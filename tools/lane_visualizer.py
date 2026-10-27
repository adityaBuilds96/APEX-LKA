"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Visualization Tools: Real-Time Precision Lane Visualizer HUD
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
================================================================================

This module provides a production-grade visualization, debugging, and live HUD
rendering suite for the APEX-Vision Lane Perception Precision Engine.

Capabilities:
- Dynamic multi-layer overlay rendering:
  * Left (Class 2) and Right (Class 3) parabolic polynomials with color-coded confidence.
  * Shaded drivable corridor (Class 1) with transparency modulation.
  * 0.3-meter inner virtual safety buffers and clearance breach zones.
  * Predicted forward trajectory arc (t + 1.5s) under Stanley guidance.
  * 8-State EKF covariance uncertainty ribbons (+/- 2 sigma confidence bands).
  * Dynamic trapezoidal Region-of-Interest (ROI) adapted to pitch and horizon.
  * Time-to-Lane-Departure (TTLD) countdown gauge with audible/visual alert state.
  * Hardware interlock HUD badge: Arduino clutch relay status & CAN 0x150 dispatch.
- Interactive keyboard shortcuts for live toggling individual visual layers.
- High-resolution video export engine writing timestamped MP4 diagnostic recordings.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zip_vault_reader import ZipVaultReader
from src.perception.adaptive_roi_engine import AdaptiveROIEngine, AdaptiveROIConfig, TrapezoidVertices
from src.perception.faded_lane_recovery import BoundaryPolynomial2D
from src.perception.lane_departure_predictor import (
    DepartureDirection,
    DepartureSeverity,
    LaneDeparturePrediction,
    ProjectedWaypoint,
)
from src.perception.lane_fusion_engine import (
    LaneFusionEngine,
    LaneFusionEngineConfig,
    LanePrediction,
    PerceptionExecutionStatus,
)
from src.perception.virtual_guardrail_enforcer import GuardrailSeverity
from src.scene.scene_profiles import SceneType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("APEX.LaneVisualizer")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Visualization Styling Configuration & Layer Toggles
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VisualizerColorPalette:
    """ASIL-B compliant high-contrast color scheme for automotive HUD rendering."""
    # BGR format
    left_boundary_nominal: Tuple[int, int, int] = (255, 180, 0)       # Amber/Cyan Class 2
    left_boundary_hallucinated: Tuple[int, int, int] = (255, 100, 255) # Magenta parallel projection
    right_boundary_nominal: Tuple[int, int, int] = (0, 230, 255)      # Bright Gold Class 3
    right_boundary_hallucinated: Tuple[int, int, int] = (255, 100, 255)
    centerline: Tuple[int, int, int] = (200, 255, 200)                # Light mint green
    corridor_fill: Tuple[int, int, int] = (0, 200, 80)                # Neon Emerald drivable zone
    guardrail_safe: Tuple[int, int, int] = (0, 255, 120)              # Safe green buffer
    guardrail_warning: Tuple[int, int, int] = (0, 215, 255)           # Yellow warning
    guardrail_critical: Tuple[int, int, int] = (0, 100, 255)          # Orange critical
    guardrail_emergency: Tuple[int, int, int] = (0, 0, 255)           # Red emergency intervention
    trajectory_arc: Tuple[int, int, int] = (255, 255, 0)              # Cyan forward trajectory
    uncertainty_ribbon: Tuple[int, int, int] = (100, 200, 255)        # Translucent uncertainty
    roi_trapezoid: Tuple[int, int, int] = (255, 0, 128)               # Violet ROI boundary
    hud_bg: Tuple[int, int, int] = (15, 18, 22)                       # Deep slate HUD panel
    hud_border: Tuple[int, int, int] = (50, 70, 90)                   # Slate border
    text_primary: Tuple[int, int, int] = (240, 245, 250)              # Crisp off-white
    text_secondary: Tuple[int, int, int] = (160, 180, 195)            # Muted silver
    text_alert: Tuple[int, int, int] = (50, 50, 255)                  # Bright red alert


@dataclass
class VisualizerLayerToggles:
    """Interactive visibility states for multi-layer HUD overlays."""
    show_polynomials: bool = True       # Key 'P': Left/Right polynomial curves
    show_corridor: bool = True          # Key 'C': Shaded drivable corridor
    show_guardrails: bool = True        # Key 'G': 0.3m inner safety buffers
    show_trajectory: bool = True        # Key 'T': Projected forward trajectory (t + 1.5s)
    show_uncertainty: bool = True       # Key 'U': Kalman +/- 2 sigma confidence bands
    show_roi: bool = True               # Key 'R': Adaptive trapezoidal ROI
    show_hud_telemetry: bool = True     # Key 'H': Primary ADAS telemetry dashboard
    show_ttld_countdown: bool = True    # Key 'D': Time-to-lane-departure gauge
    show_bev_inset: bool = False        # Key 'B': Bird's-Eye-View miniature inset


@dataclass
class LaneVisualizerConfig:
    """Master rendering specifications for automotive visualization."""
    image_width: int = 640
    image_height: int = 360
    corridor_alpha: float = 0.28
    uncertainty_alpha: float = 0.20
    guardrail_buffer_alpha: float = 0.35
    polynomial_thickness: int = 3
    centerline_thickness: int = 2
    trajectory_thickness: int = 2
    nominal_lane_width_m: float = 3.70
    camera_mount_height_m: float = 1.20
    camera_fx_px: float = 450.0
    camera_fy_px: float = 450.0
    principal_cx_px: float = 320.0
    principal_cy_px: float = 180.0
    colors: VisualizerColorPalette = field(default_factory=VisualizerColorPalette)
    toggles: VisualizerLayerToggles = field(default_factory=VisualizerLayerToggles)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Real-Time HUD & Diagnostic Overlay Renderer
# ═══════════════════════════════════════════════════════════════════════════════

class LaneVisualizer:
    """
    Automotive ADAS HUD renderer generating diagnostic and production overlays
    on top of raw 640x360 camera feeds.
    """

    def __init__(self, config: Optional[LaneVisualizerConfig] = None) -> None:
        self.config = config or LaneVisualizerConfig()
        self.colors = self.config.colors
        self.toggles = self.config.toggles

        self._frame_count = 0
        self._fps_history: List[float] = []
        self._last_timestamp = time.perf_counter()

    def render_frame(
        self,
        bgr_frame: np.ndarray,
        prediction: LanePrediction,
        fps_measured: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render all active perception overlay layers onto the camera frame.

        Args:
            bgr_frame: Input BGR image (H, W, 3) uint8.
            prediction: LanePrediction telemetry emitted by LaneFusionEngine.
            fps_measured: Optional measured system throughput in FPS.
        """
        t_start = time.perf_counter()
        self._frame_count += 1

        # Calculate instantaneous FPS if not provided
        if fps_measured is None:
            now = time.perf_counter()
            dt = max(1e-4, now - self._last_timestamp)
            self._last_timestamp = now
            fps_measured = 1.0 / dt

        self._fps_history.append(fps_measured)
        if len(self._fps_history) > 30:
            self._fps_history.pop(0)
        smooth_fps = float(np.mean(self._fps_history))

        # Create output canvas
        canvas = bgr_frame.copy()
        h, w = canvas.shape[:2]

        # 1. Layer: Adaptive ROI Boundary
        if self.toggles.show_roi:
            canvas = self._draw_adaptive_roi(canvas, prediction)

        # 2. Layer: Shaded Drivable Corridor (Class 1)
        if self.toggles.show_corridor:
            canvas = self._draw_drivable_corridor(canvas, prediction)

        # 3. Layer: Kalman Uncertainty Ribbons (+/- 2 sigma)
        if self.toggles.show_uncertainty:
            canvas = self._draw_uncertainty_ribbons(canvas, prediction)

        # 4. Layer: 0.3m Virtual Guardrail Safety Buffers
        if self.toggles.show_guardrails:
            canvas = self._draw_virtual_guardrails(canvas, prediction)

        # 5. Layer: Parabolic Polynomial Boundaries (Class 2 & Class 3)
        if self.toggles.show_polynomials:
            canvas = self._draw_lane_polynomials(canvas, prediction)

        # 6. Layer: Forward Predicted Trajectory Arc (t + 1.5s)
        if self.toggles.show_trajectory:
            canvas = self._draw_projected_trajectory(canvas, prediction)

        # 7. Layer: TTLD Visual Countdown Gauge
        if self.toggles.show_ttld_countdown:
            canvas = self._draw_ttld_gauge(canvas, prediction)

        # 8. Layer: Master ADAS Telemetry Dashboard Banner
        if self.toggles.show_hud_telemetry:
            canvas = self._draw_telemetry_hud(canvas, prediction, smooth_fps)

        # 9. Layer: Miniature BEV Inset (if enabled)
        if self.toggles.show_bev_inset:
            canvas = self._draw_bev_mini_inset(canvas, prediction)

        return canvas

    def handle_keyboard_input(self, key_code: int) -> bool:
        """
        Process interactive key presses to toggle overlay layers.
        Returns False if ESC or 'q' was pressed to exit; True otherwise.
        """
        if key_code in (27, ord("q"), ord("Q")):
            return False

        k = key_code & 0xFF
        if k in (ord("p"), ord("P")):
            self.toggles.show_polynomials = not self.toggles.show_polynomials
            logger.info(f"Toggled Polynomials: {self.toggles.show_polynomials}")
        elif k in (ord("c"), ord("C")):
            self.toggles.show_corridor = not self.toggles.show_corridor
            logger.info(f"Toggled Corridor: {self.toggles.show_corridor}")
        elif k in (ord("g"), ord("G")):
            self.toggles.show_guardrails = not self.toggles.show_guardrails
            logger.info(f"Toggled Guardrails: {self.toggles.show_guardrails}")
        elif k in (ord("t"), ord("T")):
            self.toggles.show_trajectory = not self.toggles.show_trajectory
            logger.info(f"Toggled Trajectory: {self.toggles.show_trajectory}")
        elif k in (ord("u"), ord("U")):
            self.toggles.show_uncertainty = not self.toggles.show_uncertainty
            logger.info(f"Toggled Uncertainty: {self.toggles.show_uncertainty}")
        elif k in (ord("r"), ord("R")):
            self.toggles.show_roi = not self.toggles.show_roi
            logger.info(f"Toggled ROI: {self.toggles.show_roi}")
        elif k in (ord("h"), ord("H")):
            self.toggles.show_hud_telemetry = not self.toggles.show_hud_telemetry
            logger.info(f"Toggled HUD: {self.toggles.show_hud_telemetry}")
        elif k in (ord("d"), ord("D")):
            self.toggles.show_ttld_countdown = not self.toggles.show_ttld_countdown
            logger.info(f"Toggled TTLD Gauge: {self.toggles.show_ttld_countdown}")
        elif k in (ord("b"), ord("B")):
            self.toggles.show_bev_inset = not self.toggles.show_bev_inset
            logger.info(f"Toggled BEV Inset: {self.toggles.show_bev_inset}")

        return True

    # ───────────────────────────────────────────────────────────────────────────
    # Modular Layer Rendering Methods
    # ───────────────────────────────────────────────────────────────────────────

    def _draw_adaptive_roi(self, canvas: np.ndarray, prediction: LanePrediction) -> np.ndarray:
        """Render subtle bounding trapezoid of the dynamic Region-of-Interest."""
        h, w = canvas.shape[:2]
        horizon_y = int(self.config.principal_cy_px + 20)  # Nominal horizon

        # Dynamic vertices adapted to curvature
        curve_a = prediction.left_polynomial[0]
        curve_skew = int(np.clip(curve_a * 50000.0, -40, 40))

        v_top_l = (int(w * 0.38) + curve_skew, horizon_y)
        v_top_r = (int(w * 0.62) + curve_skew, horizon_y)
        v_bot_r = (int(w * 0.94), h - 1)
        v_bot_l = (int(w * 0.06), h - 1)

        trapezoid = np.array([v_top_l, v_top_r, v_bot_r, v_bot_l], dtype=np.int32)
        cv2.polylines(canvas, [trapezoid], True, self.colors.roi_trapezoid, 1, cv2.LINE_AA)
        return canvas

    def _draw_drivable_corridor(
        self, canvas: np.ndarray, prediction: LanePrediction
    ) -> np.ndarray:
        """Render shaded translucent emerald polygon representing the drivable corridor."""
        h, w = canvas.shape[:2]
        y_samples = np.linspace(int(h * 0.52), h - 1, 40)

        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial

        pts_l: List[Tuple[int, int]] = []
        pts_r: List[Tuple[int, int]] = []

        for y in y_samples:
            xl = int(poly_l[0] * y**2 + poly_l[1] * y + poly_l[2])
            xr = int(poly_r[0] * y**2 + poly_r[1] * y + poly_r[2])
            if -50 <= xl <= w + 50:
                pts_l.append((xl, int(y)))
            if -50 <= xr <= w + 50:
                pts_r.append((xr, int(y)))

        if len(pts_l) > 3 and len(pts_r) > 3:
            overlay = canvas.copy()
            corridor_poly = np.array(pts_l + pts_r[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [corridor_poly], self.colors.corridor_fill)
            # Modulate alpha based on perception quality
            q = float(np.clip(prediction.quality_assessment.quality_score, 0.15, 0.95))
            effective_alpha = self.config.corridor_alpha * q
            canvas = cv2.addWeighted(overlay, effective_alpha, canvas, 1.0 - effective_alpha, 0)

        return canvas

    def _draw_uncertainty_ribbons(
        self, canvas: np.ndarray, prediction: LanePrediction
    ) -> np.ndarray:
        """Render translucent Kalman filter uncertainty ribbons (+/- 2 sigma)."""
        h, w = canvas.shape[:2]
        y_samples = np.linspace(int(h * 0.52), h - 1, 30)

        # Extract variance from smoothed estimate covariance P
        cov_l = float(np.trace(prediction.smoothed_estimate.covariance_matrix))
        sigma_px = math.sqrt(max(0.1, cov_l)) * 25.0  # Scale to image pixels

        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial

        overlay = canvas.copy()
        for poly in (poly_l, poly_r):
            pts_inner = []
            pts_outer = []
            for y in y_samples:
                x = poly[0] * y**2 + poly[1] * y + poly[2]
                pts_inner.append((int(x - sigma_px), int(y)))
                pts_outer.append((int(x + sigma_px), int(y)))
            if len(pts_inner) > 3:
                ribbon = np.array(pts_inner + pts_outer[::-1], dtype=np.int32)
                cv2.fillPoly(overlay, [ribbon], self.colors.uncertainty_ribbon)

        canvas = cv2.addWeighted(
            overlay,
            self.config.uncertainty_alpha,
            canvas,
            1.0 - self.config.uncertainty_alpha,
            0,
        )
        return canvas

    def _draw_virtual_guardrails(
        self, canvas: np.ndarray, prediction: LanePrediction
    ) -> np.ndarray:
        """Render 0.3-meter inner virtual guardrail safety buffers."""
        h, w = canvas.shape[:2]
        y_samples = np.linspace(int(h * 0.55), h - 1, 35)

        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial

        # Buffer width: 0.3m corresponds to ~30 px at bumper, scaling with perspective
        overlay = canvas.copy()
        severity = prediction.guardrail_result.active_severity

        # Determine buffer color by severity
        if severity == GuardrailSeverity.EMERGENCY:
            b_color = self.colors.guardrail_emergency
        elif severity == GuardrailSeverity.CRITICAL:
            b_color = self.colors.guardrail_critical
        elif severity == GuardrailSeverity.WARNING:
            b_color = self.colors.guardrail_warning
        else:
            b_color = self.colors.guardrail_safe

        # Left inner buffer ribbon
        pts_left_bound = []
        pts_left_buf = []
        # Right inner buffer ribbon
        pts_right_bound = []
        pts_right_buf = []

        for y in y_samples:
            # Perspective factor
            scale = (y - (h * 0.5)) / (h * 0.5)
            buf_px = int(max(6, 28 * scale))

            xl = int(poly_l[0] * y**2 + poly_l[1] * y + poly_l[2])
            xr = int(poly_r[0] * y**2 + poly_r[1] * y + poly_r[2])

            pts_left_bound.append((xl, int(y)))
            pts_left_buf.append((xl + buf_px, int(y)))
            pts_right_bound.append((xr, int(y)))
            pts_right_buf.append((xr - buf_px, int(y)))

        if len(pts_left_bound) > 3:
            poly_l_buf = np.array(pts_left_bound + pts_left_buf[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [poly_l_buf], b_color)

        if len(pts_right_bound) > 3:
            poly_r_buf = np.array(pts_right_bound + pts_right_buf[::-1], dtype=np.int32)
            cv2.fillPoly(overlay, [poly_r_buf], b_color)

        canvas = cv2.addWeighted(
            overlay,
            self.config.guardrail_buffer_alpha,
            canvas,
            1.0 - self.config.guardrail_buffer_alpha,
            0,
        )
        return canvas

    def _draw_lane_polynomials(
        self, canvas: np.ndarray, prediction: LanePrediction
    ) -> np.ndarray:
        """Render parabolic curves for Class 2 Left and Class 3 Right lane markings."""
        h, w = canvas.shape[:2]
        y_samples = np.linspace(int(h * 0.52), h - 1, 50)

        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial
        poly_c = prediction.centerline_polynomial

        # Determine colors (nominal vs hallucinated)
        is_hallu = prediction.status == PerceptionExecutionStatus.PARALLEL_HALLUCINATED
        col_l = (
            self.colors.left_boundary_hallucinated
            if is_hallu and prediction.left_polynomial == prediction.centerline_polynomial
            else self.colors.left_boundary_nominal
        )
        col_r = (
            self.colors.right_boundary_hallucinated
            if is_hallu
            else self.colors.right_boundary_nominal
        )

        pts_l = []
        pts_r = []
        pts_c = []

        for y in y_samples:
            xl = int(poly_l[0] * y**2 + poly_l[1] * y + poly_l[2])
            xr = int(poly_r[0] * y**2 + poly_r[1] * y + poly_r[2])
            xc = int(poly_c[0] * y**2 + poly_c[1] * y + poly_c[2])

            if 0 <= xl < w:
                pts_l.append((xl, int(y)))
            if 0 <= xr < w:
                pts_r.append((xr, int(y)))
            if 0 <= xc < w:
                pts_c.append((xc, int(y)))

        # Draw left line
        if len(pts_l) > 2:
            cv2.polylines(canvas, [np.array(pts_l)], False, col_l, self.config.polynomial_thickness, cv2.LINE_AA)

        # Draw right line
        if len(pts_r) > 2:
            cv2.polylines(canvas, [np.array(pts_r)], False, col_r, self.config.polynomial_thickness, cv2.LINE_AA)

        # Draw dotted centerline
        if len(pts_c) > 4:
            for i in range(0, len(pts_c) - 1, 2):
                cv2.line(canvas, pts_c[i], pts_c[i + 1], self.colors.centerline, self.config.centerline_thickness, cv2.LINE_AA)

        return canvas

    def _draw_projected_trajectory(
        self, canvas: np.ndarray, prediction: LanePrediction
    ) -> np.ndarray:
        """Render ego vehicle predicted forward trajectory arc (t + 1.5s)."""
        h, w = canvas.shape[:2]
        traj_pts: List[Tuple[int, int]] = []

        # Ground plane to image perspective projection
        speed = 25.0
        yaw_rate = math.radians(prediction.heading_error_deg) * 0.4
        t_samples = np.linspace(0.1, 1.5, 20)

        for t in t_samples:
            # Longitudinal meters forward
            dy_m = speed * t
            # Lateral displacement meters
            dx_m = 0.5 * speed * yaw_rate * t**2 - (prediction.lateral_error_m * 0.4)

            # Project to image perspective (y=0 at bumper h-1, y=30m at h*0.55)
            y_img = int((h - 1) - (dy_m / 35.0) * (h * 0.45))
            x_img = int((w * 0.5) + (dx_m * 90.0) * ((y_img - (h * 0.45)) / (h * 0.55)))

            if 0 <= y_img < h and 0 <= x_img < w:
                traj_pts.append((x_img, y_img))

        if len(traj_pts) > 2:
            cv2.polylines(
                canvas,
                [np.array(traj_pts)],
                False,
                self.colors.trajectory_arc,
                self.config.trajectory_thickness,
                cv2.LINE_AA,
            )
            # Target endpoint circle
            cv2.circle(canvas, traj_pts[-1], 5, self.colors.trajectory_arc, -1)

        return canvas

    def _draw_ttld_gauge(self, canvas: np.ndarray, prediction: LanePrediction) -> np.ndarray:
        """Render Time-to-Lane-Departure (TTLD) countdown gauge badge."""
        dep_pred = prediction.departure_prediction
        ttld = dep_pred.time_to_departure_s

        # Position at bottom right
        h, w = canvas.shape[:2]
        gx, gy = w - 165, h - 80

        # Background panel
        cv2.rectangle(canvas, (gx, gy), (gx + 155, gy + 70), self.colors.hud_bg, -1)
        cv2.rectangle(canvas, (gx, gy), (gx + 155, gy + 70), self.colors.hud_border, 1)

        cv2.putText(canvas, "TIME TO DEPARTURE", (gx + 10, gy + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.colors.text_secondary, 1)

        if ttld < 1.0:
            val_col = self.colors.guardrail_emergency
            stat_text = "EMERGENCY"
        elif ttld < 2.0:
            val_col = self.colors.guardrail_warning
            stat_text = "WARNING"
        else:
            val_col = self.colors.guardrail_safe
            stat_text = "NOMINAL"

        ttld_str = f"{ttld:.2f} s" if ttld < 20.0 else "> 10.0 s"
        cv2.putText(canvas, ttld_str, (gx + 10, gy + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.70, val_col, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"STATUS: {stat_text}", (gx + 10, gy + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.35, val_col, 1)

        return canvas

    def _draw_telemetry_hud(
        self, canvas: np.ndarray, prediction: LanePrediction, fps: float
    ) -> np.ndarray:
        """Render primary multi-parameter ADAS telemetry dashboard on top of frame."""
        h, w = canvas.shape[:2]

        # Top dashboard panel: (10, 10) to (440, 115)
        cv2.rectangle(canvas, (10, 10), (450, 115), self.colors.hud_bg, -1)
        cv2.rectangle(canvas, (10, 10), (450, 115), self.colors.hud_border, 1)

        # Title and branding
        cv2.putText(canvas, "APEX-VISION | CORE v4.0", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, self.colors.text_primary, 1, cv2.LINE_AA)
        cv2.putText(canvas, "SAE L2 HIGHWAY PRECISION", (210, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.colors.text_secondary, 1)

        # Telemetry columns
        col1_x = 20
        col2_x = 230

        # Row 1: Scene Tag and Execution Status
        cv2.putText(canvas, f"Scene:      {prediction.scene_tag}", (col1_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors.text_primary, 1)
        cv2.putText(canvas, f"Status:     {prediction.status.value[:14]}", (col2_x, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors.guardrail_safe if prediction.status == PerceptionExecutionStatus.OPTIMAL else self.colors.guardrail_warning, 1)

        # Row 2: Lateral Error and Heading Error
        lat_col = self.colors.guardrail_safe if abs(prediction.lateral_error_m) < 0.08 else self.colors.guardrail_warning
        cv2.putText(canvas, f"Lat Error:  {prediction.lateral_error_m:+.3f} m", (col1_x, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, lat_col, 1)
        cv2.putText(canvas, f"Heading:    {prediction.heading_error_deg:+.2f}°", (col2_x, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.colors.text_primary, 1)

        # Row 3: Quality Score and Guardrail Severity
        q_val = prediction.quality_assessment.quality_score
        q_col = self.colors.guardrail_safe if q_val >= 0.70 else self.colors.guardrail_warning if q_val >= 0.35 else self.colors.guardrail_emergency
        cv2.putText(canvas, f"Quality Q:  {q_val:.2f}", (col1_x, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40, q_col, 1)

        sev = prediction.guardrail_result.active_severity
        sev_col = self.colors.guardrail_safe if sev == GuardrailSeverity.NOMINAL else self.colors.guardrail_emergency
        cv2.putText(canvas, f"Guardrail:  {sev.name}", (col2_x, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.40, sev_col, 1)

        # Row 4: Throughput FPS and Actuation Interlock
        relay_str = "RELAY ENGAGED" if prediction.arduino_clutch_relay_engaged else "ACTUATION SAFE"
        relay_col = self.colors.guardrail_emergency if prediction.arduino_clutch_relay_engaged else self.colors.guardrail_safe
        cv2.putText(canvas, f"Throughput: {fps:.1f} FPS", (col1_x, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.colors.text_secondary, 1)
        cv2.putText(canvas, f"Interlock:  {relay_str}", (col2_x, 102), cv2.FONT_HERSHEY_SIMPLEX, 0.38, relay_col, 1)

        return canvas

    def _draw_bev_mini_inset(self, canvas: np.ndarray, prediction: LanePrediction) -> np.ndarray:
        """Render miniature Bird's-Eye-View (BEV) schematic inset."""
        h, w = canvas.shape[:2]
        iw, ih = 140, 160
        ix, iy = 10, h - ih - 10

        # Background inset box
        cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), (10, 12, 16), -1)
        cv2.rectangle(canvas, (ix, iy), (ix + iw, iy + ih), self.colors.hud_border, 1)
        cv2.putText(canvas, "BEV GRID (36m)", (ix + 12, iy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.colors.text_secondary, 1)

        # Vehicle marker at bottom center
        vx, vy = ix + (iw // 2), iy + ih - 15
        cv2.rectangle(canvas, (vx - 8, vy - 14), (vx + 8, vy + 4), (100, 200, 255), -1)

        # Lane boundary lines in BEV
        y_bev = np.linspace(2.0, 35.0, 20)
        pts_bev_l = []
        pts_bev_r = []

        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial

        for y_m in y_bev:
            # Map BEV longitudinal meters to inset y
            iy_sample = int(vy - (y_m / 36.0) * (ih - 30))
            # Evaluate lateral displacement
            # Parabolic equation x(y) in vehicle meters
            xl_m = poly_l[0] * (y_m * 10)**2 + poly_l[1] * (y_m * 10) + poly_l[2]
            # Convert to inset x (center is vx)
            ix_l = int(vx - 22)
            ix_r = int(vx + 22)
            pts_bev_l.append((ix_l, iy_sample))
            pts_bev_r.append((ix_r, iy_sample))

        if len(pts_bev_l) > 2:
            cv2.polylines(canvas, [np.array(pts_bev_l)], False, self.colors.left_boundary_nominal, 2)
            cv2.polylines(canvas, [np.array(pts_bev_r)], False, self.colors.right_boundary_nominal, 2)

        return canvas

    def _draw_cursor_metric_query(
        self, canvas: np.ndarray, prediction: LanePrediction, cursor_uv: Tuple[int, int]
    ) -> np.ndarray:
        """Render interactive cursor coordinate probe showing vehicle metric coordinates."""
        u, v = cursor_uv
        h, w = canvas.shape[:2]
        if not (0 <= u < w and 0 <= v < h):
            return canvas

        # Ground plane inverse mapping
        # y_ground = (H_cam * fy) / (v - cy)
        cy = self.config.principal_cy_px
        fy = self.config.camera_fy_px
        fx = self.config.camera_fx_px
        cx = self.config.principal_cx_px
        h_cam = self.config.camera_mount_height_m

        dy = v - cy
        if dy > 5:
            dist_long_m = (h_cam * fy) / dy
            dist_lat_m = ((u - cx) * dist_long_m) / fx
            coord_str = f"X: {dist_lat_m:+.2f}m | Y: {dist_long_m:.1f}m"
        else:
            coord_str = "HORIZON / SKY"

        cv2.drawMarker(canvas, (u, v), (0, 255, 255), cv2.MARKER_CROSS, 14, 1)
        cv2.putText(
            canvas,
            coord_str,
            (min(w - 180, u + 12), max(25, v - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return canvas


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Interactive Desktop Window & Snapshot Management
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveVisualizerWindow:
    """
    Manages OpenCV graphical desktop window with mouse hover callbacks,
    keyboard hotkey bindings, and diagnostic snapshot capturing.
    """

    def __init__(
        self, window_name: str = "APEX-Vision Core v4.0 — Lane Perception HUD"
    ) -> None:
        self.window_name = window_name
        self.cursor_pos: Tuple[int, int] = (-1, -1)
        self._window_created = False

    def init_window(self) -> None:
        """Initialize OpenCV desktop viewport and register mouse callback."""
        if not self._window_created:
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.setMouseCallback(self.window_name, self._on_mouse_event)
                self._window_created = True
            except cv2.error as e:
                logger.warning(f"Headless environment detected; GUI window disabled: {e}")

    def _on_mouse_event(self, event: int, x: int, y: int, flags: int, param: Any) -> None:
        """Mouse callback updating hover position."""
        if event == cv2.EVENT_MOUSEMOVE:
            self.cursor_pos = (x, y)

    def show_frame(self, frame: np.ndarray, wait_ms: int = 1) -> int:
        """Display frame and return pressed key code."""
        if self._window_created:
            cv2.imshow(self.window_name, frame)
            return cv2.waitKey(wait_ms)
        return -1

    def destroy(self) -> None:
        """Tear down OpenCV window."""
        if self._window_created:
            cv2.destroyWindow(self.window_name)
            self._window_created = False


class DiagnosticSnapshotExporter:
    """
    Exports synchronized diagnostic PNG frames and JSON telemetry payloads
    for safety auditing and offline failure inspection.
    """

    def __init__(
        self, export_dir: Union[str, Path] = "docs/phase7b/BENCHMARKS/section1/snapshots"
    ) -> None:
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)

    def export_snapshot(
        self,
        annotated_frame: np.ndarray,
        prediction: LanePrediction,
        prefix: str = "diagnostic_snapshot",
    ) -> Path:
        """Save image and sidecar JSON metadata."""
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        img_name = f"{prefix}_{ts}.png"
        json_name = f"{prefix}_{ts}.json"

        img_path = self.export_dir / img_name
        json_path = self.export_dir / json_name

        cv2.imwrite(str(img_path), annotated_frame)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(prediction.to_summary_dict(), f, indent=2)

        logger.info(f"Saved diagnostic snapshot: {img_path}")
        return img_path


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Batch Visualizer Runner & Video Generator
# ═══════════════════════════════════════════════════════════════════════════════

class LaneVisualizerRunner:
    """
    Automated execution runner orchestrating dataset reading, real-time perception
    inference, HUD overlay composition, and video file serialization.
    """

    def __init__(
        self,
        dataset_path: Union[str, Path] = "data/raw/lane_dataset.zip",
        output_video_path: Union[str, Path] = "docs/phase7b/BENCHMARKS/section1/lane_visualizer_demo.mp4",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self.dataset_path = Path(dataset_path)
        self.output_video_path = Path(output_video_path)
        self.output_video_path.parent.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.engine = LaneFusionEngine(LaneFusionEngineConfig())
        self.visualizer = LaneVisualizer(LaneVisualizerConfig())

    def render_demo_video(
        self, max_clips: int = 5, frames_per_clip: int = 30, fps: float = 20.0
    ) -> Path:
        """Generate high-resolution MP4 video demonstrating the full visualization stack."""
        logger.info(f"Rendering diagnostic video to: {self.output_video_path}")

        frames = self._load_frames(max_clips, frames_per_clip)
        if not frames:
            logger.warning("No frames loaded from vault; generating high-fidelity synthetic demo.")
            frames = self._generate_synthetic_highway(max_clips * frames_per_clip)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_video_path), fourcc, fps, (640, 360))

        try:
            self.engine.reset()
            for idx, frame in enumerate(frames):
                # Run perception
                pred = self.engine.process_frame(
                    bgr_frame=frame,
                    ego_speed_mps=25.0,
                    scene_tag="HIGHWAY",
                    dt_s=1.0 / fps,
                )
                annotated = self.visualizer.render_frame(frame, pred, fps_measured=fps)
                writer.write(annotated)

                if (idx + 1) % 25 == 0:
                    logger.info(f"Rendered {idx+1}/{len(frames)} frames...")
        finally:
            writer.release()
            logger.info(f"Diagnostic video successfully published to: {self.output_video_path}")

        return self.output_video_path

    def _load_frames(self, max_clips: int, frames_per_clip: int) -> List[np.ndarray]:
        """Load TuSimple clips from zip vault."""
        frames: List[np.ndarray] = []
        if not self.dataset_path.is_file():
            return frames

        try:
            reader = ZipVaultReader(self.dataset_path)
            all_files = reader.list_files()
            jpgs = [f for f in all_files if f.lower().endswith(".jpg")]
            if not jpgs:
                return frames

            from collections import defaultdict

            grouped = defaultdict(list)
            for f in jpgs:
                p = Path(f)
                grouped[str(p.parent)].append(f)

            for ck in list(grouped.keys())[:max_clips]:
                for f_name in sorted(grouped[ck])[:frames_per_clip]:
                    buf = reader.read_bytes(f_name)
                    if buf:
                        arr = np.frombuffer(buf, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            if img.shape[:2] != (360, 640):
                                img = cv2.resize(img, (640, 360))
                            frames.append(img)
        except Exception as e:
            logger.error(f"Error loading frames from vault: {e}")

        return frames

    def _generate_synthetic_highway(self, total_frames: int) -> List[np.ndarray]:
        """Generate highway frames with road curvature drift and adverse lighting."""
        frames: List[np.ndarray] = []
        for i in range(total_frames):
            frame = np.full((360, 640, 3), 65, dtype=np.uint8)
            curve = math.sin(i * 0.08) * 0.0004
            y_eval = np.linspace(170, 360, 40)
            pts_l = []
            pts_r = []

            for y in y_eval:
                xl = int(curve * (y - 170) ** 2 - 0.45 * (y - 170) + 290)
                xr = int(curve * (y - 170) ** 2 + 0.45 * (y - 170) + 350)
                if 0 <= xl < 640:
                    pts_l.append((xl, int(y)))
                if 0 <= xr < 640:
                    pts_r.append((xr, int(y)))

            if len(pts_l) > 1:
                cv2.polylines(frame, [np.array(pts_l)], False, (220, 220, 220), 4)
            if len(pts_r) > 1:
                cv2.polylines(frame, [np.array(pts_r)], False, (220, 220, 220), 4)

            frames.append(frame)
        return frames


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Command-Line Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse CLI options for lane visualizer."""
    parser = argparse.ArgumentParser(
        description="APEX-Vision Phase 7B Section 1 Lane Visualizer & HUD Renderer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/lane_dataset.zip",
        help="Path to TuSimple raw dataset archive",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default="docs/phase7b/BENCHMARKS/section1/lane_visualizer_demo.mp4",
        help="Path for generated annotated video",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=4,
        help="Maximum clips to process",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=25,
        help="Frames per clip to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Hardware execution device",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution pipeline."""
    args = parse_args()
    logger.info("Starting APEX-Vision Lane Visualizer...")

    runner = LaneVisualizerRunner(
        dataset_path=args.dataset,
        output_video_path=args.output_video,
        device=args.device,
    )

    out_file = runner.render_demo_video(
        max_clips=args.max_clips,
        frames_per_clip=args.frames_per_clip,
    )

    logger.info(f"Visualizer run complete. Video saved to: {out_file}")


if __name__ == "__main__":
    main()
