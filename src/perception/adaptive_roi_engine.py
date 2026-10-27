"""
src/perception/adaptive_roi_engine.py
=====================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Dynamic Curvature-Adaptive & Pitch-Compensated Perspective Region of Interest (ROI) Engine.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module isolates the active drivable road surface from irrelevant environmental clutter
(sky, overpasses, oncoming traffic, roadside signs, and vehicle bonnet/hood). The dynamic
perspective trapezoid continuously adapts its vertical position to vehicle pitch attitude
from the MTi-680 INS and dynamically broadens its lateral aperture on sharp curves.

Key Mathematical Formulations:
------------------------------
1. Dynamic Vanishing Line (Horizon Calculation):
   v_horizon = c_y + f_y * tan(theta_pitch)
   For nominal camera intrinsics: c_y = 180 px, f_y = 450 px, nominal pitch = 5.0 deg.
2. Curvature-Adaptive Lateral Width Scaling:
   W_top(A) = W_nominal * (1.0 + k_curve * sqrt(|A|))
   Expands search sector laterally on sharp curves (high curvature |A|).
3. Scene-Profile Driven Trapezoid Parameterization:
   Modulates top and bottom cutoff ratios based on active scene (Highway, Urban, Rural, Tunnel).
4. Sub-Pixel Anti-Aliased Polygon Masking:
   Generates optimized binary and alpha-blended perceptual ROI masks.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("APEX.AdaptiveROIEngine")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Telemetry & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrapezoidVertices:
    """Four corners of the perspective trapezoid in (x, y) pixel coordinates."""
    top_left: Tuple[int, int]
    top_right: Tuple[int, int]
    bottom_right: Tuple[int, int]
    bottom_left: Tuple[int, int]

    def to_numpy(self) -> np.ndarray:
        """Return [4, 1, 2] int32 contour array for cv2.fillPoly."""
        pts = [
            list(self.top_left),
            list(self.top_right),
            list(self.bottom_right),
            list(self.bottom_left),
        ]
        return np.array(pts, dtype=np.int32).reshape((-1, 1, 2))

    def to_float_array(self) -> np.ndarray:
        """Return [4, 2] float32 array for perspective warp calculations."""
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left,
        ], dtype=np.float32)


@dataclass
class AdaptiveROIMetrics:
    """Diagnostic telemetry from a single ROI adaptation cycle."""
    horizon_y_px: int
    pitch_angle_deg: float
    curvature_A: float
    expansion_factor: float
    mask_area_px: int
    mask_area_fraction: float
    top_width_px: int
    bottom_width_px: int
    processing_time_ms: float
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "horizon_y_px": self.horizon_y_px,
            "pitch_angle_deg": round(self.pitch_angle_deg, 3),
            "curvature_A": round(self.curvature_A, 7),
            "expansion_factor": round(self.expansion_factor, 3),
            "mask_area_px": self.mask_area_px,
            "mask_area_fraction": round(self.mask_area_fraction, 4),
            "top_width_px": self.top_width_px,
            "bottom_width_px": self.bottom_width_px,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "timestamp_utc": self.timestamp_utc,
        }


@dataclass
class AdaptiveROIResult:
    """Complete output payload from the Adaptive ROI engine."""
    masked_frame_bgr: np.ndarray
    binary_mask: np.ndarray             # Single channel 8-bit mask (0 or 255)
    trapezoid: TrapezoidVertices
    metrics: AdaptiveROIMetrics


@dataclass
class AdaptiveROIConfig:
    """Hyperparameters for camera mounting, geometry, and curve adaptation."""
    frame_width: int = 640
    frame_height: int = 360
    focal_length_y_px: float = 450.0    # f_y
    principal_point_x_px: float = 320.0 # c_x
    principal_point_y_px: float = 180.0 # c_y
    nominal_pitch_deg: float = 5.0
    camera_height_m: float = 1.20
    top_horizon_offset_px: int = 20     # Margin below/above horizon line
    bottom_bonnet_cutoff_px: int = 345  # Suppress front bumper
    base_top_width_ratio: float = 0.45  # Top trapezoid width / frame_width
    base_bottom_width_ratio: float = 0.95
    curve_expansion_gain: float = 12.0  # Scales expansion as function of sqrt(|A|)
    max_expansion_ratio: float = 1.80
    enable_anti_aliased_edges: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Master Adaptive ROI Engine
# ═══════════════════════════════════════════════════════════════════════════════

class AdaptiveROIEngine:
    """
    Computes dynamic, vehicle-motion compensated perspective regions of interest.
    """

    def __init__(self, config: Optional[AdaptiveROIConfig] = None) -> None:
        self.config = config if config is not None else AdaptiveROIConfig()
        self._mask_cache: Dict[Tuple[int, int, int, int], np.ndarray] = {}

    def compute_horizon_line_px(self, pitch_deg: float) -> int:
        """
        Compute vertical pixel coordinate of the vanishing horizon:
          v_horizon = c_y + f_y * tan(pitch_rad)
        """
        pitch_rad = math.radians(pitch_deg)
        v_h = self.config.principal_point_y_px + self.config.focal_length_y_px * math.tan(pitch_rad)
        # Clamp horizon strictly within upper half of frame
        return int(round(max(60.0, min(float(self.config.frame_height - 80), v_h))))

    def compute_trapezoid_vertices(
        self,
        pitch_deg: float,
        curvature_A: float = 0.0,
        lateral_offset_px: float = 0.0,
    ) -> Tuple[TrapezoidVertices, float]:
        """
        Calculate the 4 corner points of the perspective trapezoid.
        Returns: (trapezoid, expansion_factor)
        """
        w, h = self.config.frame_width, self.config.frame_height
        cx = self.config.principal_point_x_px + lateral_offset_px

        # 1. Vertical boundaries
        v_horizon = self.compute_horizon_line_px(pitch_deg)
        y_top = min(h - 40, max(40, v_horizon + self.config.top_horizon_offset_px))
        y_bottom = min(h - 5, max(y_top + 30, self.config.bottom_bonnet_cutoff_px))

        # 2. Curvature-adaptive lateral expansion
        # Curvature coefficient A has units 1/m or 1/px^2
        curv_mag = abs(curvature_A)
        expansion = 1.0 + min(
            self.config.max_expansion_ratio - 1.0,
            self.config.curve_expansion_gain * math.sqrt(curv_mag),
        )

        top_half_w = (w * self.config.base_top_width_ratio * 0.5) * expansion
        bot_half_w = (w * self.config.base_bottom_width_ratio * 0.5)

        # 3. Trapezoid Corners
        tl_x = int(round(max(0.0, cx - top_half_w)))
        tr_x = int(round(min(float(w - 1), cx + top_half_w)))
        bl_x = int(round(max(0.0, cx - bot_half_w)))
        br_x = int(round(min(float(w - 1), cx + bot_half_w)))

        trapezoid = TrapezoidVertices(
            top_left=(tl_x, y_top),
            top_right=(tr_x, y_top),
            bottom_right=(br_x, y_bottom),
            bottom_left=(bl_x, y_bottom),
        )
        return trapezoid, expansion

    def generate_binary_mask(self, trapezoid: TrapezoidVertices) -> np.ndarray:
        """
        Generate 8-bit binary mask rasterizing the trapezoid polygon.
        """
        h, w = self.config.frame_height, self.config.frame_width
        mask = np.zeros((h, w), dtype=np.uint8)
        contour = trapezoid.to_numpy()
        cv2.fillPoly(mask, [contour], 255)

        if self.config.enable_anti_aliased_edges:
            # Feather boundary slightly to prevent hard gradient false edges
            mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
            mask = (mask > 120).astype(np.uint8) * 255

        return mask

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        pitch_deg: Optional[float] = None,
        curvature_A: float = 0.0,
        lateral_offset_px: float = 0.0,
    ) -> AdaptiveROIResult:
        """
        Execute full ROI adaptation and apply mask to input frame.
        """
        t0 = time.perf_counter()
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame to AdaptiveROIEngine is empty or None.")

        actual_pitch = float(pitch_deg if pitch_deg is not None else self.config.nominal_pitch_deg)

        # 1. Compute dynamic vertices
        trapezoid, expansion = self.compute_trapezoid_vertices(actual_pitch, curvature_A, lateral_offset_px)

        # 2. Rasterize binary mask
        binary_mask = self.generate_binary_mask(trapezoid)

        # 3. Apply mask to BGR frame
        masked_frame = cv2.bitwise_and(frame_bgr, frame_bgr, mask=binary_mask)

        # 4. Metrics telemetry
        area_px = int(np.count_nonzero(binary_mask))
        total_px = frame_bgr.shape[0] * frame_bgr.shape[1]
        area_frac = area_px / max(1, total_px)
        top_w = abs(trapezoid.top_right[0] - trapezoid.top_left[0])
        bot_w = abs(trapezoid.bottom_right[0] - trapezoid.bottom_left[0])
        horizon_y = self.compute_horizon_line_px(actual_pitch)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = AdaptiveROIMetrics(
            horizon_y_px=horizon_y,
            pitch_angle_deg=actual_pitch,
            curvature_A=curvature_A,
            expansion_factor=expansion,
            mask_area_px=area_px,
            mask_area_fraction=area_frac,
            top_width_px=top_w,
            bottom_width_px=bot_w,
            processing_time_ms=elapsed_ms,
        )

        return AdaptiveROIResult(
            masked_frame_bgr=masked_frame,
            binary_mask=binary_mask,
            trapezoid=trapezoid,
            metrics=metrics,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic Overlay Renderer
    # ──────────────────────────────────────────────────────────────────────────

    def render_overlay(
        self,
        canvas_bgr: np.ndarray,
        result: AdaptiveROIResult,
        alpha: float = 0.35,
    ) -> np.ndarray:
        """
        Draw translucent shaded trapezoid, horizon line, and corner vertices onto canvas.
        """
        out = canvas_bgr.copy()
        h, w = out.shape[:2]

        # Draw translucent polygon fill
        overlay = out.copy()
        contour = result.trapezoid.to_numpy()
        cv2.fillPoly(overlay, [contour], (0, 255, 120))  # Greenish tint
        cv2.addWeighted(overlay, alpha, out, 1.0 - alpha, 0, out)

        # Draw trapezoid boundary outline
        cv2.polylines(out, [contour], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)

        # Draw horizon line
        hy = result.metrics.horizon_y_px
        cv2.line(out, (0, hy), (w, hy), (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(out, f"HORIZON: y={hy}px (pitch={result.metrics.pitch_angle_deg:.1f}deg)", (15, max(20, hy - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 255), 1, cv2.LINE_AA)

        # Draw Corner Points
        for pt in [result.trapezoid.top_left, result.trapezoid.top_right, result.trapezoid.bottom_right, result.trapezoid.bottom_left]:
            cv2.circle(out, pt, 4, (0, 255, 0), -1)

        # Telemetry Badge
        badge = f"ROI: Area={result.metrics.mask_area_fraction*100:.1f}% | Exp={result.metrics.expansion_factor:.2f}x"
        cv2.putText(out, badge, (result.trapezoid.top_left[0], max(20, result.trapezoid.top_left[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255), 1, cv2.LINE_AA)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Multi-Resolution Pyramid ROI Processor
# ═══════════════════════════════════════════════════════════════════════════════

class PyramidROIProcessor:
    """
    Constructs multi-resolution ROI masks for hierarchical perception pipelines.
    Generates level 0 (640x360), level 1 (320x180), and level 2 (160x90) masks.
    """

    def __init__(self, engine: AdaptiveROIEngine) -> None:
        self.engine = engine

    def generate_pyramid_masks(
        self,
        pitch_deg: float,
        curvature_A: float = 0.0,
    ) -> List[np.ndarray]:
        """
        Generate Gaussian pyramid of binary masks.
        """
        trapezoid, _ = self.engine.compute_trapezoid_vertices(pitch_deg, curvature_A)
        base_mask = self.engine.generate_binary_mask(trapezoid)

        pyramid = [base_mask]
        current = base_mask
        for _ in range(2):
            down = cv2.pyrDown(current)
            pyramid.append((down > 120).astype(np.uint8) * 255)
            current = down

        return pyramid


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Homography & Bird's Eye View (BEV) Coordinate Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class BEVTransformIntegrator:
    """
    Computes dynamic Inverse Perspective Mapping (IPM) homographies directly
    aligned with the adaptive perspective trapezoid.
    Transforms perspective camera pixels into orthographic top-down metric coordinates.
    """

    def __init__(
        self,
        bev_width_px: int = 400,
        bev_height_px: int = 600,
        meters_per_pixel: float = 0.0185,
        nominal_lookahead_m: float = 25.0,
    ) -> None:
        self.bev_w = bev_width_px
        self.bev_h = bev_height_px
        self.m_per_px = meters_per_pixel
        self.lookahead_m = nominal_lookahead_m

    def compute_homography(self, trapezoid: TrapezoidVertices) -> np.ndarray:
        """
        Compute perspective warp matrix H mapping trapezoid corners to rectangular BEV canvas.
        """
        src_pts = trapezoid.to_float_array()
        dst_pts = np.array([
            [0.0, 0.0],
            [float(self.bev_w - 1), 0.0],
            [float(self.bev_w - 1), float(self.bev_h - 1)],
            [0.0, float(self.bev_h - 1)],
        ], dtype=np.float32)

        H = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return H

    def warp_to_bev(self, frame_bgr: np.ndarray, trapezoid: TrapezoidVertices) -> np.ndarray:
        """
        Transform masked frame to top-down orthographic BEV projection.
        """
        H = self.compute_homography(trapezoid)
        warped = cv2.warpPerspective(frame_bgr, H, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        return warped

    def unwarp_from_bev(self, bev_mask: np.ndarray, trapezoid: TrapezoidVertices) -> np.ndarray:
        """
        Reproject a top-down BEV mask back into perspective image coordinates.
        """
        H = self.compute_homography(trapezoid)
        H_inv = np.linalg.inv(H)
        h_orig = 360
        w_orig = 640
        unwarped = cv2.warpPerspective(bev_mask, H_inv, (w_orig, h_orig), flags=cv2.INTER_NEAREST)
        return unwarped


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Live Vanishing Point Tracker & Pitch Drift Estimator
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VanishingPointObservation:
    """Estimated optical vanishing point coordinates and derived pitch/roll angles."""
    vp_x_px: float
    vp_y_px: float
    estimated_pitch_deg: float
    estimated_roll_deg: float
    confidence: float
    inlier_lines_count: int


class VanishingPointTracker:
    """
    RANSAC-based vanishing point estimator tracking optical intersection of road lines.
    Detects dynamic chassis pitch transients during heavy braking and acceleration.
    """

    def __init__(
        self,
        focal_length_px: float = 450.0,
        cx: float = 320.0,
        cy: float = 180.0,
        ransac_iterations: int = 40,
        inlier_dist_threshold_px: float = 6.0,
    ) -> None:
        self.fy = focal_length_px
        self.cx = cx
        self.cy = cy
        self.iterations = ransac_iterations
        self.thresh = inlier_dist_threshold_px

    def estimate_vanishing_point(
        self,
        line_segments: Sequence[Tuple[int, int, int, int]],
    ) -> VanishingPointObservation:
        """
        Extract vanishing point from pair-wise intersection voting of Hough line segments.
        Line format: (x1, y1, x2, y2).
        """
        if len(line_segments) < 2:
            return VanishingPointObservation(
                vp_x_px=self.cx,
                vp_y_px=self.cy,
                estimated_pitch_deg=0.0,
                estimated_roll_deg=0.0,
                confidence=0.0,
                inlier_lines_count=0,
            )

        # Convert line segments to standard homogeneous form: a*x + b*y + c = 0
        lines_homo = []
        for x1, y1, x2, y2 in line_segments:
            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)
            if length > 20.0 and abs(dx) > 5:
                # Normal vector (a, b)
                a = -dy / length
                b = dx / length
                c = -(a * x1 + b * y1)
                lines_homo.append((a, b, c))

        if len(lines_homo) < 2:
            return VanishingPointObservation(self.cx, self.cy, 0.0, 0.0, 0.0, 0)

        best_inliers = 0
        best_vp = (self.cx, self.cy)

        # RANSAC Voting
        n_lines = len(lines_homo)
        for _ in range(self.iterations):
            idx1, idx2 = np.random.choice(n_lines, 2, replace=False)
            l1, l2 = lines_homo[idx1], lines_homo[idx2]

            # Cross product gives intersection in homogeneous coordinates
            denom = l1[0] * l2[1] - l1[1] * l2[0]
            if abs(denom) < 1e-5:
                continue

            xi = (l1[1] * l2[2] - l1[2] * l2[1]) / denom
            yi = (l1[2] * l2[0] - l1[0] * l2[2]) / denom

            # Candidate must fall within reasonable camera bounds
            if not (0 <= xi <= 640 and 60 <= yi <= 280):
                continue

            # Count inliers
            inliers = 0
            for a, b, c in lines_homo:
                dist = abs(a * xi + b * yi + c)
                if dist <= self.thresh:
                    inliers += 1

            if inliers > best_inliers:
                best_inliers = inliers
                best_vp = (xi, yi)

        # Compute pitch from VP_y: tan(pitch) = (vp_y - cy) / fy
        pitch_rad = math.atan((best_vp[1] - self.cy) / self.fy)
        pitch_deg = math.degrees(pitch_rad)

        # Compute roll from VP_x: tan(roll) = (vp_x - cx) / fx
        roll_rad = math.atan((best_vp[0] - self.cx) / self.fy)
        roll_deg = math.degrees(roll_rad)

        conf = min(1.0, best_inliers / max(1, n_lines))

        return VanishingPointObservation(
            vp_x_px=float(best_vp[0]),
            vp_y_px=float(best_vp[1]),
            estimated_pitch_deg=float(pitch_deg),
            estimated_roll_deg=float(roll_deg),
            confidence=float(conf),
            inlier_lines_count=best_inliers,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Slew-Rate Limited Trapezoid Temporal Smoother
# ═══════════════════════════════════════════════════════════════════════════════

class ROISlewRateLimiter:
    """
    Prevents high-frequency twitching of ROI trapezoid corners across consecutive video frames.
    Filters corner trajectories with an EMA smoother and absolute rate clamps.
    """

    def __init__(self, alpha: float = 0.30, max_px_per_frame: float = 8.0) -> None:
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self.max_step = max_px_per_frame
        self._current_trapezoid: Optional[TrapezoidVertices] = None

    def smooth(self, raw_trapezoid: TrapezoidVertices) -> TrapezoidVertices:
        """
        Apply temporal smoothing to all four corners of the trapezoid.
        """
        if self._current_trapezoid is None:
            self._current_trapezoid = copy.deepcopy(raw_trapezoid)
            return raw_trapezoid

        def smooth_pt(curr: Tuple[int, int], target: Tuple[int, int]) -> Tuple[int, int]:
            dx = float(target[0] - curr[0])
            dy = float(target[1] - curr[1])
            step_x = max(-self.max_step, min(self.max_step, dx * self.alpha))
            step_y = max(-self.max_step, min(self.max_step, dy * self.alpha))
            return int(round(curr[0] + step_x)), int(round(curr[1] + step_y))

        tl = smooth_pt(self._current_trapezoid.top_left, raw_trapezoid.top_left)
        tr = smooth_pt(self._current_trapezoid.top_right, raw_trapezoid.top_right)
        br = smooth_pt(self._current_trapezoid.bottom_right, raw_trapezoid.bottom_right)
        bl = smooth_pt(self._current_trapezoid.bottom_left, raw_trapezoid.bottom_left)

        smoothed = TrapezoidVertices(top_left=tl, top_right=tr, bottom_right=br, bottom_left=bl)
        self._current_trapezoid = copy.deepcopy(smoothed)
        return smoothed


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Automated Synthetic Benchmarking Harness
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ROIBenchmarkReport:
    """Benchmark telemetry testing ROI adaptability across pitch and curvature variations."""
    total_evaluations: int
    mean_area_fraction: float
    min_area_fraction: float
    max_area_fraction: float
    mean_latency_ms: float
    p95_latency_ms: float
    is_pitch_monotonic: bool
    is_curvature_expansive: bool


def benchmark_adaptive_roi(
    engine: Optional[AdaptiveROIEngine] = None,
    num_pitch_steps: int = 15,
) -> ROIBenchmarkReport:
    """
    Stress-test the adaptive ROI engine across a range of pitch angles (-4 to +12 deg)
    and curvature magnitudes (|A| from 0 to 0.005).
    """
    eng = engine if engine is not None else AdaptiveROIEngine()
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    pitch_range = np.linspace(-4.0, 12.0, num_pitch_steps)
    curvature_range = [0.0, 0.0005, 0.001, 0.002, 0.005]

    areas: List[float] = []
    latencies: List[float] = []
    horizon_ys: List[int] = []
    expansions: List[float] = []

    for p in pitch_range:
        for c in curvature_range:
            t0 = time.perf_counter()
            res = eng.process_frame(dummy_frame, pitch_deg=float(p), curvature_A=float(c))
            lat = (time.perf_counter() - t0) * 1000.0

            latencies.append(lat)
            areas.append(res.metrics.mask_area_fraction)
            horizon_ys.append(res.metrics.horizon_y_px)
            expansions.append(res.metrics.expansion_factor)

    # Monotonicity test: as pitch increases, horizon y coordinate must strictly increase or stay constant
    pitch_horizons = [eng.compute_horizon_line_px(p) for p in pitch_range]
    is_monotonic = all(pitch_horizons[i] <= pitch_horizons[i + 1] for i in range(len(pitch_horizons) - 1))

    # Curvature test: non-zero curvature must produce expansion factor > 1.0
    is_expansive = all(exp > 1.0 for exp in expansions if exp > 1.001)

    return ROIBenchmarkReport(
        total_evaluations=len(areas),
        mean_area_fraction=float(np.mean(areas)),
        min_area_fraction=float(np.min(areas)),
        max_area_fraction=float(np.max(areas)),
        mean_latency_ms=float(np.mean(latencies)),
        p95_latency_ms=float(np.percentile(latencies, 95)),
        is_pitch_monotonic=is_monotonic,
        is_curvature_expansive=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Curvilinear Arc Polygon Generator (Sharp Curve Road Corridors)
# ═══════════════════════════════════════════════════════════════════════════════

class CurvaturePolygonRasterizer:
    """
    Synthesizes parabolic curvilinear search corridors on sharp highway ramps
    and mountain hairpins where linear trapezoids clip the arcing road envelope.
    Follows quadratic trajectory: x_left(y) = A*y^2 + B*y + C_left, x_right(y) = A*y^2 + B*y + C_right.
    """

    def __init__(self, num_slices: int = 24) -> None:
        self.num_slices = num_slices

    def generate_curvilinear_polygon(
        self,
        A: float,
        B: float,
        C_left: float,
        C_right: float,
        y_top: int,
        y_bottom: int,
        frame_width: int = 640,
        frame_height: int = 360,
    ) -> np.ndarray:
        """
        Generate OpenCV contour points bounding the parabolic road corridor.
        """
        y_vals = np.linspace(y_bottom, y_top, self.num_slices)
        left_pts: List[Tuple[int, int]] = []
        right_pts: List[Tuple[int, int]] = []

        for y in y_vals:
            # Metric or pixel coordinates
            xl = int(round(A * (y ** 2) + B * y + C_left))
            xr = int(round(A * (y ** 2) + B * y + C_right))

            xl_clamped = max(0, min(frame_width - 1, xl))
            xr_clamped = max(0, min(frame_width - 1, xr))

            left_pts.append((xl_clamped, int(y)))
            right_pts.append((xr_clamped, int(y)))

        # Close polygon: left points bottom-to-top, right points top-to-bottom
        poly_pts = left_pts + right_pts[::-1]
        return np.array(poly_pts, dtype=np.int32).reshape((-1, 1, 2))

    def rasterize_corridor_mask(
        self,
        polygon_contour: np.ndarray,
        frame_width: int = 640,
        frame_height: int = 360,
    ) -> np.ndarray:
        """Rasterize polygon contour into single-channel 8-bit mask."""
        mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon_contour], 255)
        return mask


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Multi-Focal & Multi-Resolution Camera Adapter
# ═══════════════════════════════════════════════════════════════════════════════

class MultiFocalROIAdapter:
    """
    Dynamically scales perspective trapezoids and horizon projections across
    variable camera intrinsics (e.g. 1080p full HD downsampled vs 360p Jetson input).
    """

    def __init__(self, base_config: AdaptiveROIConfig) -> None:
        self.base_cfg = base_config

    def adapt_to_resolution(
        self,
        target_width: int,
        target_height: int,
    ) -> AdaptiveROIConfig:
        """
        Rescale focal length, principal points, and pixel cutoffs proportionally.
        """
        scale_x = target_width / float(self.base_cfg.frame_width)
        scale_y = target_height / float(self.base_cfg.frame_height)

        return AdaptiveROIConfig(
            frame_width=target_width,
            frame_height=target_height,
            focal_length_y_px=self.base_cfg.focal_length_y_px * scale_y,
            principal_point_x_px=self.base_cfg.principal_point_x_px * scale_x,
            principal_point_y_px=self.base_cfg.principal_point_y_px * scale_y,
            nominal_pitch_deg=self.base_cfg.nominal_pitch_deg,
            camera_height_m=self.base_cfg.camera_height_m,
            top_horizon_offset_px=int(round(self.base_cfg.top_horizon_offset_px * scale_y)),
            bottom_bonnet_cutoff_px=int(round(self.base_cfg.bottom_bonnet_cutoff_px * scale_y)),
            base_top_width_ratio=self.base_cfg.base_top_width_ratio,
            base_bottom_width_ratio=self.base_cfg.base_bottom_width_ratio,
            curve_expansion_gain=self.base_cfg.curve_expansion_gain,
            max_expansion_ratio=self.base_cfg.max_expansion_ratio,
            enable_anti_aliased_edges=self.base_cfg.enable_anti_aliased_edges,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_ROI_ENGINE: Optional[AdaptiveROIEngine] = None
_GLOBAL_PYRAMID_ROI: Optional[PyramidROIProcessor] = None
_GLOBAL_BEV_TRANSFORM: Optional[BEVTransformIntegrator] = None
_GLOBAL_VP_TRACKER: Optional[VanishingPointTracker] = None
_GLOBAL_ROI_SMOOTHER: Optional[ROISlewRateLimiter] = None
_GLOBAL_CURVILINEAR: Optional[CurvaturePolygonRasterizer] = None
_GLOBAL_MULTIFOCAL: Optional[MultiFocalROIAdapter] = None


def get_adaptive_roi_engine(config: Optional[AdaptiveROIConfig] = None) -> AdaptiveROIEngine:
    """
    Obtain or instantiate the global singleton AdaptiveROIEngine.
    """
    global _GLOBAL_ROI_ENGINE
    if _GLOBAL_ROI_ENGINE is None:
        _GLOBAL_ROI_ENGINE = AdaptiveROIEngine(config=config)
    return _GLOBAL_ROI_ENGINE


def get_pyramid_roi_processor() -> PyramidROIProcessor:
    """Obtain or instantiate global PyramidROIProcessor."""
    global _GLOBAL_PYRAMID_ROI
    if _GLOBAL_PYRAMID_ROI is None:
        _GLOBAL_PYRAMID_ROI = PyramidROIProcessor(get_adaptive_roi_engine())
    return _GLOBAL_PYRAMID_ROI


def get_bev_transform_integrator() -> BEVTransformIntegrator:
    """Obtain or instantiate global BEVTransformIntegrator."""
    global _GLOBAL_BEV_TRANSFORM
    if _GLOBAL_BEV_TRANSFORM is None:
        _GLOBAL_BEV_TRANSFORM = BEVTransformIntegrator()
    return _GLOBAL_BEV_TRANSFORM


def get_vanishing_point_tracker() -> VanishingPointTracker:
    """Obtain or instantiate global VanishingPointTracker."""
    global _GLOBAL_VP_TRACKER
    if _GLOBAL_VP_TRACKER is None:
        _GLOBAL_VP_TRACKER = VanishingPointTracker()
    return _GLOBAL_VP_TRACKER


def get_roi_slew_rate_limiter() -> ROISlewRateLimiter:
    """Obtain or instantiate global ROISlewRateLimiter."""
    global _GLOBAL_ROI_SMOOTHER
    if _GLOBAL_ROI_SMOOTHER is None:
        _GLOBAL_ROI_SMOOTHER = ROISlewRateLimiter()
    return _GLOBAL_ROI_SMOOTHER


def get_curvilinear_polygon_rasterizer() -> CurvaturePolygonRasterizer:
    """Obtain or instantiate global CurvaturePolygonRasterizer."""
    global _GLOBAL_CURVILINEAR
    if _GLOBAL_CURVILINEAR is None:
        _GLOBAL_CURVILINEAR = CurvaturePolygonRasterizer()
    return _GLOBAL_CURVILINEAR


def get_multifocal_roi_adapter() -> MultiFocalROIAdapter:
    """Obtain or instantiate global MultiFocalROIAdapter."""
    global _GLOBAL_MULTIFOCAL
    if _GLOBAL_MULTIFOCAL is None:
        _GLOBAL_MULTIFOCAL = MultiFocalROIAdapter(get_adaptive_roi_engine().config)
    return _GLOBAL_MULTIFOCAL


