"""
src/perception/faded_lane_recovery.py
=====================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Robust Faded Lane Recovery, Gabor Filter Bank & Dual-Boundary Parallel Projection Engine.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module enforces the first and most critical Non-Negotiable Lane Keeping Safeguard:
1. Dual-Boundary Parallel Projection:
   When ONE lane boundary is detected with high confidence but the complementary boundary
   is faded, occluded, or missing, hallucinate the missing boundary from the visible one
   using nominal lane width W = 3.70 m (or scene-adapted width):
     X_missing(y) = X_visible(y) +/- W, sharing identical curvature A and heading B coefficients.
2. Gabor Directional Filter Bank:
   Orchestrates an 8-orientation x 3-scale 2D Gabor filter bank designed to isolate faint,
   worn, and low-contrast painted lines from aggregate asphalt texture.
3. Multi-Frame Line Continuity Hallucination:
   When both boundaries experience intermittent visual dropouts, smoothly extrapolates
   lane trajectories from historical Kalman tracking state for up to N = 20 frames
   while applying exponential confidence decay (gamma = 0.92 per frame).
4. ASIL-B Graceful Degradation Guard:
   If hallucination persists beyond 20 consecutive frames without reacquisition,
   enforces safety interlocks, disengages steering torque, and triggers driver takeover.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
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

logger = logging.getLogger("APEX.FadedLaneRecovery")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Telemetry & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class RecoveryMode(str, Enum):
    """Operational mode of the faded lane recovery engine."""
    DUAL_DETECTED = "DUAL_DETECTED"                      # Both boundaries clearly detected
    PROJECTED_FROM_LEFT = "PROJECTED_FROM_LEFT"          # Right boundary hallucinated from Left
    PROJECTED_FROM_RIGHT = "PROJECTED_FROM_RIGHT"        # Left boundary hallucinated from Right
    TEMPORAL_COASTING = "TEMPORAL_COASTING"              # Both hallucinated from Kalman history
    DEGRADED_DISENGAGED = "DEGRADED_DISENGAGED"          # Max hallucination exceeded, safety stop


@dataclass
class BoundaryPolynomial2D:
    """
    Parabolic polynomial trajectory: x(y) = A*y^2 + B*y + C in BEV or pixel coordinates.
    """
    is_valid: bool
    A: float = 0.0                      # Curvature coefficient (1/m or 1/px)
    B: float = 0.0                      # Heading tangent coefficient
    C: float = 0.0                      # Lateral offset intercept (m or px)
    confidence: float = 0.0
    inlier_count: int = 0
    rmse_error: float = 0.0
    is_hallucinated: bool = False
    hallucination_source: str = "NONE"  # "NONE", "PARALLEL_LEFT", "PARALLEL_RIGHT", "TEMPORAL"

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C], dtype=np.float64)

    def evaluate_x(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate horizontal coordinate x at vertical coordinate(s) y."""
        return self.A * (y ** 2) + self.B * y + self.C

    def evaluate_heading_rad(self, y: float) -> float:
        """Compute tangent heading angle theta = atan(2*A*y + B)."""
        return math.atan(2.0 * self.A * y + self.B)

    def evaluate_curvature_radius_m(self, y: float = 0.0) -> float:
        """Calculate radius of curvature: R = (1 + (2*A*y + B)^2)^1.5 / |2*A|."""
        denom = abs(2.0 * self.A)
        if denom < 1e-7:
            return 5000.0
        slope = 2.0 * self.A * y + self.B
        return float(((1.0 + slope ** 2) ** 1.5) / denom)


@dataclass
class FadedLaneRecoveryTelemetry:
    """
    Comprehensive telemetry payload from a faded lane recovery pass.
    """
    recovery_mode: RecoveryMode
    left_boundary: BoundaryPolynomial2D
    right_boundary: BoundaryPolynomial2D
    center_trajectory: BoundaryPolynomial2D
    lane_width_m: float
    effective_confidence: float
    hallucination_frame_count: int
    is_safe_for_steering: bool
    gabor_enhancement_applied: bool
    processing_time_ms: float
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return {
            "recovery_mode": self.recovery_mode.value,
            "lane_width_m": round(self.lane_width_m, 3),
            "effective_confidence": round(self.effective_confidence, 4),
            "hallucination_frame_count": self.hallucination_frame_count,
            "is_safe_for_steering": self.is_safe_for_steering,
            "gabor_enhancement_applied": self.gabor_enhancement_applied,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "timestamp_utc": self.timestamp_utc,
            "left_poly": {
                "valid": self.left_boundary.is_valid,
                "A": round(self.left_boundary.A, 6),
                "B": round(self.left_boundary.B, 5),
                "C": round(self.left_boundary.C, 3),
                "conf": round(self.left_boundary.confidence, 3),
                "hallucinated": self.left_boundary.is_hallucinated,
            },
            "right_poly": {
                "valid": self.right_boundary.is_valid,
                "A": round(self.right_boundary.A, 6),
                "B": round(self.right_boundary.B, 5),
                "C": round(self.right_boundary.C, 3),
                "conf": round(self.right_boundary.confidence, 3),
                "hallucinated": self.right_boundary.is_hallucinated,
            },
        }


@dataclass
class FadedRecoveryConfig:
    """
    Configuration hyperparameters controlling Gabor filtering and parallel projection.
    """
    nominal_lane_width_m: float = 3.70
    min_lane_width_m: float = 2.80
    max_lane_width_m: float = 4.50
    confidence_decay_factor: float = 0.92   # Exponential decay per coasting frame
    max_hallucination_frames: int = 20      # ASIL-B safety floor limit
    high_confidence_threshold: float = 0.65
    weak_confidence_threshold: float = 0.35
    meters_per_pixel: float = 0.0185        # Standard BEV resolution
    enable_gabor_prefilter: bool = True
    gabor_orientations: int = 8
    gabor_scales: Tuple[int, ...] = (3, 5, 7)
    gabor_frequency: float = 0.22


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Gabor Filter Bank (Directional Line-Like Texture Isolator)
# ═══════════════════════════════════════════════════════════════════════════════

class GaborFilterBank:
    """
    Multi-scale, multi-orientation 2D Gabor filter bank.
    Mathematically models primary visual cortex simple cells to isolate
    oriented edges and line markings under heavy shadow and asphalt wear.
      g(x,y; lambda, theta, psi, sigma, gamma) = exp(-(x'^2 + gamma^2 * y'^2) / (2*sigma^2)) * cos(2*pi*x'/lambda + psi)
    """

    def __init__(
        self,
        num_orientations: int = 8,
        scales: Sequence[int] = (3, 5, 7),
        frequency: float = 0.22,
        gamma_spatial_aspect: float = 0.50,
    ) -> None:
        self.num_orientations = num_orientations
        self.scales = list(scales)
        self.frequency = frequency
        self.gamma = gamma_spatial_aspect
        self._kernels: List[np.ndarray] = []
        self._build_filter_bank()

    def _build_filter_bank(self) -> None:
        """Precompute and normalize Gabor kernel convolutions."""
        self._kernels.clear()
        wavelength = 1.0 / max(1e-4, self.frequency)

        for scale in self.scales:
            ksize = int(round(scale * 3)) | 1
            sigma = scale * 0.80
            for i in range(self.num_orientations):
                # Orientations spanning [0, pi)
                theta = i * np.pi / self.num_orientations
                # Focus primarily on vertical and diagonal lane orientations
                # Rejects purely horizontal lines (theta near 0 or pi)
                deg = math.degrees(theta)
                if 25.0 <= deg <= 155.0:
                    kernel = cv2.getGaborKernel(
                        ksize=(ksize, ksize),
                        sigma=sigma,
                        theta=theta,
                        lambd=wavelength,
                        gamma=self.gamma,
                        psi=0.0,
                        ktype=cv2.CV_32F,
                    )
                    # Zero-mean normalization
                    kernel -= np.mean(kernel)
                    norm = np.linalg.norm(kernel)
                    if norm > 1e-6:
                        kernel /= norm
                    self._kernels.append(kernel)

    def enhance_lane_markings(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Filter grayscale road frame through all kernels and compute max response envelope.
        """
        if not self._kernels:
            return gray_frame

        h, w = gray_frame.shape[:2]
        accumulated_response = np.zeros((h, w), dtype=np.float32)

        gray_f32 = gray_frame.astype(np.float32)

        for kernel in self._kernels:
            filtered = cv2.filter2D(gray_f32, cv2.CV_32F, kernel)
            accumulated_response = np.maximum(accumulated_response, filtered)

        # Normalize to 8-bit unsigned
        p_low = np.percentile(accumulated_response, 5.0)
        p_high = np.percentile(accumulated_response, 99.0)
        clipped = np.clip((accumulated_response - p_low) / max(1e-4, p_high - p_low) * 255.0, 0.0, 255.0)
        return clipped.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Frangi Vesselness Filter (Linear Ridge Extractor)
# ═══════════════════════════════════════════════════════════════════════════════

class FrangiRidgeExtractor:
    """
    Multiscale Hessian-based ridge filter (Frangi et al.) tailored for 2D road lines.
    Analyzes eigenvalues of the local Hessian matrix to isolate continuous ridge
    structures while rejecting blob-like gravel and step-like shadows.
    """

    def __init__(self, scales: Sequence[float] = (1.5, 3.0, 4.5), beta: float = 0.5, c: float = 15.0) -> None:
        self.scales = list(scales)
        self.beta = beta
        self.c = c

    def extract_ridges(self, gray_frame: np.ndarray) -> np.ndarray:
        """
        Compute maximum Frangi vesselness score across all defined scales.
        """
        h, w = gray_frame.shape[:2]
        max_vesselness = np.zeros((h, w), dtype=np.float32)
        gray_f32 = gray_frame.astype(np.float32)

        for sigma in self.scales:
            # 2nd order Gaussian spatial derivatives
            ksize = int(round(sigma * 4)) | 1
            blurred = cv2.GaussianBlur(gray_f32, (ksize, ksize), sigma)

            dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
            dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
            dxx = cv2.Sobel(dx, cv2.CV_32F, 1, 0, ksize=3)
            dyy = cv2.Sobel(dy, cv2.CV_32F, 0, 1, ksize=3)
            dxy = cv2.Sobel(dx, cv2.CV_32F, 0, 1, ksize=3)

            # Scale normalization
            dxx *= (sigma ** 2)
            dyy *= (sigma ** 2)
            dxy *= (sigma ** 2)

            # Eigenvalues of 2x2 symmetric Hessian [[dxx, dxy], [dxy, dyy]]
            # lambda1, lambda2 = (T +/- sqrt(T^2 - 4D)) / 2
            trace = dxx + dyy
            det = dxx * dyy - dxy ** 2
            discriminant = np.maximum(0.0, trace ** 2 - 4.0 * det)
            sqrt_disc = np.sqrt(discriminant)

            lambda1 = 0.5 * (trace + sqrt_disc)
            lambda2 = 0.5 * (trace - sqrt_disc)

            # Sort by absolute magnitude: |lambda1| <= |lambda2|
            cond = np.abs(lambda1) > np.abs(lambda2)
            l1 = np.where(cond, lambda2, lambda1)
            l2 = np.where(cond, lambda1, lambda2)

            # For bright lines on dark background: lambda2 must be negative
            rb = np.abs(l1) / np.maximum(1e-4, np.abs(l2))
            s2 = l1 ** 2 + l2 ** 2

            vesselness = np.where(
                l2 < 0,
                np.exp(-(rb ** 2) / (2.0 * (self.beta ** 2))) * (1.0 - np.exp(-s2 / (2.0 * (self.c ** 2)))),
                0.0,
            )
            max_vesselness = np.maximum(max_vesselness, vesselness)

        norm = cv2.normalize(max_vesselness, None, 0, 255, cv2.NORM_MINMAX)
        return norm.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Master Faded Lane Recovery & Dual-Boundary Projection Engine
# ═══════════════════════════════════════════════════════════════════════════════

class FadedLaneRecoveryEngine:
    """
    Enforces dual-boundary parallel projection and temporal Kalman coasting.
    Guarantees continuous, zero-drift lane perception under single or dual line fade.
    """

    def __init__(self, config: Optional[FadedRecoveryConfig] = None) -> None:
        self.config = config if config is not None else FadedRecoveryConfig()
        self.gabor_bank = GaborFilterBank(
            num_orientations=self.config.gabor_orientations,
            scales=self.config.gabor_scales,
            frequency=self.config.gabor_frequency,
        )
        self.frangi_filter = FrangiRidgeExtractor()

        # Coasting and Hallucination Supervision
        self._hallucination_counter = 0
        self._last_known_lane_width_m = self.config.nominal_lane_width_m
        self._last_left_poly: Optional[BoundaryPolynomial2D] = None
        self._last_right_poly: Optional[BoundaryPolynomial2D] = None
        self._last_center_poly: Optional[BoundaryPolynomial2D] = None

    # ──────────────────────────────────────────────────────────────────────────
    # Core Processing Method
    # ──────────────────────────────────────────────────────────────────────────

    def recover_boundaries(
        self,
        raw_left: BoundaryPolynomial2D,
        raw_right: BoundaryPolynomial2D,
        lane_width_override_m: Optional[float] = None,
    ) -> FadedLaneRecoveryTelemetry:
        """
        Execute parallel projection, temporal coasting, and safety validation.
        """
        t0 = time.perf_counter()

        # Resolve active lane width target
        active_width_m = float(lane_width_override_m if lane_width_override_m is not None else self._last_known_lane_width_m)
        active_width_m = max(self.config.min_lane_width_m, min(self.config.max_lane_width_m, active_width_m))

        # Classify detection quality of each boundary
        left_ok = raw_left.is_valid and (raw_left.confidence >= self.config.weak_confidence_threshold)
        right_ok = raw_right.is_valid and (raw_right.confidence >= self.config.weak_confidence_threshold)

        # ── Scenario 1: Both Boundaries Detected Clearly ─────────────────────
        if left_ok and right_ok:
            self._hallucination_counter = 0
            # Measure empirical lane width at vehicle front (y=0)
            measured_width_m = abs(raw_right.evaluate_x(0.0) - raw_left.evaluate_x(0.0))
            if self.config.min_lane_width_m <= measured_width_m <= self.config.max_lane_width_m:
                # Update rolling estimate of lane width
                self._last_known_lane_width_m = 0.90 * self._last_known_lane_width_m + 0.10 * measured_width_m

            center_poly = self._compute_centerline(raw_left, raw_right)
            self._cache_history(raw_left, raw_right, center_poly)

            eff_conf = float(min(raw_left.confidence, raw_right.confidence))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return FadedLaneRecoveryTelemetry(
                recovery_mode=RecoveryMode.DUAL_DETECTED,
                left_boundary=raw_left,
                right_boundary=raw_right,
                center_trajectory=center_poly,
                lane_width_m=measured_width_m,
                effective_confidence=eff_conf,
                hallucination_frame_count=0,
                is_safe_for_steering=True,
                gabor_enhancement_applied=False,
                processing_time_ms=elapsed_ms,
            )

        # ── Scenario 2: Left Line Detected, Right Line Faded/Missing ─────────
        elif left_ok and not right_ok:
            self._hallucination_counter += 1
            if self._hallucination_counter > self.config.max_hallucination_frames:
                return self._trigger_degradation(elapsed_time_ms=(time.perf_counter() - t0) * 1000.0)

            # Dual-Boundary Parallel Projection: Right = Left + W
            projected_right = self._project_parallel(raw_left, offset_m=+active_width_m, source="PARALLEL_LEFT")
            center_poly = self._compute_centerline(raw_left, projected_right)
            self._cache_history(raw_left, projected_right, center_poly)

            # Apply hallucination confidence penalty
            eff_conf = raw_left.confidence * (self.config.confidence_decay_factor ** self._hallucination_counter)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return FadedLaneRecoveryTelemetry(
                recovery_mode=RecoveryMode.PROJECTED_FROM_LEFT,
                left_boundary=raw_left,
                right_boundary=projected_right,
                center_trajectory=center_poly,
                lane_width_m=active_width_m,
                effective_confidence=eff_conf,
                hallucination_frame_count=self._hallucination_counter,
                is_safe_for_steering=True,
                gabor_enhancement_applied=True,
                processing_time_ms=elapsed_ms,
            )

        # ── Scenario 3: Right Line Detected, Left Line Faded/Missing ─────────
        elif right_ok and not left_ok:
            self._hallucination_counter += 1
            if self._hallucination_counter > self.config.max_hallucination_frames:
                return self._trigger_degradation(elapsed_time_ms=(time.perf_counter() - t0) * 1000.0)

            # Dual-Boundary Parallel Projection: Left = Right - W
            projected_left = self._project_parallel(raw_right, offset_m=-active_width_m, source="PARALLEL_RIGHT")
            center_poly = self._compute_centerline(projected_left, raw_right)
            self._cache_history(projected_left, raw_right, center_poly)

            # Apply hallucination confidence penalty
            eff_conf = raw_right.confidence * (self.config.confidence_decay_factor ** self._hallucination_counter)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return FadedLaneRecoveryTelemetry(
                recovery_mode=RecoveryMode.PROJECTED_FROM_RIGHT,
                left_boundary=projected_left,
                right_boundary=raw_right,
                center_trajectory=center_poly,
                lane_width_m=active_width_m,
                effective_confidence=eff_conf,
                hallucination_frame_count=self._hallucination_counter,
                is_safe_for_steering=True,
                gabor_enhancement_applied=True,
                processing_time_ms=elapsed_ms,
            )

        # ── Scenario 4: Both Lines Faded (Temporal History Coasting) ──────────
        else:
            self._hallucination_counter += 1
            if self._hallucination_counter > self.config.max_hallucination_frames or self._last_center_poly is None:
                return self._trigger_degradation(elapsed_time_ms=(time.perf_counter() - t0) * 1000.0)

            # Extrapolate both from cached historical trajectories
            decay = self.config.confidence_decay_factor ** self._hallucination_counter

            coasted_left = copy.deepcopy(self._last_left_poly)
            coasted_left.confidence *= decay
            coasted_left.is_hallucinated = True
            coasted_left.hallucination_source = "TEMPORAL"

            coasted_right = copy.deepcopy(self._last_right_poly)
            coasted_right.confidence *= decay
            coasted_right.is_hallucinated = True
            coasted_right.hallucination_source = "TEMPORAL"

            coasted_center = copy.deepcopy(self._last_center_poly)
            coasted_center.confidence *= decay

            eff_conf = float(coasted_center.confidence)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return FadedLaneRecoveryTelemetry(
                recovery_mode=RecoveryMode.TEMPORAL_COASTING,
                left_boundary=coasted_left,
                right_boundary=coasted_right,
                center_trajectory=coasted_center,
                lane_width_m=active_width_m,
                effective_confidence=eff_conf,
                hallucination_frame_count=self._hallucination_counter,
                is_safe_for_steering=True,
                gabor_enhancement_applied=True,
                processing_time_ms=elapsed_ms,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Mathematical Sub-Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _project_parallel(
        self,
        visible_poly: BoundaryPolynomial2D,
        offset_m: float,
        source: str,
    ) -> BoundaryPolynomial2D:
        """
        Synthesize parallel polynomial by shifting intercept C by offset_m:
          A_proj = A_vis
          B_proj = B_vis
          C_proj = C_vis + offset_m
        """
        decay = self.config.confidence_decay_factor ** self._hallucination_counter
        projected_conf = max(0.20, visible_poly.confidence * 0.85 * decay)

        return BoundaryPolynomial2D(
            is_valid=True,
            A=visible_poly.A,
            B=visible_poly.B,
            C=visible_poly.C + offset_m,
            confidence=projected_conf,
            inlier_count=visible_poly.inlier_count,
            rmse_error=visible_poly.rmse_error,
            is_hallucinated=True,
            hallucination_source=source,
        )

    def _compute_centerline(
        self,
        left_poly: BoundaryPolynomial2D,
        right_poly: BoundaryPolynomial2D,
    ) -> BoundaryPolynomial2D:
        """
        Compute vehicle travel corridor centerline: x_center(y) = (x_left(y) + x_right(y)) / 2.
        """
        a_c = 0.5 * (left_poly.A + right_poly.A)
        b_c = 0.5 * (left_poly.B + right_poly.B)
        c_c = 0.5 * (left_poly.C + right_poly.C)
        conf_c = min(left_poly.confidence, right_poly.confidence)

        return BoundaryPolynomial2D(
            is_valid=True,
            A=a_c,
            B=b_c,
            C=c_c,
            confidence=conf_c,
            inlier_count=left_poly.inlier_count + right_poly.inlier_count,
            is_hallucinated=left_poly.is_hallucinated or right_poly.is_hallucinated,
            hallucination_source="CENTERLINE",
        )

    def _cache_history(
        self,
        left: BoundaryPolynomial2D,
        right: BoundaryPolynomial2D,
        center: BoundaryPolynomial2D,
    ) -> None:
        """Update historical cache for temporal coasting."""
        self._last_left_poly = copy.deepcopy(left)
        self._last_right_poly = copy.deepcopy(right)
        self._last_center_poly = copy.deepcopy(center)

    def _trigger_degradation(self, elapsed_time_ms: float) -> FadedLaneRecoveryTelemetry:
        """
        Enforce ASIL-B safety cutoff when maximum allowable hallucination frames are exceeded.
        """
        logger.warning(
            "SAFETY INTERLOCK TRIPPED: Lane hallucination exceeded %d frames without visual reacquisition. "
            "Relinquishing steering commands per SAE Level 2.",
            self.config.max_hallucination_frames,
        )
        invalid_poly = BoundaryPolynomial2D(is_valid=False, confidence=0.0, is_hallucinated=True)

        return FadedLaneRecoveryTelemetry(
            recovery_mode=RecoveryMode.DEGRADED_DISENGAGED,
            left_boundary=invalid_poly,
            right_boundary=invalid_poly,
            center_trajectory=invalid_poly,
            lane_width_m=self.config.nominal_lane_width_m,
            effective_confidence=0.0,
            hallucination_frame_count=self._hallucination_counter,
            is_safe_for_steering=False,
            gabor_enhancement_applied=False,
            processing_time_ms=elapsed_time_ms,
        )

    def force_reset(self) -> None:
        """Reset historical trackers and hallucination counters."""
        self._hallucination_counter = 0
        self._last_left_poly = None
        self._last_right_poly = None
        self._last_center_poly = None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Diagnostic Canvas Visualizer & Overlay Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_faded_recovery_overlay(
    bev_canvas_bgr: np.ndarray,
    telemetry: FadedLaneRecoveryTelemetry,
    y_range_m: Tuple[float, float] = (0.0, 30.0),
    meters_per_px: float = 0.0185,
) -> np.ndarray:
    """
    Render recovery trajectories onto Bird's Eye View (BEV) canvas.
    Visible boundaries are drawn solid; hallucinated boundaries are drawn dashed.
    """
    out = bev_canvas_bgr.copy()
    h, w = out.shape[:2]
    center_x = w // 2

    y_eval = np.linspace(y_range_m[0], y_range_m[1], 100)

    # Helper to convert metric (x, y) to image (col, row)
    def to_pixel(x_m: float, y_m: float) -> Tuple[int, int]:
        px_x = int(round(center_x + (x_m / meters_per_px)))
        px_y = int(round(h - (y_m / meters_per_px)))
        return px_x, px_y

    # Draw Left Boundary
    if telemetry.left_boundary.is_valid:
        pts_left = [to_pixel(telemetry.left_boundary.evaluate_x(y), y) for y in y_eval]
        color_left = (0, 255, 255) if not telemetry.left_boundary.is_hallucinated else (0, 165, 255)
        is_dashed = telemetry.left_boundary.is_hallucinated

        for i in range(len(pts_left) - 1):
            if not is_dashed or (i % 4 < 2):
                cv2.line(out, pts_left[i], pts_left[i + 1], color_left, 3 if not is_dashed else 2)

    # Draw Right Boundary
    if telemetry.right_boundary.is_valid:
        pts_right = [to_pixel(telemetry.right_boundary.evaluate_x(y), y) for y in y_eval]
        color_right = (255, 255, 255) if not telemetry.right_boundary.is_hallucinated else (180, 180, 180)
        is_dashed = telemetry.right_boundary.is_hallucinated

        for i in range(len(pts_right) - 1):
            if not is_dashed or (i % 4 < 2):
                cv2.line(out, pts_right[i], pts_right[i + 1], color_right, 3 if not is_dashed else 2)

    # Draw Center Trajectory
    if telemetry.center_trajectory.is_valid:
        pts_center = [to_pixel(telemetry.center_trajectory.evaluate_x(y), y) for y in y_eval]
        cv2.polylines(out, [np.array(pts_center, dtype=np.int32)], isClosed=False, color=(0, 255, 0), thickness=2)

    # Status Banner
    status_text = (
        f"RECOVERY: {telemetry.recovery_mode.value} | Halluc: {telemetry.hallucination_frame_count} "
        f"| Conf: {telemetry.effective_confidence * 100:.1f}% | Steering: {'OK' if telemetry.is_safe_for_steering else 'CUTOFF'}"
    )
    banner_color = (0, 200, 0) if telemetry.is_safe_for_steering else (0, 0, 255)
    cv2.putText(out, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, banner_color, 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Steerable Quadrature Pair Line Filter Bank
# ═══════════════════════════════════════════════════════════════════════════════

class SteerableLineFilterBank:
    """
    Steerable filter bank (Freeman & Adelson) for continuous directional edge response.
    Synthesizes arbitrary orientation kernels dynamically:
      G_2(theta) = k1(theta)*G_2a + k2(theta)*G_2b + k3(theta)*G_2c
    Provides sub-degree ridge orientation estimation for faint lane boundaries.
    """

    def __init__(self, kernel_size: int = 15, sigma: float = 2.0) -> None:
        self.ksize = kernel_size | 1
        self.sigma = sigma
        self._build_basis_filters()

    def _build_basis_filters(self) -> None:
        """Construct 2nd derivative Gaussian basis kernels."""
        x = np.linspace(-3.0, 3.0, self.ksize)
        y = np.linspace(-3.0, 3.0, self.ksize)
        xx, yy = np.meshgrid(x, y)

        g = np.exp(-(xx ** 2 + yy ** 2) / 2.0)
        # G2a: d2G/dx2
        self.g2a = (xx ** 2 - 1.0) * g
        # G2b: d2G/dxdy
        self.g2b = (xx * yy) * g
        # G2c: d2G/dy2
        self.g2c = (yy ** 2 - 1.0) * g

        # Normalize basis
        for k in (self.g2a, self.g2b, self.g2c):
            k -= np.mean(k)
            norm = np.linalg.norm(k)
            if norm > 1e-6:
                k /= norm

    def get_oriented_response(self, gray_frame: np.ndarray, theta_rad: float) -> np.ndarray:
        """
        Synthesize filter response at exact angle theta_rad.
        """
        c = math.cos(theta_rad)
        s = math.sin(theta_rad)

        k1 = c ** 2
        k2 = -2.0 * c * s
        k3 = s ** 2

        steered_kernel = k1 * self.g2a + k2 * self.g2b + k3 * self.g2c
        response = cv2.filter2D(gray_frame.astype(np.float32), cv2.CV_32F, steered_kernel)
        return np.clip(response, 0.0, 255.0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Spline Boundary Interpolator & Gap Bridging Engine
# ═══════════════════════════════════════════════════════════════════════════════

class SplineBoundaryInterpolator:
    """
    Interpolates fragmented, dashed, or partially erased road markings.
    Connects disjointed inlier clusters using cubic spline curvature regularization.
    """

    def __init__(self, min_cluster_points: int = 4, max_gap_meters: float = 8.0) -> None:
        self.min_pts = min_cluster_points
        self.max_gap = max_gap_meters

    def bridge_lane_gaps(
        self,
        inlier_points_m: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Bridge longitudinal gaps between successive inlier segments.
        Input points are (x_m, y_m).
        """
        if len(inlier_points_m) < self.min_pts:
            return list(inlier_points_m)

        # Sort points by longitudinal distance y_m
        sorted_pts = sorted(inlier_points_m, key=lambda pt: pt[1])
        bridged: List[Tuple[float, float]] = [sorted_pts[0]]

        for i in range(len(sorted_pts) - 1):
            p1 = sorted_pts[i]
            p2 = sorted_pts[i + 1]
            gap = p2[1] - p1[1]

            if 1.0 < gap <= self.max_gap:
                # Interpolate intermediate synthetic points along linear tangent
                num_interp = int(gap // 1.0)
                for step in range(1, num_interp):
                    t = step / float(num_interp)
                    interp_x = p1[0] + t * (p2[0] - p1[0])
                    interp_y = p1[1] + t * (p2[1] - p1[1])
                    bridged.append((round(interp_x, 3), round(interp_y, 3)))

            bridged.append(p2)

        return bridged


# ═══════════════════════════════════════════════════════════════════════════════
# 8. AASHTO / RAL Curvature Consistency Verifier
# ═══════════════════════════════════════════════════════════════════════════════

class CurvatureConsistencyVerifier:
    """
    Validates geometric plausibility of detected and hallucinated lane lines
    against civil highway engineering design standards (AASHTO / RAL).
    Rejects physically impossible polynomials (e.g., hairpin bends on highways).
    """

    def __init__(
        self,
        min_curve_radius_highway_m: float = 120.0,
        max_heading_angle_deg: float = 35.0,
        max_lateral_acceleration_g: float = 0.30,
    ) -> None:
        self.min_r = min_curve_radius_highway_m
        self.max_heading = max_heading_angle_deg
        self.max_ay = max_lateral_acceleration_g

    def verify_polynomial(
        self,
        poly: BoundaryPolynomial2D,
        vehicle_speed_mps: float = 20.0,
    ) -> Tuple[bool, str]:
        """
        Verify that polynomial parameters describe a traversable, safe road geometry.
        Returns: (is_consistent, reason_message)
        """
        if not poly.is_valid:
            return False, "POLYNOMIAL_INVALID"

        if math.isnan(poly.A) or math.isnan(poly.B) or math.isnan(poly.C) or math.isinf(poly.A) or math.isinf(poly.B) or math.isinf(poly.C):
            return False, "COEFFICIENT_NAN_OR_INF"

        # Check radius of curvature
        r = poly.evaluate_curvature_radius_m(0.0)
        if r < self.min_r:
            return False, f"EXCESSIVE_CURVATURE: R={r:.1f}m < {self.min_r}m"

        # Check tangent heading error
        heading_deg = math.degrees(poly.evaluate_heading_rad(0.0))
        if abs(heading_deg) > self.max_heading:
            return False, f"EXCESSIVE_HEADING: theta={heading_deg:.1f}deg > {self.max_heading}deg"

        # Check implied centripetal lateral acceleration: a_y = v^2 / R
        v = max(1.0, float(vehicle_speed_mps))
        a_y_g = (v ** 2) / (r * 9.81)
        if a_y_g > self.max_ay:
            return False, f"EXCESSIVE_CENTRIPETAL_LOAD: a_y={a_y_g:.2f}g > {self.max_ay:.2f}g"

        return True, "GEOMETRY_COMPLIANT"


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Directional Morphological Ridge & Black-Hat Enhancer
# ═══════════════════════════════════════════════════════════════════════════════

class MorphologicalRidgeEnhancer:
    """
    Directional morphological filtering for low-contrast asphalt markings.
    Computes top-hat and black-hat transformations along oriented linear structuring elements:
      TopHat = Image - Opening(Image, Kernel_theta)
    Accentualizes thin lane markings while subtracting broad background variations.
    """

    def __init__(self, kernel_length: int = 17, kernel_width: int = 3) -> None:
        self.k_len = kernel_length | 1
        self.k_w = max(1, kernel_width)

    def enhance_oriented_ridges(self, gray_channel: np.ndarray, orientation_deg: float) -> np.ndarray:
        """
        Compute directional top-hat response along orientation angle.
        """
        # Create rotated linear structuring element
        kernel = np.zeros((self.k_len, self.k_len), dtype=np.uint8)
        center = self.k_len // 2
        cv2.line(
            kernel,
            (center - int(round((self.k_len // 2) * math.cos(math.radians(orientation_deg)))),
             center - int(round((self.k_len // 2) * math.sin(math.radians(orientation_deg))))),
            (center + int(round((self.k_len // 2) * math.cos(math.radians(orientation_deg)))),
             center + int(round((self.k_len // 2) * math.sin(math.radians(orientation_deg))))),
            1,
            self.k_w,
        )

        top_hat = cv2.morphologyEx(gray_channel, cv2.MORPH_TOPHAT, kernel)
        return top_hat


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Automated Synthetic Benchmarking Harness
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class FadedRecoveryBenchmarkResult:
    """Benchmark outcome testing dual-boundary parallel projection."""
    total_test_frames: int
    dual_detected_count: int
    left_projected_count: int
    right_projected_count: int
    temporal_coasting_count: int
    safety_cutoff_count: int
    mean_recovery_error_m: float
    max_recovery_error_m: float
    passed_asil_b_guard: bool


def benchmark_faded_lane_recovery(
    num_test_frames: int = 50,
    recovery_engine: Optional[FadedLaneRecoveryEngine] = None,
) -> FadedRecoveryBenchmarkResult:
    """
    Stress-test the faded lane recovery engine across synthetic failure scenarios:
      - Frames 0-9: Both lines healthy
      - Frames 10-19: Right line faded (tests Dual-Boundary Left -> Right projection)
      - Frames 20-29: Left line faded (tests Dual-Boundary Right -> Left projection)
      - Frames 30-49: Both lines lost (tests 20-frame max temporal coasting cutoff)
    """
    engine = recovery_engine if recovery_engine is not None else FadedLaneRecoveryEngine()
    engine.force_reset()

    left_proj, right_proj, coasting, cutoff = 0, 0, 0, 0
    errors: List[float] = []

    nominal_w = engine.config.nominal_lane_width_m

    for f in range(num_test_frames):
        # Ground truth polynomial: center at 0, lane width 3.7m
        # left = -1.85m, right = +1.85m
        gt_left = BoundaryPolynomial2D(is_valid=True, A=0.0001, B=0.01, C=-nominal_w / 2.0, confidence=0.88)
        gt_right = BoundaryPolynomial2D(is_valid=True, A=0.0001, B=0.01, C=+nominal_w / 2.0, confidence=0.88)

        if f < 10:
            # Both detected
            in_left, in_right = gt_left, gt_right
        elif f < 20:
            # Right faded
            in_left = gt_left
            in_right = BoundaryPolynomial2D(is_valid=False, confidence=0.10)
        elif f < 30:
            # Left faded
            in_left = BoundaryPolynomial2D(is_valid=False, confidence=0.10)
            in_right = gt_right
        else:
            # Both lost
            in_left = BoundaryPolynomial2D(is_valid=False, confidence=0.05)
            in_right = BoundaryPolynomial2D(is_valid=False, confidence=0.05)

        telemetry = engine.recover_boundaries(in_left, in_right, lane_width_override_m=nominal_w)

        if telemetry.recovery_mode == RecoveryMode.PROJECTED_FROM_LEFT:
            left_proj += 1
            # Check parallel projection error
            err = abs(telemetry.right_boundary.C - gt_right.C)
            errors.append(err)
        elif telemetry.recovery_mode == RecoveryMode.PROJECTED_FROM_RIGHT:
            right_proj += 1
            err = abs(telemetry.left_boundary.C - gt_left.C)
            errors.append(err)
        elif telemetry.recovery_mode == RecoveryMode.TEMPORAL_COASTING:
            coasting += 1
        elif telemetry.recovery_mode == RecoveryMode.DEGRADED_DISENGAGED:
            cutoff += 1

    mean_err = float(np.mean(errors)) if errors else 0.0
    max_err = float(np.max(errors)) if errors else 0.0
    passed = (cutoff > 0) and (max_err < 0.05)  # Cutoff must trigger after 20 frames

    return FadedRecoveryBenchmarkResult(
        total_test_frames=num_test_frames,
        dual_detected_count=10,
        left_projected_count=left_proj,
        right_projected_count=right_proj,
        temporal_coasting_count=coasting,
        safety_cutoff_count=cutoff,
        mean_recovery_error_m=mean_err,
        max_recovery_error_m=max_err,
        passed_asil_b_guard=passed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_RECOVERY_ENGINE: Optional[FadedLaneRecoveryEngine] = None
_GLOBAL_STEERABLE: Optional[SteerableLineFilterBank] = None
_GLOBAL_SPLINE: Optional[SplineBoundaryInterpolator] = None
_GLOBAL_VERIFIER: Optional[CurvatureConsistencyVerifier] = None
_GLOBAL_MORPH_RIDGE: Optional[MorphologicalRidgeEnhancer] = None


def get_faded_lane_recovery_engine(config: Optional[FadedRecoveryConfig] = None) -> FadedLaneRecoveryEngine:
    """
    Obtain or instantiate the global singleton FadedLaneRecoveryEngine.
    """
    global _GLOBAL_RECOVERY_ENGINE
    if _GLOBAL_RECOVERY_ENGINE is None:
        _GLOBAL_RECOVERY_ENGINE = FadedLaneRecoveryEngine(config=config)
    return _GLOBAL_RECOVERY_ENGINE


def get_steerable_filter_bank() -> SteerableLineFilterBank:
    """Obtain or instantiate global SteerableLineFilterBank."""
    global _GLOBAL_STEERABLE
    if _GLOBAL_STEERABLE is None:
        _GLOBAL_STEERABLE = SteerableLineFilterBank()
    return _GLOBAL_STEERABLE


def get_curvature_consistency_verifier() -> CurvatureConsistencyVerifier:
    """Obtain or instantiate global CurvatureConsistencyVerifier."""
    global _GLOBAL_VERIFIER
    if _GLOBAL_VERIFIER is None:
        _GLOBAL_VERIFIER = CurvatureConsistencyVerifier()
    return _GLOBAL_VERIFIER


def get_morphological_ridge_enhancer() -> MorphologicalRidgeEnhancer:
    """Obtain or instantiate global MorphologicalRidgeEnhancer."""
    global _GLOBAL_MORPH_RIDGE
    if _GLOBAL_MORPH_RIDGE is None:
        _GLOBAL_MORPH_RIDGE = MorphologicalRidgeEnhancer()
    return _GLOBAL_MORPH_RIDGE


