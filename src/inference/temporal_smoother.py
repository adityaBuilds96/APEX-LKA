"""
src/inference/temporal_smoother.py
===================================
Temporal Lane Smoothing & Frame-to-Frame Stabilization Engine.

Key Features & Guarantees
-------------------------
1. Exponential Moving Average (EMA) on Polynomial Coefficients:
   smoothed_coefs = alpha * current + (1 - alpha) * previous_smoothed
   Eliminates jitter and flickering across streaming video frames.
2. Outlier Detection:
   Rejects abrupt coordinate or lateral offset jumps (> 30% change) caused
   by momentary shadows, glare, or guardrail reflections.
3. Gap Interpolation & Graceful Degradation:
   Interpolates lane trajectory across short occlusions/missing detections
   (up to max_gap_frames=3) using smoothed historical models.
4. Gradual Confidence Decay:
   Decays confidence progressively during occlusions rather than dropping
   abruptly to 0, ensuring smooth control transitions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np

from src.inference.predictor import DetectionStatus, LanePrediction
from src.control.ekf_tracker import EKF8StateLaneTracker


@dataclass
class SmootherConfig:
    """Configuration parameters for temporal lane smoothing."""
    enabled: bool = True
    buffer_size: int = 5
    ema_alpha: float = 0.6
    outlier_threshold: float = 0.30
    max_gap_frames: int = 3
    confidence_decay: float = 0.85


def rasterize_poly_mask(
    coeffs: Optional[np.ndarray],
    h: int,
    w: int,
    horizon_y: int,
    thickness: int = 6,
) -> np.ndarray:
    """
    Render 2nd-degree polynomial curve x = a*y^2 + b*y + c into a binary mask (uint8).
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    if coeffs is None or len(coeffs) != 3:
        return mask

    y_vals = np.arange(max(0, horizon_y), h, dtype=np.float32)
    x_vals = np.polyval(coeffs, y_vals)

    valid = (x_vals >= 0) & (x_vals < w)
    if not np.any(valid):
        return mask

    pts = np.column_stack((x_vals[valid].astype(np.int32), y_vals[valid].astype(np.int32)))
    if len(pts) >= 2:
        cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)

    return mask


class TemporalLaneSmoother:
    """
    Stateful rolling smoother for video and real-time camera streaming.
    """

    def __init__(
        self,
        buffer_size: int = 5,
        ema_alpha: float = 0.6,
        outlier_threshold: float = 0.30,
        max_gap_frames: int = 3,
        confidence_decay: float = 0.85,
        enable_ekf: bool = False,
    ) -> None:
        self.buffer_size = buffer_size
        self.ema_alpha = ema_alpha
        self.outlier_threshold = outlier_threshold
        self.max_gap_frames = max_gap_frames
        self.confidence_decay = confidence_decay
        self.enable_ekf = enable_ekf

        # State buffers
        self.left_history: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.right_history: Deque[np.ndarray] = deque(maxlen=buffer_size)

        self.smoothed_left_coeffs: Optional[np.ndarray] = None
        self.smoothed_right_coeffs: Optional[np.ndarray] = None

        self.last_left_conf: float = 0.0
        self.last_right_conf: float = 0.0
        self.last_bottom_center: Optional[float] = None

        self.gap_count: int = 0
        self.total_frames_processed: int = 0
        self.ekf = EKF8StateLaneTracker()

    def reset(self) -> None:
        """Reset state across video switches or stream interruptions."""
        self.left_history.clear()
        self.right_history.clear()
        self.smoothed_left_coeffs = None
        self.smoothed_right_coeffs = None
        self.last_left_conf = 0.0
        self.last_right_conf = 0.0
        self.last_bottom_center = None
        self.gap_count = 0
        self.total_frames_processed = 0
        self.ekf.reset()

    def _extract_poly_coeffs(self, mask: Optional[np.ndarray], h: int, w: int) -> Optional[np.ndarray]:
        """Fit polynomial coefficients x = a*y^2 + b*y + c from binary mask."""
        if mask is None or not mask.any():
            return None
        ys, xs = np.where(mask > 0)
        if len(ys) < 15:
            return None
        try:
            coeffs = np.polyfit(ys, xs, deg=2)
            return coeffs
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _calculate_bottom_intercept(self, coeffs: Optional[np.ndarray], y_bottom: int) -> Optional[float]:
        """Compute x intercept at bottom of image."""
        if coeffs is None:
            return None
        return float(np.polyval(coeffs, y_bottom))

    def smooth(
        self,
        prediction: LanePrediction,
        horizon_y: Optional[int] = None,
    ) -> LanePrediction:
        """
        Apply temporal smoothing, outlier rejection, and gap interpolation to a prediction.
        """
        self.total_frames_processed += 1
        h, w = prediction.model_h, prediction.model_w
        if horizon_y is None:
            horizon_y = int(0.35 * h)

        # ── 1. Extract polynomial coefficients ──────────────────────────────
        curr_left_coeffs = prediction.left_poly_coeffs
        if curr_left_coeffs is None and prediction.left_mask is not None:
            curr_left_coeffs = self._extract_poly_coeffs(prediction.left_mask, h, w)

        curr_right_coeffs = prediction.right_poly_coeffs
        if curr_right_coeffs is None and prediction.right_mask is not None:
            curr_right_coeffs = self._extract_poly_coeffs(prediction.right_mask, h, w)

        has_detection = (
            prediction.status != DetectionStatus.LANE_NOT_DETECTED
            and ((curr_left_coeffs is not None) or (curr_right_coeffs is not None))
        )

        # ── 2. Outlier Detection ───────────────────────────────────────────
        is_outlier = False
        if has_detection and self.last_bottom_center is not None:
            left_bot = self._calculate_bottom_intercept(curr_left_coeffs, h - 1)
            right_bot = self._calculate_bottom_intercept(curr_right_coeffs, h - 1)
            prev_left = self._calculate_bottom_intercept(self.smoothed_left_coeffs, h - 1) if self.smoothed_left_coeffs is not None else None
            prev_right = self._calculate_bottom_intercept(self.smoothed_right_coeffs, h - 1) if self.smoothed_right_coeffs is not None else None

            jumps = []
            if left_bot is not None and prev_left is not None:
                jumps.append(abs(left_bot - prev_left) / float(w))
            if right_bot is not None and prev_right is not None:
                jumps.append(abs(right_bot - prev_right) / float(w))

            curr_center = None
            if left_bot is not None and right_bot is not None:
                curr_center = (left_bot + right_bot) / 2.0
            elif left_bot is not None and prev_right is not None:
                curr_center = (left_bot + prev_right) / 2.0
            elif right_bot is not None and prev_left is not None:
                curr_center = (prev_left + right_bot) / 2.0

            if curr_center is not None:
                jumps.append(abs(curr_center - self.last_bottom_center) / float(w))

            if jumps and max(jumps) > self.outlier_threshold:
                is_outlier = True


        # ── 3. Handle Valid Detections (EMA Smoothing) ─────────────────────
        if has_detection and not is_outlier:
            self.gap_count = 0

            # Left Lane EMA
            if curr_left_coeffs is not None:
                if self.smoothed_left_coeffs is not None:
                    self.smoothed_left_coeffs = (
                        self.ema_alpha * curr_left_coeffs
                        + (1.0 - self.ema_alpha) * self.smoothed_left_coeffs
                    )
                else:
                    self.smoothed_left_coeffs = curr_left_coeffs.copy()
                self.left_history.append(curr_left_coeffs)
                self.last_left_conf = float(prediction.left_confidence)
            elif self.smoothed_left_coeffs is not None:
                # Left momentarily missing; decay confidence
                self.last_left_conf *= self.confidence_decay

            # Right Lane EMA
            if curr_right_coeffs is not None:
                if self.smoothed_right_coeffs is not None:
                    self.smoothed_right_coeffs = (
                        self.ema_alpha * curr_right_coeffs
                        + (1.0 - self.ema_alpha) * self.smoothed_right_coeffs
                    )
                else:
                    self.smoothed_right_coeffs = curr_right_coeffs.copy()
                self.right_history.append(curr_right_coeffs)
                self.last_right_conf = float(prediction.right_confidence)
            elif self.smoothed_right_coeffs is not None:
                # Right momentarily missing; decay confidence
                self.last_right_conf *= self.confidence_decay

            # Update bottom center tracking
            l_b = self._calculate_bottom_intercept(self.smoothed_left_coeffs, h - 1)
            r_b = self._calculate_bottom_intercept(self.smoothed_right_coeffs, h - 1)
            if l_b is not None and r_b is not None:
                self.last_bottom_center = (l_b + r_b) / 2.0
            elif l_b is not None:
                self.last_bottom_center = l_b + (0.25 * w)
            elif r_b is not None:
                self.last_bottom_center = r_b - (0.25 * w)

            # 8-state EKF Temporal Smoothing: [a_L, b_L, c_L, a_R, b_R, c_R, e_lat, theta_e]
            if self.enable_ekf:
                e_lat = (self.last_bottom_center - (w / 2.0)) if self.last_bottom_center is not None else None
                theta_e = None
                if self.smoothed_left_coeffs is not None and self.smoothed_right_coeffs is not None:
                    mean_b = 0.5 * (self.smoothed_left_coeffs[1] + self.smoothed_right_coeffs[1])
                    theta_e = float(np.arctan(mean_b))

                ekf_l, ekf_r, _, _, ekf_ok = self.ekf.update(
                    left_coeffs=self.smoothed_left_coeffs,
                    right_coeffs=self.smoothed_right_coeffs,
                    e_lat=e_lat,
                    theta_e=theta_e,
                )
                if ekf_ok:
                    if ekf_l is not None and self.smoothed_left_coeffs is not None:
                        self.smoothed_left_coeffs = ekf_l
                    if ekf_r is not None and self.smoothed_right_coeffs is not None:
                        self.smoothed_right_coeffs = ekf_r

            # Rasterize smoothed masks
            left_mask = rasterize_poly_mask(self.smoothed_left_coeffs, h, w, horizon_y)
            right_mask = rasterize_poly_mask(self.smoothed_right_coeffs, h, w, horizon_y)

            both = left_mask.any() and right_mask.any()
            one = left_mask.any() or right_mask.any()
            det_status = (
                DetectionStatus.LANE_DETECTED if both
                else (DetectionStatus.PARTIAL_LANE_DETECTED if one else DetectionStatus.LANE_NOT_DETECTED)
            )

            prediction.status = det_status
            prediction.left_mask = left_mask
            prediction.right_mask = right_mask
            prediction.left_poly_coeffs = self.smoothed_left_coeffs
            prediction.right_poly_coeffs = self.smoothed_right_coeffs
            prediction.left_confidence = self.last_left_conf
            prediction.right_confidence = self.last_right_conf
            prediction.accuracy_score = float(np.clip(
                0.4 * self.last_left_conf + 0.4 * self.last_right_conf + 0.2 * prediction.geometry_plausibility,
                0.0, 1.0,
            ))
            return prediction

        # ── 4. Missing Detection or Outlier: Gap Interpolation ──────────────
        self.gap_count += 1
        can_interpolate = (
            self.gap_count <= self.max_gap_frames
            and (self.smoothed_left_coeffs is not None or self.smoothed_right_coeffs is not None)
        )

        if can_interpolate:
            # EKF prediction hold across 1–3 frame line dropouts if enabled
            if self.enable_ekf:
                ekf_l, ekf_r, _, _, ekf_ok = self.ekf.update(None, None)
                if ekf_ok:
                    if ekf_l is not None and self.smoothed_left_coeffs is not None:
                        self.smoothed_left_coeffs = ekf_l
                    if ekf_r is not None and self.smoothed_right_coeffs is not None:
                        self.smoothed_right_coeffs = ekf_r

            # Decay confidence gradually
            decay_factor = self.confidence_decay ** self.gap_count
            interp_left_conf = self.last_left_conf * decay_factor
            interp_right_conf = self.last_right_conf * decay_factor

            left_mask = rasterize_poly_mask(self.smoothed_left_coeffs, h, w, horizon_y)
            right_mask = rasterize_poly_mask(self.smoothed_right_coeffs, h, w, horizon_y)

            both = left_mask.any() and right_mask.any()
            one = left_mask.any() or right_mask.any()
            det_status = (
                DetectionStatus.LANE_DETECTED if both
                else (DetectionStatus.PARTIAL_LANE_DETECTED if one else DetectionStatus.LANE_NOT_DETECTED)
            )

            prediction.status = det_status
            prediction.left_mask = left_mask
            prediction.right_mask = right_mask
            prediction.left_poly_coeffs = self.smoothed_left_coeffs
            prediction.right_poly_coeffs = self.smoothed_right_coeffs
            prediction.left_confidence = interp_left_conf
            prediction.right_confidence = interp_right_conf
            prediction.accuracy_score = float(np.clip(
                0.4 * interp_left_conf + 0.4 * interp_right_conf + 0.2 * 0.5,
                0.0, 1.0,
            ))
            return prediction

        # Gap exceeded or no prior smoothed state → Reset
        if self.gap_count > self.max_gap_frames:
            self.smoothed_left_coeffs = None
            self.smoothed_right_coeffs = None
            self.last_bottom_center = None
            self.ekf.reset()


        prediction.status = DetectionStatus.LANE_NOT_DETECTED
        prediction.left_confidence = 0.0
        prediction.right_confidence = 0.0
        prediction.accuracy_score = 0.0
        return prediction


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4: Autonomous Multi-Task Scene Perception Path Smoother
# ═══════════════════════════════════════════════════════════════════════════

class PathTrackingState(str, Enum):
    TRACKING = "TRACKING"
    INTERPOLATING = "INTERPOLATING"
    TEMPORARILY_LOST = "TEMPORARILY_LOST"


@dataclass
class SmoothedPathResult:
    """Stabilized polynomial coefficients and temporal tracking telemetry."""
    left_poly: Optional[np.ndarray] = None
    right_poly: Optional[np.ndarray] = None
    center_poly: Optional[np.ndarray] = None
    state: PathTrackingState = PathTrackingState.TEMPORARILY_LOST
    gap_count: int = 0
    is_outlier: bool = False
    interpolated: bool = False


class TemporalPathSmoother:
    """
    Temporal anti-flicker filter for drivable corridor boundaries and center path.

    Maintains a rolling ring buffer of N=5 polynomial coefficients (a, b, c).
    Applies Exponential Moving Average (alpha=0.65) to eliminate video flicker.
    Rejects lateral coordinate jumps exceeding max_lateral_jump_m (0.50m) as anomalies.
    Maintains last known trajectory for up to hold_gap_frames (3) dropped frames
    before transitioning to TEMPORARILY_LOST.
    """

    def __init__(
        self,
        buffer_depth: int = 5,
        ema_alpha: float = 0.65,
        max_lateral_jump_m: float = 0.50,
        hold_gap_frames: int = 3,
    ) -> None:
        self.buffer_depth = int(buffer_depth)
        self.ema_alpha = float(ema_alpha)
        self.max_lateral_jump_m = float(max_lateral_jump_m)
        self.hold_gap_frames = int(hold_gap_frames)

        # Ring buffers for coefficients [a, b, c]
        self.left_history: Deque[np.ndarray] = deque(maxlen=self.buffer_depth)
        self.right_history: Deque[np.ndarray] = deque(maxlen=self.buffer_depth)
        self.center_history: Deque[np.ndarray] = deque(maxlen=self.buffer_depth)

        # Smoothed state
        self.smoothed_left: Optional[np.ndarray] = None
        self.smoothed_right: Optional[np.ndarray] = None
        self.smoothed_center: Optional[np.ndarray] = None

        self.state: PathTrackingState = PathTrackingState.TEMPORARILY_LOST
        self.gap_count: int = 0
        self.total_frames: int = 0

    def reset(self) -> None:
        """Reset history buffers and smoothed trajectory state."""
        self.left_history.clear()
        self.right_history.clear()
        self.center_history.clear()
        self.smoothed_left = None
        self.smoothed_right = None
        self.smoothed_center = None
        self.state = PathTrackingState.TEMPORARILY_LOST
        self.gap_count = 0
        self.total_frames = 0

    def _eval_bottom_intercept(self, poly: Optional[np.ndarray], y_bottom: float) -> Optional[float]:
        """Compute horizontal coordinate x at y = y_bottom."""
        if poly is None or len(poly) != 3:
            return None
        try:
            val = float(np.polyval(poly, y_bottom))
            return val if np.isfinite(val) else None
        except Exception:
            return None

    def _check_outlier(
        self,
        new_poly: Optional[np.ndarray],
        prev_poly: Optional[np.ndarray],
        y_bottom: float,
        meters_per_pixel: float,
    ) -> bool:
        """Return True if single-frame lateral jump exceeds max_lateral_jump_m."""
        if new_poly is None or prev_poly is None:
            return False

        x_new = self._eval_bottom_intercept(new_poly, y_bottom)
        x_prev = self._eval_bottom_intercept(prev_poly, y_bottom)

        if x_new is None or x_prev is None:
            return False

        lateral_jump_m = abs(x_new - x_prev) * meters_per_pixel
        return lateral_jump_m > self.max_lateral_jump_m

    def _apply_ema(
        self,
        new_poly: Optional[np.ndarray],
        prev_poly: Optional[np.ndarray],
        history: Deque[np.ndarray],
    ) -> Optional[np.ndarray]:
        """Apply exponential moving average to polynomial coefficients."""
        if new_poly is None:
            return prev_poly

        new_poly_arr = np.asarray(new_poly, dtype=np.float32)
        if prev_poly is not None:
            smoothed = (
                self.ema_alpha * new_poly_arr
                + (1.0 - self.ema_alpha) * np.asarray(prev_poly, dtype=np.float32)
            )
        else:
            smoothed = new_poly_arr.copy()

        history.append(new_poly_arr)
        return smoothed

    def smooth(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        center_poly: Optional[np.ndarray],
        img_h: int = 360,
        img_w: int = 640,
        meters_per_pixel: Optional[float] = None,
    ) -> SmoothedPathResult:
        """
        Smooth left, right, and center polynomial paths across consecutive frames.

        Parameters
        ----------
        left_poly : Optional[np.ndarray]
            Raw detected left boundary polynomial [a, b, c].
        right_poly : Optional[np.ndarray]
            Raw detected right boundary polynomial [a, b, c].
        center_poly : Optional[np.ndarray]
            Raw detected center trajectory polynomial [a, b, c].
        img_h : int
            Frame height.
        img_w : int
            Frame width.
        meters_per_pixel : Optional[float]
            Lateral scale factor at bottom bumper row. Defaults to ground calibration (~0.0055 m/px).

        Returns
        -------
        SmoothedPathResult
            Anti-flicker stabilized polynomials and tracking status.
        """
        self.total_frames += 1
        y_bottom = float(img_h - 1)

        # Scale at bottom of image: ~3.7m road width across ~600px of perspective width
        if meters_per_pixel is None or meters_per_pixel <= 0.0:
            m_per_px = 3.7 / max(300.0, img_w * 0.55)
        else:
            m_per_px = float(meters_per_pixel)

        # ── 1. Outlier Detection ───────────────────────────────────────────
        is_left_outlier = self._check_outlier(left_poly, self.smoothed_left, y_bottom, m_per_px)
        is_right_outlier = self._check_outlier(right_poly, self.smoothed_right, y_bottom, m_per_px)
        is_center_outlier = self._check_outlier(center_poly, self.smoothed_center, y_bottom, m_per_px)

        is_any_outlier = is_left_outlier or is_right_outlier or is_center_outlier

        valid_left = left_poly if not is_left_outlier else None
        valid_right = right_poly if not is_right_outlier else None
        valid_center = center_poly if not is_center_outlier else None

        has_valid = (valid_left is not None) or (valid_right is not None) or (valid_center is not None)

        # ── 2. Tracking State: Normal EMA update ───────────────────────────
        if has_valid:
            self.gap_count = 0
            self.state = PathTrackingState.TRACKING

            if valid_left is not None:
                self.smoothed_left = self._apply_ema(valid_left, self.smoothed_left, self.left_history)
            if valid_right is not None:
                self.smoothed_right = self._apply_ema(valid_right, self.smoothed_right, self.right_history)
            if valid_center is not None:
                self.smoothed_center = self._apply_ema(valid_center, self.smoothed_center, self.center_history)

            return SmoothedPathResult(
                left_poly=self.smoothed_left,
                right_poly=self.smoothed_right,
                center_poly=self.smoothed_center,
                state=PathTrackingState.TRACKING,
                gap_count=0,
                is_outlier=is_any_outlier,
                interpolated=False,
            )

        # ── 3. Missing or Outlier: Gap Holding / Interpolation ──────────────
        self.gap_count += 1
        has_prior = (
            self.smoothed_left is not None
            or self.smoothed_right is not None
            or self.smoothed_center is not None
        )

        if self.gap_count <= self.hold_gap_frames and has_prior:
            self.state = PathTrackingState.INTERPOLATING
            return SmoothedPathResult(
                left_poly=self.smoothed_left,
                right_poly=self.smoothed_right,
                center_poly=self.smoothed_center,
                state=PathTrackingState.INTERPOLATING,
                gap_count=self.gap_count,
                is_outlier=is_any_outlier,
                interpolated=True,
            )

        # ── 4. Gap Exceeded: Reset to TEMPORARILY_LOST ─────────────────────
        self.state = PathTrackingState.TEMPORARILY_LOST
        self.smoothed_left = None
        self.smoothed_right = None
        self.smoothed_center = None

        return SmoothedPathResult(
            left_poly=None,
            right_poly=None,
            center_poly=None,
            state=PathTrackingState.TEMPORARILY_LOST,
            gap_count=self.gap_count,
            is_outlier=is_any_outlier,
            interpolated=False,
        )
