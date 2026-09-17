"""
src/lka_engine/lka_state.py
===========================
APEX LKA — Software Lane Keep Assist Analysis Engine.

Performs:
1. Lane geometry analysis (lane width, heading error, curvature).
2. Temporal stabilization (STABLE, UNCERTAIN, TEMPORARILY_LOST, LANE_LOST).
3. Lane Departure Warning (LDW) with multi-frame hysteresis.
4. Failure and edge-case diagnostics.
5. Software steering correction recommendation (NO HARDWARE ACTUATION).
"""

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional, Tuple

import numpy as np

from src.inference.pipeline import InferenceResult
from src.lane_geometry.lane_estimator import LaneGeometry
from src.lane_geometry.offset_calculator import (
    DriftDirection,
    OffsetResult,
    SteeringRecommendation,
)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════

class LaneStabilityState(Enum):
    STABLE = "STABLE"
    UNCERTAIN = "UNCERTAIN"
    TEMPORARILY_LOST = "TEMPORARILY LOST"
    LANE_LOST = "LANE LOST"


class LDWState(Enum):
    NORMAL = "NORMAL"
    DRIFT_LEFT = "DRIFT LEFT"
    DRIFT_RIGHT = "DRIFT RIGHT"
    WARNING = "WARNING"
    LANE_LOST = "LANE LOST"


class FailureCondition(Enum):
    NONE = "NONE"
    ONE_BOUNDARY_MISSING = "ONE BOUNDARY MISSING"
    BOTH_BOUNDARIES_MISSING = "BOTH BOUNDARIES MISSING"
    POOR_CONFIDENCE = "POOR CONFIDENCE"
    SUDDEN_LANE_SHIFT = "SUDDEN LANE SHIFT"
    EXCESSIVE_CURVATURE = "EXCESSIVE CURVATURE"
    CAMERA_UNAVAILABLE = "CAMERA UNAVAILABLE"
    INFERENCE_FAILURE = "INFERENCE FAILURE"


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Calibration estimate: standard lane ~3.7m spans ~200px at bottom of 640x360 frame
METERS_PER_PIXEL = 3.7 / 200.0  # ~0.0185 meters per pixel

# Thresholds
NORMAL_OFFSET_NORM_THRESH = 0.08   # Within ±8% of lane center = NORMAL
WARNING_OFFSET_NORM_THRESH = 0.25  # Beyond ±25% = WARNING
HYSTERESIS_FRAMES = 3              # Frames required to trigger / clear LDW WARNING
MAX_TEMPORARY_LOST_FRAMES = 3      # Frames to hold brief dropout before LANE_LOST
SUDDEN_SHIFT_THRESH_NORM = 0.18    # Sudden lateral jump threshold per frame
EXCESSIVE_CURVATURE_RADIUS_M = 150.0  # Sharp curve threshold


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry Output Container
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LKATelemetry:
    """
    Comprehensive software-only LKA telemetry packet.
    """
    # Timestamps & counters
    frame_id: int = 0
    timestamp: float = 0.0

    # Geometry & Positions
    vehicle_center_x: float = 320.0
    lane_center_x: Optional[float] = None
    lateral_offset_px: Optional[float] = None
    lateral_offset_m: Optional[float] = None
    lateral_offset_norm: Optional[float] = None
    smoothed_offset_norm: Optional[float] = None

    # Heading & Width
    heading_error_deg: Optional[float] = None
    smoothed_heading_deg: Optional[float] = None
    lane_width_px: Optional[float] = None
    lane_width_m: Optional[float] = None
    curvature_radius_m: Optional[float] = None

    # Visibility & Confidence
    left_detected: bool = False
    right_detected: bool = False
    left_confidence: float = 0.0
    right_confidence: float = 0.0
    combined_confidence: float = 0.0

    # System & ADAS States
    stability_state: LaneStabilityState = LaneStabilityState.LANE_LOST
    ldw_state: LDWState = LDWState.LANE_LOST
    drift_direction: DriftDirection = DriftDirection.UNKNOWN
    failure_condition: FailureCondition = FailureCondition.NONE

    # Software Steering Recommendation (STRICTLY NON-ACTUATING)
    steering_recommendation: SteeringRecommendation = SteeringRecommendation.NO_RECOMMENDATION
    recommended_angle_deg: float = 0.0
    steering_command: Optional[float] = None

    # Diagnostics & Timings
    consecutive_detected_frames: int = 0
    consecutive_lost_frames: int = 0
    inference_ms: float = 0.0
    total_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# LKA State Tracker
# ═══════════════════════════════════════════════════════════════════════════

class LKAStateTracker:
    """
    Stateful analysis engine for real-time lane tracking.
    Maintains temporal stabilization, heading error estimation, LDW hysteresis,
    and diagnostic edge-case detection across sequential frames.
    """

    def __init__(self, history_len: int = 30) -> None:
        self.history_len = history_len
        self.frame_count = 0
        self.consecutive_detected = 0
        self.consecutive_lost = 0

        # Rolling history of offsets and headings for smoothing
        self._offset_history: Deque[float] = deque(maxlen=history_len)
        self._heading_history: Deque[float] = deque(maxlen=history_len)

        # Last valid measurements for temporal stabilization during dropouts
        self._last_valid_geometry: Optional[LaneGeometry] = None
        self._last_valid_offset_norm: Optional[float] = None
        self._last_valid_heading_deg: Optional[float] = None

        # Hysteresis counters
        self._warning_counter = 0
        self._normal_counter = 0
        self._current_ldw_state = LDWState.NORMAL

    def reset(self) -> None:
        """Reset state when switching streams or resetting video."""
        self.frame_count = 0
        self.consecutive_detected = 0
        self.consecutive_lost = 0
        self._offset_history.clear()
        self._heading_history.clear()
        self._last_valid_geometry = None
        self._last_valid_offset_norm = None
        self._last_valid_heading_deg = None
        self._warning_counter = 0
        self._normal_counter = 0
        self._current_ldw_state = LDWState.NORMAL

    def update(self, result: InferenceResult) -> LKATelemetry:
        """
        Ingest an InferenceResult, perform temporal analysis, and return LKATelemetry.
        """
        self.frame_count += 1
        now = time.time()

        # Extract basic inference data
        has_pred = result.prediction is not None
        left_det = bool(result.prediction.left_detected) if has_pred else False
        right_det = bool(result.prediction.right_detected) if has_pred else False
        left_conf = float(result.prediction.left_confidence) if has_pred else 0.0
        right_conf = float(result.prediction.right_confidence) if has_pred else 0.0
        combined_conf = (left_conf + right_conf) / 2.0 if (left_det and right_det) else max(left_conf, right_conf)

        geo = result.geometry
        off = result.offset

        # Check detection validity
        is_detected = (
            result.success
            and geo is not None
            and geo.geometry_valid
            and off is not None
            and off.lateral_error_norm is not None
        )

        if is_detected:
            self.consecutive_detected += 1
            self.consecutive_lost = 0
            self._last_valid_geometry = geo
        else:
            self.consecutive_lost += 1
            self.consecutive_detected = 0

        # ── 1. Calculate Geometry & Width ──────────────────────────────────
        lane_width_px: Optional[float] = None
        lane_width_m: Optional[float] = None
        if geo and geo.x_left_bottom is not None and geo.x_right_bottom is not None:
            lane_width_px = round(abs(geo.x_right_bottom - geo.x_left_bottom), 1)
            lane_width_m = round(lane_width_px * METERS_PER_PIXEL, 2)
        elif self._last_valid_geometry and self.consecutive_lost <= MAX_TEMPORARY_LOST_FRAMES:
            prev_geo = self._last_valid_geometry
            if prev_geo.x_left_bottom is not None and prev_geo.x_right_bottom is not None:
                lane_width_px = round(abs(prev_geo.x_right_bottom - prev_geo.x_left_bottom), 1)
                lane_width_m = round(lane_width_px * METERS_PER_PIXEL, 2)

        # Curvature radius
        curv_radius_m = geo.curvature_left_m if (geo and geo.curvature_left_m is not None) else None
        if curv_radius_m is None and geo and geo.curvature_right_m is not None:
            curv_radius_m = geo.curvature_right_m

        # ── 2. Calculate Heading Error (degrees) ───────────────────────────
        raw_heading_deg = self._compute_heading_deg(geo)

        # ── 3. Temporal Stability Analysis ────────────────────────────────
        raw_offset_norm = off.lateral_error_norm if (off and off.lateral_error_norm is not None) else None
        stability_state, effective_offset_norm, effective_heading_deg = self._determine_stability(
            is_detected, raw_offset_norm, raw_heading_deg, combined_conf
        )

        # Update smoothing buffers
        if effective_offset_norm is not None:
            self._offset_history.append(effective_offset_norm)
            smoothed_offset_norm = float(np.mean(self._offset_history))
            self._last_valid_offset_norm = smoothed_offset_norm
        else:
            smoothed_offset_norm = None

        if effective_heading_deg is not None:
            self._heading_history.append(effective_heading_deg)
            smoothed_heading_deg = float(np.mean(self._heading_history))
            self._last_valid_heading_deg = smoothed_heading_deg
        else:
            smoothed_heading_deg = None

        # Lateral offset in meters and pixels
        veh_ctr = off.vehicle_center_x if off else 320.0
        lane_ctr = off.lane_center_x if off else None
        lat_offset_px = off.lateral_error_px if off else None
        lat_offset_m = round(lat_offset_px * METERS_PER_PIXEL, 3) if lat_offset_px is not None else None

        # ── 4. Lane Departure Warning (LDW) with Hysteresis ────────────────
        ldw_state = self._compute_ldw_state(stability_state, effective_offset_norm)

        # ── 5. Failure / Edge-Case Diagnostics ────────────────────────────
        failure_cond = self._detect_failure_condition(
            result, left_det, right_det, combined_conf, raw_offset_norm, curv_radius_m
        )

        # ── 6. Software Steering Recommendation (Software Only) ───────────
        rec_angle_deg, rec_direction = self._compute_steering_recommendation(
            stability_state, smoothed_offset_norm, smoothed_heading_deg, off
        )

        # Build telemetry object
        inf_ms = result.timings.get("inference_ms", 0.0)
        tot_ms = result.timings.get("total_ms", 0.0)

        return LKATelemetry(
            frame_id=self.frame_count,
            timestamp=now,
            vehicle_center_x=veh_ctr,
            lane_center_x=lane_ctr,
            lateral_offset_px=lat_offset_px,
            lateral_offset_m=lat_offset_m,
            lateral_offset_norm=effective_offset_norm,
            smoothed_offset_norm=smoothed_offset_norm,
            heading_error_deg=effective_heading_deg,
            smoothed_heading_deg=smoothed_heading_deg,
            lane_width_px=lane_width_px,
            lane_width_m=lane_width_m,
            curvature_radius_m=curv_radius_m,
            left_detected=left_det,
            right_detected=right_det,
            left_confidence=left_conf,
            right_confidence=right_conf,
            combined_confidence=combined_conf,
            stability_state=stability_state,
            ldw_state=ldw_state,
            drift_direction=off.drift_direction if off else DriftDirection.UNKNOWN,
            failure_condition=failure_cond,
            steering_recommendation=rec_direction,
            recommended_angle_deg=rec_angle_deg,
            steering_command=off.steering_command if off else None,
            consecutive_detected_frames=self.consecutive_detected,
            consecutive_lost_frames=self.consecutive_lost,
            inference_ms=inf_ms,
            total_ms=tot_ms,
        )

    def _compute_heading_deg(self, geo: Optional[LaneGeometry]) -> Optional[float]:
        """
        Estimate heading error angle (degrees) from fitted polynomial tangent at image bottom (y = model_h).
        dx/dy = 2*a*y + b (for quadratic x = a*y^2 + b*y + c).
        Heading angle relative to lane centerline = -arctan(dx/dy).
        """
        if geo is None:
            return None

        h = float(geo.model_h)
        slopes: List[float] = []

        for poly in (geo.left_poly, geo.right_poly):
            if poly is not None and len(poly) >= 2:
                if len(poly) == 3:
                    # quadratic: a*y^2 + b*y + c -> dx/dy = 2*a*y + b
                    a, b, _ = poly
                    dx_dy = 2.0 * a * h + b
                else:
                    # linear: m*y + c -> dx/dy = m
                    dx_dy = float(poly[0])
                slopes.append(dx_dy)

        if not slopes:
            return None

        avg_dx_dy = float(np.mean(slopes))
        # Negative sign: in image coordinates, y increases downwards.
        # An angle tilting rightwards (dx/dy > 0) means vehicle heading right (+ heading error).
        heading_rad = math.atan(avg_dx_dy)
        return round(math.degrees(heading_rad), 1)

    def _determine_stability(
        self,
        is_detected: bool,
        raw_offset_norm: Optional[float],
        raw_heading_deg: Optional[float],
        conf: float,
    ) -> Tuple[LaneStabilityState, Optional[float], Optional[float]]:
        """
        Determine stability state and return stabilized offset & heading values.
        """
        if is_detected and raw_offset_norm is not None:
            # Check sudden lateral jump
            is_sudden_jump = False
            if self._last_valid_offset_norm is not None:
                if abs(raw_offset_norm - self._last_valid_offset_norm) > SUDDEN_SHIFT_THRESH_NORM:
                    is_sudden_jump = True

            if self.consecutive_detected >= 3 and conf >= 0.4 and not is_sudden_jump:
                return LaneStabilityState.STABLE, raw_offset_norm, raw_heading_deg
            else:
                return LaneStabilityState.UNCERTAIN, raw_offset_norm, raw_heading_deg

        # Not detected in this frame:
        if self.consecutive_lost <= MAX_TEMPORARY_LOST_FRAMES and self._last_valid_offset_norm is not None:
            # Temporarily lost grace period: hold previous values briefly
            return LaneStabilityState.TEMPORARILY_LOST, self._last_valid_offset_norm, self._last_valid_heading_deg

        # Completely lost
        return LaneStabilityState.LANE_LOST, None, None

    def _compute_ldw_state(
        self, stability: LaneStabilityState, offset_norm: Optional[float]
    ) -> LDWState:
        """
        Lane Departure Warning state machine with multi-frame hysteresis.
        Prevents flickering alerts from noisy single-frame spikes.
        """
        if stability == LaneStabilityState.LANE_LOST or offset_norm is None:
            self._current_ldw_state = LDWState.LANE_LOST
            return LDWState.LANE_LOST

        abs_off = abs(offset_norm)

        if abs_off >= WARNING_OFFSET_NORM_THRESH:
            self._warning_counter += 1
            self._normal_counter = 0
            if self._warning_counter >= HYSTERESIS_FRAMES:
                self._current_ldw_state = LDWState.WARNING
            elif self._current_ldw_state != LDWState.WARNING:
                self._current_ldw_state = LDWState.DRIFT_RIGHT if offset_norm > 0 else LDWState.DRIFT_LEFT
        elif abs_off <= NORMAL_OFFSET_NORM_THRESH:
            self._normal_counter += 1
            self._warning_counter = 0
            if self._normal_counter >= HYSTERESIS_FRAMES:
                self._current_ldw_state = LDWState.NORMAL
        else:
            # Drift zone
            self._warning_counter = 0
            self._normal_counter = 0
            if self._current_ldw_state != LDWState.WARNING:
                self._current_ldw_state = LDWState.DRIFT_RIGHT if offset_norm > 0 else LDWState.DRIFT_LEFT

        return self._current_ldw_state

    def _detect_failure_condition(
        self,
        result: InferenceResult,
        left_det: bool,
        right_det: bool,
        conf: float,
        raw_offset_norm: Optional[float],
        curv_m: Optional[float],
    ) -> FailureCondition:
        """
        Detect difficult edge-cases and diagnostic failure modes.
        """
        if result.error or (result.prediction and result.prediction.error_message):
            return FailureCondition.INFERENCE_FAILURE

        if not left_det and not right_det:
            return FailureCondition.BOTH_BOUNDARIES_MISSING

        if left_det ^ right_det:
            return FailureCondition.ONE_BOUNDARY_MISSING

        if conf < 0.25:
            return FailureCondition.POOR_CONFIDENCE

        if (
            raw_offset_norm is not None
            and self._last_valid_offset_norm is not None
            and abs(raw_offset_norm - self._last_valid_offset_norm) > SUDDEN_SHIFT_THRESH_NORM
        ):
            return FailureCondition.SUDDEN_LANE_SHIFT

        if curv_m is not None and 0 < curv_m < EXCESSIVE_CURVATURE_RADIUS_M:
            return FailureCondition.EXCESSIVE_CURVATURE

        return FailureCondition.NONE

    def _compute_steering_recommendation(
        self,
        stability: LaneStabilityState,
        offset_norm: Optional[float],
        heading_deg: Optional[float],
        off: Optional[OffsetResult],
    ) -> Tuple[float, SteeringRecommendation]:
        """
        Calculate software steering angle recommendation.
        Sign: positive angle = steer LEFT (correcting rightward drift).
              negative angle = steer RIGHT (correcting leftward drift).
        Clamped to [-30°, +30°]. Strictly for software visualization.
        """
        if stability == LaneStabilityState.LANE_LOST or offset_norm is None:
            return 0.0, SteeringRecommendation.NO_RECOMMENDATION

        # Proportional terms
        kp_angle = 35.0  # degrees per 1.0 normalized offset
        k_head = 0.5     # degrees per degree of heading error

        h_val = heading_deg if heading_deg is not None else 0.0
        # Positive offset (right of center) -> positive steering angle (turn LEFT)
        angle = (offset_norm * kp_angle - h_val * k_head)
        angle = max(-30.0, min(30.0, angle))
        angle = round(angle, 1)

        if abs(angle) < 1.5:
            rec = SteeringRecommendation.KEEP_CENTER
        elif angle > 0:
            rec = SteeringRecommendation.STEER_LEFT
        else:
            rec = SteeringRecommendation.STEER_RIGHT

        return angle, rec
