"""
src/lane_geometry/offset_calculator.py
========================================
Calculate lateral offset and generate a steering recommendation.

Coordinate convention (EXPLICIT — never assume)
------------------------------------------------
  vehicle_center_x = model_w / 2
      (camera is assumed to be mounted at the vehicle center line)

  lateral_error = vehicle_center_x - lane_center_x

  Sign convention:
    lateral_error > 0  →  vehicle is RIGHT of lane center  →  steer LEFT
    lateral_error < 0  →  vehicle is LEFT  of lane center  →  steer RIGHT
    lateral_error ≈ 0  →  vehicle is centered              →  KEEP CENTER

  Normalized lateral error:
    lateral_error_norm = lateral_error / (model_w / 2)
    Range: [-1, 1]
    -1 = far left of lane, +1 = far right of lane

Steering controller
-------------------
PD controller (Proportional-Derivative):
    steering_cmd = Kp * error_norm + Kd * d(error_norm)/dt

At single-image inference, dt is undefined so we use P-only.
When video/stream inference is running, the previous error is maintained
and the derivative term is used.

Kp, Kd are configurable in configs/project_config.yaml.

Safety
------
This module produces a SOFTWARE recommendation ONLY.
It DOES NOT send any signal to a motor, actuator, or CAN bus.
"""

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import cfg
from src.lane_geometry.lane_estimator import LaneGeometry

# ── Config parameters ──────────────────────────────────────────────────────
_Kp                     = cfg["steering"]["Kp"]
_Kd                     = cfg["steering"]["Kd"]
_CENTER_THRESHOLD_NORM  = cfg["steering"]["keep_center_threshold"]
_LOW_CONF_IOU           = cfg["steering"]["low_confidence_iou"]
_MAX_CMD                = cfg["steering"]["max_steering_command"]


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

class DriftDirection(Enum):
    LEFT        = "LEFT"
    RIGHT       = "RIGHT"
    CENTERED    = "CENTERED"
    UNKNOWN     = "UNKNOWN"


class SteeringRecommendation(Enum):
    STEER_LEFT       = "STEER LEFT"
    STEER_RIGHT      = "STEER RIGHT"
    KEEP_CENTER      = "KEEP CENTER"
    NO_RECOMMENDATION = "NO RECOMMENDATION"
    LOW_CONFIDENCE   = "LOW CONFIDENCE"


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class OffsetResult:
    """
    Full lateral-offset analysis result.

    All distances are in model-space pixels unless labelled _norm.

    Attributes
    ----------
    vehicle_center_x : float
        Pixel x of the camera/vehicle centerline (model_w / 2).
    lane_center_x : float | None
        Pixel x of the estimated lane center.
    lateral_error_px : float | None
        vehicle_center_x - lane_center_x  (+ = vehicle right, - = vehicle left)
    lateral_error_norm : float | None
        Normalized to [-1, 1] range.
    drift_direction : DriftDirection
    steering_command : float | None
        Signed float in [-max_cmd, +max_cmd]. Positive = steer left.
    recommendation : SteeringRecommendation
    confidence : float [0, 1]
        Geometric confidence: 1.0 if both lanes detected, 0.5 if one (estimated), 0.0 if none.
    one_lane_estimated : bool
        True if lane center was estimated from a single visible lane.
    calc_ms : float
    """
    # Positions
    vehicle_center_x:    float = 0.0
    lane_center_x:       Optional[float] = None

    # Error
    lateral_error_px:    Optional[float] = None
    lateral_error_norm:  Optional[float] = None

    # Direction & recommendation
    drift_direction:     DriftDirection          = DriftDirection.UNKNOWN
    steering_command:    Optional[float]         = None
    recommendation:      SteeringRecommendation  = SteeringRecommendation.NO_RECOMMENDATION

    # Confidence
    confidence:          float = 0.0
    one_lane_estimated:  bool  = False

    # Timing
    calc_ms:             float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Stateful calculator (maintains previous error for derivative term)
# ═══════════════════════════════════════════════════════════════════════════

class OffsetCalculator:
    """
    Stateful calculator that maintains previous error for PD control.

    For single-image inference, call calculate() once and the derivative
    term will be zero (no previous state).

    For video/streaming, keep one OffsetCalculator instance and call
    calculate() for each frame to get the derivative term.
    """

    def __init__(self) -> None:
        self._prev_error_norm: Optional[float] = None

    def reset(self) -> None:
        """Reset state (call when switching to a new video stream)."""
        self._prev_error_norm = None

    def calculate(
        self,
        geometry: LaneGeometry,
        detection_confidence: float = 1.0,
    ) -> OffsetResult:
        """
        Calculate lateral offset and steering recommendation.

        Parameters
        ----------
        geometry : LaneGeometry
            Output of lane_estimator.estimate_geometry()
        detection_confidence : float
            Combined ML/CV detection confidence [0, 1].
            Below _LOW_CONF_IOU → LOW_CONFIDENCE recommendation.

        Returns
        -------
        OffsetResult
        """
        t0 = time.perf_counter()
        model_w = geometry.model_w

        vehicle_cx = model_w / 2.0

        # ── No geometry → cannot compute ──────────────────────────────────
        if not geometry.geometry_valid or geometry.lane_center_x is None:
            self._prev_error_norm = None
            return OffsetResult(
                vehicle_center_x = vehicle_cx,
                recommendation   = SteeringRecommendation.NO_RECOMMENDATION,
                drift_direction  = DriftDirection.UNKNOWN,
                calc_ms          = (time.perf_counter() - t0) * 1000.0,
            )

        lane_cx = geometry.lane_center_x

        # ── Lateral error ──────────────────────────────────────────────────
        # vehicle_cx - lane_cx:
        #   positive → vehicle right of center → steer LEFT
        #   negative → vehicle left of center  → steer RIGHT
        err_px   = vehicle_cx - lane_cx
        err_norm = err_px / (model_w / 2.0)
        err_norm = max(-1.0, min(1.0, err_norm))   # clip to [-1, 1]

        # ── Drift direction ────────────────────────────────────────────────
        if abs(err_norm) < _CENTER_THRESHOLD_NORM:
            drift = DriftDirection.CENTERED
        elif err_norm > 0:
            drift = DriftDirection.RIGHT    # vehicle right → should steer left
        else:
            drift = DriftDirection.LEFT

        # ── PD steering command ────────────────────────────────────────────
        d_err_norm = 0.0
        if self._prev_error_norm is not None:
            d_err_norm = err_norm - self._prev_error_norm
        self._prev_error_norm = err_norm

        # steering_cmd positive → steer left (matches sign convention above)
        raw_cmd = _Kp * err_norm + _Kd * d_err_norm
        cmd     = max(-_MAX_CMD, min(_MAX_CMD, raw_cmd))

        # ── Confidence ────────────────────────────────────────────────────
        if geometry.has_left and geometry.has_right:
            geom_conf = 1.0
        elif geometry.has_left or geometry.has_right:
            geom_conf = 0.5
        else:
            geom_conf = 0.0

        confidence = min(detection_confidence, geom_conf)

        # ── Recommendation ─────────────────────────────────────────────────
        if confidence < _LOW_CONF_IOU:
            rec = SteeringRecommendation.LOW_CONFIDENCE
        elif abs(err_norm) < _CENTER_THRESHOLD_NORM:
            rec = SteeringRecommendation.KEEP_CENTER
        elif err_norm > 0:
            rec = SteeringRecommendation.STEER_LEFT
        else:
            rec = SteeringRecommendation.STEER_RIGHT

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return OffsetResult(
            vehicle_center_x   = vehicle_cx,
            lane_center_x      = lane_cx,
            lateral_error_px   = round(err_px,   2),
            lateral_error_norm = round(err_norm, 4),
            drift_direction    = drift,
            steering_command   = round(cmd, 4),
            recommendation     = rec,
            confidence         = round(confidence, 3),
            one_lane_estimated = geometry.one_lane_estimated,
            calc_ms            = elapsed_ms,
        )
