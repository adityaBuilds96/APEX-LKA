"""
tests/test_lka_state.py
=======================
Unit tests for the LKA State Engine (temporal stability, heading error, LDW, diagnostics).
"""

import math
import numpy as np
import pytest

from src.inference.pipeline import InferenceResult
from src.inference.predictor import DetectionStatus, LanePrediction, ModelStatus
from src.lane_geometry.lane_estimator import LaneGeometry
from src.lane_geometry.offset_calculator import (
    DriftDirection,
    OffsetResult,
    SteeringRecommendation,
)
from src.lka_engine.lka_state import (
    FailureCondition,
    LaneStabilityState,
    LDWState,
    LKAStateTracker,
)


def _make_mock_result(
    left_det: bool = True,
    right_det: bool = True,
    offset_norm: float = 0.0,
    left_slope: float = 0.0,
    right_slope: float = 0.0,
    conf: float = 0.85,
) -> InferenceResult:
    """Helper to synthesize InferenceResult with specific geometric parameters."""
    status = (
        DetectionStatus.LANE_DETECTED
        if (left_det and right_det)
        else (DetectionStatus.PARTIAL_LANE_DETECTED if (left_det or right_det) else DetectionStatus.LANE_NOT_DETECTED)
    )

    left_mask = np.ones((360, 640), dtype=np.uint8) if left_det else None
    right_mask = np.ones((360, 640), dtype=np.uint8) if right_det else None

    pred = LanePrediction(
        status=status,
        model_status=ModelStatus.READY,
        backend_name="test",
        left_mask=left_mask,
        right_mask=right_mask,
        left_confidence=conf if left_det else 0.0,
        right_confidence=conf if right_det else 0.0,
    )

    # Simple linear polynomials: x = slope * y + intercept
    left_poly = np.array([left_slope, 220.0]) if left_det else None
    right_poly = np.array([right_slope, 420.0]) if right_det else None

    geo = LaneGeometry(
        left_poly=left_poly,
        right_poly=right_poly,
        x_left_bottom=left_slope * 360.0 + 220.0 if left_det else None,
        x_right_bottom=right_slope * 360.0 + 420.0 if right_det else None,
        lane_center_x=320.0 - offset_norm * 320.0,
        model_h=360,
        model_w=640,
    )

    off = OffsetResult(
        vehicle_center_x=320.0,
        lane_center_x=geo.lane_center_x,
        lateral_error_px=offset_norm * 320.0,
        lateral_error_norm=offset_norm,
        drift_direction=DriftDirection.CENTERED if abs(offset_norm) < 0.05 else (DriftDirection.RIGHT if offset_norm > 0 else DriftDirection.LEFT),
        steering_command=-offset_norm,
        confidence=conf,
    )

    return InferenceResult(
        success=(left_det or right_det),
        prediction=pred,
        geometry=geo,
        offset=off,
        timings={"inference_ms": 5.0, "total_ms": 12.0},
    )


def test_heading_error_computation():
    tracker = LKAStateTracker()

    # Straight road (slope = 0) -> Heading error ~0°
    r_straight = _make_mock_result(left_slope=0.0, right_slope=0.0)
    telem = tracker.update(r_straight)
    assert telem.heading_error_deg is not None
    assert abs(telem.heading_error_deg) < 0.5

    # Slanted right (dx/dy > 0) -> Positive heading error
    r_right = _make_mock_result(left_slope=0.1, right_slope=0.1)
    telem = tracker.update(r_right)
    assert telem.heading_error_deg > 2.0


def test_temporal_stabilization_states():
    tracker = LKAStateTracker()

    # 1st & 2nd frames: UNCERTAIN (requires 3 frames to be STABLE)
    t1 = tracker.update(_make_mock_result(conf=0.9))
    assert t1.stability_state == LaneStabilityState.UNCERTAIN

    t2 = tracker.update(_make_mock_result(conf=0.9))
    assert t2.stability_state == LaneStabilityState.UNCERTAIN

    # 3rd frame: STABLE
    t3 = tracker.update(_make_mock_result(conf=0.9))
    assert t3.stability_state == LaneStabilityState.STABLE

    # 1 dropped frame: TEMPORARILY_LOST (grace period holds previous values)
    r_lost = _make_mock_result(left_det=False, right_det=False, conf=0.0)
    t_lost1 = tracker.update(r_lost)
    assert t_lost1.stability_state == LaneStabilityState.TEMPORARILY_LOST
    assert t_lost1.smoothed_offset_norm is not None  # Preserved

    # 2nd and 3rd dropped frames: Still TEMPORARILY_LOST
    tracker.update(r_lost)
    t_lost3 = tracker.update(r_lost)
    assert t_lost3.stability_state == LaneStabilityState.TEMPORARILY_LOST

    # 4th dropped frame: Exceeds grace period -> LANE_LOST
    t_lost4 = tracker.update(r_lost)
    assert t_lost4.stability_state == LaneStabilityState.LANE_LOST
    assert t_lost4.smoothed_offset_norm is None  # Dropped completely, no fake lane!


def test_ldw_hysteresis():
    tracker = LKAStateTracker()

    # Initialize to STABLE centered
    for _ in range(3):
        tracker.update(_make_mock_result(offset_norm=0.0))

    # Single frame drift into warning zone (> 0.25) -> should NOT trigger WARNING immediately
    t_spike = tracker.update(_make_mock_result(offset_norm=0.35))
    assert t_spike.ldw_state != LDWState.WARNING

    # After 3 consecutive warning frames -> transitions to WARNING
    tracker.update(_make_mock_result(offset_norm=0.35))
    t_warn = tracker.update(_make_mock_result(offset_norm=0.35))
    assert t_warn.ldw_state == LDWState.WARNING

    # Single normal frame should NOT instantly clear warning
    t_rec1 = tracker.update(_make_mock_result(offset_norm=0.0))
    assert t_rec1.ldw_state == LDWState.WARNING

    # 3 consecutive normal frames -> clears warning back to NORMAL
    tracker.update(_make_mock_result(offset_norm=0.0))
    t_rec3 = tracker.update(_make_mock_result(offset_norm=0.0))
    assert t_rec3.ldw_state == LDWState.NORMAL


def test_failure_conditions():
    tracker = LKAStateTracker()

    # One lane missing
    r_partial = _make_mock_result(left_det=True, right_det=False)
    t_part = tracker.update(r_partial)
    assert t_part.failure_condition == FailureCondition.ONE_BOUNDARY_MISSING

    # Both missing
    r_none = _make_mock_result(left_det=False, right_det=False)
    t_none = tracker.update(r_none)
    assert t_none.failure_condition == FailureCondition.BOTH_BOUNDARIES_MISSING

    # Poor confidence
    r_low_conf = _make_mock_result(conf=0.15)
    t_low = tracker.update(r_low_conf)
    assert t_low.failure_condition == FailureCondition.POOR_CONFIDENCE


def test_steering_recommendation_clamping():
    tracker = LKAStateTracker()
    for _ in range(3):
        tracker.update(_make_mock_result(offset_norm=0.0))

    # Extreme offset: check clamping to [-30°, +30°]
    t_extreme = tracker.update(_make_mock_result(offset_norm=0.9))
    assert -30.0 <= t_extreme.recommended_angle_deg <= 30.0
    assert t_extreme.steering_recommendation == SteeringRecommendation.STEER_LEFT
