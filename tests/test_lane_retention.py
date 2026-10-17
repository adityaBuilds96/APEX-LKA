"""
tests/test_lane_retention.py
============================
Unit tests for Kinematic Lane Retention & Dead Reckoning Path Hold.
Validates:
- Lane anchor caching on valid visual perception
- Kinematic lane retention during vision blackout (up to 5.0s)
- 2D rigid polynomial projection under ESKF displacement
- Confidence decay over time
- Max hold duration cutoff (> 5.0s expiration)
- Nominal vision re-acquisition and anchor reset
"""

import time
import pytest
import numpy as np

from src.fusion.lane_retention import (
    KinematicLaneRetainer,
    RetainedLaneResult,
)
from src.fusion.eskf import VehiclePose, LocalizationStatus
from src.inference.predictor import DetectionStatus, LanePrediction, ModelStatus
from src.lane_geometry.lane_estimator import LaneGeometry


def make_valid_prediction(confidence: float = 0.90) -> LanePrediction:
    return LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        model_status=ModelStatus.READY,
        left_confidence=confidence,
        right_confidence=confidence,
        accuracy_score=confidence,
        backend_name="classical_cv",
    )


def make_valid_geometry() -> LaneGeometry:
    # Parabolic lane lines: x = a*y^2 + b*y + c
    # Left line: x ~ 200 at bottom
    # Right line: x ~ 440 at bottom
    left_poly = np.array([0.0001, -0.05, 200.0])
    right_poly = np.array([0.0001, -0.05, 440.0])
    return LaneGeometry(
        left_poly=left_poly,
        right_poly=right_poly,
        x_left_bottom=200.0,
        x_right_bottom=440.0,
        lane_center_x=320.0,
    )


@pytest.fixture
def retainer():
    return KinematicLaneRetainer(
        max_hold_seconds=5.0,
        confidence_threshold=0.35,
        confidence_decay_per_sec=0.15,
        model_h=360,
        model_w=640,
    )


def test_retainer_nominal_visual_update(retainer):
    """Test nominal visual frame updates retention anchor."""
    pred = make_valid_prediction(confidence=0.88)
    geom = make_valid_geometry()
    pose0 = VehiclePose(x=0.0, y=0.0, yaw=0.0, vx=5.0)

    res = retainer.update(pred, geom, pose0, timestamp=100.0)

    assert not res.is_retained
    assert res.status_label == "VISUAL_ACTIVE"
    assert res.confidence == 0.88
    assert retainer.last_valid_geometry is not None
    assert retainer.last_anchor_pose is not None


def test_retention_activation_on_dropout(retainer):
    """Test retention takes over when vision detection drops to NO_DETECTION."""
    # Step 1: Valid visual anchor at t=100.0s
    pred_valid = make_valid_prediction(0.90)
    geom_valid = make_valid_geometry()
    pose0 = VehiclePose(x=0.0, y=0.0, yaw=0.0, vx=5.0)
    retainer.update(pred_valid, geom_valid, pose0, timestamp=100.0)

    # Step 2: Visual dropout at t=101.0s (1.0s later)
    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    geom_lost = LaneGeometry()
    # Vehicle has moved 5m forward
    pose1 = VehiclePose(x=0.0, y=5.0, yaw=0.0, vx=5.0)

    res_retained = retainer.update(pred_lost, geom_lost, pose1, timestamp=101.0)

    assert res_retained.is_retained
    assert res_retained.status_label == "LANE_RETENTION_ACTIVE"
    assert res_retained.geometry is not None
    assert res_retained.geometry.geometry_valid
    assert res_retained.prediction is not None
    assert res_retained.prediction.status == DetectionStatus.LANE_DETECTED
    assert res_retained.offset is not None
    # Confidence should have decayed by 0.15 * 1.0 = 0.15
    assert np.isclose(res_retained.confidence, 0.90 - 0.15, atol=0.02)


def test_confidence_decay_over_time(retainer):
    """Verify confidence decays monotonically during retention."""
    pred_valid = make_valid_prediction(0.85)
    geom_valid = make_valid_geometry()
    pose = VehiclePose(x=0.0, y=0.0, yaw=0.0)
    retainer.update(pred_valid, geom_valid, pose, timestamp=0.0)

    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    geom_lost = LaneGeometry()

    res_1s = retainer.update(pred_lost, geom_lost, pose, timestamp=1.0)
    res_2s = retainer.update(pred_lost, geom_lost, pose, timestamp=2.0)
    res_3s = retainer.update(pred_lost, geom_lost, pose, timestamp=3.0)

    assert res_1s.confidence > res_2s.confidence > res_3s.confidence
    assert np.isclose(res_3s.confidence, 0.85 - (0.15 * 3.0), atol=0.02)


def test_max_hold_duration_expiration(retainer):
    """Retention should expire after max_hold_seconds (5.0s)."""
    pred_valid = make_valid_prediction(0.95)
    geom_valid = make_valid_geometry()
    pose = VehiclePose(x=0.0, y=0.0, yaw=0.0)
    retainer.update(pred_valid, geom_valid, pose, timestamp=10.0)

    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    geom_lost = LaneGeometry()

    # At 4.5s: still retained
    res_4_5 = retainer.update(pred_lost, geom_lost, pose, timestamp=14.5)
    assert res_4_5.is_retained

    # At 5.1s: expired
    res_5_1 = retainer.update(pred_lost, geom_lost, pose, timestamp=15.1)
    assert not res_5_1.is_retained
    assert res_5_1.status_label == "RETENTION_EXPIRED"


def test_recovery_on_new_visual_frame(retainer):
    """Test re-acquisition of visual detection terminates retention immediately."""
    pred_valid = make_valid_prediction(0.90)
    geom_valid = make_valid_geometry()
    pose = VehiclePose(x=0.0, y=0.0, yaw=0.0)
    retainer.update(pred_valid, geom_valid, pose, timestamp=10.0)

    # Dropout
    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    geom_lost = LaneGeometry()
    res_drop = retainer.update(pred_lost, geom_lost, pose, timestamp=11.0)
    assert res_drop.is_retained

    # Recovery
    pred_recovered = make_valid_prediction(0.92)
    geom_recovered = make_valid_geometry()
    res_rec = retainer.update(pred_recovered, geom_recovered, pose, timestamp=12.0)
    assert not res_rec.is_retained
    assert res_rec.status_label == "VISUAL_ACTIVE"
    assert res_rec.confidence == 0.92


def test_retainer_reset(retainer):
    """Verify reset clears anchor state completely."""
    pred = make_valid_prediction(0.85)
    geom = make_valid_geometry()
    retainer.update(pred, geom, VehiclePose(), timestamp=0.0)
    retainer.reset()

    assert retainer.last_valid_geometry is None
    assert retainer.last_valid_prediction is None
    assert not retainer.is_retaining
