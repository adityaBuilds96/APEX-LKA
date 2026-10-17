"""
tests/test_risk_assessment.py
=============================
Unit tests for PriorityThreatRiskAssessor, risk formula, and emergency brake trigger.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perception.object_detector import Detection, DetectionResult
from src.perception.risk_assessment import (
    CLASS_SEVERITY_MAP,
    PriorityRiskAssessor,
    RiskAssessmentReport,
)


def test_class_severity_specification():
    """Verify class severity mappings conform to autonomous safety standards."""
    assert CLASS_SEVERITY_MAP["pedestrian"] == 1.0
    assert CLASS_SEVERITY_MAP["animal"] == 1.0
    assert CLASS_SEVERITY_MAP["vehicle"] == 0.8
    assert CLASS_SEVERITY_MAP["barrier"] == 0.8
    assert CLASS_SEVERITY_MAP["traffic_cone"] == 0.5
    assert CLASS_SEVERITY_MAP["debris"] == 0.5
    assert CLASS_SEVERITY_MAP["traffic_sign"] == 0.1
    assert CLASS_SEVERITY_MAP["traffic_light"] == 0.1


def test_critical_pedestrian_emergency_brake():
    """Verify pedestrian in center corridor near bottom triggers emergency brake."""
    assessor = PriorityRiskAssessor(
        emergency_brake_risk_thresh=85.0,
        proximity_brake_thresh=0.75,
    )

    # Pedestrian directly in vehicle path at bottom (y2=324 in 360h image -> S_prox=0.90)
    det = Detection(
        box=np.array([280.0, 180.0, 360.0, 324.0]),
        score=1.0,
        class_id=1,
        class_name="pedestrian",
    )

    report = assessor.assess(
        detections=[det],
        drivable_corridor_poly=None,
        frame_shape=(360, 640),
    )

    assert report.highest_risk_object is not None
    # S_prox = 0.90, InCorridor = 1.0, Severity = 1.0
    # Risk = 100 * (0.5*0.9 + 0.3*1.0 + 0.2*1.0) = 95.0
    assert np.isclose(report.highest_risk_score, 95.0, atol=1.0)
    assert report.emergency_brake_flag is True
    assert report.is_critical is True
    assert report.highest_risk_distance_mm < 6000  # Within 6 meters


def test_informational_traffic_sign_low_risk():
    """Verify roadside traffic sign produces low risk and does not trigger brake."""
    assessor = PriorityRiskAssessor()

    # Traffic sign on far right near horizon (y2=140, x=580, not in corridor)
    det = Detection(
        box=np.array([550.0, 80.0, 610.0, 140.0]),
        score=0.90,
        class_id=7,
        class_name="traffic_sign",
    )

    report = assessor.assess(
        detections=[det],
        drivable_corridor_poly=None,
        frame_shape=(360, 640),
    )

    assert report.highest_risk_object is not None
    assert report.highest_risk_score < 30.0
    assert report.emergency_brake_flag is False
    assert report.has_threat is False


def test_empty_detections_safe_report():
    """Verify empty detections produce valid empty report."""
    assessor = PriorityRiskAssessor()
    report = assessor.assess(detections=DetectionResult())
    assert report.highest_risk_object is None
    assert report.highest_risk_score == 0.0
    assert report.emergency_brake_flag is False
    assert len(report.assessed_objects) == 0


def test_custom_drivable_corridor_polygon():
    """Verify custom drivable corridor polygon correctly filters lateral threats."""
    assessor = PriorityRiskAssessor(corridor_padding_px=10.0)

    # Defined corridor polygon (lane between x=200 and x=440 at bottom)
    corridor = np.array([
        [200.0, 360.0],
        [280.0, 150.0],
        [360.0, 150.0],
        [440.0, 360.0],
    ], dtype=np.float32)

    # Object 1: Cone inside corridor at (x=300, y=300)
    det_inside = Detection(
        box=np.array([280.0, 260.0, 320.0, 300.0]),
        score=0.90,
        class_id=3,
        class_name="traffic_cone",
    )

    # Object 2: Cone outside corridor on shoulder at (x=50, y=300)
    det_outside = Detection(
        box=np.array([30.0, 260.0, 70.0, 300.0]),
        score=0.90,
        class_id=3,
        class_name="traffic_cone",
    )

    report = assessor.assess(
        detections=[det_inside, det_outside],
        drivable_corridor_poly=corridor,
        frame_shape=(360, 640),
    )

    assert len(report.assessed_objects) == 2
    obj_in = report.assessed_objects[0]
    obj_out = report.assessed_objects[1]

    assert obj_in.in_corridor is True
    assert obj_out.in_corridor is False
    assert obj_in.risk_score > obj_out.risk_score
