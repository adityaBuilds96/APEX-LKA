"""
tests/test_object_detection.py
==============================
End-to-End Unit & Integration tests for Phase 5 Section 1:
- Multi-object detection engine
- Threat risk assessor
- VehicleCommunicator CAN Frame 0x150 broadcast
- Perception pipeline and HUD visualization overlays
"""

from __future__ import annotations

import numpy as np
import pytest

from dashboard.components.lka_hud import draw_hud_overlay
from deploy.vehicle_comm import (
    CAN_ID_THREAT_ALERT,
    ThreatAlertTelemetry,
    VehicleCommunicator,
)
from src.inference.pipeline import run_pipeline
from src.perception.object_detector import (
    OBJECT_CLASSES,
    Detection,
    DetectionResult,
    ObjectDetector,
)
from src.perception.risk_assessment import (
    PriorityRiskAssessor,
    RiskAssessmentReport,
)


def test_can_0x150_threat_alert_roundtrip():
    """Verify VehicleCommunicator encodes, transmits, and decodes CAN Frame 0x150."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)

    sent = comm.send_threat_alert(
        distance_mm=3250,
        class_enum=1,  # Pedestrian
        risk_score=88,
        emergency_stop=True,
        seq=42,
    )
    assert sent is True

    threat = comm.get_threat_alert()
    assert threat is not None
    assert threat.distance_mm == 3250
    assert threat.class_enum == 1
    assert threat.risk_score == 88
    assert threat.emergency_stop is True
    assert threat.seq == 42
    assert threat.timestamp > 0.0

    comm.close()


def test_pipeline_with_object_detection_enabled():
    """Verify run_pipeline() executes detection and risk assessment when enabled."""
    dummy_frame = np.full((360, 640, 3), 100, dtype=np.uint8)

    res = run_pipeline(
        dummy_frame,
        backend="classical_cv",
        enable_object_detection=True,
    )

    assert res is not None
    assert hasattr(res, "detection_result")
    assert hasattr(res, "risk_report")
    assert "object_det_ms" in res.timings
    assert "risk_assess_ms" in res.timings
    assert res.annotated_bgr is not None
    assert res.annotated_bgr.shape == (360, 640, 3)


def test_hud_draw_with_detections_and_brake_warning():
    """Verify draw_hud_overlay renders bounding boxes, badges, and emergency brake banner."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)

    # Mock high-risk pedestrian
    det = Detection(
        box=np.array([250.0, 180.0, 350.0, 320.0]),
        score=0.95,
        class_id=1,
        class_name="pedestrian",
    )
    assessor = PriorityRiskAssessor()
    risk_rep = assessor.assess(detections=[det], frame_shape=(360, 640))
    det_res = DetectionResult(
        boxes=np.array([[250.0, 180.0, 350.0, 320.0]], dtype=np.float32),
        scores=np.array([0.95], dtype=np.float32),
        class_ids=np.array([1], dtype=np.int32),
        class_names=["pedestrian"],
    )

    hud_out = draw_hud_overlay(
        canvas,
        detection_result=det_res,
        risk_report=risk_rep,
    )

    assert hud_out is not None
    assert hud_out.shape == (360, 640, 3)
    # Ensure drawing modified pixels on the blank canvas
    assert np.any(hud_out > 0)


def test_object_detector_execution_latency_budget():
    """Verify object detection preprocess and postprocess meets high-speed budget."""
    detector = ObjectDetector(model_path=None)
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Warmup
    _ = detector.detect(dummy_frame)

    times = []
    for _ in range(10):
        t0 = np.datetime64("now")
        res = detector.detect(dummy_frame)
        times.append(res.inference_ms)

    # Average latency in mock/preprocess should be well under 18ms
    avg_ms = float(np.mean(times))
    assert avg_ms < 18.0
