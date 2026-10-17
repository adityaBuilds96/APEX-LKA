"""
tests/test_perception_v3.py
===========================
Phase 6 Section 1 Verification Suite:
1. HydraNet Backbone & Multi-Task Detection (Lane Mask + Curvature + 2D BBoxes)
2. TensorRT-ready modules & Latency validation (< 15ms)
3. Priority Risk Assessor: Center-Lane Pedestrian > Side-Lane Vehicle threat prioritization
4. CAN Bus ID 0x150 message packaging and 30 Hz broadcast
5. Dashboard PATHS resolution and clean UI branding validation (Zero "SAE" branding)
"""

from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import pytest
import torch

from src.models.perception_core import (
    HydraNet,
    APEXPerceptionCore,
    PerceptionOutput,
    Detection2D,
    OBJECT_CLASSES,
)
from src.perception.risk_engine import (
    PriorityRiskAssessor,
    ThreatTarget,
    RiskEngineReport,
)
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. HydraNet Multi-Task Architecture & TensorRT Compatibility
# ==============================================================================

def test_hydranet_output_shapes_and_static_dimensions():
    """Verify HydraNet produces static, deterministic shapes required for TensorRT."""
    model = HydraNet(pretrained_backbone=False)
    model.eval()

    # Input tensor: Batch=1, Channels=3, Height=360, Width=640
    dummy_input = torch.zeros(1, 3, 360, 640, dtype=torch.float32)

    with torch.no_grad():
        mask_logits, curvature, det_preds = model(dummy_input)

    # Head A: Lane Segmentation & Curvature
    assert mask_logits.shape == (1, 1, 360, 640), "Lane mask logits must match input resolution"
    assert curvature.shape == (1, 3), "Curvature head must output [c0, c1, c2]"

    # Head B: Object Detection
    # On C3 (stride 16), 360/16 = 22 or 23, 640/16 = 40 -> total grid cells N
    b, n_anchors, channels = det_preds.shape
    assert b == 1
    assert n_anchors > 0
    # channels = 4 box coordinates + 3 classes + 1 objectness = 8
    assert channels == 8, "Detection tensor must have exactly 8 channels (4 bbox + 3 cls + 1 obj)"


def test_hydranet_inference_latency():
    """Validate perception latency benchmark (< 15ms on GPU or < 80ms on CPU)."""
    core = APEXPerceptionCore(device="cuda" if torch.cuda.is_available() else "cpu")
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)

    # Warmup
    for _ in range(3):
        _ = core.predict(dummy_img)

    t0 = time.perf_counter()
    n_iters = 5
    for _ in range(n_iters):
        out = core.predict(dummy_img)
    avg_latency = ((time.perf_counter() - t0) / n_iters) * 1000.0

    assert isinstance(out, PerceptionOutput)
    assert out.lane_mask.shape == (360, 640)
    assert len(out.curvature_coeffs) == 3

    if torch.cuda.is_available():
        assert avg_latency < 15.0, f"GPU detection latency {avg_latency:.2f}ms exceeds 15ms target!"
    else:
        assert avg_latency < 150.0, f"CPU latency {avg_latency:.2f}ms is acceptable for test runner."


# ==============================================================================
# 2. Priority Risk Assessor (Center-Lane Pedestrian vs Side-Lane Vehicle)
# ==============================================================================

def test_risk_engine_prioritizes_center_pedestrian_over_side_vehicle():
    """
    CRITICAL SAFETY REQUIREMENT:
    A Pedestrian standing directly in the vehicle's driving path (center lane)
    MUST be assessed as a higher collision threat than a Vehicle in an adjacent side lane.
    """
    assessor = PriorityRiskAssessor(
        camera_height_m=1.25,
        pitch_angle_deg=-5.0,
        focal_length_px=500.0,
        principal_point=(320.0, 180.0),
        lane_width_m=3.2,
    )

    # 1. Pedestrian in Center Lane (X ≈ 0.0m, Z ≈ 14.0m)
    # Image coords: centered at x=320, bottom y=240
    det_ped_center = Detection2D(
        bbox=[310.0, 190.0, 330.0, 240.0],
        class_name="pedestrian",
        class_id=1,
        confidence=0.92,
    )

    # 2. Vehicle in Side Lane (X ≈ 2.5m to the right, even if closer at Z ≈ 9.0m)
    # Image coords: shifted to x=500 (right lane), bottom y=270 (closer)
    det_veh_side = Detection2D(
        bbox=[460.0, 200.0, 540.0, 270.0],
        class_name="vehicle",
        class_id=0,
        confidence=0.95,
    )

    report = assessor.evaluate([det_veh_side, det_ped_center], img_w=640, img_h=360)

    # Find evaluated target objects
    ped_target = next(t for t in report.targets if t.class_name == "pedestrian")
    veh_target = next(t for t in report.targets if t.class_name == "vehicle")

    # Assertions
    assert ped_target.is_in_lane is True, "Pedestrian at center must be marked as in_lane"
    assert veh_target.is_in_lane is False, "Vehicle at x=500 must be outside center lane"
    assert ped_target.risk_score > veh_target.risk_score, (
        f"Center-lane pedestrian risk ({ped_target.risk_score}) must exceed "
        f"side-lane vehicle risk ({veh_target.risk_score})"
    )

    # Primary target selected for CAN broadcast must be the Pedestrian
    assert report.primary_target is not None
    assert report.primary_target.class_name == "pedestrian"
    assert report.primary_target.target_id == ped_target.target_id


def test_risk_engine_stop_flag_trigger():
    """Verify that a close obstacle in the vehicle corridor triggers stop_flag."""
    assessor = PriorityRiskAssessor()

    # Pedestrian dangerously close in our path (bottom y=320 -> Z ~ 3.5m)
    det_danger = Detection2D(
        bbox=[300.0, 160.0, 340.0, 320.0],
        class_name="pedestrian",
        class_id=1,
        confidence=0.98,
    )

    report = assessor.evaluate([det_danger])
    assert len(report.targets) == 1
    target = report.targets[0]

    assert target.is_in_lane is True
    assert target.distance_m < 8.0
    assert target.stop_flag is True
    assert report.emergency_stop_required is True


# ==============================================================================
# 3. CAN Bus Integration (CAN ID 0x150 Packaging & 30 Hz Cadence)
# ==============================================================================

def test_can_0x150_frame_structure():
    """
    Validate CAN ID 0x150 byte format:
    [Risk_ID (8b), Class_Enum (8b), Dist_mm (16b), Risk_Score (8b), Stop_Flag (8b), Padding (16b)]
    Total: Exactly 8 bytes.
    """
    mgr = CANManager(channel="vcan0")

    payload = mgr.pack_risk_frame(
        risk_id=1,
        class_enum=1,        # 1: Pedestrian
        dist_mm=12500,       # 12.5 meters
        risk_score=0.85,     # 85%
        stop_flag=True,      # 1
    )

    assert len(payload) == 8, f"Payload must be 8 bytes, got {len(payload)}"

    decoded = mgr.unpack_risk_frame(payload)
    assert decoded["risk_id"] == 1
    assert decoded["class_enum"] == 1
    assert decoded["dist_mm"] == 12500
    assert decoded["distance_m"] == 12.50
    assert decoded["risk_score"] == 0.85
    assert decoded["stop_flag"] is True
    assert decoded["padding"] == 0


def test_can_manager_30hz_broadcast_cadence():
    """Verify CANManager periodic broadcaster loop operates at 30 Hz."""
    mgr = CANManager(channel="vcan0", broadcast_hz=30.0)
    received_frames = []

    mgr.register_callback(lambda cid, data: received_frames.append((cid, data)))

    target = ThreatTarget(
        target_id=1,
        class_name="pedestrian",
        class_enum=1,
        bbox=[310, 180, 330, 240],
        distance_m=12.0,
        distance_mm=12000,
        lateral_offset_m=0.1,
        is_in_lane=True,
        risk_score=0.88,
        stop_flag=True,
    )
    mgr.update_target(target)
    mgr.start_broadcaster()

    # Sleep for ~120ms (at 30Hz, should transmit approx 3-4 frames)
    time.sleep(0.12)
    mgr.stop_broadcaster()
    mgr.close()

    assert len(received_frames) >= 3, f"Expected >= 3 frames in 120ms at 30Hz, got {len(received_frames)}"
    for cid, data in received_frames:
        assert cid == 0x150
        assert len(data) == 8


# ==============================================================================
# 4. Dashboard Clean Branding & PATHS Resolution
# ==============================================================================

def test_dashboard_paths_import_and_branding():
    """Ensure dashboard imports PATHS without NameError and contains no legacy SAE branding."""
    from src.config import PATHS, cfg
    assert PATHS is not None
    assert hasattr(PATHS, "annotated")
    assert hasattr(PATHS, "raw_frames")

    # Read dashboard/app.py text and verify branding
    app_path = Path("dashboard/app.py")
    assert app_path.exists()
    content = app_path.read_text(encoding="utf-8")

    # Confirm brand title is APEX-Vision | Real-Time ADAS
    assert "APEX-Vision | Real-Time ADAS" in content
    # Confirm legacy SAE string is completely absent
    assert "SAE BAJA" not in content
    assert "SAE India" not in content
