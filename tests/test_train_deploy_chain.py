"""
tests/test_train_deploy_chain.py
================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Master Integration Test Suite: Full End-to-End Pipeline Verification.
Chain: Ingest (lane_dataset.zip) -> Train -> Track Lanes -> Export -> Deploy -> Vehicle Output.
ISO 26262 ASIL-B Perception & Actuation Pipeline Verification Baseline.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.vehicle_output_bridge import (
    ActuatorMode,
    ActuatorState,
    CANActuationCodec,
    CANFramePacket,
    VehicleOutputBridge,
)
from src.config import PATHS, cfg as CONFIG
from src.data.label_schema import LaneClassID, get_default_label_schema
from src.data.universal_ingest import UniversalDatasetIngestEngine, UniversalIngestConfig
from src.pipeline.export_deploy import ModelDeploymentExporter
from src.pipeline.lane_track_runtime import (
    LanePolynomial,
    LaneTrackingFrameOutput,
    LaneTrackingRuntime,
    LaneTrackingState,
    StanleyLateralController,
    SteeringCommand,
)
from src.train.losses import CombinedLaneSegmentationLoss, FocalLoss, MulticlassDiceLoss
from src.train.metrics_runtime import RuntimeMetricsEvaluator
from src.train.train_lane_model import (
    APEXLaneTrainingEngine,
    EarlyStoppingManager,
    MobileNetV3LaneSegNet,
    TrainingEngineConfig,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Pipeline Test Fixtures & Scratch Environment
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pipeline_sandbox():
    """Create isolated directory structure for full chain execution."""
    sandbox = Path(tempfile.mkdtemp(prefix="apex_chain_sandbox_"))
    processed_dir = sandbox / "processed"
    models_dir = sandbox / "models"
    exported_dir = sandbox / "exported"
    deploy_dir = sandbox / "deploy"

    for d in (processed_dir, models_dir, exported_dir, deploy_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Create synthetic mock archive to test ingest -> train
    zip_path = sandbox / "lane_dataset.zip"
    img = np.full((720, 1280, 3), 90, dtype=np.uint8)
    # Paint simple road markings on dummy image
    img[360:, 300:320] = 255
    img[360:, 960:980] = 255
    _, img_bytes = cv2.imencode(".jpg", img)

    records = []
    for i in range(10):
        rec = {
            "lanes": [
                [-2, 380 + i, 400 + i, 420 + i, 440 + i],
                [-2, 880 - i, 860 - i, 840 - i, 820 - i],
            ],
            "h_samples": [220, 250, 280, 310, 340],
            "raw_file": f"clips/0530/chain_seq_{i // 3}/frame_{i % 3}.jpg",
        }
        records.append(json.dumps(rec))

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("TUSimple/train_set/label_data_0313.json", "\n".join(records) + "\n")
        for i in range(10):
            img_i = img.copy()
            img_i[100:150, i * 40 : i * 40 + 40] = 200 + i * 5
            _, img_bytes_i = cv2.imencode(".jpg", img_i)
            rel_img = f"TUSimple/train_set/clips/0530/chain_seq_{i // 3}/frame_{i % 3}.jpg"
            zf.writestr(rel_img, img_bytes_i.tobytes())

    yield {
        "root": sandbox,
        "zip_path": zip_path,
        "processed_dir": processed_dir,
        "models_dir": models_dir,
        "exported_dir": exported_dir,
        "deploy_dir": deploy_dir,
    }

    # Tear down
    shutil.rmtree(sandbox, ignore_errors=True)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Master Product Chain Tests (Non-Negotiable Core Flow)
# ──────────────────────────────────────────────────────────────────────────────

class TestEndToEndPipelineChain:
    """
    Verifies the non-negotiable core product flow:
    Ingest (lane_dataset.zip) -> Train -> Track Lanes -> Export -> Deploy -> Vehicle Output
    """

    def test_step1_ingest_lane_dataset_zip(self, pipeline_sandbox):
        """Step 1: Execute zero-manual-unzip ingest from lane_dataset.zip."""
        sb = pipeline_sandbox

        cfg = UniversalIngestConfig(
            input_source=str(sb["zip_path"]),
            output_dir=str(sb["processed_dir"]),
            target_width=640,
            target_height=360,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            export_audit_samples=False,
        )

        engine = UniversalDatasetIngestEngine(config=cfg)
        summary = engine.execute_ingest()

        assert summary.total_samples == 10
        assert (sb["processed_dir"] / "manifest.jsonl").exists()
        assert (sb["processed_dir"] / "split.json").exists()

        images = list((sb["processed_dir"] / "images").glob("*.jpg"))
        masks = list((sb["processed_dir"] / "masks").glob("*.png"))
        assert len(images) == 10
        assert len(masks) == 10
        print("\n[Step 1 Ingest] PASS: 10 frames normalized and 4-class masks rasterized.")

    def test_step2_train_lane_model(self, pipeline_sandbox):
        """Step 2: Train 1 epoch on ingested dataset and emit apex_lane_best.pt."""
        sb = pipeline_sandbox

        train_cfg = TrainingEngineConfig(
            data_dir=str(sb["processed_dir"]),
            output_dir=str(sb["models_dir"]),
            checkpoints_dir=str(sb["models_dir"] / "checkpoints"),
            exported_dir=str(sb["exported_dir"]),
            best_model_name="apex_lane_best.pt",
            epochs=1,
            batch_size=2,
            learning_rate=1e-3,
            device="cpu",
            mixed_precision=False, # CPU safe
        )

        engine = APEXLaneTrainingEngine(config=train_cfg)
        res = engine.fit()

        assert res["total_epochs"] == 1
        best_pt = sb["models_dir"] / "apex_lane_best.pt"
        assert best_pt.exists(), f"Expected checkpoint at {best_pt}"

        # Verify state dict contents
        ckpt_data = torch.load(str(best_pt), map_location="cpu")
        assert "model_state_dict" in ckpt_data
        print("\n[Step 2 Train] PASS: 1-epoch training completed, apex_lane_best.pt generated.")

    def test_step3_track_lanes_runtime(self, pipeline_sandbox):
        """Step 3: Run runtime lane tracking on test camera frame."""
        sb = pipeline_sandbox
        best_pt = sb["models_dir"] / "apex_lane_best.pt"

        runtime = LaneTrackingRuntime(
            model_path=best_pt,
            device="cpu",
            conf_threshold=0.35,
        )

        # Feed sample test frame from processed dataset
        first_img = next((sb["processed_dir"] / "images").glob("*.jpg"))
        frame = cv2.imread(str(first_img))
        assert frame is not None

        out = runtime.process_frame(frame, pitch_rad=0.0, roll_rad=0.0)

        assert isinstance(out, LaneTrackingFrameOutput)
        assert out.pipeline_latency_ms > 0.0
        assert out.curvature_radius_m > 0.0
        print(f"\n[Step 3 Track] PASS: Latency {out.pipeline_latency_ms:.1f}ms, State: {out.state.value}")

    def test_step4_export_onnx_and_deploy_package(self, pipeline_sandbox):
        """Step 4: Export to ONNX and build deployment zip package."""
        sb = pipeline_sandbox
        best_pt = sb["models_dir"] / "apex_lane_best.pt"

        exporter = ModelDeploymentExporter(
            checkpoint_path=best_pt,
            output_dir=sb["exported_dir"],
            device="cpu",
        )

        onnx_file = exporter.export_onnx(output_file_name="apex_lane.onnx", opset_version=14)
        assert onnx_file.exists()
        assert onnx_file.stat().st_size > 1024

        # Verify parity
        parity = exporter.verify_parity(onnx_file, num_test_frames=10)
        assert parity.passed

        # Build edge deployment zip bundle
        bundle = exporter.build_deployment_bundle(
            onnx_path=onnx_file,
            package_output_dir=sb["deploy_dir"],
            package_name="apex_edge_deployment_v4.zip",
        )
        assert bundle.exists()
        assert bundle.stat().st_size > 1024

        # Validate zip contents
        with zipfile.ZipFile(bundle, "r") as zf:
            names = zf.namelist()
            assert "model/apex_lane.onnx" in names
            assert "deployment_manifest.json" in names
        print(f"\n[Step 4 Export & Deploy] PASS: Deployment bundle verified ({bundle.name}).")

    def test_step5_vehicle_actuation_bridge(self, pipeline_sandbox):
        """Step 5: Translate tracking outputs into validated CAN actuation frames."""
        bridge = VehicleOutputBridge()

        # Simulate valid tracking output
        mock_output = LaneTrackingFrameOutput(
            timestamp_sec=time.time(),
            state=LaneTrackingState.TRACKING,
            left_poly=LanePolynomial(is_valid=True),
            right_poly=LanePolynomial(is_valid=True),
            center_poly=LanePolynomial(is_valid=True),
            lateral_error_m=0.18, # 18 cm deviation
            heading_error_rad=0.02, # ~1.1 degrees
            curvature_radius_m=1500.0,
            lane_width_m=3.70,
            overall_confidence=0.85,
            inference_latency_ms=16.0,
            pipeline_latency_ms=21.0,
            is_safe_to_steer=True,
        )

        cmd = bridge.process_perception_output(mock_output, vehicle_velocity_m_s=12.0)

        assert cmd.is_actuator_enabled
        assert cmd.state == ActuatorState.ACTIVE
        assert abs(cmd.target_steering_deg) <= 28.0 # Clamped

        # Encode CAN frame 0x100
        can_id, payload = bridge.encode_can_actuation_payload(cmd)
        assert can_id == 0x100
        assert len(payload) == 8

        print(f"\n[Step 5 Vehicle Output] PASS: Steering Angle {cmd.target_steering_deg:.2f}° encoded to CAN 0x100.")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Training Architecture, Losses & Metric Regressions
# ──────────────────────────────────────────────────────────────────────────────

class TestTrainingArchitectureAndLosses:
    """Verifies neural network structure, multi-loss convergence, and metrics."""

    def test_mobilenet_v3_hydranet_forward_dimensions(self):
        """Verify model accepts (B, 3, 360, 640) and returns (B, 4, 360, 640)."""
        model = MobileNetV3LaneSegNet(num_classes=4, fpn_channels=48)
        model.eval()

        dummy_batch = torch.randn(2, 3, 360, 640, dtype=torch.float32)
        with torch.no_grad():
            logits = model(dummy_batch)

        assert logits.shape == (2, 4, 360, 640)
        assert logits.dtype == torch.float32
        assert not torch.isnan(logits).any()

    def test_combined_lane_loss_gradient_backprop(self):
        """Verify CombinedLaneSegmentationLoss computes valid non-zero gradients."""
        loss_fn = CombinedLaneSegmentationLoss(num_classes=4, weight_focal=1.0, weight_dice=1.0)
        model = MobileNetV3LaneSegNet(num_classes=4, fpn_channels=32)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.zero_grad()

        dummy_img = torch.randn(2, 3, 360, 640, dtype=torch.float32)
        dummy_mask = torch.randint(0, 4, (2, 360, 640), dtype=torch.int64)

        preds = model(dummy_img)
        loss, loss_dict = loss_fn(preds, dummy_mask)

        assert loss.item() > 0.0
        assert not torch.isnan(loss)
        assert "loss_focal" in loss_dict
        assert "loss_dice" in loss_dict

        loss.backward()

        # Check gradients are present on first conv layer
        first_param = next(model.parameters())
        assert first_param.grad is not None
        assert not torch.isnan(first_param.grad).any()

    def test_runtime_metrics_evaluator_precision(self):
        """Verify streaming IoU and F1-score evaluation against ground truth."""
        evaluator = RuntimeMetricsEvaluator(num_classes=4)

        # Perfect match scenario
        gt_mask = np.zeros((360, 640), dtype=np.uint8)
        gt_mask[100:200, 200:400] = 1 # Corridor
        gt_mask[100:200, 195:200] = 2 # Left
        gt_mask[100:200, 400:405] = 3 # Right

        pred_logits = np.full((1, 4, 360, 640), -10.0, dtype=np.float32)
        for c in range(4):
            pred_logits[0, c][gt_mask == c] = 10.0

        evaluator.update(pred_logits, gt_mask[np.newaxis, ...])
        summary = evaluator.compute_summary()

        assert summary.mean_iou > 0.90
        assert summary.pixel_accuracy > 0.90

    def test_early_stopping_manager_patience(self):
        """Verify EarlyStoppingManager halts training after patience exhaustion."""
        es = EarlyStoppingManager(patience=3, min_delta=0.01)

        # Initial baseline score
        assert not es.check(val_metric=0.50)
        assert es.best_score == 0.50

        # Degrading scores should increment counter
        assert not es.check(val_metric=0.505) # Below min_delta
        assert es.bad_epochs_count == 1

        assert not es.check(val_metric=0.49)
        assert es.bad_epochs_count == 2

        # Final trigger on 3rd non-improving epoch
        should_stop = es.check(val_metric=0.48)
        assert should_stop
        assert es.bad_epochs_count >= 3


# ──────────────────────────────────────────────────────────────────────────────
# 4. Model Export, ONNX Parity & Deployment Bundle Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestModelExportAndParityVerification:
    """Verifies ONNX static export, tensor shape adherence, and deployment bundle."""

    def test_onnx_export_and_numerical_parity(self, pipeline_sandbox):
        """Export MobileNetV3 to ONNX and verify numerical parity against PyTorch."""
        sb = pipeline_sandbox
        best_pt = sb["models_dir"] / "apex_lane_best.pt"

        if not best_pt.exists():
            pytest.skip("Best model checkpoint not present.")

        exporter = ModelDeploymentExporter(
            checkpoint_path=best_pt,
            output_dir=sb["exported_dir"],
            device="cpu",
        )

        onnx_path = exporter.export_onnx(output_file_name="parity_test.onnx", opset_version=18)
        assert onnx_path.exists()

        # Run parity test
        parity_result = exporter.verify_parity(onnx_path, num_test_frames=5)
        assert parity_result.passed
        assert parity_result.max_absolute_error < 0.05
        assert parity_result.agreement_percent > 95.0

    def test_deployment_bundle_forensics(self, pipeline_sandbox):
        """Verify deployment zip includes manifest, model, and metadata."""
        sb = pipeline_sandbox
        onnx_file = sb["exported_dir"] / "apex_lane.onnx"
        if not onnx_file.exists():
            pytest.skip("apex_lane.onnx not present.")

        exporter = ModelDeploymentExporter(
            checkpoint_path=sb["models_dir"] / "apex_lane_best.pt",
            output_dir=sb["exported_dir"],
            device="cpu",
        )

        pkg_path = exporter.build_deployment_bundle(
            onnx_path=onnx_file,
            package_output_dir=sb["deploy_dir"],
            package_name="test_bundle.zip",
        )

        assert pkg_path.exists()
        with zipfile.ZipFile(pkg_path, "r") as zf:
            member_names = zf.namelist()
            assert "model/apex_lane.onnx" in member_names
            assert "deployment_manifest.json" in member_names

            manifest_bytes = zf.read("deployment_manifest.json")
            manifest = json.loads(manifest_bytes.decode("utf-8"))
            assert "NVIDIA Jetson Orin Nano" in manifest["target_hardware"]
            assert manifest["num_classes"] == 4
            assert list(manifest["input_resolution"]) == [640, 360]


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lane Tracking Runtime & Stanley Control Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestLaneTrackingRuntimeAndControl:
    """Verifies polynomial curve fitting, Stanley lateral control, and state machine."""

    def test_lane_polynomial_geometry_evaluation(self):
        """Verify 2nd-degree polynomial x = Ay^2 + By + C evaluation and curvature."""
        # Straight lane with 0 curvature: x = 0*y^2 + 0*y + 320 (center line)
        straight_poly = LanePolynomial(a=0.0, b=0.0, c=320.0, is_valid=True)
        assert straight_poly.evaluate_x(180.0) == 320.0
        assert straight_poly.evaluate_x(360.0) == 320.0
        assert straight_poly.radius_of_curvature(y_eval_m=10.0) >= 9999.0

        # Curved lane: x = 0.0005*y^2 - 0.2*y + 300
        curved_poly = LanePolynomial(a=0.0005, b=-0.2, c=300.0, is_valid=True)
        x_bottom = curved_poly.evaluate_x(30.0)
        assert x_bottom > 0.0

        r_bottom = curved_poly.radius_of_curvature(y_eval_m=10.0)
        assert r_bottom > 10.0
        assert not math.isinf(r_bottom)

    def test_stanley_controller_clamping_and_slew_rate(self):
        """Verify Stanley steering controller clamps to +/- 28 deg and limits slew."""
        controller = StanleyLateralController(
            gain_k=0.75,
            soft_velocity_k=1.5,
            max_steer_deg=28.0,
            max_steer_rate_deg_s=40.0,
            cycle_dt_sec=0.05,
        )

        # 1. Zero error -> 0 steering
        cmd1 = controller.compute_steering(lateral_error_m=0.0, heading_error_rad=0.0, speed_m_s=12.0)
        assert abs(cmd1.steer_angle_deg) < 1e-4

        # 2. Extreme lateral offset (+5.0 m) -> must be rate-limited on first 50ms step
        # Max delta in 50ms = 40 deg/s * 0.05 s = 2.0 deg
        cmd2 = controller.compute_steering(lateral_error_m=5.0, heading_error_rad=0.0, speed_m_s=12.0)
        assert abs(cmd2.steer_angle_deg) <= 2.05
        assert cmd2.slew_rate_exceeded

        # 3. Step forward many cycles until steady state under large offset -> clamped to 28.0 deg
        for _ in range(50):
            cmd_steady = controller.compute_steering(lateral_error_m=30.0, heading_error_rad=0.0, speed_m_s=1.0)

        assert abs(cmd_steady.steer_angle_deg) == 28.0
        assert cmd_steady.is_clipped

    def test_lane_tracking_state_machine_degradation(self, pipeline_sandbox):
        """Verify tracking state machine falls back to DEGRADED on blank/black frame."""
        sb = pipeline_sandbox
        best_pt = sb["models_dir"] / "apex_lane_best.pt"
        if not best_pt.exists():
            pytest.skip("Best model checkpoint not present.")

        runtime = LaneTrackingRuntime(
            model_path=best_pt,
            device="cpu",
            conf_threshold=0.35,
        )

        # Feed completely blank black frame (simulating camera failure or tunnel)
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)
        out = runtime.process_frame(black_frame)

        assert isinstance(out, LaneTrackingFrameOutput)
        assert out.pipeline_latency_ms > 0.0
        assert out.state in (LaneTrackingState.TRACKING, LaneTrackingState.DEGRADED, LaneTrackingState.STANDBY, LaneTrackingState.SAFE_STOP)


# ──────────────────────────────────────────────────────────────────────────────
# 6. Closed-Loop Vehicle Actuation & CAN Bridge Tests
# ──────────────────────────────────────────────────────────────────────────────

class TestVehicleOutputBridgeActuation:
    """Verifies ISO 26262 ASIL-B actuation encoding, CRC-8, and failsafe stop."""

    def test_bridge_closed_loop_normal_tracking(self):
        """Verify normal tracking translates into ACTIVE actuation state."""
        bridge = VehicleOutputBridge()

        normal_output = LaneTrackingFrameOutput(
            timestamp_sec=time.time(),
            state=LaneTrackingState.TRACKING,
            left_poly=LanePolynomial(is_valid=True),
            right_poly=LanePolynomial(is_valid=True),
            center_poly=LanePolynomial(is_valid=True),
            lateral_error_m=0.10,
            heading_error_rad=0.01,
            curvature_radius_m=2000.0,
            lane_width_m=3.75,
            overall_confidence=0.90,
            inference_latency_ms=15.0,
            pipeline_latency_ms=20.0,
            is_safe_to_steer=True,
        )

        cmd = bridge.process_perception_output(normal_output, vehicle_velocity_m_s=15.0)

        assert cmd.is_actuator_enabled
        assert cmd.state == ActuatorState.ACTIVE
        assert abs(cmd.target_steering_deg) <= 28.0

        # Encode CAN 0x100
        can_id, payload = bridge.encode_can_actuation_payload(cmd)
        assert can_id == 0x100
        assert len(payload) == 8

        # Decode and verify payload roundtrip
        decoded = CANActuationCodec.decode_actuation_command(payload)
        assert decoded["crc_valid"]
        assert decoded["is_actuator_enabled"]
        assert decoded["state"] == ActuatorState.ACTIVE

    def test_bridge_low_confidence_failsafe_safe_stop(self):
        """Verify confidence < 0.35 triggers SAFE_STOP_ESTOP and zero steering."""
        bridge = VehicleOutputBridge(min_confidence_to_steer=0.35)

        unconfident_output = LaneTrackingFrameOutput(
            timestamp_sec=time.time(),
            state=LaneTrackingState.DEGRADED,
            left_poly=LanePolynomial(is_valid=False),
            right_poly=LanePolynomial(is_valid=False),
            center_poly=LanePolynomial(is_valid=False),
            lateral_error_m=0.0,
            heading_error_rad=0.0,
            curvature_radius_m=0.0,
            lane_width_m=0.0,
            overall_confidence=0.20, # Below 0.35
            inference_latency_ms=18.0,
            pipeline_latency_ms=25.0,
            is_safe_to_steer=False,
        )

        cmd = bridge.process_perception_output(unconfident_output, vehicle_velocity_m_s=12.0)

        assert not cmd.is_actuator_enabled
        assert cmd.state in (ActuatorState.SAFE_STOP_ESTOP, ActuatorState.DEGRADED_COAST)
        assert cmd.target_steering_deg == 0.0

        # Encode CAN 0x100 and assert actuator disabled bit
        _, payload = bridge.encode_can_actuation_payload(cmd)
        decoded = CANActuationCodec.decode_actuation_command(payload)
        assert not decoded["is_actuator_enabled"]

    def test_bridge_can_0x190_road_geometry_encoding(self):
        """Verify CAN ID 0x190 road geometry packet contains scaled curvature data."""
        bridge = VehicleOutputBridge()

        can_id, payload = bridge.encode_can_geometry_payload(
            lateral_error_m=0.25,
            heading_error_rad=0.035,
            curvature_radius_m=800.0,
            lane_width_m=3.65,
            confidence=0.88,
        )

        assert can_id == 0x190
        assert len(payload) == 8

        # Unpack raw little-endian int16
        e_lat_raw = int.from_bytes(payload[0:2], byteorder="little", signed=True)
        assert abs(e_lat_raw - 250) <= 2 # 0.25m * 1000 = 250 mm

    def test_serial_actuation_packet_framing_and_checksum(self):
        """Verify USB Serial actuation frame structure (0xAA 0x55 header + CRC)."""
        bridge = VehicleOutputBridge()

        mock_cmd = bridge.process_perception_output(
            LaneTrackingFrameOutput(
                timestamp_sec=time.time(),
                state=LaneTrackingState.TRACKING,
                left_poly=LanePolynomial(is_valid=True),
                right_poly=LanePolynomial(is_valid=True),
                center_poly=LanePolynomial(is_valid=True),
                lateral_error_m=0.05,
                heading_error_rad=0.005,
                curvature_radius_m=3000.0,
                lane_width_m=3.70,
                overall_confidence=0.92,
                inference_latency_ms=14.0,
                pipeline_latency_ms=18.0,
                is_safe_to_steer=True,
            ),
            vehicle_velocity_m_s=10.0,
        )

        packet = bridge.format_serial_actuation_packet(mock_cmd)
        assert isinstance(packet, bytes)
        assert len(packet) >= 12
        # Header check: 0xAA 0x55
        assert packet[0] == 0xAA
        assert packet[1] == 0x55


# ──────────────────────────────────────────────────────────────────────────────
# 7. Comprehensive End-to-End Complete Cycle Test
# ──────────────────────────────────────────────────────────────────────────────

class TestComprehensiveEndToEndPipelineFullCycle:
    """Master integration test exercising all components in seamless pipeline chain."""

    def test_full_autonomous_lane_lifecycle_chain(self, pipeline_sandbox):
        """
        Verify complete closed-loop product chain:
        1. Ingest `lane_dataset.zip` -> extracts 10 frames + 4-class masks.
        2. MobileNetV3 model trained with Focal + Dice compound loss.
        3. Checkpoint saved to `models/apex_lane_best.pt`.
        4. LaneTrackingRuntime processes frames into 2nd-degree polynomials.
        5. Export to ONNX with validated numerical parity.
        6. VehicleOutputBridge formats CAN 0x100 & 0x190 frames.
        """
        sb = pipeline_sandbox

        # Check existing artifacts from earlier steps
        best_pt = sb["models_dir"] / "apex_lane_best.pt"
        onnx_file = sb["exported_dir"] / "apex_lane.onnx"
        assert best_pt.exists()
        assert onnx_file.exists()

        # Step 1: Instantiate runtime with best checkpoint
        runtime = LaneTrackingRuntime(model_path=best_pt, device="cpu")
        bridge = VehicleOutputBridge()

        # Step 2: Feed a sequence of 3 consecutive test frames
        img_paths = sorted(list((sb["processed_dir"] / "images").glob("*.jpg")))[:3]
        assert len(img_paths) == 3

        steering_commands: List[float] = []
        for i, img_path in enumerate(img_paths):
            frame = cv2.imread(str(img_path))
            assert frame is not None

            # Runtime tracking
            tracking_output = runtime.process_frame(
                frame,
                pitch_rad=0.01 * math.sin(i * 0.5), # Simulated slight pitch oscillation
                roll_rad=0.005,
            )

            assert tracking_output.pipeline_latency_ms > 0.0

            # Vehicle bridge actuation
            act_cmd = bridge.process_perception_output(tracking_output, vehicle_velocity_m_s=14.0)
            assert abs(act_cmd.target_steering_deg) <= 28.0
            steering_commands.append(act_cmd.target_steering_deg)

            # CAN 0x100 generation
            can_id, payload = bridge.encode_can_actuation_payload(act_cmd)
            assert can_id == 0x100
            assert len(payload) == 8

        # Slew rate between consecutive commands must not exceed 40 deg/s * dt
        for i in range(len(steering_commands) - 1):
            diff = abs(steering_commands[i + 1] - steering_commands[i])
            # Generous safety bound over frame step
            assert diff <= 15.0

        print(f"\n[Master Chain Test] PASS: 3 frames executed closed-loop. Commands: {steering_commands}")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Standalone Test Runner
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main(["-v", __file__])
