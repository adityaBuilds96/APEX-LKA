"""
tests/test_fleet_ota.py
=======================
Comprehensive integration test suite (16 tests) for Phase 5 Section 6:
- Cryptographic Over-The-Air (OTA) Model Manager & Ed25519 verification
- Atomic Hot-Swapping (<= 5ms) and Thread-Safe Predictor Ingress
- Silent Shadow-Mode Execution Engine & Divergence Benchmarking
- Closed-Loop Edge Failure Harvester & Rolling FIFO Quota Management
- CAN Frame 0x1A0 Fleet Remote Telemetry Broadcast & Loopback
- Streamlit / OpenCV HUD Fleet Status Overlay Rendering
- Full Autonomous Inference Pipeline Fleet Integration
"""

import hashlib
import json
import struct
import threading
import time
from pathlib import Path
import numpy as np
import pytest

# Fleet modules
from src.fleet.ota_updater import (
    OTAModelManager,
    OTAManifest,
    OTASyncState,
    ModelBackendType,
    MockPredictor,
    generate_keypair,
    sign_artifact,
    verify_signature,
)
from src.fleet.shadow_mode import (
    ShadowModeEngine,
    ShadowDivergenceReport,
    PromotionStatus,
    compute_mask_miou,
)
from src.fleet.data_collector import (
    ClosedLoopDataHarvester,
    HarvestTriggerReason,
    HarvestEvent,
)

# Vehicle Communicator & CAN ID 0x1A0
from deploy.vehicle_comm import (
    VehicleCommunicator,
    FleetStatusTelemetry,
    CAN_ID_FLEET_STATUS,
)

# Dashboard HUD
from dashboard.components.lka_hud import (
    _draw_fleet_ota_panel,
    _draw_ota_hotswap_banner,
    draw_hud_overlay,
)

# Inference Pipeline
from src.inference.pipeline import run_pipeline, get_ota_manager


@pytest.fixture
def fleet_suite_env(tmp_path):
    """Provides isolated testing environment with genuine Ed25519 keys and dirs."""
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir(parents=True, exist_ok=True)
    pub_key_path, priv_key_path = generate_keypair(keys_dir)

    staging_dir = tmp_path / "staging"
    active_dir = tmp_path / "active"
    quarantine_dir = tmp_path / "quarantine"
    harvest_dir = tmp_path / "harvested"
    shadow_csv = tmp_path / "shadow_log.csv"

    ota_mgr = OTAModelManager(
        public_key_path=pub_key_path,
        staging_dir=staging_dir,
        active_dir=active_dir,
        quarantine_dir=quarantine_dir,
    )

    shadow_pred = MockPredictor(name="candidate_v2", version="2.0.0")
    shadow_eng = ShadowModeEngine(
        shadow_predictor=shadow_pred,
        divergence_threshold_m=0.40,
        log_csv_path=shadow_csv,
        min_eval_frames=5,
    )

    harvester = ClosedLoopDataHarvester(
        output_dir=harvest_dir,
        storage_quota_mb=5,
        trigger_confidence_low=0.40,
        trigger_divergence_m=0.40,
        pre_event_frames=1,
        post_event_frames=1,
        consecutive_low_conf=2,
    )

    return {
        "tmp_path": tmp_path,
        "pub_key_path": pub_key_path,
        "priv_key_path": priv_key_path,
        "ota_mgr": ota_mgr,
        "shadow_eng": shadow_eng,
        "harvester": harvester,
    }


# ── TEST 1: CAN ID 0x1A0 Packing and Unpacking ──────────────────────────────
def test_can_0x1a0_serialization_and_deserialization():
    """Verify CAN frame 0x1A0 encodes and decodes (<HBBBBH, 8 bytes)."""
    raw_payload = struct.pack("<HBBBBH", 0x4A1F, 5, 18, 42, 2, 0)
    assert len(raw_payload) == 8

    v_hash, sync_u8, s_disc, h_count, b_enum, reserved = struct.unpack("<HBBBBH", raw_payload)
    assert v_hash == 0x4A1F
    assert sync_u8 == 5
    assert s_disc == 18
    assert h_count == 42
    assert b_enum == 2
    assert reserved == 0


# ── TEST 2: CAN ID 0x1A0 Telemetry Bounds and Clamping ──────────────────────
def test_can_0x1a0_telemetry_bounds_and_clamping():
    """Verify VehicleCommunicator clamps fields to valid byte ranges."""
    comm = VehicleCommunicator(can_channel="vcan_test", use_can=False, use_serial=False)

    comm.send_fleet_status(
        model_version_hash=0xFFFF + 50,  # Overflow uint16
        sync_state_enum=255,             # Out of enum range
        shadow_discrepancy_score=150,    # Above 100
        harvest_queue_count=300,         # Above uint8
        backend_enum=99,                 # Out of range
    )

    stat = comm.get_fleet_status()
    assert stat is not None
    assert stat.model_version_hash == 0xFFFF
    assert stat.sync_state_enum == 6  # Fallback to ERROR
    assert stat.shadow_discrepancy_score == 100
    assert stat.harvest_queue_count == 255
    assert stat.backend_enum == 0     # Fallback to CLASSICAL
    comm.close()


# ── TEST 3: VehicleCommunicator Loopback & Callback ────────────────────────
def test_vehicle_communicator_fleet_status_send_and_receive():
    """Verify fleet status broadcast and retrieval via VehicleCommunicator."""
    comm = VehicleCommunicator(can_channel="vcan_test", use_can=False, use_serial=False)
    received = []

    def on_fleet_status(t: FleetStatusTelemetry):
        received.append(t)

    comm.register_fleet_status_callback(on_fleet_status)

    ok = comm.send_fleet_status(
        model_version_hash=0x1234,
        sync_state_enum=5,
        shadow_discrepancy_score=25,
        harvest_queue_count=7,
        backend_enum=2,
    )
    assert ok is True
    assert len(received) == 1
    assert received[0].model_version_hash == 0x1234
    assert received[0].sync_state_str == "SYNC_OK"
    assert received[0].shadow_discrepancy_score == 25
    assert received[0].harvest_queue_count == 7
    assert received[0].backend_str == "ONNX"
    comm.close()


# ── TEST 4: Full OTA Lifecycle (Check, Download, Verify, Hot-Swap) ──────────
def test_fleet_ota_full_lifecycle(fleet_suite_env):
    """Verify full cryptographic OTA workflow from candidate creation to active inference."""
    ota_mgr = fleet_suite_env["ota_mgr"]
    priv_key = fleet_suite_env["priv_key_path"]

    # 1. Create candidate model artifact
    candidate_path = fleet_suite_env["tmp_path"] / "candidate_model.onnx"
    raw_weights = b"VERIFIED_ONNX_MODEL_BINARY_STAGED_ARTIFACT"
    candidate_path.write_bytes(raw_weights)

    sha256 = hashlib.sha256(raw_weights).hexdigest()
    sig = sign_artifact(raw_weights, priv_key)

    manifest = OTAManifest(
        model_version="3.0.0",
        model_type=ModelBackendType.ONNX,
        sha256_hash=sha256,
        signature_b64=sig,
        artifact_filename="candidate_model.onnx",
        file_size_bytes=len(raw_weights),
    )

    # 2. Download and verify
    verified_file = ota_mgr.download_and_verify(manifest, source_url_or_path=candidate_path)
    assert verified_file is not None
    assert ota_mgr.sync_state == OTASyncState.SYNC_OK

    # 3. Instantiate and Atomic Hot-Swap
    new_pred = MockPredictor(name="verified_v3", version="3.0.0")
    swap_ok = ota_mgr.atomic_hotswap(new_pred, target_version="3.0.0")
    assert swap_ok is True
    assert ota_mgr.active_version == "3.0.0"
    assert ota_mgr.active_predictor.version == "3.0.0"


# ── TEST 5: Atomic Hot-Swap Concurrency Thread Safety ───────────────────────
def test_atomic_hotswap_concurrency_thread_safety(fleet_suite_env):
    """Verify concurrent reads never observe corrupted or None pointer during hot-swap."""
    ota_mgr = fleet_suite_env["ota_mgr"]
    ota_mgr.active_predictor = MockPredictor(name="v1", version="1.0.0")

    stop_flag = False
    observed_versions = set()
    read_errors = []

    def reader_loop():
        while not stop_flag:
            pred = ota_mgr.active_predictor
            if pred is None:
                read_errors.append("Observed None predictor")
            else:
                observed_versions.add(pred.version)
            time.sleep(0.0001)

    threads = [threading.Thread(target=reader_loop) for _ in range(3)]
    for t in threads:
        t.start()

    # Perform rapid hot-swaps
    for v in ["2.0.0", "3.0.0", "4.0.0"]:
        time.sleep(0.01)
        p = MockPredictor(name=f"v_{v}", version=v)
        ota_mgr.atomic_hotswap(p, target_version=v)

    time.sleep(0.05)
    stop_flag = True
    for t in threads:
        t.join()

    assert len(read_errors) == 0
    assert "4.0.0" in observed_versions


# ── TEST 6: Quarantine Tampered Artifact Leaves Active Predictor Safe ────────
def test_quarantine_tampered_model_and_preserve_active(fleet_suite_env):
    """Verify tampered model is quarantined and active predictor remains unchanged."""
    ota_mgr = fleet_suite_env["ota_mgr"]
    safe_pred = MockPredictor(name="safe_v1", version="1.0.0")
    ota_mgr.active_predictor = safe_pred
    ota_mgr.active_version = "1.0.0"

    corrupted_file = fleet_suite_env["tmp_path"] / "tampered.onnx"
    corrupted_file.write_bytes(b"MALICIOUS_TAMPERED_INJECTED_CODE")

    manifest = OTAManifest(
        model_version="6.6.6",
        model_type=ModelBackendType.ONNX,
        sha256_hash="0000000000000000000000000000000000000000000000000000000000000000",
        signature_b64="SW52YWxpZA==",
        artifact_filename="tampered.onnx",
    )

    verified = ota_mgr.download_and_verify(manifest, source_url_or_path=corrupted_file)
    assert verified is None
    assert ota_mgr.sync_state == OTASyncState.ERROR
    # Active predictor is completely untouched
    assert ota_mgr.active_predictor.version == "1.0.0"
    assert ota_mgr.active_version == "1.0.0"


# ── TEST 7: Shadow Mode Parallel Background Execution ────────────────────────
def test_shadow_mode_async_execution_stream(fleet_suite_env):
    """Verify shadow mode runs asynchronously without delay to primary loop."""
    shadow_eng = fleet_suite_env["shadow_eng"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    rep = shadow_eng.step(
        frame=dummy_frame,
        primary_prediction=None,
        primary_offset_m=0.10,
        primary_confidence=0.88,
        frame_idx=1,
    )
    dispatch_ms = (time.perf_counter() - t0) * 1000.0

    assert dispatch_ms <= 1.0
    assert rep is not None


# ── TEST 8: Shadow Mode Discrepancy Score Normalization ──────────────────────
def test_shadow_mode_discrepancy_score_mapping(fleet_suite_env):
    """Verify discrepancy score is strictly within [0, 100]."""
    shadow_eng = fleet_suite_env["shadow_eng"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Moderate divergence
    rep = shadow_eng.evaluate_divergence_synchronous(
        dummy_frame,
        primary_offset_m=0.30,
        primary_confidence=0.80,
    )
    assert 0 <= rep.discrepancy_score_u8 <= 100


# ── TEST 9: Shadow Candidate Auto-Promotion Lifecycle ────────────────────────
def test_shadow_mode_auto_promotion_lifecycle(fleet_suite_env):
    """Verify candidate model is flagged for promotion upon superior performance."""
    shadow_eng = fleet_suite_env["shadow_eng"]
    dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Feed 6 frames with low primary confidence (< 0.40) and high shadow confidence
    for i in range(6):
        shadow_eng.evaluate_divergence_synchronous(
            dummy_frame,
            primary_offset_m=0.15,
            primary_confidence=0.30,  # Hard scene
            frame_idx=i,
        )

    assert shadow_eng.promotion_status == PromotionStatus.PROMOTION_RECOMMENDED


# ── TEST 10: Harvester All Trigger Conditions ────────────────────────────────
def test_harvester_all_triggers_matrix(fleet_suite_env):
    """Verify each of the 4 harvest triggers initiates capture."""
    harvester = fleet_suite_env["harvester"]
    dummy_frame = np.zeros((50, 50, 3), dtype=np.uint8)

    # 1. RSS intervention
    evt1 = harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, rss_override_active=True)
    assert evt1 is not None and evt1.trigger_reason == HarvestTriggerReason.RSS_INTERVENTION
    harvester._active_capture = None  # Reset for next trigger

    # 2. Driver override
    evt2 = harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, driver_override_active=True)
    assert evt2 is not None and evt2.trigger_reason == HarvestTriggerReason.DRIVER_OVERRIDE
    harvester._active_capture = None

    # 3. High divergence
    evt3 = harvester.observe(dummy_frame, confidence=0.8, offset_m=0.0, shadow_divergence_m=0.60)
    assert evt3 is not None and evt3.trigger_reason == HarvestTriggerReason.SHADOW_DIVERGENCE
    harvester._active_capture = None

    # 4. Low confidence (2 consecutive)
    harvester.observe(dummy_frame, confidence=0.30, offset_m=0.0)
    evt4 = harvester.observe(dummy_frame, confidence=0.30, offset_m=0.0)
    assert evt4 is not None and evt4.trigger_reason == HarvestTriggerReason.LOW_CONFIDENCE


# ── TEST 11: Harvester FIFO Rolling Storage Quota ────────────────────────────
def test_harvester_fifo_storage_quota_preservation(fleet_suite_env):
    """Verify FIFO rolling deletion purges oldest packages when quota exceeded."""
    harvester = fleet_suite_env["harvester"]
    out_dir = harvester.output_dir
    harvester.storage_quota_bytes = 100 * 1024  # 100 KB limit

    # Create 3 packages with 50 KB each -> 150 KB total > 100 KB
    for i in range(3):
        d = out_dir / f"event_20261011_00000{i}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "data.bin").write_bytes(b"A" * (50 * 1024))
        time.sleep(0.02)

    harvester._enforce_storage_quota()
    remaining = [d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("event_")]
    assert len(remaining) <= 2
    # Oldest package should be gone
    assert not (out_dir / "event_20261011_000000").exists()


# ── TEST 12: HUD Fleet OTA Panel Overlay Rendering ──────────────────────────
def test_hud_fleet_panel_rendering():
    """Verify fleet status panel renders onto canvas without errors."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    fleet_status = {
        "model_version_hash": 0x3F1A,
        "sync_state_str": "SYNC_OK",
        "shadow_discrepancy": 15,
        "harvest_queue_count": 3,
        "backend_str": "ONNX",
    }
    _draw_fleet_ota_panel(canvas, fleet_status, x=14, y=280, bw=220)
    # Check that pixels were drawn
    assert np.any(canvas[270:350, 10:240] > 0)


# ── TEST 13: HUD Hot-Swap Banner Rendering ──────────────────────────────────
def test_hud_hotswap_complete_banner_rendering():
    """Verify hot-swap banner renders centered text onto canvas."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)
    _draw_ota_hotswap_banner(canvas, w=640, h=360)
    assert np.any(canvas > 0)


# ── TEST 14: End-to-End Pipeline Inference with Shadow & Harvesting ──────────
def test_pipeline_inference_result_fleet_integration():
    """Verify run_pipeline executes shadow mode and data harvesting cleanly."""
    dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
    res = run_pipeline(
        dummy_img,
        enable_shadow_mode=True,
        enable_data_harvesting=True,
    )
    assert res is not None
    assert res.shadow_report is not None
    assert res.fleet_status is not None
    assert "model_version_hash" in res.fleet_status
    assert "sync_state_str" in res.fleet_status


# ── TEST 15: Mock Predictor Fallback and Inference ───────────────────────────
def test_mock_predictor_inference():
    """Verify MockPredictor behaves as expected for testing."""
    pred = MockPredictor(name="test_pred", version="1.5.0")
    dummy = np.zeros((100, 100, 3), dtype=np.uint8)
    res = pred.infer(dummy)
    assert res.left_confidence >= 0.8
    assert res.right_confidence >= 0.8
    assert res.model_version == "1.5.0"


# ── TEST 16: Rollback on Faulty Predictor Staging Failure ────────────────────
def test_ota_manager_rollback_on_faulty_predictor(fleet_suite_env):
    """Verify rollback restores previous predictor if new predictor is rolled back."""
    ota_mgr = fleet_suite_env["ota_mgr"]
    p1 = MockPredictor(name="v1", version="1.0.0")
    p2 = MockPredictor(name="v2_broken", version="2.0.0")

    ota_mgr.active_predictor = p1
    ota_mgr.active_version = "1.0.0"

    # Swap to p2
    ota_mgr.atomic_hotswap(p2, target_version="2.0.0")
    assert ota_mgr.active_predictor.version == "2.0.0"

    # Trigger rollback
    ok = ota_mgr.rollback()
    assert ok is True
    assert ota_mgr.active_predictor.version == "1.0.0"
    assert ota_mgr.active_version == "1.0.0"
