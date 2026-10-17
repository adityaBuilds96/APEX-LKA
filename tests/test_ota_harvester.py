"""
tests/test_ota_harvester.py
===========================
Phase 6 Section 6 Verification Suite:
1. Cryptographic OTA Signature verification failure properly rejects corrupt package.
2. Atomic hot-swap completes within 100ms without crashing perception.
3. Edge-case data harvester auto-trigger saves 5-second context slice to disk.
4. CAN ID 0x1A0 Fleet & OTA status serialization format.
"""

from __future__ import annotations

import hashlib
import json
import time
import zipfile
import numpy as np
import pytest

from src.ota.ota_manager import CryptographicOTAManager, OTAModelPackage, OTASyncState
from src.ota.shadow_mode import SilentShadowModeEngine
from src.ota.data_harvester import EdgeCaseDataHarvester, HarvesterTriggerReason
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. Signature Verification Failure Rejects Corrupted Package
# ==============================================================================

def test_ota_signature_verification_rejects_corrupted_payload(tmp_path):
    """
    Verify that CryptographicOTAManager rejects any model package where the
    binary payload has been modified or the signature is invalid.
    """
    ota_mgr = CryptographicOTAManager(models_root=tmp_path / "models")

    valid_binary = b"VALID_ENCRYPTED_TENSORRT_MODEL_WEIGHTS_V2.0.0"
    hasher = hashlib.sha256()
    hasher.update(valid_binary)
    valid_sha = hasher.hexdigest()

    # Sign legitimate content
    legit_signable = valid_sha.encode("utf-8") + b"v2.0.0"
    valid_sig = ota_mgr.sign_payload_for_testing(legit_signable)

    # 1. Verify legitimate package succeeds
    valid_pkg = OTAModelPackage(
        version="v2.0.0",
        model_name="model.plan",
        binary_data=valid_binary,
        sha256_hash=valid_sha,
        signature_bytes=valid_sig,
    )
    assert ota_mgr.verify_package(valid_pkg) is True

    # 2. Corrupt binary payload (simulating MITM tampering or bit-flip)
    tampered_pkg = OTAModelPackage(
        version="v2.0.0",
        model_name="model.plan",
        binary_data=b"TAMPERED_MALICIOUS_PAYLOAD",
        sha256_hash=valid_sha,
        signature_bytes=valid_sig,
    )
    assert ota_mgr.verify_package(tampered_pkg) is False
    assert ota_mgr.sync_state == OTASyncState.VERIFICATION_FAILED

    # 3. Forged signature on modified version
    forged_pkg = OTAModelPackage(
        version="v2.0.1",
        model_name="model.plan",
        binary_data=valid_binary,
        sha256_hash=valid_sha,
        signature_bytes=b"CORRUPT_INVALID_SIGNATURE_BYTES_1234567890",
    )
    assert ota_mgr.verify_package(forged_pkg) is False


# ==============================================================================
# 2. Atomic Hot-Swap Completes Within 100ms
# ==============================================================================

def test_atomic_hot_swap_under_100ms(tmp_path):
    """
    Validate that atomic hot-swap finishes well below the 100ms ceiling.
    """
    ota_mgr = CryptographicOTAManager(models_root=tmp_path / "models")

    new_model_data = b"TENSORRT_ENGINE_V2_3_0_FP16"
    hasher = hashlib.sha256()
    hasher.update(new_model_data)
    new_sha = hasher.hexdigest()

    signable = new_sha.encode("utf-8") + b"v2.3.0"
    sig = ota_mgr.sign_payload_for_testing(signable)

    pkg = OTAModelPackage(
        version="v2.3.0",
        model_name="model.plan",
        binary_data=new_model_data,
        sha256_hash=new_sha,
        signature_bytes=sig,
    )

    t0 = time.perf_counter()
    swap_ok = ota_mgr.deploy_and_swap_atomic(pkg)
    swap_duration_ms = (time.perf_counter() - t0) * 1000.0

    assert swap_ok is True
    assert swap_duration_ms < 100.0, f"Atomic hot-swap latency {swap_duration_ms:.2f}ms exceeds 100ms ceiling!"
    assert ota_mgr.current_active_version == "v2.3.0"

    active_dir = ota_mgr.get_active_model_path()
    assert (active_dir / "model.plan").exists()


# ==============================================================================
# 3. Edge Case Harvester Saves 5-Second Video Slice
# ==============================================================================

def test_edge_case_harvester_saves_bundle(tmp_path):
    """
    Verify that an auto-trigger (e.g. low confidence < 0.35) packages
    the 5-second buffer and metadata into a valid zip archive.
    """
    harvest_dir = tmp_path / "harvested"
    harvester = EdgeCaseDataHarvester(harvest_dir=harvest_dir, buffer_duration_s=5.0)

    # Populate 10 frames (simulating rolling 5-second buffer)
    start_ts = 1000.0
    for i in range(10):
        dummy_frame = np.zeros((120, 160, 3), dtype=np.uint8)
        dummy_frame[:, :, i % 3] = 200
        harvester.add_frame(frame_bgr=dummy_frame, confidence=0.85, timestamp=start_ts + (i * 0.5))

    # Trigger with low confidence < 0.35
    bundle = harvester.check_and_trigger(
        confidence=0.20,
        rss_override=False,
        now=start_ts + 5.5,
    )

    assert bundle is not None
    assert bundle.bundle_path.exists()
    assert bundle.trigger_reason == HarvesterTriggerReason.LOW_CONFIDENCE
    assert bundle.frame_count == 10

    # Verify contents inside zipped package
    with zipfile.ZipFile(bundle.bundle_path, "r") as z:
        names = z.namelist()
        assert "metadata.json" in names
        meta = json.loads(z.read("metadata.json").decode("utf-8"))
        assert meta["trigger_reason"] == "LOW_CONFIDENCE"
        assert meta["frame_count"] == 10
        # Verify frame images were written
        assert any(n.startswith("frames/frame_") for n in names)


# ==============================================================================
# 4. CAN ID 0x1A0 Fleet & OTA Status Frame Format
# ==============================================================================

def test_can_0x1a0_fleet_ota_serialization():
    """Verify CAN ID 0x1A0 frame serialization."""
    can_mgr = CANManager(channel="vcan0")

    active_hash_b0 = 0x3F
    sync_state = int(OTASyncState.SYNC_OK)
    shadow_disagreement = 18
    harvest_queue_count = 4
    backend_enum = 3  # TensorRT

    payload = can_mgr.pack_fleet_ota_frame(
        active_model_hash_byte0=active_hash_b0,
        sync_state=sync_state,
        shadow_disagreement_score=shadow_disagreement,
        harvest_queue_count=harvest_queue_count,
        backend_enum=backend_enum,
    )

    assert len(payload) == 8, f"CAN 0x1A0 frame must be exactly 8 bytes, got {len(payload)}"

    decoded = can_mgr.unpack_fleet_ota_frame(payload)
    assert decoded["active_model_hash_byte0"] == 0x3F
    assert decoded["sync_state"] == int(OTASyncState.SYNC_OK)
    assert decoded["shadow_disagreement_score"] == 18
    assert decoded["harvest_queue_count"] == 4
    assert decoded["backend_enum"] == 3
