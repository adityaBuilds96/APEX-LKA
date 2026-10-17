"""
tests/test_ota_updater.py
=========================
Comprehensive unit tests for Cryptographic OTA Model Manager,
Ed25519 signature verification, manifest validation, quarantine,
atomic hot-swapping (<= 5 ms benchmark), and rollback recovery.
"""

import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
import pytest

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


@pytest.fixture
def ota_env(tmp_path):
    """Fixture providing isolated temporary workspace and authentic keys."""
    keys_dir = tmp_path / "keys"
    keys_dir.mkdir()
    pub_key_path, priv_key_path = generate_keypair(keys_dir)

    staging_dir = tmp_path / "models" / "staging"
    active_dir = tmp_path / "models" / "active"
    quarantine_dir = tmp_path / "models" / "quarantine"

    manager = OTAModelManager(
        public_key_path=pub_key_path,
        staging_dir=staging_dir,
        active_dir=active_dir,
        quarantine_dir=quarantine_dir,
    )
    return {
        "tmp_path": tmp_path,
        "manager": manager,
        "pub_key_path": pub_key_path,
        "priv_key_path": priv_key_path,
    }


def test_keypair_generation_and_signature(ota_env):
    """Verify Ed25519 keypair generation, signing, and signature verification."""
    priv_key = ota_env["priv_key_path"]
    pub_key = ota_env["pub_key_path"]

    test_data = b"APEX_LKA_EDGE_TENSORRT_PLAN_ARTIFACT_V2.1.0"
    sig = sign_artifact(test_data, priv_key)
    assert isinstance(sig, str)
    assert len(sig) > 20

    # Signature verifies against authentic public key
    assert verify_signature(test_data, sig, pub_key) is True

    # Tampered data fails verification
    tampered_data = b"TAMPERED_LKA_EDGE_ARTIFACT"
    assert verify_signature(tampered_data, sig, pub_key) is False


def test_ota_manifest_creation_and_hash(ota_env):
    """Verify manifest creation and SHA-256 calculation."""
    artifact_bytes = b"MODEL_BINARY_MOCK_CONTENT"
    sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    sig = sign_artifact(artifact_bytes, ota_env["priv_key_path"])

    manifest = OTAManifest(
        model_version="2.0.0",
        model_type=ModelBackendType.ONNX,
        sha256_hash=sha256,
        signature_b64=sig,
        min_app_version="1.0.0",
        artifact_filename="model.onnx",
        file_size_bytes=len(artifact_bytes),
    )

    d = manifest.to_dict()
    assert d["model_version"] == "2.0.0"
    assert d["sha256_hash"] == sha256

    reconstructed = OTAManifest.from_dict(d)
    assert reconstructed.model_version == "2.0.0"
    assert reconstructed.model_type == ModelBackendType.ONNX


def test_download_and_verify_valid_artifact(ota_env):
    """Verify downloading and authenticating a valid package."""
    manager = ota_env["manager"]
    priv_key = ota_env["priv_key_path"]

    # Create dummy artifact file
    source_file = ota_env["tmp_path"] / "source_model.onnx"
    payload = b"VALID_TRAINED_NEURAL_NETWORK_WEIGHTS_FOR_TESTING"
    source_file.write_bytes(payload)

    sha256 = hashlib.sha256(payload).hexdigest()
    sig = sign_artifact(payload, priv_key)

    manifest = OTAManifest(
        model_version="2.1.0",
        model_type=ModelBackendType.ONNX,
        sha256_hash=sha256,
        signature_b64=sig,
        artifact_filename="source_model.onnx",
        file_size_bytes=len(payload),
    )

    verified_path = manager.download_and_verify(manifest, source_url_or_path=source_file)
    assert verified_path is not None
    assert verified_path.exists()
    assert manager.sync_state == OTASyncState.SYNC_OK


def test_rejection_and_quarantine_tampered_payload(ota_env):
    """Verify that artifact with mismatched SHA-256 is quarantined."""
    manager = ota_env["manager"]
    priv_key = ota_env["priv_key_path"]

    source_file = ota_env["tmp_path"] / "corrupted_model.onnx"
    payload = b"ORIGINAL_PAYLOAD"
    corrupted_payload = b"CORRUPTED_PAYLOAD_MODIFIED"
    source_file.write_bytes(corrupted_payload)

    # Sign original payload but provide corrupted file
    sig = sign_artifact(payload, priv_key)
    manifest = OTAManifest(
        model_version="2.1.1",
        model_type=ModelBackendType.ONNX,
        sha256_hash=hashlib.sha256(payload).hexdigest(),
        signature_b64=sig,
        artifact_filename="corrupted_model.onnx",
    )

    verified_path = manager.download_and_verify(manifest, source_url_or_path=source_file)
    assert verified_path is None
    assert manager.sync_state == OTASyncState.ERROR

    # Check quarantine directory
    quarantined_files = list(manager.quarantine_dir.glob("quarantined_*"))
    assert len(quarantined_files) >= 1


def test_rejection_and_quarantine_invalid_signature(ota_env):
    """Verify that an artifact with invalid signature is rejected."""
    manager = ota_env["manager"]

    source_file = ota_env["tmp_path"] / "bad_sig_model.onnx"
    payload = b"SOME_UNAUTHENTICATED_ARTIFACT"
    source_file.write_bytes(payload)

    manifest = OTAManifest(
        model_version="2.2.0",
        model_type=ModelBackendType.ONNX,
        sha256_hash=hashlib.sha256(payload).hexdigest(),
        signature_b64="SW52YWxpZFNpZ25hdHVyZUZvckVkMjU1MTk=",
        artifact_filename="bad_sig_model.onnx",
    )

    verified_path = manager.download_and_verify(manifest, source_url_or_path=source_file)
    assert verified_path is None
    assert manager.sync_state == OTASyncState.ERROR


def test_atomic_hotswap_performance_benchmark(ota_env):
    """
    CRITICAL HARD CONSTRAINT:
    Swapping active predictor pointer from model A -> B MUST execute in <= 5 ms.
    """
    manager = ota_env["manager"]
    initial_predictor = MockPredictor(name="prod_v1", version="1.0.0")
    manager.active_predictor = initial_predictor
    manager.active_version = "1.0.0"

    new_predictor = MockPredictor(name="candidate_v2", version="2.0.0")

    # Time atomic hot-swap
    t0 = time.perf_counter()
    swap_ok = manager.atomic_hotswap(new_predictor, target_version="2.0.0")
    t1 = time.perf_counter()

    elapsed_ms = (t1 - t0) * 1000.0
    assert swap_ok is True
    assert manager.active_predictor.version == "2.0.0"
    assert manager.previous_predictor.version == "1.0.0"
    assert elapsed_ms <= 5.0, f"Hot-swap took {elapsed_ms:.3f} ms, exceeding 5.0 ms ceiling!"


def test_rollback_mechanism(ota_env):
    """Verify automatic rollback restoring previous valid predictor."""
    manager = ota_env["manager"]
    pred1 = MockPredictor(name="v1", version="1.0.0")
    pred2 = MockPredictor(name="v2", version="2.0.0")

    manager.active_predictor = pred1
    manager.active_version = "1.0.0"

    # Hotswap to v2
    manager.atomic_hotswap(pred2, target_version="2.0.0")
    assert manager.active_predictor.version == "2.0.0"

    # Simulate issue and trigger rollback
    rollback_ok = manager.rollback()
    assert rollback_ok is True
    assert manager.active_predictor.version == "1.0.0"
    assert manager.active_version == "1.0.0"


def test_ota_version_hash_derivation(ota_env):
    """Verify 16-bit version hash matches expected lower 16 bits of SHA-256."""
    manager = ota_env["manager"]
    manager.active_version = "v1.2.3"
    expected_hash = int.from_bytes(hashlib.sha256(b"v1.2.3").digest()[:2], byteorder="little")
    assert manager.active_version_hash == expected_hash
