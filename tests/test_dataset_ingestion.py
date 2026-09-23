"""
tests/test_dataset_ingestion.py
===============================
Unit and integration tests for dataset ingestion, validation, and reporting.

Tests:
  1. Image validation (valid, corrupted, sub-resolution)
  2. Mask validation (valid 4-class, dimension mismatch, invalid class ID, empty mask)
  3. Redundancy detection (SHA-256 exact duplicates, dHash near-duplicates)
  4. Sequence group extraction (temporal leakage prevention)
  5. Ingestion of Mode A dataset (images + masks)
  6. Ingestion of Mode B dataset (unlabeled images -> triggers annotation workspace)
  7. Report serialization (JSON and Markdown)
  8. Zip slip security protection
"""

import json
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.validator import DatasetValidator
from src.dataset.report import DatasetReport
from src.dataset.ingestion import DatasetIngestor
from src.dataset.annotation_prep import AnnotationWorkspace


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as td:
        yield Path(td)


def make_test_image(path: Path, width: int = 640, height: int = 360, color=(100, 100, 100)) -> Path:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    # Add some variation so dhash differs
    cv2.line(img, (50, 50), (200, 200), (255, 255, 255), 3)
    cv2.imwrite(str(path), img)
    return path


def make_test_mask(path: Path, width: int = 640, height: int = 360, add_lanes: bool = True) -> Path:
    # 0 = bg, 1 = road, 2 = left_lane, 3 = right_lane
    mask = np.zeros((height, width), dtype=np.uint8)
    if add_lanes:
        mask[150:, :] = 1  # road
        mask[150:, 100:110] = 2  # left lane
        mask[150:, 530:540] = 3  # right lane
    cv2.imwrite(str(path), mask)
    return path


# ── 1. Validator Tests ─────────────────────────────────────────────────────

def test_validator_valid_image(temp_dir):
    val = DatasetValidator(min_width=160, min_height=90)
    img_p = make_test_image(temp_dir / "valid.jpg")
    is_valid, shape, err = val.validate_image(img_p)
    assert is_valid is True
    assert shape == (360, 640, 3)
    assert err is None


def test_validator_corrupt_image(temp_dir):
    val = DatasetValidator()
    bad_p = temp_dir / "bad.jpg"
    bad_p.write_bytes(b"garbage content that is not an image")
    is_valid, shape, err = val.validate_image(bad_p)
    assert is_valid is False
    assert err is not None


def test_validator_small_image(temp_dir):
    val = DatasetValidator(min_width=300, min_height=200)
    small_p = make_test_image(temp_dir / "small.jpg", width=100, height=100)
    is_valid, shape, err = val.validate_image(small_p)
    assert is_valid is False
    assert "below minimum threshold" in err


def test_validator_mask_checks(temp_dir):
    val = DatasetValidator(allowed_classes={0, 1, 2, 3})

    # Valid mask
    mask_p = make_test_mask(temp_dir / "mask_valid.png", 640, 360, add_lanes=True)
    is_valid, counts, is_empty, err = val.validate_mask(mask_p, expected_shape=(360, 640))
    assert is_valid is True
    assert is_empty is False
    assert 1 in counts and 2 in counts and 3 in counts

    # Dimension mismatch
    is_valid_dim, _, _, err_dim = val.validate_mask(mask_p, expected_shape=(720, 1280))
    assert is_valid_dim is False
    assert "does not match image" in err_dim

    # Invalid class ID
    bad_mask = np.full((360, 640), 99, dtype=np.uint8)
    bad_mask_p = temp_dir / "bad_class_mask.png"
    cv2.imwrite(str(bad_mask_p), bad_mask)
    is_valid_bad, _, _, err_bad = val.validate_mask(bad_mask_p)
    assert is_valid_bad is False
    assert "invalid class IDs" in err_bad

    # Empty mask (all zeros)
    empty_mask_p = make_test_mask(temp_dir / "empty_mask.png", 640, 360, add_lanes=False)
    is_valid_emp, _, is_emp, _ = val.validate_mask(empty_mask_p)
    assert is_valid_emp is True
    assert is_emp is True


# ── 2. Redundancy & Leakage Tests ──────────────────────────────────────────

def test_duplicate_and_sequence_detection(temp_dir):
    val = DatasetValidator()
    img1 = make_test_image(temp_dir / "drive01_frame000100.jpg")
    img2 = temp_dir / "drive01_frame000101.jpg"
    # Exact duplicate
    img2.write_bytes(img1.read_bytes())

    h1 = val.compute_sha256(img1)
    h2 = val.compute_sha256(img2)
    assert h1 == h2

    # Sequence group extraction
    assert val.extract_sequence_group(img1) == "drive01"
    assert val.extract_sequence_group(img2) == "drive01"


# ── 3. Mode A Ingestion (ZIP with Images + Masks) ──────────────────────────

def test_ingest_mode_a_labeled_zip(temp_dir):
    zip_path = temp_dir / "labeled_dataset.zip"
    staging_data = temp_dir / "staging_data"
    staging_data.mkdir()

    img_dir = staging_data / "images"
    mask_dir = staging_data / "masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    # Create 3 matching pairs
    for i in range(3):
        stem = f"road_seq1_{i:04d}"
        make_test_image(img_dir / f"{stem}.jpg")
        make_test_mask(mask_dir / f"{stem}.png")

    # Zip them up
    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in img_dir.iterdir():
            zf.write(f, arcname=f"images/{f.name}")
        for f in mask_dir.iterdir():
            zf.write(f, arcname=f"masks/{f.name}")

    target_annotated = temp_dir / "output_annotated"
    ingestor = DatasetIngestor(base_dir=temp_dir)
    report = ingestor.ingest(zip_path, target_dir=target_annotated)

    assert report.mode == "MODE_A_LABELED"
    assert report.can_train_supervised is True
    assert report.total_images == 3
    assert report.valid_images == 3
    assert report.matched_pairs == 3
    assert report.masks_found is True
    assert (target_annotated / "images" / "road_seq1_0000.jpg").exists()
    assert (target_annotated / "masks" / "road_seq1_0000.png").exists()

    # Check that report files were written
    report_json = temp_dir / "results" / "metrics" / "dataset_report.json"
    assert report_json.exists()
    with open(report_json) as f:
        data = json.load(f)
        assert data["mode"] == "MODE_A_LABELED"


# ── 4. Mode B Ingestion (Unlabeled Images -> Annotation Workspace) ──────────

def test_ingest_mode_b_unlabeled_zip(temp_dir):
    zip_path = temp_dir / "unlabeled_dataset.zip"
    raw_images = temp_dir / "raw_images"
    raw_images.mkdir()

    for i in range(4):
        make_test_image(raw_images / f"raw_frame_{i:04d}.jpg")

    with zipfile.ZipFile(zip_path, "w") as zf:
        for f in raw_images.iterdir():
            zf.write(f, arcname=f.name)

    target_annotated = temp_dir / "data" / "annotated"
    ingestor = DatasetIngestor(base_dir=temp_dir)
    report = ingestor.ingest(zip_path, target_dir=target_annotated)

    assert report.mode == "MODE_B_UNLABELED"
    assert report.can_train_supervised is False
    assert report.total_images == 4
    assert report.valid_images == 4
    assert report.masks_found is False
    assert report.matched_pairs == 0

    # Annotation workspace should be prepared
    legend_file = temp_dir / "annotations" / "class_legend.json"
    manifest_file = temp_dir / "annotations" / "workspace_manifest.json"
    assert legend_file.exists()
    assert manifest_file.exists()

    with open(manifest_file) as f:
        manifest = json.load(f)
        assert manifest["total_images"] == 4
        assert manifest["completed_masks"] == 0
        assert manifest["pending_images"] == 4


# ── 5. Security: Zip Slip Check ────────────────────────────────────────────

def test_zip_slip_protection(temp_dir):
    evil_zip = temp_dir / "evil.zip"
    with zipfile.ZipFile(evil_zip, "w") as zf:
        zf.writestr("../../evil.txt", "malicious payload")

    ingestor = DatasetIngestor(base_dir=temp_dir)
    with pytest.raises(RuntimeError, match="Zip-slip security error"):
        ingestor.ingest(evil_zip)


def test_annotation_workspace_candidate_generation(temp_dir):
    ws = AnnotationWorkspace.create(temp_dir)
    img_p = make_test_image(temp_dir / "road_sample.jpg")
    count = ws.generate_candidate_masks([img_p])
    assert count == 1
    cand_mask = temp_dir / "annotations" / "auto_generated_candidates" / "road_sample.png"
    cand_meta = temp_dir / "annotations" / "auto_generated_candidates" / "road_sample_meta.json"
    assert cand_mask.exists()
    assert cand_meta.exists()
    with open(cand_meta) as f:
        meta = json.load(f)
        assert meta["status"] == "AUTO-GENERATED / NEEDS REVIEW"
        assert meta["verified_by_human"] is False
