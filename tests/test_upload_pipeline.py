"""
tests/test_upload_pipeline.py
=============================
Integration tests for PipelineOrchestrator and staging lifecycle (Section 6 & Section 8).
"""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.pipeline_orchestrator import PipelineOrchestrator, PipelineState


def _make_dummy_image(path: Path):
    """Helper to save a valid test image."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    pts = np.array([[200, 150], [440, 150], [639, 359], [0, 359]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    cv2.line(img, (220, 150), (120, 359), (255, 255, 255), 4)
    cv2.line(img, (420, 150), (520, 359), (255, 255, 255), 4)
    cv2.imwrite(str(path), img)


def test_file_type_support_constants():
    orch = PipelineOrchestrator()
    assert ".jpg" in orch.SUPPORTED_IMG_EXTS
    assert ".png" in orch.SUPPORTED_IMG_EXTS
    assert ".mp4" in orch.SUPPORTED_VID_EXTS
    assert ".zip" in orch.SUPPORTED_ARCHIVES


def test_staging_directory_creation_and_cleanup(tmp_path):
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()

    img_p = input_dir / "sample1.jpg"
    _make_dummy_image(img_p)

    orch = PipelineOrchestrator(base_dir=base_dir)
    res = orch.run(source=input_dir, auto_split=False)

    assert res.success is True
    # Verify staging directory is cleaned up
    staging_base = base_dir / "data" / ".staging" / res.batch_id
    assert not staging_base.exists()


def test_batch_id_uniqueness(tmp_path):
    base_dir = tmp_path / "project"
    base_dir.mkdir()
    img_p = tmp_path / "sample.jpg"
    _make_dummy_image(img_p)

    orch = PipelineOrchestrator(base_dir=base_dir)
    res1 = orch.run(source=img_p, auto_split=False)
    res2 = orch.run(source=img_p, auto_split=False)

    assert res1.batch_id != res2.batch_id


def test_pipeline_state_persistence(tmp_path):
    state_file = tmp_path / "pipeline_state.json"
    state = PipelineState(
        batch_id="test1234",
        stage="auto_annotation",
        progress=0.50,
        started_at="2026-09-30T10:00:00",
        frames_total=10,
        frames_processed=5,
    )
    state.save(state_file)

    assert state_file.exists()
    with open(state_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["batch_id"] == "test1234"
    assert data["progress"] == 0.50


def test_pipeline_integration_zip_to_report(tmp_path):
    base_dir = tmp_path / "project"
    base_dir.mkdir()

    # Create test zip archive
    zip_path = tmp_path / "dataset.zip"
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    pts = np.array([[200, 150], [440, 150], [639, 359], [0, 359]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    cv2.line(img, (220, 150), (120, 359), (255, 255, 255), 4)
    cv2.line(img, (420, 150), (520, 359), (255, 255, 255), 4)
    img_bytes = cv2.imencode(".jpg", img)[1].tobytes()

    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("frame_zip_01.jpg", img_bytes)

    orch = PipelineOrchestrator(base_dir=base_dir)
    summary = orch.run(source=zip_path, auto_split=False)

    assert summary.success is True
    assert summary.total_staged == 1
    assert summary.valid_frames == 1
    assert summary.auto_annotated_count == 1
    assert summary.report_json is not None
    assert summary.report_json.exists()
    assert summary.auto_annotation_report_md is not None
    assert summary.auto_annotation_report_md.exists()
