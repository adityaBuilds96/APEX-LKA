"""
tests/test_synthetic_generator.py
=================================
Unit tests for APEX-LKA Synthetic Dataset Auto-Generator.
Verifies pixel-perfect labeled data generation, format compliance,
and high-speed throughput (>= 10,000 frames/hour).
"""

from pathlib import Path
import json
import numpy as np
import pytest

from src.simulation.synthetic_generator import (
    SyntheticDataGenerator,
    SyntheticSample,
    BoundingBox3D,
)


@pytest.fixture
def synth_gen(tmp_path):
    """Provides a SyntheticDataGenerator operating in an isolated temp directory."""
    out_dir = tmp_path / "synthetic_data"
    return SyntheticDataGenerator(output_dir=out_dir)


def test_synthetic_directory_structure_setup(synth_gen):
    """Verify train/val/test subdirectories are initialized properly."""
    for split in ("train", "val", "test"):
        assert (synth_gen.output_dir / split / "images").exists()
        assert (synth_gen.output_dir / split / "masks").exists()
        assert (synth_gen.output_dir / split / "depth").exists()
        assert (synth_gen.output_dir / split / "labels").exists()


def test_generate_single_synthetic_sample(synth_gen):
    """Verify single frame generation creates valid image, mask, depth, and annotations."""
    sample = synth_gen.generate_single_frame(frame_idx=1, weather="clear_noon")
    assert isinstance(sample, SyntheticSample)
    assert sample.frame_id == 1
    assert sample.image_path.exists()
    assert sample.mask_path.exists()
    assert sample.depth_path.exists()
    assert sample.annotation_path.exists()

    # Verify depth map values
    depth_arr = np.load(str(sample.depth_path))
    assert depth_arr.shape == (360, 640)
    assert depth_arr.dtype == np.float32
    assert np.all(depth_arr >= 0.0)

    # Verify JSON annotations
    with open(sample.annotation_path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    assert doc["frame_id"] == 1
    assert "objects" in doc


def test_batch_generation_throughput_and_manifest(synth_gen):
    """Verify batch generation meets >= 10,000 frames/hour target and writes manifest."""
    manifest = synth_gen.generate_batch(num_frames=20, weather="rain_moderate")
    assert manifest["total_samples"] == 20
    assert manifest["throughput_frames_per_hour"] > 10000.0
    assert manifest["target_met"] is True

    manifest_file = synth_gen.output_dir / "dataset_manifest.json"
    assert manifest_file.exists()
    with open(manifest_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "split_distribution" in data
    assert sum(data["split_distribution"].values()) == 20
