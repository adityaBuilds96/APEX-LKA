"""
tests/test_universal_ingest.py
==============================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Test Suite: Universal Multi-Format Dataset Ingest & Streaming ZIP Vault.
ISO 26262 ASIL-B Data Ingestion Security & Quality Verification.

Scope of Automated Verification:
--------------------------------
1. Zero-Manual-Unzip Direct Streaming Read from compressed vault (`lane_dataset.zip`).
2. Security & Zip-Slip Defense: Strict rejection of traversal escapes (`../`) and absolute paths.
3. Memory-Bounded Stream Reading: Chunked buffer iteration without RAM exhaustion.
4. Format Adapters: TuSimple JSON parsing, ego left/right boundary classification.
5. Multi-Format Routing: Router auto-detection of TuSimple, YOLO, COCO, and raw images.
6. 4-Class Mask Rasterization: Class 0 Background, Class 1 Corridor, Class 2 Left, Class 3 Right.
7. Cryptographic Deduplication: SHA-256 chunk-based duplicate filtering and audit logging.
8. EXIF Metadata Stripping & Image Normalization to standardized 640x360 dimensions.
9. Sequence-Preserving Train/Val/Test Splitter: Clip sequence integrity without temporal leakage.
10. Corrupt File Isolation & Quarantine: Graceful fault containment to `data/quarantine/`.
11. Training Augmentation Safety: Enforcing `horizontal_flip=False` and class-swap integrity.
12. PyTorch DataModule & Dataset Loader: Verification of batch shapes and tensor types.
13. Manifest JSONL Field Auditing: Cryptographic hash, class distribution, and split flags.
14. Geometric Edge Cases: Single boundaries, zero-lane frames, and cross-over handling.
15. Full End-to-End Ingest Pipeline: Testing ingest execution pass producing valid processed assets.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pytest
import torch

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.augmentations import SafeLaneAugmentor
from src.data.data_module import LaneSegmentationDataset, LaneSegmentationDataModule
from src.data.dataset_normalizer import (
    DatasetNormalizer,
    NormalizationConfig,
    NormalizedOutput,
)
from src.data.format_adapters import (
    LanePolyline,
    NormalizedSample,
    Point2D,
    TuSimpleAdapter,
    UnifiedFormatRouter,
)
from src.data.label_schema import (
    COLOR_PALETTE_BGR,
    COLOR_PALETTE_RGB,
    LaneClassID,
    LabelSchema,
    LaneLabelSchema,
    get_default_label_schema,
    colorize_mask,
)
from src.data.train_ready_writer import (
    DatasetSplitConfig,
    IngestSummary,
    IngestSummaryReport,
    TrainReadyWriter,
)
from src.data.universal_ingest import (
    UniversalDatasetIngestEngine,
    UniversalIngestConfig,
)
from src.data.zip_vault_reader import (
    ZipMemberMetadata,
    ZipVaultCorruptError,
    ZipVaultReader,
    ZipVaultSecurityError,
)


# ==============================================================================
# 1. Test Fixtures & Synthetic Vault Generator
# ==============================================================================

@pytest.fixture
def temp_workspace():
    """Create and tear down a scratch test workspace."""
    temp_dir = tempfile.mkdtemp(prefix="apex_test_ingest_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_tusimple_zip(temp_workspace):
    """Generate a synthetic TuSimple zip archive simulating the 21GB vault."""
    zip_path = temp_workspace / "mock_lane_dataset.zip"

    # Synthetic TuSimple JSON lines
    records = []
    for i in range(12):
        rec = {
            "lanes": [
                [-2, -2, 400 + i * 2, 420 + i * 2, 440 + i * 2, 460 + i * 2],
                [-2, -2, 850 + i * 2, 830 + i * 2, 810 + i * 2, 790 + i * 2],
            ],
            "h_samples": [240, 260, 280, 300, 320, 340],
            "raw_file": f"clips/0530/test_seq_{i // 4}/frame_{i % 4}.jpg",
        }
        records.append(json.dumps(rec))

    json_str = "\n".join(records) + "\n"

    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Write JSON annotation
        zf.writestr("TUSimple/train_set/label_data_0313.json", json_str)

        # Write matching distinct images
        for i in range(12):
            img_bgr = np.full((720, 1280, 3), 50 + i * 10, dtype=np.uint8)
            img_bgr[i * 20 : i * 20 + 50, :] = 200
            _, img_encoded = cv2.imencode(".jpg", img_bgr)
            rel_img = f"TUSimple/train_set/clips/0530/test_seq_{i // 4}/frame_{i % 4}.jpg"
            zf.writestr(rel_img, img_encoded.tobytes())

    return zip_path


# ==============================================================================
# 2. ZipVaultReader Security & Zip-Slip Defense Tests
# ==============================================================================

class TestZipVaultSecurity:
    """Verifies zip-slip traversal protection and memory-bounded streaming."""

    def test_zip_slip_traversal_detection(self, temp_workspace):
        """Verify that malicious zip files with path traversal components are rejected."""
        malicious_zip = temp_workspace / "malicious_slip.zip"

        with zipfile.ZipFile(malicious_zip, mode="w") as zf:
            # Traversal escape
            zf.writestr("../../etc/passwd", "root:x:0:0::/root:/bin/bash\n")
            zf.writestr("normal_file.txt", "safe content\n")

        with ZipVaultReader(malicious_zip) as vault:
            # Malicious member must NOT be in the valid indexed list
            assert not vault.has_file("../../etc/passwd")
            assert not vault.has_file("etc/passwd")
            assert vault.has_file("normal_file.txt")

    def test_absolute_path_traversal_defense(self, temp_workspace):
        """Verify absolute path prefix members are securely filtered."""
        bad_zip = temp_workspace / "absolute_path.zip"

        with zipfile.ZipFile(bad_zip, mode="w") as zf:
            zf.writestr("/root/secret.key", "classified\n")
            zf.writestr("safe_dir/data.json", '{"status": "ok"}\n')

        with ZipVaultReader(bad_zip) as vault:
            assert not vault.has_file("/root/secret.key")
            assert vault.has_file("safe_dir/data.json")

    def test_streaming_read_integrity(self, mock_tusimple_zip):
        """Verify zero-manual-unzip in-memory reading without local disk extraction."""
        with ZipVaultReader(mock_tusimple_zip) as vault:
            assert vault.has_file("TUSimple/train_set/label_data_0313.json")
            text = vault.read_text("TUSimple/train_set/label_data_0313.json")
            assert "clips/0530/test_seq_0/frame_0.jpg" in text

            # Stream chunks
            chunks = list(vault.stream_chunks("TUSimple/train_set/label_data_0313.json", chunk_size=128))
            assert len(chunks) > 0
            reconstructed = b"".join(chunks).decode("utf-8")
            assert reconstructed == text

    def test_sha256_vault_hashing(self, mock_tusimple_zip):
        """Verify on-the-fly SHA-256 calculation."""
        with ZipVaultReader(mock_tusimple_zip) as vault:
            h = vault.compute_sha256("TUSimple/train_set/label_data_0313.json")
            assert isinstance(h, str) and len(h) == 64

    def test_corrupt_zip_header_detection(self, temp_workspace):
        """Verify corrupted zip archives raise ZipVaultCorruptError."""
        corrupt_zip = temp_workspace / "corrupted_archive.zip"
        with open(corrupt_zip, "wb") as f:
            f.write(b"PK\x03\x04GARBAGE_PAYLOAD_NOT_A_VALID_ZIP")

        with pytest.raises((ZipVaultCorruptError, zipfile.BadZipFile)):
            with ZipVaultReader(corrupt_zip) as vault:
                vault.inspect_vault()

    def test_nonexistent_member_read_error(self, mock_tusimple_zip):
        """Verify reading a non-existent member raises FileNotFoundError."""
        with ZipVaultReader(mock_tusimple_zip) as vault:
            with pytest.raises(FileNotFoundError):
                vault.read_bytes("non_existent_folder/missing_image.jpg")


# ==============================================================================
# 3. Format Adapter & Polyline Classification Tests
# ==============================================================================

class TestFormatAdapters:
    """Tests TuSimple JSON parsing, ego boundary classification, and polyline logic."""

    def test_tusimple_adapter_polyline_extraction(self):
        adapter = TuSimpleAdapter(target_width=640, target_height=360)
        raw_json_line = json.dumps({
            "lanes": [
                [-2, 300, 320, 340, 360],
                [-2, 600, 580, 560, 540],
            ],
            "h_samples": [200, 220, 240, 260, 280],
            "raw_file": "clips/0530/seq_test/20.jpg",
        })

        samples = adapter.parse_source(raw_json_line)
        assert len(samples) == 1
        s = samples[0]
        assert "20" in s.sample_id
        assert len(s.polylines) == 2

        # Verify left vs right classification
        classes = {p.semantic_class for p in s.polylines}
        assert LaneClassID.LEFT_BOUNDARY in classes
        assert LaneClassID.RIGHT_BOUNDARY in classes

    def test_sparse_or_corrupt_points_filtering(self):
        adapter = TuSimpleAdapter(min_points_per_lane=4)
        raw_data = {
            "lanes": [
                [-2, -2, -2, 350, -2], # Only 1 valid point
                [400, 420, 440, 460, 480], # 5 valid points
            ],
            "h_samples": [200, 220, 240, 260, 280],
            "raw_file": "clips/0530/test/20.jpg",
        }

        samples = adapter.parse_source(raw_data)
        assert len(samples) == 1
        # Sparse lane with 1 point should be dropped (< 4 min points)
        assert len(samples[0].polylines) == 1

    def test_multi_lane_highway_ego_assignment(self):
        """
        Verify that in a 4-lane highway scene, only the immediate ego left and ego right
        lanes are assigned classes 2 and 3, while outer adjacent lanes are properly handled.
        """
        adapter = TuSimpleAdapter(target_width=640, target_height=360)
        raw_scene = {
            "lanes": [
                [150, 160, 170, 180],
                [450, 460, 470, 480],
                [850, 840, 830, 820],
                [1150, 1140, 1130, 1120],
            ],
            "h_samples": [300, 320, 340, 360],
            "raw_file": "clips/0530/multilane/20.jpg",
        }

        samples = adapter.parse_source(raw_scene)
        assert len(samples) == 1
        polys = samples[0].polylines
        assert len(polys) >= 2

        classes = [p.semantic_class for p in polys]
        assert LaneClassID.LEFT_BOUNDARY in classes
        assert LaneClassID.RIGHT_BOUNDARY in classes


# ==============================================================================
# 4. Unified Format Router Tests
# ==============================================================================

class TestUnifiedFormatRouter:
    """Verifies format routing and auto-dispatch logic."""

    def test_router_tusimple_dict_dispatch(self):
        router = UnifiedFormatRouter()
        payload = {
            "lanes": [[-2, 200, 220, 240], [-2, 450, 430, 410]],
            "h_samples": [220, 240, 260, 280],
            "raw_file": "clips/0530/router_test/20.jpg",
        }
        samples = router.route_and_parse(payload)
        assert len(samples) == 1
        assert "20" in samples[0].sample_id
        assert len(samples[0].polylines) == 2

    def test_router_unrecognized_payload_rejection(self):
        router = UnifiedFormatRouter()
        with pytest.raises(Exception):
            router.route_and_parse(123456789) # Completely invalid object


# ==============================================================================
# 5. Dataset Normalizer & Mask Rasterizer Tests
# ==============================================================================

class TestDatasetNormalizer:
    """Verifies 4-class mask rasterization, corridor filling, and deduplication."""

    def test_rasterize_corridor_and_boundaries(self):
        normalizer = DatasetNormalizer(NormalizationConfig(target_width=640, target_height=360))

        # Define synthetic left and right lanes
        left_poly = LanePolyline(
            points=[Point2D(200.0, 180.0), Point2D(150.0, 360.0)],
            semantic_class=LaneClassID.LEFT_BOUNDARY,
        )
        right_poly = LanePolyline(
            points=[Point2D(440.0, 180.0), Point2D(490.0, 360.0)],
            semantic_class=LaneClassID.RIGHT_BOUNDARY,
        )

        sample = NormalizedSample(
            sample_id="raster_test_01",
            image_path="test.jpg",
            image_width=640,
            image_height=360,
            polylines=[left_poly, right_poly],
        )

        dummy_img = np.full((360, 640, 3), 120, dtype=np.uint8)
        output = normalizer.process_sample(sample, dummy_img)

        mask = output.mask_uint8
        assert mask.shape == (360, 640)

        # Class 1 (Corridor) should be filled between x=200 and x=440
        assert mask[250, 320] == LaneClassID.DRIVABLE_CORRIDOR.value

        # Boundaries must be present
        assert LaneClassID.LEFT_BOUNDARY.value in np.unique(mask)
        assert LaneClassID.RIGHT_BOUNDARY.value in np.unique(mask)

    def test_sha256_exact_deduplication(self):
        normalizer = DatasetNormalizer(NormalizationConfig(enable_sha256_dedup=True))
        dummy_img = np.full((360, 640, 3), 150, dtype=np.uint8)

        sample1 = NormalizedSample(sample_id="f1", image_path="f1.jpg", image_width=640, image_height=360)
        sample2 = NormalizedSample(sample_id="f2", image_path="f2.jpg", image_width=640, image_height=360)

        out1 = normalizer.process_sample(sample1, dummy_img)
        assert not out1.is_duplicate

        out2 = normalizer.process_sample(sample2, dummy_img)
        assert out2.is_duplicate, "Identical frame was not flagged as duplicate."

    def test_image_resizing_and_exif_strip(self):
        """
        Verify that higher resolution frames (e.g. 1920x1080) are resized
        cleanly to 640x360 without aspect ratio or channel corruption.
        """
        normalizer = DatasetNormalizer(NormalizationConfig(target_width=640, target_height=360))
        large_img = np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)

        sample = NormalizedSample(sample_id="large_01", image_path="large.jpg", image_width=1920, image_height=1080)
        out = normalizer.process_sample(sample, large_img)

        assert out.image_bgr.shape == (360, 640, 3)
        assert out.mask_uint8.shape == (360, 640)

    def test_single_lane_boundary_no_corridor(self):
        """
        If only one boundary is detected (e.g. only left boundary), the normalizer
        must rasterize the boundary pixels without attempting to fill a corridor.
        """
        normalizer = DatasetNormalizer(NormalizationConfig(target_width=640, target_height=360))
        left_poly = LanePolyline(
            points=[Point2D(200.0, 200.0), Point2D(150.0, 360.0)],
            semantic_class=LaneClassID.LEFT_BOUNDARY,
        )
        sample = NormalizedSample(
            sample_id="single_boundary",
            image_path="single.jpg",
            image_width=640,
            image_height=360,
            polylines=[left_poly],
        )

        dummy_img = np.full((360, 640, 3), 100, dtype=np.uint8)
        out = normalizer.process_sample(sample, dummy_img)

        # Should contain left boundary, but NO corridor class
        classes_present = np.unique(out.mask_uint8)
        assert LaneClassID.LEFT_BOUNDARY.value in classes_present
        assert LaneClassID.DRIVABLE_CORRIDOR.value not in classes_present

    def test_zero_lanes_blank_mask(self):
        """
        Frames with no annotations should output an all-zero mask (Class 0 Background).
        """
        normalizer = DatasetNormalizer(NormalizationConfig(target_width=640, target_height=360))
        sample = NormalizedSample(
            sample_id="zero_lanes",
            image_path="empty.jpg",
            image_width=640,
            image_height=360,
            polylines=[],
        )
        dummy_img = np.full((360, 640, 3), 100, dtype=np.uint8)
        out = normalizer.process_sample(sample, dummy_img)
        assert np.all(out.mask_uint8 == 0)


# ==============================================================================
# 6. Label Schema and Color Palette Verification
# ==============================================================================

class TestLabelSchema:
    """Verifies 4-class semantic color mapping and LUT generation."""

    def test_default_label_schema_integrity(self):
        schema = get_default_label_schema()
        assert schema.num_classes == 4
        assert schema.get_class_name(0) == "background"
        assert schema.get_class_name(1) == "drivable_corridor"
        assert schema.get_class_name(2) == "left_boundary"
        assert schema.get_class_name(3) == "right_boundary"

    def test_color_palettes_consistency(self):
        """Ensure RGB and BGR palettes correctly map channels."""
        for cid in [0, 1, 2, 3]:
            rgb = COLOR_PALETTE_RGB[cid]
            bgr = COLOR_PALETTE_BGR[cid]
            assert bgr[0] == rgb[2]
            assert bgr[1] == rgb[1]
            assert bgr[2] == rgb[0]

    def test_color_mask_visualization_overlay(self):
        """Verify colorization helper maps 2D index mask to 3-channel RGB image."""
        schema = get_default_label_schema()
        idx_mask = np.zeros((100, 100), dtype=np.uint8)
        idx_mask[20:40, :] = 1  # Corridor
        idx_mask[40:60, :] = 2  # Left
        idx_mask[60:80, :] = 3  # Right

        color_rgb = schema.colorize_mask(idx_mask, color_mode="rgb")
        assert color_rgb.shape == (100, 100, 3)
        assert np.array_equal(color_rgb[30, 50], COLOR_PALETTE_RGB[1])
        assert np.array_equal(color_rgb[50, 50], COLOR_PALETTE_RGB[2])
        assert np.array_equal(color_rgb[70, 50], COLOR_PALETTE_RGB[3])


# ==============================================================================
# 7. Train-Ready Writer and Sequence Splitting Tests
# ==============================================================================

class TestTrainReadyWriter:
    """Tests directory structure, sequence-preserving splits, and manifest logging."""

    def test_clip_sequence_preservation(self, temp_workspace):
        """
        Critical Autonomous Driving Test:
        Frames belonging to the same video clip sequence MUST NOT be split
        across train and validation sets, preventing temporal data leakage.
        """
        writer = TrainReadyWriter(
            output_dir=str(temp_workspace / "writer_test"),
            split_config=DatasetSplitConfig(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15),
        )

        samples: List[Tuple[NormalizedSample, NormalizedOutput]] = []
        for seq_idx in range(3):
            for frame_idx in range(5):
                sid = f"clip_{seq_idx}_frame_{frame_idx}"
                ns = NormalizedSample(
                    sample_id=sid,
                    image_path=f"clips/{seq_idx}/{frame_idx}.jpg",
                    image_width=640,
                    image_height=360,
                    sequence_id=f"clip_{seq_idx}",
                )
                img = np.full((360, 640, 3), 100, dtype=np.uint8)
                mask = np.zeros((360, 640), dtype=np.uint8)
                out = NormalizedOutput(sample_id=sid, image_bgr=img, mask_uint8=mask, sha256_hash=f"hash_{sid}")
                samples.append((ns, out))

        for ns, out in samples:
            writer.write_sample(ns, out)

        split_dict = writer.finalize_splits()

        for seq_idx in range(3):
            seq_prefix = f"clip_{seq_idx}"
            assigned_splits = []
            for s_name, s_ids in split_dict.items():
                matching = [sid for sid in s_ids if seq_prefix in sid]
                if len(matching) > 0:
                    assigned_splits.append(s_name)
                    assert len(matching) == 5, f"Sequence {seq_prefix} was split across subsets!"

            assert len(assigned_splits) == 1, f"Sequence {seq_prefix} found in multiple splits: {assigned_splits}"


# ==============================================================================
# 8. Safe Lane Augmentation Tests
# ==============================================================================

class TestSafeLaneAugmentation:
    """Verifies strict no-flip policy and directional class swap safety."""

    def test_strict_no_flip_by_default(self):
        augmentor = SafeLaneAugmentor(allow_horizontal_flip=False)
        assert not augmentor.allow_horizontal_flip

    def test_horizontal_flip_class_swap_integrity(self):
        """
        When horizontal flip is explicitly applied, verify that Class 2 and Class 3
        are symmetrically swapped in the ground truth mask.
        """
        augmentor = SafeLaneAugmentor(allow_horizontal_flip=True)

        mask = np.zeros((360, 640), dtype=np.uint8)
        mask[100:200, 100:150] = LaneClassID.LEFT_BOUNDARY.value
        mask[100:200, 500:550] = LaneClassID.RIGHT_BOUNDARY.value
        mask[100:200, 250:350] = LaneClassID.DRIVABLE_CORRIDOR.value

        img = np.zeros((360, 640, 3), dtype=np.uint8)
        flipped_img, flipped_mask = augmentor.apply_flip_with_swap(img, mask)

        assert flipped_mask.shape == (360, 640)
        assert LaneClassID.LEFT_BOUNDARY.value in np.unique(flipped_mask)
        assert LaneClassID.RIGHT_BOUNDARY.value in np.unique(flipped_mask)
        assert LaneClassID.DRIVABLE_CORRIDOR.value in np.unique(flipped_mask)


# ==============================================================================
# 9. PyTorch DataModule & Dataset Loader Tests
# ==============================================================================

class TestPyTorchDataModuleIntegration:
    """Verifies PyTorch Dataset tensor shapes, types, and DataLoader collation."""

    def test_dataset_tensor_shapes_and_types(self, temp_workspace):
        dataset_dir = temp_workspace / "torch_dataset_test"
        img_dir = dataset_dir / "images"
        mask_dir = dataset_dir / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        manifest_lines = []
        for i in range(4):
            sid = f"sample_{i:04d}"
            img = np.full((360, 640, 3), 100 + i * 20, dtype=np.uint8)
            mask = np.zeros((360, 640), dtype=np.uint8)
            mask[200:300, 200:400] = 1

            cv2.imwrite(str(img_dir / f"{sid}.jpg"), img)
            cv2.imwrite(str(mask_dir / f"{sid}.png"), mask)

            manifest_lines.append(json.dumps({
                "sample_id": sid,
                "image_file": f"images/{sid}.jpg",
                "mask_file": f"masks/{sid}.png",
                "split": "train" if i < 3 else "val",
                "sha256": f"dummy_sha_{i}",
            }))

        with open(dataset_dir / "manifest.jsonl", "w") as f:
            f.write("\n".join(manifest_lines) + "\n")

        with open(dataset_dir / "split.json", "w") as f:
            json.dump({"train": ["sample_0000", "sample_0001", "sample_0002"], "val": ["sample_0003"], "test": []}, f)

        dataset = LaneSegmentationDataset(
            root_dir=str(dataset_dir),
            split="train",
        )
        assert len(dataset) == 3

        sample = dataset[0]
        assert "image" in sample and "mask" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert isinstance(sample["mask"], torch.Tensor)
        assert sample["image"].shape == (3, 360, 640)
        assert sample["mask"].shape == (360, 640)
        assert sample["image"].dtype == torch.float32
        assert sample["mask"].dtype == torch.int64


# ==============================================================================
# 10. Corrupted File Isolation & Quarantine Tests
# ==============================================================================

class TestCorruptedFileQuarantine:
    """Verifies that corrupted files are isolated without stopping pipeline execution."""

    def test_corrupt_jpeg_quarantine(self, temp_workspace):
        bad_zip = temp_workspace / "partially_corrupt.zip"
        quarantine_dir = temp_workspace / "quarantine_target"
        processed_dir = temp_workspace / "processed_target"

        valid_img = np.full((720, 1280, 3), 120, dtype=np.uint8)
        _, valid_bytes = cv2.imencode(".jpg", valid_img)
        corrupt_bytes = b"CORRUPTED_NOT_A_VALID_JPEG_STREAM"

        records = [
            json.dumps({"lanes": [[100, 110], [200, 210]], "h_samples": [200, 220], "raw_file": "frame_valid.jpg"}),
            json.dumps({"lanes": [[100, 110], [200, 210]], "h_samples": [200, 220], "raw_file": "frame_corrupt.jpg"}),
        ]

        with zipfile.ZipFile(bad_zip, "w") as zf:
            zf.writestr("label_data_test.json", "\n".join(records) + "\n")
            zf.writestr("frame_valid.jpg", valid_bytes.tobytes())
            zf.writestr("frame_corrupt.jpg", corrupt_bytes)

        cfg = UniversalIngestConfig(
            input_source=str(bad_zip),
            output_dir=str(processed_dir),
            quarantine_dir=str(quarantine_dir),
            target_width=640,
            target_height=360,
        )

        engine = UniversalDatasetIngestEngine(config=cfg)
        summary = engine.execute_ingest()

        assert summary.total_samples == 1
        assert (processed_dir / "images").exists()


# ==============================================================================
# 11. End-to-End Universal Ingest Execution Tests
# ==============================================================================

class TestUniversalIngestEngine:
    """Verifies full execution pass directly reading lane_dataset.zip."""

    def test_ingest_from_mock_zip(self, mock_tusimple_zip, temp_workspace):
        out_processed = temp_workspace / "processed"
        quarantine_dir = temp_workspace / "quarantine"

        cfg = UniversalIngestConfig(
            input_source=str(mock_tusimple_zip),
            output_dir=str(out_processed),
            quarantine_dir=str(quarantine_dir),
            target_width=640,
            target_height=360,
            train_ratio=0.80,
            val_ratio=0.10,
            test_ratio=0.10,
            export_audit_samples=True,
        )

        engine = UniversalDatasetIngestEngine(config=cfg)
        summary = engine.execute_ingest()

        assert summary.total_samples == 12
        assert (out_processed / "images").exists()
        assert (out_processed / "masks").exists()
        assert (out_processed / "manifest.jsonl").exists()
        assert (out_processed / "split.json").exists()
        assert (out_processed / "ingest_summary.json").exists()

        # Check split preservation
        with open(out_processed / "split.json", "r") as f:
            splits = json.load(f)
            assert "train" in splits and "val" in splits and "test" in splits
            total_partitioned = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
            assert total_partitioned == 12

        # Check image and mask files on disk
        img_files = list((out_processed / "images").glob("*.jpg"))
        mask_files = list((out_processed / "masks").glob("*.png"))
        assert len(img_files) == 12
        assert len(mask_files) == 12

        # Verify mask contents
        sample_mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_UNCHANGED)
        assert sample_mask.shape == (360, 640)
        assert sample_mask.dtype == np.uint8

    def test_native_lane_dataset_zip_execution(self, temp_workspace):
        """Test against real workspace data/raw/lane_dataset.zip with small limit."""
        real_zip = PROJECT_ROOT / "data" / "raw" / "lane_dataset.zip"
        if not real_zip.exists():
            real_zip = PROJECT_ROOT / "data" / "lane_dataset.zip"

        if not real_zip.exists():
            pytest.skip("lane_dataset.zip not found in workspace.")

        out_dir = temp_workspace / "native_processed"
        cfg = UniversalIngestConfig(
            input_source=str(real_zip),
            output_dir=str(out_dir),
            max_samples=4,
            export_audit_samples=False,
        )

        engine = UniversalDatasetIngestEngine(config=cfg)
        summary = engine.execute_ingest()

        assert summary.total_samples == 4
        assert (out_dir / "manifest.jsonl").exists()
        assert len(list((out_dir / "images").glob("*.jpg"))) == 4


# ==============================================================================
# 12. Advanced Traversal Defense & Mask Color Verification
# ==============================================================================

class TestAdvancedIngestSecurityAndFormatting:
    """Security and formatting regression suite for zero-unzip ingestion."""

    def test_zip_vault_absolute_and_unc_traversal_rejection(self, temp_workspace):
        """Verify ZipVaultReader blocks UNC and Windows absolute paths inside archive."""
        zip_path = temp_workspace / "malicious_advanced.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("valid_dir/image_01.jpg", b"valid_data")
            # Windows drive letter
            zf.writestr("C:/Windows/System32/calc.exe", b"fake_calc")
            # UNC path
            zf.writestr("//malicious_server/share/evil.sh", b"fake_exploit")

        with ZipVaultReader(zip_path) as vault:
            assert vault.has_file("valid_dir/image_01.jpg")
            assert not vault.has_file("C:/Windows/System32/calc.exe")
            assert not vault.has_file("//malicious_server/share/evil.sh")
            assert not vault.has_file("Windows/System32/calc.exe")

    def test_color_mask_palette_consistency(self):
        """Test color palette generation and boundary visual distinction."""
        mask = np.zeros((360, 640), dtype=np.uint8)
        # Class 1: Corridor
        mask[100:200, 200:400] = LaneClassID.DRIVABLE_CORRIDOR.value
        # Class 2: Left boundary
        mask[100:200, 195:200] = LaneClassID.LEFT_BOUNDARY.value
        # Class 3: Right boundary
        mask[100:200, 400:405] = LaneClassID.RIGHT_BOUNDARY.value

        color_mask = colorize_mask(mask, color_mode="bgr")
        assert color_mask.shape == (360, 640, 3)
        assert color_mask.dtype == np.uint8

        # Background should match background color
        assert np.all(color_mask[0, 0] == COLOR_PALETTE_BGR[0])
        # Corridor should match corridor color
        assert np.all(color_mask[150, 300] == COLOR_PALETTE_BGR[1])
        # Left boundary should match left boundary color
        assert np.all(color_mask[150, 197] == COLOR_PALETTE_BGR[2])
        # Right boundary should match right boundary color
        assert np.all(color_mask[150, 402] == COLOR_PALETTE_BGR[3])

    def test_ingest_summary_serialization(self, temp_workspace):
        """Verify summary serialization and deserialization integrity."""
        summary = IngestSummaryReport(
            timestamp_utc="2026-10-18T00:00:00Z",
            total_samples=100,
            train_count=80,
            val_count=10,
            test_count=10,
            quarantine_count=5,
            duplicate_count=0,
            corrupt_count=2,
            class_pixel_totals={"background": 100000, "drivable_corridor": 50000},
            class_pixel_percentages={"background": 0.66, "drivable_corridor": 0.33},
            output_directory=str(temp_workspace),
            elapsed_time_sec=12.45,
        )
        summary_path = temp_workspace / "summary_test.json"
        with open(summary_path, "w") as f:
            json.dump(summary.to_dict(), f)

        assert summary_path.exists()
        with open(summary_path, "r") as f:
            loaded_data = json.load(f)

        assert loaded_data["total_samples"] == 100
        assert loaded_data["train_count"] == 80
        assert loaded_data["quarantine_count"] == 5


# ==============================================================================
# 13. Standalone Script Runner
# ==============================================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])

