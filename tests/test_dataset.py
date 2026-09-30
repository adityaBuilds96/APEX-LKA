"""
tests/test_dataset.py
======================
Pytest suite for src/training/dataset.py (LaneSegDataset).

Strategy
--------
All tests operate on a synthetic in-memory dataset written to a tmp_path
fixture — no real data required, no network access.

Coverage
--------
- Dataset discovery: finds the right number of pairs.
- __getitem__ returns correct tensor shapes and dtypes.
- Mask class IDs in {0,1,2,3} after transform.
- Corrupt mask (illegal class ID) is skipped gracefully.
- Unreadable mask file is skipped gracefully.
- Val/test split disables augmentation by default.
- class_pixel_counts sums correctly over synthetic masks.
- Missing images/ directory raises FileNotFoundError.
- Missing masks/ directory raises FileNotFoundError.
- Dataset length reflects number of valid pairs.
"""

import struct
import zlib
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pytest
import torch

from src.training.dataset import LaneSegDataset, NUM_CLASSES, VALID_CLASS_IDS


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers — synthetic data factory
# ═══════════════════════════════════════════════════════════════════════════════

def _write_image(path: Path, h: int = 36, w: int = 64) -> None:
    """Write a small synthetic BGR JPEG to path."""
    img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    cv2.imwrite(str(path), img)


def _write_mask(path: Path, class_id: int = 0, h: int = 36, w: int = 64) -> None:
    """Write a synthetic single-class PNG mask to path."""
    mask = np.full((h, w), class_id, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def _write_multiclass_mask(path: Path, h: int = 36, w: int = 64) -> None:
    """Write a mask with all 4 class IDs present."""
    mask = np.zeros((h, w), dtype=np.uint8)
    # Divide into 4 horizontal bands
    band = h // 4
    for c in range(4):
        mask[c * band: (c + 1) * band, :] = c
    cv2.imwrite(str(path), mask)


def _write_corrupt_mask(path: Path) -> None:
    """Write a mask with an illegal class ID (255)."""
    mask = np.full((36, 64), 255, dtype=np.uint8)
    cv2.imwrite(str(path), mask)


def _make_split_dir(root: Path, n_images: int = 4) -> Tuple[Path, Path]:
    """
    Create a split directory with n_images valid pairs.
    Returns (images_dir, masks_dir).
    """
    img_dir  = root / "images"
    mask_dir = root / "masks"
    img_dir.mkdir(parents=True)
    mask_dir.mkdir(parents=True)

    for i in range(n_images):
        _write_image(img_dir / f"frame_{i:04d}.jpg")
        _write_multiclass_mask(mask_dir / f"frame_{i:04d}.png")

    return img_dir, mask_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture()
def train_dir(tmp_path) -> Path:
    root = tmp_path / "train"
    _make_split_dir(root, n_images=6)
    return root


@pytest.fixture()
def val_dir(tmp_path) -> Path:
    root = tmp_path / "val"
    _make_split_dir(root, n_images=3)
    return root


# ═══════════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneSegDatasetDiscovery:

    def test_correct_pair_count(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train")
        assert len(ds) == 6

    def test_missing_images_dir_raises(self, tmp_path):
        root = tmp_path / "bad_split"
        root.mkdir()
        (root / "masks").mkdir()
        with pytest.raises(FileNotFoundError, match="Images directory"):
            LaneSegDataset(root, split="train")

    def test_missing_masks_dir_raises(self, tmp_path):
        root = tmp_path / "bad_split2"
        root.mkdir()
        (root / "images").mkdir()
        with pytest.raises(FileNotFoundError, match="Masks directory"):
            LaneSegDataset(root, split="train")

    def test_images_without_matching_mask_excluded(self, tmp_path):
        root = tmp_path / "partial"
        img_dir  = root / "images"
        mask_dir = root / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        # 3 images, only 2 masks
        for i in range(3):
            _write_image(img_dir / f"img_{i}.jpg")
        for i in range(2):
            _write_mask(mask_dir / f"img_{i}.png")

        ds = LaneSegDataset(root, split="val")
        assert len(ds) == 2


class TestLaneSegDatasetGetItem:

    def test_image_shape(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", height=36, width=64, augment=False)
        img, mask = ds[0]
        assert img.shape == (3, 36, 64), f"Unexpected image shape: {img.shape}"

    def test_image_dtype(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", height=36, width=64, augment=False)
        img, _ = ds[0]
        assert img.dtype == torch.float32

    def test_mask_shape(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", height=36, width=64, augment=False)
        _, mask = ds[0]
        assert mask.shape == (36, 64), f"Unexpected mask shape: {mask.shape}"

    def test_mask_dtype(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", height=36, width=64, augment=False)
        _, mask = ds[0]
        assert mask.dtype == torch.int64

    def test_mask_class_ids_valid(self, train_dir):
        """All class IDs in returned mask must be in {0,1,2,3}."""
        ds = LaneSegDataset(train_dir, split="val", height=36, width=64, augment=False)
        for i in range(len(ds)):
            _, mask = ds[i]
            unique = set(mask.unique().tolist())
            illegal = unique - {0, 1, 2, 3}
            assert not illegal, f"Illegal class IDs at idx {i}: {illegal}"

    def test_returns_tensor_types(self, train_dir):
        ds = LaneSegDataset(train_dir, split="val", augment=False)
        img, mask = ds[0]
        assert isinstance(img,  torch.Tensor)
        assert isinstance(mask, torch.Tensor)


class TestLaneSegDatasetCorruptHandling:

    def test_corrupt_mask_skipped(self, tmp_path):
        """A pair with an illegal class ID should be skipped, not crash."""
        root = tmp_path / "corrupt_split"
        img_dir  = root / "images"
        mask_dir = root / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        # 1 corrupt pair + 2 valid pairs
        _write_image(img_dir / "bad.jpg")
        _write_corrupt_mask(mask_dir / "bad.png")

        for i in range(2):
            _write_image(img_dir / f"good_{i}.jpg")
            _write_multiclass_mask(mask_dir / f"good_{i}.png")

        ds = LaneSegDataset(root, split="train", augment=False)
        # Should still load without raising — falls back to valid pair
        img, mask = ds[0]
        assert img.shape[0] == 3

    def test_corrupt_mask_logged_in_skipped_pairs(self, tmp_path):
        """Skipped corrupt pairs should be accessible via .skipped_pairs."""
        root = tmp_path / "log_corrupt"
        img_dir  = root / "images"
        mask_dir = root / "masks"
        img_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)

        _write_image(img_dir / "bad.jpg")
        _write_corrupt_mask(mask_dir / "bad.png")
        _write_image(img_dir / "good.jpg")
        _write_multiclass_mask(mask_dir / "good.png")

        ds = LaneSegDataset(root, split="train", augment=False)
        # Trigger loading to populate skipped list
        ds[0]

        # The bad.jpg must appear in skipped_pairs
        skipped_names = [p.name for p in ds.skipped_pairs]
        assert "bad.jpg" in skipped_names


class TestLaneSegDatasetAugmentation:

    def test_val_split_does_not_augment_by_default(self, val_dir):
        """Val split must have augmentation off by default."""
        ds = LaneSegDataset(val_dir, split="val")
        assert not ds._use_augment

    def test_train_split_augments_by_default(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train")
        assert ds._use_augment

    def test_augment_override(self, train_dir):
        """augment=False must override default for train split."""
        ds = LaneSegDataset(train_dir, split="train", augment=False)
        assert not ds._use_augment


class TestClassPixelCounts:

    def test_counts_sum_to_total_pixels(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", augment=False)
        counts = ds.class_pixel_counts()
        total  = sum(counts.values())
        # Each of the 6 synthetic images has a 36×64 = 2304 pixel mask
        expected = 6 * 36 * 64
        assert total == expected, f"Expected {expected} total pixels, got {total}"

    def test_all_classes_present(self, train_dir):
        ds = LaneSegDataset(train_dir, split="train", augment=False)
        counts = ds.class_pixel_counts()
        for c in range(NUM_CLASSES):
            assert counts[c] > 0, f"Class {c} has zero pixels — multi-class mask broken"


class TestLaneSegDatasetMetadata:

    def test_metadata_returned_when_flag_enabled(self, train_dir):
        ds = LaneSegDataset(train_dir, split="val", return_metadata=True)
        item = ds[0]
        assert len(item) == 3
        img, mask, meta = item
        assert isinstance(img, torch.Tensor)
        assert isinstance(mask, torch.Tensor)
        assert isinstance(meta, dict)
        assert "image_path" in meta
        assert "mask_path" in meta
        assert "session_id" in meta
        assert "height" in meta
        assert "width" in meta
        assert isinstance(meta["session_id"], str) and len(meta["session_id"]) > 0

    def test_default_does_not_return_metadata(self, train_dir):
        ds = LaneSegDataset(train_dir, split="val")
        item = ds[0]
        assert len(item) == 2


class TestLaneSegDatasetAugmentationSafety:

    def test_augmentation_preserves_mask_validity(self, train_dir):
        """Augmentations must NEVER create values outside {0, 1, 2, 3}."""
        ds = LaneSegDataset(train_dir, split="train", augment=True)
        for _ in range(10):
            img, mask = ds[0]
            unique_vals = set(mask.unique().tolist())
            assert unique_vals.issubset({0, 1, 2, 3}), f"Illegal mask values generated: {unique_vals}"

    def test_forbidden_augmentations_not_in_pipeline(self):
        """HorizontalFlip, ElasticTransform, GridDistortion, CoarseDropout MUST NOT be present."""
        from src.training.dataset import build_safe_train_augmentations
        pipeline = build_safe_train_augmentations(height=72, width=128)
        
        transform_types = [t.__class__.__name__ for t in pipeline.transforms]
        
        forbidden = [
            "HorizontalFlip",
            "RandomHorizontalFlip",
            "Fliplr",
            "ElasticTransform",
            "GridDistortion",
            "OpticalDistortion",
            "CoarseDropout",
            "Cutout",
            "RandomRotate90",
        ]
        for f in forbidden:
            assert f not in transform_types, f"Forbidden augmentation '{f}' found in pipeline!"

    def test_rotation_angle_strictly_bounded(self):
        """Rotation must not exceed 5 degrees to protect steering geometry."""
        from src.training.dataset import build_safe_train_augmentations
        pipeline = build_safe_train_augmentations(height=72, width=128)
        for t in pipeline.transforms:
            if "Rotate" in t.__class__.__name__:
                limit = getattr(t, "limit", None)
                if limit is not None:
                    max_angle = max(abs(limit[0]), abs(limit[1])) if isinstance(limit, (tuple, list)) else abs(limit)
                    assert max_angle <= 5, f"Rotation limit {max_angle}° exceeds safe 5° threshold"


class TestWeightedAndSequenceSamplers:

    def test_compute_lane_pixel_fractions(self, tmp_path):
        from src.training.samplers import compute_lane_pixel_fractions
        mask_dir = tmp_path / "masks"
        mask_dir.mkdir(parents=True)

        # Mask 1: only background (0 lane pixels)
        p1 = mask_dir / "m1.png"
        cv2.imwrite(str(p1), np.zeros((32, 32), dtype=np.uint8))

        # Mask 2: 50% left lane (class 2)
        p2 = mask_dir / "m2.png"
        m2 = np.zeros((32, 32), dtype=np.uint8)
        m2[:16, :] = 2
        cv2.imwrite(str(p2), m2)

        fractions = compute_lane_pixel_fractions([p1, p2])
        assert len(fractions) == 2
        assert fractions[0] == 0.0
        assert pytest.approx(fractions[1], rel=1e-3) == 0.5

    def test_compute_class_balanced_weights(self):
        from src.training.samplers import compute_class_balanced_weights
        fractions = np.array([0.0, 0.2, 0.4], dtype=np.float32)
        weights = compute_class_balanced_weights(fractions)
        assert len(weights) == 3
        # Weight formula: 1.0 + 1.5 * (lane_fraction / mean_lane_fraction)
        # Higher lane fraction => strictly higher weight
        assert weights[2] > weights[1] > weights[0]
        assert weights[0] == 1.0

    def test_class_balanced_sampler_samples_correctly(self):
        from src.training.samplers import ClassBalancedSampler
        dummy_dataset = list(range(10))
        fractions = np.array([0.0] * 5 + [0.3] * 5, dtype=np.float32)
        sampler = ClassBalancedSampler(dummy_dataset, lane_fractions=fractions, num_samples=50)
        samples = list(sampler)
        assert len(samples) == 50
        # High lane fraction indices (5..9) should be sampled more often than (0..4)
        high_count = sum(1 for idx in samples if idx >= 5)
        low_count = sum(1 for idx in samples if idx < 5)
        assert high_count > low_count

    def test_weighted_session_sampler_diversity(self, tmp_path):
        from src.training.samplers import WeightedSessionSampler
        # Create dataset items belonging to 4 distinct sessions
        session_ids = [
            "session_A", "session_A", "session_A",
            "session_B", "session_B", "session_B",
            "session_C", "session_C", "session_C",
            "session_D", "session_D", "session_D",
        ]
        dummy_dataset = list(range(len(session_ids)))
        batch_size = 4
        sampler = WeightedSessionSampler(
            dataset=dummy_dataset,
            batch_size=batch_size,
            session_ids=session_ids,
        )
        batches = list(sampler)
        assert len(batches) >= 3
        for b in batches:
            if len(b) == batch_size:
                batch_sessions = [session_ids[i] for i in b]
                # Each batch should have high diversity (distinct sessions)
                assert len(set(batch_sessions)) == len(batch_sessions), (
                    f"Batch has duplicate sessions: {batch_sessions}"
                )

    def test_weighted_session_sampler_with_dataloader(self, train_dir):
        from torch.utils.data import DataLoader
        from src.training.samplers import WeightedSessionSampler

        ds = LaneSegDataset(train_dir, split="train", height=36, width=64, augment=False)
        sampler = WeightedSessionSampler(ds, batch_size=2)
        loader = DataLoader(ds, batch_sampler=sampler)

        batch_count = 0
        total_items = 0
        for imgs, masks in loader:
            batch_count += 1
            total_items += imgs.shape[0]
            assert imgs.shape[1] == 3
            assert masks.shape[1:] == (36, 64)
        assert batch_count > 0
        assert total_items == len(ds)

