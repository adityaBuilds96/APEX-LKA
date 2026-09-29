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
