"""
src/training/dataset.py
========================
LaneSegDataset — PyTorch Dataset for 4-class lane segmentation.

Class map
---------
0  background
1  road
2  left_lane
3  right_lane

Directory layout (expected)
---------------------------
data/
  train/
    images/   *.jpg | *.png
    masks/    *.png  (same stem, uint8, values in {0,1,2,3})
  val/
    images/
    masks/
  test/
    images/
    masks/

Augmentation policy
-------------------
✅  Allowed  : Brightness, Contrast, Gamma, HueSaturation,
               GaussianBlur, GaussNoise, CLAHE,
               Shadow simulation (synthetic occlusion).
❌  FORBIDDEN: HorizontalFlip  — would mirror left/right lanes incorrectly.
❌  FORBIDDEN: VerticalFlip    — road geometry is up/down directed.
❌  FORBIDDEN: RandomRotate >5° — breaks lane angle assumptions.
❌  FORBIDDEN: Perspective/Elastic — distorts lane curvature cues.

Corrupt mask handling
---------------------
If a mask fails to load or contains out-of-range class IDs,
the pair is skipped silently and the next valid pair is loaded.
The dataset logs all skipped pairs so QC can flag them later.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

# ── Optional albumentations import (graceful degradation) ─────────────────────
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    _ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    _ALBUMENTATIONS_AVAILABLE = False
    log.warning(
        "albumentations not installed — augmentation disabled. "
        "Run: pip install albumentations"
    )


# ── Constants ──────────────────────────────────────────────────────────────────
NUM_CLASSES: int = 4
VALID_CLASS_IDS: frozenset = frozenset(range(NUM_CLASSES))   # {0, 1, 2, 3}
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp")
MASK_EXTENSION:   str             = ".png"

# ImageNet stats (RGB) — must match src/config.py preprocessing section
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation pipelines
# ═══════════════════════════════════════════════════════════════════════════════

def _build_train_transforms(
    height: int,
    width: int,
    brightness_limit: float = 0.3,
    contrast_limit:   float = 0.3,
    blur_limit:       int   = 3,
    shadow_simulation: bool = True,
) -> Optional[object]:
    """
    Build the training augmentation pipeline.

    ONLY photometric transforms are applied.
    Geometric transforms that could break lane geometry are excluded.

    Returns None if albumentations is not available (caller must handle).
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        return None

    transforms: List[A.BasicTransform] = [
        A.Resize(height, width),
        # ── Photometric ───────────────────────────────────────────────────────
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.6,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=25,
            val_shift_limit=15,
            p=0.4,
        ),
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        A.CLAHE(clip_limit=3.0, tile_grid_size=(4, 4), p=0.2),
        # ── Blur / Noise ──────────────────────────────────────────────────────
        A.GaussianBlur(blur_limit=(3, blur_limit), p=0.2),
        A.GaussNoise(p=0.2),
    ]

    # ── Shadow simulation (synthetic occlusion) ───────────────────────────────
    # Shadow is safe — it does not alter spatial lane geometry.
    if shadow_simulation:
        transforms.append(
            A.RandomShadow(
                shadow_roi=(0, 0.5, 1, 1),      # lower half only
                num_shadows_limit=(1, 1),        # exactly one shadow
                shadow_dimension=4,
                p=0.3,
            )
        )

    # ── Mild geometric (≤5° rotation, NO flip, NO perspective) ───────────────
    transforms.append(
        A.Affine(
            translate_percent={"x": (-0.03, 0.03), "y": (-0.02, 0.02)},
            scale=(0.95, 1.05),
            rotate=(-4, 4),           # Max ±4° — safe for lane geometry
            shear=0,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.4,
        )
    )

    # ── Normalise + to tensor ─────────────────────────────────────────────────
    transforms += [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ]

    return A.Compose(transforms)


def _build_val_transforms(height: int, width: int) -> Optional[object]:
    """Validation / test: only resize + normalise — NO augmentation."""
    if not _ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


# ── Manual (no-albumentations) transform fallback ─────────────────────────────

def _manual_transform(
    image: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Minimal resize + normalise without albumentations.
    Returns (image_tensor, mask_tensor).
    """
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask,  (width, height), interpolation=cv2.INTER_NEAREST)

    # Normalise
    img_f = image.astype(np.float32) / 255.0
    mean  = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std   = np.array(_IMAGENET_STD,  dtype=np.float32)
    img_f = (img_f - mean) / std

    img_t  = torch.from_numpy(img_f.transpose(2, 0, 1))           # (3, H, W)
    mask_t = torch.from_numpy(mask.astype(np.int64))               # (H, W)
    return img_t, mask_t


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class LaneSegDataset(Dataset):
    """
    PyTorch Dataset for APEX LKA 4-class lane segmentation.

    Parameters
    ----------
    root_dir   : Path to split directory, e.g. ``data/train``.
                 Must contain ``images/`` and ``masks/`` sub-dirs.
    split      : ``"train"`` | ``"val"`` | ``"test"`` — controls augmentation.
    height     : Target image height (default 360, from config).
    width      : Target image width  (default 640, from config).
    augment    : Override to force augmentation on/off.
                 Defaults to True for train, False for val/test.

    Notes
    -----
    - Masks are expected as single-channel PNG uint8 with values in {0,1,2,3}.
    - Corrupted or out-of-range masks are skipped, not crashed on.
    - Skipped pairs are logged at WARNING level.
    """

    def __init__(
        self,
        root_dir: str | Path,
        split:    str = "train",
        height:   int = 360,
        width:    int = 640,
        augment:  Optional[bool] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split    = split
        self.height   = height
        self.width    = width

        self._use_augment = augment if augment is not None else (split == "train")

        self._img_dir  = self.root_dir / "images"
        self._mask_dir = self.root_dir / "masks"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self._img_dir}")
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self._mask_dir}")

        # Build the matched (image, mask) pair list
        self._pairs: List[Tuple[Path, Path]] = self._discover_pairs()

        # Build transforms
        if self._use_augment:
            self._transform = _build_train_transforms(height, width)
        else:
            self._transform = _build_val_transforms(height, width)

        # Track any corrupted pairs found during __getitem__
        self._skipped: List[Path] = []

        log.info(
            "LaneSegDataset [%s] — %d pairs | augment=%s | albumentations=%s",
            split, len(self._pairs), self._use_augment, _ALBUMENTATIONS_AVAILABLE,
        )

    # ── Pair discovery ─────────────────────────────────────────────────────────

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """
        Find all (image, mask) pairs where the mask file exists.

        Supports image files with any extension in IMAGE_EXTENSIONS.
        Mask must be a ``.png`` with the same stem.
        """
        pairs: List[Tuple[Path, Path]] = []
        for img_path in sorted(self._img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self._mask_dir / (img_path.stem + MASK_EXTENSION)
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        image : (3, H, W)  float32  ImageNet-normalised
        mask  : (H, W)     int64    class indices in {0,1,2,3}
        """
        # Loop through pairs starting at idx until a valid one is found.
        n = len(self._pairs)
        for attempt in range(n):
            real_idx = (idx + attempt) % n
            img_path, mask_path = self._pairs[real_idx]

            try:
                image, mask = self._load_and_validate(img_path, mask_path)
            except _SkipPair as exc:
                log.warning("Skipping pair %s: %s", img_path.name, exc)
                self._skipped.append(img_path)
                continue

            # Apply transforms
            if self._transform is not None and _ALBUMENTATIONS_AVAILABLE:
                transformed = self._transform(image=image, mask=mask)
                img_tensor  = transformed["image"].float()          # (3, H, W)
                mask_tensor = transformed["mask"].long()            # (H, W)
            else:
                img_tensor, mask_tensor = _manual_transform(
                    image, mask, self.height, self.width
                )
            return img_tensor, mask_tensor

        raise RuntimeError(
            f"LaneSegDataset [{self.split}]: all {n} pairs are corrupted."
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_and_validate(
        self, img_path: Path, mask_path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load image and mask; raise _SkipPair on any corruption.

        Returns
        -------
        image : (H, W, 3)  uint8  RGB
        mask  : (H, W)     uint8  class IDs
        """
        # ── Load image ────────────────────────────────────────────────────────
        raw = cv2.imdecode(
            np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if raw is None:
            raise _SkipPair("Image failed to decode")
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        # ── Load mask ─────────────────────────────────────────────────────────
        raw_mask = cv2.imdecode(
            np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if raw_mask is None:
            raise _SkipPair("Mask failed to decode")

        mask = raw_mask.astype(np.uint8)

        # ── Validate class IDs ────────────────────────────────────────────────
        unique_ids = set(np.unique(mask).tolist())
        illegal = unique_ids - VALID_CLASS_IDS
        if illegal:
            raise _SkipPair(f"Illegal class IDs in mask: {illegal}")

        return image, mask

    # ── Introspection helpers ─────────────────────────────────────────────────

    @property
    def skipped_pairs(self) -> List[Path]:
        """Paths of image files that were skipped due to mask corruption."""
        return list(self._skipped)

    def class_pixel_counts(self) -> Dict[int, int]:
        """
        Count total pixels per class across the entire dataset.

        This is intentionally O(N × H × W) — call offline, not during training.

        Returns
        -------
        {class_id: pixel_count}
        """
        counts: Dict[int, int] = {c: 0 for c in range(NUM_CLASSES)}
        for _, mask_path in self._pairs:
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue
            for cid in range(NUM_CLASSES):
                counts[cid] += int((raw_mask == cid).sum())
        return counts


# ── Internal sentinel exception ───────────────────────────────────────────────

class _SkipPair(Exception):
    """Raised internally when a (image, mask) pair should be skipped."""
