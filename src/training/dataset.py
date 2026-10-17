"""
src/training/dataset.py
========================
LaneSegDataset — Universal PyTorch Dataset for 4-Class Semantic Segmentation.

Class Map:
  0 : Background
  1 : Road (Drivable Surface)
  2 : Left Lane Line
  3 : Right Lane Line

Augmentation Policy
-------------------
✅ SAFE AUGMENTATIONS (both image AND mask):
  - RandomResizedCrop (scale 0.8-1.0)
  - VerticalShift ±5%
  - Rotate ±3 degrees only
  - RandomBrightnessContrast (±30%, ±20%)
  - GaussNoise (var_limit 10-50)
  - GaussianBlur (blur_limit 3-5)
  - RandomShadow (num_shadows 1-3)
  - HueSaturationValue (hue ±10, sat ±20)

❌ EXPLICITLY FORBIDDEN AUGMENTATIONS:
  - HorizontalFlip:
      CATASTROPHIC for autonomous driving. Inverts left vs right lane classes (2 vs 3)
      and mirrors road curvature, destroying steering geometry.
  - Large Rotations (> 5°):
      Violates upright vehicle camera perspective and produces impossible horizon angles.
  - ElasticTransform & GridDistortion:
      Non-rigid warping distorts straight highway lanes into artificial S-curves.
  - CoarseDropout / Cutout:
      Deletes thin continuous lane markings, confusing the network into learning broken lines.
"""

from __future__ import annotations

import inspect
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.samplers import extract_session_id

log = logging.getLogger(__name__)

# ── Optional albumentations import ───────────────────────────────────────────
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
IMAGE_EXTENSIONS: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
MASK_EXTENSION:   str             = ".png"

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


# ═══════════════════════════════════════════════════════════════════════════════
# Augmentation Pipelines
# ═══════════════════════════════════════════════════════════════════════════════

# List of explicitly forbidden augmentations that destroy autonomous perception geometry
FORBIDDEN_TRANSFORMS: Tuple[str, ...] = (
    "HorizontalFlip",   # INVERTS left (2) vs right (3) lane classes and mirrors steering sign
    "VerticalFlip",     # Inverts ground-plane geometry and horizon perspective
    "ElasticTransform", # Non-rigid warping creates artificial non-physical road curvatures
    "GridDistortion",   # Distorts straight highway geometry into non-planar artifacts
)


def _build_train_transforms(
    height: int = 360,
    width: int = 640,
    brightness_limit: float = 0.20,
    contrast_limit: float = 0.20,
    blur_limit: int = 5,
    shadow_simulation: bool = True,
) -> Optional[object]:
    """
    Build albumentations pipeline strictly containing SAFE transforms for lane segmentation.
    Explicitly excludes HorizontalFlip, VerticalFlip, ElasticTransform, and GridDistortion.
    Horizontal flips invert left/right lane classes and mirror steering geometry sign conventions.
    """
    if not _ALBUMENTATIONS_AVAILABLE:
        return None

    # Allowed Geometric Augmentation 1: RandomResizedCrop (scale 0.85 - 1.0, p=0.5)
    try:
        crop_transform = A.RandomResizedCrop(
            size=(height, width),
            scale=(0.85, 1.0),
            ratio=(width / height * 0.95, width / height * 1.05),
            p=0.5,
        )
    except (TypeError, ValueError):
        crop_transform = A.RandomResizedCrop(
            height=height,
            width=width,
            scale=(0.85, 1.0),
            ratio=(width / height * 0.95, width / height * 1.05),
            p=0.5,
        )

    # Allowed Geometric Augmentation 2: ShiftScaleRotate (shift_limit=0.05, scale_limit=0.0, rotate_limit=3, p=0.3)
    try:
        shift_rotate = A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.0,
            rotate_limit=3,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        )
    except Exception:
        shift_rotate = A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            rotate=(-3, 3),
            scale=(1.0, 1.0),
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.3,
        )

    # Allowed Photometric Augmentations:
    # 1. RandomBrightnessContrast(p=0.5)
    # 2. GaussNoise(var_limit=(10, 50), p=0.3)
    # 3. GaussianBlur(blur_limit=(3, 5), p=0.3)
    # 4. RandomShadow(num_shadows_lower=1, num_shadows_upper=3, p=0.4)
    # 5. HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.4)
    transforms: List[Any] = [
        crop_transform,
        shift_rotate,
        A.Resize(height=height, width=width),
        A.RandomBrightnessContrast(
            brightness_limit=brightness_limit,
            contrast_limit=contrast_limit,
            p=0.5,
        ),
        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.4,
        ),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3)
        if "var_limit" in inspect.signature(A.GaussNoise.__init__).parameters
        else A.GaussNoise(std_range=(0.02, 0.1), p=0.3),
        A.GaussianBlur(blur_limit=(3, blur_limit), p=0.3),
    ]

    # Synthetic shadow simulation (p=0.4)
    if shadow_simulation:
        if "num_shadows_lower" in inspect.signature(A.RandomShadow.__init__).parameters:
            transforms.append(
                A.RandomShadow(
                    shadow_roi=(0, 0.4, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=3,
                    shadow_dimension=5,
                    p=0.4,
                )
            )
        elif "num_shadows_limit" in inspect.signature(A.RandomShadow.__init__).parameters:
            transforms.append(
                A.RandomShadow(
                    shadow_roi=(0, 0.4, 1, 1),
                    num_shadows_limit=(1, 3),
                    shadow_dimension=5,
                    p=0.4,
                )
            )
        else:
            transforms.append(A.RandomShadow(p=0.4))

    # Normalization + Tensor conversion
    transforms += [
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ]

    pipeline = A.Compose(transforms)

    # Runtime assertion: Guarantee no geometry-inverting transforms exist
    for t in transforms:
        t_name = t.__class__.__name__
        if t_name in FORBIDDEN_TRANSFORMS:
            raise ValueError(
                f"Forbidden augmentation '{t_name}' detected! Horizontal flips and non-rigid "
                "distortions invert left/right lane classes and corrupt steering geometry."
            )

    return pipeline


# Public alias for safe augmentation builder
build_safe_train_augmentations = _build_train_transforms




def _build_val_transforms(height: int, width: int) -> Optional[object]:
    """Validation/testing: pure deterministic resize and normalization."""
    if not _ALBUMENTATIONS_AVAILABLE:
        return None
    return A.Compose([
        A.Resize(height, width),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])


def _manual_transform(
    image: np.ndarray,
    mask: np.ndarray,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fallback manual transforms when albumentations is unavailable."""
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    img_f = image.astype(np.float32) / 255.0
    mean = np.array(_IMAGENET_MEAN, dtype=np.float32)
    std = np.array(_IMAGENET_STD, dtype=np.float32)
    img_f = (img_f - mean) / std

    img_t = torch.from_numpy(img_f.transpose(2, 0, 1))
    mask_t = torch.from_numpy(mask.astype(np.int64))
    return img_t, mask_t


# ═══════════════════════════════════════════════════════════════════════════════
# LaneSegDataset
# ═══════════════════════════════════════════════════════════════════════════════

class LaneSegDataset(Dataset):
    """
    Universal 4-class lane segmentation dataset loader.
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "train",
        height: int = 360,
        width: int = 640,
        augment: Optional[bool] = None,
        return_metadata: bool = False,
        transform: Optional[Any] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.split = split
        self.height = height
        self.width = width
        self.return_metadata = return_metadata

        self._use_augment = augment if augment is not None else (split == "train")

        self._img_dir = self.root_dir / "images"
        self._mask_dir = self.root_dir / "masks"

        if not self._img_dir.exists():
            raise FileNotFoundError(f"Images directory not found: {self._img_dir}")
        if not self._mask_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self._mask_dir}")

        self._pairs: List[Tuple[Path, Path]] = self._discover_pairs()

        if transform is not None:
            # Runtime validation: Strictly prohibit geometry-inverting transforms
            transforms_list = getattr(transform, "transforms", [transform])
            for t in transforms_list:
                t_name = t.__class__.__name__
                if t_name in FORBIDDEN_TRANSFORMS:
                    raise ValueError(
                        f"CRITICAL: Forbidden augmentation '{t_name}' detected! Horizontal flips "
                        "and non-rigid distortions invert steering geometry sign conventions."
                    )
            self._transform = transform
        elif self._use_augment:
            self._transform = _build_train_transforms(height, width)
        else:
            self._transform = _build_val_transforms(height, width)

        self._skipped: List[Path] = []

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """Find matching image-mask pairs."""
        pairs: List[Tuple[Path, Path]] = []
        for img_path in sorted(self._img_dir.iterdir()):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            mask_path = self._mask_dir / (img_path.stem + MASK_EXTENSION)
            if mask_path.exists():
                pairs.append((img_path, mask_path))
        return pairs

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(
        self,
        idx: int,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Returns
        -------
        If return_metadata is False:
            (image_tensor, mask_tensor)
        If return_metadata is True:
            (image_tensor, mask_tensor, metadata_dict)
        """
        n = len(self._pairs)
        for attempt in range(n):
            real_idx = (idx + attempt) % n
            img_path, mask_path = self._pairs[real_idx]

            try:
                image, mask = self._load_and_validate(img_path, mask_path)
            except _SkipPair as exc:
                log.warning("Skipping corrupted pair %s: %s", img_path.name, exc)
                self._skipped.append(img_path)
                continue

            orig_h, orig_w = image.shape[:2]

            # Apply transforms
            if self._transform is not None and _ALBUMENTATIONS_AVAILABLE:
                transformed = self._transform(image=image, mask=mask)
                img_tensor = transformed["image"].float()
                mask_tensor = transformed["mask"].long()
            else:
                img_tensor, mask_tensor = _manual_transform(
                    image, mask, self.height, self.width
                )

            if self.return_metadata:
                metadata: Dict[str, Any] = {
                    "image_path": str(img_path),
                    "mask_path": str(mask_path),
                    "stem": img_path.stem,
                    "session_id": extract_session_id(img_path),
                    "height": self.height,
                    "width": self.width,
                    "orig_h": orig_h,
                    "orig_w": orig_w,
                }
                return img_tensor, mask_tensor, metadata

            return img_tensor, mask_tensor

        raise RuntimeError(f"LaneSegDataset [{self.split}]: all {n} pairs corrupted.")

    def _load_and_validate(
        self,
        img_path: Path,
        mask_path: Path,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load image and mask, validating class range {0, 1, 2, 3}."""
        raw = cv2.imdecode(np.fromfile(str(img_path), dtype=np.uint8), cv2.IMREAD_COLOR)
        if raw is None:
            raise _SkipPair("Image failed to decode")
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

        raw_mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if raw_mask is None:
            raise _SkipPair("Mask failed to decode")
        mask = raw_mask.astype(np.uint8)

        # Validate class IDs
        unique_ids = set(np.unique(mask).tolist())
        illegal = unique_ids - VALID_CLASS_IDS
        if illegal:
            raise _SkipPair(f"Illegal class IDs in mask: {illegal}")

        return image, mask

    @property
    def skipped_pairs(self) -> List[Path]:
        """List of image files skipped due to corruption."""
        return list(self._skipped)

    def class_pixel_counts(self) -> Dict[int, int]:
        """Count total pixels per class across dataset."""
        counts = {c: 0 for c in range(NUM_CLASSES)}
        for _, mask_path in self._pairs:
            raw_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if raw_mask is None:
                continue
            for cid in range(NUM_CLASSES):
                counts[cid] += int((raw_mask == cid).sum())
        return counts


class _SkipPair(Exception):
    """Raised internally when an image-mask pair is corrupted."""
