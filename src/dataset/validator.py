"""
src/dataset/validator.py
========================
Quality control, integrity validation, and leakage prevention for road datasets.

Performs:
  - Corrupted/unreadable image detection
  - Dimension & channel validation
  - Mask validity, dimension matching, and class ID verification
  - Exact duplicate detection (SHA-256)
  - Near-duplicate detection (perceptual difference hash)
  - Sequence-aware group identification (temporal leakage prevention)
  - Class pixel distribution & extreme imbalance audits
"""

import hashlib
import re
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np


class DatasetValidator:
    """
    Validates images, masks, and metadata integrity.
    """

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    SUPPORTED_MASK_EXTS = {".png", ".bmp", ".npy", ".tiff"}

    def __init__(
        self,
        min_width: int = 160,
        min_height: int = 90,
        allowed_classes: Optional[set[int]] = None,
        dhash_threshold: int = 2,
    ):
        self.min_width = min_width
        self.min_height = min_height
        self.allowed_classes = allowed_classes or {0, 1, 2, 3}
        self.dhash_threshold = dhash_threshold

    def validate_image(self, image_path: Path) -> Tuple[bool, Optional[Tuple[int, int, int]], Optional[str]]:
        """
        Validate an image file for decodability, dimensions, and channels.

        Returns:
            (is_valid, (height, width, channels), error_message)
        """
        if not image_path.exists():
            return False, None, f"File does not exist: {image_path}"

        if image_path.suffix.lower() not in self.SUPPORTED_IMG_EXTS:
            return False, None, f"Unsupported image extension: {image_path.suffix}"

        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False, None, "Corrupted or unreadable image (cv2.imread returned None)"
            if img.size == 0:
                return False, None, "Empty image buffer (0 bytes decoded)"

            h, w = img.shape[:2]
            c = img.shape[2] if len(img.shape) > 2 else 1

            if w < self.min_width or h < self.min_height:
                return False, (h, w, c), f"Resolution {w}x{h} below minimum threshold ({self.min_width}x{self.min_height})"

            if c != 3:
                return False, (h, w, c), f"Expected 3 channels (BGR/RGB), got {c} channel(s)"

            return True, (h, w, c), None

        except Exception as exc:
            return False, None, f"Exception reading image: {exc}"

    def validate_mask(
        self,
        mask_path: Path,
        expected_shape: Optional[Tuple[int, int]] = None,
        allowed_classes: Optional[set[int]] = None,
    ) -> Tuple[bool, dict[int, int], bool, Optional[str]]:
        """
        Validate a segmentation mask file.

        Returns:
            (is_valid, class_pixel_counts, is_empty, error_message)
        """
        if not mask_path.exists():
            return False, {}, True, f"Mask file does not exist: {mask_path}"

        allowed = allowed_classes or self.allowed_classes

        try:
            mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
            if mask is None:
                return False, {}, True, "Corrupted or unreadable mask file"

            # If 3-channel (color-indexed or RGB mask), convert to 1-channel 2D
            if len(mask.shape) == 3:
                # If all channels identical, take single channel
                if np.array_equal(mask[:, :, 0], mask[:, :, 1]) and np.array_equal(mask[:, :, 1], mask[:, :, 2]):
                    mask_2d = mask[:, :, 0]
                else:
                    # Treat as grayscale representation of class IDs
                    mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_2d = mask

            h, w = mask_2d.shape[:2]

            # Dimension consistency
            if expected_shape is not None:
                exp_h, exp_w = expected_shape[:2]
                if (h, w) != (exp_h, exp_w):
                    return False, {}, False, f"Mask dimension ({w}x{h}) does not match image ({exp_w}x{exp_h})"

            # Unique class IDs check
            unique_classes = set(np.unique(mask_2d).tolist())
            invalid_classes = unique_classes - allowed
            if invalid_classes:
                return (
                    False,
                    {},
                    False,
                    f"Mask contains invalid class IDs: {sorted(invalid_classes)}. Allowed: {sorted(allowed)}",
                )

            # Class counts
            counts = {}
            for cls_id in unique_classes:
                counts[int(cls_id)] = int(np.sum(mask_2d == cls_id))

            # Is empty (only background class 0 present)
            is_empty = (unique_classes == {0}) or (len(unique_classes) == 0)

            return True, counts, is_empty, None

        except Exception as exc:
            return False, {}, True, f"Exception validating mask: {exc}"

    @staticmethod
    def compute_sha256(path: Path) -> str:
        """Compute SHA-256 hash of file content."""
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_dhash(img: np.ndarray, hash_size: int = 8) -> int:
        """
        Compute difference hash (dHash) for fast perceptual duplicate matching.
        """
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        resized = cv2.resize(gray, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
        diff = resized[:, 1:] > resized[:, :-1]
        decimal_val = 0
        for bit in diff.flatten():
            decimal_val = (decimal_val << 1) | int(bit)
        return decimal_val

    @staticmethod
    def hamming_distance(h1: int, h2: int) -> int:
        """Compute Hamming distance between two integer hashes."""
        return bin(h1 ^ h2).count("1")

    @staticmethod
    def extract_sequence_group(path: Path) -> str:
        """
        Extract video / recording session group name from filename or parent directory.
        Used to prevent temporal data leakage across train/val/test splits.
        """
        stem = path.stem

        # Common naming conventions:
        # e.g., "drive01_frame000150" -> "drive01"
        # e.g., "seq_A_00123" -> "seq_A"
        # e.g., "cam0_16892348_0001" -> "cam0_16892348"
        if "_frame" in stem:
            return stem.rsplit("_frame", 1)[0]
        if "_step" in stem:
            return stem.rsplit("_step", 1)[0]

        # Regex match trailing frame numbering like _000123 or -000123
        match = re.match(r"^(.*?)[_-]\d{3,}$", stem)
        if match:
            return match.group(1)

        # Fallback to parent directory name if meaningful
        parent = path.parent.name
        if parent not in {"images", "raw_frames", "data", "annotated", "train", "val", "test"}:
            return parent

        return "default_sequence"
