"""
src/data/label_schema.py
========================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Universal 4-Class Semantic Label Schema & Validation Engine.
ISO 26262 ASIL-B Data Quality & Perception Baseline Aligned.

Class Map Definition:
--------------------
  Class 0 : Background (Non-drivable environment, sky, hood, vegetation, barriers)
  Class 1 : Drivable Corridor (Drivable asphalt bounded between ego lane boundaries)
  Class 2 : Left Lane Boundary (Immediate ego left boundary line marking)
  Class 3 : Right Lane Boundary (Immediate ego right boundary line marking)

Mathematical & Safety Principles:
---------------------------------
1. Exclusivity: Each pixel in a standardized semantic mask must map strictly to {0, 1, 2, 3}.
2. Geometry Integrity: Left (2) and Right (3) lines define the boundaries of Class 1 (Corridor).
3. No-Flip Inversion Defense: Left and Right line classes MUST NOT be horizontally mirrored
   without explicit class-index swapping to preserve vehicle steering directionality.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import IntEnum, unique
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# 1. Class Enumeration & Core Constants
# ──────────────────────────────────────────────────────────────────────────────

@unique
class LaneClassID(IntEnum):
    """
    Standardized integer class identifiers for APEX-Vision 4-class segmentation.
    """
    BACKGROUND = 0
    DRIVABLE_CORRIDOR = 1
    LEFT_BOUNDARY = 2
    RIGHT_BOUNDARY = 3

    @classmethod
    def all_ids(cls) -> Set[int]:
        """Return set of valid class IDs."""
        return {item.value for item in cls}

    @classmethod
    def is_valid(cls, value: int) -> bool:
        """Check if integer is a valid class ID."""
        return value in cls.all_ids()


NUM_CLASSES: int = 4
VALID_CLASS_IDS: frozenset[int] = frozenset(LaneClassID.all_ids())

# Default color palettes
COLOR_RGB_MAP: Dict[int, Tuple[int, int, int]] = {
    LaneClassID.BACKGROUND.value: (0, 0, 0),
    LaneClassID.DRIVABLE_CORRIDOR.value: (128, 64, 128),
    LaneClassID.LEFT_BOUNDARY.value: (0, 255, 0),
    LaneClassID.RIGHT_BOUNDARY.value: (255, 100, 0),
}

COLOR_BGR_MAP: Dict[int, Tuple[int, int, int]] = {
    LaneClassID.BACKGROUND.value: (0, 0, 0),
    LaneClassID.DRIVABLE_CORRIDOR.value: (128, 64, 128),
    LaneClassID.LEFT_BOUNDARY.value: (0, 255, 0),
    LaneClassID.RIGHT_BOUNDARY.value: (0, 100, 255),
}

HEX_COLOR_MAP: Dict[int, str] = {
    LaneClassID.BACKGROUND.value: "#000000",
    LaneClassID.DRIVABLE_CORRIDOR.value: "#804080",
    LaneClassID.LEFT_BOUNDARY.value: "#00FF00",
    LaneClassID.RIGHT_BOUNDARY.value: "#FF6400",
}

# Aliases for color palettes
COLOR_PALETTE_BGR = COLOR_BGR_MAP
COLOR_PALETTE_RGB = COLOR_RGB_MAP


# ──────────────────────────────────────────────────────────────────────────────
# 2. Dataclasses: Semantic Class Metadata & Statistics
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassInfo:
    """
    Complete metadata definition for a single semantic class.
    """
    id: int
    name: str
    description: str
    color_rgb: Tuple[int, int, int]
    color_bgr: Tuple[int, int, int]
    hex_color: str
    is_drivable: bool
    is_lane_line: bool
    weight: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize class definition to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "color_rgb": list(self.color_rgb),
            "color_bgr": list(self.color_bgr),
            "hex_color": self.hex_color,
            "is_drivable": self.is_drivable,
            "is_lane_line": self.is_lane_line,
            "weight": self.weight,
        }


@dataclass
class MaskClassStatistics:
    """
    Audit metrics summarizing pixel distribution across the 4 semantic classes.
    """
    total_pixels: int
    background_pixels: int
    corridor_pixels: int
    left_boundary_pixels: int
    right_boundary_pixels: int
    unknown_pixels: int

    background_pct: float
    corridor_pct: float
    left_boundary_pct: float
    right_boundary_pct: float
    unknown_pct: float

    has_left_boundary: bool
    has_right_boundary: bool
    has_corridor: bool
    is_valid_schema: bool
    validation_errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to serializable dictionary."""
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Master Label Schema Engine
# ──────────────────────────────────────────────────────────────────────────────

class LabelSchema:
    """
    Production-grade schema registry, validator, and colorizer for APEX-Vision masks.
    Enforces strict ISO 26262 ASIL-B data integrity gates across all ingest stages.
    """

    def __init__(self, custom_weights: Optional[Dict[int, float]] = None) -> None:
        """
        Initialize the label schema registry with default class specifications.
        """
        weights = custom_weights or {
            LaneClassID.BACKGROUND.value: 0.15,
            LaneClassID.DRIVABLE_CORRIDOR.value: 1.0,
            LaneClassID.LEFT_BOUNDARY.value: 4.5,
            LaneClassID.RIGHT_BOUNDARY.value: 4.5,
        }

        self._classes: Dict[int, ClassInfo] = {
            LaneClassID.BACKGROUND.value: ClassInfo(
                id=LaneClassID.BACKGROUND.value,
                name="background",
                description="Non-drivable regions, environment, horizon, sky, hoods, off-road terrain",
                color_rgb=COLOR_RGB_MAP[LaneClassID.BACKGROUND.value],
                color_bgr=COLOR_BGR_MAP[LaneClassID.BACKGROUND.value],
                hex_color=HEX_COLOR_MAP[LaneClassID.BACKGROUND.value],
                is_drivable=False,
                is_lane_line=False,
                weight=weights.get(LaneClassID.BACKGROUND.value, 0.15),
            ),
            LaneClassID.DRIVABLE_CORRIDOR.value: ClassInfo(
                id=LaneClassID.DRIVABLE_CORRIDOR.value,
                name="drivable_corridor",
                description="Drivable travel corridor bounded between ego left and ego right lanes",
                color_rgb=COLOR_RGB_MAP[LaneClassID.DRIVABLE_CORRIDOR.value],
                color_bgr=COLOR_BGR_MAP[LaneClassID.DRIVABLE_CORRIDOR.value],
                hex_color=HEX_COLOR_MAP[LaneClassID.DRIVABLE_CORRIDOR.value],
                is_drivable=True,
                is_lane_line=False,
                weight=weights.get(LaneClassID.DRIVABLE_CORRIDOR.value, 1.0),
            ),
            LaneClassID.LEFT_BOUNDARY.value: ClassInfo(
                id=LaneClassID.LEFT_BOUNDARY.value,
                name="left_boundary",
                description="Immediate ego left lane boundary marking (continuous or dashed)",
                color_rgb=COLOR_RGB_MAP[LaneClassID.LEFT_BOUNDARY.value],
                color_bgr=COLOR_BGR_MAP[LaneClassID.LEFT_BOUNDARY.value],
                hex_color=HEX_COLOR_MAP[LaneClassID.LEFT_BOUNDARY.value],
                is_drivable=False,
                is_lane_line=True,
                weight=weights.get(LaneClassID.LEFT_BOUNDARY.value, 4.5),
            ),
            LaneClassID.RIGHT_BOUNDARY.value: ClassInfo(
                id=LaneClassID.RIGHT_BOUNDARY.value,
                name="right_boundary",
                description="Immediate ego right lane boundary marking (continuous or dashed)",
                color_rgb=COLOR_RGB_MAP[LaneClassID.RIGHT_BOUNDARY.value],
                color_bgr=COLOR_BGR_MAP[LaneClassID.RIGHT_BOUNDARY.value],
                hex_color=HEX_COLOR_MAP[LaneClassID.RIGHT_BOUNDARY.value],
                is_drivable=False,
                is_lane_line=True,
                weight=weights.get(LaneClassID.RIGHT_BOUNDARY.value, 4.5),
            ),
        }

        # Precompute lookup tables for fast colorization (uint8 [4, 3])
        self._lut_bgr = np.zeros((256, 3), dtype=np.uint8)
        self._lut_rgb = np.zeros((256, 3), dtype=np.uint8)
        for class_id, info in self._classes.items():
            self._lut_bgr[class_id] = np.array(info.color_bgr, dtype=np.uint8)
            self._lut_rgb[class_id] = np.array(info.color_rgb, dtype=np.uint8)

    @property
    def num_classes(self) -> int:
        """Return the count of registered classes (4)."""
        return NUM_CLASSES

    @property
    def valid_class_ids(self) -> frozenset[int]:
        """Return the set of valid class IDs."""
        return VALID_CLASS_IDS

    def get_class_info(self, class_id: int) -> ClassInfo:
        """
        Retrieve ClassInfo for a specified class index.
        Raises ValueError if index is out of bounds.
        """
        if class_id not in self._classes:
            raise ValueError(f"Class ID {class_id} is outside valid 4-class schema {VALID_CLASS_IDS}")
        return self._classes[class_id]

    def get_class_name(self, class_id: int) -> str:
        """Return human-readable class name."""
        return self.get_class_info(class_id).name

    def get_color_rgb(self, class_id: int) -> Tuple[int, int, int]:
        """Return RGB tuple for class."""
        return self.get_class_info(class_id).color_rgb

    def get_color_bgr(self, class_id: int) -> Tuple[int, int, int]:
        """Return BGR tuple for class."""
        return self.get_class_info(class_id).color_bgr

    def get_class_weights_tensor(self) -> np.ndarray:
        """
        Return class loss weighting array for imbalanced cross-entropy training.
        """
        weights = [self._classes[i].weight for i in range(NUM_CLASSES)]
        return np.array(weights, dtype=np.float32)

    # ──────────────────────────────────────────────────────────────────────────
    # Validation & Quality Integrity Checks
    # ──────────────────────────────────────────────────────────────────────────

    def validate_mask_array(
        self,
        mask: np.ndarray,
        expected_shape: Optional[Tuple[int, int]] = (360, 640),
    ) -> Tuple[bool, List[str]]:
        """
        Verify that a candidate mask array satisfies all schema requirements:
        - Must be 2D uint8 or int32 array.
        - Must match expected height and width (if specified).
        - Pixel values must strictly belong to {0, 1, 2, 3}.
        """
        errors: List[str] = []

        if not isinstance(mask, np.ndarray):
            return False, ["Mask is not a numpy ndarray."]

        if mask.ndim != 2:
            errors.append(f"Expected 2D mask, but got {mask.ndim}D array with shape {mask.shape}.")

        if expected_shape is not None and mask.ndim == 2:
            if mask.shape != expected_shape:
                errors.append(f"Mask shape {mask.shape} does not match expected {expected_shape}.")

        # Check unique values
        unique_vals = np.unique(mask)
        illegal_vals = [val for val in unique_vals if val not in VALID_CLASS_IDS]
        if illegal_vals:
            errors.append(f"Mask contains illegal class indices: {illegal_vals}. Allowed: {list(VALID_CLASS_IDS)}.")

        return len(errors) == 0, errors

    def audit_mask_statistics(
        self,
        mask: np.ndarray,
        expected_shape: Optional[Tuple[int, int]] = (360, 640),
    ) -> MaskClassStatistics:
        """
        Calculate pixel distribution and safety metrics for a given semantic mask.
        """
        is_valid, errors = self.validate_mask_array(mask, expected_shape)

        total_pixels = int(mask.size)
        if total_pixels == 0:
            return MaskClassStatistics(
                total_pixels=0,
                background_pixels=0,
                corridor_pixels=0,
                left_boundary_pixels=0,
                right_boundary_pixels=0,
                unknown_pixels=0,
                background_pct=0.0,
                corridor_pct=0.0,
                left_boundary_pct=0.0,
                right_boundary_pct=0.0,
                unknown_pct=0.0,
                has_left_boundary=False,
                has_right_boundary=False,
                has_corridor=False,
                is_valid_schema=False,
                validation_errors=["Mask is empty."],
            )

        bg_count = int(np.count_nonzero(mask == LaneClassID.BACKGROUND.value))
        corr_count = int(np.count_nonzero(mask == LaneClassID.DRIVABLE_CORRIDOR.value))
        left_count = int(np.count_nonzero(mask == LaneClassID.LEFT_BOUNDARY.value))
        right_count = int(np.count_nonzero(mask == LaneClassID.RIGHT_BOUNDARY.value))

        valid_sum = bg_count + corr_count + left_count + right_count
        unknown_count = total_pixels - valid_sum

        return MaskClassStatistics(
            total_pixels=total_pixels,
            background_pixels=bg_count,
            corridor_pixels=corr_count,
            left_boundary_pixels=left_count,
            right_boundary_pixels=right_count,
            unknown_pixels=unknown_count,
            background_pct=float(bg_count / total_pixels * 100.0),
            corridor_pct=float(corr_count / total_pixels * 100.0),
            left_boundary_pct=float(left_count / total_pixels * 100.0),
            right_boundary_pct=float(right_count / total_pixels * 100.0),
            unknown_pct=float(unknown_count / total_pixels * 100.0),
            has_left_boundary=(left_count > 0),
            has_right_boundary=(right_count > 0),
            has_corridor=(corr_count > 0),
            is_valid_schema=(len(errors) == 0 and unknown_count == 0),
            validation_errors=errors,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Colorization & Overlay Visualization
    # ──────────────────────────────────────────────────────────────────────────

    def colorize_mask(self, mask: np.ndarray, color_mode: str = "bgr") -> np.ndarray:
        """
        Convert single-channel indexed mask into high-contrast RGB/BGR visualization image.
        Uses vectorized lookup table mapping for microsecond latency.
        """
        if mask.ndim != 2:
            raise ValueError(f"colorize_mask requires 2D array, got shape {mask.shape}")

        lut = self._lut_bgr if color_mode.lower() == "bgr" else self._lut_rgb
        # Clip values to 0-255 safely
        clamped = np.clip(mask, 0, 255).astype(np.uint8)
        color_image = lut[clamped]
        return color_image

    def create_alpha_overlay(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        alpha: float = 0.50,
        highlight_lanes_only: bool = False,
    ) -> np.ndarray:
        """
        Blend semantic colorized mask onto original RGB/BGR camera frame.
        """
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f"Shape mismatch between image {image.shape[:2]} and mask {mask.shape[:2]}")

        color_mask = self.colorize_mask(mask, color_mode="bgr")

        if highlight_lanes_only:
            # Only blend where class > 0 (corridor and boundaries)
            foreground = (mask > 0).astype(np.uint8)[:, :, np.newaxis]
            blended = np.where(
                foreground == 1,
                cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0),
                image,
            )
        else:
            blended = cv2.addWeighted(image, 1.0 - alpha, color_mask, alpha, 0)

        return blended

    # ──────────────────────────────────────────────────────────────────────────
    # Class Transformation & Filtering Utilities
    # ──────────────────────────────────────────────────────────────────────────

    def extract_binary_lane_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract binary mask where Left Boundary (2) or Right Boundary (3) is 1, else 0.
        """
        binary = np.zeros_like(mask, dtype=np.uint8)
        binary[(mask == LaneClassID.LEFT_BOUNDARY.value) | (mask == LaneClassID.RIGHT_BOUNDARY.value)] = 1
        return binary

    def extract_corridor_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Extract binary mask of Drivable Corridor (1).
        """
        binary = np.zeros_like(mask, dtype=np.uint8)
        binary[mask == LaneClassID.DRIVABLE_CORRIDOR.value] = 1
        return binary

    def swap_left_right_boundaries(self, mask: np.ndarray) -> np.ndarray:
        """
        Invert Left (2) and Right (3) lane boundaries.
        MANDATORY transformation whenever a horizontal flip augmentation is applied.
        """
        swapped = mask.copy()
        left_mask = mask == LaneClassID.LEFT_BOUNDARY.value
        right_mask = mask == LaneClassID.RIGHT_BOUNDARY.value

        swapped[left_mask] = LaneClassID.RIGHT_BOUNDARY.value
        swapped[right_mask] = LaneClassID.LEFT_BOUNDARY.value
        return swapped

    # ──────────────────────────────────────────────────────────────────────────
    # Export & Serialization
    # ──────────────────────────────────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """Export entire schema specification to dictionary."""
        return {
            "version": "4.0.0",
            "num_classes": self.num_classes,
            "classes": [info.to_dict() for info in self._classes.values()],
            "valid_class_ids": sorted(list(self.valid_class_ids)),
        }

    def to_json(self, indent: int = 2) -> str:
        """Export schema to formatted JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save_schema_file(self, target_path: Union[str, Path]) -> None:
        """Save schema configuration to JSON file on disk."""
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.to_json())


# ──────────────────────────────────────────────────────────────────────────────
# 4. Global Singleton Schema Instance & Convenient Functions
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_SCHEMA: LabelSchema = LabelSchema()
LaneLabelSchema = LabelSchema


def get_default_label_schema() -> LabelSchema:
    """Return the global default APEX-Vision 4-class label schema."""
    return DEFAULT_SCHEMA


def validate_mask(mask: np.ndarray) -> bool:
    """Quick validation check: returns True if mask adheres to 4-class schema."""
    valid, _ = DEFAULT_SCHEMA.validate_mask_array(mask)
    return valid


def colorize_mask(mask: np.ndarray, color_mode: str = "bgr") -> np.ndarray:
    """Quick colorization of mask to 3-channel visualization image."""
    return DEFAULT_SCHEMA.colorize_mask(mask, color_mode=color_mode)


def swap_left_right_classes(mask: np.ndarray) -> np.ndarray:
    """Quick swap of left and right boundary classes."""
    return DEFAULT_SCHEMA.swap_left_right_boundaries(mask)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing LabelSchema ISO 26262 ASIL-B self-check...")
    schema = LabelSchema()
    assert schema.num_classes == 4

    # Test synthetic mask validation
    dummy_mask = np.zeros((360, 640), dtype=np.uint8)
    dummy_mask[180:360, 200:440] = LaneClassID.DRIVABLE_CORRIDOR.value
    dummy_mask[180:360, 195:200] = LaneClassID.LEFT_BOUNDARY.value
    dummy_mask[180:360, 440:445] = LaneClassID.RIGHT_BOUNDARY.value

    is_valid, errs = schema.validate_mask_array(dummy_mask)
    assert is_valid, f"Validation failed: {errs}"

    stats = schema.audit_mask_statistics(dummy_mask)
    assert stats.has_left_boundary and stats.has_right_boundary and stats.has_corridor
    assert stats.is_valid_schema

    # Test swap
    swapped = schema.swap_left_right_boundaries(dummy_mask)
    assert np.all(swapped[180:360, 195:200] == LaneClassID.RIGHT_BOUNDARY.value)
    assert np.all(swapped[180:360, 440:445] == LaneClassID.LEFT_BOUNDARY.value)

    # Test colorization
    colored = schema.colorize_mask(dummy_mask)
    assert colored.shape == (360, 640, 3)

    print("LabelSchema self-check passed successfully.")
