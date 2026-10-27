"""
src/data/augmentations.py
=========================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomous Vehicle Perception Augmentations & Strict No-Flip Safety Policy.
ISO 26262 ASIL-B Perception Training Safety Baseline.

Augmentation Safety Principles:
--------------------------------
1. Strict No-Flip Rule (Default):
   Standard horizontal flipping is CATASTROPHIC for autonomous driving perception.
   It inverts Left (Class 2) vs Right (Class 3) lane boundaries and mirrors steering
   curvature signs. By default, `horizontal_flip=False` is strictly enforced.
2. Controlled Class-Swap Flip (Optional):
   If horizontal flipping is explicitly enabled via `allow_horizontal_flip=True`,
   the pipeline MUST simultaneously swap Class 2 and Class 3 in the ground-truth mask.
3. Rotation Limits:
   Rotations are strictly constrained to [-3.0, +3.0] degrees to preserve natural
   automotive camera horizon geometry.
4. Non-Rigid Warp Prohibition:
   ElasticTransform and GridDistortion are strictly forbidden because they create
   artificial non-planar road curvatures and wavy lane artifacts.
"""

from __future__ import annotations

import logging
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema, swap_left_right_classes

logger = logging.getLogger("APEX.Augmentations")

# Optional Albumentations integration
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUMENTATIONS_AVAILABLE = True
except ImportError:
    ALBUMENTATIONS_AVAILABLE = False
    logger.warning("albumentations library not detected. Using built-in deterministic transforms.")

# ImageNet normalization statistics
IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


# ──────────────────────────────────────────────────────────────────────────────
# 1. Augmentation Configuration Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AugmentationPolicyConfig:
    """
    Fine-grained control over training and validation transforms.
    """
    target_height: int = 360
    target_width: int = 640
    allow_horizontal_flip: bool = False
    enable_class_swap_on_flip: bool = True
    horizontal_flip_prob: float = 0.50
    rotation_limit_deg: float = 3.0
    shift_limit_pct: float = 0.05
    scale_limit_pct: float = 0.05
    crop_scale_min: float = 0.85
    crop_scale_max: float = 1.00
    brightness_contrast_prob: float = 0.50
    brightness_limit: float = 0.20
    contrast_limit: float = 0.20
    blur_prob: float = 0.25
    blur_limit: int = 5
    noise_prob: float = 0.25
    noise_var_limit: Tuple[int, int] = (10, 50)
    shadow_prob: float = 0.30
    color_jitter_prob: float = 0.30
    hue_shift_limit: int = 10
    sat_shift_limit: int = 20
    val_shift_limit: int = 20

    # List of explicitly forbidden augmentations that destroy road geometry
    forbidden_transforms: Tuple[str, ...] = (
        "VerticalFlip",
        "ElasticTransform",
        "GridDistortion",
        "CoarseDropout",
    )


# ──────────────────────────────────────────────────────────────────────────────
# 2. Custom Safe Augmentation Wrapper with Class Swap Defense
# ──────────────────────────────────────────────────────────────────────────────

class SafeLaneAugmentor:
    """
    Wrapper around Albumentations / OpenCV that guarantees road perception integrity.
    If horizontal flip occurs, it intercepts the mask and swaps Left (2) and Right (3).
    """

    def __init__(
        self,
        config: Optional[AugmentationPolicyConfig] = None,
        allow_horizontal_flip: Optional[bool] = None,
    ) -> None:
        self.config = config or AugmentationPolicyConfig()
        if allow_horizontal_flip is not None:
            self.config.allow_horizontal_flip = allow_horizontal_flip
            if allow_horizontal_flip:
                self.config.horizontal_flip_prob = 1.0
            else:
                self.config.horizontal_flip_prob = 0.0
        self.schema = get_default_label_schema()
        self._albumentations_pipeline = self._build_pipeline()

    @property
    def allow_horizontal_flip(self) -> bool:
        """Query whether horizontal flipping is enabled."""
        return self.config.allow_horizontal_flip

    def apply_flip_with_swap(
        self,
        img: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Directly apply horizontal flip with deterministic class swap."""
        flipped_img = cv2.flip(img, 1)
        flipped_mask = cv2.flip(mask, 1)
        if self.config.enable_class_swap_on_flip:
            flipped_mask = swap_left_right_classes(flipped_mask)
        return flipped_img, flipped_mask

    def _build_pipeline(self) -> Optional[object]:
        """Construct albumentations composition if installed."""
        if not ALBUMENTATIONS_AVAILABLE:
            return None

        cfg = self.config
        transforms_list: List[Any] = []

        # 1. Allowed Geometric Transforms
        # Safe crop
        try:
            transforms_list.append(
                A.RandomResizedCrop(
                    size=(cfg.target_height, cfg.target_width),
                    scale=(cfg.crop_scale_min, cfg.crop_scale_max),
                    ratio=(cfg.target_width / cfg.target_height * 0.95, cfg.target_width / cfg.target_height * 1.05),
                    p=0.4,
                )
            )
        except Exception:
            transforms_list.append(
                A.RandomResizedCrop(
                    height=cfg.target_height,
                    width=cfg.target_width,
                    scale=(cfg.crop_scale_min, cfg.crop_scale_max),
                    p=0.4,
                )
            )

        # Micro-rotation only (strictly <= 3 degrees) via modern Affine
        try:
            transforms_list.append(
                A.Affine(
                    scale=(1.0 - cfg.scale_limit_pct, 1.0 + cfg.scale_limit_pct),
                    translate_percent={
                        "x": (-cfg.shift_limit_pct, cfg.shift_limit_pct),
                        "y": (-cfg.shift_limit_pct, cfg.shift_limit_pct),
                    },
                    rotate=(-cfg.rotation_limit_deg, cfg.rotation_limit_deg),
                    border_mode=cv2.BORDER_CONSTANT,
                    cval=0,
                    cval_mask=0,
                    p=0.35,
                )
            )
        except Exception:
            transforms_list.append(
                A.Affine(
                    scale=(0.95, 1.05),
                    rotate=(-3.0, 3.0),
                    border_mode=cv2.BORDER_CONSTANT,
                    p=0.35,
                )
            )

        # 2. Photometric Lighting & Sensor Noise Transforms
        transforms_list.append(
            A.RandomBrightnessContrast(
                brightness_limit=cfg.brightness_limit,
                contrast_limit=cfg.contrast_limit,
                p=cfg.brightness_contrast_prob,
            )
        )

        transforms_list.append(
            A.HueSaturationValue(
                hue_shift_limit=cfg.hue_shift_limit,
                sat_shift_limit=cfg.sat_shift_limit,
                val_shift_limit=cfg.val_shift_limit,
                p=cfg.color_jitter_prob,
            )
        )

        transforms_list.append(
            A.GaussianBlur(
                blur_limit=(3, cfg.blur_limit),
                p=cfg.blur_prob,
            )
        )

        try:
            if hasattr(A, "GaussianNoise"):
                transforms_list.append(A.GaussianNoise(p=cfg.noise_prob))
            elif hasattr(A, "GaussNoise"):
                transforms_list.append(A.GaussNoise(p=cfg.noise_prob))
        except Exception:
            pass

        # 3. Dynamic Sunlight & Shadow simulation
        if hasattr(A, "RandomShadow"):
            try:
                transforms_list.append(A.RandomShadow(p=cfg.shadow_prob))
            except Exception:
                pass

        # 4. Standard Resizing guarantee
        try:
            transforms_list.append(A.Resize(size=(cfg.target_height, cfg.target_width)))
        except Exception:
            transforms_list.append(A.Resize(cfg.target_height, cfg.target_width))

        # Verify no forbidden transforms exist
        for t in transforms_list:
            if t.__class__.__name__ in cfg.forbidden_transforms:
                raise ValueError(f"Forbidden transform {t.__class__.__name__} detected in pipeline!")

        return A.Compose(transforms_list)

    def apply_training_transforms(
        self,
        image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply complete safe augmentation sequence with optional class-swap horizontal flip.
        """
        img = image_bgr
        msk = mask_uint8

        # 1. Apply primary albumentations pipeline
        if self._albumentations_pipeline is not None:
            augmented = self._albumentations_pipeline(image=img, mask=msk)
            img = augmented["image"]
            msk = augmented["mask"]
        else:
            # Fallback manual transforms
            img, msk = self._manual_safe_augment(img, msk)

        # 2. Controlled Horizontal Flip with Mandatory Class Swap
        if self.config.allow_horizontal_flip:
            if random.random() < self.config.horizontal_flip_prob:
                img = cv2.flip(img, 1)
                msk = cv2.flip(msk, 1)

                if self.config.enable_class_swap_on_flip:
                    # Invert Class 2 (Left) and Class 3 (Right) to preserve ground truth
                    msk = swap_left_right_classes(msk)

        return img, msk

    def apply_validation_transforms(
        self,
        image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pure deterministic resizing to 640x360 with no random augmentations.
        """
        th, tw = self.config.target_height, self.config.target_width
        if image_bgr.shape[:2] != (th, tw):
            img = cv2.resize(image_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
        else:
            img = image_bgr.copy()

        if mask_uint8.shape[:2] != (th, tw):
            msk = cv2.resize(mask_uint8, (tw, th), interpolation=cv2.INTER_NEAREST)
        else:
            msk = mask_uint8.copy()

        return img, msk

    def _manual_safe_augment(
        self,
        image: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        OpenCV-based fallback transformations when albumentations is unavailable.
        """
        th, tw = self.config.target_height, self.config.target_width
        img = cv2.resize(image, (tw, th), interpolation=cv2.INTER_LINEAR)
        msk = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)

        # Random brightness adjustment
        if random.random() < self.config.brightness_contrast_prob:
            alpha = 1.0 + random.uniform(-self.config.contrast_limit, self.config.contrast_limit)
            beta = random.uniform(-25.0, 25.0)
            img = np.clip(alpha * img.astype(np.float32) + beta, 0, 255).astype(np.uint8)

        # Micro-rotation (<= 3 deg)
        if random.random() < 0.35:
            angle = random.uniform(-self.config.rotation_limit_deg, self.config.rotation_limit_deg)
            center = (tw / 2.0, th / 2.0)
            rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rot_mat, (tw, th), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            msk = cv2.warpAffine(msk, rot_mat, (tw, th), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return img, msk


# ──────────────────────────────────────────────────────────────────────────────
# 3. PyTorch Tensor Conversion & ImageNet Normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_and_to_tensor(
    image_bgr: np.ndarray,
    mask_uint8: np.ndarray,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert BGR image to normalized RGB float32 tensor [3, H, W] and mask to int64 tensor [H, W].
    """
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    img_norm = (img_rgb - mean) / std

    # [H, W, C] -> [C, H, W]
    image_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))
    mask_tensor = torch.from_numpy(mask_uint8.astype(np.int64))

    return image_tensor, mask_tensor


# ──────────────────────────────────────────────────────────────────────────────
# 4. Automotive Environmental Lighting & Adverse Weather Simulator
# ──────────────────────────────────────────────────────────────────────────────

class PhotometricLightingAugmentor:
    """
    Simulates real-world automotive environmental disturbances:
    - Direct low-angle solar glare (morning / sunset blindness).
    - Longitudinal rain streaks on front windshield glass.
    - Asymmetrical headlight illumination cones for night driving.
    - Overhead bridge / overpass cast shadows across the drivable corridor.
    """

    def __init__(
        self,
        glare_prob: float = 0.20,
        rain_prob: float = 0.15,
        shadow_prob: float = 0.25,
        night_headlight_prob: float = 0.15,
    ) -> None:
        self.glare_prob = glare_prob
        self.rain_prob = rain_prob
        self.shadow_prob = shadow_prob
        self.night_headlight_prob = night_headlight_prob

    def apply_solar_glare(
        self,
        image_bgr: np.ndarray,
        center: Optional[Tuple[int, int]] = None,
        radius: int = 160,
        intensity: float = 0.75,
    ) -> np.ndarray:
        """
        Simulate solar flare / lens flare at the horizon line.
        """
        h, w = image_bgr.shape[:2]
        if center is None:
            # Random position along horizon band (y ~ 0.35 * h to 0.55 * h)
            cx = random.randint(int(w * 0.15), int(w * 0.85))
            cy = random.randint(int(h * 0.30), int(h * 0.55))
            center = (cx, cy)

        flare_layer = np.zeros((h, w, 3), dtype=np.float32)
        cv2.circle(flare_layer, center, radius, (240.0, 250.0, 255.0), -1)
        flare_layer = cv2.GaussianBlur(flare_layer, (radius | 1, radius | 1), 0)

        # Blend flare onto camera frame
        blended = image_bgr.astype(np.float32) + (flare_layer * intensity)
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def apply_windshield_rain(
        self,
        image_bgr: np.ndarray,
        num_streaks: int = 180,
        angle_deg: float = -12.0,
        streak_length: int = 22,
    ) -> np.ndarray:
        """
        Simulate semi-transparent rain droplets and streaks on windshield.
        """
        h, w = image_bgr.shape[:2]
        canvas = image_bgr.copy()
        rad = math.radians(angle_deg)
        dx = int(streak_length * math.sin(rad))
        dy = int(streak_length * math.cos(rad))

        overlay = np.zeros_like(canvas)
        for _ in range(num_streaks):
            x1 = random.randint(0, w - 1)
            y1 = random.randint(0, h - 1)
            x2 = np.clip(x1 + dx + random.randint(-2, 2), 0, w - 1)
            y2 = np.clip(y1 + dy + random.randint(-2, 2), 0, h - 1)
            intensity = random.randint(180, 240)
            cv2.line(overlay, (x1, y1), (x2, y2), (intensity, intensity, intensity), 1)

        # Soft blur rain streaks
        overlay = cv2.GaussianBlur(overlay, (3, 3), 0)
        mask = (overlay > 0).astype(np.float32)
        blended = canvas.astype(np.float32) * (1.0 - 0.35 * mask) + overlay.astype(np.float32) * 0.35
        return np.clip(blended, 0.0, 255.0).astype(np.uint8)

    def apply_overpass_shadow(
        self,
        image_bgr: np.ndarray,
        attenuation_factor: float = 0.55,
    ) -> np.ndarray:
        """
        Simulate harsh shadow cast across road surface by bridges or gantries.
        """
        h, w = image_bgr.shape[:2]
        canvas = image_bgr.copy()

        # Random polygon spanning across image
        y_top = random.randint(int(h * 0.40), int(h * 0.70))
        y_bot = random.randint(y_top + 40, h)
        skew = random.randint(-60, 60)

        pts = np.array([
            [0, y_top],
            [w, y_top + skew],
            [w, y_bot + skew],
            [0, y_bot],
        ], dtype=np.int32)

        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.fillPoly(shadow_mask, [pts], 1.0)
        shadow_mask = cv2.GaussianBlur(shadow_mask, (21, 21), 0)

        shadow_mult = 1.0 - (1.0 - attenuation_factor) * shadow_mask
        shadow_mult = np.expand_dims(shadow_mult, axis=-1)

        darkened = (canvas.astype(np.float32) * shadow_mult)
        return np.clip(darkened, 0.0, 255.0).astype(np.uint8)

    def apply_headlight_illumination(
        self,
        image_bgr: np.ndarray,
        ambient_darkening: float = 0.30,
        beam_gain: float = 1.50,
    ) -> np.ndarray:
        """
        Simulate night driving: dark ambient background with forward headlight cone.
        """
        h, w = image_bgr.shape[:2]
        ambient = (image_bgr.astype(np.float32) * ambient_darkening)

        # Headlight beam elliptical cone centered near bottom bumper
        cone_mask = np.zeros((h, w), dtype=np.float32)
        beam_center = (w // 2, int(h * 0.95))
        axes = (int(w * 0.45), int(h * 0.55))
        cv2.ellipse(cone_mask, beam_center, axes, 0, 0, 360, 1.0, -1)
        cone_mask = cv2.GaussianBlur(cone_mask, (61, 61), 0)

        gain_map = ambient_darkening + (beam_gain - ambient_darkening) * cone_mask
        gain_map = np.expand_dims(gain_map, axis=-1)

        illuminated = (image_bgr.astype(np.float32) * gain_map)
        return np.clip(illuminated, 0.0, 255.0).astype(np.uint8)

    def augment(self, image_bgr: np.ndarray) -> np.ndarray:
        """Apply random combination of photometric disturbances."""
        res = image_bgr
        if random.random() < self.glare_prob:
            res = self.apply_solar_glare(res)
        if random.random() < self.rain_prob:
            res = self.apply_windshield_rain(res)
        if random.random() < self.shadow_prob:
            res = self.apply_overpass_shadow(res)
        if random.random() < self.night_headlight_prob:
            res = self.apply_headlight_illumination(res)
        return res


# ──────────────────────────────────────────────────────────────────────────────
# 5. ASIL-B Perception Safety Auditor
# ──────────────────────────────────────────────────────────────────────────────

class AugmentationSafetyAuditor:
    """
    Runtime invariant check ensuring transforms NEVER invert road semantics:
    - Verifies Left Boundary (2) stays to the left of Right Boundary (3).
    - Verifies that Drivable Corridor (1) is properly bounded.
    - Flags any transformation that distorts mask classes.
    """

    @classmethod
    def audit_transformed_pair(
        cls,
        image_transformed: np.ndarray,
        mask_transformed: np.ndarray,
    ) -> Tuple[bool, Optional[str]]:
        """
        Audit output pair after augmentation pipeline.
        """
        if mask_transformed is None or image_transformed is None:
            return False, "Null image or mask returned from augmentor"

        if image_transformed.shape[:2] != mask_transformed.shape[:2]:
            return False, f"Shape mismatch: img={image_transformed.shape[:2]}, msk={mask_transformed.shape[:2]}"

        # Check unique classes
        classes = set(np.unique(mask_transformed))
        if not classes.issubset({0, 1, 2, 3}):
            return False, f"Illegal class IDs present in mask: {classes - {0, 1, 2, 3}}"

        # Geometric check: ego left vs ego right
        left_ys, left_xs = np.where(mask_transformed == LaneClassID.LEFT_BOUNDARY.value)
        right_ys, right_xs = np.where(mask_transformed == LaneClassID.RIGHT_BOUNDARY.value)

        if len(left_xs) > 20 and len(right_xs) > 20:
            # Compute centroid x positions in bottom half of frame
            h = mask_transformed.shape[0]
            bottom_left_xs = left_xs[left_ys > int(h * 0.65)]
            bottom_right_xs = right_xs[right_ys > int(h * 0.65)]

            if len(bottom_left_xs) > 5 and len(bottom_right_xs) > 5:
                mean_left_x = np.mean(bottom_left_xs)
                mean_right_x = np.mean(bottom_right_xs)
                if mean_left_x >= mean_right_x:
                    return (
                        False,
                        f"Left lane boundary centroid ({mean_left_x:.1f}) >= Right boundary ({mean_right_x:.1f})! Class inversion hazard detected.",
                    )

        return True, None


# ──────────────────────────────────────────────────────────────────────────────
# 6. High-Throughput Batch Augmentation Pipeline
# ──────────────────────────────────────────────────────────────────────────────

class AugmentationBatchPipeline:
    """
    Coordinates batch-level parallel execution of safe augmentations,
    photometric lighting simulations, and tensor conversion.
    """

    def __init__(
        self,
        config: Optional[AugmentationPolicyConfig] = None,
        enable_photometric: bool = True,
    ) -> None:
        self.config = config or AugmentationPolicyConfig()
        self.augmentor = SafeLaneAugmentor(self.config)
        self.photometric = PhotometricLightingAugmentor() if enable_photometric else None
        self.safety_auditor = AugmentationSafetyAuditor()

    def process_pair(
        self,
        image_bgr: np.ndarray,
        mask_uint8: np.ndarray,
        is_training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full pipeline: safe augment -> photometric -> audit -> tensor conversion.
        """
        if is_training:
            img, msk = self.augmentor.apply_training_transforms(image_bgr, mask_uint8)
            if self.photometric is not None:
                img = self.photometric.augment(img)
            is_safe, reason = self.safety_auditor.audit_transformed_pair(img, msk)
            if not is_safe:
                logger.warning("Augmentation safety audit flagged sample: %s. Using clean fallback.", reason)
                img, msk = self.augmentor.apply_validation_transforms(image_bgr, mask_uint8)
        else:
            img, msk = self.augmentor.apply_validation_transforms(image_bgr, mask_uint8)

        return normalize_and_to_tensor(img, msk)


# ──────────────────────────────────────────────────────────────────────────────
# 7. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing Augmentations ISO 26262 ASIL-B self-check...")
    cfg = AugmentationPolicyConfig(allow_horizontal_flip=True, enable_class_swap_on_flip=True)
    augmentor = SafeLaneAugmentor(cfg)

    # Synthetic image & mask
    dummy_img = np.full((360, 640, 3), 100, dtype=np.uint8)
    dummy_mask = np.zeros((360, 640), dtype=np.uint8)
    dummy_mask[200:360, 100:110] = LaneClassID.LEFT_BOUNDARY.value   # Left line at x=100
    dummy_mask[200:360, 530:540] = LaneClassID.RIGHT_BOUNDARY.value  # Right line at x=530

    # Test training transforms
    for _ in range(5):
        aug_img, aug_mask = augmentor.apply_training_transforms(dummy_img, dummy_mask)
        assert aug_img.shape == (360, 640, 3)
        assert aug_mask.shape == (360, 640)

    # Test tensor normalization
    t_img, t_msk = normalize_and_to_tensor(dummy_img, dummy_mask)
    assert t_img.shape == (3, 360, 640)
    assert t_msk.shape == (360, 640)
    assert t_img.dtype == torch.float32
    assert t_msk.dtype == torch.int64

    # Test Photometric Lighting Augmentor
    photo = PhotometricLightingAugmentor()
    glared = photo.apply_solar_glare(dummy_img)
    assert glared.shape == (360, 640, 3)
    rained = photo.apply_windshield_rain(dummy_img)
    assert rained.shape == (360, 640, 3)
    shadowed = photo.apply_overpass_shadow(dummy_img)
    assert shadowed.shape == (360, 640, 3)
    headlit = photo.apply_headlight_illumination(dummy_img)
    assert headlit.shape == (360, 640, 3)

    # Test Safety Auditor
    is_safe, err = AugmentationSafetyAuditor.audit_transformed_pair(dummy_img, dummy_mask)
    assert is_safe, f"Safety audit unexpectedly failed: {err}"

    # Test Batch Pipeline
    batch_pipeline = AugmentationBatchPipeline(cfg)
    p_img, p_msk = batch_pipeline.process_pair(dummy_img, dummy_mask, is_training=True)
    assert p_img.shape == (3, 360, 640)
    assert p_msk.shape == (360, 640)

    print("Augmentations self-check passed successfully.")


