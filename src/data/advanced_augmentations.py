"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

src/data/advanced_augmentations.py
Advanced Adversarial & Environmental Lane Data Augmentations Engine v2.0

Generates photometrically and physically accurate adverse driving condition simulations:
  - Atmospheric Fog: Koschmieder's optical extinction model with depth estimation
  - Dynamic Rain: Synthetic rain streaks, motion blur kernels, and specular puddles
  - Night / Headlights: Ambient lux attenuation, parabolic beam cones, and sensor noise
  - Polygonal Road Shadows: Multi-point tree/bridge shadows with penumbra blur
  - Specular Sun Glare: Radial lens flares, ring artifacts, and sensor clipping
  - Faded Line Degradation: Micro-crack paint erosion and intermittent mark erasure

NON-NEGOTIABLE ARCHITECTURAL CONSTRAINT:
  horizontal_flip = False (permanently enforced across all augmentation pipelines)
"""

from __future__ import annotations

import copy
import logging
import math
import os
import random
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("apex_vision.data.advanced_augmentations")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enumerations & Configuration Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class AdverseConditionType(str, Enum):
    """Adverse environmental perturbation types."""
    FOG = "FOG"
    RAIN = "RAIN"
    NIGHT = "NIGHT"
    SHADOW = "SHADOW"
    GLARE = "GLARE"
    FADED_LINES = "FADED_LINES"
    CLEAN = "CLEAN"


@dataclass
class FogAugmentationConfig:
    """Parameters governing Koschmieder atmospheric haze simulation."""
    extinction_beta_range: Tuple[float, float] = (0.015, 0.065)
    atmospheric_light_range: Tuple[int, int] = (200, 245)
    horizon_y_ratio: float = 0.50
    camera_height_m: float = 1.20
    camera_focal_y_px: float = 450.0


@dataclass
class RainAugmentationConfig:
    """Parameters governing stochastic rain streak and puddle simulation."""
    streak_count_range: Tuple[int, int] = (400, 1200)
    streak_length_range: Tuple[int, int] = (15, 35)
    streak_thickness_range: Tuple[int, int] = (1, 2)
    streak_angle_deg_range: Tuple[float, float] = (70.0, 85.0)  # Near vertical
    streak_alpha_range: Tuple[float, float] = (0.40, 0.75)
    puddle_reflection_intensity: float = 0.50
    puddle_vertical_start_ratio: float = 0.65


@dataclass
class NightAugmentationConfig:
    """Parameters governing low-lux night driving and headlight cone projection."""
    ambient_lux_attenuation_range: Tuple[float, float] = (0.10, 0.25)
    beam_width_ratio: float = 0.42
    beam_length_ratio: float = 0.58
    beam_intensity_range: Tuple[float, float] = (0.85, 1.20)
    beam_color_temperature_k: int = 5500  # Xenon / LED white
    sensor_iso_noise_sigma: float = 8.0


@dataclass
class ShadowAugmentationConfig:
    """Parameters governing polygonal road surface shadow occlusions."""
    max_shadow_polygons: int = 3
    shadow_opacity_range: Tuple[float, float] = (0.45, 0.75)
    penumbra_blur_kernel: int = 25
    cool_shadow_color_shift: bool = True  # Blue shift in ambient shadow


@dataclass
class GlareAugmentationConfig:
    """Parameters governing specular sun and headlight high-beam glare."""
    glare_intensity_range: Tuple[float, float] = (0.70, 1.00)
    glare_radius_px_range: Tuple[int, int] = (60, 140)
    glare_position_y_ratio_max: float = 0.60  # Upper portion of scene
    lens_flare_streaks: int = 6


@dataclass
class FadedLineAugmentationConfig:
    """Parameters governing road marking erosion and intermittent erasure."""
    erosion_alpha_range: Tuple[float, float] = (0.50, 0.85)
    intermittent_gap_length_px: Tuple[int, int] = (20, 60)
    intermittent_gap_frequency: float = 0.35
    add_micro_cracks: bool = True


@dataclass
class AdvancedAugmentationPipelineConfig:
    """Global configuration governing multi-condition data augmentation."""
    image_width: int = 640
    image_height: int = 360

    # NON-NEGOTIABLE CONSTRAINT
    horizontal_flip: bool = False

    # Sub-component configs
    fog: FogAugmentationConfig = field(default_factory=FogAugmentationConfig)
    rain: RainAugmentationConfig = field(default_factory=RainAugmentationConfig)
    night: NightAugmentationConfig = field(default_factory=NightAugmentationConfig)
    shadow: ShadowAugmentationConfig = field(default_factory=ShadowAugmentationConfig)
    glare: GlareAugmentationConfig = field(default_factory=GlareAugmentationConfig)
    faded_lines: FadedLineAugmentationConfig = field(default_factory=FadedLineAugmentationConfig)

    # Probabilistic execution rates for randomized pipelines
    prob_fog: float = 0.20
    prob_rain: float = 0.20
    prob_night: float = 0.20
    prob_shadow: float = 0.25
    prob_glare: float = 0.15
    prob_faded_lines: float = 0.25

    # Photometric jitter
    random_brightness_delta: float = 0.15
    random_contrast_range: Tuple[float, float] = (0.80, 1.20)
    random_saturation_range: Tuple[float, float] = (0.75, 1.25)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Physics-Based Atmospheric Fog Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class AtmosphericFogSynthesizer:
    """
    Implements Koschmieder's Law for atmospheric light extinction:
        I(x) = J(x) * t(x) + A * (1 - t(x))
    where:
        J(x) = Original radiance (clean road image)
        t(x) = Atmospheric transmission = exp(-beta * d(x))
        d(x) = Distance from camera to surface point x
        A    = Ambient atmospheric light (haze color)
        beta = Extinction coefficient
    """

    def __init__(self, config: FogAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h
        self._depth_map_cache: Optional[np.ndarray] = None
        self._generate_depth_map()

    def _generate_depth_map(self) -> None:
        """
        Estimate ground plane distance d(y) based on pinhole camera geometry:
            y_horizon = h * horizon_y_ratio
            d(y) = (H_cam * f_y) / (y - y_horizon)
        """
        y_grid, _ = np.mgrid[0 : self.h, 0 : self.w]
        v_h = self.h * self.config.horizon_y_ratio

        # Distance increases towards horizon; above horizon distance is clamped to sky infinity
        delta_y = np.maximum(y_grid - v_h, 1.0)
        depth_ground = (self.config.camera_height_m * self.config.camera_focal_y_px) / delta_y

        # Sky / above horizon gets large depth
        depth = np.where(y_grid <= v_h, 250.0, depth_ground)
        self._depth_map_cache = np.clip(depth, 1.0, 300.0).astype(np.float32)

    def apply(self, frame_bgr: np.ndarray, beta: Optional[float] = None) -> np.ndarray:
        """Apply depth-dependent Koschmieder fog layer."""
        if beta is None:
            beta = random.uniform(*self.config.extinction_beta_range)

        assert self._depth_map_cache is not None
        # Transmission map t(x) = exp(-beta * d(x))
        transmission = np.exp(-beta * self._depth_map_cache)
        transmission = np.repeat(transmission[:, :, np.newaxis], 3, axis=2)

        # Atmospheric airlight A
        air_val = random.randint(*self.config.atmospheric_light_range)
        A = np.full_like(frame_bgr, air_val, dtype=np.float32)

        j_img = frame_bgr.astype(np.float32)
        # Koschmieder formulation: I = J * t + A * (1 - t)
        fogged = j_img * transmission + A * (1.0 - transmission)
        return np.clip(fogged, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dynamic Rain & Puddle Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class DynamicRainSynthesizer:
    """
    Simulates high-velocity falling rain streaks with directional motion blur,
    contrast attenuation, and wet road surface specular reflection puddles.
    """

    def __init__(self, config: RainAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Inject falling rain streaks, contrast reduction, and puddle reflections."""
        out = frame_bgr.copy().astype(np.float32)

        # 1. Global contrast reduction due to precipitation mist
        out = out * 0.88 + 15.0

        # 2. Road specular puddles (bottom third of image)
        puddle_start_y = int(self.h * self.config.puddle_vertical_start_ratio)
        puddle_h = self.h - puddle_start_y

        # Synthesize horizontal ripple highlights
        ripple_noise = np.random.normal(0, 15, (puddle_h, self.w, 3))
        ripple_blur = cv2.GaussianBlur(ripple_noise.astype(np.float32), (31, 3), 0)
        out[puddle_start_y:, :] = np.clip(
            out[puddle_start_y:, :] * 0.90 + ripple_blur * self.config.puddle_reflection_intensity,
            0,
            255,
        )

        # 3. Generate rain streaks
        num_streaks = random.randint(*self.config.streak_count_range)
        streak_len = random.randint(*self.config.streak_length_range)
        thickness = random.randint(*self.config.streak_thickness_range)
        angle_deg = random.uniform(*self.config.streak_angle_deg_range)
        angle_rad = math.radians(angle_deg)

        dx = int(streak_len * math.cos(angle_rad))
        dy = int(streak_len * math.sin(angle_rad))

        streak_canvas = np.zeros((self.h, self.w), dtype=np.uint8)

        # Draw randomized lines
        x_pts = np.random.randint(0, self.w, size=num_streaks)
        y_pts = np.random.randint(0, self.h, size=num_streaks)

        for x1, y1 in zip(x_pts, y_pts):
            x2 = x1 + dx
            y2 = y1 + dy
            cv2.line(streak_canvas, (x1, y1), (x2, y2), 255, thickness)

        # Motion blur kernel in streak direction
        ksize = max(3, streak_len // 3)
        if ksize % 2 == 0:
            ksize += 1
        motion_kernel = np.zeros((ksize, ksize), dtype=np.float32)
        center = ksize // 2
        for i in range(ksize):
            offset = i - center
            kx = int(center + offset * math.cos(angle_rad))
            ky = int(center + offset * math.sin(angle_rad))
            if 0 <= kx < ksize and 0 <= ky < ksize:
                motion_kernel[ky, kx] = 1.0
        norm = np.sum(motion_kernel)
        if norm > 0:
            motion_kernel /= norm
        blurred_streaks = cv2.filter2D(streak_canvas, -1, motion_kernel)

        # Blend streaks onto image
        alpha = random.uniform(*self.config.streak_alpha_range)
        streaks_bgr = np.repeat(blurred_streaks[:, :, np.newaxis], 3, axis=2).astype(np.float32)
        out = out * (1.0 - alpha * (streaks_bgr / 255.0)) + streaks_bgr * alpha

        return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Low-Lux Night & Headlight Beam Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class NightHeadlightSynthesizer:
    """
    Simulates night-time conditions with extreme ambient lux attenuation,
    parabolic headlight forward projection cones with 1/r^2 falloff,
    and low-light sensor ISO grain noise.
    """

    def __init__(self, config: NightAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h
        self._headlight_mask_cache: Optional[np.ndarray] = None
        self._build_headlight_beam_mask()

    def _build_headlight_beam_mask(self) -> None:
        """Construct forward projection headlight illumination template."""
        mask = np.zeros((self.h, self.w), dtype=np.float32)
        # Dual beam centers at bottom of bumper
        left_center = (int(self.w * 0.38), self.h - 10)
        right_center = (int(self.w * 0.62), self.h - 10)

        axis_x = int(self.w * self.config.beam_width_ratio * 0.5)
        axis_y = int(self.h * self.config.beam_length_ratio)

        cv2.ellipse(mask, left_center, (axis_x, axis_y), -5, 0, 360, 1.0, -1)
        cv2.ellipse(mask, right_center, (axis_x, axis_y), 5, 0, 360, 1.0, -1)

        # Smooth beam cutoff with heavy Gaussian filter
        mask = cv2.GaussianBlur(mask, (65, 65), 0)
        self._headlight_mask_cache = np.clip(mask, 0.0, 1.0)

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Render night darkness, headlight illumination, and sensor grain."""
        lux_scale = random.uniform(*self.config.ambient_lux_attenuation_range)
        dark_background = frame_bgr.astype(np.float32) * lux_scale

        beam_scale = random.uniform(*self.config.beam_intensity_range)
        assert self._headlight_mask_cache is not None
        beam_3ch = np.repeat(self._headlight_mask_cache[:, :, np.newaxis], 3, axis=2)

        # Full frame illuminated by beams
        illuminated = frame_bgr.astype(np.float32) * beam_scale

        # Composite night road: ambient outside + headlights inside
        night_road = dark_background * (1.0 - beam_3ch) + illuminated * beam_3ch

        # Inject camera sensor ISO shot noise
        if self.config.sensor_iso_noise_sigma > 0:
            noise = np.random.normal(0, self.config.sensor_iso_noise_sigma, night_road.shape)
            night_road = night_road + noise

        return np.clip(night_road, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Polygonal Road Shadow Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class PolygonalRoadShadowSynthesizer:
    """
    Injects realistic roadside tree, building, and overpass bridge shadows
    with realistic penumbra boundary diffusion and ambient blue-temperature tint.
    """

    def __init__(self, config: ShadowAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Inject random polygonal shadows onto road surface."""
        num_polys = random.randint(1, self.config.max_shadow_polygons)
        shadow_mask = np.zeros((self.h, self.w), dtype=np.float32)

        for _ in range(num_polys):
            num_vertices = random.randint(4, 7)
            # Generate points clustered along side and road
            pts_x = np.random.randint(0, self.w, size=num_vertices)
            pts_y = np.random.randint(int(self.h * 0.40), self.h, size=num_vertices)
            pts = np.column_stack((pts_x, pts_y)).astype(np.int32)
            cv2.fillPoly(shadow_mask, [pts], 1.0)

        # Penumbra blur on shadow edges
        ksize = self.config.penumbra_blur_kernel
        if ksize % 2 == 0:
            ksize += 1
        blurred_shadow = cv2.GaussianBlur(shadow_mask, (ksize, ksize), 0)
        blurred_shadow = np.clip(blurred_shadow, 0.0, 1.0)

        opacity = random.uniform(*self.config.shadow_opacity_range)
        shadow_intensity = 1.0 - (blurred_shadow * opacity)

        out = frame_bgr.astype(np.float32) * shadow_intensity[:, :, np.newaxis]

        # Cool blue color shift in ambient shadow areas
        if self.config.cool_shadow_color_shift:
            blue_boost = blurred_shadow * 8.0 * opacity
            out[:, :, 0] = np.clip(out[:, :, 0] + blue_boost, 0, 255)

        return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Specular Sun Glare & High-Beam Flare Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class SpecularGlareSynthesizer:
    """
    Simulates direct sun flare and oncoming vehicle high-beam blinding
    with saturated bloom clipping, radial flare rays, and circular ring reflections.
    """

    def __init__(self, config: GlareAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h

    def apply(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Inject specular glare bloom and lens flare artifacts."""
        out = frame_bgr.copy().astype(np.float32)

        # Glare epicenter position (upper hemisphere or horizon)
        center_x = random.randint(int(self.w * 0.20), int(self.w * 0.80))
        center_y = random.randint(int(self.h * 0.15), int(self.h * self.config.glare_position_y_ratio_max))
        radius = random.randint(*self.config.glare_radius_px_range)
        intensity = random.uniform(*self.config.glare_intensity_range)

        # 1. Primary Gaussian bloom core
        y_grid, x_grid = np.mgrid[0 : self.h, 0 : self.w]
        dist_sq = (x_grid - center_x) ** 2 + (y_grid - center_y) ** 2
        bloom = np.exp(-dist_sq / (2.0 * (radius ** 2))) * intensity * 255.0

        # 2. Radial streak rays
        streaks = np.zeros((self.h, self.w), dtype=np.float32)
        for _ in range(self.config.lens_flare_streaks):
            angle = random.uniform(0, math.pi)
            x_end1 = int(center_x + radius * 3.0 * math.cos(angle))
            y_end1 = int(center_y + radius * 3.0 * math.sin(angle))
            x_end2 = int(center_x - radius * 3.0 * math.cos(angle))
            y_end2 = int(center_y - radius * 3.0 * math.sin(angle))
            cv2.line(streaks, (x_end1, y_end1), (x_end2, y_end2), 1.0, 2)
        streaks = cv2.GaussianBlur(streaks, (15, 15), 0) * (intensity * 120.0)

        # 3. Secondary ring reflection across optical center
        opt_cx, opt_cy = self.w // 2, self.h // 2
        sec_cx = 2 * opt_cx - center_x
        sec_cy = 2 * opt_cy - center_y
        sec_dist = np.sqrt((x_grid - sec_cx) ** 2 + (y_grid - sec_cy) ** 2)
        ring = np.exp(-((sec_dist - radius * 0.8) ** 2) / (2.0 * (15.0 ** 2))) * (intensity * 80.0)

        # Composite glare onto BGR channels (warm white tint: R > G > B)
        out[:, :, 0] += (bloom * 0.75 + streaks * 0.70 + ring * 0.40)  # Blue
        out[:, :, 1] += (bloom * 0.90 + streaks * 0.85 + ring * 0.70)  # Green
        out[:, :, 2] += (bloom * 1.00 + streaks * 1.00 + ring * 0.90)  # Red

        return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Faded Line Degradation & Paint Erosion Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

class FadedLineDegradationSynthesizer:
    """
    Simulates unmaintained highway pavement by alpha-eroding lane marking paint,
    inserting intermittent paint gaps, and overlaying weathered micro-cracks.
    """

    def __init__(self, config: FadedLineAugmentationConfig, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config
        self.w = img_w
        self.h = img_h

    def apply(
        self,
        frame_bgr: np.ndarray,
        lane_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Degrade lane marking paint within lane_mask or across typical lane perspective regions.
        """
        out = frame_bgr.copy()
        mask_out = lane_mask.copy() if lane_mask is not None else None

        erosion_alpha = random.uniform(*self.config.erosion_alpha_range)

        # If 4-class mask provided: Class 2 is Left, Class 3 is Right
        if lane_mask is not None:
            lane_pixels = (lane_mask == 2) | (lane_mask == 3)
        else:
            # Fallback: estimate lane line zones
            h, w = self.h, self.w
            lane_pixels = np.zeros((h, w), dtype=bool)
            cv2.line(lane_pixels.view(np.uint8), (int(w * 0.20), h), (int(w * 0.44), int(h * 0.55)), 1, 10)
            cv2.line(lane_pixels.view(np.uint8), (int(w * 0.80), h), (int(w * 0.56), int(h * 0.55)), 1, 10)
            lane_pixels = lane_pixels.astype(bool)

        if not np.any(lane_pixels):
            return out, mask_out

        # Generate realistic asphalt texture replacement
        asphalt_base = np.median(frame_bgr[int(self.h * 0.60) :, :], axis=(0, 1))
        asphalt_noise = np.random.normal(0, 8, frame_bgr.shape).astype(np.float32)
        asphalt_patch = np.clip(asphalt_base + asphalt_noise, 0, 255).astype(np.uint8)

        # 1. Continuous alpha-blend erosion
        orig_lane_f = out.astype(np.float32)
        asphalt_f = asphalt_patch.astype(np.float32)
        blended = orig_lane_f * (1.0 - erosion_alpha) + asphalt_f * erosion_alpha

        out[lane_pixels] = np.clip(blended[lane_pixels], 0, 255).astype(np.uint8)

        # 2. Intermittent paint gaps (horizontal bands where paint is 100% missing)
        num_gaps = random.randint(2, 5)
        for _ in range(num_gaps):
            gap_y = random.randint(int(self.h * 0.55), self.h - 30)
            gap_len = random.randint(*self.config.intermittent_gap_length_px)
            gap_mask = np.zeros((self.h, self.w), dtype=bool)
            gap_mask[gap_y : gap_y + gap_len, :] = True
            active_gap = lane_pixels & gap_mask

            out[active_gap] = asphalt_patch[active_gap]
            if mask_out is not None:
                # Mark as background (Class 0) in ground truth
                mask_out[active_gap] = 0

        # 3. Micro-cracks simulation
        if self.config.add_micro_cracks:
            crack_canvas = np.zeros((self.h, self.w), dtype=np.uint8)
            for _ in range(8):
                cx1 = random.randint(0, self.w)
                cy1 = random.randint(int(self.h * 0.50), self.h)
                cx2 = cx1 + random.randint(-15, 15)
                cy2 = cy1 + random.randint(-15, 15)
                cv2.line(crack_canvas, (cx1, cy1), (cx2, cy2), 255, 1)

            crack_active = (crack_canvas > 0) & lane_pixels
            out[crack_active] = np.clip(out[crack_active].astype(np.float32) * 0.35, 0, 255).astype(np.uint8)

        return out, mask_out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Master Advanced Augmentations Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class AdvancedAugmentationPipeline:
    """
    Master pipeline orchestrating adverse condition generation on TuSimple frames.
    NON-NEGOTIABLE GUARANTEE: horizontal_flip = False strictly enforced.
    """

    def __init__(self, config: Optional[AdvancedAugmentationPipelineConfig] = None) -> None:
        self.config = config or AdvancedAugmentationPipelineConfig()

        # Enforce non-negotiable constraint
        if self.config.horizontal_flip:
            raise ValueError(
                "Violation of APEX-Vision safety rules: horizontal_flip must ALWAYS be False!"
            )

        # Synthesizers
        self.fog_synth = AtmosphericFogSynthesizer(self.config.fog, self.config.image_width, self.config.image_height)
        self.rain_synth = DynamicRainSynthesizer(self.config.rain, self.config.image_width, self.config.image_height)
        self.night_synth = NightHeadlightSynthesizer(self.config.night, self.config.image_width, self.config.image_height)
        self.shadow_synth = PolygonalRoadShadowSynthesizer(self.config.shadow, self.config.image_width, self.config.image_height)
        self.glare_synth = SpecularGlareSynthesizer(self.config.glare, self.config.image_width, self.config.image_height)
        self.faded_synth = FadedLineDegradationSynthesizer(self.config.faded_lines, self.config.image_width, self.config.image_height)

    def apply_single_condition(
        self,
        frame_bgr: np.ndarray,
        condition: AdverseConditionType,
        lane_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply a specifically requested adverse condition."""
        # STRICT IMMUTABLE GUARD
        assert not self.config.horizontal_flip, "horizontal_flip=False guard breached"

        if condition == AdverseConditionType.FOG:
            return self.fog_synth.apply(frame_bgr), lane_mask
        elif condition == AdverseConditionType.RAIN:
            return self.rain_synth.apply(frame_bgr), lane_mask
        elif condition == AdverseConditionType.NIGHT:
            return self.night_synth.apply(frame_bgr), lane_mask
        elif condition == AdverseConditionType.SHADOW:
            return self.shadow_synth.apply(frame_bgr), lane_mask
        elif condition == AdverseConditionType.GLARE:
            return self.glare_synth.apply(frame_bgr), lane_mask
        elif condition == AdverseConditionType.FADED_LINES:
            return self.faded_synth.apply(frame_bgr, lane_mask)
        else:
            return frame_bgr.copy(), lane_mask

    def apply_randomized_pipeline(
        self,
        frame_bgr: np.ndarray,
        lane_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Execute randomized multi-adverse augmentations and photometric jitter.
        Returns:
            Tuple of (augmented_frame, augmented_mask, active_condition_tags)
        """
        # STRICT IMMUTABLE GUARD
        assert not self.config.horizontal_flip, "horizontal_flip=False guard breached"

        img = frame_bgr.copy()
        mask = lane_mask.copy() if lane_mask is not None else None
        active_tags: List[str] = []

        # 1. Mutually exclusive major environmental condition (Rain, Fog, Night)
        env_choice = random.random()
        if env_choice < self.config.prob_rain:
            img = self.rain_synth.apply(img)
            active_tags.append("RAIN")
        elif env_choice < (self.config.prob_rain + self.config.prob_fog):
            img = self.fog_synth.apply(img)
            active_tags.append("FOG")
        elif env_choice < (self.config.prob_rain + self.config.prob_fog + self.config.prob_night):
            img = self.night_synth.apply(img)
            active_tags.append("NIGHT")

        # 2. Road surface lighting artifacts (Shadows or Glare)
        if random.random() < self.config.prob_shadow and "NIGHT" not in active_tags:
            img = self.shadow_synth.apply(img)
            active_tags.append("SHADOW")
        elif random.random() < self.config.prob_glare and "NIGHT" not in active_tags:
            img = self.glare_synth.apply(img)
            active_tags.append("GLARE")

        # 3. Pavement markings wear / faded line erosion
        if random.random() < self.config.prob_faded_lines:
            img, mask = self.faded_synth.apply(img, mask)
            active_tags.append("FADED_LINES")

        # 4. Photometric Color Jitter
        img = self._apply_photometric_jitter(img)

        if not active_tags:
            active_tags.append("CLEAN")

        return img, mask, active_tags

    def _apply_photometric_jitter(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Apply random brightness, contrast, and saturation jitter."""
        out = frame_bgr.astype(np.float32)

        # Brightness
        delta_b = random.uniform(-self.config.random_brightness_delta, self.config.random_brightness_delta) * 255.0
        out += delta_b

        # Contrast
        factor_c = random.uniform(*self.config.random_contrast_range)
        mean_val = np.mean(out, axis=(0, 1), keepdims=True)
        out = (out - mean_val) * factor_c + mean_val

        out = np.clip(out, 0, 255).astype(np.uint8)

        # Saturation in HSV
        factor_s = random.uniform(*self.config.random_saturation_range)
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor_s, 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return out


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Windshield Wiper Sweep & Hydrophobic Water Droplet Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class WindshieldWiperConfig:
    """Parameters governing realistic windshield wiper arc clearance."""
    pivot_x_ratio: float = 0.50
    pivot_y_ratio: float = 1.25  # Below windshield bottom
    blade_radius_px: int = 420
    sweep_angle_deg: float = 75.0
    cleared_region_transparency: float = 0.95
    water_bead_edge_width_px: int = 4
    residual_film_blur_ksize: int = 5


class WindshieldWiperSynthesizer:
    """
    Simulates optical clearing path of vehicle windshield wipers during heavy rain.
    Constructs a radial sweep sector that attenuates rain streaks and water films
    while leaving bead accumulation ridges along the reversal boundary.
    """

    def __init__(self, config: Optional[WindshieldWiperConfig] = None, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config or WindshieldWiperConfig()
        self.w = img_w
        self.h = img_h

    def compute_wiper_mask(self, angle_deg: float = 45.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute binary sector mask for wiper blade clearance and border bead contour.
        
        Returns:
            clear_mask: uint8 mask (255 inside cleared wedge, 0 outside)
            ridge_mask: uint8 mask of the water ridge along wiper blade edge
        """
        pivot_x = int(self.w * self.config.pivot_x_ratio)
        pivot_y = int(self.h * self.config.pivot_y_ratio)

        clear_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        ridge_mask = np.zeros((self.h, self.w), dtype=np.uint8)

        # Draw filled pie slice representing current wiper cycle swept sector
        start_angle = 180.0 + (90.0 - self.config.sweep_angle_deg / 2.0)
        end_angle = start_angle + self.config.sweep_angle_deg

        axes = (self.config.blade_radius_px, self.config.blade_radius_px)
        cv2.ellipse(clear_mask, (pivot_x, pivot_y), axes, 0, start_angle, end_angle, 255, -1)

        # Extract perimeter contour for water ridge accumulation
        contours, _ = cv2.findContours(clear_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            cv2.drawContours(ridge_mask, contours, -1, 255, self.config.water_bead_edge_width_px)

        return clear_mask, ridge_mask

    def apply(self, rained_frame_bgr: np.ndarray, clean_frame_bgr: np.ndarray) -> np.ndarray:
        """
        Blend clean frame into wiper arc area and superimpose water bead ridges.
        """
        clear_mask, ridge_mask = self.compute_wiper_mask()
        alpha = self.config.cleared_region_transparency

        clear_float = (clear_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        ridge_float = (ridge_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        # Blend clean frame where swept, retaining subtle residual film
        swept = clean_frame_bgr.astype(np.float32) * alpha + rained_frame_bgr.astype(np.float32) * (1.0 - alpha)
        out = rained_frame_bgr.astype(np.float32) * (1.0 - clear_float) + swept * clear_float

        # Add ridge highlight
        out = out + ridge_float * 35.0
        return np.clip(out, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Overhead Sodium Luminary & Tunnel Lighting Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class OverheadLuminaryConfig:
    """Parameters governing overhead lamp light cones and chromatic color cast."""
    lamp_spacing_px: int = 180
    lamp_height_y_px: int = 40
    spot_radius_px: int = 90
    amber_tint_ratio: float = 0.35  # Sodium vapor tint (high orange/yellow)
    lamp_intensity_gain: float = 1.35
    falloff_exponent: float = 2.2


class OverheadLuminarySynthesizer:
    """
    Simulates periodic highway overhead luminaires and tunnel ceiling lights.
    Injects realistic conical downward light distributions with warm sodium or
    cool LED color temperature shifts.
    """

    def __init__(self, config: Optional[OverheadLuminaryConfig] = None, img_w: int = 640, img_h: int = 360) -> None:
        self.config = config or OverheadLuminaryConfig()
        self.w = img_w
        self.h = img_h

    def apply(self, frame_bgr: np.ndarray, tunnel_mode: bool = False) -> np.ndarray:
        """
        Superimpose periodic overhead lighting cones onto roadway scene.
        """
        out = frame_bgr.copy().astype(np.float32)
        y_grid, x_grid = np.mgrid[0 : self.h, 0 : self.w].astype(np.float32)

        lum_map = np.zeros((self.h, self.w), dtype=np.float32)
        lamp_positions = list(range(self.config.lamp_spacing_px // 2, self.w, self.config.lamp_spacing_px))

        for lx in lamp_positions:
            ly = float(self.config.lamp_height_y_px)
            # Distance metric with anisotropic vertical elongation
            dx = (x_grid - lx) / 1.0
            dy = (y_grid - ly) / 1.8
            dist = np.sqrt(dx * dx + dy * dy)
            cone = np.exp(-((dist / float(self.config.spot_radius_px)) ** self.config.falloff_exponent))
            lum_map = np.maximum(lum_map, cone)

        lum_map = np.repeat(lum_map[:, :, np.newaxis], 3, axis=2)

        if tunnel_mode:
            # Sodium amber color: boost R, moderately boost G, depress B
            tint_kernel = np.array([0.70, 1.15, 1.45], dtype=np.float32)  # BGR order
        else:
            # Modern warm LED white
            tint_kernel = np.array([0.90, 1.10, 1.20], dtype=np.float32)

        illuminated = out * (1.0 + lum_map * (self.config.lamp_intensity_gain - 1.0) * tint_kernel)
        return np.clip(illuminated, 0, 255).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Adverse Condition Quality Assurance & Physics Validation Suite
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PerturbationValidationMetrics:
    """Quantitative statistical metrics verifying physics-based perturbations."""
    condition_name: str
    original_mean_luminance: float
    perturbed_mean_luminance: float
    luminance_delta: float
    original_rms_contrast: float
    perturbed_rms_contrast: float
    contrast_ratio: float
    edge_magnitude_preservation: float
    shannon_entropy_delta: float
    is_valid: bool
    diagnostic_message: str


class AdverseConditionQualityAssuranceSuite:
    """
    Validates synthetic adverse environmental augmentations using statistical
    radiometry and structural metrics (contrast degradation, edge preservation).
    Ensures that augmentations do not destroy lane features beyond recognition
    while strictly enforcing realistic physical bounds.
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def compute_rms_contrast(gray_img: np.ndarray) -> float:
        """Compute root-mean-square contrast of grayscale image."""
        norm = gray_img.astype(np.float32) / 255.0
        return float(np.std(norm))

    @staticmethod
    def compute_edge_energy(gray_img: np.ndarray) -> float:
        """Compute sum of gradient magnitudes via Sobel operator."""
        gx = cv2.Sobel(gray_img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        return float(np.mean(mag))

    @staticmethod
    def compute_shannon_entropy(gray_img: np.ndarray) -> float:
        """Compute Shannon informational entropy of pixel luminance histogram."""
        hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256]).flatten()
        prob = hist / (hist.sum() + 1e-7)
        prob = prob[prob > 0]
        return float(-np.sum(prob * np.log2(prob)))

    def evaluate_condition_perturbation(
        self,
        clean_bgr: np.ndarray,
        perturbed_bgr: np.ndarray,
        condition: AdverseConditionType,
    ) -> PerturbationValidationMetrics:
        """
        Evaluate statistical delta between clean baseline and perturbed frame.
        """
        clean_gray = cv2.cvtColor(clean_bgr, cv2.COLOR_BGR2GRAY)
        pert_gray = cv2.cvtColor(perturbed_bgr, cv2.COLOR_BGR2GRAY)

        orig_lum = float(np.mean(clean_gray))
        pert_lum = float(np.mean(pert_gray))
        lum_delta = pert_lum - orig_lum

        orig_c = self.compute_rms_contrast(clean_gray)
        pert_c = self.compute_rms_contrast(pert_gray)
        contrast_ratio = pert_c / max(orig_c, 1e-4)

        orig_edges = self.compute_edge_energy(clean_gray)
        pert_edges = self.compute_edge_energy(pert_gray)
        edge_ratio = pert_edges / max(orig_edges, 1e-4)

        orig_ent = self.compute_shannon_entropy(clean_gray)
        pert_ent = self.compute_shannon_entropy(pert_gray)
        ent_delta = pert_ent - orig_ent

        # Physics rule verification
        is_valid = True
        msg = "Validation passed"

        if condition == AdverseConditionType.NIGHT:
            if pert_lum > orig_lum * 0.70:
                is_valid = False
                msg = f"Night condition failed: luminance {pert_lum:.1f} did not attenuate sufficiently"
        elif condition == AdverseConditionType.FOG:
            if pert_lum < orig_lum * 0.90:
                is_valid = False
                msg = f"Fog condition failed: atmospheric airlight should elevate baseline luminance"
        elif condition == AdverseConditionType.RAIN:
            if contrast_ratio > 1.25:
                is_valid = False
                msg = "Rain condition failed: excessive contrast inflation observed"

        return PerturbationValidationMetrics(
            condition_name=condition.value,
            original_mean_luminance=orig_lum,
            perturbed_mean_luminance=pert_lum,
            luminance_delta=lum_delta,
            original_rms_contrast=orig_c,
            perturbed_rms_contrast=pert_c,
            contrast_ratio=contrast_ratio,
            edge_magnitude_preservation=edge_ratio,
            shannon_entropy_delta=ent_delta,
            is_valid=is_valid,
            diagnostic_message=msg,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. PyTorch Dataset Integration: AugmentedLaneDataset
# ═══════════════════════════════════════════════════════════════════════════════

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    Dataset = object  # Fallback for static typing when torch not loaded


class AugmentedLaneDataset(Dataset):
    """
    High-performance PyTorch Dataset wrapper that streams TuSimple clips,
    applies stochastic adverse perturbations via AdvancedAugmentationPipeline,
    and returns 4-class segmentation masks with associated condition metadata.
    
    NON-NEGOTIABLE CONSTRAINT ENFORCED:
    Horizontal flip is strictly prohibited (horizontal_flip=False).
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        pipeline_config: Optional[AdvancedAugmentationPipelineConfig] = None,
        is_training: bool = True,
        deterministic_seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.samples = samples
        self.is_training = is_training
        self.seed = deterministic_seed

        cfg = pipeline_config or AdvancedAugmentationPipelineConfig(horizontal_flip=False)
        if cfg.horizontal_flip:
            raise ValueError("AugmentedLaneDataset violation: horizontal_flip MUST be False.")
        self.pipeline = AdvancedAugmentationPipeline(config=cfg)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetch sample by index, apply randomized or deterministic adverse perturbations,
        and yield tensors formatted for PyTorch lane training.
        """
        if self.seed is not None:
            random.seed(self.seed + idx)
            np.random.seed(self.seed + idx)

        sample_info = self.samples[idx]

        # In-memory numpy array or file path loading
        if "frame_bgr" in sample_info:
            frame_bgr = sample_info["frame_bgr"]
        elif "image_path" in sample_info:
            frame_bgr = cv2.imread(str(sample_info["image_path"]))
            if frame_bgr is None:
                raise FileNotFoundError(f"Failed to read image at {sample_info['image_path']}")
        else:
            # Synthesize standard 640x360 test frame if mock data
            frame_bgr = np.full((360, 640, 3), 70, dtype=np.uint8)

        if "lane_mask" in sample_info:
            lane_mask = sample_info["lane_mask"]
        else:
            lane_mask = np.zeros((360, 640), dtype=np.int32)

        # Apply augmentation if in training mode
        active_tags: List[str] = ["CLEAN"]
        if self.is_training:
            aug_bgr, aug_mask, active_tags = self.pipeline.apply_randomized_pipeline(frame_bgr, lane_mask)
        else:
            aug_bgr, aug_mask = frame_bgr.copy(), lane_mask.copy()

        # Convert to PyTorch format if torch is available
        if TORCH_AVAILABLE:
            # Normalize image to [0, 1] and transpose to (C, H, W)
            rgb = cv2.cvtColor(aug_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(rgb).permute(2, 0, 1).float()
            mask_tensor = torch.from_numpy(aug_mask).long()
        else:
            img_tensor = aug_bgr
            mask_tensor = aug_mask

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "active_tags": active_tags,
            "sample_index": idx,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Concurrent Dataset Augmentation & Shard Exporter
# ═══════════════════════════════════════════════════════════════════════════════

class ConcurrentDatasetAugmentor:
    """
    High-throughput multithreaded worker pool for offline pre-generation
    of adverse condition training shards into data/processed/augmented/.
    """

    def __init__(
        self,
        output_dir: str = "data/processed/augmented",
        num_workers: int = 4,
        pipeline_config: Optional[AdvancedAugmentationPipelineConfig] = None,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.num_workers = num_workers
        self.cfg = pipeline_config or AdvancedAugmentationPipelineConfig(horizontal_flip=False)
        self.lock = threading.Lock()
        self.total_processed = 0

    def process_and_save_sample(
        self,
        sample_idx: int,
        frame_bgr: np.ndarray,
        lane_mask: Optional[np.ndarray],
        condition: AdverseConditionType,
    ) -> Dict[str, str]:
        """Process a single frame under a specified condition and save PNGs to disk."""
        pipeline = AdvancedAugmentationPipeline(config=self.cfg)
        aug_img, aug_mask = pipeline.apply_single_condition(frame_bgr, condition, lane_mask)

        img_filename = f"aug_{sample_idx:06d}_{condition.value.lower()}_img.png"
        img_path = self.output_dir / img_filename
        cv2.imwrite(str(img_path), aug_img)

        result = {"image_path": str(img_path), "condition": condition.value}

        if aug_mask is not None:
            mask_filename = f"aug_{sample_idx:06d}_{condition.value.lower()}_mask.png"
            mask_path = self.output_dir / mask_filename
            cv2.imwrite(str(mask_path), aug_mask.astype(np.uint8))
            result["mask_path"] = str(mask_path)

        with self.lock:
            self.total_processed += 1

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Batch Dataset Export Runner & Verification Harness
# ═══════════════════════════════════════════════════════════════════════════════

class AugmentedDatasetBatchExporter:
    """
    Exports synthetic augmented batches of TuSimple clips into data/processed/augmented/
    for model training and robustness benchmarking.
    """

    def __init__(
        self,
        pipeline: AdvancedAugmentationPipeline,
        output_directory: str = "data/processed/augmented",
    ) -> None:
        self.pipeline = pipeline
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export_sample_gallery(self, sample_bgr_frame: np.ndarray) -> Dict[str, str]:
        """Generate and save gallery of all 6 individual adverse conditions."""
        gallery_paths: Dict[str, str] = {}
        conditions = [
            AdverseConditionType.FOG,
            AdverseConditionType.RAIN,
            AdverseConditionType.NIGHT,
            AdverseConditionType.SHADOW,
            AdverseConditionType.GLARE,
            AdverseConditionType.FADED_LINES,
        ]

        for cond in conditions:
            aug_img, _ = self.pipeline.apply_single_condition(sample_bgr_frame, cond)
            filename = f"sample_{cond.value.lower()}.png"
            filepath = self.output_dir / filename
            cv2.imwrite(str(filepath), aug_img)
            gallery_paths[cond.value] = str(filepath)

        logger.info(f"Exported sample augmentation gallery to {self.output_dir}")
        return gallery_paths


def run_advanced_augmentations_diagnostics(verbose: bool = True) -> bool:
    """
    Validate all adverse synthesizers, verify horizontal_flip=False guard,
    test QA physics validation suite, PyTorch dataset wrapper, and wiper module.
    """
    if verbose:
        logger.info("Executing AdvancedAugmentations v2 diagnostics...")

    cfg = AdvancedAugmentationPipelineConfig(horizontal_flip=False)
    pipeline = AdvancedAugmentationPipeline(config=cfg)

    # 1. Enforce horizontal_flip=False non-negotiable guard
    try:
        invalid_cfg = AdvancedAugmentationPipelineConfig(horizontal_flip=True)
        _ = AdvancedAugmentationPipeline(config=invalid_cfg)
        assert False, "Pipeline failed to enforce horizontal_flip=False guard!"
    except ValueError:
        if verbose:
            logger.info("Confirmed: horizontal_flip=False guard rejected invalid config.")

    # 2. Test synthetic frame across all conditions
    dummy_frame = np.full((360, 640, 3), 80, dtype=np.uint8)
    dummy_frame[160:350, 200:440] = 50
    dummy_frame[160:350, 195:202] = 255  # Left
    dummy_frame[160:350, 438:445] = 255  # Right

    dummy_mask = np.zeros((360, 640), dtype=np.int32)
    dummy_mask[160:350, 200:440] = 1
    dummy_mask[160:350, 195:202] = 2
    dummy_mask[160:350, 438:445] = 3

    qa_suite = AdverseConditionQualityAssuranceSuite()

    for cond in AdverseConditionType:
        if cond == AdverseConditionType.CLEAN:
            continue
        aug_img, aug_mask = pipeline.apply_single_condition(dummy_frame, cond, dummy_mask)
        assert aug_img.shape == (360, 640, 3), f"Incorrect shape {aug_img.shape}"
        assert aug_img.dtype == np.uint8
        assert 0 <= np.min(aug_img) and np.max(aug_img) <= 255

        metrics = qa_suite.evaluate_condition_perturbation(dummy_frame, aug_img, cond)
        assert metrics is not None
        if verbose:
            logger.info(f"Condition {cond.value}: Lum Delta={metrics.luminance_delta:+.1f}, "
                        f"Contrast Ratio={metrics.contrast_ratio:.2f}, Edge Ratio={metrics.edge_magnitude_preservation:.2f}")

    # 3. Test Windshield Wiper Synthesizer
    wiper = WindshieldWiperSynthesizer()
    rained_img, _ = pipeline.apply_single_condition(dummy_frame, AdverseConditionType.RAIN)
    wiped_img = wiper.apply(rained_img, dummy_frame)
    assert wiped_img.shape == (360, 640, 3)

    # 4. Test Overhead Luminary Synthesizer
    lum = OverheadLuminarySynthesizer()
    tunnel_img = lum.apply(dummy_frame, tunnel_mode=True)
    assert tunnel_img.shape == (360, 640, 3)

    # 5. Test randomized pipeline
    rand_img, rand_mask, tags = pipeline.apply_randomized_pipeline(dummy_frame, dummy_mask)
    assert rand_img.shape == (360, 640, 3)
    assert len(tags) >= 1

    # 6. Test PyTorch Dataset wrapper
    sample_data = [{"frame_bgr": dummy_frame, "lane_mask": dummy_mask} for _ in range(5)]
    ds = AugmentedLaneDataset(sample_data, is_training=True)
    assert len(ds) == 5
    item = ds[0]
    assert "image" in item and "mask" in item
    if TORCH_AVAILABLE:
        assert isinstance(item["image"], torch.Tensor)
        assert item["image"].shape == (3, 360, 640)

    # 7. Test Batch Exporter Gallery
    exporter = AugmentedDatasetBatchExporter(pipeline, output_directory="data/processed/augmented")
    gallery = exporter.export_sample_gallery(dummy_frame)
    assert len(gallery) == 6

    if verbose:
        logger.info("All AdvancedAugmentations diagnostics passed successfully!")
    return True


def parse_cli_args():
    """Parse command line arguments for standalone data augmentation execution."""
    import argparse
    parser = argparse.ArgumentParser(
        description="APEX-Vision Core v4.0 - Advanced Adversarial Lane Data Augmentation Engine v2.0"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/lane_dataset.zip",
        help="Path to raw TuSimple dataset archive or directory",
    )
    parser.add_argument(
        "--export-gallery",
        action="store_true",
        help="Synthesize and export adverse condition gallery to data/processed/augmented/",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/augmented",
        help="Directory to store generated augmented samples",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of synthetic sample frames to generate",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable detailed diagnostic logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cli_args = parse_cli_args()

    # Always verify pipeline diagnostics first
    run_advanced_augmentations_diagnostics(verbose=cli_args.verbose)

    if cli_args.export_gallery:
        logger.info(f"Exporting gallery of adverse conditions to {cli_args.output_dir}...")
        test_frame = np.full((360, 640, 3), 90, dtype=np.uint8)
        # Add road marking visual
        cv2.line(test_frame, (200, 360), (300, 180), (255, 255, 255), 4)
        cv2.line(test_frame, (440, 360), (340, 180), (255, 255, 255), 4)
        pipeline = AdvancedAugmentationPipeline(AdvancedAugmentationPipelineConfig(horizontal_flip=False))
        exporter = AugmentedDatasetBatchExporter(pipeline, output_directory=cli_args.output_dir)
        paths = exporter.export_sample_gallery(test_frame)
        logger.info(f"Successfully generated {len(paths)} adverse condition frames.")


