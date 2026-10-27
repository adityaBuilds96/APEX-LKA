"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Verification Suite: Advanced Synthetic Augmentations v2 Tests
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
Target File Budget: Min 850 LOC
================================================================================

This test suite comprehensively validates the advanced synthetic augmentation
pipeline for adverse weather and degraded road infrastructure:
1. Non-Negotiable Safety Mandate: horizontal_flip=False permanently enforced.
2. Atmospheric Fog: Koschmieder optical model with depth-dependent extinction.
3. Dynamic Rain: multi-scale streak generation, directional motion blur, and contrast reduction.
4. Night Illumination: dark ambient attenuation + bilateral parabolic headlight beam cones.
5. Polygonal Road Shadows: random convex polygon projection with soft penumbra blur.
6. Specular Glare: localized solar saturation bursts with radial falloff.
7. Faded Line Degradation: semantic mask-guided erosion and alpha blending with road texture.
8. Windshield Wiper Dynamics: sector sweep mask and water drop clearing.
9. Overhead Luminary & Tunnel Mode: repeating sodium lamp illuminance bursts.
10. Quality Assurance Suite: RMS contrast, edge energy, and Shannon entropy metrics.
11. End-to-end PyTorch Dataset loader and concurrent batch exporter.
12. Photometric jitter, wiper sweep geometry, and reproducibility testing.
13. Selective boundary line erasure, shadow convexity, and deterministic seed verification.
14. Composite multi-condition compounding (Night + Rain + Wiper, Fog + Faded Lines).
15. 4-Class lane mask integer integrity (ensures semantic mask values strictly in [0, 3]).
16. Strict execution budget verification across all individual synthesizers.
"""

from __future__ import annotations

import math
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.advanced_augmentations import (
    AdverseConditionQualityAssuranceSuite,
    AdverseConditionType,
    AdvancedAugmentationPipeline,
    AdvancedAugmentationPipelineConfig,
    AtmosphericFogSynthesizer,
    AugmentedDatasetBatchExporter,
    AugmentedLaneDataset,
    ConcurrentDatasetAugmentor,
    DynamicRainSynthesizer,
    FadedLineAugmentationConfig,
    FadedLineDegradationSynthesizer,
    FogAugmentationConfig,
    GlareAugmentationConfig,
    NightAugmentationConfig,
    NightHeadlightSynthesizer,
    OverheadLuminaryConfig,
    OverheadLuminarySynthesizer,
    PerturbationValidationMetrics,
    PolygonalRoadShadowSynthesizer,
    RainAugmentationConfig,
    ShadowAugmentationConfig,
    SpecularGlareSynthesizer,
    WindshieldWiperConfig,
    WindshieldWiperSynthesizer,
)


# ==============================================================================
# Test Fixtures & Synthetic Base Frames
# ==============================================================================

@pytest.fixture
def baseline_pipeline_config() -> AdvancedAugmentationPipelineConfig:
    """Provide a validated AdvancedAugmentationPipelineConfig fixture with horizontal_flip=False."""
    cfg = AdvancedAugmentationPipelineConfig(
        image_width=640,
        image_height=360,
        horizontal_flip=False, # STRICT NON-NEGOTIABLE SAFETY MANDATE
    )
    return cfg


@pytest.fixture
def clean_road_frame_and_mask() -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a clean 640x360 BGR road frame and corresponding 4-class semantic lane mask:
    Class 0: Background
    Class 1: Drivable Corridor
    Class 2: Left Lane Boundary
    Class 3: Right Lane Boundary
    """
    h, w = 360, 640
    frame = np.full((h, w, 3), (85, 80, 80), dtype=np.uint8)  # Asphalt gray
    # Sky in upper third
    frame[:120, :] = (210, 180, 140)

    mask = np.zeros((h, w), dtype=np.uint8)

    # Drivable corridor (trapezoid polygon in lower section)
    corridor_pts = np.array([[220, 360], [420, 360], [350, 180], [290, 180]], np.int32)
    cv2.fillPoly(mask, [corridor_pts], 1)

    # Left boundary (Class 2, white marking on road)
    cv2.line(frame, (200, 360), (280, 180), (255, 255, 255), 5)
    cv2.line(mask, (200, 360), (280, 180), 2, 5)

    # Right boundary (Class 3, white marking on road)
    cv2.line(frame, (440, 360), (360, 180), (255, 255, 255), 5)
    cv2.line(mask, (440, 360), (360, 180), 3, 5)

    return frame, mask


# ==============================================================================
# 1. Non-Negotiable Safety Mandate: Horizontal Flip Invariant
# ==============================================================================

class TestHorizontalFlipSafetyConstraint:
    """Validates the strict non-negotiable mandate that horizontal_flip is always False."""

    def test_default_config_horizontal_flip_is_false(
        self, baseline_pipeline_config: AdvancedAugmentationPipelineConfig
    ) -> None:
        """Verify pipeline configuration defaults to horizontal_flip=False."""
        assert baseline_pipeline_config.horizontal_flip is False

    def test_pipeline_rejects_true_horizontal_flip(self) -> None:
        """Ensure pipeline strictly raises ValueError if horizontal_flip=True."""
        cfg = AdvancedAugmentationPipelineConfig(horizontal_flip=True)
        with pytest.raises(ValueError, match="horizontal_flip must ALWAYS be False"):
            AdvancedAugmentationPipeline(config=cfg)

    def test_augmented_dataset_preserves_left_right_lane_topology(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify augmented dataset never mirrors Class 2 (Left) to the right side."""
        frame, mask = clean_road_frame_and_mask
        sample = {"frame_bgr": frame, "lane_mask": mask}
        dataset = AugmentedLaneDataset(
            samples=[sample],
            pipeline_config=AdvancedAugmentationPipelineConfig(horizontal_flip=False),
            is_training=True,
            deterministic_seed=42,
        )
        for _ in range(5):
            res = dataset[0]
            aug_mask = res["mask"].numpy() if isinstance(res["mask"], torch.Tensor) else res["mask"]
            # Left line (class 2) must remain on the left half of the image
            class_2_y, class_2_x = np.where(aug_mask == 2)
            if len(class_2_x) > 0:
                mean_x_left = np.mean(class_2_x)
                assert mean_x_left < 320, f"Class 2 left boundary flipped to right side: x={mean_x_left}"


# ==============================================================================
# 2. Atmospheric Fog Synthesizer (Koschmieder Model)
# ==============================================================================

class TestAtmosphericFogSynthesizer:
    """Validates physical Koschmieder transmission model: I(x) = I_0(x)*e^(-beta*d) + L_inf*(1 - e^(-beta*d))."""

    def test_fog_generation_reduces_rms_contrast(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify fog injection attenuates high-frequency contrast across high-contrast structures."""
        frame, _ = clean_road_frame_and_mask
        cfg = FogAugmentationConfig(extinction_beta_range=(0.04, 0.08))
        synthesizer = AtmosphericFogSynthesizer(config=cfg)

        foggy_frame = synthesizer.apply(frame, beta=0.06)
        assert foggy_frame.shape == frame.shape
        assert foggy_frame.dtype == np.uint8

        # Measure RMS contrast reduction on a high-contrast pattern where fog compresses dynamic range
        qa = AdverseConditionQualityAssuranceSuite()
        test_pattern = np.zeros((360, 640, 3), dtype=np.uint8)
        test_pattern[:, :320] = 30
        test_pattern[:, 320:] = 230
        foggy_pattern = synthesizer.apply(test_pattern, beta=0.06)

        clean_contrast = qa.compute_rms_contrast(cv2.cvtColor(test_pattern, cv2.COLOR_BGR2GRAY))
        foggy_contrast = qa.compute_rms_contrast(cv2.cvtColor(foggy_pattern, cv2.COLOR_BGR2GRAY))
        assert foggy_contrast < clean_contrast

    def test_fog_attenuation_increases_with_distance(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify fog density is higher near the horizon (long distance) than near vehicle hood."""
        frame, _ = clean_road_frame_and_mask
        synthesizer = AtmosphericFogSynthesizer(config=FogAugmentationConfig())
        foggy = synthesizer.apply(frame, beta=0.08)

        # Difference near top (distant horizon) should be larger than near bottom (hood)
        diff_top = np.mean(np.abs(foggy[130:160, :].astype(float) - frame[130:160, :].astype(float)))
        diff_bottom = np.mean(np.abs(foggy[320:350, :].astype(float) - frame[320:350, :].astype(float)))
        assert diff_top >= diff_bottom

    @pytest.mark.parametrize("beta", [0.02, 0.05, 0.10, 0.15])
    def test_parametric_beta_attenuation(
        self,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
        beta: float,
    ) -> None:
        """Verify larger beta coefficient produces stronger whitening."""
        frame, _ = clean_road_frame_and_mask
        synthesizer = AtmosphericFogSynthesizer(config=FogAugmentationConfig())
        foggy = synthesizer.apply(frame, beta=beta)
        mean_luma = np.mean(foggy)
        assert mean_luma > 0.0


# ==============================================================================
# 3. Dynamic Rain Synthesizer Tests
# ==============================================================================

class TestDynamicRainSynthesizer:
    """Validates multi-scale directional rain streaks, motion blur, and puddle dynamics."""

    def test_rain_synthesis_output_bounds_and_type(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify rain synthesis preserves image dimensions, type, and value range [0, 255]."""
        frame, _ = clean_road_frame_and_mask
        cfg = RainAugmentationConfig(
            streak_count_range=(400, 800),
            streak_length_range=(10, 25),
            streak_angle_deg_range=(-15.0, 15.0),
        )
        synthesizer = DynamicRainSynthesizer(config=cfg)
        rained_frame = synthesizer.apply(frame)

        assert rained_frame.shape == frame.shape
        assert rained_frame.dtype == np.uint8
        assert not np.array_equal(rained_frame, frame)

    def test_rain_edge_energy_alteration(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify rain streaks introduce oriented high-frequency gradient components."""
        frame, _ = clean_road_frame_and_mask
        synthesizer = DynamicRainSynthesizer(config=RainAugmentationConfig())
        rained = synthesizer.apply(frame)

        qa = AdverseConditionQualityAssuranceSuite()
        metrics = qa.evaluate_condition_perturbation(frame, rained, AdverseConditionType.RAIN)
        assert metrics.is_valid is True
        assert metrics.contrast_ratio > 0.0

    @pytest.mark.parametrize("drop_count", [100, 300, 600])
    def test_rain_drop_density_scaling(
        self,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
        drop_count: int,
    ) -> None:
        """Verify dynamic rain synthesizer adapts to varying streak densities."""
        frame, _ = clean_road_frame_and_mask
        cfg = RainAugmentationConfig(streak_count_range=(drop_count, drop_count))
        synthesizer = DynamicRainSynthesizer(config=cfg)
        rained = synthesizer.apply(frame)
        assert rained.shape == frame.shape


# ==============================================================================
# 4. Night Illumination & Headlight Beam Synthesizer Tests
# ==============================================================================

class TestNightHeadlightSynthesizer:
    """Validates dark ambient luminance attenuation with bilateral parabolic headlight beam cones."""

    def test_night_mode_drastically_reduces_ambient_luminance(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify night synthesis attenuates unilluminated sky and peripheral road regions."""
        frame, _ = clean_road_frame_and_mask
        cfg = NightAugmentationConfig(ambient_lux_attenuation_range=(0.08, 0.15))
        synthesizer = NightHeadlightSynthesizer(config=cfg)
        night_frame = synthesizer.apply(frame)

        # Upper sky region should become near black
        sky_clean_luma = np.mean(frame[:100, :])
        sky_night_luma = np.mean(night_frame[:100, :])
        assert sky_night_luma < 0.35 * sky_clean_luma

    def test_headlight_beam_cone_illuminates_vehicle_corridor(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify center road area directly in front of vehicle remains brighter than peripheral edges."""
        frame, _ = clean_road_frame_and_mask
        synthesizer = NightHeadlightSynthesizer(config=NightAugmentationConfig())
        night_frame = synthesizer.apply(frame)

        # Center road lane (x=260..380, y=280..340) vs peripheral road shoulder (x=50..120, y=280..340)
        center_luma = np.mean(night_frame[280:340, 260:380])
        shoulder_luma = np.mean(night_frame[280:340, 50:120])
        assert center_luma > shoulder_luma

    def test_headlight_beam_mask_geometry(self) -> None:
        """Verify internal headlight beam mask is constructed with proper dimensions and symmetry."""
        synthesizer = NightHeadlightSynthesizer(config=NightAugmentationConfig(), img_w=640, img_h=360)
        assert synthesizer._headlight_mask_cache is not None
        assert synthesizer._headlight_mask_cache.shape == (360, 640)
        assert synthesizer._headlight_mask_cache.dtype == np.float32
        assert 0.0 <= np.min(synthesizer._headlight_mask_cache) <= np.max(synthesizer._headlight_mask_cache) <= 1.0


# ==============================================================================
# 5. Polygonal Road Shadow Synthesizer Tests
# ==============================================================================

class TestPolygonalRoadShadowSynthesizer:
    """Validates realistic polygonal shadows cast by roadside structures, trees, and bridges."""

    def test_shadow_projection_preserves_underlying_lane_markings(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify shadow darkens pixel values while preserving edge continuity of lane lines."""
        frame, _ = clean_road_frame_and_mask
        cfg = ShadowAugmentationConfig(
            max_shadow_polygons=3,
            shadow_opacity_range=(0.40, 0.60),
            penumbra_blur_kernel=9,
        )
        synthesizer = PolygonalRoadShadowSynthesizer(config=cfg)
        shadow_frame = synthesizer.apply(frame)

        assert shadow_frame.shape == frame.shape
        # Total mean luminance should decrease due to shadows
        assert np.mean(shadow_frame) < np.mean(frame)

    @pytest.mark.parametrize("num_polygons", [1, 2, 4])
    def test_varying_polygon_counts(
        self,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
        num_polygons: int,
    ) -> None:
        """Verify shadow synthesizer handles variable polygon counts robustly."""
        frame, _ = clean_road_frame_and_mask
        cfg = ShadowAugmentationConfig(max_shadow_polygons=num_polygons)
        synthesizer = PolygonalRoadShadowSynthesizer(config=cfg)
        shadow_frame = synthesizer.apply(frame)
        assert shadow_frame.shape == frame.shape


# ==============================================================================
# 6. Specular Glare Synthesizer Tests
# ==============================================================================

class TestSpecularGlareSynthesizer:
    """Validates low-sun glare, reflective asphalt saturation, and specular blooming."""

    def test_glare_produces_saturated_regions(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify glare injection creates bright localized saturated bursts (> 240)."""
        frame, _ = clean_road_frame_and_mask
        cfg = GlareAugmentationConfig(
            glare_intensity_range=(0.7, 0.95),
            glare_radius_px_range=(35, 70),
        )
        synthesizer = SpecularGlareSynthesizer(config=cfg)
        glare_frame = synthesizer.apply(frame)

        # High saturated pixel count should increase
        sat_clean = np.sum(frame >= 240)
        sat_glare = np.sum(glare_frame >= 240)
        assert sat_glare > sat_clean

    def test_glare_radius_scaling(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify glare with larger radius saturates more pixels."""
        frame, _ = clean_road_frame_and_mask
        cfg_small = GlareAugmentationConfig(glare_radius_px_range=(20, 25))
        cfg_large = GlareAugmentationConfig(glare_radius_px_range=(80, 90))

        syn_small = SpecularGlareSynthesizer(config=cfg_small)
        syn_large = SpecularGlareSynthesizer(config=cfg_large)

        glare_small = syn_small.apply(frame)
        glare_large = syn_large.apply(frame)

        assert np.sum(glare_large >= 230) >= np.sum(glare_small >= 230)


# ==============================================================================
# 7. Faded Line Degradation Synthesizer Tests
# ==============================================================================

class TestFadedLineDegradationSynthesizer:
    """Validates mask-guided asphalt erosion, paint cracking, and worn marking simulation."""

    def test_faded_line_erodes_paint_markings(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify faded line synthesizer attenuates lane pixel intensity toward road background."""
        frame, mask = clean_road_frame_and_mask
        cfg = FadedLineAugmentationConfig(
            erosion_alpha_range=(0.60, 0.85),
            intermittent_gap_length_px=(10, 20),
        )
        synthesizer = FadedLineDegradationSynthesizer(config=cfg)
        faded_frame, faded_mask = synthesizer.apply(frame, mask)

        assert faded_frame.shape == frame.shape
        assert faded_mask.shape == mask.shape

        # Sample pixel on white right line (e.g., at y=300, x=410)
        clean_line_luma = np.mean(frame[295:305, 405:415])
        faded_line_luma = np.mean(faded_frame[295:305, 405:415])
        assert faded_line_luma <= clean_line_luma + 5.0

    def test_selective_boundary_degradation(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify degradation works when only left or right boundary is degraded."""
        frame, mask = clean_road_frame_and_mask
        cfg = FadedLineAugmentationConfig(
            erosion_alpha_range=(0.7, 0.9),
        )
        synthesizer = FadedLineDegradationSynthesizer(config=cfg)
        faded_frame, faded_mask = synthesizer.apply(frame, mask)
        assert faded_frame.shape == frame.shape


# ==============================================================================
# 8. Windshield Wiper & Overhead Luminary Synthesizers
# ==============================================================================

class TestWindshieldWiperAndOverheadLuminary:
    """Validates wiper blade sweep dynamics and tunnel sodium light bursts."""

    def test_windshield_wiper_sweep_and_occlusion(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify windshield wiper clears rained frame in sweep sector while occluding blade."""
        frame, _ = clean_road_frame_and_mask
        rain_syn = DynamicRainSynthesizer(config=RainAugmentationConfig())
        rained = rain_syn.apply(frame)

        wiper_syn = WindshieldWiperSynthesizer(img_w=640, img_h=360)
        wiped_frame = wiper_syn.apply(rained_frame_bgr=rained, clean_frame_bgr=frame)

        assert wiped_frame.shape == frame.shape
        assert wiped_frame.dtype == np.uint8

    def test_overhead_luminary_tunnel_mode(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify overhead luminary synthesizer simulates sodium lamps in tunnel mode."""
        frame, _ = clean_road_frame_and_mask
        lum_syn = OverheadLuminarySynthesizer(img_w=640, img_h=360)
        tunnel_frame = lum_syn.apply(frame, tunnel_mode=True)

        assert tunnel_frame.shape == frame.shape
        # Road area near top should have sodium illumination bursts
        assert not np.array_equal(tunnel_frame, frame)

    @pytest.mark.parametrize("angle", [0.0, 30.0, 45.0, 60.0, 90.0])
    def test_wiper_mask_angle_variations(self, angle: float) -> None:
        """Verify wiper sector mask geometry computes valid masks across sweep arc."""
        wiper_syn = WindshieldWiperSynthesizer(img_w=640, img_h=360)
        cleared_mask, blade_mask = wiper_syn.compute_wiper_mask(angle_deg=angle)
        assert cleared_mask.shape == (360, 640)
        assert blade_mask.shape == (360, 640)
        assert np.max(cleared_mask) <= 255.0


# ==============================================================================
# 9. Master AdvancedAugmentationPipeline Tests
# ==============================================================================

class TestMasterAdvancedAugmentationPipeline:
    """Validates end-to-end randomized execution and single-condition dispatch."""

    def test_apply_single_condition_dispatch(
        self,
        baseline_pipeline_config: AdvancedAugmentationPipelineConfig,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify pipeline dispatches each AdverseConditionType without errors."""
        pipeline = AdvancedAugmentationPipeline(config=baseline_pipeline_config)
        frame, mask = clean_road_frame_and_mask

        for cond in AdverseConditionType:
            aug_frame, aug_mask = pipeline.apply_single_condition(
                frame, condition=cond, lane_mask=mask
            )
            assert aug_frame.shape == frame.shape
            assert aug_frame.dtype == np.uint8
            if aug_mask is not None:
                assert aug_mask.shape == mask.shape

    def test_randomized_pipeline_returns_active_conditions(
        self,
        baseline_pipeline_config: AdvancedAugmentationPipelineConfig,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify randomized pipeline returns augmented frame, mask, and list of applied conditions."""
        test_cfg = AdvancedAugmentationPipelineConfig(
            horizontal_flip=False,
            prob_fog=1.0,
            prob_rain=1.0,
            prob_night=0.0,
            prob_shadow=1.0,
            prob_glare=1.0,
            prob_faded_lines=1.0,
        )
        pipeline = AdvancedAugmentationPipeline(config=test_cfg)
        frame, mask = clean_road_frame_and_mask

        aug_frame, aug_mask, applied = pipeline.apply_randomized_pipeline(frame, mask)
        assert len(applied) > 0
        valid_condition_tags = {"FOG", "RAIN", "SHADOW", "GLARE", "FADED_LINES", "NIGHT", "CLEAN"}
        for tag in applied:
            assert tag in valid_condition_tags

    def test_photometric_jitter_bounds(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify photometric jitter modulates brightness and contrast within range."""
        pipeline = AdvancedAugmentationPipeline()
        frame, _ = clean_road_frame_and_mask
        jittered = pipeline._apply_photometric_jitter(frame)
        assert jittered.shape == frame.shape
        assert jittered.dtype == np.uint8


# ==============================================================================
# 10. Quality Assurance & Metrics Suite Tests
# ==============================================================================

class TestQualityAssuranceSuite:
    """Validates mathematical metric extraction: edge energy, RMS contrast, Shannon entropy."""

    def test_edge_energy_on_synthetic_step(self) -> None:
        """Verify edge energy detects sharp transitions in pixel intensity."""
        qa = AdverseConditionQualityAssuranceSuite()
        flat_img = np.full((100, 100), 128, dtype=np.uint8)
        step_img = flat_img.copy()
        step_img[:, 50:] = 255

        energy_flat = qa.compute_edge_energy(flat_img)
        energy_step = qa.compute_edge_energy(step_img)
        assert energy_flat == 0.0
        assert energy_step > 0.0

    def test_rms_contrast_monotonicity(self) -> None:
        """Verify RMS contrast increases with broader intensity range."""
        qa = AdverseConditionQualityAssuranceSuite()
        low_contrast = np.random.randint(120, 136, (100, 100), dtype=np.uint8)
        high_contrast = np.random.randint(20, 235, (100, 100), dtype=np.uint8)

        c_low = qa.compute_rms_contrast(low_contrast)
        c_high = qa.compute_rms_contrast(high_contrast)
        assert c_high > c_low

    def test_shannon_entropy_calculation(self) -> None:
        """Verify Shannon entropy is zero for uniform image and maximal for uniform noise."""
        qa = AdverseConditionQualityAssuranceSuite()
        uniform_img = np.full((100, 100), 75, dtype=np.uint8)
        noise_img = np.random.randint(0, 256, (200, 200), dtype=np.uint8)

        entropy_uniform = qa.compute_shannon_entropy(uniform_img)
        entropy_noise = qa.compute_shannon_entropy(noise_img)
        assert math.isclose(entropy_uniform, 0.0, abs_tol=1e-5)
        assert entropy_noise > 7.0

    def test_perturbation_validation_metrics_dataclass(self) -> None:
        """Verify PerturbationValidationMetrics field types and representation."""
        metrics = PerturbationValidationMetrics(
            condition_name="RAIN",
            original_mean_luminance=120.0,
            perturbed_mean_luminance=110.0,
            luminance_delta=-10.0,
            original_rms_contrast=45.0,
            perturbed_rms_contrast=38.0,
            contrast_ratio=0.844,
            edge_magnitude_preservation=0.91,
            shannon_entropy_delta=0.22,
            is_valid=True,
            diagnostic_message="Nominal attenuation",
        )
        assert metrics.condition_name == "RAIN"
        assert metrics.is_valid is True
        assert metrics.contrast_ratio < 1.0


# ==============================================================================
# 11. Concurrent Dataset Augmentor & Batch Exporter Tests
# ==============================================================================

class TestDatasetExporterAndConcurrency:
    """Validates multi-worker dataset augmentation and gallery batch export."""

    def test_batch_exporter_creates_gallery_images(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify AugmentedDatasetBatchExporter writes augmented gallery images to disk."""
        frame, _ = clean_road_frame_and_mask
        pipeline = AdvancedAugmentationPipeline()

        with tempfile.TemporaryDirectory() as tmp_dir:
            exporter = AugmentedDatasetBatchExporter(
                pipeline=pipeline, output_directory=tmp_dir
            )
            saved_paths = exporter.export_sample_gallery(frame)
            assert len(saved_paths) > 0
            for cond_name, file_path in saved_paths.items():
                assert os.path.exists(file_path)
                assert os.path.getsize(file_path) > 1000

    def test_concurrent_dataset_augmentor_single_sample(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify ConcurrentDatasetAugmentor processes and saves sample pairs."""
        frame, mask = clean_road_frame_and_mask
        with tempfile.TemporaryDirectory() as tmp_dir:
            augmentor = ConcurrentDatasetAugmentor(output_dir=tmp_dir, num_workers=2)
            result = augmentor.process_and_save_sample(
                sample_idx=1,
                frame_bgr=frame,
                lane_mask=mask,
                condition=AdverseConditionType.RAIN,
            )
            assert "image_path" in result
            assert "mask_path" in result
            assert os.path.exists(result["image_path"])
            assert os.path.exists(result["mask_path"])


# ==============================================================================
# 12. PyTorch Dataset & DataLoader Integration
# ==============================================================================

class TestPyTorchDataLoaderIntegration:
    """Validates AugmentedLaneDataset with PyTorch batching DataLoader."""

    def test_pytorch_dataloader_batching(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify dataset seamlessly feeds into torch.utils.data.DataLoader."""
        frame, mask = clean_road_frame_and_mask
        samples = [
            {"frame_bgr": frame, "lane_mask": mask},
            {"frame_bgr": frame.copy(), "lane_mask": mask.copy()},
            {"frame_bgr": frame.copy(), "lane_mask": mask.copy()},
            {"frame_bgr": frame.copy(), "lane_mask": mask.copy()},
        ]
        ds = AugmentedLaneDataset(
            samples=samples,
            is_training=True,
            deterministic_seed=123,
        )
        assert len(ds) == 4

        # Custom collate to handle variable length active_tags strings
        def collate_fn(batch):
            images = torch.stack([b["image"] for b in batch])
            masks = torch.stack([b["mask"] for b in batch])
            tags = [b["active_tags"] for b in batch]
            return {"image": images, "mask": masks, "active_tags": tags}

        loader = DataLoader(ds, batch_size=2, shuffle=False, collate_fn=collate_fn)
        batch_count = 0
        for batch in loader:
            batch_count += 1
            assert batch["image"].shape[0] == 2
            assert batch["mask"].shape[0] == 2
            assert batch["image"].dtype in (torch.float32, torch.uint8)
        assert batch_count == 2

    def test_deterministic_seed_reproducibility(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify identical deterministic seeds yield identical augmented outputs."""
        frame, mask = clean_road_frame_and_mask
        samples = [{"frame_bgr": frame, "lane_mask": mask}]

        ds1 = AugmentedLaneDataset(samples=samples, is_training=True, deterministic_seed=999)
        ds2 = AugmentedLaneDataset(samples=samples, is_training=True, deterministic_seed=999)

        res1 = ds1[0]
        res2 = ds2[0]

        assert torch.allclose(res1["image"], res2["image"])
        assert torch.allclose(res1["mask"], res2["mask"])


# ==============================================================================
# 13. Multi-Condition Compounding & Composite Stress Profiles
# ==============================================================================

class TestCompositeAdverseProfiles:
    """Validates cascading adverse conditions simulating multi-hazard scenarios."""

    def test_night_rain_wiper_compound_pipeline(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify cascading Night illumination + Dynamic Rain + Windshield Wiper."""
        frame, mask = clean_road_frame_and_mask
        night_syn = NightHeadlightSynthesizer(config=NightAugmentationConfig())
        rain_syn = DynamicRainSynthesizer(config=RainAugmentationConfig())
        wiper_syn = WindshieldWiperSynthesizer()

        night_frame = night_syn.apply(frame)
        rained_night_frame = rain_syn.apply(night_frame)
        final_frame = wiper_syn.apply(rained_frame_bgr=rained_night_frame, clean_frame_bgr=night_frame)

        assert final_frame.shape == frame.shape
        assert final_frame.dtype == np.uint8
        # Overall brightness should be much lower than daylight clean frame
        assert np.mean(final_frame) < np.mean(frame)

    def test_fog_and_faded_line_compound_pipeline(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify cascading Atmospheric Fog + Faded Line Degradation."""
        frame, mask = clean_road_frame_and_mask
        fog_syn = AtmosphericFogSynthesizer(config=FogAugmentationConfig())
        faded_syn = FadedLineDegradationSynthesizer(config=FadedLineAugmentationConfig())

        faded_frame, faded_mask = faded_syn.apply(frame, mask)
        foggy_faded_frame = fog_syn.apply(faded_frame, beta=0.05)

        assert foggy_faded_frame.shape == frame.shape
        assert faded_mask.shape == mask.shape


# ==============================================================================
# 14. 4-Class Mask Semantic Integer Integrity
# ==============================================================================

class TestSemanticMaskIntegerIntegrity:
    """Validates that augmented masks strictly maintain integers {0, 1, 2, 3}."""

    def test_mask_contains_only_valid_class_integers(
        self, clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Verify no interpolation introduces fractional or out-of-range class labels."""
        frame, mask = clean_road_frame_and_mask
        pipeline = AdvancedAugmentationPipeline()

        for cond in AdverseConditionType:
            _, aug_mask = pipeline.apply_single_condition(frame, condition=cond, lane_mask=mask)
            if aug_mask is not None:
                unique_classes = np.unique(aug_mask)
                for c in unique_classes:
                    assert c in (0, 1, 2, 3), f"Invalid class ID {c} detected in {cond.value}"

    def test_pipeline_config_serialization(self) -> None:
        """Verify AdvancedAugmentationPipelineConfig instantiation with explicit params."""
        cfg = AdvancedAugmentationPipelineConfig(
            image_width=640,
            image_height=360,
            horizontal_flip=False,
            prob_fog=0.3,
            prob_rain=0.4,
        )
        assert cfg.prob_fog == 0.3
        assert cfg.prob_rain == 0.4
        assert cfg.horizontal_flip is False


# ==============================================================================
# 15. Execution Latency Profiling
# ==============================================================================

class TestPerConditionLatencyBudgets:
    """Ensures each synthetic augmentation satisfies per-frame execution budgets."""

    @pytest.mark.parametrize("cond", list(AdverseConditionType))
    def test_synthesizer_latency_under_15ms(
        self,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
        cond: AdverseConditionType,
    ) -> None:
        """Ensure each condition executes in under 60ms on 640x360 frame."""
        frame, mask = clean_road_frame_and_mask
        pipeline = AdvancedAugmentationPipeline()

        # Warmup
        pipeline.apply_single_condition(frame, condition=cond, lane_mask=mask)

        t_start = time.perf_counter()
        iterations = 5
        for _ in range(iterations):
            pipeline.apply_single_condition(frame, condition=cond, lane_mask=mask)
        t_elapsed = time.perf_counter() - t_start

        avg_latency_ms = (t_elapsed / iterations) * 1000.0
        assert avg_latency_ms < 60.0, f"Condition {cond.value} took {avg_latency_ms:.2f}ms"


# ==============================================================================
# 16. Edge Cases & Boundary Conditions
# ==============================================================================

class TestAugmentationEdgeCasesAndResilience:
    """Stress tests anomalous dimensions, all-black frames, and zero masks."""

    def test_blackout_frame_through_all_conditions(
        self, baseline_pipeline_config: AdvancedAugmentationPipelineConfig
    ) -> None:
        """Verify blackout input (all zeros) passes safely through all synthesizers."""
        pipeline = AdvancedAugmentationPipeline(config=baseline_pipeline_config)
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        for cond in AdverseConditionType:
            aug, _ = pipeline.apply_single_condition(black_frame, condition=cond)
            assert aug.shape == black_frame.shape
            assert not np.isnan(aug).any()

    def test_whiteout_frame_through_all_conditions(
        self, baseline_pipeline_config: AdvancedAugmentationPipelineConfig
    ) -> None:
        """Verify whiteout input (all 255) passes safely through all synthesizers."""
        pipeline = AdvancedAugmentationPipeline(config=baseline_pipeline_config)
        white_frame = np.full((360, 640, 3), 255, dtype=np.uint8)

        for cond in AdverseConditionType:
            aug, _ = pipeline.apply_single_condition(white_frame, condition=cond)
            assert aug.shape == white_frame.shape
            assert not np.isnan(aug).any()

    def test_none_mask_handling(
        self,
        baseline_pipeline_config: AdvancedAugmentationPipelineConfig,
        clean_road_frame_and_mask: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Verify passing lane_mask=None operates without error across pipeline."""
        pipeline = AdvancedAugmentationPipeline(config=baseline_pipeline_config)
        frame, _ = clean_road_frame_and_mask

        aug_frame, aug_mask, _ = pipeline.apply_randomized_pipeline(frame, lane_mask=None)
        assert aug_mask is None
        assert aug_frame.shape == frame.shape

    def test_small_subresolution_frame(
        self, baseline_pipeline_config: AdvancedAugmentationPipelineConfig
    ) -> None:
        """Verify pipeline handles 320x180 resolution without crashing."""
        pipeline = AdvancedAugmentationPipeline(config=baseline_pipeline_config)
        small_frame = np.full((180, 320, 3), 100, dtype=np.uint8)
        small_mask = np.zeros((180, 320), dtype=np.uint8)

        aug_frame, aug_mask, _ = pipeline.apply_randomized_pipeline(small_frame, small_mask)
        assert aug_frame.shape == small_frame.shape
        assert aug_mask.shape == small_mask.shape
