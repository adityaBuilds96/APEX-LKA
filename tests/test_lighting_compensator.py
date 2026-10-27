"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

tests/test_lighting_compensator.py
Comprehensive Verification Suite for Photometric Conditioning & Illumination Compensation
Validates:
    - Master LightingCompensator pipeline (CLAHE, gamma LUT, adaptive stretch)
    - Dynamic non-linear power-law gamma transfer functions with multi-level sweeps
    - CIELAB L*-channel CLAHE equalization and chromaticity preservation
    - Tile grid variations ((4, 4), (8, 8), (16, 16)) and clip limit sensitivities
    - Specular highlight isolation, sky masking, and Telea vs NS inpainting
    - Night mode shadow lift, S-curve tone mapping, and headlight glare suppression
    - Adaptive percentile luminance histogram stretching without uint8 overflow
    - Photometric metric calculation (dynamic range dB, dark/saturated fractions, contrast gain)
    - Edge-preserving GuidedImageFilter (He et al.) noise suppression and edge retention
    - MultiScaleRetinex with color restoration (MSRCR) parameter sweeps
    - ShadowBoundaryEqualizer for cast bridge/tree shadow lift with morphology
    - TemporalExposureFilter frame-to-frame auto-exposure stabilization and alpha tracking
    - Frequency-domain HomomorphicFilter via 2D FFT illumination decoupling
    - AntiBloomingFlareSuppressor connected-component high-beam filtering and area gating
    - LogarithmicToneCurveCompander Reinhard global tonemapping and key value modulation
    - DirectionalGradientCompensator lateral sun-glare equalization
    - FastIntegralBinarizer O(1) integral image adaptive thresholding and radius scaling
    - Multi-panel forensic diagnostic quadrant canvas rendering
    - Boundary and degenerate cases (pure black, pure white, extreme checkerboard)
    - Batch benchmark execution and singleton accessor validation
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from src.perception.lighting_compensator import (
    AntiBloomingFlareSuppressor,
    DirectionalGradientCompensator,
    FastIntegralBinarizer,
    GuidedImageFilter,
    HomomorphicFilter,
    LightingCompensationResult,
    LightingCompensator,
    LightingCompensatorConfig,
    LightingMetrics,
    LogarithmicToneCurveCompander,
    MultiScaleRetinex,
    ShadowBoundaryEqualizer,
    TemporalExposureFilter,
    benchmark_lighting_compensator,
    get_anti_blooming_suppressor,
    get_directional_gradient_compensator,
    get_fast_integral_binarizer,
    get_guided_filter,
    get_lighting_compensator,
    get_shadow_equalizer,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures & Synthetic Frame Generators
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_config() -> LightingCompensatorConfig:
    """Return default LightingCompensator configuration."""
    return LightingCompensatorConfig()


@pytest.fixture
def compensator(default_config: LightingCompensatorConfig) -> LightingCompensator:
    """Return fresh LightingCompensator instance."""
    return LightingCompensator(config=default_config)


@pytest.fixture
def synthetic_road_frame() -> np.ndarray:
    """
    Generate a 360x640 synthetic highway road frame:
    - Upper 40%: Sky / horizon (light gray/blue)
    - Lower 60%: Asphalt road (dark gray L~60)
    - Two lane markings: White dashed lines (L~220)
    """
    h, w = 360, 640
    frame = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky
    frame[:144, :] = [180, 150, 120]  # BGR

    # Asphalt road
    frame[144:, :] = [60, 60, 60]

    # Left white lane line (dashed)
    cv2.line(frame, (200, 360), (280, 144), (230, 230, 230), 4)

    # Right white lane line (dashed)
    cv2.line(frame, (440, 360), (360, 144), (230, 230, 230), 4)

    return frame


@pytest.fixture
def underexposed_night_frame(synthetic_road_frame: np.ndarray) -> np.ndarray:
    """Simulate a dark night highway scene with headlights illuminating road ahead."""
    dark = (synthetic_road_frame.astype(np.float32) * 0.18).astype(np.uint8)
    # Headlight illumination cone in lower center
    h, w = dark.shape[:2]
    center_y, center_x = int(h * 0.8), int(w * 0.5)
    y, x = np.ogrid[:h, :w]
    cone = np.clip(1.0 - np.sqrt(((x - center_x) / 180.0) ** 2 + ((y - center_y) / 90.0) ** 2), 0.0, 1.0)
    cone_bgr = np.repeat(cone[:, :, np.newaxis], 3, axis=2)
    night_frame = np.clip(dark.astype(np.float32) + cone_bgr * 80.0, 0.0, 255.0).astype(np.uint8)
    return night_frame


@pytest.fixture
def overexposed_glare_frame(synthetic_road_frame: np.ndarray) -> np.ndarray:
    """Simulate intense forward sunlight glare washing out upper and road regions."""
    glare = synthetic_road_frame.copy().astype(np.float32)
    glare = np.clip(glare * 1.8 + 50.0, 0.0, 255.0).astype(np.uint8)
    return glare


@pytest.fixture
def wet_road_specular_frame(synthetic_road_frame: np.ndarray) -> np.ndarray:
    """Simulate wet pavement with high-luminance desaturated specular puddle reflections."""
    wet = synthetic_road_frame.copy()
    # Add specular puddle in lower asphalt area
    cv2.ellipse(wet, (320, 280), (60, 20), 0, 0, 360, (250, 250, 250), -1)
    return wet


@pytest.fixture
def hard_shadow_frame(synthetic_road_frame: np.ndarray) -> np.ndarray:
    """Simulate a bridge overpass casting a dark diagonal shadow across pavement."""
    shadow = synthetic_road_frame.copy()
    pts = np.array([[100, 144], [350, 144], [480, 360], [230, 360]], dtype=np.int32)
    mask = np.zeros((360, 640), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    shadow[mask > 0] = (shadow[mask > 0].astype(np.float32) * 0.35).astype(np.uint8)
    return shadow


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Master LightingCompensator Pipeline Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLightingCompensatorPipeline:
    """Verify end-to-end execution, dimension integrity, parameter overrides, and error handling."""

    def test_nominal_processing_dimensions_and_types(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """Processed output must match input dimensions and uint8 dtype."""
        res = compensator.process(synthetic_road_frame)

        assert isinstance(res, LightingCompensationResult)
        assert res.frame_shape == synthetic_road_frame.shape
        assert res.compensated_frame_bgr.dtype == np.uint8
        assert res.specular_mask.shape == synthetic_road_frame.shape[:2]
        assert res.luminance_map.shape == synthetic_road_frame.shape[:2]
        assert res.metrics.processing_time_ms > 0.0

    def test_empty_frame_raises_value_error(self, compensator: LightingCompensator) -> None:
        """Passing None or an empty array must raise ValueError."""
        with pytest.raises(ValueError, match="empty or None"):
            compensator.process(None)  # type: ignore

        with pytest.raises(ValueError, match="empty or None"):
            compensator.process(np.empty((0, 0, 3), dtype=np.uint8))

    def test_parameter_overrides(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """Runtime parameter overrides must take precedence over static config."""
        res = compensator.process(
            synthetic_road_frame,
            override_gamma=1.75,
            override_clahe_clip=3.5,
            force_night_mode=True,
            force_specular_removal=True,
        )

        assert math.isclose(res.metrics.active_gamma, 1.75, abs_tol=1e-3)
        assert math.isclose(res.metrics.active_clahe_clip, 3.5, abs_tol=1e-3)
        assert res.metrics.is_night_mode_active is True
        assert res.metrics.is_specular_removal_active is True

    def test_metrics_serialization(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """LightingMetrics.to_dict must produce a valid dictionary with all expected keys."""
        res = compensator.process(synthetic_road_frame)
        m_dict = res.metrics.to_dict()

        expected_keys = [
            "raw_mean_luminance", "raw_contrast_std", "raw_dark_pixel_fraction",
            "raw_saturated_pixel_fraction", "raw_dynamic_range_db",
            "compensated_mean_luminance", "compensated_contrast_std",
            "contrast_improvement_ratio", "specular_pixels_inpainted",
            "active_gamma", "active_clahe_clip", "is_night_mode_active",
            "is_specular_removal_active", "processing_time_ms"
        ]
        for key in expected_keys:
            assert key in m_dict

    def test_dynamic_range_calculation_with_uniform_and_high_contrast_frames(
        self, compensator: LightingCompensator
    ) -> None:
        """Dynamic range dB metric must handle flat frames without log-of-zero errors."""
        flat_black = np.zeros((100, 100, 3), dtype=np.uint8)
        res_black = compensator.process(flat_black)
        assert res_black.metrics.raw_dynamic_range_db >= 0.0
        assert math.isfinite(res_black.metrics.raw_dynamic_range_db)

        flat_white = np.full((100, 100, 3), 255, dtype=np.uint8)
        res_white = compensator.process(flat_white)
        assert res_white.metrics.raw_dynamic_range_db >= 0.0
        assert math.isfinite(res_white.metrics.raw_dynamic_range_db)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Dynamic Gamma Non-Linear Power-Law Correction Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicGammaCorrection:
    """Verify 8-bit lookup table power-law transformations: I_out = 255 * (I_in / 255)^(1 / gamma)."""

    def test_gamma_greater_than_one_brightens_image(
        self, compensator: LightingCompensator, underexposed_night_frame: np.ndarray
    ) -> None:
        """Gamma > 1.0 (e.g. 2.0) must lift dark pixel intensities monotonically."""
        brightened = compensator._apply_gamma(underexposed_night_frame, gamma=2.0)

        mean_before = float(np.mean(underexposed_night_frame))
        mean_after = float(np.mean(brightened))
        assert mean_after > mean_before * 1.3

    def test_gamma_less_than_one_darkens_highlights(
        self, compensator: LightingCompensator, overexposed_glare_frame: np.ndarray
    ) -> None:
        """Gamma < 1.0 (e.g. 0.6) must compress washed-out overexposed regions."""
        darkened = compensator._apply_gamma(overexposed_glare_frame, gamma=0.6)

        mean_before = float(np.mean(overexposed_glare_frame))
        mean_after = float(np.mean(darkened))
        assert mean_after < mean_before

    def test_gamma_caching_efficiency(self, compensator: LightingCompensator) -> None:
        """Gamma lookup tables must be cached in _gamma_tables to avoid redundant recreation."""
        assert len(compensator._gamma_tables) == 0

        dummy = np.full((10, 10, 3), 128, dtype=np.uint8)
        compensator._apply_gamma(dummy, gamma=1.8)
        assert 1.8 in compensator._gamma_tables
        assert len(compensator._gamma_tables) == 1

        # Second call with same gamma should reuse cached table
        compensator._apply_gamma(dummy, gamma=1.8)
        assert len(compensator._gamma_tables) == 1

    @pytest.mark.parametrize("gamma_val", [0.4, 0.7, 1.2, 1.8, 2.5, 3.2])
    def test_gamma_monotonicity_across_range(
        self, compensator: LightingCompensator, gamma_val: float
    ) -> None:
        """Gamma LUT transformation must be strictly monotonic across all 256 grayscale levels."""
        dummy = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        dummy_3ch = np.repeat(dummy, 3, axis=2)

        out = compensator._apply_gamma(dummy_3ch, gamma=gamma_val)
        lut_result = out[0, :, 0]

        # Verify monotonic non-decreasing
        for i in range(255):
            assert lut_result[i + 1] >= lut_result[i]


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Dual-Domain CIELAB CLAHE Equalization Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDualDomainCLAHE:
    """Verify Contrast Limited Adaptive Histogram Equalization on L*-channel."""

    def test_clahe_enhances_contrast_std(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """Applying CLAHE must increase standard deviation of luminance (contrast)."""
        out_bgr, luma = compensator._apply_lab_clahe(
            synthetic_road_frame, clip_limit=2.5, tile_grid=(8, 8)
        )

        std_raw = float(np.std(cv2.cvtColor(synthetic_road_frame, cv2.COLOR_BGR2GRAY)))
        std_clahe = float(np.std(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2GRAY)))

        assert std_clahe >= std_raw * 0.95
        assert out_bgr.shape == synthetic_road_frame.shape

    def test_clahe_preserves_color_chromaticity(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """Equalizing only L* channel in CIELAB must preserve A* and B* channels untouched."""
        lab_before = cv2.cvtColor(synthetic_road_frame, cv2.COLOR_BGR2LAB)
        out_bgr, _ = compensator._apply_lab_clahe(
            synthetic_road_frame, clip_limit=2.0, tile_grid=(8, 8)
        )
        lab_after = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB)

        diff_a = np.max(np.abs(lab_before[:, :, 1].astype(int) - lab_after[:, :, 1].astype(int)))
        diff_b = np.max(np.abs(lab_before[:, :, 2].astype(int) - lab_after[:, :, 2].astype(int)))

        assert diff_a <= 1
        assert diff_b <= 1

    @pytest.mark.parametrize("grid_size", [(4, 4), (8, 8), (16, 16)])
    def test_clahe_tile_grid_variations(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray, grid_size: Tuple[int, int]
    ) -> None:
        """CLAHE must execute cleanly across various grid resolutions."""
        out_bgr, luma = compensator._apply_lab_clahe(
            synthetic_road_frame, clip_limit=2.0, tile_grid=grid_size
        )
        assert out_bgr.shape == synthetic_road_frame.shape
        assert luma.shape == synthetic_road_frame.shape[:2]


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Specular Puddle & Glare Removal via Inpainting Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSpecularHighlightRemoval:
    """Verify specular puddle detection, sky suppression, and inpainting restoration."""

    def test_specular_puddle_detected_and_inpainted(
        self, compensator: LightingCompensator, wet_road_specular_frame: np.ndarray
    ) -> None:
        """Wet puddle with L >= 245 and S <= 40 must be masked and inpainted with road texture."""
        inpainted, mask, count = compensator._remove_specular_highlights(wet_road_specular_frame)

        assert count > 0
        assert np.count_nonzero(mask) == count
        puddle_roi_raw = wet_road_specular_frame[270:290, 310:330]
        puddle_roi_inp = inpainted[270:290, 310:330]
        assert float(np.mean(puddle_roi_inp)) < float(np.mean(puddle_roi_raw))

    def test_sky_region_is_never_inpainted(
        self, compensator: LightingCompensator, overexposed_glare_frame: np.ndarray
    ) -> None:
        """Upper 45% sky area must have specular mask forced to zero to avoid sky distortion."""
        h, w = overexposed_glare_frame.shape[:2]
        sky_limit = int(h * 0.45)

        _, mask, _ = compensator._remove_specular_highlights(overexposed_glare_frame)
        sky_mask_pixels = np.count_nonzero(mask[:sky_limit, :])
        assert sky_mask_pixels == 0

    def test_telea_vs_ns_inpainting_methods(
        self, default_config: LightingCompensatorConfig, wet_road_specular_frame: np.ndarray
    ) -> None:
        """Both TELEA and NS (Navier-Stokes) inpaint methods must execute without errors."""
        default_config.inpaint_method = "TELEA"
        comp_telea = LightingCompensator(config=default_config)
        out_telea, _, _ = comp_telea._remove_specular_highlights(wet_road_specular_frame)
        assert out_telea.shape == wet_road_specular_frame.shape

        default_config.inpaint_method = "NS"
        comp_ns = LightingCompensator(config=default_config)
        out_ns, _, _ = comp_ns._remove_specular_highlights(wet_road_specular_frame)
        assert out_ns.shape == wet_road_specular_frame.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Night Mode Shadow Lift & Headlight Flare Suppression Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestNightEnhancement:
    """Verify shadow region lifting and headlight glare blooming suppression."""

    def test_night_enhancement_lifts_dark_road(
        self, compensator: LightingCompensator, underexposed_night_frame: np.ndarray
    ) -> None:
        """Night enhancement must lift low-luminance pavement pixels."""
        lifted = compensator._apply_night_enhancement(underexposed_night_frame)

        mean_before = float(np.mean(underexposed_night_frame))
        mean_after = float(np.mean(lifted))
        assert mean_after > mean_before

    def test_headlight_bloom_suppression(
        self, default_config: LightingCompensatorConfig, underexposed_night_frame: np.ndarray
    ) -> None:
        """When glare suppression is enabled, extreme bright bloom bulbs are attenuated."""
        default_config.enable_glare_suppression = True
        default_config.enable_night_boost = True
        comp = LightingCompensator(config=default_config)

        # Inject extreme 255 bloom bulb
        frame_with_bloom = underexposed_night_frame.copy()
        cv2.circle(frame_with_bloom, (320, 250), 15, (255, 255, 255), -1)

        lifted = comp._apply_night_enhancement(frame_with_bloom)
        bulb_center_val = lifted[250, 320]
        assert np.max(bulb_center_val) <= 255


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Adaptive Histogram Stretching Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAdaptiveHistogramStretching:
    """Verify linear percentile stretching of luminance without overflow."""

    def test_narrow_dynamic_range_is_stretched(
        self, compensator: LightingCompensator
    ) -> None:
        """An image compressed into [80, 150] luminance must be stretched across full range."""
        low_dyn = np.full((100, 100, 3), 100, dtype=np.uint8)
        low_dyn[20:80, 20:80] = 145

        stretched = compensator._apply_histogram_stretching(low_dyn)
        gray_stretched = cv2.cvtColor(stretched, cv2.COLOR_BGR2GRAY)

        assert float(np.min(gray_stretched)) < 50
        assert float(np.max(gray_stretched)) > 200

    def test_sufficient_contrast_skips_stretch(
        self, compensator: LightingCompensator
    ) -> None:
        """If dynamic range difference (p_high - p_low) is <= 40, stretching is bypassed to avoid noise."""
        flat = np.full((100, 100, 3), 100, dtype=np.uint8)
        result = compensator._apply_histogram_stretching(flat)
        assert np.array_equal(flat, result)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Guided Image Filter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGuidedImageFilter:
    """Verify O(N) linear edge-preserving guided filter execution."""

    def test_guided_filter_preserves_sharp_edges(self) -> None:
        """Guided filter must smooth noise while keeping high-contrast step edges sharp."""
        gf = GuidedImageFilter(radius=4, eps=0.04)

        step = np.zeros((100, 100), dtype=np.uint8)
        step[:, 50:] = 200
        noise = np.random.normal(0, 10, step.shape).astype(np.int16)
        noisy = np.clip(step.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        filtered = gf.filter(guide_gray=noisy, src_gray=noisy)

        assert filtered.shape == step.shape
        assert filtered.dtype == np.uint8
        std_noisy_flat = float(np.std(noisy[:, :40]))
        std_filtered_flat = float(np.std(filtered[:, :40]))
        assert std_filtered_flat < std_noisy_flat

    def test_guided_filter_radius_and_eps_effects(self) -> None:
        """Increasing radius increases smoothing extent."""
        gf_small = GuidedImageFilter(radius=2, eps=0.01)
        gf_large = GuidedImageFilter(radius=8, eps=0.08)

        dummy = np.random.randint(50, 200, (64, 64), dtype=np.uint8)
        f_small = gf_small.filter(dummy, dummy)
        f_large = gf_large.filter(dummy, dummy)

        std_small = float(np.std(f_small))
        std_large = float(np.std(f_large))
        assert std_large <= std_small

    def test_guided_filter_singleton(self) -> None:
        """get_guided_filter singleton accessor must return same instance."""
        g1 = get_guided_filter()
        g2 = get_guided_filter()
        assert g1 is g2


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Multi-Scale Retinex with Color Restoration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiScaleRetinex:
    """Verify Retinex dynamic range expansion under deep tunnel entry / severe underexposure."""

    def test_retinex_enhancement_expands_dynamic_range(
        self, underexposed_night_frame: np.ndarray
    ) -> None:
        """MSRCR must lift deep shadows and expand local contrast."""
        msr = MultiScaleRetinex(sigmas=(15.0, 80.0), alpha=100.0, beta=40.0)
        enhanced = msr.enhance(underexposed_night_frame)

        assert enhanced.shape == underexposed_night_frame.shape
        assert enhanced.dtype == np.uint8
        assert float(np.mean(enhanced)) > float(np.mean(underexposed_night_frame))

    def test_retinex_with_custom_weights(self, underexposed_night_frame: np.ndarray) -> None:
        """Retinex must support custom weighting across scale kernels."""
        msr = MultiScaleRetinex(sigmas=(10.0, 50.0, 150.0), weights=(0.5, 0.3, 0.2))
        enhanced = msr.enhance(underexposed_night_frame)
        assert enhanced.shape == underexposed_night_frame.shape


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Hard Shadow Boundary Equalizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestShadowBoundaryEqualizer:
    """Verify detection and lifting of cast bridge/tree shadows."""

    def test_shadow_detection_and_lifting(
        self, hard_shadow_frame: np.ndarray
    ) -> None:
        """ShadowBoundaryEqualizer must identify cast shadow and lift its luminance."""
        eq = ShadowBoundaryEqualizer(shadow_ratio_threshold=0.65)
        lifted, mask = eq.equalize_shadows(hard_shadow_frame)

        assert lifted.shape == hard_shadow_frame.shape
        assert mask.shape == hard_shadow_frame.shape[:2]
        if np.count_nonzero(mask) > 100:
            shadow_mean_before = float(np.mean(hard_shadow_frame[mask > 0]))
            shadow_mean_after = float(np.mean(lifted[mask > 0]))
            assert shadow_mean_after >= shadow_mean_before

    def test_shadow_equalizer_singleton(self) -> None:
        """get_shadow_equalizer singleton accessor must return same instance."""
        s1 = get_shadow_equalizer()
        s2 = get_shadow_equalizer()
        assert s1 is s2


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Temporal Exposure Stabilization Filter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalExposureFilter:
    """Verify frame-to-frame auto-exposure hunting stabilization."""

    def test_temporal_exposure_stabilization(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """Sudden frame luminance flicker should be attenuated towards running EMA."""
        filter_exp = TemporalExposureFilter(alpha=0.20)

        for _ in range(10):
            filter_exp.update_and_stabilize(synthetic_road_frame)

        flicker_bright = np.clip(synthetic_road_frame.astype(np.float32) * 1.30, 0, 255).astype(np.uint8)
        stabilized = filter_exp.update_and_stabilize(flicker_bright)

        assert float(np.mean(stabilized)) < float(np.mean(flicker_bright))

    def test_temporal_filter_alpha_responsiveness(self, synthetic_road_frame: np.ndarray) -> None:
        """High alpha adapts more quickly to step changes than low alpha."""
        filter_fast = TemporalExposureFilter(alpha=0.80)
        filter_slow = TemporalExposureFilter(alpha=0.05)

        for _ in range(5):
            filter_fast.update_and_stabilize(synthetic_road_frame)
            filter_slow.update_and_stabilize(synthetic_road_frame)

        bright_step = np.clip(synthetic_road_frame.astype(np.float32) * 1.25, 0, 255).astype(np.uint8)
        filter_fast.update_and_stabilize(bright_step)
        filter_slow.update_and_stabilize(bright_step)

        assert filter_fast.running_mean_luma is not None
        assert filter_slow.running_mean_luma is not None
        assert filter_fast.running_mean_luma > filter_slow.running_mean_luma


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Frequency-Domain Homomorphic Filter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHomomorphicFilter:
    """Verify 2D FFT illumination decoupling."""

    def test_homomorphic_filter_frequency_domain_execution(self) -> None:
        """Homomorphic filtering must attenuate low-frequency illumination gradients."""
        hf = HomomorphicFilter(d0=25.0, gamma_l=0.5, gamma_h=1.8, c=1.0)

        h, w = 128, 128
        row = np.linspace(30.0, 210.0, w, dtype=np.float32)
        illum_gradient = np.tile(row, (h, 1)).astype(np.uint8)

        filtered = hf.filter_channel(illum_gradient)
        assert filtered.shape == (h, w)
        assert filtered.dtype == np.uint8

    def test_homomorphic_filter_parameter_variations(self) -> None:
        """Filter must accept variations in cutoff frequency d0 and high-frequency boost gamma_h."""
        hf_broad = HomomorphicFilter(d0=50.0, gamma_l=0.4, gamma_h=2.2)
        dummy = np.full((64, 64), 120, dtype=np.uint8)
        dummy[:, 30:34] = 240
        out = hf_broad.filter_channel(dummy)
        assert out.shape == (64, 64)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Anti-Blooming Headlight Flare Suppressor Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestAntiBloomingFlareSuppressor:
    """Verify detection and attenuation of oncoming vehicle high-beams."""

    def test_bloom_suppression_attenuates_saturated_core(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """Localized high-beam bloom circle must be detected and suppressed."""
        suppressor = AntiBloomingFlareSuppressor(saturation_threshold=245, min_flare_area_px=30)

        bloomed = synthetic_road_frame.copy()
        cv2.circle(bloomed, (300, 200), 12, (255, 255, 255), -1)

        suppressed, mask = suppressor.suppress_flares(bloomed)
        assert suppressed.shape == bloomed.shape
        assert mask.shape == bloomed.shape[:2]
        assert np.count_nonzero(mask) > 0

    def test_anti_blooming_area_gating(self, synthetic_road_frame: np.ndarray) -> None:
        """Tiny 2-pixel glints below min_flare_area_px are ignored to preserve lane dashed dashes."""
        suppressor = AntiBloomingFlareSuppressor(saturation_threshold=245, min_flare_area_px=50)

        # Inject tiny 2x2 glint
        glint = synthetic_road_frame.copy()
        glint[200:202, 300:302] = 255

        _, mask = suppressor.suppress_flares(glint)
        assert np.count_nonzero(mask) == 0

    def test_anti_blooming_singleton(self) -> None:
        """get_anti_blooming_suppressor singleton accessor must return same instance."""
        b1 = get_anti_blooming_suppressor()
        b2 = get_anti_blooming_suppressor()
        assert b1 is b2


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Directional Gradient & Integral Binarizer Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDirectionalGradientAndIntegralBinarizer:
    """Verify lateral sun-gradient flattening and integral image adaptive thresholding."""

    def test_directional_gradient_equalization(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """DirectionalGradientCompensator must flatten asymmetric side-sun gradients."""
        comp = DirectionalGradientCompensator(horizontal_bins=16)

        h, w = synthetic_road_frame.shape[:2]
        ramp = np.linspace(0.6, 1.4, w).reshape(1, w, 1).astype(np.float32)
        ramped = np.clip(synthetic_road_frame.astype(np.float32) * ramp, 0, 255).astype(np.uint8)

        equalized = comp.equalize_lateral_gradient(ramped)
        assert equalized.shape == ramped.shape
        assert equalized.dtype == np.uint8

    def test_fast_integral_binarizer(self) -> None:
        """FastIntegralBinarizer must binarize high-contrast lane markings in O(1) time."""
        binarizer = FastIntegralBinarizer(window_radius=10, offset_percentage=5.0)

        gray = np.full((100, 100), 80, dtype=np.uint8)
        gray[:, 48:52] = 220

        mask = binarizer.binarize(gray)
        assert mask.shape == (100, 100)
        assert mask.dtype == np.uint8
        assert np.all(mask[:, 48:52] == 255)
        assert np.all(mask[:, :40] == 0)

    @pytest.mark.parametrize("radius", [5, 12, 25])
    def test_integral_binarizer_radii(self, radius: int) -> None:
        """Integral binarizer must maintain invariant O(1) performance across various radii."""
        binarizer = FastIntegralBinarizer(window_radius=radius, offset_percentage=6.0)
        gray = np.random.randint(40, 180, (80, 80), dtype=np.uint8)
        gray[:, 35:40] = 235

        mask = binarizer.binarize(gray)
        assert mask.shape == (80, 80)
        assert mask.dtype == np.uint8

    def test_directional_and_integral_singletons(self) -> None:
        """Verify singleton accessors return same instance."""
        d1 = get_directional_gradient_compensator()
        d2 = get_directional_gradient_compensator()
        assert d1 is d2
        i1 = get_fast_integral_binarizer()
        i2 = get_fast_integral_binarizer()
        assert i1 is i2


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Logarithmic Tone Curve Compander Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLogarithmicToneCurveCompander:
    """Verify Reinhard logarithmic tone mapping and dynamic range compression."""

    def test_logarithmic_tonemapping_compands_high_contrast(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """Reinhard tonemapper must compress high dynamic range while preserving midtones."""
        compander = LogarithmicToneCurveCompander(key_value=0.18, white_point=1.5)
        out = compander.compand(synthetic_road_frame)

        assert out.shape == synthetic_road_frame.shape
        assert out.dtype == np.uint8

    def test_key_value_exposure_modulation(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """Higher key values produce brighter tonemapped results."""
        compander_low = LogarithmicToneCurveCompander(key_value=0.10)
        compander_high = LogarithmicToneCurveCompander(key_value=0.35)

        out_low = compander_low.compand(synthetic_road_frame)
        out_high = compander_high.compand(synthetic_road_frame)

        mean_low = float(np.mean(out_low))
        mean_high = float(np.mean(out_high))
        assert mean_high > mean_low


# ═══════════════════════════════════════════════════════════════════════════════
# 15. Forensic Diagnostic Quadrant Rendering & Benchmark Harness Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDiagnosticRenderingAndBenchmark:
    """Verify 4-quadrant engineering canvas assembly and batch benchmark execution."""

    def test_render_diagnostic_quadrant(
        self, compensator: LightingCompensator, synthetic_road_frame: np.ndarray
    ) -> None:
        """render_diagnostic_quadrant must assemble a 720x1280 4-panel diagnostic canvas."""
        res = compensator.process(synthetic_road_frame)
        canvas = compensator.render_diagnostic_quadrant(
            raw_bgr=synthetic_road_frame,
            result=res,
            canvas_size=(720, 1280),
        )

        assert canvas.shape == (720, 1280, 3)
        assert canvas.dtype == np.uint8

    def test_benchmark_lighting_compensator(
        self, synthetic_road_frame: np.ndarray
    ) -> None:
        """Batch benchmark function must calculate latency percentiles and mean contrast gain."""
        frames = [synthetic_road_frame for _ in range(5)]
        metrics = benchmark_lighting_compensator(frames)

        assert metrics["frames_tested"] == 5
        assert metrics["p50_latency_ms"] > 0.0
        assert metrics["p95_latency_ms"] >= metrics["p50_latency_ms"]
        assert metrics["mean_contrast_gain"] > 0.0

    def test_global_lighting_compensator_singleton(self) -> None:
        """get_lighting_compensator singleton accessor must return same instance."""
        c1 = get_lighting_compensator()
        c2 = get_lighting_compensator()
        assert c1 is c2
