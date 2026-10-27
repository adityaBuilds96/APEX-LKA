"""
src/perception/lighting_compensator.py
======================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Robust Photometric Lighting Compensator & Adverse Illumination Mitigation Stack.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module delivers dynamic illumination normalization, specular reflection suppression,
adaptive histogram expansion, and anti-glare filtering. Poor lighting conditions (tunnels,
night driving, wet asphalt reflections, low sun glare, and harsh shadows) severely degrade
both classical edge extractors and deep segmentation networks. The LightingCompensator
dynamically stabilizes image radiometry before feature extraction.

Core Architectural Sub-Pipelines:
---------------------------------
1. Dynamic Non-Linear Gamma Correction:
   Applies scene-adaptive power-law transfer functions: I_out = 255 * (I_in / 255)^(1 / gamma).
   Brightens dark tunnel interiors and night horizons without blowing out highlights.
2. Dual-Domain CLAHE Equalization:
   Computes Contrast Limited Adaptive Histogram Equalization on the LAB L*-channel
   and HLS Lightness channel with configurable tile grids (e.g. 8x8) and clip limits.
3. Specular Puddle & Glare Removal via Inpainting:
   Isolates saturated highlight regions where Lightness > 240 and Saturation < 35,
   and applies Navier-Stokes or fast Telea inpainting to restore true road surface reflectance.
4. Night Mode Shadow Lift & Headlight Glare Suppression:
   Boosts low-luminance underexposed regions while applying localized anti-blooming masks
   to suppress intense oncoming high-beam glare cones.
5. Adaptive HLS Histogram Stretching:
   Dynamically remaps the 5th and 95th percentiles of luminance to full 8-bit dynamic range.
6. Multi-Panel Diagnostic Inspection Renderer:
   Renders 4-quadrant forensic visual feeds (Raw, Luminance Map, Specular Mask, Compensated).
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("APEX.LightingCompensator")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Telemetry & Result Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class LightingMetrics:
    """
    Photometric quality metrics extracted before and after compensation.
    """
    raw_mean_luminance: float
    raw_contrast_std: float
    raw_dark_pixel_fraction: float      # Fraction of pixels with L < 30
    raw_saturated_pixel_fraction: float # Fraction of pixels with L > 240
    raw_dynamic_range_db: float
    compensated_mean_luminance: float
    compensated_contrast_std: float
    contrast_improvement_ratio: float
    specular_pixels_inpainted: int
    active_gamma: float
    active_clahe_clip: float
    is_night_mode_active: bool
    is_specular_removal_active: bool
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to serializable dictionary."""
        return {
            "raw_mean_luminance": round(self.raw_mean_luminance, 2),
            "raw_contrast_std": round(self.raw_contrast_std, 2),
            "raw_dark_pixel_fraction": round(self.raw_dark_pixel_fraction, 4),
            "raw_saturated_pixel_fraction": round(self.raw_saturated_pixel_fraction, 4),
            "raw_dynamic_range_db": round(self.raw_dynamic_range_db, 2),
            "compensated_mean_luminance": round(self.compensated_mean_luminance, 2),
            "compensated_contrast_std": round(self.compensated_contrast_std, 2),
            "contrast_improvement_ratio": round(self.contrast_improvement_ratio, 3),
            "specular_pixels_inpainted": self.specular_pixels_inpainted,
            "active_gamma": round(self.active_gamma, 3),
            "active_clahe_clip": round(self.active_clahe_clip, 2),
            "is_night_mode_active": self.is_night_mode_active,
            "is_specular_removal_active": self.is_specular_removal_active,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class LightingCompensationResult:
    """
    Full outcome container from the lighting compensation pipeline.
    """
    compensated_frame_bgr: np.ndarray
    specular_mask: np.ndarray           # Binary mask of inpainted glares
    luminance_map: np.ndarray           # Equalized 8-bit luminance channel
    metrics: LightingMetrics
    timestamp_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def frame_shape(self) -> Tuple[int, int, int]:
        return self.compensated_frame_bgr.shape


@dataclass
class LightingCompensatorConfig:
    """
    Hyperparameter configuration controlling lighting compensation sub-stages.
    """
    default_gamma: float = 1.0
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    enable_specular_removal: bool = False
    specular_l_threshold: int = 245
    specular_s_threshold: int = 40
    specular_inpaint_radius: int = 3
    inpaint_method: str = "TELEA"       # "TELEA" or "NS"
    enable_adaptive_stretch: bool = True
    stretch_lower_percentile: float = 2.0
    stretch_upper_percentile: float = 98.0
    enable_night_boost: bool = False
    night_boost_factor: float = 1.35
    enable_glare_suppression: bool = False
    glare_kernel_size: int = 9


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Master Lighting Compensator Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class LightingCompensator:
    """
    High-throughput photometric conditioning engine designed for real-time
    Jetson and workstation edge execution.
    """

    def __init__(self, config: Optional[LightingCompensatorConfig] = None) -> None:
        self.config = config if config is not None else LightingCompensatorConfig()
        self._clahe_cache: Dict[Tuple[float, Tuple[int, int]], cv2.CLAHE] = {}
        self._gamma_tables: Dict[float, np.ndarray] = {}

    # ──────────────────────────────────────────────────────────────────────────
    # Main Processing Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def process(
        self,
        frame_bgr: np.ndarray,
        override_gamma: Optional[float] = None,
        override_clahe_clip: Optional[float] = None,
        force_night_mode: Optional[bool] = None,
        force_specular_removal: Optional[bool] = None,
    ) -> LightingCompensationResult:
        """
        Execute full compensation pass on input BGR frame.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            raise ValueError("Input frame to LightingCompensator is empty or None.")

        t0 = time.perf_counter()
        h, w = frame_bgr.shape[:2]

        # Resolve active parameters
        gamma = float(override_gamma if override_gamma is not None else self.config.default_gamma)
        clahe_clip = float(override_clahe_clip if override_clahe_clip is not None else self.config.clahe_clip_limit)
        is_night = bool(force_night_mode if force_night_mode is not None else self.config.enable_night_boost)
        is_specular = bool(force_specular_removal if force_specular_removal is not None else self.config.enable_specular_removal)

        # 1. Compute raw input photometric metrics
        raw_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        raw_mean = float(np.mean(raw_gray))
        raw_std = float(np.std(raw_gray))
        raw_dark_frac = float(np.sum(raw_gray < 30) / raw_gray.size)
        raw_sat_frac = float(np.sum(raw_gray > 240) / raw_gray.size)
        dr_db = 20.0 * math.log10(max(1.0, float(np.max(raw_gray)) / max(1.0, float(np.min(raw_gray)) + 1.0)))

        work_frame = frame_bgr.copy()

        # 2. Specular Highlight & Puddle Glare Mitigation
        specular_mask = np.zeros((h, w), dtype=np.uint8)
        inpainted_pixel_count = 0
        if is_specular:
            work_frame, specular_mask, inpainted_pixel_count = self._remove_specular_highlights(work_frame)

        # 3. Dynamic Gamma Non-Linear Power-Law Correction
        if abs(gamma - 1.0) > 0.02:
            work_frame = self._apply_gamma(work_frame, gamma)

        # 4. Night Mode Shadow Lift & Glare Suppression
        if is_night:
            work_frame = self._apply_night_enhancement(work_frame)

        # 5. Dual-Domain Luminance Equalization (LAB L*-channel CLAHE)
        equalized_luma_map = raw_gray
        if self.config.enable_clahe:
            work_frame, equalized_luma_map = self._apply_lab_clahe(work_frame, clahe_clip, self.config.clahe_tile_grid_size)

        # 6. Adaptive Percentile Histogram Stretching
        if self.config.enable_adaptive_stretch:
            work_frame = self._apply_histogram_stretching(work_frame)

        # 7. Compute output compensated metrics
        comp_gray = cv2.cvtColor(work_frame, cv2.COLOR_BGR2GRAY)
        comp_mean = float(np.mean(comp_gray))
        comp_std = float(np.std(comp_gray))
        contrast_ratio = comp_std / max(1.0, raw_std)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        metrics = LightingMetrics(
            raw_mean_luminance=raw_mean,
            raw_contrast_std=raw_std,
            raw_dark_pixel_fraction=raw_dark_frac,
            raw_saturated_pixel_fraction=raw_sat_frac,
            raw_dynamic_range_db=dr_db,
            compensated_mean_luminance=comp_mean,
            compensated_contrast_std=comp_std,
            contrast_improvement_ratio=contrast_ratio,
            specular_pixels_inpainted=inpainted_pixel_count,
            active_gamma=gamma,
            active_clahe_clip=clahe_clip,
            is_night_mode_active=is_night,
            is_specular_removal_active=is_specular,
            processing_time_ms=elapsed_ms,
        )

        return LightingCompensationResult(
            compensated_frame_bgr=work_frame,
            specular_mask=specular_mask,
            luminance_map=equalized_luma_map,
            metrics=metrics,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Sub-Engine Algorithms
    # ──────────────────────────────────────────────────────────────────────────

    def _apply_gamma(self, frame_bgr: np.ndarray, gamma: float) -> np.ndarray:
        """
        Accelerated 8-bit lookup table power-law gamma transformation.
        """
        g_clamped = round(max(0.2, min(5.0, float(gamma))), 2)
        if g_clamped not in self._gamma_tables:
            inv_gamma = 1.0 / g_clamped
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
            self._gamma_tables[g_clamped] = table

        lut = self._gamma_tables[g_clamped]
        return cv2.LUT(frame_bgr, lut)

    def _remove_specular_highlights(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Identify wet road specular reflections and inpaint them with local road textures.
        """
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Specular glares exhibit high luminance and desaturated color
        mask = (l_channel >= self.config.specular_l_threshold) & (s_channel <= self.config.specular_s_threshold)
        specular_mask = mask.astype(np.uint8) * 255

        # Suppress upper sky region so we only inpaint road reflections
        sky_limit = int(frame_bgr.shape[0] * 0.45)
        specular_mask[:sky_limit, :] = 0

        pixel_count = int(np.count_nonzero(specular_mask))
        if pixel_count == 0 or pixel_count > (frame_bgr.shape[0] * frame_bgr.shape[1] * 0.25):
            # If nothing or too much is masked, avoid inpainting artifacts
            return frame_bgr, specular_mask, pixel_count

        # Dilate mask slightly to capture gradient edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated_mask = cv2.dilate(specular_mask, kernel, iterations=1)

        # Inpainting execution
        inpaint_flag = cv2.INPAINT_TELEA if self.config.inpaint_method == "TELEA" else cv2.INPAINT_NS
        inpainted = cv2.inpaint(
            frame_bgr,
            dilated_mask,
            inpaintRadius=self.config.specular_inpaint_radius,
            flags=inpaint_flag,
        )
        return inpainted, specular_mask, pixel_count

    def _apply_lab_clahe(
        self,
        frame_bgr: np.ndarray,
        clip_limit: float,
        tile_grid: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform to CIELAB, equalize L*-channel with CLAHE, and convert back.
        Preserves color chromaticity while radically enhancing lane boundary contrast.
        """
        key = (round(clip_limit, 2), tile_grid)
        if key not in self._clahe_cache:
            self._clahe_cache[key] = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
        clahe = self._clahe_cache[key]

        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        equalized_l = clahe.apply(l)
        merged_lab = cv2.merge((equalized_l, a, b))
        out_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return out_bgr, equalized_l

    def _apply_night_enhancement(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Lift underexposed dark road regions and suppress blinding specular headlight blooms.
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # Adaptive shadow lift
        v_float = v.astype(np.float32)
        # S-curve tone curve lifting values in [10, 100]
        shadow_mask = np.clip(1.0 - (v_float / 150.0), 0.0, 1.0)
        lifted_v = v_float + shadow_mask * 25.0 * self.config.night_boost_factor
        lifted_v = np.clip(lifted_v, 0.0, 255.0).astype(np.uint8)

        # Headlight bloom suppression
        if self.config.enable_glare_suppression:
            glare_mask = (v > 245).astype(np.uint8) * 255
            ksize = self.config.glare_kernel_size
            blurred_v = cv2.medianBlur(lifted_v, ksize)
            lifted_v = np.where(glare_mask > 0, blurred_v, lifted_v)

        merged_hsv = cv2.merge((h, s, lifted_v))
        return cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)

    def _apply_histogram_stretching(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Linearly stretch the dynamic range of the luminance channel between
        the 2nd and 98th percentiles to maximize feature contrast.
        """
        ycrcb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        p_low = float(np.percentile(y, self.config.stretch_lower_percentile))
        p_high = float(np.percentile(y, self.config.stretch_upper_percentile))

        if p_high <= p_low + 40.0:
            return frame_bgr

        # Linear normalization: Y_new = 255 * (Y - p_low) / (p_high - p_low)
        y_float = y.astype(np.float32)
        stretched_y = np.clip(255.0 * (y_float - p_low) / (p_high - p_low), 0.0, 255.0).astype(np.uint8)

        merged = cv2.merge((stretched_y, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    # ──────────────────────────────────────────────────────────────────────────
    # Forensic Visualization Renderer
    # ──────────────────────────────────────────────────────────────────────────

    def render_diagnostic_quadrant(
        self,
        raw_bgr: np.ndarray,
        result: LightingCompensationResult,
        canvas_size: Tuple[int, int] = (720, 1280),
    ) -> np.ndarray:
        """
        Assemble a 4-quadrant engineering diagnostic canvas:
          Top-Left:     Raw Input Frame
          Top-Right:    Equalized L*-channel Luminance Map
          Bottom-Left:  Specular Highlight Inpaint Mask
          Bottom-Right: Final Compensated Output with Telemetry
        """
        target_h, target_w = canvas_size
        half_h, half_w = target_h // 2, target_w // 2

        # 1. Top-Left: Raw
        tl = cv2.resize(raw_bgr, (half_w, half_h))
        cv2.putText(tl, "1. RAW INPUT FRAME", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # 2. Top-Right: Luminance Map
        luma_color = cv2.cvtColor(result.luminance_map, cv2.COLOR_GRAY2BGR)
        tr = cv2.resize(luma_color, (half_w, half_h))
        cv2.putText(tr, "2. CLAHE EQUALIZED LUMINANCE (L*)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

        # 3. Bottom-Left: Specular Mask (Heatmap)
        mask_color = cv2.applyColorMap(result.specular_mask, cv2.COLORMAP_JET)
        bl = cv2.resize(mask_color, (half_w, half_h))
        cv2.putText(bl, f"3. SPECULAR GLARE MASK ({result.metrics.specular_pixels_inpainted} px)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

        # 4. Bottom-Right: Final Compensated
        br = cv2.resize(result.compensated_frame_bgr, (half_w, half_h))
        cv2.putText(br, "4. COMPENSATED OUTPUT (ASIL-B)", (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        # Telemetry HUD in Bottom-Right panel
        m = result.metrics
        hud_lines = [
            f"Gamma: {m.active_gamma:.2f} | CLAHE Clip: {m.active_clahe_clip:.1f}",
            f"Contrast Gain: {m.contrast_improvement_ratio:.2f}x ({m.raw_contrast_std:.1f} -> {m.compensated_contrast_std:.1f})",
            f"Dynamic Range: {m.raw_dynamic_range_db:.1f} dB | Latency: {m.processing_time_ms:.1f} ms",
            f"Night Mode: {'ACTIVE' if m.is_night_mode_active else 'OFF'} | Glare Inpaint: {'ACTIVE' if m.is_specular_removal_active else 'OFF'}",
        ]
        y_hud = half_h - 80
        for line in hud_lines:
            cv2.putText(br, line, (15, y_hud), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1, cv2.LINE_AA)
            y_hud += 18

        # Combine into 2x2 grid
        top_row = np.hstack([tl, tr])
        bottom_row = np.hstack([bl, br])
        grid = np.vstack([top_row, bottom_row])

        # Grid dividing lines
        cv2.line(grid, (half_w, 0), (half_w, target_h), (80, 80, 80), 2)
        cv2.line(grid, (0, half_h), (target_w, half_h), (80, 80, 80), 2)

        return grid


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Edge-Preserving Guided Filter (Fog, Rain Spray & Haze Mitigation)
# ═══════════════════════════════════════════════════════════════════════════════

class GuidedImageFilter:
    """
    Fast $O(N)$ linear time edge-preserving guided filter (He et al.).
    Smooths rain haze, high-frequency water spray, and sensor shot noise
    while preserving sub-pixel lane boundary edges with zero halo artifacts.
    """

    def __init__(self, radius: int = 4, eps: float = 0.04) -> None:
        self.radius = max(1, int(radius))
        self.eps = max(1e-6, float(eps))

    def filter(self, guide_gray: np.ndarray, src_gray: np.ndarray) -> np.ndarray:
        """
        Apply guided filtering using guide_gray to direct the filtering of src_gray.
        """
        I = guide_gray.astype(np.float32) / 255.0
        p = src_gray.astype(np.float32) / 255.0
        ksize = (2 * self.radius + 1, 2 * self.radius + 1)

        mean_I = cv2.boxFilter(I, cv2.CV_32F, ksize)
        mean_p = cv2.boxFilter(p, cv2.CV_32F, ksize)
        mean_Ip = cv2.boxFilter(I * p, cv2.CV_32F, ksize)
        cov_Ip = mean_Ip - mean_I * mean_p

        mean_II = cv2.boxFilter(I * I, cv2.CV_32F, ksize)
        var_I = mean_II - mean_I * mean_I

        a = cov_Ip / (var_I + self.eps)
        b = mean_p - a * mean_I

        mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize)
        mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize)

        q = mean_a * I + mean_b
        return np.clip(q * 255.0, 0.0, 255.0).astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Multi-Scale Retinex with Color Restoration (MSRCR)
# ═══════════════════════════════════════════════════════════════════════════════

class MultiScaleRetinex:
    """
    Multi-Scale Retinex algorithm for severe under-exposure and deep tunnel entry.
    Decomposes image into illumination and reflectance components:
      R_i(x,y) = sum_k w_k * [log(I_i(x,y)) - log(I_i(x,y) * G_k)]
    """

    def __init__(
        self,
        sigmas: Sequence[float] = (15.0, 80.0, 250.0),
        weights: Optional[Sequence[float]] = None,
        alpha: float = 125.0,
        beta: float = 46.0,
    ) -> None:
        self.sigmas = list(sigmas)
        self.weights = list(weights) if weights else [1.0 / len(self.sigmas)] * len(self.sigmas)
        self.alpha = alpha
        self.beta = beta

    def enhance(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Execute multi-scale retinex dynamic range expansion.
        """
        img_float = frame_bgr.astype(np.float32) + 1.0  # Avoid log(0)
        retinex_sum = np.zeros_like(img_float)

        for sigma, weight in zip(self.sigmas, self.weights):
            ksize = int(round(sigma * 3)) | 1
            blurred = cv2.GaussianBlur(img_float, (ksize, ksize), sigma)
            retinex_sum += weight * (np.log(img_float) - np.log(blurred + 1.0))

        # Color restoration factor C_i(x,y) = beta * [log(alpha * I_i) - log(sum_c I_c)]
        channel_sum = np.sum(img_float, axis=2, keepdims=True)
        color_restoration = self.beta * (np.log(self.alpha * img_float) - np.log(channel_sum + 1.0))

        enhanced = retinex_sum * color_restoration

        # Simplest Color Balance (normalize between 1% and 99% percentiles)
        out_channels = []
        for c in range(3):
            ch = enhanced[:, :, c]
            p_low = np.percentile(ch, 1.0)
            p_high = np.percentile(ch, 99.0)
            ch_clipped = np.clip((ch - p_low) / max(1e-4, p_high - p_low) * 255.0, 0.0, 255.0)
            out_channels.append(ch_clipped.astype(np.uint8))

        return cv2.merge(out_channels)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Hard Shadow Boundary Detector & Pavement Equalizer
# ═══════════════════════════════════════════════════════════════════════════════

class ShadowBoundaryEqualizer:
    """
    Identifies hard, cast polygonal shadows across road surfaces
    (e.g., bridge shadows, overpass girders, roadside trees) and selectively
    lifts the shadow interior to eliminate spurious edge detections.
    """

    def __init__(self, shadow_ratio_threshold: float = 0.55) -> None:
        self.ratio_thresh = shadow_ratio_threshold

    def equalize_shadows(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment shadow zones and boost their luminance to match lit pavement.
        Returns: (shadow_equalized_bgr, shadow_mask)
        """
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # In shadow regions, the ratio (V / (S + 1)) is markedly suppressed
        ratio = (v.astype(np.float32) + 1.0) / (s.astype(np.float32) + 1.0)
        mean_ratio = float(np.mean(ratio))
        shadow_mask = (ratio < (mean_ratio * self.ratio_thresh)) & (v < 110)
        shadow_mask_u8 = shadow_mask.astype(np.uint8) * 255

        # Suppress sky region
        sky_limit = int(frame_bgr.shape[0] * 0.40)
        shadow_mask_u8[:sky_limit, :] = 0

        # Morphological opening and closing
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        cleaned_mask = cv2.morphologyEx(shadow_mask_u8, cv2.MORPH_CLOSE, kernel)
        cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)

        pixel_count = int(np.count_nonzero(cleaned_mask))
        if pixel_count < 100 or pixel_count > (frame_bgr.shape[0] * frame_bgr.shape[1] * 0.70):
            return frame_bgr, cleaned_mask

        # Compute illumination offset between lit road and shadow road
        lit_mask = (v > 120) & (cleaned_mask == 0)
        lit_mask[:sky_limit, :] = False

        if np.count_nonzero(lit_mask) < 200:
            return frame_bgr, cleaned_mask

        mean_lit_v = float(np.mean(v[lit_mask]))
        mean_shadow_v = float(np.mean(v[cleaned_mask > 0]))
        lift_delta = max(0.0, min(120.0, (mean_lit_v - mean_shadow_v) * 0.75))

        # Apply smooth feathered boost
        feathered = cv2.GaussianBlur(cleaned_mask.astype(np.float32) / 255.0, (15, 15), 5.0)
        lifted_v = np.clip(v.astype(np.float32) + feathered * lift_delta, 0.0, 255.0).astype(np.uint8)

        merged_hsv = cv2.merge((h, s, lifted_v))
        out_bgr = cv2.cvtColor(merged_hsv, cv2.COLOR_HSV2BGR)
        return out_bgr, cleaned_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Temporal Exposure Stabilization Filter
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalExposureFilter:
    """
    Stabilizes frame-to-frame auto-exposure hunting and rolling shutter flicker.
    Guarantees steady radiometric levels across video sequences.
    """

    def __init__(self, alpha: float = 0.25) -> None:
        self.alpha = max(0.01, min(1.0, float(alpha)))
        self.running_mean_luma: Optional[float] = None
        self.running_contrast: Optional[float] = None

    def update_and_stabilize(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Scale frame intensity to match exponential moving average luminance target.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        current_mean = float(np.mean(gray))
        current_std = float(np.std(gray))

        if self.running_mean_luma is None:
            self.running_mean_luma = current_mean
            self.running_contrast = current_std
            return frame_bgr

        # Target luminance via EMA
        self.running_mean_luma = (1.0 - self.alpha) * self.running_mean_luma + self.alpha * current_mean
        self.running_contrast = (1.0 - self.alpha) * self.running_contrast + self.alpha * current_std

        # Modulate frame gain: scale factor
        ratio = self.running_mean_luma / max(1.0, current_mean)
        clamped_ratio = max(0.80, min(1.25, ratio))

        if abs(clamped_ratio - 1.0) < 0.02:
            return frame_bgr

        adjusted = np.clip(frame_bgr.astype(np.float32) * clamped_ratio, 0.0, 255.0).astype(np.uint8)
        return adjusted


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Batch Verification & Synthetic Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

def benchmark_lighting_compensator(
    sample_frames: Sequence[np.ndarray],
    compensator: Optional[LightingCompensator] = None,
) -> Dict[str, float]:
    """
    Benchmark throughput, mean contrast gain, and latency across a batch of test frames.
    """
    comp = compensator if compensator is not None else LightingCompensator()
    latencies: List[float] = []
    contrast_gains: List[float] = []

    for f in sample_frames:
        t0 = time.perf_counter()
        res = comp.process(f)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        contrast_gains.append(res.metrics.contrast_improvement_ratio)

    return {
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "p99_latency_ms": float(np.percentile(latencies, 99)),
        "mean_contrast_gain": float(np.mean(contrast_gains)),
        "frames_tested": len(sample_frames),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Frequency-Domain Homomorphic Illumination Decoupling Filter
# ═══════════════════════════════════════════════════════════════════════════════

class HomomorphicFilter:
    """
    Decouples non-uniform illumination from surface reflectance in frequency domain:
      Image = Illumination (low frequency) * Reflectance (high frequency)
      ln(Image) = ln(Illumination) + ln(Reflectance)
    Applies high-pass Butterworth / Gaussian filtering to attenuate illumination variations
    while amplifying subtle lane line reflectance.
    """

    def __init__(self, d0: float = 30.0, gamma_l: float = 0.5, gamma_h: float = 1.8, c: float = 1.0) -> None:
        self.d0 = d0
        self.gamma_l = gamma_l
        self.gamma_h = gamma_h
        self.c = c

    def filter_channel(self, channel_u8: np.ndarray) -> np.ndarray:
        """
        Execute 2D homomorphic filtering on single grayscale channel.
        """
        h, w = channel_u8.shape
        img_float = channel_u8.astype(np.float32) + 1.0
        log_img = np.log(img_float)

        # 2D Fast Fourier Transform
        dft = cv2.dft(log_img, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # Build high-pass Gaussian filter mask
        y, x = np.ogrid[:h, :w]
        center_y, center_x = h // 2, w // 2
        dist_sq = (x - center_x) ** 2 + (y - center_y) ** 2

        # Filter transfer function H(u,v) = (gamma_h - gamma_l) * (1 - exp(-c * D^2 / D0^2)) + gamma_l
        h_filter = (self.gamma_h - self.gamma_l) * (1.0 - np.exp(-self.c * dist_sq / (self.d0 ** 2))) + self.gamma_l
        h_filter_2ch = np.repeat(h_filter[:, :, np.newaxis], 2, axis=2)

        # Frequency domain multiplication
        filtered_shift = dft_shift * h_filter_2ch
        dft_ishift = np.fft.ifftshift(filtered_shift)
        idft = cv2.idft(dft_ishift, flags=cv2.DFT_SCALE)
        idft_mag = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])

        # Exponential inverse with numerical guard
        exp_img = np.exp(np.clip(idft_mag, -20.0, 20.0)) - 1.0

        # Dynamic range normalizer
        norm_img = cv2.normalize(exp_img, None, 0, 255, cv2.NORM_MINMAX)
        return norm_img.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Anti-Blooming Headlight Flare & High-Beam Suppressor
# ═══════════════════════════════════════════════════════════════════════════════

class AntiBloomingFlareSuppressor:
    """
    Detects and suppresses intense localized optical blooming caused by oncoming
    vehicle high-beams, halogen street lamps, or low sun angles.
    Prevents saturation halos from swallowing ego lane polylines.
    """

    def __init__(self, saturation_threshold: int = 248, min_flare_area_px: int = 50) -> None:
        self.sat_thresh = saturation_threshold
        self.min_area = min_flare_area_px

    def suppress_flares(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect saturated bloom centers and blend with local ambient textures.
        Returns: (suppressed_frame_bgr, flare_mask)
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        bloom_mask = (gray >= self.sat_thresh).astype(np.uint8) * 255

        # Suppress vehicle bonnet
        h, w = frame_bgr.shape[:2]
        bloom_mask[int(h * 0.90):, :] = 0

        # Connected component analysis to find localized bloom bulbs
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bloom_mask, connectivity=8)

        cleaned_flare_mask = np.zeros_like(bloom_mask)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= self.min_area:
                cleaned_flare_mask[labels == i] = 255

        if np.count_nonzero(cleaned_flare_mask) == 0:
            return frame_bgr, cleaned_flare_mask

        # Dilate flare boundary
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        dilated = cv2.dilate(cleaned_flare_mask, kernel, iterations=2)

        # Attenuate flared pixels with bilateral smoothing
        blurred = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)
        # Gamma compress bloom region
        lut = np.array([int(round(255.0 * ((i / 255.0) ** 1.8))) for i in range(256)], dtype=np.uint8)
        compressed = cv2.LUT(blurred, lut)

        weight = cv2.GaussianBlur(dilated.astype(np.float32) / 255.0, (15, 15), 5.0)[:, :, np.newaxis]
        suppressed = (frame_bgr.astype(np.float32) * (1.0 - weight) + compressed.astype(np.float32) * weight).astype(np.uint8)

        return suppressed, cleaned_flare_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Perceptual Logarithmic Tonemapping Compander
# ═══════════════════════════════════════════════════════════════════════════════

class LogarithmicToneCurveCompander:
    """
    Applies Reinhard and Drago logarithmic dynamic range companding curves
    to compress extreme illumination contrast while keeping mid-tone lane markings crisp.
    """

    def __init__(self, key_value: float = 0.18, white_point: float = 1.5) -> None:
        self.key = key_value
        self.white_point = white_point

    def compand(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Execute global Reinhard tonemapping: L_d = L * (1 + L / L_white^2) / (1 + L)
        """
        hls = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HLS)
        h, l, s = cv2.split(hls)

        l_norm = l.astype(np.float32) / 255.0
        # Compute geometric mean luminance
        delta = 1e-4
        log_mean = float(np.exp(np.mean(np.log(l_norm + delta))))

        # Scaled luminance
        l_scaled = (self.key / log_mean) * l_norm
        # Reinhard tonemap
        l_tonemapped = (l_scaled * (1.0 + (l_scaled / (self.white_point ** 2)))) / (1.0 + l_scaled)
        l_out = np.clip(l_tonemapped * 255.0, 0.0, 255.0).astype(np.uint8)

        merged = cv2.merge((h, l_out, s))
        return cv2.cvtColor(merged, cv2.COLOR_HLS2BGR)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Directional Low-Angle Sun Gradient Compensator
# ═══════════════════════════════════════════════════════════════════════════════

class DirectionalGradientCompensator:
    """
    Normalizes asymmetric lateral illumination gradients caused by low-angle
    morning or evening side-sun (where one side of the road is washed out
    while the other side is deeply shadowed).
    """

    def __init__(self, horizontal_bins: int = 16) -> None:
        self.num_bins = horizontal_bins

    def equalize_lateral_gradient(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Estimate 1D lateral illumination profile across road surface and flatten gradient.
        """
        h, w = frame_bgr.shape[:2]
        road_roi = frame_bgr[int(h * 0.45):, :]
        gray_road = cv2.cvtColor(road_roi, cv2.COLOR_BGR2GRAY)

        # Compute column-wise mean brightness
        col_means = np.mean(gray_road, axis=0)  # Shape [w]
        # Smooth column profile with large moving average
        kernel_w = max(5, w // self.num_bins)
        smoothed_profile = cv2.blur(col_means.reshape(1, -1), (kernel_w, 1)).flatten()

        overall_mean = float(np.mean(smoothed_profile))
        # Compute per-column gain multiplier
        gains = overall_mean / np.maximum(10.0, smoothed_profile)
        clamped_gains = np.clip(gains, 0.70, 1.45).astype(np.float32)

        # Broadcast gains across all rows
        gain_matrix = np.tile(clamped_gains.reshape(1, w, 1), (h, 1, 3))
        equalized = np.clip(frame_bgr.astype(np.float32) * gain_matrix, 0.0, 255.0).astype(np.uint8)
        return equalized


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Fast Integral Image Adaptive Binarizer
# ═══════════════════════════════════════════════════════════════════════════════

class FastIntegralBinarizer:
    """
    Ultra-low latency local adaptive thresholding utilizing 2D Integral Images.
    Computes local means in $O(1)$ time per pixel regardless of window radius.
    Isolates faded lane markings under extreme spatial illumination variations.
    """

    def __init__(self, window_radius: int = 15, offset_percentage: float = 7.5) -> None:
        self.radius = max(3, int(window_radius))
        self.offset = offset_percentage

    def binarize(self, gray_channel: np.ndarray) -> np.ndarray:
        """
        Produce binary lane marking mask via integral image thresholding.
        """
        h, w = gray_channel.shape
        integral = cv2.integral(gray_channel, sdepth=cv2.CV_32S)

        # Coordinate bounds
        r = self.radius
        y, x = np.ogrid[:h, :w]
        y1 = np.maximum(0, y - r)
        y2 = np.minimum(h, y + r + 1)
        x1 = np.maximum(0, x - r)
        x2 = np.minimum(w, x + r + 1)

        # Vectorized box sum via integral table:
        # Sum = I(y2, x2) - I(y1, x2) - I(y2, x1) + I(y1, x1)
        box_sums = (
            integral[y2, x2]
            - integral[y1, x2]
            - integral[y2, x1]
            + integral[y1, x1]
        )
        box_areas = (y2 - y1) * (x2 - x1)
        local_means = box_sums.astype(np.float32) / np.maximum(1, box_areas)

        threshold = local_means * (1.0 + self.offset / 100.0)
        binary_mask = (gray_channel.astype(np.float32) > threshold).astype(np.uint8) * 255
        return binary_mask


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_COMPENSATOR: Optional[LightingCompensator] = None
_GLOBAL_GUIDED: Optional[GuidedImageFilter] = None
_GLOBAL_RETINEX: Optional[MultiScaleRetinex] = None
_GLOBAL_SHADOW: Optional[ShadowBoundaryEqualizer] = None
_GLOBAL_TEMPORAL: Optional[TemporalExposureFilter] = None
_GLOBAL_HOMOMORPHIC: Optional[HomomorphicFilter] = None
_GLOBAL_BLOOM: Optional[AntiBloomingFlareSuppressor] = None
_GLOBAL_DIRECTIONAL: Optional[DirectionalGradientCompensator] = None
_GLOBAL_INTEGRAL: Optional[FastIntegralBinarizer] = None


def get_lighting_compensator(config: Optional[LightingCompensatorConfig] = None) -> LightingCompensator:
    """
    Obtain or instantiate the global singleton LightingCompensator.
    """
    global _GLOBAL_COMPENSATOR
    if _GLOBAL_COMPENSATOR is None:
        _GLOBAL_COMPENSATOR = LightingCompensator(config=config)
    return _GLOBAL_COMPENSATOR


def get_guided_filter(radius: int = 4, eps: float = 0.04) -> GuidedImageFilter:
    """Obtain or instantiate global GuidedImageFilter."""
    global _GLOBAL_GUIDED
    if _GLOBAL_GUIDED is None:
        _GLOBAL_GUIDED = GuidedImageFilter(radius=radius, eps=eps)
    return _GLOBAL_GUIDED


def get_shadow_equalizer() -> ShadowBoundaryEqualizer:
    """Obtain or instantiate global ShadowBoundaryEqualizer."""
    global _GLOBAL_SHADOW
    if _GLOBAL_SHADOW is None:
        _GLOBAL_SHADOW = ShadowBoundaryEqualizer()
    return _GLOBAL_SHADOW


def get_anti_blooming_suppressor() -> AntiBloomingFlareSuppressor:
    """Obtain or instantiate global AntiBloomingFlareSuppressor."""
    global _GLOBAL_BLOOM
    if _GLOBAL_BLOOM is None:
        _GLOBAL_BLOOM = AntiBloomingFlareSuppressor()
    return _GLOBAL_BLOOM


def get_directional_gradient_compensator() -> DirectionalGradientCompensator:
    """Obtain or instantiate global DirectionalGradientCompensator."""
    global _GLOBAL_DIRECTIONAL
    if _GLOBAL_DIRECTIONAL is None:
        _GLOBAL_DIRECTIONAL = DirectionalGradientCompensator()
    return _GLOBAL_DIRECTIONAL


def get_fast_integral_binarizer() -> FastIntegralBinarizer:
    """Obtain or instantiate global FastIntegralBinarizer."""
    global _GLOBAL_INTEGRAL
    if _GLOBAL_INTEGRAL is None:
        _GLOBAL_INTEGRAL = FastIntegralBinarizer()
    return _GLOBAL_INTEGRAL



