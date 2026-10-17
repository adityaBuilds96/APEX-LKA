"""
src/inference/environmental_adaptor.py
======================================
Automated Environmental & Illumination Auto-Adaptor for APEX-LKA.

Detects dynamic driving scene conditions:
- Low-light / Night (Mean Lightness < 40)
- Sun Glare / Overexposure (Mean Lightness > 210 & Variance > 8000)
- Tree Shadows & High Contrast Occlusions (Shadow Ratio > 0.35)
- Normal Daylight (Pass-through)

Applies high-speed hardware-friendly contrast enhancement (CLAHE, Gamma LUT,
adaptive shadow equalization) to normalize frames before perception.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np

from src.config import cfg


class EnvironmentalCondition(str, Enum):
    NORMAL = "NORMAL"
    NIGHT_LOW_LIGHT = "NIGHT_LOW_LIGHT"
    HIGH_GLARE = "HIGH_GLARE"
    SHADOW_DOMINATED = "SHADOW_DOMINATED"


@dataclass
class EnvironmentalState:
    """Telemetry describing scene illumination and applied adaptation."""
    condition: EnvironmentalCondition = EnvironmentalCondition.NORMAL
    mean_lightness: float = 128.0
    contrast_variance: float = 2500.0
    shadow_ratio: float = 0.05
    adaptation_applied: bool = False
    latency_ms: float = 0.0
    badge_text: str = "NORMAL"


class EnvironmentalAutoAdaptor:
    """
    Real-time dynamic illumination analyzer and pre-processing tone mapper.
    Processes 640x360 frames in < 3 ms on CPU with zero memory accumulation.
    """

    def __init__(
        self,
        clahe_clip_limit: float = 3.0,
        clahe_grid_size: Tuple[int, int] = (8, 8),
        gamma_glare: float = 0.6,
    ) -> None:
        sp_cfg = cfg.get("scene_perception", {})
        self.enabled = sp_cfg.get("environmental_adaptation", True)
        self.clip_limit = float(sp_cfg.get("clahe_clip_limit", clahe_clip_limit))
        grid = sp_cfg.get("clahe_grid_size", clahe_grid_size)
        self.grid_size = (int(grid[0]), int(grid[1])) if len(grid) == 2 else (8, 8)
        self.gamma_glare = float(gamma_glare)

        # Pre-instantiate reusable OpenCV CLAHE processor
        self._clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit,
            tileGridSize=self.grid_size,
        )

        # Pre-compute 256-byte LUT for gamma correction (instant ~0.05 ms lookup)
        # For gamma=0.6: output = ((x / 255.0) ** (1.0 / 0.6)) * 255.0
        inv_gamma = 1.0 / max(self.gamma_glare, 1e-4)
        lut_vals = np.clip(
            np.power(np.arange(256, dtype=np.float32) / 255.0, inv_gamma) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        self._glare_lut = lut_vals

    def analyze_and_adapt(
        self,
        img_bgr: np.ndarray,
        roi_fraction: float = 0.60,
    ) -> Tuple[np.ndarray, EnvironmentalState]:
        """
        Analyze scene illumination and return the tone-adapted frame.

        Parameters
        ----------
        img_bgr : np.ndarray
            Input BGR image (uint8).
        roi_fraction : float
            Fraction of the lower image height to analyze (focus on road / vehicle view).

        Returns
        -------
        Tuple[np.ndarray, EnvironmentalState]
            Adapted BGR image and structured environmental state.
        """
        if img_bgr is None or img_bgr.size == 0:
            return img_bgr, EnvironmentalState()

        t0 = time.perf_counter()
        h, w = img_bgr.shape[:2]

        # ── 1. Calculate Frame Lighting Metrics in ROI ──────────────────────
        roi_y_start = max(0, int((1.0 - roi_fraction) * h))
        roi_bgr = img_bgr[roi_y_start:, :]

        hls = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]

        gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

        mean_l = float(np.mean(l_channel))
        contrast_var = float(np.var(gray_roi))

        total_roi_pixels = max(1, l_channel.size)
        shadow_pixels = int(np.count_nonzero(l_channel < 30))
        shadow_ratio = float(shadow_pixels / total_roi_pixels)

        # ── 2. Environmental Condition Classification ───────────────────────
        condition = EnvironmentalCondition.NORMAL
        badge_text = "NORMAL"

        if mean_l < 40.0:
            condition = EnvironmentalCondition.NIGHT_LOW_LIGHT
            badge_text = "NIGHT_BOOST"
        elif mean_l > 210.0 and contrast_var > 8000.0:
            condition = EnvironmentalCondition.HIGH_GLARE
            badge_text = "GLARE_COMPENSATED"
        elif shadow_ratio > 0.35:
            condition = EnvironmentalCondition.SHADOW_DOMINATED
            badge_text = "SHADOW_COMPENSATED"

        # ── 3. Apply Dynamic Adaptation (if enabled) ────────────────────────
        adapted_bgr = img_bgr
        applied = False

        if self.enabled:
            if condition == EnvironmentalCondition.NIGHT_LOW_LIGHT:
                # Apply CLAHE to L channel in LAB color space (preserves chroma balance)
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = self._clahe.apply(lab[:, :, 0])
                adapted_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                applied = True

            elif condition == EnvironmentalCondition.HIGH_GLARE:
                # Apply high-speed hardware LUT gamma correction
                adapted_bgr = cv2.LUT(img_bgr, self._glare_lut)
                applied = True

            elif condition == EnvironmentalCondition.SHADOW_DOMINATED:
                # Apply adaptive tone curve to lift deep road shadows without blowing out highlights
                lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
                l_lab = lab[:, :, 0].astype(np.float32)
                # Selectively enhance dark regions (L < 80)
                shadow_mask = l_lab < 80.0
                l_lab[shadow_mask] = np.clip(l_lab[shadow_mask] * 1.45 + 10.0, 0, 255)
                lab[:, :, 0] = l_lab.astype(np.uint8)
                adapted_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                applied = True

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        state = EnvironmentalState(
            condition=condition,
            mean_lightness=round(mean_l, 1),
            contrast_variance=round(contrast_var, 1),
            shadow_ratio=round(shadow_ratio, 3),
            adaptation_applied=applied,
            latency_ms=round(elapsed_ms, 2),
            badge_text=badge_text,
        )

        return adapted_bgr, state
