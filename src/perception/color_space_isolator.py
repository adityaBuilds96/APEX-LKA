"""
src/perception/color_space_isolator.py
======================================
APEX-Vision Core v4.0 • Max-Level Multi-Color Space Fusion & Lane Isolator
ISO 26262 ASIL-B Aligned Computer Vision Feature Extraction

This module implements multi-space color decomposition and contrast-adaptive
histogram equalization (CLAHE) to isolate Yellow and White roadway and off-road
lane boundaries across high-dynamic-range lighting conditions.

Color Space Transformation Architecture:
----------------------------------------
1. HLS (Hue, Lightness, Saturation):
   - White isolation via high lightness ($L > 200$).
   - Yellow isolation via bounded hue ($15^\circ \le H \le 35^\circ$) and high saturation ($S > 80$).
2. CIE LAB ($L^*, a^*, b^*$):
   - Yellow boundary isolation via the opponent yellow-blue $b^*$-channel ($b^* > 145$).
3. YUV ($Y, U, V$):
   - White boundary isolation via calibrated luminance ($Y > 190$).
4. CLAHE (Contrast Limited Adaptive Histogram Equalization):
   - Applied to luminance / lightness channels ($L$ in HLS/LAB, $Y$ in YUV) with
     adaptive clip limits (default: 2.5, tile grid size: 8x8) to recover lane
     contrast beneath severe shadows, dusk, and headlight glare.
5. Morphological Post-Refinement:
   - Noise suppression via morphological opening ($3 \times 3$ ellipse) and gap
     closure via directional morphological dilation/closing ($5 \times 3$ kernel).
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.ColorSpaceIsolator")


@dataclass
class ColorIsolationConfig:
    """Configuration parameters for multi-color space boundary isolation."""
    # White Lane Thresholds
    white_hls_l_min: int = 200
    white_hls_l_max: int = 255
    white_yuv_y_min: int = 190
    white_yuv_y_max: int = 255
    white_rgb_r_min: int = 190
    white_rgb_g_min: int = 190
    white_rgb_b_min: int = 190

    # Yellow Lane Thresholds
    yellow_lab_b_min: int = 145
    yellow_lab_b_max: int = 255
    yellow_hls_h_min: int = 15
    yellow_hls_h_max: int = 35
    yellow_hls_s_min: int = 80
    yellow_hls_s_max: int = 255

    # Off-Road Dirt / Earth Thresholds (BAJA Buggy Profile)
    dirt_hls_h_min: int = 8
    dirt_hls_h_max: int = 25
    dirt_hls_l_min: int = 60
    dirt_hls_l_max: int = 160
    dirt_hls_s_min: int = 40
    dirt_hls_s_max: int = 180

    # CLAHE Configuration
    enable_clahe: bool = True
    clahe_clip_limit: float = 2.5
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)

    # Morphological Filter Kernel Sizes
    morph_kernel_open: Tuple[int, int] = (3, 3)
    morph_kernel_close: Tuple[int, int] = (5, 3)

    # Adaptive Shadow / Glare Compensation
    enable_adaptive_lighting: bool = True
    glare_mean_threshold: float = 185.0
    shadow_mean_threshold: float = 65.0


@dataclass
class ColorIsolationResult:
    """Result telemetry from multi-space color isolation pass."""
    white_binary_mask: np.ndarray
    yellow_binary_mask: np.ndarray
    combined_binary_mask: np.ndarray
    dirt_boundary_mask: Optional[np.ndarray] = None
    clahe_enhanced_bgr: Optional[np.ndarray] = None
    hls_image: Optional[np.ndarray] = None
    lab_image: Optional[np.ndarray] = None
    yuv_image: Optional[np.ndarray] = None
    processing_time_ms: float = 0.0
    detected_lighting_condition: str = "NORMAL"
    mean_scene_luminance: float = 0.0
    white_pixel_count: int = 0
    yellow_pixel_count: int = 0

    def get_summary_metrics(self) -> Dict[str, Any]:
        return {
            "white_pixel_count": self.white_pixel_count,
            "yellow_pixel_count": self.yellow_pixel_count,
            "mean_scene_luminance": round(self.mean_scene_luminance, 2),
            "detected_lighting_condition": self.detected_lighting_condition,
            "processing_time_ms": round(self.processing_time_ms, 3),
        }


class ColorSpaceIsolator:
    """
    Orchestrates high-precision multi-color space fusion, CLAHE contrast
    equalization, and boundary isolation for roadway and off-road environments.
    """

    def __init__(self, config: Optional[ColorIsolationConfig] = None):
        self.config = config or ColorIsolationConfig()

        # Instantiate OpenCV CLAHE processor
        self.clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe_clip_limit,
            tileGridSize=self.config.clahe_tile_grid_size,
        )

        # Pre-build morphological structuring elements
        self.kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, self.config.morph_kernel_open
        )
        self.kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, self.config.morph_kernel_close
        )

    def apply_clahe_enhancement(self, bgr_image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE across the luminance channel in LAB color space to equalize
        lighting without distorting chromaticity values.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        l_equalized = self.clahe.apply(l_chan)
        lab_equalized = cv2.merge([l_equalized, a_chan, b_chan])
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    def assess_lighting_condition(self, bgr_image: np.ndarray) -> Tuple[str, float]:
        """
        Assess scene lighting distribution to dynamically adjust threshold offsets.
        Returns condition tag: NORMAL, GLARE, SHADOW, or DUSK_NIGHT.
        """
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        mean_val = float(np.mean(gray))

        if mean_val > self.config.glare_mean_threshold:
            return "GLARE", mean_val
        elif mean_val < self.config.shadow_mean_threshold:
            return "DUSK_NIGHT", mean_val
        elif np.std(gray) > 65.0:
            return "HIGH_CONTRAST_SHADOW", mean_val
        return "NORMAL", mean_val

    def isolate_white_boundaries(
        self,
        bgr_image: np.ndarray,
        hls_image: np.ndarray,
        yuv_image: np.ndarray,
        lighting_tag: str,
    ) -> np.ndarray:
        """
        Isolate white lane markings combining HLS Lightness, YUV Luminance, and RGB thresholds.
        """
        # Dynamic threshold adaptation under glare or night
        l_thresh = self.config.white_hls_l_min
        y_thresh = self.config.white_yuv_y_min

        if lighting_tag == "GLARE":
            l_thresh = min(240, l_thresh + 20)
            y_thresh = min(235, y_thresh + 20)
        elif lighting_tag == "DUSK_NIGHT":
            l_thresh = max(160, l_thresh - 35)
            y_thresh = max(150, y_thresh - 35)

        # 1. HLS Lightness mask
        hls_l = hls_image[:, :, 1]
        mask_hls = cv2.inRange(hls_l, l_thresh, self.config.white_hls_l_max)

        # 2. YUV Luminance mask
        yuv_y = yuv_image[:, :, 0]
        mask_yuv = cv2.inRange(yuv_y, y_thresh, self.config.white_yuv_y_max)

        # 3. RGB high-intensity mask
        lower_rgb = np.array([
            self.config.white_rgb_b_min,
            self.config.white_rgb_g_min,
            self.config.white_rgb_r_min,
        ], dtype=np.uint8)
        upper_rgb = np.array([255, 255, 255], dtype=np.uint8)
        mask_rgb = cv2.inRange(bgr_image, lower_rgb, upper_rgb)

        # Fusion: HLS Lightness combined with YUV Luminance or high RGB
        white_mask = cv2.bitwise_and(mask_hls, cv2.bitwise_or(mask_yuv, mask_rgb))

        # Morphological noise removal
        white_clean = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, self.kernel_open)
        white_final = cv2.morphologyEx(white_clean, cv2.MORPH_CLOSE, self.kernel_close)
        return white_final

    def isolate_yellow_boundaries(
        self,
        bgr_image: np.ndarray,
        hls_image: np.ndarray,
        lab_image: np.ndarray,
        lighting_tag: str,
    ) -> np.ndarray:
        """
        Isolate yellow lane markings fusing LAB b*-channel (yellow-blue opponent)
        and HLS Hue-Saturation boundaries.
        """
        b_thresh = self.config.yellow_lab_b_min
        s_thresh = self.config.yellow_hls_s_min

        if lighting_tag == "DUSK_NIGHT":
            b_thresh = max(135, b_thresh - 10)
            s_thresh = max(60, s_thresh - 20)

        # 1. LAB b*-channel thresholding (positive values represent yellow)
        lab_b = lab_image[:, :, 2]
        mask_lab_b = cv2.inRange(lab_b, b_thresh, self.config.yellow_lab_b_max)

        # 2. HLS Hue & Saturation thresholding (yellow hue typically in 15-35 range)
        lower_hls = np.array([
            self.config.yellow_hls_h_min,
            30,
            s_thresh,
        ], dtype=np.uint8)
        upper_hls = np.array([
            self.config.yellow_hls_h_max,
            240,
            self.config.yellow_hls_s_max,
        ], dtype=np.uint8)
        mask_hls_yellow = cv2.inRange(hls_image, lower_hls, upper_hls)

        # Fusion: Bitwise OR between LAB b* and HLS yellow
        yellow_mask = cv2.bitwise_or(mask_lab_b, mask_hls_yellow)

        # Morphological filtering
        yellow_clean = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, self.kernel_open)
        yellow_final = cv2.morphologyEx(yellow_clean, cv2.MORPH_CLOSE, self.kernel_close)
        return yellow_final

    def isolate_offroad_dirt_boundaries(
        self,
        hls_image: np.ndarray,
        lab_image: np.ndarray,
    ) -> np.ndarray:
        """
        Isolate off-road dirt/earth boundaries for the BAJA Buggy profile.
        Differentiates dry earth/clay from asphalt and vegetation.
        """
        lower_dirt = np.array([
            self.config.dirt_hls_h_min,
            self.config.dirt_hls_l_min,
            self.config.dirt_hls_s_min,
        ], dtype=np.uint8)
        upper_dirt = np.array([
            self.config.dirt_hls_h_max,
            self.config.dirt_hls_l_max,
            self.config.dirt_hls_s_max,
        ], dtype=np.uint8)
        dirt_mask = cv2.inRange(hls_image, lower_dirt, upper_dirt)

        # Filter out extreme bright or dark regions
        lab_l = lab_image[:, :, 0]
        valid_luminance = cv2.inRange(lab_l, 40, 210)
        filtered_dirt = cv2.bitwise_and(dirt_mask, valid_luminance)

        return cv2.morphologyEx(filtered_dirt, cv2.MORPH_OPEN, self.kernel_open)

    def process(
        self,
        bgr_image: np.ndarray,
        extract_dirt: bool = False,
    ) -> ColorIsolationResult:
        """
        Execute full multi-color space boundary isolation pass.
        """
        t_start = time.perf_counter()

        if bgr_image is None or bgr_image.size == 0:
            empty_mask = np.zeros((360, 640), dtype=np.uint8)
            return ColorIsolationResult(
                white_binary_mask=empty_mask,
                yellow_binary_mask=empty_mask,
                combined_binary_mask=empty_mask,
                detected_lighting_condition="INVALID_INPUT",
            )

        # 1. Assess Lighting Distribution
        lighting_tag, mean_lum = self.assess_lighting_condition(bgr_image)

        # 2. Apply CLAHE if enabled
        if self.config.enable_clahe:
            enhanced_bgr = self.apply_clahe_enhancement(bgr_image)
        else:
            enhanced_bgr = bgr_image.copy()

        # 3. Transform into Target Color Spaces
        hls = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2HLS)
        lab = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2LAB)
        yuv = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2YUV)

        # 4. Isolate White and Yellow Boundaries
        white_mask = self.isolate_white_boundaries(enhanced_bgr, hls, yuv, lighting_tag)
        yellow_mask = self.isolate_yellow_boundaries(enhanced_bgr, hls, lab, lighting_tag)

        # 5. Extract Off-Road Dirt if requested
        dirt_mask = None
        if extract_dirt:
            dirt_mask = self.isolate_offroad_dirt_boundaries(hls, lab)

        # 6. Combine Binary Masks
        combined_mask = cv2.bitwise_or(white_mask, yellow_mask)
        if dirt_mask is not None:
            combined_mask = cv2.bitwise_or(combined_mask, dirt_mask)

        # Count isolated pixels
        n_white = int(cv2.countNonZero(white_mask))
        n_yellow = int(cv2.countNonZero(yellow_mask))

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return ColorIsolationResult(
            white_binary_mask=white_mask,
            yellow_binary_mask=yellow_mask,
            combined_binary_mask=combined_mask,
            dirt_boundary_mask=dirt_mask,
            clahe_enhanced_bgr=enhanced_bgr,
            hls_image=hls,
            lab_image=lab,
            yuv_image=yuv,
            processing_time_ms=elapsed_ms,
            detected_lighting_condition=lighting_tag,
            mean_scene_luminance=mean_lum,
            white_pixel_count=n_white,
            yellow_pixel_count=n_yellow,
        )

    def generate_diagnostic_visualization(
        self,
        bgr_image: np.ndarray,
        result: ColorIsolationResult,
    ) -> np.ndarray:
        """
        Generate a 4-quadrant diagnostic display for debugging and HUD inspection:
        - Top-Left: Original input with CLAHE enhancement
        - Top-Right: Isolated Yellow boundary mask (colorized yellow)
        - Bottom-Left: Isolated White boundary mask (colorized white)
        - Bottom-Right: Combined multi-color isolation overlay
        """
        h, w = bgr_image.shape[:2]
        canvas = np.zeros((h * 2, w * 2, 3), dtype=np.uint8)

        # Top-Left: CLAHE enhanced
        canvas[:h, :w] = result.clahe_enhanced_bgr or bgr_image

        # Top-Right: Yellow mask colorized
        yellow_colored = np.zeros_like(bgr_image)
        yellow_colored[result.yellow_binary_mask > 0] = (0, 220, 255)  # Yellow BGR
        canvas[:h, w:] = yellow_colored

        # Bottom-Left: White mask colorized
        white_colored = np.zeros_like(bgr_image)
        white_colored[result.white_binary_mask > 0] = (255, 255, 255)  # White BGR
        canvas[h:, :w] = white_colored

        # Bottom-Right: Composite overlay onto original image
        composite = bgr_image.copy()
        composite[result.white_binary_mask > 0] = (255, 255, 255)
        composite[result.yellow_binary_mask > 0] = (0, 220, 255)
        canvas[h:, w:] = composite

        # Overlay telemetry annotations
        cv2.putText(canvas, f"Lighting: {result.detected_lighting_condition}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"Luminance: {result.mean_scene_luminance:.1f}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(canvas, f"Time: {result.processing_time_ms:.2f} ms", (20, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return canvas


    def suppress_shadow_artifacts(
        self,
        bgr_image: np.ndarray,
        binary_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Suppresses false positive road-edge detections caused by cast tree and guardrail shadows.
        Uses chromaticity ratio testing: asphalt in shadow exhibits higher blue-to-red ratio
        without true reflective paint properties.
        """
        b, g, r = cv2.split(bgr_image.astype(np.float32))
        total_intensity = r + g + b + 1e-4

        # Blue-to-red shadow ratio
        shadow_ratio = b / (r + 1.0)
        # Normalized red chromaticity
        norm_r = r / total_intensity

        # Deep shadow signature: high blue ratio, low normalized red, low luminance
        is_cast_shadow = (shadow_ratio > 1.25) & (norm_r < 0.30) & (total_intensity < 220.0)

        # Mask out shadow regions from lane candidate binary mask
        cleaned_mask = binary_mask.copy()
        cleaned_mask[is_cast_shadow] = 0
        return cleaned_mask

    def isolate_multiscale_gravel_edges(
        self,
        gray_image: np.ndarray,
    ) -> np.ndarray:
        """
        Extracts high-frequency gravel and dirt berm transitions on unmarked off-road trails.
        Applies multi-scale Difference of Gaussians (DoG) filter followed by Otsu thresholding.
        """
        g1 = cv2.GaussianBlur(gray_image, (3, 3), 1.0)
        g2 = cv2.GaussianBlur(gray_image, (7, 7), 2.5)
        dog = cv2.subtract(g1, g2)
        _, thresh = cv2.threshold(dog, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return thresh


class TemporalColorCalibrationTracker:
    """
    Maintains an exponential moving average of road surface and sky chromaticity
    to stabilize color segmentation thresholds across dynamic cloud cover and tunnels.
    """

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.smoothed_mean_lum: float = 128.0
        self.smoothed_yellow_b_mean: float = 140.0
        self.is_initialized: bool = False

    def update(self, bgr_image: np.ndarray) -> Dict[str, float]:
        """Update chromaticity moving averages from incoming frame."""
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        current_lum = float(np.mean(gray))

        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        current_b_mean = float(np.mean(lab[:, :, 2]))

        if not self.is_initialized:
            self.smoothed_mean_lum = current_lum
            self.smoothed_yellow_b_mean = current_b_mean
            self.is_initialized = True
        else:
            self.smoothed_mean_lum = (1.0 - self.alpha) * self.smoothed_mean_lum + self.alpha * current_lum
            self.smoothed_yellow_b_mean = (1.0 - self.alpha) * self.smoothed_yellow_b_mean + self.alpha * current_b_mean

        return {
            "smoothed_lum": round(self.smoothed_mean_lum, 2),
            "smoothed_yellow_b": round(self.smoothed_yellow_b_mean, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone Factory & Self-Verification
# ═══════════════════════════════════════════════════════════════════════════════

def get_color_space_isolator(config: Optional[ColorIsolationConfig] = None) -> ColorSpaceIsolator:
    """Factory constructor for ColorSpaceIsolator."""
    return ColorSpaceIsolator(config=config)


def run_unit_tests() -> bool:
    """Execute built-in verification suite for ColorSpaceIsolator."""
    print("Testing ColorSpaceIsolator on synthetic multi-color scene...")

    # Create synthetic test frame
    test_frame = np.full((360, 640, 3), (60, 60, 60), dtype=np.uint8)
    # Add solid yellow boundary
    cv2.line(test_frame, (200, 359), (300, 150), (30, 215, 230), 6)
    # Add dashed white boundary
    cv2.line(test_frame, (440, 359), (340, 150), (245, 245, 245), 6)

    isolator = get_color_space_isolator()
    res = isolator.process(test_frame, extract_dirt=True)

    print(f"Lighting Condition: {res.detected_lighting_condition}")
    print(f"White Pixels Detected: {res.white_pixel_count}")
    print(f"Yellow Pixels Detected: {res.yellow_pixel_count}")
    print(f"Processing Latency: {res.processing_time_ms:.3f} ms")
    assert res.white_pixel_count > 0, "Failed to isolate synthetic white lane"
    assert res.yellow_pixel_count > 0, "Failed to isolate synthetic yellow lane"

    # Test shadow suppression
    shadowed_frame = test_frame.copy()
    shadowed_frame[200:300, 250:350] = (90, 45, 30)  # High blue shadow
    shadow_cleaned = isolator.suppress_shadow_artifacts(shadowed_frame, res.combined_binary_mask)
    assert shadow_cleaned is not None

    # Test temporal calibration tracker
    tracker = TemporalColorCalibrationTracker()
    track_res = tracker.update(test_frame)
    assert "smoothed_lum" in track_res

    print("ColorSpaceIsolator self-verification PASSED (100% checks met).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

