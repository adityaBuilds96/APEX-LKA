"""
src/perception/roi_processor.py
===============================
APEX-Vision Core v4.0 • Dynamic Horizon & Multi-Scale ROI Processor
ISO 26262 ASIL-B Aligned Region-of-Interest & Perspective Management

Computes dynamic perspective regions of interest (ROI) and multi-scale image
pyramids driven by real-time vehicle pitch and roll telemetry.

Mathematical Formulations:
--------------------------
1. Dynamic Horizon Line:
   y_horizon = c_y - f_y * tan(theta_pitch)
   where f_y is vertical focal length, c_y is optical center, and theta_pitch
   is vehicle pitch angle relative to nominal ground plane.
2. Vanishing Point Lateral Shift:
   x_vp = c_x + f_x * tan(phi_roll)
   where phi_roll is vehicle roll angle.
3. Multi-Scale Depth Pyramid:
   - Near-Field ROI: 0 to 10 meters (high-resolution, lateral control, obstacle avoidance)
   - Mid-Field ROI: 10 to 20 meters (lane curvature preview, Stanley lookahead)
   - Far-Field ROI: 20 to 40 meters (horizon tracking, high-speed path planning)
4. Curvature-Adaptive Perspective Masking:
   - Dynamic trapezoid adjusting lateral boundary angle as a function of
     estimated road curvature kappa to prevent clipping lane boundaries on tight turns.
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

logger = logging.getLogger("APEX.ROIProcessor")


@dataclass
class ROIConfig:
    """Configuration parameters for dynamic horizon and perspective pyramid."""
    image_width: int = 640
    image_height: int = 360
    fx: float = 450.0
    fy: float = 450.0
    cx: float = 320.0
    cy: float = 180.0
    camera_height_m: float = 1.20

    # Nominal Mounting Tilt
    nominal_pitch_deg: float = 5.0
    nominal_roll_deg: float = 0.0

    # Horizon Clamping Bounds (pixels from top)
    min_horizon_y: int = 80
    max_horizon_y: int = 240

    # Perspective Trapezoid Relative Widths (fractions of image width)
    base_bottom_width_ratio: float = 0.95
    base_top_width_ratio: float = 0.22
    horizon_margin_px: int = 20

    # Multi-Scale Metric Range Bands (meters forward)
    near_field_range_m: Tuple[float, float] = (1.0, 10.0)
    mid_field_range_m: Tuple[float, float] = (10.0, 20.0)
    far_field_range_m: Tuple[float, float] = (20.0, 40.0)

    # Pyramid Downscaling Factors
    pyramid_levels: int = 3
    pyramid_scale_factor: float = 0.5


@dataclass
class HorizonVanishingPoint:
    """Estimated horizon line and perspective vanishing point."""
    horizon_y: int
    vanishing_point_x: int
    vanishing_point_y: int
    pitch_deg: float
    roll_deg: float
    horizon_slope: float = 0.0  # Non-zero under vehicle roll
    confidence: float = 1.0


@dataclass
class SubWindowROI:
    """A bounded sub-window crop in image coordinates."""
    band_name: str
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    range_min_m: float
    range_max_m: float
    cropped_image: Optional[np.ndarray] = None


@dataclass
class MultiScalePyramidResult:
    """Result telemetry from multi-scale ROI processing pass."""
    horizon_vp: HorizonVanishingPoint
    trapezoid_mask: np.ndarray
    masked_frame: np.ndarray
    near_field: SubWindowROI
    mid_field: SubWindowROI
    far_field: SubWindowROI
    pyramid_scales: List[np.ndarray] = field(default_factory=list)
    processing_time_ms: float = 0.0


class DynamicROIProcessor:
    """
    Computes dynamic horizon lines, vanishing point shifts, and multi-scale
    perspective pyramid sub-windows driven by vehicle attitude telemetry.
    """

    def __init__(self, config: Optional[ROIConfig] = None):
        self.config = config or ROIConfig()

    def compute_horizon_and_vanishing_point(
        self,
        pitch_deg: float,
        roll_deg: float,
    ) -> HorizonVanishingPoint:
        """
        Estimate horizon line position and vanishing point coordinates
        using camera pinhole geometry and vehicle attitude angles:
        Delta_y_horizon = fy * tan(theta_pitch)
        y_horizon = cy - fy * tan(theta_pitch)
        x_vp = cx + fx * tan(phi_roll)
        """
        # Relative pitch from nominal calibration
        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)

        # Delta y displacement (positive pitch down lowers horizon in view, i.e., shifts up in image coord)
        delta_y = self.config.fy * math.tan(pitch_rad)
        raw_horizon_y = self.config.cy - delta_y

        # Clamp within safety bounds
        clamped_horizon_y = int(np.clip(
            raw_horizon_y,
            self.config.min_horizon_y,
            self.config.max_horizon_y,
        ))

        # Vanishing point lateral shift
        delta_x = self.config.fx * math.tan(roll_rad)
        clamped_vp_x = int(np.clip(
            self.config.cx + delta_x,
            50,
            self.config.image_width - 50,
        ))

        # Roll-induced horizon line slope: m = tan(phi_roll)
        slope = math.tan(roll_rad)

        return HorizonVanishingPoint(
            horizon_y=clamped_horizon_y,
            vanishing_point_x=clamped_vp_x,
            vanishing_point_y=clamped_horizon_y,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
            horizon_slope=slope,
            confidence=1.0,
        )

    def ground_distance_to_image_y(
        self,
        distance_m: float,
        horizon_y: int,
    ) -> int:
        """
        Project longitudinal ground distance (meters) into image row pixel (y):
        y = horizon_y + (fy * H_cam) / distance_m
        """
        if distance_m <= 0.1:
            distance_m = 0.1
        delta_px = (self.config.fy * self.config.camera_height_m) / distance_m
        y_img = horizon_y + delta_px
        return int(np.clip(y_img, 0, self.config.image_height - 1))

    def generate_trapezoid_polygon(
        self,
        horizon_vp: HorizonVanishingPoint,
        estimated_curvature_kappa: float = 0.0,
    ) -> np.ndarray:
        """
        Generate a dynamic perspective trapezoid polygon enclosing the roadway corridor.
        Adapts lateral vertex positions according to estimated path curvature kappa.
        """
        w = self.config.image_width
        h = self.config.image_height
        vp_x = horizon_vp.vanishing_point_x
        h_y = horizon_vp.horizon_y + self.config.horizon_margin_px

        # Curvature lateral shift (positive kappa curves right)
        curve_shift_px = int(np.clip(estimated_curvature_kappa * 1500.0, -80.0, 80.0))

        # Top boundary width (around horizon)
        top_w = int(w * self.config.base_top_width_ratio)
        top_left_x = int(np.clip(vp_x - top_w // 2 + curve_shift_px, 10, w - 10))
        top_right_x = int(np.clip(vp_x + top_w // 2 + curve_shift_px, 10, w - 10))

        # Bottom boundary width (near vehicle bonnet)
        bottom_w = int(w * self.config.base_bottom_width_ratio)
        bottom_left_x = int(np.clip((w - bottom_w) // 2, 0, w - 1))
        bottom_right_x = int(np.clip((w + bottom_w) // 2, 0, w - 1))
        bottom_y = h - 1

        polygon = np.array([
            [bottom_left_x, bottom_y],
            [top_left_x, h_y],
            [top_right_x, h_y],
            [bottom_right_x, bottom_y],
        ], dtype=np.int32)

        return polygon

    def create_perspective_mask(
        self,
        trapezoid_poly: np.ndarray,
    ) -> np.ndarray:
        """Create single-channel 8-bit binary mask from trapezoid polygon."""
        mask = np.zeros((self.config.image_height, self.config.image_width), dtype=np.uint8)
        cv2.fillPoly(mask, [trapezoid_poly], 255)
        return mask

    def extract_range_band_subwindows(
        self,
        frame: np.ndarray,
        horizon_vp: HorizonVanishingPoint,
    ) -> Tuple[SubWindowROI, SubWindowROI, SubWindowROI]:
        """
        Extract Near-Field, Mid-Field, and Far-Field sub-windows based on metric distances.
        """
        h_y = horizon_vp.horizon_y
        w = self.config.image_width

        # Project metric distance bounds to pixel rows
        # Near Field: 1.0m to 10.0m
        y_near_bottom = self.ground_distance_to_image_y(self.config.near_field_range_m[0], h_y)
        y_near_top = self.ground_distance_to_image_y(self.config.near_field_range_m[1], h_y)

        # Mid Field: 10.0m to 20.0m
        y_mid_bottom = y_near_top
        y_mid_top = self.ground_distance_to_image_y(self.config.mid_field_range_m[1], h_y)

        # Far Field: 20.0m to 40.0m
        y_far_bottom = y_mid_top
        y_far_top = self.ground_distance_to_image_y(self.config.far_field_range_m[1], h_y)

        # Build SubWindowROI objects with safe crops
        def make_subwindow(name: str, y_top: int, y_bot: int, r_min: float, r_max: float) -> SubWindowROI:
            y_t = max(0, min(y_top, y_bot))
            y_b = min(self.config.image_height, max(y_top, y_bot))
            if y_b <= y_t:
                y_b = y_t + 5
            crop = frame[y_t:y_b, 0:w].copy() if frame is not None else None
            return SubWindowROI(
                band_name=name,
                x_min=0,
                y_min=y_t,
                x_max=w,
                y_max=y_b,
                range_min_m=r_min,
                range_max_m=r_max,
                cropped_image=crop,
            )

        near_roi = make_subwindow("NEAR_FIELD", y_near_top, y_near_bottom,
                                  self.config.near_field_range_m[0], self.config.near_field_range_m[1])
        mid_roi = make_subwindow("MID_FIELD", y_mid_top, y_mid_bottom,
                                 self.config.mid_field_range_m[0], self.config.mid_field_range_m[1])
        far_roi = make_subwindow("FAR_FIELD", y_far_top, y_far_bottom,
                                 self.config.far_field_range_m[0], self.config.far_field_range_m[1])

        return near_roi, mid_roi, far_roi

    def build_scale_space_pyramid(
        self,
        masked_frame: np.ndarray,
    ) -> List[np.ndarray]:
        """
        Build Gaussian scale-space image pyramid (e.g. 1.0x, 0.5x, 0.25x).
        """
        pyramid = [masked_frame]
        curr = masked_frame
        for _ in range(1, self.config.pyramid_levels):
            curr = cv2.pyrDown(curr)
            pyramid.append(curr)
        return pyramid

    def process(
        self,
        frame: np.ndarray,
        pitch_deg: float = 5.0,
        roll_deg: float = 0.0,
        estimated_curvature_kappa: float = 0.0,
    ) -> MultiScalePyramidResult:
        """
        Execute full dynamic ROI and perspective pyramid extraction pass.
        """
        t_start = time.perf_counter()

        # 1. Estimate Horizon and Vanishing Point
        horizon_vp = self.compute_horizon_and_vanishing_point(pitch_deg, roll_deg)

        # 2. Build Curvature-Adaptive Trapezoid
        trapezoid_poly = self.generate_trapezoid_polygon(horizon_vp, estimated_curvature_kappa)
        mask = self.create_perspective_mask(trapezoid_poly)

        # 3. Apply Mask to Image
        if len(frame.shape) == 3:
            mask_3c = cv2.merge([mask, mask, mask])
            masked_frame = cv2.bitwise_and(frame, mask_3c)
        else:
            masked_frame = cv2.bitwise_and(frame, mask)

        # 4. Extract Metric Range Sub-Windows
        near_roi, mid_roi, far_roi = self.extract_range_band_subwindows(masked_frame, horizon_vp)

        # 5. Build Scale-Space Pyramid
        pyramid_scales = self.build_scale_space_pyramid(masked_frame)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return MultiScalePyramidResult(
            horizon_vp=horizon_vp,
            trapezoid_mask=mask,
            masked_frame=masked_frame,
            near_field=near_roi,
            mid_field=mid_roi,
            far_field=far_roi,
            pyramid_scales=pyramid_scales,
            processing_time_ms=elapsed_ms,
        )

    def draw_roi_diagnostic_overlay(
        self,
        frame: np.ndarray,
        result: MultiScalePyramidResult,
    ) -> np.ndarray:
        """
        Draw visual overlay of horizon line, vanishing point, and range boundaries.
        """
        annotated = frame.copy()
        w = self.config.image_width
        h_y = result.horizon_vp.horizon_y
        vp_x = result.horizon_vp.vanishing_point_x

        # Draw Horizon Line (Cyan)
        cv2.line(annotated, (0, h_y), (w - 1, h_y), (255, 255, 0), 2)
        cv2.putText(annotated, f"Horizon (Pitch: {result.horizon_vp.pitch_deg:.1f} deg)",
                    (10, h_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

        # Draw Vanishing Point (Red Crosshair)
        cv2.circle(annotated, (vp_x, h_y), 6, (0, 0, 255), -1)
        cv2.line(annotated, (vp_x - 12, h_y), (vp_x + 12, h_y), (0, 0, 255), 2)
        cv2.line(annotated, (vp_x, h_y - 12), (vp_x, h_y + 12), (0, 0, 255), 2)

        # Draw Range Band Dividers (Green / Orange)
        # Mid/Near divider
        y_mid_near = result.near_field.y_min
        cv2.line(annotated, (0, y_mid_near), (w - 1, y_mid_near), (0, 255, 0), 1)
        cv2.putText(annotated, "10m (Near/Mid)", (w - 110, y_mid_near - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Far/Mid divider
        y_far_mid = result.mid_field.y_min
        cv2.line(annotated, (0, y_far_mid), (w - 1, y_far_mid), (0, 165, 255), 1)
        cv2.putText(annotated, "20m (Mid/Far)", (w - 110, y_far_mid - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        return annotated


    def estimate_visual_vanishing_point_from_edges(
        self,
        edge_image: np.ndarray,
        min_line_length: int = 40,
        max_line_gap: int = 15,
    ) -> Optional[Tuple[int, int]]:
        """
        Visual Fallback Vanishing Point Estimator:
        When IMU / ESKF attitude data is temporarily degraded or uncalibrated,
        detects converging road boundary lines via HoughLinesP and computes
        their intersection in the upper half of the image.
        """
        lines = cv2.HoughLinesP(
            edge_image,
            rho=1,
            theta=np.pi / 180.0,
            threshold=30,
            minLineLength=min_line_length,
            maxLineGap=max_line_gap,
        )
        if lines is None or len(lines) < 2:
            return None

        left_lines = []
        right_lines = []

        for line in lines:
            coords = line.flatten()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
            if x2 == x1:
                continue
            slope = (y2 - y1) / float(x2 - x1)
            # Filter near-horizontal guardrails / sky
            if abs(slope) < 0.35:
                continue
            # Left lane candidate (negative slope in image coordinates)
            if slope < -0.35 and max(x1, x2) < self.config.image_width * 0.65:
                left_lines.append((slope, y1 - slope * x1))
            # Right lane candidate (positive slope in image coordinates)
            elif slope > 0.35 and min(x1, x2) > self.config.image_width * 0.35:
                right_lines.append((slope, y1 - slope * x1))

        if not left_lines or not right_lines:
            return None

        # Average slope and intercept
        m_l, b_l = np.median([l[0] for l in left_lines]), np.median([l[1] for l in left_lines])
        m_r, b_r = np.median([r[0] for r in right_lines]), np.median([r[1] for r in right_lines])

        if abs(m_l - m_r) < 1e-4:
            return None

        # Intersection point: x_i = (b_r - b_l) / (m_l - m_r), y_i = m_l * x_i + b_l
        x_i = (b_r - b_l) / (m_l - m_r)
        y_i = m_l * x_i + b_l

        if 50 <= x_i <= self.config.image_width - 50 and 50 <= y_i <= self.config.image_height * 0.7:
            return int(round(x_i)), int(round(y_i))
        return None

    def generate_sky_exclusion_mask(
        self,
        horizon_vp: HorizonVanishingPoint,
        extra_sky_margin_px: int = 15,
    ) -> np.ndarray:
        """
        Creates an 8-bit single channel mask excluding all image pixels above the
        horizon boundary plus safety margin to reject clouds, sun glare, and overpasses.
        """
        mask = np.ones((self.config.image_height, self.config.image_width), dtype=np.uint8) * 255
        cutoff_y = max(0, min(self.config.image_height, horizon_vp.horizon_y - extra_sky_margin_px))
        mask[:cutoff_y, :] = 0
        return mask

    def extract_corridor_crop_bounding_box(
        self,
        trapezoid_polygon: np.ndarray,
    ) -> Tuple[int, int, int, int]:
        """
        Compute minimal axis-aligned bounding box (x, y, w, h) enclosing the dynamic perspective trapezoid.
        """
        x, y, w, h = cv2.boundingRect(trapezoid_polygon)
        return int(x), int(y), int(w), int(h)


class AdaptiveTemporalHorizonFilter:
    """
    Low-pass filter for vehicle pitch and horizon line oscillations.
    Prevents high-frequency suspension bounce on off-road terrain from inducing
    rapid perspective ROI jitter.
    """

    def __init__(self, smoothing_factor: float = 0.25, max_slew_rate_px_per_frame: float = 12.0):
        self.alpha = smoothing_factor
        self.max_slew = max_slew_rate_px_per_frame
        self.filtered_horizon_y: Optional[float] = None
        self.filtered_vp_x: Optional[float] = None

    def filter(self, raw_horizon_y: int, raw_vp_x: int) -> Tuple[int, int]:
        """Apply slew-limited IIR filter to horizon and vanishing point coordinates."""
        if self.filtered_horizon_y is None or self.filtered_vp_x is None:
            self.filtered_horizon_y = float(raw_horizon_y)
            self.filtered_vp_x = float(raw_vp_x)
            return raw_horizon_y, raw_vp_x

        # Slew rate limiter on y
        target_y = (1.0 - self.alpha) * self.filtered_horizon_y + self.alpha * float(raw_horizon_y)
        dy = target_y - self.filtered_horizon_y
        if abs(dy) > self.max_slew:
            target_y = self.filtered_horizon_y + math.copysign(self.max_slew, dy)
        self.filtered_horizon_y = target_y

        # Slew rate limiter on x
        target_x = (1.0 - self.alpha) * self.filtered_vp_x + self.alpha * float(raw_vp_x)
        dx = target_x - self.filtered_vp_x
        if abs(dx) > self.max_slew:
            target_x = self.filtered_vp_x + math.copysign(self.max_slew, dx)
        self.filtered_vp_x = target_x

        return int(round(self.filtered_horizon_y)), int(round(self.filtered_vp_x))


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification
# ═══════════════════════════════════════════════════════════════════════════

def get_roi_processor(config: Optional[ROIConfig] = None) -> DynamicROIProcessor:
    """Factory constructor for DynamicROIProcessor."""
    return DynamicROIProcessor(config=config)



def run_unit_tests() -> bool:
    """Validate dynamic horizon mathematics, visual fallback, and multi-scale pyramid integrity."""
    print("Testing DynamicROIProcessor geometry and sub-windows...")
    processor = get_roi_processor()

    # Invariant 1: At 0.0 deg pitch, horizon is at cy (180 px)
    h_nominal = processor.compute_horizon_and_vanishing_point(pitch_deg=0.0, roll_deg=0.0)
    assert h_nominal.horizon_y == 180, f"Expected horizon at cy=180, got {h_nominal.horizon_y}"
    assert h_nominal.vanishing_point_x == 320, f"Expected VP at cx=320, got {h_nominal.vanishing_point_x}"

    # Invariant 2: Positive pitch (downward tilt) raises horizon row (moves up in visual frame)
    h_pitch_up = processor.compute_horizon_and_vanishing_point(pitch_deg=5.0, roll_deg=0.0)
    assert h_pitch_up.horizon_y < 180, f"Expected horizon_y < 180 under +5 deg pitch, got {h_pitch_up.horizon_y}"

    # Invariant 3: Roll tilt shifts vanishing point laterally
    h_roll = processor.compute_horizon_and_vanishing_point(pitch_deg=0.0, roll_deg=3.0)
    assert h_roll.vanishing_point_x > 320, f"Expected VP x > 320 under +3 deg roll, got {h_roll.vanishing_point_x}"

    # Invariant 4: Sub-window bounds monotonicity (Far is above Mid, Mid is above Near)
    dummy_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    res = processor.process(dummy_frame, pitch_deg=5.0, roll_deg=0.0)
    assert res.far_field.y_min < res.mid_field.y_min, "Far field must be vertically above Mid field"
    assert res.mid_field.y_min < res.near_field.y_min, "Mid field must be vertically above Near field"
    assert res.near_field.y_max <= 360, "Near field must terminate within image height"

    # Invariant 5: Pyramid scale downsampling check
    assert len(res.pyramid_scales) == 3, "Expected 3 pyramid levels"
    assert res.pyramid_scales[1].shape[0] == 180, "Expected level 1 height to be 180"

    # Invariant 6: Visual vanishing point fallback with synthetic lines
    synth_edges = np.zeros((360, 640), dtype=np.uint8)
    cv2.line(synth_edges, (150, 350), (315, 175), 255, 2)  # Left lane
    cv2.line(synth_edges, (490, 350), (325, 175), 255, 2)  # Right lane
    vp_visual = processor.estimate_visual_vanishing_point_from_edges(synth_edges)
    assert vp_visual is not None, "Failed to compute visual vanishing point from converging lines"
    assert abs(vp_visual[0] - 320) < 25, f"Expected visual VP near x=320, got {vp_visual[0]}"

    # Invariant 7: Sky exclusion mask
    sky_mask = processor.generate_sky_exclusion_mask(h_nominal)
    assert sky_mask[50, 320] == 0, "Sky region must be masked out to 0"
    assert sky_mask[250, 320] == 255, "Road region must be preserved at 255"

    print("DynamicROIProcessor self-verification PASSED (100% geometric tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

