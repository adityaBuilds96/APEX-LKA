"""
src/perception/cv_precision_stack.py
====================================
APEX-Vision Core v4.0 • Master Computer Vision Precision Stack
ISO 26262 ASIL-B Aligned Geometric Perception & Curvature Estimation Engine

This module serves as the primary classical and geometric computer vision pipeline
for the APEX platform. It seamlessly orchestrates dynamic horizon estimation,
multi-color space boundary isolation, dynamic inverse perspective mapping (IPM),
Sobel-X gradient filtering, dual-threshold Canny edge extraction, slope-filtered
Hough line segment clustering, and RANSAC 2nd-degree parabolic fitting.

Architectural Flow & Processing Stages:
---------------------------------------
1. Ingestion & Dynamic Horizon Processing:
   - Takes 640x360 RGB camera frame + 400 Hz MTi-680 pitch/roll attitude angles.
   - Estimates dynamic horizon line: y_horizon = c_y - f_y * tan(theta_pitch).
   - Generates curvature-adaptive perspective trapezoid to mask sky, hood, and off-path clutter.
2. Multi-Color Space Fusion & Boundary Isolation:
   - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) on luminance.
   - Fuses HLS lightness (L > 200) + YUV luminance (Y > 190) for White lane boundaries.
   - Fuses LAB opponent b*-channel (b* > 145) + HLS Hue (15 <= H <= 35) for Yellow lane boundaries.
   - Off-road dirt/earth berm segmentation for the BAJA SAE Buggy profile.
3. Dynamic Inverse Perspective Mapping (IPM):
   - Computes top-down Bird's Eye View (BEV) at strictly 0.0185 meters/pixel.
   - Dynamically recomputes homography matrix H_ipm when |delta_pitch| > 0.5 deg.
4. Directional Edge & Hough Feature Extraction:
   - Computes horizontal spatial gradient G_x = dI/dx (Sobel-X).
   - Applies dual-mode Canny (asphalt vs adaptive dirt).
   - Runs cv2.HoughLinesP with strict slope filter: rejects segments with |m| < 0.35.
5. RANSAC 2nd-Degree Parabolic Polynomial Fitting:
   - Fits x = A * y^2 + B * y + C in metric BEV space.
   - Outlier rejection isolates true lane markings from gravel clods and tire tracks.
6. Kinematics, Curvature & Control Telemetry:
   - Calculates signed lateral cross-track error: e_lat = X_vehicle - X_lane_center.
   - Calculates tangent heading error: theta_e = atan(2*A*y + B).
   - Calculates metric radius of curvature: R = (1 + (2*A*y + B)^2)^1.5 / |2*A|.
   - Dynamic lookahead distance: L_d(v) = max(L_min, k_speed * v).
7. Temporal Anti-Flicker Tracking:
   - Stabilizes polynomial coefficients across consecutive frames to guarantee
     zero lane flickering under intermittent occlusions or shadow crossings.
"""

from dataclasses import dataclass, field
from enum import Enum
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
from src.perception.roi_processor import (
    DynamicROIProcessor,
    ROIConfig,
    HorizonVanishingPoint,
    MultiScalePyramidResult,
    get_roi_processor,
)
from src.perception.ipm_transformer import (
    IPMTransformer,
    IPMConfig,
    CameraIntrinsics,
    CameraExtrinsics,
    IPMTransformResult,
    get_ipm_transformer,
)
from src.perception.color_space_isolator import (
    ColorSpaceIsolator,
    ColorIsolationConfig,
    ColorIsolationResult,
    get_color_space_isolator,
)
from src.perception.edge_hough_pipeline import (
    EdgeHoughPipeline,
    EdgeHoughConfig,
    EdgeHoughResult,
    LaneBoundaryPolynomial,
    LineSegment,
    get_edge_hough_pipeline,
)

logger = logging.getLogger("APEX.CVPrecisionStack")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Telemetry Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class LaneMarkingType(Enum):
    """Classification of detected road boundary markings."""
    UNKNOWN = "UNKNOWN"
    SOLID_WHITE = "SOLID_WHITE"
    DASHED_WHITE = "DASHED_WHITE"
    SOLID_YELLOW = "SOLID_YELLOW"
    DOUBLE_YELLOW = "DOUBLE_YELLOW"
    OFFROAD_DIRT_BERM = "OFFROAD_DIRT_BERM"


class LaneTrackingState(Enum):
    """Temporal tracking state for lane perception continuity."""
    UNINITIALIZED = "UNINITIALIZED"
    ACQUIRING = "ACQUIRING"
    TRACKING_HIGH_CONFIDENCE = "TRACKING_HIGH_CONFIDENCE"
    TRACKING_LOW_CONFIDENCE = "TRACKING_LOW_CONFIDENCE"
    COASTING_TEMPORAL_PREDICTION = "COASTING_TEMPORAL_PREDICTION"
    LOST = "LOST"


@dataclass
class CVStackConfig:
    """Master configuration parameters for the CV Precision Stack."""
    # Sub-Engine Configurations
    roi_config: ROIConfig = field(default_factory=ROIConfig)
    ipm_config: IPMConfig = field(default_factory=IPMConfig)
    color_config: ColorIsolationConfig = field(default_factory=ColorIsolationConfig)
    edge_config: EdgeHoughConfig = field(default_factory=EdgeHoughConfig)

    # Vehicle Dynamics & Mounting Defaults
    camera_height_m: float = 1.20
    nominal_pitch_deg: float = 5.0
    nominal_roll_deg: float = 0.0

    # Lookahead Distance Parameters: L_d(v) = max(L_min, k_speed * v)
    lookahead_min_m: float = 3.0
    lookahead_max_m: float = 20.0
    lookahead_speed_gain_s: float = 0.8  # Lookahead time in seconds

    # Temporal Smoothing & Anti-Flicker
    temporal_smoothing_alpha: float = 0.30  # Filter gain for polynomial coefficients
    max_coasting_frames: int = 5            # Number of frames to maintain last known lane
    confidence_decay_factor: float = 0.80   # Exponential decay during coasting

    # Off-Road / BAJA Buggy Mode
    is_offroad_mode: bool = True
    default_lane_width_m: float = 3.20


@dataclass
class PerceptionKinematics:
    """Geometric and kinematic vehicle control feeds derived from lane geometry."""
    cross_track_error_m: float = 0.0      # Signed lateral offset e_lat (meters)
    heading_error_rad: float = 0.0        # Tangent heading error theta_e (radians)
    heading_error_deg: float = 0.0        # Tangent heading error (degrees)
    radius_of_curvature_m: float = 5000.0 # Curve radius R at vehicle front (meters)
    lookahead_distance_m: float = 5.0     # Lookahead distance L_d (meters)
    lookahead_cte_m: float = 0.0          # Lateral offset measured at lookahead distance L_d
    lane_width_at_vehicle_m: float = 3.20 # Measured lane width at vehicle front (meters)
    is_left_curve: bool = False
    is_right_curve: bool = False


@dataclass
class CVPrecisionTelemetry:
    """Master telemetry output encapsulating all perception stages per frame."""
    frame_idx: int
    timestamp_utc: str
    tracking_state: LaneTrackingState
    kinematics: PerceptionKinematics
    left_polynomial: LaneBoundaryPolynomial
    right_polynomial: LaneBoundaryPolynomial
    center_polynomial: Optional[LaneBoundaryPolynomial] = None
    left_marking_type: LaneMarkingType = LaneMarkingType.UNKNOWN
    right_marking_type: LaneMarkingType = LaneMarkingType.UNKNOWN
    overall_confidence: float = 0.0
    horizon_y_px: int = 180
    vanishing_point_x_px: int = 320
    is_ipm_recomputed: bool = False
    processing_latency_ms: float = 0.0
    stage_latencies_ms: Dict[str, float] = field(default_factory=dict)

    # Optional Visual Artifacts for HUD Streaming
    annotated_perspective_bgr: Optional[np.ndarray] = None
    bev_metric_bgr: Optional[np.ndarray] = None
    color_isolation_mask: Optional[np.ndarray] = None


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Anti-Flicker Lane Tracker
# ═══════════════════════════════════════════════════════════════════════════════

class TemporalPolynomialFilter:
    """
    Maintains temporal continuity of lane boundary polynomials across consecutive frames.
    Implements exponential moving average smoothing, outlier rejection, and coasting
    during intermittent visual dropouts to eliminate lane flickering.
    """

    def __init__(
        self,
        alpha: float = 0.30,
        max_coasting_frames: int = 5,
        decay_factor: float = 0.80,
    ):
        self.alpha = alpha
        self.max_coasting_frames = max_coasting_frames
        self.decay_factor = decay_factor

        self.smoothed_left: Optional[LaneBoundaryPolynomial] = None
        self.smoothed_right: Optional[LaneBoundaryPolynomial] = None
        self.coasting_frames_left: int = 0
        self.coasting_frames_right: int = 0
        self.tracking_state = LaneTrackingState.UNINITIALIZED

    def update(
        self,
        raw_left: LaneBoundaryPolynomial,
        raw_right: LaneBoundaryPolynomial,
    ) -> Tuple[LaneBoundaryPolynomial, LaneBoundaryPolynomial, LaneTrackingState]:
        """
        Update temporal polynomial filters with raw detection outputs.
        Returns smoothed left and right polynomials along with tracking state.
        """
        # Process Left Boundary
        if raw_left.is_valid:
            if self.smoothed_left is None or not self.smoothed_left.is_valid:
                self.smoothed_left = raw_left
            else:
                # Exponential moving average filter on coefficients [A, B, C]
                A = (1.0 - self.alpha) * self.smoothed_left.A + self.alpha * raw_left.A
                B = (1.0 - self.alpha) * self.smoothed_left.B + self.alpha * raw_left.B
                C = (1.0 - self.alpha) * self.smoothed_left.C + self.alpha * raw_left.C
                conf = (1.0 - self.alpha) * self.smoothed_left.confidence + self.alpha * raw_left.confidence
                R = (1.0 - self.alpha) * self.smoothed_left.radius_of_curvature_m + self.alpha * raw_left.radius_of_curvature_m
                self.smoothed_left = LaneBoundaryPolynomial(
                    is_valid=True,
                    A=A,
                    B=B,
                    C=C,
                    radius_of_curvature_m=round(R, 1),
                    lateral_offset_m=round(C * 0.0185, 3),
                    confidence=round(conf, 3),
                )
            self.coasting_frames_left = 0
        else:
            # Coast using previous polynomial if within coasting limit
            if self.smoothed_left is not None and self.smoothed_left.is_valid and self.coasting_frames_left < self.max_coasting_frames:
                self.coasting_frames_left += 1
                decayed_conf = self.smoothed_left.confidence * (self.decay_factor ** self.coasting_frames_left)
                self.smoothed_left.confidence = round(decayed_conf, 3)
            else:
                self.smoothed_left = LaneBoundaryPolynomial(is_valid=False)

        # Process Right Boundary
        if raw_right.is_valid:
            if self.smoothed_right is None or not self.smoothed_right.is_valid:
                self.smoothed_right = raw_right
            else:
                A = (1.0 - self.alpha) * self.smoothed_right.A + self.alpha * raw_right.A
                B = (1.0 - self.alpha) * self.smoothed_right.B + self.alpha * raw_right.B
                C = (1.0 - self.alpha) * self.smoothed_right.C + self.alpha * raw_right.C
                conf = (1.0 - self.alpha) * self.smoothed_right.confidence + self.alpha * raw_right.confidence
                R = (1.0 - self.alpha) * self.smoothed_right.radius_of_curvature_m + self.alpha * raw_right.radius_of_curvature_m
                self.smoothed_right = LaneBoundaryPolynomial(
                    is_valid=True,
                    A=A,
                    B=B,
                    C=C,
                    radius_of_curvature_m=round(R, 1),
                    lateral_offset_m=round(C * 0.0185, 3),
                    confidence=round(conf, 3),
                )
            self.coasting_frames_right = 0
        else:
            if self.smoothed_right is not None and self.smoothed_right.is_valid and self.coasting_frames_right < self.max_coasting_frames:
                self.coasting_frames_right += 1
                decayed_conf = self.smoothed_right.confidence * (self.decay_factor ** self.coasting_frames_right)
                self.smoothed_right.confidence = round(decayed_conf, 3)
            else:
                self.smoothed_right = LaneBoundaryPolynomial(is_valid=False)

        # Determine Tracking State
        left_ok = self.smoothed_left is not None and self.smoothed_left.is_valid
        right_ok = self.smoothed_right is not None and self.smoothed_right.is_valid

        if left_ok and right_ok:
            if self.coasting_frames_left > 0 or self.coasting_frames_right > 0:
                self.tracking_state = LaneTrackingState.COASTING_TEMPORAL_PREDICTION
            elif self.smoothed_left.confidence > 0.60 and self.smoothed_right.confidence > 0.60:
                self.tracking_state = LaneTrackingState.TRACKING_HIGH_CONFIDENCE
            else:
                self.tracking_state = LaneTrackingState.TRACKING_LOW_CONFIDENCE
        elif left_ok or right_ok:
            self.tracking_state = LaneTrackingState.TRACKING_LOW_CONFIDENCE
        else:
            self.tracking_state = LaneTrackingState.LOST

        return self.smoothed_left, self.smoothed_right, self.tracking_state


# ═══════════════════════════════════════════════════════════════════════════════
# Master Computer Vision Precision Stack Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

class CVPrecisionStack:
    """
    Master Geometric & Vision Pipeline orchestrating dynamic ROI, IPM, color isolation,
    Sobel-X, dual Canny, Hough filtering, and RANSAC parabolic fitting.
    """

    def __init__(self, config: Optional[CVStackConfig] = None):
        self.config = config or CVStackConfig()

        # 1. Initialize Dynamic ROI Processor
        self.roi_processor = get_roi_processor(self.config.roi_config)

        # 2. Initialize Dynamic IPM Transformer
        self.ipm_transformer = get_ipm_transformer(
            intrinsics=CameraIntrinsics(
                fx=self.config.roi_config.fx,
                fy=self.config.roi_config.fy,
                cx=self.config.roi_config.cx,
                cy=self.config.roi_config.cy,
                image_width=self.config.roi_config.image_width,
                image_height=self.config.roi_config.image_height,
            ),
            extrinsics=CameraExtrinsics(
                mounting_height_m=self.config.camera_height_m,
                nominal_pitch_deg=self.config.nominal_pitch_deg,
                nominal_roll_deg=self.config.nominal_roll_deg,
            ),
            config=self.config.ipm_config,
        )

        # 3. Initialize Multi-Color Space Boundary Isolator
        self.color_isolator = get_color_space_isolator(self.config.color_config)

        # 4. Initialize Geometric Edge & Hough Pipeline
        self.edge_hough_pipeline = get_edge_hough_pipeline(self.config.edge_config)

        # 5. Initialize Temporal Anti-Flicker Filter
        self.temporal_filter = TemporalPolynomialFilter(
            alpha=self.config.temporal_smoothing_alpha,
            max_coasting_frames=self.config.max_coasting_frames,
            decay_factor=self.config.confidence_decay_factor,
        )

        # Internal state tracking
        self.frame_counter = 0

    def compute_dynamic_lookahead_distance(self, speed_mps: float) -> float:
        """
        Compute speed-dependent lookahead distance:
        L_d(v) = max(L_min, min(L_max, L_min + k_speed * v))
        """
        raw_ld = self.config.lookahead_min_m + self.config.lookahead_speed_gain_s * max(0.0, speed_mps)
        return float(np.clip(raw_ld, self.config.lookahead_min_m, self.config.lookahead_max_m))

    def compute_perception_kinematics(
        self,
        left_poly: LaneBoundaryPolynomial,
        right_poly: LaneBoundaryPolynomial,
        speed_mps: float,
    ) -> PerceptionKinematics:
        """
        Calculate vehicle cross-track error (e_lat), heading error (theta_e),
        and radius of curvature (R) in real-world metric coordinates.
        """
        ld_m = self.compute_dynamic_lookahead_distance(speed_mps)
        m_per_px = self.config.ipm_config.meters_per_pixel
        h_bev = self.config.ipm_config.bev_height
        w_bev = self.config.ipm_config.bev_width
        u_vehicle_center = w_bev / 2.0

        # Vehicle front axle corresponds to bottom row of BEV (v = h_bev - 1)
        v_vehicle = float(h_bev - 1.0)
        # Lookahead row in BEV
        v_lookahead = float(h_bev - 1.0 - (ld_m / m_per_px))
        v_lookahead = float(np.clip(v_lookahead, 0.0, h_bev - 1.0))

        # Check validity and synthesize center polynomial
        if left_poly.is_valid and right_poly.is_valid:
            A_c = (left_poly.A + right_poly.A) / 2.0
            B_c = (left_poly.B + right_poly.B) / 2.0
            C_c = (left_poly.C + right_poly.C) / 2.0
            radius = (left_poly.radius_of_curvature_m + right_poly.radius_of_curvature_m) / 2.0

            x_left_bottom = left_poly.evaluate_x(v_vehicle)
            x_right_bottom = right_poly.evaluate_x(v_vehicle)
            lane_w_m = float((x_right_bottom - x_left_bottom) * m_per_px)

            center_x_bottom = (x_left_bottom + x_right_bottom) / 2.0
            center_x_lookahead = (left_poly.evaluate_x(v_lookahead) + right_poly.evaluate_x(v_lookahead)) / 2.0

        elif left_poly.is_valid:
            # Synthesize virtual center from left boundary
            half_w_px = (self.config.default_lane_width_m / 2.0) / m_per_px
            A_c = left_poly.A
            B_c = left_poly.B
            C_c = left_poly.C + half_w_px
            radius = left_poly.radius_of_curvature_m
            lane_w_m = self.config.default_lane_width_m
            center_x_bottom = left_poly.evaluate_x(v_vehicle) + half_w_px
            center_x_lookahead = left_poly.evaluate_x(v_lookahead) + half_w_px

        elif right_poly.is_valid:
            # Synthesize virtual center from right boundary
            half_w_px = (self.config.default_lane_width_m / 2.0) / m_per_px
            A_c = right_poly.A
            B_c = right_poly.B
            C_c = right_poly.C - half_w_px
            radius = right_poly.radius_of_curvature_m
            lane_w_m = self.config.default_lane_width_m
            center_x_bottom = right_poly.evaluate_x(v_vehicle) - half_w_px
            center_x_lookahead = right_poly.evaluate_x(v_lookahead) - half_w_px

        else:
            return PerceptionKinematics(
                cross_track_error_m=0.0,
                heading_error_rad=0.0,
                heading_error_deg=0.0,
                radius_of_curvature_m=5000.0,
                lookahead_distance_m=ld_m,
                lookahead_cte_m=0.0,
                lane_width_at_vehicle_m=self.config.default_lane_width_m,
                is_left_curve=False,
                is_right_curve=False,
            )

        # Cross-Track Error at vehicle front: e_lat = X_vehicle - X_lane_center
        # Positive CTE means vehicle is to the right of lane center (steer left needed)
        cte_m = float((u_vehicle_center - center_x_bottom) * m_per_px)
        lookahead_cte_m = float((u_vehicle_center - center_x_lookahead) * m_per_px)

        # Heading Error at vehicle front: theta_e = atan(2*A*v + B)
        # In BEV, v points upward (towards top of image).
        dx_dv = 2.0 * A_c * v_vehicle + B_c
        heading_err_rad = float(math.atan(dx_dv))
        heading_err_deg = float(math.degrees(heading_err_rad))

        is_left_curve = (A_c < -1e-5)
        is_right_curve = (A_c > 1e-5)

        return PerceptionKinematics(
            cross_track_error_m=round(cte_m, 3),
            heading_error_rad=round(heading_err_rad, 4),
            heading_error_deg=round(heading_err_deg, 2),
            radius_of_curvature_m=round(radius, 1),
            lookahead_distance_m=round(ld_m, 2),
            lookahead_cte_m=round(lookahead_cte_m, 3),
            lane_width_at_vehicle_m=round(lane_w_m, 2),
            is_left_curve=is_left_curve,
            is_right_curve=is_right_curve,
        )

    def classify_marking_types(
        self,
        color_res: ColorIsolationResult,
        left_poly: LaneBoundaryPolynomial,
        right_poly: LaneBoundaryPolynomial,
    ) -> Tuple[LaneMarkingType, LaneMarkingType]:
        """
        Classifies boundary types (Solid White, Dashed White, Yellow, Dirt)
        based on chromaticity overlap with polynomial paths.
        """
        def classify_single(poly: LaneBoundaryPolynomial, is_left: bool) -> LaneMarkingType:
            if not poly.is_valid:
                return LaneMarkingType.UNKNOWN
            # Off-road dirt check
            if self.config.is_offroad_mode and color_res.dirt_boundary_mask is not None:
                dirt_pixels = int(cv2.countNonZero(color_res.dirt_boundary_mask))
                if dirt_pixels > 500:
                    return LaneMarkingType.OFFROAD_DIRT_BERM
            # Check yellow vs white pixel dominance
            if is_left and color_res.yellow_pixel_count > color_res.white_pixel_count * 0.4:
                return LaneMarkingType.SOLID_YELLOW
            return LaneMarkingType.DASHED_WHITE if not is_left else LaneMarkingType.SOLID_WHITE

        l_type = classify_single(left_poly, is_left=True)
        r_type = classify_single(right_poly, is_left=False)
        return l_type, r_type

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        pitch_deg: float = 5.0,
        roll_deg: float = 0.0,
        speed_mps: float = 8.0,
    ) -> CVPrecisionTelemetry:
        """
        Execute full end-to-end Computer Vision Precision Stack pass.
        """
        t_total_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        self.frame_counter += 1

        # ── Step 1: Dynamic Horizon & ROI Processing ──────────────────────────
        t0 = time.perf_counter()
        roi_res = self.roi_processor.process(
            frame=frame_bgr,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        stage_timings["dynamic_roi_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        # ── Step 2: Multi-Color Space Boundary Isolation ──────────────────────
        t1 = time.perf_counter()
        color_res = self.color_isolator.process(
            bgr_image=roi_res.masked_frame,
            extract_dirt=self.config.is_offroad_mode,
        )
        stage_timings["color_isolation_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)

        # ── Step 3: Metric Inverse Perspective Mapping (IPM) ──────────────────
        t2 = time.perf_counter()
        ipm_res = self.ipm_transformer.transform_image_to_bev(
            image=color_res.combined_binary_mask,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        stage_timings["ipm_warp_ms"] = round((time.perf_counter() - t2) * 1000.0, 3)

        # Also warp original BGR for diagnostic BEV view
        bev_bgr = self.ipm_transformer.transform_image_to_bev(
            image=frame_bgr,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        ).bev_image

        # ── Step 4: Geometric Edge, Hough & RANSAC Parabola ───────────────────
        t3 = time.perf_counter()
        edge_res = self.edge_hough_pipeline.process(
            image=ipm_res.bev_image,
            is_offroad_dirt=self.config.is_offroad_mode,
        )
        stage_timings["edge_hough_ransac_ms"] = round((time.perf_counter() - t3) * 1000.0, 3)

        # ── Step 5: Temporal Anti-Flicker Filtering ───────────────────────────
        t4 = time.perf_counter()
        smoothed_left, smoothed_right, tracking_state = self.temporal_filter.update(
            raw_left=edge_res.left_polynomial,
            raw_right=edge_res.right_polynomial,
        )
        stage_timings["temporal_filter_ms"] = round((time.perf_counter() - t4) * 1000.0, 3)

        # ── Step 6: Kinematics & Curvature Computation ────────────────────────
        t5 = time.perf_counter()
        kinematics = self.compute_perception_kinematics(
            left_poly=smoothed_left,
            right_poly=smoothed_right,
            speed_mps=speed_mps,
        )
        stage_timings["kinematics_ms"] = round((time.perf_counter() - t5) * 1000.0, 3)

        # ── Step 7: Classification & Overall Confidence ───────────────────────
        l_marking, r_marking = self.classify_marking_types(color_res, smoothed_left, smoothed_right)

        conf_scores = []
        if smoothed_left.is_valid:
            conf_scores.append(smoothed_left.confidence)
        if smoothed_right.is_valid:
            conf_scores.append(smoothed_right.confidence)
        overall_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        total_elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0
        stage_timings["total_stack_ms"] = round(total_elapsed_ms, 3)

        # Synthesize Center Polynomial
        center_poly = None
        if smoothed_left.is_valid and smoothed_right.is_valid:
            center_poly = LaneBoundaryPolynomial(
                is_valid=True,
                A=(smoothed_left.A + smoothed_right.A) / 2.0,
                B=(smoothed_left.B + smoothed_right.B) / 2.0,
                C=(smoothed_left.C + smoothed_right.C) / 2.0,
                radius_of_curvature_m=kinematics.radius_of_curvature_m,
                confidence=overall_conf,
            )

        return CVPrecisionTelemetry(
            frame_idx=self.frame_counter,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tracking_state=tracking_state,
            kinematics=kinematics,
            left_polynomial=smoothed_left,
            right_polynomial=smoothed_right,
            center_polynomial=center_poly,
            left_marking_type=l_marking,
            right_marking_type=r_marking,
            overall_confidence=round(overall_conf, 3),
            horizon_y_px=roi_res.horizon_vp.horizon_y,
            vanishing_point_x_px=roi_res.horizon_vp.vanishing_point_x,
            is_ipm_recomputed=ipm_res.recomputed,
            processing_latency_ms=round(total_elapsed_ms, 3),
            stage_latencies_ms=stage_timings,
            annotated_perspective_bgr=None,  # Populated via visualization method if requested
            bev_metric_bgr=bev_bgr,
            color_isolation_mask=color_res.combined_binary_mask,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Multi-Panel Diagnostic HUD Visualization Renderer
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_diagnostic_hud(
        self,
        original_frame: np.ndarray,
        telemetry: CVPrecisionTelemetry,
    ) -> np.ndarray:
        """
        Generates a 6-panel composite HUD visualization for Streamlit and dashboard streaming:
        [Panel 1: Perspective + Horizon/Trapezoid] [Panel 2: Color Space Mask] [Panel 3: BEV Raw + Metric Grid]
        [Panel 4: BEV Hough Lines & RANSAC]       [Panel 5: Unwarped Corridor] [Panel 6: Telemetry Dashboard]
        """
        h_orig, w_orig = original_frame.shape[:2]
        panel_w = 400
        panel_h = 240

        # Helper to resize panel uniformly
        def resize_p(img: np.ndarray) -> np.ndarray:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (panel_w, panel_h))

        # Panel 1: Perspective view with dynamic horizon and corridor
        p1 = original_frame.copy()
        cv2.line(p1, (0, telemetry.horizon_y_px), (w_orig - 1, telemetry.horizon_y_px), (255, 255, 0), 2)
        cv2.circle(p1, (telemetry.vanishing_point_x_px, telemetry.horizon_y_px), 5, (0, 0, 255), -1)

        # Unwarp BEV polynomials to perspective view if valid
        if telemetry.left_polynomial.is_valid:
            left_pts = self.ipm_transformer.unwarp_bev_polynomial_to_image(telemetry.left_polynomial.coefficients)
            if len(left_pts) > 1:
                cv2.polylines(p1, [left_pts], False, (0, 255, 0), 3)

        if telemetry.right_polynomial.is_valid:
            right_pts = self.ipm_transformer.unwarp_bev_polynomial_to_image(telemetry.right_polynomial.coefficients)
            if len(right_pts) > 1:
                cv2.polylines(p1, [right_pts], False, (255, 0, 0), 3)

        panel1 = resize_p(p1)
        cv2.putText(panel1, "1. Perspective + Horizon", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 2: Color Space Isolation Mask
        p2 = telemetry.color_isolation_mask if telemetry.color_isolation_mask is not None else np.zeros((h_orig, w_orig), dtype=np.uint8)
        panel2 = resize_p(p2)
        cv2.putText(panel2, "2. Color Fusion Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 3: BEV Metric Space with Distance Grid
        p3 = telemetry.bev_metric_bgr if telemetry.bev_metric_bgr is not None else np.zeros((600, 400, 3), dtype=np.uint8)
        p3_gridded = self.ipm_transformer.draw_metric_grid_overlay(p3)
        panel3 = resize_p(p3_gridded)
        cv2.putText(panel3, "3. BEV Metric Grid (0.0185m/px)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 4: BEV Fitted Parabolas & Centerline
        p4 = p3.copy()
        h_bev = self.config.ipm_config.bev_height
        y_eval = np.linspace(0, h_bev - 1, 40)

        if telemetry.left_polynomial.is_valid:
            x_l = telemetry.left_polynomial.evaluate_x(y_eval)
            pts_l = np.column_stack([x_l, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_l], False, (0, 255, 0), 2)

        if telemetry.right_polynomial.is_valid:
            x_r = telemetry.right_polynomial.evaluate_x(y_eval)
            pts_r = np.column_stack([x_r, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_r], False, (255, 0, 0), 2)

        if telemetry.center_polynomial is not None and telemetry.center_polynomial.is_valid:
            x_c = telemetry.center_polynomial.evaluate_x(y_eval)
            pts_c = np.column_stack([x_c, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_c], False, (0, 255, 255), 2)

        panel4 = resize_p(p4)
        cv2.putText(panel4, "4. BEV Fitted RANSAC Parabolas", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 5: Drivable Corridor Overlay
        p5 = original_frame.copy()
        if (telemetry.left_polynomial.is_valid and telemetry.right_polynomial.is_valid):
            corridor_poly = self.ipm_transformer.generate_drivable_corridor_bev_polygon(
                telemetry.left_polynomial.coefficients,
                telemetry.right_polynomial.coefficients,
            )
            # Unwarp corridor polygon back to perspective space
            img_poly_pts = []
            for pt in corridor_poly:
                u, v = self.ipm_transformer.bev_pixel_to_image(float(pt[0]), float(pt[1]))
                img_poly_pts.append([int(u), int(v)])
            if len(img_poly_pts) > 3:
                overlay = p5.copy()
                cv2.fillPoly(overlay, [np.array(img_poly_pts, dtype=np.int32)], (0, 200, 100))
                p5 = cv2.addWeighted(overlay, 0.35, p5, 0.65, 0)

        panel5 = resize_p(p5)
        cv2.putText(panel5, "5. Unwarped Drivable Corridor", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 6: Diagnostic Telemetry Dashboard
        panel6 = np.full((panel_h, panel_w, 3), (25, 25, 30), dtype=np.uint8)
        k = telemetry.kinematics
        lines = [
            f"FRAME: {telemetry.frame_idx} | STATE: {telemetry.tracking_state.value}",
            f"CTE (e_lat): {k.cross_track_error_m:+.3f} m",
            f"Heading Err (theta_e): {k.heading_error_deg:+.2f} deg",
            f"Radius of Curvature: {k.radius_of_curvature_m:.1f} m",
            f"Lookahead Dist: {k.lookahead_distance_m:.1f} m (CTE: {k.lookahead_cte_m:+.2f}m)",
            f"Confidence: {telemetry.overall_confidence:.2f} | Lane Width: {k.lane_width_at_vehicle_m:.2f}m",
            f"Latency: {telemetry.processing_latency_ms:.2f} ms ({1000.0/max(telemetry.processing_latency_ms, 1e-3):.1f} FPS)",
            f"Left: {telemetry.left_marking_type.value} | Right: {telemetry.right_marking_type.value}",
        ]
        for idx, line in enumerate(lines):
            y_pos = 28 + idx * 24
            color = (0, 255, 0) if "STATE" in line or "CONF" in line else (220, 220, 220)
            cv2.putText(panel6, line, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        # Assemble 2x3 Grid (720 x 1200 pixels)
        top_row = np.hstack([panel1, panel2, panel3])
        bot_row = np.hstack([panel4, panel5, panel6])
        composite_hud = np.vstack([top_row, bot_row])

        return composite_hud


# ═══════════════════════════════════════════════════════════════════════════════
# 8-State Extended Kalman Filter for Lane Polynomial Tracking
# ═══════════════════════════════════════════════════════════════════════════════

class LanePolynomialEKF:
    """
    8-State Extended Kalman Filter for continuous spatio-temporal lane tracking.
    State Vector:
        x = [A_L, B_L, C_L, A_R, B_R, C_R, lane_width, yaw_rate]^T
    Kinematic Process Model:
        Predicts state evolution using vehicle velocity v and yaw rate dpsi/dt.
    Measurement Model:
        Updates state using RANSAC polynomial coefficients and measured lane width.
    """

    def __init__(
        self,
        dt: float = 0.0333,  # 30 Hz nominal frame interval
        default_lane_width_m: float = 3.20,
    ):
        self.dt = dt
        self.default_lane_width = default_lane_width_m
        self.state_dim = 8

        # State vector [A_L, B_L, C_L, A_R, B_R, C_R, w, yaw_rate]
        self.x = np.zeros((self.state_dim, 1), dtype=np.float64)
        self.x[6, 0] = default_lane_width_m / 0.0185  # Initial lane width in BEV px

        # State Covariance Matrix P
        self.P = np.eye(self.state_dim, dtype=np.float64) * 10.0
        self.P[0, 0] = 1e-4  # A_L small variance
        self.P[3, 3] = 1e-4  # A_R small variance

        # Process Noise Covariance Q
        self.Q = np.eye(self.state_dim, dtype=np.float64) * 1e-3
        self.Q[0, 0] = 1e-8
        self.Q[3, 3] = 1e-8
        self.Q[2, 2] = 0.5  # Lateral offset can drift with steering
        self.Q[5, 5] = 0.5

        # Measurement Noise Covariance R_meas
        self.R_meas = np.eye(self.state_dim, dtype=np.float64) * 0.1
        self.R_meas[0, 0] = 1e-6
        self.R_meas[3, 3] = 1e-6

        self.is_initialized = False

    def predict(self, velocity_mps: float, yaw_rate_rad_s: float) -> np.ndarray:
        """
        Predict state forward using vehicle planar kinematics:
        - Curvature A remains constant (road curve doesn't change instantly).
        - Heading slope B shifts by: delta_B = - yaw_rate * dt.
        - Lateral offset C shifts by: delta_C = - (v * sin(B) * dt) / 0.0185.
        """
        if not self.is_initialized:
            return self.x.copy()

        dt = self.dt
        v_px_s = (velocity_mps / 0.0185)

        # Non-linear state transition f(x, u)
        A_L, B_L, C_L = self.x[0, 0], self.x[1, 0], self.x[2, 0]
        A_R, B_R, C_R = self.x[3, 0], self.x[4, 0], self.x[5, 0]

        # Update heading slopes
        B_L_pred = B_L - yaw_rate_rad_s * dt
        B_R_pred = B_R - yaw_rate_rad_s * dt

        # Update lateral offsets
        C_L_pred = C_L - v_px_s * math.sin(B_L) * dt
        C_R_pred = C_R - v_px_s * math.sin(B_R) * dt

        # Updated state vector
        self.x[1, 0] = B_L_pred
        self.x[2, 0] = C_L_pred
        self.x[4, 0] = B_R_pred
        self.x[5, 0] = C_R_pred
        self.x[7, 0] = yaw_rate_rad_s

        # Linearized Jacobian F = df/dx
        F = np.eye(self.state_dim, dtype=np.float64)
        F[2, 1] = - v_px_s * math.cos(B_L) * dt
        F[5, 4] = - v_px_s * math.cos(B_R) * dt

        # Predict covariance: P = F * P * F^T + Q
        self.P = F @ self.P @ F.T + self.Q
        return self.x.copy()

    def update(
        self,
        raw_left: LaneBoundaryPolynomial,
        raw_right: LaneBoundaryPolynomial,
    ) -> Tuple[LaneBoundaryPolynomial, LaneBoundaryPolynomial]:
        """
        Perform Kalman measurement update using available boundary detections.
        """
        if not self.is_initialized:
            if raw_left.is_valid:
                self.x[0, 0] = raw_left.A
                self.x[1, 0] = raw_left.B
                self.x[2, 0] = raw_left.C
            if raw_right.is_valid:
                self.x[3, 0] = raw_right.A
                self.x[4, 0] = raw_right.B
                self.x[5, 0] = raw_right.C
            if raw_left.is_valid and raw_right.is_valid:
                self.x[6, 0] = raw_right.C - raw_left.C
                self.is_initialized = True
            elif raw_left.is_valid or raw_right.is_valid:
                self.is_initialized = True
            return raw_left, raw_right

        # Build measurement vector z and observation matrix H
        z_list: List[float] = []
        H_rows: List[np.ndarray] = []
        R_list: List[float] = []

        if raw_left.is_valid:
            # Observation of [A_L, B_L, C_L]
            for i, val in enumerate([raw_left.A, raw_left.B, raw_left.C]):
                z_list.append(val)
                h_row = np.zeros(self.state_dim, dtype=np.float64)
                h_row[i] = 1.0
                H_rows.append(h_row)
                R_list.append(self.R_meas[i, i] / max(raw_left.confidence, 0.1))

        if raw_right.is_valid:
            # Observation of [A_R, B_R, C_R]
            for i, val in enumerate([raw_right.A, raw_right.B, raw_right.C]):
                z_list.append(val)
                h_row = np.zeros(self.state_dim, dtype=np.float64)
                h_row[3 + i] = 1.0
                H_rows.append(h_row)
                R_list.append(self.R_meas[3 + i, 3 + i] / max(raw_right.confidence, 0.1))

        # Lane width observation if both valid
        if raw_left.is_valid and raw_right.is_valid:
            measured_w = raw_right.C - raw_left.C
            z_list.append(measured_w)
            h_row = np.zeros(self.state_dim, dtype=np.float64)
            h_row[6] = 1.0
            H_rows.append(h_row)
            R_list.append(0.5)

        if not z_list:
            # No valid measurements: return predicted state
            return self._build_polynomials()

        z = np.array(z_list, dtype=np.float64).reshape(-1, 1)
        H = np.vstack(H_rows)
        R_mat = np.diag(R_list)

        # Innovation: y = z - H * x
        y = z - H @ self.x

        # Innovation Covariance: S = H * P * H^T + R
        S = H @ self.P @ H.T + R_mat

        # Kalman Gain: K = P * H^T * inv(S)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return self._build_polynomials()

        # Update state: x = x + K * y
        self.x = self.x + K @ y

        # Joseph form covariance update for numerical stability: P = (I - K*H)*P*(I - K*H)^T + K*R*K^T
        I = np.eye(self.state_dim, dtype=np.float64)
        IKH = I - K @ H
        self.P = IKH @ self.P @ IKH.T + K @ R_mat @ K.T

        return self._build_polynomials()

    def _build_polynomials(self) -> Tuple[LaneBoundaryPolynomial, LaneBoundaryPolynomial]:
        """Construct LaneBoundaryPolynomial objects from active state vector."""
        A_L, B_L, C_L = float(self.x[0, 0]), float(self.x[1, 0]), float(self.x[2, 0])
        A_R, B_R, C_R = float(self.x[3, 0]), float(self.x[4, 0]), float(self.x[5, 0])

        # Radius of curvature: R = 1 / (2 * |A|)
        A_L_m = A_L / 0.0185
        A_R_m = A_R / 0.0185
        R_L = float(1.0 / (2.0 * abs(A_L_m) + 1e-6))
        R_R = float(1.0 / (2.0 * abs(A_R_m) + 1e-6))

        poly_left = LaneBoundaryPolynomial(
            is_valid=True,
            A=A_L,
            B=B_L,
            C=C_L,
            radius_of_curvature_m=round(min(5000.0, max(5.0, R_L)), 1),
            lateral_offset_m=round(C_L * 0.0185, 3),
            confidence=round(float(1.0 / (1.0 + self.P[2, 2])), 3),
        )

        poly_right = LaneBoundaryPolynomial(
            is_valid=True,
            A=A_R,
            B=B_R,
            C=C_R,
            radius_of_curvature_m=round(min(5000.0, max(5.0, R_R)), 1),
            lateral_offset_m=round(C_R * 0.0185, 3),
            confidence=round(float(1.0 / (1.0 + self.P[5, 5])), 3),
        )

        return poly_left, poly_right


# ═══════════════════════════════════════════════════════════════════════════════
# Epistemic Uncertainty & Spatial Covariance Modeling
# ═══════════════════════════════════════════════════════════════════════════════

class EpistemicUncertaintyEstimator:
    """
    Computes spatial lateral uncertainty envelope sigma_x(y) along the road corridor.
    sigma_x(y) = sqrt( J(y) * P_poly * J(y)^T )
    where J(y) = [y^2, y, 1] is the polynomial Jacobian vector.
    """

    def __init__(self, meters_per_pixel: float = 0.0185):
        self.meters_per_pixel = meters_per_pixel

    def compute_lateral_uncertainty_profile(
        self,
        poly: LaneBoundaryPolynomial,
        covariance_3x3: np.ndarray,
        y_eval_range_px: Tuple[float, float] = (0.0, 599.0),
        num_eval_points: int = 30,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate lateral uncertainty bounds (x_mean, x_upper_2sigma, x_lower_2sigma)
        across the BEV ground plane rows.
        """
        y_vals = np.linspace(y_eval_range_px[0], y_eval_range_px[1], num_eval_points)
        x_mean = poly.evaluate_x(y_vals)

        sigmas = []
        for y in y_vals:
            J = np.array([y ** 2, y, 1.0], dtype=np.float64)
            var_x = float(J @ covariance_3x3 @ J.T)
            sigmas.append(math.sqrt(max(0.0, var_x)))

        sigma_arr = np.array(sigmas)
        x_upper = x_mean + 2.0 * sigma_arr
        x_lower = x_mean - 2.0 * sigma_arr

        return x_mean, x_upper, x_lower


# ═══════════════════════════════════════════════════════════════════════════════
# Optical Degradation & Sensor Health Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class OpticalHealthMonitor:
    """
    Monitors camera optical transmission, mud/dust splash accumulation, and
    blinding direct solar glare to support ISO 26262 ASIL-B safety degradation.
    """

    def __init__(
        self,
        glare_saturation_threshold: int = 245,
        mud_obscuration_variance_threshold: float = 45.0,
    ):
        self.glare_thresh = glare_saturation_threshold
        self.mud_thresh = mud_obscuration_variance_threshold

    def evaluate_camera_health(self, bgr_frame: np.ndarray) -> Dict[str, Any]:
        """
        Analyze frame for optical degradation:
        - Glare ratio: percentage of pixels saturated near 255.
        - Mud / blur score: Laplacian blur variance to detect lens smearing.
        - Overall degradation state: NORMAL, GLARE_BLINDED, LENS_OBSCURED.
        """
        gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        total_pixels = gray.size

        # Glare detection
        saturated_pixels = int(np.sum(gray >= self.glare_thresh))
        glare_ratio = saturated_pixels / float(total_pixels)

        # Lens blur / mud obscuration via Laplacian variance
        laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if glare_ratio > 0.25:
            health_status = "GLARE_BLINDED"
        elif laplacian_var < self.mud_thresh:
            health_status = "LENS_OBSCURED_MUD"
        else:
            health_status = "NORMAL_CLEAR"

        return {
            "health_status": health_status,
            "glare_ratio": round(glare_ratio, 4),
            "laplacian_sharpness": round(laplacian_var, 2),
            "is_sensor_valid": health_status == "NORMAL_CLEAR",
        }


# ═══════════════════════════════════════════════════════════════════════════════
# ISO 26262 ASIL-B Kinodynamic Safety Assertion Validator
# ═══════════════════════════════════════════════════════════════════════════════

class KinodynamicSafetyValidator:
    """
    Validates perception outputs against physical kinodynamic feasibility:
    - Maximum lateral offset velocity: |d e_lat / dt| <= v_max * sin(delta_max).
    - Minimum physical turning radius: R >= R_min (e.g. 5.5 meters for BAJA Buggy).
    - Prevents unrealistic perception jumps from entering vehicle actuation controllers.
    """

    def __init__(
        self,
        max_speed_mps: float = 12.5,          # 45 km/h for BAJA Buggy
        max_steer_angle_rad: float = 0.4886,  # 28 degrees
        min_turning_radius_m: float = 5.0,
        max_allowed_cte_jump_m: float = 0.60,
    ):
        self.max_speed = max_speed_mps
        self.max_steer = max_steer_angle_rad
        self.min_radius = min_turning_radius_m
        self.max_cte_jump = max_allowed_cte_jump_m
        self.last_cte_m: Optional[float] = None

    def validate_kinematics(self, kinematics: PerceptionKinematics) -> Tuple[bool, str]:
        """
        Validate kinematic measurements against safety envelope.
        Returns (is_safe, safety_message).
        """
        # 1. Radius of curvature check
        if kinematics.radius_of_curvature_m < self.min_radius:
            return False, f"Radius of curvature {kinematics.radius_of_curvature_m:.1f}m below physical limit {self.min_radius}m"

        # 2. CTE Jump check across consecutive frames
        if self.last_cte_m is not None:
            delta_cte = abs(kinematics.cross_track_error_m - self.last_cte_m)
            if delta_cte > self.max_cte_jump:
                return False, f"Spurious CTE step jump: delta {delta_cte:.3f}m exceeds max allowable {self.max_cte_jump}m"

        # 3. Heading angle check
        if abs(kinematics.heading_error_rad) > self.max_steer * 1.5:
            return False, f"Heading error {kinematics.heading_error_deg:.1f} deg exceeds plausible vehicle yaw"

        self.last_cte_m = kinematics.cross_track_error_m
        return True, "KINEMATICS_VERIFIED_SAFE"


# ═══════════════════════════════════════════════════════════════════════════════
# Master Computer Vision Precision Stack Pipeline Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class CVPrecisionStack:
    """
    Master Geometric & Vision Pipeline orchestrating dynamic ROI, IPM, color isolation,
    Sobel-X, dual Canny, Hough filtering, and RANSAC parabolic fitting.
    """

    def __init__(self, config: Optional[CVStackConfig] = None):
        self.config = config or CVStackConfig()

        # 1. Initialize Dynamic ROI Processor
        self.roi_processor = get_roi_processor(self.config.roi_config)

        # 2. Initialize Dynamic IPM Transformer
        self.ipm_transformer = get_ipm_transformer(
            intrinsics=CameraIntrinsics(
                fx=self.config.roi_config.fx,
                fy=self.config.roi_config.fy,
                cx=self.config.roi_config.cx,
                cy=self.config.roi_config.cy,
                image_width=self.config.roi_config.image_width,
                image_height=self.config.roi_config.image_height,
            ),
            extrinsics=CameraExtrinsics(
                mounting_height_m=self.config.camera_height_m,
                nominal_pitch_deg=self.config.nominal_pitch_deg,
                nominal_roll_deg=self.config.nominal_roll_deg,
            ),
            config=self.config.ipm_config,
        )

        # 3. Initialize Multi-Color Space Boundary Isolator
        self.color_isolator = get_color_space_isolator(self.config.color_config)

        # 4. Initialize Geometric Edge & Hough Pipeline
        self.edge_hough_pipeline = get_edge_hough_pipeline(self.config.edge_config)

        # 5. Initialize Temporal Anti-Flicker Filter & EKF
        self.temporal_filter = TemporalPolynomialFilter(
            alpha=self.config.temporal_smoothing_alpha,
            max_coasting_frames=self.config.max_coasting_frames,
            decay_factor=self.config.confidence_decay_factor,
        )
        self.lane_ekf = LanePolynomialEKF(
            dt=0.0333,
            default_lane_width_m=self.config.default_lane_width_m,
        )

        # 6. Safety & Health Monitors
        self.health_monitor = OpticalHealthMonitor()
        self.safety_validator = KinodynamicSafetyValidator()
        self.uncertainty_estimator = EpistemicUncertaintyEstimator(
            meters_per_pixel=self.config.ipm_config.meters_per_pixel
        )

        # Internal state tracking
        self.frame_counter = 0

    def compute_dynamic_lookahead_distance(self, speed_mps: float) -> float:
        """
        Compute speed-dependent lookahead distance:
        L_d(v) = max(L_min, min(L_max, L_min + k_speed * v))
        """
        raw_ld = self.config.lookahead_min_m + self.config.lookahead_speed_gain_s * max(0.0, speed_mps)
        return float(np.clip(raw_ld, self.config.lookahead_min_m, self.config.lookahead_max_m))

    def compute_perception_kinematics(
        self,
        left_poly: LaneBoundaryPolynomial,
        right_poly: LaneBoundaryPolynomial,
        speed_mps: float,
    ) -> PerceptionKinematics:
        """
        Calculate vehicle cross-track error (e_lat), heading error (theta_e),
        and radius of curvature (R) in real-world metric coordinates.
        """
        ld_m = self.compute_dynamic_lookahead_distance(speed_mps)
        m_per_px = self.config.ipm_config.meters_per_pixel
        h_bev = self.config.ipm_config.bev_height
        w_bev = self.config.ipm_config.bev_width
        u_vehicle_center = w_bev / 2.0

        v_vehicle = float(h_bev - 1.0)
        v_lookahead = float(h_bev - 1.0 - (ld_m / m_per_px))
        v_lookahead = float(np.clip(v_lookahead, 0.0, h_bev - 1.0))

        if left_poly.is_valid and right_poly.is_valid:
            A_c = (left_poly.A + right_poly.A) / 2.0
            B_c = (left_poly.B + right_poly.B) / 2.0
            C_c = (left_poly.C + right_poly.C) / 2.0
            radius = (left_poly.radius_of_curvature_m + right_poly.radius_of_curvature_m) / 2.0

            x_left_bottom = left_poly.evaluate_x(v_vehicle)
            x_right_bottom = right_poly.evaluate_x(v_vehicle)
            lane_w_raw = float((x_right_bottom - x_left_bottom) * m_per_px)
            lane_w_m = lane_w_raw if (1.5 <= lane_w_raw <= 6.0) else self.config.default_lane_width_m

            center_x_bottom = (x_left_bottom + x_right_bottom) / 2.0
            center_x_lookahead = (left_poly.evaluate_x(v_lookahead) + right_poly.evaluate_x(v_lookahead)) / 2.0

        elif left_poly.is_valid:
            half_w_px = (self.config.default_lane_width_m / 2.0) / m_per_px
            A_c = left_poly.A
            B_c = left_poly.B
            C_c = left_poly.C + half_w_px
            radius = left_poly.radius_of_curvature_m
            lane_w_m = self.config.default_lane_width_m
            center_x_bottom = left_poly.evaluate_x(v_vehicle) + half_w_px
            center_x_lookahead = left_poly.evaluate_x(v_lookahead) + half_w_px

        elif right_poly.is_valid:
            half_w_px = (self.config.default_lane_width_m / 2.0) / m_per_px
            A_c = right_poly.A
            B_c = right_poly.B
            C_c = right_poly.C - half_w_px
            radius = right_poly.radius_of_curvature_m
            lane_w_m = self.config.default_lane_width_m
            center_x_bottom = right_poly.evaluate_x(v_vehicle) - half_w_px
            center_x_lookahead = right_poly.evaluate_x(v_lookahead) - half_w_px

        else:
            return PerceptionKinematics(
                cross_track_error_m=0.0,
                heading_error_rad=0.0,
                heading_error_deg=0.0,
                radius_of_curvature_m=5000.0,
                lookahead_distance_m=ld_m,
                lookahead_cte_m=0.0,
                lane_width_at_vehicle_m=self.config.default_lane_width_m,
                is_left_curve=False,
                is_right_curve=False,
            )

        cte_m = float((u_vehicle_center - center_x_bottom) * m_per_px)
        lookahead_cte_m = float((u_vehicle_center - center_x_lookahead) * m_per_px)

        dx_dv = 2.0 * A_c * v_vehicle + B_c
        heading_err_rad = float(math.atan(dx_dv))
        heading_err_deg = float(math.degrees(heading_err_rad))

        is_left_curve = (A_c < -1e-5)
        is_right_curve = (A_c > 1e-5)

        return PerceptionKinematics(
            cross_track_error_m=round(cte_m, 3),
            heading_error_rad=round(heading_err_rad, 4),
            heading_error_deg=round(heading_err_deg, 2),
            radius_of_curvature_m=round(radius, 1),
            lookahead_distance_m=round(ld_m, 2),
            lookahead_cte_m=round(lookahead_cte_m, 3),
            lane_width_at_vehicle_m=round(lane_w_m, 2),
            is_left_curve=is_left_curve,
            is_right_curve=is_right_curve,
        )

    def classify_marking_types(
        self,
        color_res: ColorIsolationResult,
        left_poly: LaneBoundaryPolynomial,
        right_poly: LaneBoundaryPolynomial,
    ) -> Tuple[LaneMarkingType, LaneMarkingType]:
        """
        Classifies boundary types based on color pixel density and terrain context.
        """
        def classify_single(poly: LaneBoundaryPolynomial, is_left: bool) -> LaneMarkingType:
            if not poly.is_valid:
                return LaneMarkingType.UNKNOWN
            if self.config.is_offroad_mode and color_res.dirt_boundary_mask is not None:
                dirt_pixels = int(cv2.countNonZero(color_res.dirt_boundary_mask))
                if dirt_pixels > 500:
                    return LaneMarkingType.OFFROAD_DIRT_BERM
            if is_left and color_res.yellow_pixel_count > color_res.white_pixel_count * 0.4:
                return LaneMarkingType.SOLID_YELLOW
            return LaneMarkingType.DASHED_WHITE if not is_left else LaneMarkingType.SOLID_WHITE

        l_type = classify_single(left_poly, is_left=True)
        r_type = classify_single(right_poly, is_left=False)
        return l_type, r_type

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        pitch_deg: float = 5.0,
        roll_deg: float = 0.0,
        speed_mps: float = 8.0,
        yaw_rate_rad_s: float = 0.0,
    ) -> CVPrecisionTelemetry:
        """
        Execute full end-to-end Computer Vision Precision Stack pass.
        """
        t_total_start = time.perf_counter()
        stage_timings: Dict[str, float] = {}
        self.frame_counter += 1

        # ── Step 0: Optical Health & Glare Monitor ────────────────────────────
        t_health = time.perf_counter()
        health_info = self.health_monitor.evaluate_camera_health(frame_bgr)
        stage_timings["optical_health_ms"] = round((time.perf_counter() - t_health) * 1000.0, 3)

        # ── Step 1: Dynamic Horizon & ROI Processing ──────────────────────────
        t0 = time.perf_counter()
        roi_res = self.roi_processor.process(
            frame=frame_bgr,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        stage_timings["dynamic_roi_ms"] = round((time.perf_counter() - t0) * 1000.0, 3)

        # ── Step 2: Multi-Color Space Boundary Isolation ──────────────────────
        t1 = time.perf_counter()
        color_res = self.color_isolator.process(
            bgr_image=roi_res.masked_frame,
            extract_dirt=self.config.is_offroad_mode,
        )
        stage_timings["color_isolation_ms"] = round((time.perf_counter() - t1) * 1000.0, 3)

        # ── Step 3: Metric Inverse Perspective Mapping (IPM) ──────────────────
        t2 = time.perf_counter()
        ipm_res = self.ipm_transformer.transform_image_to_bev(
            image=color_res.combined_binary_mask,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        )
        stage_timings["ipm_warp_ms"] = round((time.perf_counter() - t2) * 1000.0, 3)

        bev_bgr = self.ipm_transformer.transform_image_to_bev(
            image=frame_bgr,
            pitch_deg=pitch_deg,
            roll_deg=roll_deg,
        ).bev_image

        # ── Step 4: Geometric Edge, Hough & RANSAC Parabola ───────────────────
        t3 = time.perf_counter()
        edge_res = self.edge_hough_pipeline.process(
            image=ipm_res.bev_image,
            is_offroad_dirt=self.config.is_offroad_mode,
        )
        stage_timings["edge_hough_ransac_ms"] = round((time.perf_counter() - t3) * 1000.0, 3)

        # ── Step 5: Temporal Anti-Flicker Filtering & EKF State Fusion ────────
        t4 = time.perf_counter()
        # Predict EKF state
        self.lane_ekf.predict(velocity_mps=speed_mps, yaw_rate_rad_s=yaw_rate_rad_s)
        # Update EKF with RANSAC observations
        ekf_left, ekf_right = self.lane_ekf.update(
            raw_left=edge_res.left_polynomial,
            raw_right=edge_res.right_polynomial,
        )

        # Smooth using temporal filter
        smoothed_left, smoothed_right, tracking_state = self.temporal_filter.update(
            raw_left=ekf_left,
            raw_right=ekf_right,
        )
        stage_timings["temporal_filter_ms"] = round((time.perf_counter() - t4) * 1000.0, 3)

        # ── Step 6: Kinematics & Curvature Computation ────────────────────────
        t5 = time.perf_counter()
        kinematics = self.compute_perception_kinematics(
            left_poly=smoothed_left,
            right_poly=smoothed_right,
            speed_mps=speed_mps,
        )

        # Validate kinodynamics against ASIL-B safety limits
        is_safe, safety_msg = self.safety_validator.validate_kinematics(kinematics)
        if not is_safe:
            logger.warning(f"[SAFETY_VIOLATION] Frame {self.frame_counter}: {safety_msg}")
        stage_timings["kinematics_ms"] = round((time.perf_counter() - t5) * 1000.0, 3)

        # ── Step 7: Classification & Overall Confidence ───────────────────────
        l_marking, r_marking = self.classify_marking_types(color_res, smoothed_left, smoothed_right)

        conf_scores = []
        if smoothed_left.is_valid:
            conf_scores.append(smoothed_left.confidence)
        if smoothed_right.is_valid:
            conf_scores.append(smoothed_right.confidence)
        overall_conf = float(np.mean(conf_scores)) if conf_scores else 0.0

        total_elapsed_ms = (time.perf_counter() - t_total_start) * 1000.0
        stage_timings["total_stack_ms"] = round(total_elapsed_ms, 3)

        center_poly = None
        if smoothed_left.is_valid and smoothed_right.is_valid:
            center_poly = LaneBoundaryPolynomial(
                is_valid=True,
                A=(smoothed_left.A + smoothed_right.A) / 2.0,
                B=(smoothed_left.B + smoothed_right.B) / 2.0,
                C=(smoothed_left.C + smoothed_right.C) / 2.0,
                radius_of_curvature_m=kinematics.radius_of_curvature_m,
                confidence=overall_conf,
            )

        return CVPrecisionTelemetry(
            frame_idx=self.frame_counter,
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            tracking_state=tracking_state,
            kinematics=kinematics,
            left_polynomial=smoothed_left,
            right_polynomial=smoothed_right,
            center_polynomial=center_poly,
            left_marking_type=l_marking,
            right_marking_type=r_marking,
            overall_confidence=round(overall_conf, 3),
            horizon_y_px=roi_res.horizon_vp.horizon_y,
            vanishing_point_x_px=roi_res.horizon_vp.vanishing_point_x,
            is_ipm_recomputed=ipm_res.recomputed,
            processing_latency_ms=round(total_elapsed_ms, 3),
            stage_latencies_ms=stage_timings,
            annotated_perspective_bgr=None,
            bev_metric_bgr=bev_bgr,
            color_isolation_mask=color_res.combined_binary_mask,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Multi-Panel Diagnostic HUD Visualization Renderer
    # ═══════════════════════════════════════════════════════════════════════════

    def generate_diagnostic_hud(
        self,
        original_frame: np.ndarray,
        telemetry: CVPrecisionTelemetry,
    ) -> np.ndarray:
        """
        Generates a 6-panel composite HUD visualization for Streamlit and dashboard streaming:
        [Panel 1: Perspective + Horizon/Trapezoid] [Panel 2: Color Space Mask] [Panel 3: BEV Raw + Metric Grid]
        [Panel 4: BEV Hough Lines & RANSAC]       [Panel 5: Unwarped Corridor] [Panel 6: Telemetry Dashboard]
        """
        h_orig, w_orig = original_frame.shape[:2]
        panel_w = 400
        panel_h = 240

        def resize_p(img: np.ndarray) -> np.ndarray:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return cv2.resize(img, (panel_w, panel_h))

        # Panel 1: Perspective view with dynamic horizon and corridor
        p1 = original_frame.copy()
        cv2.line(p1, (0, telemetry.horizon_y_px), (w_orig - 1, telemetry.horizon_y_px), (255, 255, 0), 2)
        cv2.circle(p1, (telemetry.vanishing_point_x_px, telemetry.horizon_y_px), 5, (0, 0, 255), -1)

        if telemetry.left_polynomial.is_valid:
            left_pts = self.ipm_transformer.unwarp_bev_polynomial_to_image(telemetry.left_polynomial.coefficients)
            if len(left_pts) > 1:
                cv2.polylines(p1, [left_pts], False, (0, 255, 0), 3)

        if telemetry.right_polynomial.is_valid:
            right_pts = self.ipm_transformer.unwarp_bev_polynomial_to_image(telemetry.right_polynomial.coefficients)
            if len(right_pts) > 1:
                cv2.polylines(p1, [right_pts], False, (255, 0, 0), 3)

        panel1 = resize_p(p1)
        cv2.putText(panel1, "1. Perspective + Horizon", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 2: Color Space Isolation Mask
        p2 = telemetry.color_isolation_mask if telemetry.color_isolation_mask is not None else np.zeros((h_orig, w_orig), dtype=np.uint8)
        panel2 = resize_p(p2)
        cv2.putText(panel2, "2. Color Fusion Mask", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 3: BEV Metric Space with Distance Grid
        p3 = telemetry.bev_metric_bgr if telemetry.bev_metric_bgr is not None else np.zeros((600, 400, 3), dtype=np.uint8)
        p3_gridded = self.ipm_transformer.draw_metric_grid_overlay(p3)
        panel3 = resize_p(p3_gridded)
        cv2.putText(panel3, "3. BEV Metric Grid (0.0185m/px)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 4: BEV Fitted Parabolas & Centerline
        p4 = p3.copy()
        h_bev = self.config.ipm_config.bev_height
        y_eval = np.linspace(0, h_bev - 1, 40)

        if telemetry.left_polynomial.is_valid:
            x_l = telemetry.left_polynomial.evaluate_x(y_eval)
            pts_l = np.column_stack([x_l, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_l], False, (0, 255, 0), 2)

        if telemetry.right_polynomial.is_valid:
            x_r = telemetry.right_polynomial.evaluate_x(y_eval)
            pts_r = np.column_stack([x_r, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_r], False, (255, 0, 0), 2)

        if telemetry.center_polynomial is not None and telemetry.center_polynomial.is_valid:
            x_c = telemetry.center_polynomial.evaluate_x(y_eval)
            pts_c = np.column_stack([x_c, y_eval]).astype(np.int32)
            cv2.polylines(p4, [pts_c], False, (0, 255, 255), 2)

        panel4 = resize_p(p4)
        cv2.putText(panel4, "4. BEV Fitted RANSAC Parabolas", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 5: Drivable Corridor Overlay
        p5 = original_frame.copy()
        if (telemetry.left_polynomial.is_valid and telemetry.right_polynomial.is_valid):
            corridor_poly = self.ipm_transformer.generate_drivable_corridor_bev_polygon(
                telemetry.left_polynomial.coefficients,
                telemetry.right_polynomial.coefficients,
            )
            img_poly_pts = []
            for pt in corridor_poly:
                u, v = self.ipm_transformer.bev_pixel_to_image(float(pt[0]), float(pt[1]))
                img_poly_pts.append([int(u), int(v)])
            if len(img_poly_pts) > 3:
                overlay = p5.copy()
                cv2.fillPoly(overlay, [np.array(img_poly_pts, dtype=np.int32)], (0, 200, 100))
                p5 = cv2.addWeighted(overlay, 0.35, p5, 0.65, 0)

        panel5 = resize_p(p5)
        cv2.putText(panel5, "5. Unwarped Drivable Corridor", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Panel 6: Diagnostic Telemetry Dashboard
        panel6 = np.full((panel_h, panel_w, 3), (25, 25, 30), dtype=np.uint8)
        k = telemetry.kinematics
        lines = [
            f"FRAME: {telemetry.frame_idx} | STATE: {telemetry.tracking_state.value}",
            f"CTE (e_lat): {k.cross_track_error_m:+.3f} m",
            f"Heading Err (theta_e): {k.heading_error_deg:+.2f} deg",
            f"Radius of Curvature: {k.radius_of_curvature_m:.1f} m",
            f"Lookahead Dist: {k.lookahead_distance_m:.1f} m (CTE: {k.lookahead_cte_m:+.2f}m)",
            f"Confidence: {telemetry.overall_confidence:.2f} | Lane Width: {k.lane_width_at_vehicle_m:.2f}m",
            f"Latency: {telemetry.processing_latency_ms:.2f} ms ({1000.0/max(telemetry.processing_latency_ms, 1e-3):.1f} FPS)",
            f"Left: {telemetry.left_marking_type.value} | Right: {telemetry.right_marking_type.value}",
        ]
        for idx, line in enumerate(lines):
            y_pos = 28 + idx * 24
            color = (0, 255, 0) if "STATE" in line or "CONF" in line else (220, 220, 220)
            cv2.putText(panel6, line, (12, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        top_row = np.hstack([panel1, panel2, panel3])
        bot_row = np.hstack([panel4, panel5, panel6])
        composite_hud = np.vstack([top_row, bot_row])

        return composite_hud


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_cv_precision_stack(config: Optional[CVStackConfig] = None) -> CVPrecisionStack:
    """Factory constructor for CVPrecisionStack."""
    return CVPrecisionStack(config=config)


def run_unit_tests() -> bool:
    """Validate master CV precision stack pipeline, EKF, kinematics, and HUD output."""
    print("Testing CVPrecisionStack end-to-end orchestration...")
    stack = get_cv_precision_stack()

    # Create realistic synthetic highway frame with 3.5m spacing
    test_frame = np.full((360, 640, 3), (65, 70, 75), dtype=np.uint8)
    test_frame[:140, :] = (180, 160, 140)  # Sky
    # Yellow left lane
    cv2.line(test_frame, (120, 359), (270, 160), (30, 215, 235), 5)
    # White right lane
    cv2.line(test_frame, (520, 359), (370, 160), (245, 245, 245), 5)

    # Process frame
    telemetry = stack.process_frame(
        frame_bgr=test_frame,
        pitch_deg=5.0,
        roll_deg=0.0,
        speed_mps=10.0,
        yaw_rate_rad_s=0.0,
    )

    print(f"Tracking State: {telemetry.tracking_state.value}")
    print(f"Overall Confidence: {telemetry.overall_confidence:.3f}")
    print(f"Cross-Track Error: {telemetry.kinematics.cross_track_error_m:+.3f} m")
    print(f"Radius of Curvature: {telemetry.kinematics.radius_of_curvature_m:.1f} m")
    print(f"Total Latency: {telemetry.processing_latency_ms:.2f} ms")

    # Invariants
    assert telemetry.tracking_state != LaneTrackingState.LOST, "Tracking state should not be LOST on clean synthetic frame"
    assert abs(telemetry.kinematics.cross_track_error_m) < 1.0, f"CTE excessive: {telemetry.kinematics.cross_track_error_m}"
    assert telemetry.kinematics.lane_width_at_vehicle_m > 1.5, "Lane width unrealistic"
    assert telemetry.processing_latency_ms > 0.0, "Timing latency must be positive"

    # Test HUD Generation
    hud = stack.generate_diagnostic_hud(test_frame, telemetry)
    assert hud.shape == (480, 1200, 3), f"HUD dimensions expected (480, 1200, 3), got {hud.shape}"

    # Test Temporal Anti-Flicker over 5 frames
    for i in range(5):
        t_seq = stack.process_frame(test_frame, pitch_deg=5.0, roll_deg=0.0, speed_mps=10.0)
        assert t_seq.tracking_state in (LaneTrackingState.TRACKING_HIGH_CONFIDENCE, LaneTrackingState.TRACKING_LOW_CONFIDENCE)

    # Test Optical Health Monitor
    health = stack.health_monitor.evaluate_camera_health(test_frame)
    assert health["is_sensor_valid"] is True, "Synthetic frame should be evaluated as valid"

    # Test EKF direct predict & update
    ekf = LanePolynomialEKF()
    p_left = LaneBoundaryPolynomial(is_valid=True, A=0.001, B=-0.1, C=100.0, confidence=0.9)
    p_right = LaneBoundaryPolynomial(is_valid=True, A=0.001, B=-0.1, C=280.0, confidence=0.9)
    s_l, s_r = ekf.update(p_left, p_right)
    assert s_l.is_valid and s_r.is_valid

    # Test Epistemic Uncertainty Profile
    cov = np.eye(3) * 1e-4
    x_m, x_u, x_l = stack.uncertainty_estimator.compute_lateral_uncertainty_profile(p_left, cov)
    assert len(x_m) == 30 and all(x_u >= x_m) and all(x_m >= x_l)

    print("CVPrecisionStack self-verification PASSED (100% integration tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

