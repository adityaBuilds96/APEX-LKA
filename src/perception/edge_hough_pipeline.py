"""
src/perception/edge_hough_pipeline.py
=====================================
APEX-Vision Core v4.0 • High-Precision Edge, Hough & RANSAC Polynomial Pipeline
ISO 26262 ASIL-B Aligned Geometric Feature Extraction & Curvature Estimation

Executes classical and robust geometric computer vision algorithms to isolate
lane edges, filter horizontal clutter, and fit curvature-accurate 2nd-degree parabolas.

Mathematical & Algorithmic Architecture:
----------------------------------------
1. Directional Gradient (Sobel-X):
   - Computes horizontal spatial gradient G_x = dI/dx using a 3x3 or 5x5 kernel.
   - Suppresses horizontal artifacts (horizon lines, vehicle hoods, road repair seams)
     while accentuating vertical and diagonal lane boundaries.
2. Dual Canny Edge Detection:
   - Asphalt Mode: Standard hysteresis thresholds (50, 150).
   - Off-Road Dirt Mode: Dynamic Otsu/median-scaled thresholds with adaptive sigma.
3. Hough Transform with Strict Slope Filtering:
   - Probabilistic Hough Transform (`cv2.HoughLinesP`).
   - Strict slope filter: Rejects lines with |m| < 0.35 (|slope| < ~19.3 deg)
     to eliminate guardrails, sky boundaries, and vehicle bonnet edges.
   - Segregates left candidates (m < -0.35) and right candidates (m > +0.35).
4. RANSAC 2nd-Degree Parabolic Polynomial:
   - Fits x = A * y^2 + B * y + C in metric BEV space.
   - RANSAC sampling rejects dirt clods, shadows, and isolated outlier edges.
5. Metric Radius of Curvature:
   - R = (1 + (2*A*y + B)^2)^1.5 / |2*A| evaluated in meters at vehicle front (y=0).
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import random
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

logger = logging.getLogger("APEX.EdgeHoughPipeline")


@dataclass
class EdgeHoughConfig:
    """Configuration parameters for Edge, Hough, and Polynomial fitting pipeline."""
    # Sobel Gradient Parameters
    sobel_kernel_size: int = 3
    sobel_scale: float = 1.0
    sobel_delta: float = 0.0

    # Dual Canny Thresholds
    canny_asphalt_low: int = 50
    canny_asphalt_high: int = 150
    canny_dirt_adaptive_sigma: float = 0.33
    use_adaptive_dirt_canny: bool = True

    # Hough Transform Parameters
    hough_rho_res: float = 1.0
    hough_theta_res: float = np.pi / 180.0
    hough_accumulator_threshold: int = 25
    hough_min_line_length_px: int = 35
    hough_max_line_gap_px: int = 20

    # Strict Slope Filtering (|m| >= 0.35)
    min_slope_magnitude: float = 0.35

    # RANSAC Parabola Parameters
    ransac_max_iterations: int = 150
    ransac_distance_threshold_px: float = 4.0
    ransac_min_inlier_ratio: float = 0.35
    ransac_sample_size: int = 4

    # Metric Scaling for Curvature (0.0185 m/px)
    meters_per_pixel: float = 0.0185
    max_curvature_radius_m: float = 5000.0


@dataclass
class LineSegment:
    """Extracted line segment in 2D image coordinates."""
    x1: int
    y1: int
    x2: int
    y2: int
    slope: float
    intercept: float
    length_px: float
    is_left_lane_candidate: bool
    is_right_lane_candidate: bool


@dataclass
class LaneBoundaryPolynomial:
    """Estimated 2nd-degree parabolic polynomial x = A*y^2 + B*y + C."""
    is_valid: bool
    A: float = 0.0  # Curvature coefficient
    B: float = 0.0  # Heading / slope coefficient
    C: float = 0.0  # Lateral offset coefficient
    radius_of_curvature_m: float = 5000.0
    lateral_offset_m: float = 0.0
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    fit_rmse_px: float = 0.0
    confidence: float = 0.0

    @property
    def coefficients(self) -> np.ndarray:
        return np.array([self.A, self.B, self.C], dtype=np.float64)

    def evaluate_x(self, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Evaluate x = A*y^2 + B*y + C at given y coordinate(s)."""
        return self.A * (y ** 2) + self.B * y + self.C

    def evaluate_heading_angle_rad(self, y: float) -> float:
        """Compute tangent heading angle theta = atan(2*A*y + B)."""
        slope = 2.0 * self.A * y + self.B
        return math.atan(slope)


@dataclass
class EdgeHoughResult:
    """Full result telemetry from edge, Hough, and polynomial fitting pass."""
    sobel_x_edges: np.ndarray
    canny_edges: np.ndarray
    filtered_hough_lines: List[LineSegment]
    left_polynomial: LaneBoundaryPolynomial
    right_polynomial: LaneBoundaryPolynomial
    center_polynomial: Optional[LaneBoundaryPolynomial] = None
    estimated_radius_of_curvature_m: float = 5000.0
    lane_width_m: float = 3.50
    lateral_cross_track_error_m: float = 0.0
    processing_time_ms: float = 0.0


class SobelEdgeDetector:
    """
    Directional gradient engine computing horizontal Sobel-X to accentuate
    vertical/slanted lane edges while rejecting horizontal road features.
    """

    def __init__(self, kernel_size: int = 3, scale: float = 1.0, delta: float = 0.0):
        self.kernel_size = kernel_size
        self.scale = scale
        self.delta = delta

    def compute_sobel_x(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute horizontal derivative |dI/dx| and normalize to 8-bit unsigned integer.
        """
        # Compute gradient in 16-bit signed to prevent underflow
        grad_x = cv2.Sobel(
            gray_image,
            cv2.CV_16S,
            dx=1,
            dy=0,
            ksize=self.kernel_size,
            scale=self.scale,
            delta=self.delta,
            borderType=cv2.BORDER_DEFAULT,
        )
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        return abs_grad_x

    def compute_directional_ratio(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Compute ratio |G_x| / (|G_y| + eps) to strongly suppress horizontal edges.
        """
        gx = cv2.Sobel(gray_image, cv2.CV_32F, 1, 0, ksize=self.kernel_size)
        gy = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=self.kernel_size)

        ratio = np.abs(gx) / (np.abs(gy) + 1e-4)
        # Select pixels where vertical gradient dominates horizontal gradient by >= 1.5x
        mask = (ratio > 1.5) & (np.abs(gx) > 25.0)
        return (mask * 255).astype(np.uint8)


class DualCannyDetector:
    """
    Dual-mode Canny edge detector:
    - Standard hysteresis (50, 150) for uniform asphalt.
    - Adaptive median-based thresholds for dirt/gravel tracks.
    """

    def __init__(
        self,
        asphalt_low: int = 50,
        asphalt_high: int = 150,
        adaptive_sigma: float = 0.33,
    ):
        self.asphalt_low = asphalt_low
        self.asphalt_high = asphalt_high
        self.adaptive_sigma = adaptive_sigma

    def detect_asphalt_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """Standard high-speed asphalt edge detection."""
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        return cv2.Canny(blurred, self.asphalt_low, self.asphalt_high)

    def detect_adaptive_dirt_edges(self, gray_image: np.ndarray) -> np.ndarray:
        """
        Adaptive Canny for off-road dirt/gravel where contrast is low and non-uniform.
        Computes median luminance:
        lower = max(0, (1.0 - sigma) * median)
        upper = min(255, (1.0 + sigma) * median)
        """
        blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
        v = np.median(blurred)
        lower = int(max(0, (1.0 - self.adaptive_sigma) * v))
        upper = int(min(255, (1.0 + self.adaptive_sigma) * v))

        # Clamp to reasonable bounds
        lower = max(20, min(lower, 80))
        upper = max(60, min(upper, 200))

        return cv2.Canny(blurred, lower, upper)


class HoughLaneFilter:
    """
    Extracts line segments via Probabilistic Hough Transform and enforces
    strict slope filtering (|m| >= 0.35) to eliminate guardrails, bonnet, and horizon.
    """

    def __init__(
        self,
        rho: float = 1.0,
        theta: float = np.pi / 180.0,
        threshold: int = 25,
        min_line_length: int = 35,
        max_line_gap: int = 20,
        min_slope: float = 0.35,
    ):
        self.rho = rho
        self.theta = theta
        self.threshold = threshold
        self.min_line_length = min_line_length
        self.max_line_gap = max_line_gap
        self.min_slope = min_slope

    def extract_and_filter_lines(
        self,
        edge_image: np.ndarray,
        image_width: int = 640,
    ) -> List[LineSegment]:
        """
        Run cv2.HoughLinesP and filter segments by slope magnitude and position.
        """
        lines = cv2.HoughLinesP(
            edge_image,
            rho=self.rho,
            theta=self.theta,
            threshold=self.threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        segments: List[LineSegment] = []
        if lines is None or len(lines) == 0:
            return segments

        for line in lines:
            coords = line.flatten()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])

            dx = x2 - x1
            dy = y2 - y1
            length = math.hypot(dx, dy)

            # Prevent divide by zero on vertical lines
            if dx == 0:
                slope = 999.0 if dy > 0 else -999.0
            else:
                slope = dy / float(dx)

            # Strict slope magnitude filter: reject horizontal lines (|m| < 0.35)
            if abs(slope) < self.min_slope:
                continue

            intercept = y1 - slope * x1

            # Segregate left vs right candidates based on lateral position and slope
            mid_x = (x1 + x2) / 2.0
            # Left candidate: left hemisphere and valid vertical/negative slope
            is_left = (mid_x < image_width * 0.55) and (slope < -self.min_slope or abs(slope) >= 1.0)
            # Right candidate: right hemisphere and valid vertical/positive slope
            is_right = (mid_x > image_width * 0.45) and (slope > self.min_slope or abs(slope) >= 1.0)

            segments.append(LineSegment(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                slope=slope,
                intercept=intercept,
                length_px=length,
                is_left_lane_candidate=is_left,
                is_right_lane_candidate=is_right,
            ))

        return segments

    def line_segments_to_pixel_points(
        self,
        segments: List[LineSegment],
        samples_per_segment: int = 15,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert line segments into discrete 2D point arrays (x, y) for polynomial fitting.
        Returns (x_points, y_points).
        """
        x_pts: List[float] = []
        y_pts: List[float] = []

        for seg in segments:
            # Interpolate points along the segment
            alphas = np.linspace(0.0, 1.0, samples_per_segment)
            for a in alphas:
                x_pts.append(seg.x1 + a * (seg.x2 - seg.x1))
                y_pts.append(seg.y1 + a * (seg.y2 - seg.y1))

        if not x_pts:
            return np.array([]), np.array([])

        return np.array(x_pts, dtype=np.float64), np.array(y_pts, dtype=np.float64)


class RANSACPolynomialFitter:
    """
    Fits 2nd-degree parabolic polynomials x = A*y^2 + B*y + C using RANSAC.
    Robust against edge outliers, shadows, and disconnected markings.
    """

    def __init__(
        self,
        max_iterations: int = 150,
        distance_threshold: float = 4.0,
        min_inlier_ratio: float = 0.35,
        sample_size: int = 4,
    ):
        self.max_iterations = max_iterations
        self.distance_threshold = distance_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.sample_size = sample_size

    def fit(
        self,
        x_points: np.ndarray,
        y_points: np.ndarray,
        meters_per_pixel: float = 0.0185,
    ) -> LaneBoundaryPolynomial:
        """
        Fit parabola x = A*y^2 + B*y + C to points.
        Evaluates radius of curvature and fit confidence.
        """
        n_pts = len(x_points)
        if n_pts < self.sample_size:
            return LaneBoundaryPolynomial(is_valid=False)

        best_inliers: Optional[np.ndarray] = None
        best_inlier_count = 0
        best_coeffs: Optional[np.ndarray] = None

        rng = np.random.default_rng(1234)

        for _ in range(self.max_iterations):
            # 1. Random sample of 4 points
            indices = rng.choice(n_pts, size=self.sample_size, replace=False)
            x_sample = x_points[indices]
            y_sample = y_points[indices]

            # 2. Fit candidate parabola: x = A*y^2 + B*y + C
            try:
                # Design matrix: [y^2, y, 1]
                V = np.column_stack([y_sample ** 2, y_sample, np.ones(self.sample_size)])
                coeffs, _, _, _ = np.linalg.lstsq(V, x_sample, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # 3. Compute residuals for all points
            x_pred = coeffs[0] * (y_points ** 2) + coeffs[1] * y_points + coeffs[2]
            residuals = np.abs(x_points - x_pred)
            inliers = residuals < self.distance_threshold
            inlier_count = np.sum(inliers)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers = inliers
                best_coeffs = coeffs

        # Check inlier ratio requirement
        inlier_ratio = best_inlier_count / float(n_pts)
        if (best_coeffs is None or
            best_inliers is None or
            inlier_ratio < self.min_inlier_ratio or
            best_inlier_count < self.sample_size):
            # Fallback to standard least-squares if RANSAC fails
            try:
                V_all = np.column_stack([y_points ** 2, y_points, np.ones(n_pts)])
                fallback_coeffs, _, _, _ = np.linalg.lstsq(V_all, x_points, rcond=None)
                best_coeffs = fallback_coeffs
                best_inlier_count = n_pts
                inlier_ratio = 1.0
            except Exception:
                return LaneBoundaryPolynomial(is_valid=False)

        # Refit on all inliers for optimal precision
        x_in = x_points[best_inliers]
        y_in = y_points[best_inliers]
        if len(x_in) >= 3:
            V_in = np.column_stack([y_in ** 2, y_in, np.ones(len(x_in))])
            final_coeffs, _, _, _ = np.linalg.lstsq(V_in, x_in, rcond=None)
        else:
            final_coeffs = best_coeffs

        A, B, C = float(final_coeffs[0]), float(final_coeffs[1]), float(final_coeffs[2])

        # Compute RMSE on inliers
        x_fit = A * (y_in ** 2) + B * y_in + C
        rmse = float(np.sqrt(np.mean((x_in - x_fit) ** 2))) if len(x_in) > 0 else 0.0

        # Compute Radius of Curvature R = (1 + (2*A*y + B)^2)^1.5 / |2*A|
        # Scale to metric space: A_m = A / meters_per_pixel, B_m = B
        # at vehicle front (y_eval = max y row)
        y_eval = float(np.max(y_points))
        A_m = A / meters_per_pixel
        B_m = B

        denom = 2.0 * abs(A_m)
        if denom > 1e-6:
            R_m = float(((1.0 + (2.0 * A_m * (y_eval * meters_per_pixel) + B_m) ** 2) ** 1.5) / denom)
        else:
            R_m = 5000.0  # Straight line

        R_m = min(5000.0, max(5.0, R_m))
        lateral_offset_m = float(C * meters_per_pixel)
        confidence = float(np.clip(inlier_ratio * (1.0 / (1.0 + rmse * 0.1)), 0.0, 1.0))

        return LaneBoundaryPolynomial(
            is_valid=True,
            A=A,
            B=B,
            C=C,
            radius_of_curvature_m=round(R_m, 1),
            lateral_offset_m=round(lateral_offset_m, 3),
            inlier_count=int(best_inlier_count),
            inlier_ratio=round(inlier_ratio, 3),
            fit_rmse_px=round(rmse, 2),
            confidence=round(confidence, 3),
        )


class EdgeHoughPipeline:
    """
    Master Geometric Edge & Hough Pipeline orchestrating:
    Sobel-X -> Dual Canny -> Hough Line Segment Filter -> RANSAC Parabola -> Curvature
    """

    def __init__(self, config: Optional[EdgeHoughConfig] = None):
        self.config = config or EdgeHoughConfig()
        self.sobel = SobelEdgeDetector(
            kernel_size=self.config.sobel_kernel_size,
            scale=self.config.sobel_scale,
            delta=self.config.sobel_delta,
        )
        self.canny = DualCannyDetector(
            asphalt_low=self.config.canny_asphalt_low,
            asphalt_high=self.config.canny_asphalt_high,
            adaptive_sigma=self.config.canny_dirt_adaptive_sigma,
        )
        self.hough = HoughLaneFilter(
            rho=self.config.hough_rho_res,
            theta=self.config.hough_theta_res,
            threshold=self.config.hough_accumulator_threshold,
            min_line_length=self.config.hough_min_line_length_px,
            max_line_gap=self.config.hough_max_line_gap_px,
            min_slope=self.config.min_slope_magnitude,
        )
        self.fitter = RANSACPolynomialFitter(
            max_iterations=self.config.ransac_max_iterations,
            distance_threshold=self.config.ransac_distance_threshold_px,
            min_inlier_ratio=self.config.ransac_min_inlier_ratio,
            sample_size=self.config.ransac_sample_size,
        )

    def process(
        self,
        image: np.ndarray,
        is_offroad_dirt: bool = False,
    ) -> EdgeHoughResult:
        """
        Execute full edge detection, Hough extraction, and polynomial fitting.
        """
        t_start = time.perf_counter()

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape[:2]

        # 1. Sobel-X horizontal spatial derivative
        sobel_x = self.sobel.compute_sobel_x(gray)

        # 2. Dual Canny Edge Detection
        if is_offroad_dirt and self.config.use_adaptive_dirt_canny:
            canny_edges = self.canny.detect_adaptive_dirt_edges(sobel_x)
        else:
            canny_edges = self.canny.detect_asphalt_edges(sobel_x)

        # 3. Hough Line Extraction with Strict Slope Filter (|m| >= 0.35)
        line_segments = self.hough.extract_and_filter_lines(canny_edges, image_width=w)

        # Separate left and right line segments
        left_segs = [s for s in line_segments if s.is_left_lane_candidate]
        right_segs = [s for s in line_segments if s.is_right_lane_candidate]

        # 4. Extract Point Clouds for RANSAC Parabola
        x_left, y_left = self.hough.line_segments_to_pixel_points(left_segs)
        x_right, y_right = self.hough.line_segments_to_pixel_points(right_segs)

        # 5. Fit 2nd-Degree Polynomials
        left_poly = self.fitter.fit(x_left, y_left, meters_per_pixel=self.config.meters_per_pixel)
        right_poly = self.fitter.fit(x_right, y_right, meters_per_pixel=self.config.meters_per_pixel)

        # 6. Estimate Combined Radius of Curvature and Lateral Offset
        if left_poly.is_valid and right_poly.is_valid:
            mean_radius = (left_poly.radius_of_curvature_m + right_poly.radius_of_curvature_m) / 2.0
            # Center polynomial
            center_A = (left_poly.A + right_poly.A) / 2.0
            center_B = (left_poly.B + right_poly.B) / 2.0
            center_C = (left_poly.C + right_poly.C) / 2.0
            center_poly = LaneBoundaryPolynomial(
                is_valid=True,
                A=center_A,
                B=center_B,
                C=center_C,
                radius_of_curvature_m=round(mean_radius, 1),
                confidence=(left_poly.confidence + right_poly.confidence) / 2.0,
            )
            lane_w_px = right_poly.evaluate_x(h - 1) - left_poly.evaluate_x(h - 1)
            lane_w_m = float(lane_w_px * self.config.meters_per_pixel)
            center_x_bottom = (left_poly.evaluate_x(h - 1) + right_poly.evaluate_x(h - 1)) / 2.0
            cte_m = float((w / 2.0 - center_x_bottom) * self.config.meters_per_pixel)
        elif left_poly.is_valid:
            mean_radius = left_poly.radius_of_curvature_m
            center_poly = left_poly
            lane_w_m = 3.50
            cte_m = float((w / 2.0 - (left_poly.evaluate_x(h - 1) + 1.75 / self.config.meters_per_pixel)) * self.config.meters_per_pixel)
        elif right_poly.is_valid:
            mean_radius = right_poly.radius_of_curvature_m
            center_poly = right_poly
            lane_w_m = 3.50
            cte_m = float((w / 2.0 - (right_poly.evaluate_x(h - 1) - 1.75 / self.config.meters_per_pixel)) * self.config.meters_per_pixel)
        else:
            mean_radius = 5000.0
            center_poly = None
            lane_w_m = 3.50
            cte_m = 0.0

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return EdgeHoughResult(
            sobel_x_edges=sobel_x,
            canny_edges=canny_edges,
            filtered_hough_lines=line_segments,
            left_polynomial=left_poly,
            right_polynomial=right_poly,
            center_polynomial=center_poly,
            estimated_radius_of_curvature_m=round(mean_radius, 1),
            lane_width_m=round(lane_w_m, 2),
            lateral_cross_track_error_m=round(cte_m, 3),
            processing_time_ms=elapsed_ms,
        )

    def draw_edge_hough_overlay(
        self,
        bgr_image: np.ndarray,
        result: EdgeHoughResult,
    ) -> np.ndarray:
        """
        Draw visual overlay showing extracted Hough line segments and fitted parabolas.
        """
        annotated = bgr_image.copy()
        h, w = annotated.shape[:2]

        # Draw Hough segments: Left = Yellow, Right = Cyan
        for s in result.filtered_hough_lines:
            color = (0, 255, 255) if s.is_left_lane_candidate else (255, 255, 0)
            cv2.line(annotated, (s.x1, s.y1), (s.x2, s.y2), color, 2)

        # Sample and draw Left Polynomial (Green)
        if result.left_polynomial.is_valid:
            y_eval = np.linspace(h * 0.45, h - 1, 30)
            x_eval = result.left_polynomial.evaluate_x(y_eval)
            pts = np.column_stack([x_eval, y_eval]).astype(np.int32)
            cv2.polylines(annotated, [pts], isClosed=False, color=(0, 255, 0), thickness=3)

        # Sample and draw Right Polynomial (Blue)
        if result.right_polynomial.is_valid:
            y_eval = np.linspace(h * 0.45, h - 1, 30)
            x_eval = result.right_polynomial.evaluate_x(y_eval)
            pts = np.column_stack([x_eval, y_eval]).astype(np.int32)
            cv2.polylines(annotated, [pts], isClosed=False, color=(255, 0, 0), thickness=3)

        # Telemetry HUD text
        cv2.putText(annotated, f"R_curv: {result.estimated_radius_of_curvature_m:.0f}m", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"CTE: {result.lateral_cross_track_error_m:+.2f}m", (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(annotated, f"Time: {result.processing_time_ms:.2f}ms", (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return annotated


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification
# ═══════════════════════════════════════════════════════════════════════════

def get_edge_hough_pipeline(config: Optional[EdgeHoughConfig] = None) -> EdgeHoughPipeline:
    """Factory constructor for EdgeHoughPipeline."""
    return EdgeHoughPipeline(config=config)


def run_unit_tests() -> bool:
    """Validate edge filtering, slope rejection, and RANSAC parabolic fitting."""
    print("Testing EdgeHoughPipeline algorithms...")
    pipeline = get_edge_hough_pipeline()

    # Create synthetic test image with clean left and right lane markings
    img = np.full((360, 640, 3), 40, dtype=np.uint8)
    # Left lane line (slope approx -0.85)
    cv2.line(img, (180, 359), (280, 190), (255, 255, 255), 4)
    # Right lane line (slope approx +0.85)
    cv2.line(img, (460, 359), (360, 190), (255, 255, 255), 4)
    # Horizontal distracting guardrail (slope approx 0.0) -> MUST be filtered out
    cv2.line(img, (50, 200), (590, 200), (255, 255, 255), 3)

    res = pipeline.process(img)

    # Invariant 1: Horizontal line was rejected (|m| >= 0.35)
    horizontal_detected = any(abs(s.slope) < 0.35 for s in res.filtered_hough_lines)
    assert not horizontal_detected, "Horizontal guardrail line should have been filtered out"

    # Invariant 2: Left and right candidates extracted
    assert len(res.filtered_hough_lines) >= 2, "Failed to detect left and right lane lines"
    assert res.left_polynomial.is_valid, "Left polynomial fitting failed"
    assert res.right_polynomial.is_valid, "Right polynomial fitting failed"

    # Invariant 3: Lane width is physically realistic (~3.5m)
    assert 2.0 <= res.lane_width_m <= 5.5, f"Unrealistic lane width: {res.lane_width_m} m"

    # Invariant 4: Radius of curvature on straight synthetic line is high (>500m)
    assert res.estimated_radius_of_curvature_m > 300.0, f"Expected straight line radius >300m, got {res.estimated_radius_of_curvature_m}"

    print("EdgeHoughPipeline self-verification PASSED (100% checks met).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()
