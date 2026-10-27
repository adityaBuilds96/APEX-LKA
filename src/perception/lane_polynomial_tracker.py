"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

src/perception/lane_polynomial_tracker.py
RANSAC Bird's-Eye-View (BEV) Polynomial Tracker & Geometric Metric Estimator

Extracts BEV lane boundary coordinates, executes robust RANSAC 2nd-order
polynomial fitting x(y) = A*y^2 + B*y + C, applies Mahalanobis distance outlier
rejection, estimates analytic coefficient variance-covariance matrices, and
derives essential ADAS metrics: radius of curvature R(y), lateral error e_lat,
heading error theta_e, and dual-boundary parallelism indices.
"""

from __future__ import annotations

import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("apex_vision.perception.lane_polynomial_tracker")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enumerations & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class BoundaryDesignation(str, Enum):
    """Identifier for evaluated road boundary."""
    LEFT_BOUNDARY = "LEFT_BOUNDARY"
    RIGHT_BOUNDARY = "RIGHT_BOUNDARY"
    CENTERLINE = "CENTERLINE"
    ADJACENT_LEFT = "ADJACENT_LEFT"
    ADJACENT_RIGHT = "ADJACENT_RIGHT"


class PolynomialFitStatus(str, Enum):
    """Result status of the RANSAC polynomial fit."""
    OPTIMAL_FIT = "OPTIMAL_FIT"
    DEGRADED_LOW_INLIERS = "DEGRADED_LOW_INLIERS"
    PARALLEL_HALLUCINATED = "PARALLEL_HALLUCINATED"
    FIT_FAILED = "FIT_FAILED"


@dataclass(frozen=True)
class PolynomialCoefficients:
    """
    Second-order polynomial coefficients for x(y) = A*y^2 + B*y + C in vehicle coordinates.
    All longitudinal distances y > 0 forward, lateral x positive right, negative left.
    """
    A: float  # Curvature coefficient (1/m)
    B: float  # Heading coefficient (tangent slope)
    C: float  # Lateral offset at bumper y=0 (m)
    variance_A: float = 0.0
    variance_B: float = 0.0
    variance_C: float = 0.0
    covariance_AB: float = 0.0
    covariance_AC: float = 0.0
    covariance_BC: float = 0.0

    def evaluate(self, y_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute lateral position x(y) = A*y^2 + B*y + C."""
        return self.A * (y_m ** 2) + self.B * y_m + self.C

    def evaluate_derivative(self, y_m: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Compute first derivative dx/dy = 2*A*y + B."""
        return 2.0 * self.A * y_m + self.B

    def evaluate_curvature(self, y_m: float = 15.0) -> float:
        """
        Compute road curvature kappa(y) = 2*A / (1 + (2*A*y + B)^2)^1.5.
        """
        dy = 2.0 * self.A * y_m + self.B
        denom = (1.0 + dy ** 2) ** 1.5
        if abs(denom) < 1e-6:
            return 0.0
        return float(2.0 * self.A / denom)

    def evaluate_radius_m(self, y_m: float = 15.0) -> float:
        """Compute radius of curvature R = 1 / |kappa|."""
        kappa = abs(self.evaluate_curvature(y_m))
        if kappa < 1e-5:
            return 10000.0
        return float(1.0 / kappa)

    def evaluate_uncertainty(self, y_m: float) -> float:
        """
        Calculate analytical 1-sigma coordinate variance:
        Var(x) = J * Cov * J^T where J = [y^2, y, 1].
        """
        var_x = (
            (y_m ** 4) * self.variance_A
            + (y_m ** 2) * self.variance_B
            + self.variance_C
            + 2.0 * (y_m ** 3) * self.covariance_AB
            + 2.0 * (y_m ** 2) * self.covariance_AC
            + 2.0 * y_m * self.covariance_BC
        )
        return float(math.sqrt(max(1e-6, var_x)))

    def to_tuple(self) -> Tuple[float, float, float]:
        return (float(self.A), float(self.B), float(self.C))

    def to_dict(self) -> Dict[str, float]:
        return {
            "A": round(self.A, 7),
            "B": round(self.B, 5),
            "C": round(self.C, 4),
            "std_A": round(math.sqrt(max(0.0, self.variance_A)), 7),
            "std_B": round(math.sqrt(max(0.0, self.variance_B)), 5),
            "std_C": round(math.sqrt(max(0.0, self.variance_C)), 4),
        }


@dataclass
class BoundaryTrackResult:
    """Track result for an individual lane boundary."""
    designation: BoundaryDesignation
    status: PolynomialFitStatus
    coefficients: PolynomialCoefficients
    inlier_pixel_count: int
    total_pixel_count: int
    inlier_ratio: float
    root_mean_square_error_m: float
    max_residual_error_m: float
    sample_points_bev: List[Tuple[float, float]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "designation": self.designation.value,
            "status": self.status.value,
            "coefficients": self.coefficients.to_dict(),
            "inlier_pixel_count": self.inlier_pixel_count,
            "total_pixel_count": self.total_pixel_count,
            "inlier_ratio": round(self.inlier_ratio, 3),
            "root_mean_square_error_m": round(self.root_mean_square_error_m, 4),
            "max_residual_error_m": round(self.max_residual_error_m, 4),
        }


@dataclass
class DualBoundaryLaneGeometry:
    """
    Combined geometric state of ego lane derived from left and right tracks.
    """
    timestamp_utc: str
    left_track: BoundaryTrackResult
    right_track: BoundaryTrackResult
    centerline_poly: PolynomialCoefficients
    lateral_error_m: float               # e_lat at lookahead distance (m)
    heading_error_rad: float             # theta_e at lookahead distance (rad)
    heading_error_deg: float             # theta_e in degrees
    radius_of_curvature_m: float         # R at lookahead distance (m)
    lane_width_at_bumper_m: float        # w(0) = C_R - C_L
    lane_width_at_lookahead_m: float     # w(y_eval)
    parallelism_residual_variance: float # Metric of geometric parallelism
    is_geometry_valid: bool
    evaluation_lookahead_m: float
    processing_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "left_track": self.left_track.to_dict(),
            "right_track": self.right_track.to_dict(),
            "centerline_poly": self.centerline_poly.to_dict(),
            "lateral_error_m": round(self.lateral_error_m, 4),
            "heading_error_deg": round(self.heading_error_deg, 3),
            "radius_of_curvature_m": round(self.radius_of_curvature_m, 2),
            "lane_width_at_bumper_m": round(self.lane_width_at_bumper_m, 4),
            "lane_width_at_lookahead_m": round(self.lane_width_at_lookahead_m, 4),
            "parallelism_residual_variance": round(self.parallelism_residual_variance, 5),
            "is_geometry_valid": self.is_geometry_valid,
            "evaluation_lookahead_m": round(self.evaluation_lookahead_m, 1),
            "processing_latency_ms": round(self.processing_latency_ms, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration Parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class PolynomialTrackerConfig:
    """Parameters governing BEV RANSAC fitting and metric evaluation."""
    # RANSAC fitting hyper-parameters
    ransac_max_iterations: int = 150
    ransac_inlier_distance_threshold_m: float = 0.18
    min_inliers_required: int = 25
    min_inlier_ratio_target: float = 0.40

    # Mahalanobis outlier filtering
    mahalanobis_residual_threshold: float = 2.58  # 99% Gaussian confidence
    enable_mahalanobis_refinement: bool = True

    # Physical road parameters
    nominal_lane_width_m: float = 3.70
    min_valid_lane_width_m: float = 2.60
    max_valid_lane_width_m: float = 4.80
    max_acceptable_heading_deg: float = 35.0
    evaluation_lookahead_m: float = 12.0

    # Longitudinal range in BEV coordinates (meters)
    min_longitudinal_m: float = 2.0
    max_longitudinal_m: float = 35.0

    # Sliding window search parameters for pixel extraction
    num_sliding_windows: int = 9
    sliding_window_margin_px: int = 40
    min_pixels_recenter: int = 30


# ═══════════════════════════════════════════════════════════════════════════════
# 3. BEV Polynomial Tracker Implementation
# ═══════════════════════════════════════════════════════════════════════════════

class LanePolynomialTracker:
    """
    RANSAC-based 2nd-order polynomial tracker operating in metric BEV space.

    Standards Compliance:
        - APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
        - Driver-supervised SAE Level 2 execution
        - ISO 26262-style process; ASIL-B aligned; not independently assessed
    """

    def __init__(self, config: Optional[PolynomialTrackerConfig] = None) -> None:
        self.config = config or PolynomialTrackerConfig()
        self._lock = threading.Lock()

    def track_lane_geometry(
        self,
        left_bev_points: Optional[np.ndarray],
        right_bev_points: Optional[np.ndarray],
        eval_lookahead_m: Optional[float] = None,
    ) -> DualBoundaryLaneGeometry:
        """
        Fit left and right boundaries, apply parallel hallucination if one side is missing,
        and calculate complete ADAS lane tracking geometry.

        Args:
            left_bev_points: Array of shape (N, 2) in meters [X (lat), Y (long)].
            right_bev_points: Array of shape (M, 2) in meters [X (lat), Y (long)].
            eval_lookahead_m: Forward lookahead distance in meters for error evaluation.
        """
        t_start = time.perf_counter()
        lookahead = float(eval_lookahead_m or self.config.evaluation_lookahead_m)

        with self._lock:
            # 1. Fit Left Boundary
            left_track = self._fit_boundary(
                left_bev_points, BoundaryDesignation.LEFT_BOUNDARY
            )

            # 2. Fit Right Boundary
            right_track = self._fit_boundary(
                right_bev_points, BoundaryDesignation.RIGHT_BOUNDARY
            )

            # 3. Parallel Projection Hallucination if one line failed
            left_track, right_track = self._resolve_parallel_projections(
                left_track, right_track
            )

            # 4. Construct Centerline Polynomial: x_center(y) = 0.5 * (x_L(y) + x_R(y))
            c_poly = self._construct_centerline(left_track.coefficients, right_track.coefficients)

            # 5. Compute Lateral Error e_lat = X_ego - x_center(y_eval)
            # Vehicle ego coordinate at lookahead is x=0
            x_center_at_eval = float(c_poly.evaluate(lookahead))
            e_lat = -x_center_at_eval

            # 6. Compute Heading Error theta_e = arctan(dx/dy|_{center})
            dx_dy = float(c_poly.evaluate_derivative(lookahead))
            theta_e = float(math.atan(dx_dy))

            # 7. Radius of Curvature R(y_eval)
            radius_m = float(c_poly.evaluate_radius_m(lookahead))

            # 8. Lane Width Metrics
            w_bumper = float(right_track.coefficients.C - left_track.coefficients.C)
            x_l_eval = float(left_track.coefficients.evaluate(lookahead))
            x_r_eval = float(right_track.coefficients.evaluate(lookahead))
            w_lookahead = float(x_r_eval - x_l_eval)

            # 9. Parallelism Check
            parallelism_var = self._compute_parallelism_variance(
                left_track.coefficients, right_track.coefficients
            )

            is_valid = bool(
                left_track.status != PolynomialFitStatus.FIT_FAILED
                and right_track.status != PolynomialFitStatus.FIT_FAILED
                and self.config.min_valid_lane_width_m <= w_bumper <= self.config.max_valid_lane_width_m
                and abs(math.degrees(theta_e)) <= self.config.max_acceptable_heading_deg
            )

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            return DualBoundaryLaneGeometry(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                left_track=left_track,
                right_track=right_track,
                centerline_poly=c_poly,
                lateral_error_m=float(e_lat),
                heading_error_rad=float(theta_e),
                heading_error_deg=float(math.degrees(theta_e)),
                radius_of_curvature_m=float(radius_m),
                lane_width_at_bumper_m=float(w_bumper),
                lane_width_at_lookahead_m=float(w_lookahead),
                parallelism_residual_variance=float(parallelism_var),
                is_geometry_valid=is_valid,
                evaluation_lookahead_m=lookahead,
                processing_latency_ms=t_elapsed_ms,
            )

    # ───────────────────────────────────────────────────────────────────────────
    # RANSAC Polynomial Fitting Implementation
    # ───────────────────────────────────────────────────────────────────────────

    def _fit_boundary(
        self,
        points_bev: Optional[np.ndarray],
        designation: BoundaryDesignation,
    ) -> BoundaryTrackResult:
        """
        Fit a 2nd-order polynomial x(y) = A*y^2 + B*y + C to BEV coordinates via RANSAC.
        """
        nominal_c = -self.config.nominal_lane_width_m / 2.0 if designation == BoundaryDesignation.LEFT_BOUNDARY else self.config.nominal_lane_width_m / 2.0
        fallback_coeffs = PolynomialCoefficients(A=0.0, B=0.0, C=nominal_c)

        if points_bev is None or len(points_bev) < self.config.min_inliers_required:
            return BoundaryTrackResult(
                designation=designation,
                status=PolynomialFitStatus.FIT_FAILED,
                coefficients=fallback_coeffs,
                inlier_pixel_count=0,
                total_pixel_count=len(points_bev) if points_bev is not None else 0,
                inlier_ratio=0.0,
                root_mean_square_error_m=99.0,
                max_residual_error_m=99.0,
                sample_points_bev=[],
            )

        # Filter points within longitudinal range
        x_pts = points_bev[:, 0].astype(np.float64)
        y_pts = points_bev[:, 1].astype(np.float64)
        valid_mask = (y_pts >= self.config.min_longitudinal_m) & (y_pts <= self.config.max_longitudinal_m)

        x_pts = x_pts[valid_mask]
        y_pts = y_pts[valid_mask]
        n_pts = len(x_pts)

        if n_pts < self.config.min_inliers_required:
            return BoundaryTrackResult(
                designation=designation,
                status=PolynomialFitStatus.FIT_FAILED,
                coefficients=fallback_coeffs,
                inlier_pixel_count=0,
                total_pixel_count=n_pts,
                inlier_ratio=0.0,
                root_mean_square_error_m=99.0,
                max_residual_error_m=99.0,
                sample_points_bev=[],
            )

        # RANSAC Loop
        best_inliers_mask: Optional[np.ndarray] = None
        best_inlier_count = 0
        thresh = self.config.ransac_inlier_distance_threshold_m

        for _ in range(self.config.ransac_max_iterations):
            # Sample 3 random points
            sample_idx = np.random.choice(n_pts, size=3, replace=False)
            y_sample = y_pts[sample_idx]
            x_sample = x_pts[sample_idx]

            # Fit candidate parabola x = A*y^2 + B*y + C
            # Vandermonde matrix for y: [y^2, y, 1]
            V_sample = np.column_stack((y_sample ** 2, y_sample, np.ones(3)))
            try:
                candidate_coeffs = np.linalg.solve(V_sample, x_sample)
            except np.linalg.LinAlgError:
                continue

            # Evaluate distance residuals on all points
            x_pred = (
                candidate_coeffs[0] * (y_pts ** 2)
                + candidate_coeffs[1] * y_pts
                + candidate_coeffs[2]
            )
            residuals = np.abs(x_pts - x_pred)
            inliers = residuals < thresh
            inlier_count = int(np.sum(inliers))

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_inliers_mask = inliers

        if best_inliers_mask is None or best_inlier_count < self.config.min_inliers_required:
            return BoundaryTrackResult(
                designation=designation,
                status=PolynomialFitStatus.DEGRADED_LOW_INLIERS,
                coefficients=fallback_coeffs,
                inlier_pixel_count=best_inlier_count,
                total_pixel_count=n_pts,
                inlier_ratio=float(best_inlier_count) / float(n_pts),
                root_mean_square_error_m=99.0,
                max_residual_error_m=99.0,
                sample_points_bev=[],
            )

        # Least squares re-estimation on all inliers
        x_in = x_pts[best_inliers_mask]
        y_in = y_pts[best_inliers_mask]

        coeffs, cov_matrix, rmse, max_err = self._fit_weighted_least_squares(x_in, y_in)

        # Mahalanobis refinement: drop points whose residual exceeds 2.58 sigma
        if self.config.enable_mahalanobis_refinement and rmse > 1e-4:
            x_pred_refined = coeffs.evaluate(y_in)
            residuals_ref = np.abs(x_in - x_pred_refined)
            mahalanobis_mask = (residuals_ref / rmse) < self.config.mahalanobis_residual_threshold
            if np.sum(mahalanobis_mask) >= self.config.min_inliers_required:
                x_in = x_in[mahalanobis_mask]
                y_in = y_in[mahalanobis_mask]
                coeffs, cov_matrix, rmse, max_err = self._fit_weighted_least_squares(x_in, y_in)

        inlier_ratio = float(len(x_in)) / float(n_pts)
        status = (
            PolynomialFitStatus.OPTIMAL_FIT
            if inlier_ratio >= self.config.min_inlier_ratio_target
            else PolynomialFitStatus.DEGRADED_LOW_INLIERS
        )

        sample_pts = [(float(x_in[i]), float(y_in[i])) for i in range(0, len(x_in), max(1, len(x_in) // 20))]

        return BoundaryTrackResult(
            designation=designation,
            status=status,
            coefficients=coeffs,
            inlier_pixel_count=len(x_in),
            total_pixel_count=n_pts,
            inlier_ratio=inlier_ratio,
            root_mean_square_error_m=rmse,
            max_residual_error_m=max_err,
            sample_points_bev=sample_pts,
        )

    def _fit_weighted_least_squares(
        self, x: np.ndarray, y: np.ndarray
    ) -> Tuple[PolynomialCoefficients, np.ndarray, float, float]:
        """
        Fit x(y) = A*y^2 + B*y + C using ordinary least squares and estimate
        analytic covariance matrix: Cov = (X^T X)^-1 * sigma_residual^2.
        """
        V = np.column_stack((y ** 2, y, np.ones_like(y)))  # Design matrix (N, 3)

        # Solve normal equations: (V^T V) * theta = V^T * x
        VtV = V.T @ V
        try:
            VtV_inv = np.linalg.inv(VtV)
            theta = VtV_inv @ (V.T @ x)
        except np.linalg.LinAlgError:
            theta = np.zeros(3)
            VtV_inv = np.eye(3) * 1e-3

        a, b, c = float(theta[0]), float(theta[1]), float(theta[2])

        # Residuals
        x_fit = a * (y ** 2) + b * y + c
        residuals = x - x_fit
        dof = max(1, len(x) - 3)
        residual_variance = float(np.sum(residuals ** 2) / dof)
        rmse = float(math.sqrt(residual_variance))
        max_residual = float(np.max(np.abs(residuals)))

        # Covariance matrix: Cov(theta) = residual_variance * (V^T V)^-1
        cov = VtV_inv * residual_variance

        coeffs = PolynomialCoefficients(
            A=a,
            B=b,
            C=c,
            variance_A=float(cov[0, 0]),
            variance_B=float(cov[1, 1]),
            variance_C=float(cov[2, 2]),
            covariance_AB=float(cov[0, 1]),
            covariance_AC=float(cov[0, 2]),
            covariance_BC=float(cov[1, 2]),
        )
        return coeffs, cov, rmse, max_residual

    # ───────────────────────────────────────────────────────────────────────────
    # Geometric Synthesis & Projection Helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _resolve_parallel_projections(
        self, left: BoundaryTrackResult, right: BoundaryTrackResult
    ) -> Tuple[BoundaryTrackResult, BoundaryTrackResult]:
        """
        Dual-Boundary Parallel Projection: If ONE lane boundary is detected
        with high confidence but the other is missing or weak, mirror the visible
        polynomial using nominal lane width W = 3.7m.
        """
        left_ok = left.status in (PolynomialFitStatus.OPTIMAL_FIT, PolynomialFitStatus.DEGRADED_LOW_INLIERS)
        right_ok = right.status in (PolynomialFitStatus.OPTIMAL_FIT, PolynomialFitStatus.DEGRADED_LOW_INLIERS)

        nominal_w = self.config.nominal_lane_width_m

        if left_ok and not right_ok:
            # Mirror Right from Left: A_R = A_L, B_R = B_L, C_R = C_L + W
            c_l = left.coefficients
            c_r = PolynomialCoefficients(
                A=c_l.A,
                B=c_l.B,
                C=c_l.C + nominal_w,
                variance_A=c_l.variance_A * 2.0,
                variance_B=c_l.variance_B * 2.0,
                variance_C=c_l.variance_C * 2.0 + 0.05,
                covariance_AB=c_l.covariance_AB,
                covariance_AC=c_l.covariance_AC,
                covariance_BC=c_l.covariance_BC,
            )
            right_proj = BoundaryTrackResult(
                designation=BoundaryDesignation.RIGHT_BOUNDARY,
                status=PolynomialFitStatus.PARALLEL_HALLUCINATED,
                coefficients=c_r,
                inlier_pixel_count=left.inlier_pixel_count,
                total_pixel_count=left.total_pixel_count,
                inlier_ratio=left.inlier_ratio * 0.80,
                root_mean_square_error_m=left.root_mean_square_error_m,
                max_residual_error_m=left.max_residual_error_m,
                sample_points_bev=[],
            )
            return left, right_proj

        elif right_ok and not left_ok:
            # Mirror Left from Right: A_L = A_R, B_L = B_R, C_L = C_R - W
            c_r = right.coefficients
            c_l = PolynomialCoefficients(
                A=c_r.A,
                B=c_r.B,
                C=c_r.C - nominal_w,
                variance_A=c_r.variance_A * 2.0,
                variance_B=c_r.variance_B * 2.0,
                variance_C=c_r.variance_C * 2.0 + 0.05,
                covariance_AB=c_r.covariance_AB,
                covariance_AC=c_r.covariance_AC,
                covariance_BC=c_r.covariance_BC,
            )
            left_proj = BoundaryTrackResult(
                designation=BoundaryDesignation.LEFT_BOUNDARY,
                status=PolynomialFitStatus.PARALLEL_HALLUCINATED,
                coefficients=c_l,
                inlier_pixel_count=right.inlier_pixel_count,
                total_pixel_count=right.total_pixel_count,
                inlier_ratio=right.inlier_ratio * 0.80,
                root_mean_square_error_m=right.root_mean_square_error_m,
                max_residual_error_m=right.max_residual_error_m,
                sample_points_bev=[],
            )
            return left_proj, right

        return left, right

    def _construct_centerline(
        self, left_coeffs: PolynomialCoefficients, right_coeffs: PolynomialCoefficients
    ) -> PolynomialCoefficients:
        """Construct centerline polynomial: x_c(y) = 0.5 * (x_L(y) + x_R(y))."""
        a_c = 0.5 * (left_coeffs.A + right_coeffs.A)
        b_c = 0.5 * (left_coeffs.B + right_coeffs.B)
        c_c = 0.5 * (left_coeffs.C + right_coeffs.C)

        var_a = 0.25 * (left_coeffs.variance_A + right_coeffs.variance_A)
        var_b = 0.25 * (left_coeffs.variance_B + right_coeffs.variance_B)
        var_c = 0.25 * (left_coeffs.variance_C + right_coeffs.variance_C)

        return PolynomialCoefficients(
            A=a_c,
            B=b_c,
            C=c_c,
            variance_A=var_a,
            variance_B=var_b,
            variance_C=var_c,
        )

    def _compute_parallelism_variance(
        self, left: PolynomialCoefficients, right: PolynomialCoefficients
    ) -> float:
        """
        Evaluate parallelism consistency by sampling lane width w(y) = x_R(y) - x_L(y)
        at 10 equidistant points and returning its sample variance.
        Ideal parallel lines have Var(w) = 0.
        """
        y_samples = np.linspace(self.config.min_longitudinal_m, self.config.max_longitudinal_m, 10)
        widths = (right.evaluate(y_samples) - left.evaluate(y_samples))
        return float(np.var(widths))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Sliding Window BEV Pixel Extractor & Overlay Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

class SlidingWindowBEVPixelExtractor:
    """
    Extracts continuous lane boundary coordinates from 2D Bird's-Eye-View (BEV)
    binary edge images or multi-class semantic segmentation masks using
    histogram peak detection and adaptive sliding windows.
    """

    def __init__(
        self,
        num_windows: int = 9,
        window_margin_px: int = 35,
        min_pixels_recenter: int = 25,
        bev_meter_per_pixel_x: float = 0.025,
        bev_meter_per_pixel_y: float = 0.050,
        origin_x_px: int = 160,
        origin_y_px: int = 360,
    ) -> None:
        self.num_windows = int(num_windows)
        self.margin = int(window_margin_px)
        self.min_recenter = int(min_pixels_recenter)
        self.m_per_px_x = float(bev_meter_per_pixel_x)
        self.m_per_px_y = float(bev_meter_per_pixel_y)
        self.origin_x = int(origin_x_px)
        self.origin_y = int(origin_y_px)

    def extract_boundary_coordinates(
        self, bev_binary_image: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract metric BEV coordinates for left and right lane boundaries.
        Returns:
            Tuple of (left_pts_m, right_pts_m) where each is (N, 2) [X_m, Y_m].
        """
        h, w = bev_binary_image.shape[:2]
        # Histogram on bottom half of image
        histogram = np.sum(bev_binary_image[h // 2 :, :], axis=0)
        if np.max(histogram) < 15:
            return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

        midpoint = int(w // 2)
        left_peak = int(np.argmax(histogram[:midpoint]))
        right_peak = int(np.argmax(histogram[midpoint:]) + midpoint)

        nonzero = bev_binary_image.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        left_lane_inds = []
        right_lane_inds = []

        window_height = int(h // self.num_windows)
        current_left = left_peak
        current_right = right_peak

        for win in range(self.num_windows):
            win_y_low = h - (win + 1) * window_height
            win_y_high = h - win * window_height

            win_x_left_low = current_left - self.margin
            win_x_left_high = current_left + self.margin
            win_x_right_low = current_right - self.margin
            win_x_right_high = current_right + self.margin

            good_left_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_x_left_low)
                & (nonzerox < win_x_left_high)
            ).nonzero()[0]

            good_right_inds = (
                (nonzeroy >= win_y_low)
                & (nonzeroy < win_y_high)
                & (nonzerox >= win_x_right_low)
                & (nonzerox < win_x_right_high)
            ).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            if len(good_left_inds) > self.min_recenter:
                current_left = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.min_recenter:
                current_right = int(np.mean(nonzerox[good_right_inds]))

        left_lane_inds = np.concatenate(left_lane_inds) if left_lane_inds else np.array([], dtype=int)
        right_lane_inds = np.concatenate(right_lane_inds) if right_lane_inds else np.array([], dtype=int)

        # Convert pixel indices to metric BEV coordinates
        left_pts_m = self._pixels_to_metric(nonzerox[left_lane_inds], nonzeroy[left_lane_inds])
        right_pts_m = self._pixels_to_metric(nonzerox[right_lane_inds], nonzeroy[right_lane_inds])

        return left_pts_m, right_pts_m

    def _pixels_to_metric(self, px_x: np.ndarray, px_y: np.ndarray) -> np.ndarray:
        if len(px_x) == 0:
            return np.empty((0, 2), dtype=np.float64)
        # Lateral: centered at origin_x
        x_m = (px_x.astype(np.float64) - self.origin_x) * self.m_per_px_x
        # Longitudinal: 0 at origin_y (bottom), positive forward
        y_m = (self.origin_y - px_y.astype(np.float64)) * self.m_per_px_y
        return np.column_stack((x_m, y_m))


class LaneCurvatureVisualizerOverlay:
    """
    Renders metric polynomials, curvature osculating circles, and safety
    envelopes onto a 2D BEV canvas for debugging and HUD displays.
    """

    def __init__(
        self,
        canvas_width: int = 400,
        canvas_height: int = 600,
        meter_per_pixel: float = 0.08,
    ) -> None:
        self.w = int(canvas_width)
        self.h = int(canvas_height)
        self.m_per_px = float(meter_per_pixel)
        self.origin_x = self.w // 2
        self.origin_y = self.h - 40

    def render_bev_canvas(self, geometry: DualBoundaryLaneGeometry) -> np.ndarray:
        """Render diagnostic BEV overlay with polynomials and curvature indicators."""
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Draw distance grid lines every 5 meters
        for d_m in range(5, 40, 5):
            y_px = int(self.origin_y - d_m / self.m_per_px)
            if 0 <= y_px < self.h:
                cv2.line(canvas, (0, y_px), (self.w, y_px), (30, 30, 30), 1)
                cv2.putText(canvas, f"{d_m}m", (10, y_px - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1)

        # Draw vehicle footprint at bumper (origin)
        veh_w_px = int(1.9 / self.m_per_px)
        veh_l_px = int(4.5 / self.m_per_px)
        cv2.rectangle(
            canvas,
            (self.origin_x - veh_w_px // 2, self.origin_y),
            (self.origin_x + veh_w_px // 2, self.origin_y + veh_l_px),
            (0, 140, 255),
            -1,
        )

        y_eval = np.linspace(2.0, 35.0, 50)
        # Render Left Polynomial (Cyan)
        left_x = geometry.left_track.coefficients.evaluate(y_eval)
        pts_left = self._metric_to_canvas(left_x, y_eval)
        cv2.polylines(canvas, [pts_left], isClosed=False, color=(255, 255, 0), thickness=2)

        # Render Right Polynomial (Amber)
        right_x = geometry.right_track.coefficients.evaluate(y_eval)
        pts_right = self._metric_to_canvas(right_x, y_eval)
        cv2.polylines(canvas, [pts_right], isClosed=False, color=(0, 200, 255), thickness=2)

        # Render Centerline (Green dashed)
        center_x = geometry.centerline_poly.evaluate(y_eval)
        pts_center = self._metric_to_canvas(center_x, y_eval)
        cv2.polylines(canvas, [pts_center], isClosed=False, color=(0, 255, 0), thickness=1)

        # Telemetry Text
        cv2.putText(
            canvas,
            f"R: {geometry.radius_of_curvature_m:.0f}m | e_lat: {geometry.lateral_error_m:+.2f}m",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            canvas,
            f"theta_e: {geometry.heading_error_deg:+.1f} deg | w: {geometry.lane_width_at_bumper_m:.2f}m",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
        return canvas

    def _metric_to_canvas(self, x_m: np.ndarray, y_m: np.ndarray) -> np.ndarray:
        px_x = (self.origin_x + x_m / self.m_per_px).astype(np.int32)
        px_y = (self.origin_y - y_m / self.m_per_px).astype(np.int32)
        return np.column_stack((px_x, px_y))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Global Singleton & Benchmark Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_POLYNOMIAL_TRACKER: Optional[LanePolynomialTracker] = None
_GLOBAL_PIXEL_EXTRACTOR: Optional[SlidingWindowBEVPixelExtractor] = None
_GLOBAL_CANVAS_VISUALIZER: Optional[LaneCurvatureVisualizerOverlay] = None


def get_lane_polynomial_tracker(
    config: Optional[PolynomialTrackerConfig] = None,
) -> LanePolynomialTracker:
    """Retrieve or instantiate global LanePolynomialTracker singleton."""
    global _GLOBAL_POLYNOMIAL_TRACKER
    if _GLOBAL_POLYNOMIAL_TRACKER is None:
        _GLOBAL_POLYNOMIAL_TRACKER = LanePolynomialTracker(config=config)
    return _GLOBAL_POLYNOMIAL_TRACKER


def get_sliding_window_extractor() -> SlidingWindowBEVPixelExtractor:
    """Retrieve global SlidingWindowBEVPixelExtractor singleton."""
    global _GLOBAL_PIXEL_EXTRACTOR
    if _GLOBAL_PIXEL_EXTRACTOR is None:
        _GLOBAL_PIXEL_EXTRACTOR = SlidingWindowBEVPixelExtractor()
    return _GLOBAL_PIXEL_EXTRACTOR


def get_curvature_canvas_visualizer() -> LaneCurvatureVisualizerOverlay:
    """Retrieve global LaneCurvatureVisualizerOverlay singleton."""
    global _GLOBAL_CANVAS_VISUALIZER
    if _GLOBAL_CANVAS_VISUALIZER is None:
        _GLOBAL_CANVAS_VISUALIZER = LaneCurvatureVisualizerOverlay()
    return _GLOBAL_CANVAS_VISUALIZER


@dataclass
class PolynomialBenchmarkSummary:
    """Statistical summary of 300-frame polynomial tracking stress benchmark."""
    total_frames: int
    mean_fitting_latency_ms: float
    max_fitting_latency_ms: float
    mean_lane_width_error_m: float
    mean_lateral_error_m: float
    fit_success_rate_pct: float
    passed_precision_target: bool


def run_polynomial_tracking_stress_benchmark(num_frames: int = 300) -> PolynomialBenchmarkSummary:
    """
    Stress-test BEV polynomial tracker across sharp curves (R=150m..1500m),
    synthetic noise, and parallel hallucination recovery.
    """
    tracker = LanePolynomialTracker()
    latencies: List[float] = []
    width_errors: List[float] = []
    lat_errors: List[float] = []
    success_count = 0

    np.random.seed(42)

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Dynamic curvature sweep
        a_true = 0.0015 * math.sin(i / 25.0)  # Min radius ~ 333m
        b_true = 0.02 * math.cos(i / 15.0)
        c_l_true = -1.85
        c_r_true = 1.85

        y_pts = np.linspace(3.0, 32.0, 120)
        x_l = a_true * (y_pts ** 2) + b_true * y_pts + c_l_true + np.random.normal(0, 0.04, len(y_pts))
        x_r = a_true * (y_pts ** 2) + b_true * y_pts + c_r_true + np.random.normal(0, 0.04, len(y_pts))

        # Random single-line dropout on 10% of frames
        if i % 10 == 0:
            right_input = None
        else:
            right_input = np.column_stack((x_r, y_pts))

        left_input = np.column_stack((x_l, y_pts))

        geom = tracker.track_lane_geometry(left_input, right_input, eval_lookahead_m=12.0)
        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)

        w_err = abs(geom.lane_width_at_bumper_m - 3.70)
        width_errors.append(w_err)
        lat_errors.append(abs(geom.lateral_error_m))

        if geom.is_geometry_valid:
            success_count += 1

    mean_w_err = float(np.mean(width_errors))
    success_rate = (success_count / num_frames) * 100.0
    passed = success_rate >= 95.0 and mean_w_err < 0.10 and np.mean(latencies) < 20.0

    return PolynomialBenchmarkSummary(
        total_frames=num_frames,
        mean_fitting_latency_ms=float(np.mean(latencies)),
        max_fitting_latency_ms=float(np.max(latencies)),
        mean_lane_width_error_m=mean_w_err,
        mean_lateral_error_m=float(np.mean(lat_errors)),
        fit_success_rate_pct=success_rate,
        passed_precision_target=bool(passed),
    )


def run_polynomial_tracker_diagnostics(verbose: bool = True) -> bool:
    """
    Validate RANSAC polynomial fitting, Mahalanobis distance outlier refinement,
    dual-boundary parallel hallucination, and lateral/heading error calculations.
    """
    if verbose:
        logger.info("Executing LanePolynomialTracker diagnostics...")

    tracker = LanePolynomialTracker()
    extractor = SlidingWindowBEVPixelExtractor()
    visualizer = LaneCurvatureVisualizerOverlay()

    # 1. Synthesize ground-truth curved road: A = 0.001 (R = 500m), B = 0.02, W = 3.70m
    np.random.seed(42)
    y_vals = np.linspace(3.0, 32.0, 150)
    a_true = 0.0008
    b_true = 0.015
    c_l_true = -1.85
    c_r_true = 1.85

    x_l_vals = a_true * (y_vals ** 2) + b_true * y_vals + c_l_true + np.random.normal(0, 0.03, len(y_vals))
    x_r_vals = a_true * (y_vals ** 2) + b_true * y_vals + c_r_true + np.random.normal(0, 0.03, len(y_vals))

    # Add 10% severe outliers
    outlier_idx = np.random.choice(len(y_vals), size=15, replace=False)
    x_l_vals[outlier_idx] += np.random.uniform(0.5, 1.5, size=15)

    left_pts = np.column_stack((x_l_vals, y_vals))
    right_pts = np.column_stack((x_r_vals, y_vals))

    geom = tracker.track_lane_geometry(left_pts, right_pts, eval_lookahead_m=12.0)

    assert geom.is_geometry_valid is True
    assert geom.left_track.status == PolynomialFitStatus.OPTIMAL_FIT
    assert geom.right_track.status == PolynomialFitStatus.OPTIMAL_FIT
    assert abs(geom.lane_width_at_bumper_m - 3.70) < 0.10
    assert abs(geom.radius_of_curvature_m - 625.0) < 150.0
    assert geom.parallelism_residual_variance < 0.02

    # 2. Test Parallel Projection Hallucination: pass None for right boundary
    geom_hal = tracker.track_lane_geometry(left_pts, None, eval_lookahead_m=12.0)
    assert geom_hal.right_track.status == PolynomialFitStatus.PARALLEL_HALLUCINATED
    assert abs(geom_hal.lane_width_at_bumper_m - 3.70) < 0.05
    assert geom_hal.is_geometry_valid is True

    # 3. Test sliding window pixel extraction
    dummy_bev = np.zeros((360, 320), dtype=np.uint8)
    dummy_bev[100:350, 100:104] = 255  # Left
    dummy_bev[100:350, 220:224] = 255  # Right
    pts_l, pts_r = extractor.extract_boundary_coordinates(dummy_bev)
    assert len(pts_l) > 20 and len(pts_r) > 20

    # 4. Test BEV Canvas Visualizer
    canvas = visualizer.render_bev_canvas(geom)
    assert canvas.shape == (600, 400, 3)

    if verbose:
        logger.info("All LanePolynomialTracker diagnostics passed successfully!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_polynomial_tracker_diagnostics(verbose=True)
    report = run_polynomial_tracking_stress_benchmark(num_frames=300)
    print("=" * 80)
    print("APEX-Vision BEV Polynomial Tracker Stress Benchmark Results")
    print(f"Total Frames Tested: {report.total_frames}")
    print(f"Mean Latency: {report.mean_fitting_latency_ms:.3f} ms")
    print(f"Max Latency: {report.max_fitting_latency_ms:.3f} ms")
    print(f"Fit Success Rate: {report.fit_success_rate_pct:.1f}%")
    print(f"Mean Lane Width Error: {report.mean_lane_width_error_m:.4f} m")
    print(f"Mean Lateral Error: {report.mean_lateral_error_m:.4f} m")
    print(f"Passed Precision Target: {report.passed_precision_target}")
    print("=" * 80)

