"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

src/perception/lane_temporal_smoother.py
8-State Extended Kalman Filter & Temporal Motion Compensator for Lane Tracking

Maintains an 8-state EKF vector:
    x = [A_L, B_L, C_L, A_R, B_R, C_R, w_lane, theta_e]^T
incorporating ego-motion compensation via MTi-680 IMU velocity and yaw rate,
10-frame confidence-weighted rolling history, 2-sigma Mahalanobis outlier gating,
dynamic measurement covariance adaptation based on perception quality Q,
and parallel boundary constraint enforcement.
"""

from __future__ import annotations

import collections
import copy
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("apex_vision.perception.lane_temporal_smoother")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enumerations & State Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class TrackingFilterMode(str, Enum):
    """Operational mode of the temporal Kalman filter."""
    INITIALIZING = "INITIALIZING"        # Accumulating baseline history
    DUAL_TRACKING = "DUAL_TRACKING"      # Both left and right boundaries active
    SINGLE_LEFT_COAST = "SINGLE_LEFT_COAST"    # Left boundary active, right projected
    SINGLE_RIGHT_COAST = "SINGLE_RIGHT_COAST"  # Right boundary active, left projected
    COASTING_PREDICTION = "COASTING_PREDICTION" # No valid measurements, coasting on IMU
    DIVERGED_RESET = "DIVERGED_RESET"    # Outlier covariance breach, filter re-initialized


@dataclass(frozen=True)
class PolynomialBand:
    """Evaluated polynomial coordinate with 1-sigma and 2-sigma uncertainty bounds."""
    y_eval_m: float
    x_mean_m: float
    sigma_1_m: float
    sigma_2_m: float


@dataclass
class SmoothedLaneEstimate:
    """
    Stabilized lane boundary output resulting from 8-state EKF estimation.
    """
    timestamp_utc: str
    filter_mode: TrackingFilterMode
    left_poly_coeffs: Tuple[float, float, float]    # (A_L, B_L, C_L)
    right_poly_coeffs: Tuple[float, float, float]   # (A_R, B_R, C_R)
    lane_width_m: float
    heading_error_rad: float
    heading_error_deg: float
    covariance_matrix: np.ndarray                   # (8, 8) covariance
    mahalanobis_distance: float
    outlier_rejected: bool
    motion_compensated: bool
    quality_score: float
    rolling_confidence: float
    left_uncertainty_ribbon: List[PolynomialBand]
    right_uncertainty_ribbon: List[PolynomialBand]
    processing_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "filter_mode": self.filter_mode.value,
            "left_poly_coeffs": [round(float(c), 6) for c in self.left_poly_coeffs],
            "right_poly_coeffs": [round(float(c), 6) for c in self.right_poly_coeffs],
            "lane_width_m": round(float(self.lane_width_m), 4),
            "heading_error_deg": round(float(self.heading_error_deg), 4),
            "mahalanobis_distance": round(float(self.mahalanobis_distance), 4),
            "outlier_rejected": self.outlier_rejected,
            "motion_compensated": self.motion_compensated,
            "quality_score": round(float(self.quality_score), 4),
            "rolling_confidence": round(float(self.rolling_confidence), 4),
            "processing_latency_ms": round(float(self.processing_latency_ms), 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration Parameters
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TemporalSmootherConfig:
    """Configuration for 8-state Extended Kalman Filter and Motion Compensation."""
    # State Vector indices:
    # 0: A_L (curvature), 1: B_L (heading), 2: C_L (lateral offset)
    # 3: A_R (curvature), 4: B_R (heading), 5: C_R (lateral offset)
    # 6: w_lane (lane width), 7: theta_e (centerline heading error)
    state_dim: int = 8
    measurement_dim: int = 6  # [A_L, B_L, C_L, A_R, B_R, C_R]

    # Nominal physical parameters
    nominal_lane_width_m: float = 3.70
    max_lane_width_m: float = 4.80
    min_lane_width_m: float = 2.60
    max_heading_error_rad: float = math.radians(35.0)

    # Rolling history capacity
    history_window_size: int = 10
    max_coast_frames: int = 15

    # Outlier rejection thresholds (Chi-square gating)
    # Chi2(6 dof, 0.95) = 12.59; Chi2(6 dof, 0.99) = 16.81
    mahalanobis_gating_threshold_chi2: float = 14.50
    width_outlier_sigma_factor: float = 2.0

    # Process noise covariance diagonals Q_diag
    q_diag_curvature: float = 1e-7      # A_L, A_R process noise
    q_diag_heading: float = 1e-4        # B_L, B_R process noise
    q_diag_offset: float = 1e-3         # C_L, C_R process noise
    q_diag_width: float = 1e-5          # w_lane process noise
    q_diag_theta_e: float = 1e-4        # theta_e process noise

    # Measurement noise baseline diagonals R_diag (at Q=1.0)
    r_diag_curvature: float = 5e-6
    r_diag_heading: float = 2e-3
    r_diag_offset: float = 1e-2

    # Initial state covariance P0
    p0_diag: Tuple[float, ...] = (
        1e-4, 1e-2, 0.25,
        1e-4, 1e-2, 0.25,
        0.05, 1e-2
    )

    # Uncertainty evaluation distances
    eval_longitudinal_points_m: Tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 8-State Extended Kalman Filter Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LaneTemporalSmoother:
    """
    8-State Extended Kalman Filter for robust lane tracking and motion compensation.

    Standards Compliance:
        - APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
        - Driver-supervised SAE Level 2 execution
        - ISO 26262-style process; ASIL-B aligned; not independently assessed
    """

    def __init__(self, config: Optional[TemporalSmootherConfig] = None) -> None:
        self.config = config or TemporalSmootherConfig()
        self._lock = threading.Lock()

        # State vector x: shape (8, 1)
        self._x = np.zeros((self.config.state_dim, 1), dtype=np.float64)
        # Covariance matrix P: shape (8, 8)
        self._P = np.diag(self.config.p0_diag).astype(np.float64)

        # Process Noise Matrix Q: shape (8, 8)
        self._Q = np.diag([
            self.config.q_diag_curvature,
            self.config.q_diag_heading,
            self.config.q_diag_offset,
            self.config.q_diag_curvature,
            self.config.q_diag_heading,
            self.config.q_diag_offset,
            self.config.q_diag_width,
            self.config.q_diag_theta_e,
        ]).astype(np.float64)

        # Baseline Measurement Noise Matrix R0: shape (6, 6)
        self._R0 = np.diag([
            self.config.r_diag_curvature,
            self.config.r_diag_heading,
            self.config.r_diag_offset,
            self.config.r_diag_curvature,
            self.config.r_diag_heading,
            self.config.r_diag_offset,
        ]).astype(np.float64)

        # Measurement observation matrix H: shape (6, 8)
        # Maps state [A_L, B_L, C_L, A_R, B_R, C_R, w, theta_e] to measurement [A_L, B_L, C_L, A_R, B_R, C_R]
        self._H = np.zeros((6, 8), dtype=np.float64)
        self._H[0, 0] = 1.0  # A_L
        self._H[1, 1] = 1.0  # B_L
        self._H[2, 2] = 1.0  # C_L
        self._H[3, 3] = 1.0  # A_R
        self._H[4, 4] = 1.0  # B_R
        self._H[5, 5] = 1.0  # C_R

        # Filter lifecycle variables
        self._mode: TrackingFilterMode = TrackingFilterMode.INITIALIZING
        self._frame_count: int = 0
        self._coast_count: int = 0
        self._last_timestamp: Optional[float] = None

        # Rolling history buffers for confidence and width checks
        self._history_width: Deque[float] = collections.deque(
            maxlen=self.config.history_window_size
        )
        self._history_confidence: Deque[float] = collections.deque(
            maxlen=self.config.history_window_size
        )

        # Reset filter to nominal straight road
        self._initialize_nominal_state()

    def _initialize_nominal_state(self) -> None:
        """Set state vector to nominal straight lane with 3.7m width."""
        half_w = self.config.nominal_lane_width_m / 2.0
        self._x = np.array([
            [0.0],      # A_L
            [0.0],      # B_L
            [-half_w],  # C_L
            [0.0],      # A_R
            [0.0],      # B_R
            [half_w],   # C_R
            [self.config.nominal_lane_width_m],  # w_lane
            [0.0],      # theta_e
        ], dtype=np.float64)
        self._P = np.diag(self.config.p0_diag).astype(np.float64)
        self._coast_count = 0
        self._mode = TrackingFilterMode.INITIALIZING

    # ───────────────────────────────────────────────────────────────────────────
    # Public Execution API
    # ───────────────────────────────────────────────────────────────────────────

    def update(
        self,
        left_poly_meas: Optional[Tuple[float, float, float]],
        right_poly_meas: Optional[Tuple[float, float, float]],
        perception_quality: float,
        dt_s: float = 0.0333,
        ego_vx_mps: float = 0.0,
        ego_vy_mps: float = 0.0,
        ego_yaw_rate_radps: float = 0.0,
    ) -> SmoothedLaneEstimate:
        """
        Execute prediction, motion compensation, outlier gating, and Kalman measurement update.

        Args:
            left_poly_meas: Measured (A_L, B_L, C_L) or None if occluded.
            right_poly_meas: Measured (A_R, B_R, C_R) or None if occluded.
            perception_quality: Quality score Q in [0, 1] from LaneQualityScorer.
            dt_s: Time step in seconds.
            ego_vx_mps: Ego forward longitudinal velocity in m/s.
            ego_vy_mps: Ego lateral velocity in m/s.
            ego_yaw_rate_radps: Yaw rate from MTi-680 IMU in rad/s.
        """
        t_start = time.perf_counter()

        with self._lock:
            self._frame_count += 1
            dt = float(np.clip(dt_s, 0.005, 0.200))
            q = float(np.clip(perception_quality, 0.01, 1.0))
            self._history_confidence.append(q)

            # 1. EKF Prediction Step with IMU Motion Compensation
            motion_active = self._predict_step(dt, ego_vx_mps, ego_vy_mps, ego_yaw_rate_radps)

            # 2. Measurement Synthesis & Dual-Boundary Parallel Projection Fallback
            meas_vec, active_meas_mask, synthesized = self._prepare_measurement_vector(
                left_poly_meas, right_poly_meas
            )

            # 3. Dynamic Measurement Noise Scaling based on Quality Q
            R_k = self._compute_adaptive_measurement_covariance(q, active_meas_mask)

            # 4. Outlier Rejection via Mahalanobis Distance & Lane Width 2-Sigma Gating
            outlier_rejected = False
            mahalanobis_dist = 0.0

            if active_meas_mask[2] or active_meas_mask[5]:
                is_outlier, mahalanobis_dist = self._evaluate_outlier_gating(
                    meas_vec, R_k, active_meas_mask
                )
                if is_outlier:
                    outlier_rejected = True
                    self._coast_count += 1
                else:
                    self._coast_count = 0
            else:
                self._coast_count += 1

            # 5. Kalman Measurement Update Step (Joseph Form)
            if not outlier_rejected and (active_meas_mask[2] or active_meas_mask[5]):
                self._update_step(meas_vec, R_k, active_meas_mask)
                self._enforce_parallel_boundary_constraints()
            else:
                # Coasting without update
                if self._coast_count > self.config.max_coast_frames:
                    logger.warning("Kalman smoother exceeded maximum coast frames; resetting.")
                    self._initialize_nominal_state()
                    self._mode = TrackingFilterMode.DIVERGED_RESET
                else:
                    self._mode = TrackingFilterMode.COASTING_PREDICTION

            # Determine filter operating mode
            if self._mode not in (TrackingFilterMode.DIVERGED_RESET, TrackingFilterMode.COASTING_PREDICTION):
                if active_meas_mask[2] and active_meas_mask[5]:
                    self._mode = TrackingFilterMode.DUAL_TRACKING
                elif active_meas_mask[2]:
                    self._mode = TrackingFilterMode.SINGLE_LEFT_COAST
                elif active_meas_mask[5]:
                    self._mode = TrackingFilterMode.SINGLE_RIGHT_COAST

            # 6. Extract Smoothed Coefficients & Uncertainty Ribbons
            res = self._build_output_estimate(
                outlier_rejected=outlier_rejected,
                motion_compensated=motion_active,
                mahalanobis_dist=mahalanobis_dist,
                quality=q,
                start_time=t_start,
            )

            # Update width history
            self._history_width.append(res.lane_width_m)
            return res

    # ───────────────────────────────────────────────────────────────────────────
    # EKF Math & Motion Compensation
    # ───────────────────────────────────────────────────────────────────────────

    def _predict_step(
        self, dt: float, vx: float, vy: float, omega_z: float
    ) -> bool:
        """
        Propagate 8-state vector forward in time using kinematic bicycle / SE(2)
        coordinate transformation for polynomials:
            dx = (vx * dt) * sin(theta_e) + (vy * dt)
            dy = (vx * dt) * cos(theta_e)
            dpsi = omega_z * dt
        """
        # Linearized State Transition Matrix F
        F = np.eye(self.config.state_dim, dtype=np.float64)

        motion_applied = False
        if abs(vx) > 0.1 or abs(omega_z) > 0.005:
            motion_applied = True
            delta_y = float(vx * dt)
            delta_x = float(vy * dt)
            delta_psi = float(omega_z * dt)

            # Transform polynomials x(y) under ego motion:
            # Shift in evaluation frame y -> y + delta_y, x -> x - delta_x - delta_psi * y
            # For x_L(y) = A y^2 + B y + C:
            # New C' = C - delta_x + B * delta_y + A * delta_y^2
            # New B' = B + 2 * A * delta_y - delta_psi
            # New A' = A
            for base_idx in [0, 3]:  # Left at 0, Right at 3
                a = self._x[base_idx, 0]
                b = self._x[base_idx + 1, 0]
                c = self._x[base_idx + 2, 0]

                new_c = c - delta_x + b * delta_y + a * (delta_y ** 2)
                new_b = b + 2.0 * a * delta_y - delta_psi
                new_a = a

                self._x[base_idx, 0] = new_a
                self._x[base_idx + 1, 0] = new_b
                self._x[base_idx + 2, 0] = new_c

                # Fill transition Jacobian elements
                F[base_idx + 1, base_idx] = 2.0 * delta_y
                F[base_idx + 2, base_idx + 1] = delta_y
                F[base_idx + 2, base_idx] = delta_y ** 2

            # Heading error theta_e shifts by -delta_psi
            self._x[7, 0] -= delta_psi
            self._x[7, 0] = np.clip(self._x[7, 0], -self.config.max_heading_error_rad, self.config.max_heading_error_rad)

        # Propagate state covariance: P_pred = F * P * F^T + Q
        self._P = F @ self._P @ F.T + (self._Q * dt)
        # Ensure symmetric positive-definite covariance
        self._P = 0.5 * (self._P + self._P.T)
        return motion_applied

    def _prepare_measurement_vector(
        self,
        left_poly: Optional[Tuple[float, float, float]],
        right_poly: Optional[Tuple[float, float, float]],
    ) -> Tuple[np.ndarray, List[bool], bool]:
        """
        Form 6-dimensional measurement vector. If one line is absent,
        hallucinates it via parallel projection at nominal lane width 3.7m.
        """
        z = np.zeros((6, 1), dtype=np.float64)
        mask = [False] * 6
        synthesized = False
        nominal_w = float(self._x[6, 0])  # Use currently tracked lane width

        has_left = left_poly is not None
        has_right = right_poly is not None

        if has_left and has_right:
            z[0, 0], z[1, 0], z[2, 0] = left_poly
            z[3, 0], z[4, 0], z[5, 0] = right_poly
            mask = [True] * 6
        elif has_left and not has_right:
            z[0, 0], z[1, 0], z[2, 0] = left_poly
            # Hallucinate right boundary: identical curvature & heading, shifted right by w
            z[3, 0] = left_poly[0]
            z[4, 0] = left_poly[1]
            z[5, 0] = left_poly[2] + nominal_w
            mask = [True, True, True, False, False, False]
            synthesized = True
        elif has_right and not has_left:
            z[3, 0], z[4, 0], z[5, 0] = right_poly
            # Hallucinate left boundary: shifted left by w
            z[0, 0] = right_poly[0]
            z[1, 0] = right_poly[1]
            z[2, 0] = right_poly[2] - nominal_w
            mask = [False, False, False, True, True, True]
            synthesized = True

        return z, mask, synthesized

    def _compute_adaptive_measurement_covariance(
        self, quality: float, active_mask: List[bool]
    ) -> np.ndarray:
        """
        Scale measurement covariance inversely with perception quality:
            R(t) = R_0 / (Q^2 + eps)
        Inflate noise for unmeasured/synthesized boundaries.
        """
        eps = 1e-3
        q_scale = 1.0 / (quality ** 2 + eps)
        R = copy.deepcopy(self._R0) * q_scale

        for i in range(6):
            if not active_mask[i]:
                # 10x penalty for hallucinated / occluded boundaries
                R[i, i] *= 10.0

        return R

    def _evaluate_outlier_gating(
        self, z: np.ndarray, R: np.ndarray, active_mask: List[bool]
    ) -> Tuple[bool, float]:
        """
        Gating using Mahalanobis distance on innovation:
            y = z - H * x
            S = H * P * H^T + R
            D_M^2 = y^T * S^-1 * y
        Plus lane width 2-sigma jump check.
        """
        # Active indices
        active_indices = [i for i, active in enumerate(active_mask) if active]
        if not active_indices:
            return False, 0.0

        H_act = self._H[active_indices, :]
        z_act = z[active_indices, :]
        R_act = R[np.ix_(active_indices, active_indices)]

        # Innovation
        y = z_act - H_act @ self._x
        S = H_act @ self._P @ H_act.T + R_act

        try:
            S_inv = np.linalg.inv(S)
            d_mahalanobis_sq = float((y.T @ S_inv @ y).item())
        except np.linalg.LinAlgError:
            d_mahalanobis_sq = 999.0

        is_outlier = d_mahalanobis_sq > self.config.mahalanobis_gating_threshold_chi2

        # Physical Lane Width 2-sigma check
        if active_mask[2] and active_mask[5]:
            measured_w = float(z[5, 0] - z[2, 0])
            if len(self._history_width) >= 3:
                w_mean = float(np.mean(self._history_width))
                w_std = max(0.08, float(np.std(self._history_width)))
                if abs(measured_w - w_mean) > self.config.width_outlier_sigma_factor * w_std:
                    is_outlier = True

        return is_outlier, math.sqrt(max(0.0, d_mahalanobis_sq))

    def _update_step(
        self, z: np.ndarray, R: np.ndarray, active_mask: List[bool]
    ) -> None:
        """
        Execute numerically stable Joseph-form covariance update:
            K = P * H^T * (H * P * H^T + R)^-1
            x = x + K * y
            P = (I - K * H) * P * (I - K * H)^T + K * R * K^T
        """
        active_indices = [i for i, active in enumerate(active_mask) if active]
        H_act = self._H[active_indices, :]
        z_act = z[active_indices, :]
        R_act = R[np.ix_(active_indices, active_indices)]

        y = z_act - H_act @ self._x
        S = H_act @ self._P @ H_act.T + R_act

        try:
            K = self._P @ H_act.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            logger.warning("Singular innovation covariance matrix S in Kalman update; skipping update.")
            return

        # State update
        self._x = self._x + K @ y

        # Joseph form covariance update
        I_KH = np.eye(self.config.state_dim, dtype=np.float64) - K @ H_act
        self._P = I_KH @ self._P @ I_KH.T + K @ R_act @ K.T
        self._P = 0.5 * (self._P + self._P.T)

    def _enforce_parallel_boundary_constraints(self) -> None:
        """
        Enforce highway geometric consistency:
            1. Lane width w_lane = C_R - C_L bounded in [2.6m, 4.8m]
            2. Curvature consensus: A_L and A_R blended towards average
            3. Heading consensus: B_L and B_R blended
            4. Heading error theta_e = 0.5 * (B_L + B_R)
        """
        # Clamp lane width
        c_l = self._x[2, 0]
        c_r = self._x[5, 0]
        current_w = c_r - c_l

        # Bound width
        bounded_w = float(np.clip(current_w, self.config.min_lane_width_m, self.config.max_lane_width_m))
        self._x[6, 0] = 0.85 * self._x[6, 0] + 0.15 * bounded_w

        # If width violated bounds, adjust lateral offsets symmetrically
        if abs(bounded_w - current_w) > 0.05:
            center = (c_l + c_r) / 2.0
            self._x[2, 0] = center - bounded_w / 2.0
            self._x[5, 0] = center + bounded_w / 2.0

        # Curvature consensus: Highway lanes share parallel curvature
        mean_a = 0.5 * (self._x[0, 0] + self._x[3, 0])
        self._x[0, 0] = 0.80 * self._x[0, 0] + 0.20 * mean_a
        self._x[3, 0] = 0.80 * self._x[3, 0] + 0.20 * mean_a

        # Heading error calculation
        mean_b = 0.5 * (self._x[1, 0] + self._x[4, 0])
        self._x[7, 0] = float(np.clip(mean_b, -self.config.max_heading_error_rad, self.config.max_heading_error_rad))

    def _build_output_estimate(
        self,
        outlier_rejected: bool,
        motion_compensated: bool,
        mahalanobis_dist: float,
        quality: float,
        start_time: float,
    ) -> SmoothedLaneEstimate:
        """Package internal EKF state and uncertainty ribbons into public result object."""
        left_poly = (float(self._x[0, 0]), float(self._x[1, 0]), float(self._x[2, 0]))
        right_poly = (float(self._x[3, 0]), float(self._x[4, 0]), float(self._x[5, 0]))
        w_lane = float(self._x[5, 0] - self._x[2, 0])
        theta_e = float(self._x[7, 0])

        # Evaluate uncertainty ribbons along longitudinal lookahead
        left_ribbon: List[PolynomialBand] = []
        right_ribbon: List[PolynomialBand] = []

        for y in self.config.eval_longitudinal_points_m:
            # For Left: x = A*y^2 + B*y + C, variance = J * P_L * J^T where J = [y^2, y, 1]
            j_vec = np.array([[y ** 2, y, 1.0]], dtype=np.float64)

            # Left covariance slice (indices 0..2)
            p_left = self._P[0:3, 0:3]
            var_left = float((j_vec @ p_left @ j_vec.T).item())
            sigma_left = math.sqrt(max(1e-6, var_left))
            x_l = left_poly[0] * (y ** 2) + left_poly[1] * y + left_poly[2]
            left_ribbon.append(PolynomialBand(
                y_eval_m=y,
                x_mean_m=x_l,
                sigma_1_m=sigma_left,
                sigma_2_m=2.0 * sigma_left,
            ))

            # Right covariance slice (indices 3..5)
            p_right = self._P[3:6, 3:6]
            var_right = float((j_vec @ p_right @ j_vec.T).item())
            sigma_right = math.sqrt(max(1e-6, var_right))
            x_r = right_poly[0] * (y ** 2) + right_poly[1] * y + right_poly[2]
            right_ribbon.append(PolynomialBand(
                y_eval_m=y,
                x_mean_m=x_r,
                sigma_1_m=sigma_right,
                sigma_2_m=2.0 * sigma_right,
            ))

        mean_roll_conf = float(np.mean(self._history_confidence)) if self._history_confidence else quality
        lat_ms = (time.perf_counter() - start_time) * 1000.0

        return SmoothedLaneEstimate(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            filter_mode=self._mode,
            left_poly_coeffs=left_poly,
            right_poly_coeffs=right_poly,
            lane_width_m=w_lane,
            heading_error_rad=theta_e,
            heading_error_deg=math.degrees(theta_e),
            covariance_matrix=copy.deepcopy(self._P),
            mahalanobis_distance=mahalanobis_dist,
            outlier_rejected=outlier_rejected,
            motion_compensated=motion_compensated,
            quality_score=quality,
            rolling_confidence=mean_roll_conf,
            left_uncertainty_ribbon=left_ribbon,
            right_uncertainty_ribbon=right_ribbon,
            processing_latency_ms=lat_ms,
        )

    def reset(self) -> None:
        """Manual reset of Kalman filter."""
        with self._lock:
            self._initialize_nominal_state()
            self._history_width.clear()
            self._history_confidence.clear()
            logger.info("LaneTemporalSmoother reset to nominal state.")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Telemetry Exporter, Curvature Dynamics & Stanley Trajectory Coupler
# ═══════════════════════════════════════════════════════════════════════════════

class LaneCurvatureDynamicsAnalyzer:
    """
    Computes road curvature dynamics, clothoid parameters, lateral acceleration demand,
    and ISO 26262 ASIL-B aligned speed advisory limits from smoothed polynomials.

    Polynomial formulation: x(y) = A * y^2 + B * y + C
    Curvature kappa(y) = 2*A / (1 + (2*A*y + B)^2)^1.5
    Radius of curvature R(y) = 1 / |kappa(y)|
    """

    def __init__(self, max_lateral_accel_comfort_mps2: float = 2.0) -> None:
        self.a_lat_max = float(max_lateral_accel_comfort_mps2)

    def analyze_curvature(
        self,
        poly_coeffs: Tuple[float, float, float],
        eval_distance_m: float = 15.0,
        current_speed_mps: float = 20.0,
    ) -> Dict[str, Any]:
        """
        Evaluate curvature, radius, lateral acceleration demand, and advisory speed.
        """
        a, b, _ = poly_coeffs
        y = float(eval_distance_m)

        # First and second derivatives of x with respect to y
        dx_dy = 2.0 * a * y + b
        d2x_dy2 = 2.0 * a

        denom = (1.0 + dx_dy ** 2) ** 1.5
        if abs(denom) < 1e-6:
            kappa = 0.0
        else:
            kappa = d2x_dy2 / denom

        abs_kappa = abs(kappa)
        radius_m = (1.0 / abs_kappa) if abs_kappa > 1e-5 else 10000.0

        # Lateral acceleration demand: a_lat = v^2 * kappa
        a_lat_demand = (current_speed_mps ** 2) * abs_kappa

        # Advisory maximum speed based on lateral acceleration limit
        if abs_kappa > 1e-4:
            v_advisory_mps = math.sqrt(self.a_lat_max / abs_kappa)
        else:
            v_advisory_mps = 38.89  # ~140 km/h highway ceiling

        return {
            "eval_distance_m": round(y, 2),
            "curvature_kappa_radpm": round(float(kappa), 6),
            "radius_of_curvature_m": round(float(radius_m), 2),
            "lateral_accel_demand_mps2": round(float(a_lat_demand), 3),
            "advisory_speed_mps": round(float(v_advisory_mps), 2),
            "advisory_speed_kph": round(float(v_advisory_mps * 3.6), 1),
            "exceeds_comfort_limit": bool(a_lat_demand > self.a_lat_max),
        }


class StanleyTrajectorySteeringCoupler:
    """
    Directly interfaces the smoothed EKF lane representation with the Stanley
    path tracking controller, synthesizing scheduled lateral gains and target steer angles.

    Stanley Control Law:
        delta(t) = theta_e(t) + arctan( (k * e_lat(t)) / (v_x + v_soft) )
    """

    def __init__(
        self,
        nominal_gain_k: float = 0.85,
        softening_speed_mps: float = 1.0,
        max_steer_angle_rad: float = math.radians(28.0),
    ) -> None:
        self.k_nominal = float(nominal_gain_k)
        self.v_soft = float(softening_speed_mps)
        self.max_steer = float(max_steer_angle_rad)

    def compute_steering_command(
        self,
        estimate: SmoothedLaneEstimate,
        ego_speed_mps: float,
        lookahead_distance_m: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Derive scheduled Stanley steering command from smoothed left and right polynomials.
        """
        # Centerline polynomial: x_center(y) = 0.5 * (x_L(y) + x_R(y))
        a_l, b_l, c_l = estimate.left_poly_coeffs
        a_r, b_r, c_r = estimate.right_poly_coeffs

        a_c = 0.5 * (a_l + a_r)
        b_c = 0.5 * (b_l + b_r)
        c_c = 0.5 * (c_l + c_r)

        # Cross-track lateral error at evaluation lookahead distance
        y = float(lookahead_distance_m)
        x_center_at_y = a_c * (y ** 2) + b_c * y + c_c
        # Vehicle is at x = 0; error is deviation of vehicle from centerline
        e_lat = -x_center_at_y

        # Heading error from centerline
        theta_e = estimate.heading_error_rad

        # Schedule gain based on perception quality and speed
        # Under lower quality Q, reduce aggressiveness to prevent jerky oscillation
        q_scale = float(np.clip(estimate.quality_score, 0.40, 1.0))
        speed = max(0.5, float(ego_speed_mps))
        scheduled_k = self.k_nominal * q_scale

        # Stanley cross-track correction term
        cross_track_term = math.atan2(scheduled_k * e_lat, speed + self.v_soft)

        # Combined steering command
        delta_raw = theta_e + cross_track_term
        delta_clipped = float(np.clip(delta_raw, -self.max_steer, self.max_steer))

        return {
            "lateral_error_m": round(float(e_lat), 4),
            "heading_error_deg": round(float(math.degrees(theta_e)), 3),
            "scheduled_gain_k": round(float(scheduled_k), 3),
            "cross_track_term_deg": round(float(math.degrees(cross_track_term)), 3),
            "commanded_steer_angle_deg": round(float(math.degrees(delta_clipped)), 3),
            "commanded_steer_angle_rad": round(float(delta_clipped), 5),
            "is_actuator_saturated": bool(abs(delta_raw) > self.max_steer),
        }


class ExtendedKalmanFilterCovarianceMonitor:
    """
    Supervises the numerical integrity of the 8x8 EKF state covariance matrix P.
    Monitors trace, determinant, condition number, and symmetry to prevent divergence.
    """

    def __init__(self, max_allowed_trace: float = 100.0, max_condition_number: float = 1e8) -> None:
        self.max_trace = float(max_allowed_trace)
        self.max_cond = float(max_condition_number)

    def check_covariance_health(self, P: np.ndarray) -> Dict[str, Any]:
        """Verify covariance matrix P conforms to ASIL-B numerical stability requirements."""
        trace = float(np.trace(P))
        # Check symmetry
        sym_diff = float(np.max(np.abs(P - P.T)))
        is_symmetric = sym_diff < 1e-5

        # Check positive-definiteness via eigenvalues
        try:
            eigvals = np.linalg.eigvalsh(P)
            min_eig = float(np.min(eigvals))
            max_eig = float(np.max(eigvals))
            cond_num = max_eig / (min_eig + 1e-12)
            is_pos_def = min_eig > 0.0
        except np.linalg.LinAlgError:
            min_eig = -1.0
            max_eig = 999.0
            cond_num = 1e12
            is_pos_def = False

        is_healthy = (
            is_symmetric
            and is_pos_def
            and trace < self.max_trace
            and cond_num < self.max_cond
        )

        return {
            "is_healthy": bool(is_healthy),
            "trace": round(trace, 4),
            "min_eigenvalue": round(min_eig, 8),
            "max_eigenvalue": round(max_eig, 4),
            "condition_number": round(cond_num, 2),
            "symmetry_max_error": round(sym_diff, 8),
        }


class LaneTemporalTelemetryExporter:
    """
    Formats temporal filter trajectories and uncertainty envelopes into
    structured JSON/GeoJSON for HUD overlays and ROS2 marker visualization.
    """

    def __init__(self, decimal_precision: int = 4) -> None:
        self.precision = int(decimal_precision)

    def to_uncertainty_ribbon_dict(
        self, estimate: SmoothedLaneEstimate
    ) -> Dict[str, Any]:
        """Convert uncertainty ribbons into structured plotting coordinate dictionary."""
        return {
            "timestamp": estimate.timestamp_utc,
            "filter_mode": estimate.filter_mode.value,
            "lane_width_m": round(estimate.lane_width_m, self.precision),
            "heading_error_deg": round(estimate.heading_error_deg, self.precision),
            "mahalanobis_distance": round(estimate.mahalanobis_distance, self.precision),
            "outlier_rejected": estimate.outlier_rejected,
            "left_band": [
                {
                    "y_m": round(b.y_eval_m, self.precision),
                    "x_mean_m": round(b.x_mean_m, self.precision),
                    "sigma_1_m": round(b.sigma_1_m, self.precision),
                    "sigma_2_m": round(b.sigma_2_m, self.precision),
                }
                for b in estimate.left_uncertainty_ribbon
            ],
            "right_band": [
                {
                    "y_m": round(b.y_eval_m, self.precision),
                    "x_mean_m": round(b.x_mean_m, self.precision),
                    "sigma_1_m": round(b.sigma_1_m, self.precision),
                    "sigma_2_m": round(b.sigma_2_m, self.precision),
                }
                for b in estimate.right_uncertainty_ribbon
            ],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Global Singleton & Benchmark Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_TEMPORAL_SMOOTHER: Optional[LaneTemporalSmoother] = None
_GLOBAL_TELEMETRY_EXPORTER: Optional[LaneTemporalTelemetryExporter] = None


def get_lane_temporal_smoother(
    config: Optional[TemporalSmootherConfig] = None,
) -> LaneTemporalSmoother:
    """Retrieve or instantiate global LaneTemporalSmoother singleton."""
    global _GLOBAL_TEMPORAL_SMOOTHER
    if _GLOBAL_TEMPORAL_SMOOTHER is None:
        _GLOBAL_TEMPORAL_SMOOTHER = LaneTemporalSmoother(config=config)
    return _GLOBAL_TEMPORAL_SMOOTHER


def get_temporal_telemetry_exporter() -> LaneTemporalTelemetryExporter:
    """Retrieve global LaneTemporalTelemetryExporter singleton."""
    global _GLOBAL_TELEMETRY_EXPORTER
    if _GLOBAL_TELEMETRY_EXPORTER is None:
        _GLOBAL_TELEMETRY_EXPORTER = LaneTemporalTelemetryExporter()
    return _GLOBAL_TELEMETRY_EXPORTER


@dataclass
class SmootherBenchmarkResults:
    """Statistical summary of 500-frame Kalman smoother validation."""
    total_frames: int
    mean_latency_ms: float
    max_latency_ms: float
    outlier_rejections: int
    mean_lane_width_m: float
    std_lane_width_m: float
    dual_tracking_count: int
    single_coast_count: int
    passed_ekf_stability_test: bool


def run_temporal_smoother_stress_benchmark(num_frames: int = 500) -> SmootherBenchmarkResults:
    """
    Stress-test the 8-state EKF across curves, sudden IMU yaw steps, artificial
    outlier jumps (> 1.5m), single-line occlusions, and recovery phases.
    """
    smoother = LaneTemporalSmoother()
    latencies: List[float] = []
    widths: List[float] = []
    outlier_count = 0
    dual_count = 0
    single_count = 0

    np.random.seed(42)

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Simulate vehicle dynamics
        vx = 22.0  # ~80 km/h
        vy = float(np.random.normal(0.0, 0.05))
        yaw_rate = 0.01 * math.sin(i / 20.0)  # Gentle S-curve

        # Nominal lane parameters
        a_true = 0.0005 * math.sin(i / 30.0)
        b_true = yaw_rate * 1.5
        c_left_true = -1.85 + 0.1 * math.cos(i / 15.0)
        c_right_true = c_left_true + 3.70

        # Inject noise into measurements
        a_meas = a_true + float(np.random.normal(0, 1e-4))
        b_meas = b_true + float(np.random.normal(0, 0.01))
        c_l_meas = c_left_true + float(np.random.normal(0, 0.03))
        c_r_meas = c_right_true + float(np.random.normal(0, 0.03))

        left_m: Optional[Tuple[float, float, float]] = (a_meas, b_meas, c_l_meas)
        right_m: Optional[Tuple[float, float, float]] = (a_meas, b_meas, c_r_meas)
        quality = 0.90

        # Inject anomaly: single line occlusion between frames 100..140
        if 100 <= i < 140:
            right_m = None  # Right line occluded; must hallucinate from left
            quality = 0.70

        # Inject anomaly: massive outlier jump at frame 200 (must be rejected)
        if i == 200:
            left_m = (a_meas + 0.01, b_meas + 0.2, c_l_meas + 2.50)  # Jump > 2.5m

        res = smoother.update(
            left_poly_meas=left_m,
            right_poly_meas=right_m,
            perception_quality=quality,
            dt_s=0.0333,
            ego_vx_mps=vx,
            ego_vy_mps=vy,
            ego_yaw_rate_radps=yaw_rate,
        )

        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        widths.append(res.lane_width_m)

        if res.outlier_rejected:
            outlier_count += 1
        if res.filter_mode == TrackingFilterMode.DUAL_TRACKING:
            dual_count += 1
        elif res.filter_mode in (TrackingFilterMode.SINGLE_LEFT_COAST, TrackingFilterMode.SINGLE_RIGHT_COAST):
            single_count += 1

    mean_w = float(np.mean(widths))
    std_w = float(np.std(widths))
    passed = (
        outlier_count >= 1
        and abs(mean_w - 3.70) < 0.15
        and std_w < 0.15
        and float(np.mean(latencies)) < 2.0
    )

    return SmootherBenchmarkResults(
        total_frames=num_frames,
        mean_latency_ms=float(np.mean(latencies)),
        max_latency_ms=float(np.max(latencies)),
        outlier_rejections=outlier_count,
        mean_lane_width_m=mean_w,
        std_lane_width_m=std_w,
        dual_tracking_count=dual_count,
        single_coast_count=single_count,
        passed_ekf_stability_test=bool(passed),
    )


class LaneSmootherCANPublisher:
    """
    Serializes 8-state EKF smoothed polynomial coefficients and covariance metrics
    into production CAN bus payloads for vehicle network dissemination.
    Arbitration IDs:
        0x160: Smoothed Left Polynomial (A_L, B_L, C_L, Quality)
        0x161: Smoothed Right Polynomial (A_R, B_R, C_R, Quality)
        0x162: Lane Geometry & Health (w_lane, theta_e, Outlier, Mode)
    """

    def __init__(self) -> None:
        pass

    def encode_left_boundary_frame(self, estimate: SmoothedLaneEstimate) -> Dict[str, Any]:
        """Encode Left polynomial coefficients into 8-byte CAN frame 0x160."""
        a, b, c = estimate.left_poly_coeffs
        # Scaling: A * 1e6 (int16), B * 1e3 (int16), C * 1e2 (int16), Quality * 255 (uint8)
        a_int = int(np.clip(round(a * 1e6), -32768, 32767))
        b_int = int(np.clip(round(b * 1e3), -32768, 32767))
        c_int = int(np.clip(round(c * 100.0), -32768, 32767))
        q_byte = int(np.clip(round(estimate.quality_score * 255.0), 0, 255))

        payload = [
            (a_int >> 8) & 0xFF, a_int & 0xFF,
            (b_int >> 8) & 0xFF, b_int & 0xFF,
            (c_int >> 8) & 0xFF, c_int & 0xFF,
            q_byte & 0xFF, 0x00
        ]
        return {
            "arbitration_id": 0x160,
            "dlc": 8,
            "payload_hex": "".join(f"{b:02X}" for b in payload),
            "payload_bytes": payload,
            "timestamp": estimate.timestamp_utc,
        }

    def encode_right_boundary_frame(self, estimate: SmoothedLaneEstimate) -> Dict[str, Any]:
        """Encode Right polynomial coefficients into 8-byte CAN frame 0x161."""
        a, b, c = estimate.right_poly_coeffs
        a_int = int(np.clip(round(a * 1e6), -32768, 32767))
        b_int = int(np.clip(round(b * 1e3), -32768, 32767))
        c_int = int(np.clip(round(c * 100.0), -32768, 32767))
        q_byte = int(np.clip(round(estimate.quality_score * 255.0), 0, 255))

        payload = [
            (a_int >> 8) & 0xFF, a_int & 0xFF,
            (b_int >> 8) & 0xFF, b_int & 0xFF,
            (c_int >> 8) & 0xFF, c_int & 0xFF,
            q_byte & 0xFF, 0x00
        ]
        return {
            "arbitration_id": 0x161,
            "dlc": 8,
            "payload_hex": "".join(f"{b:02X}" for b in payload),
            "payload_bytes": payload,
            "timestamp": estimate.timestamp_utc,
        }

    def encode_lane_geometry_frame(self, estimate: SmoothedLaneEstimate) -> Dict[str, Any]:
        """Encode width, heading error, and filter status into 8-byte CAN frame 0x162."""
        w_int = int(np.clip(round(estimate.lane_width_m * 100.0), 0, 65535))
        theta_int = int(np.clip(round(estimate.heading_error_deg * 100.0), -32768, 32767))
        outlier_byte = 1 if estimate.outlier_rejected else 0
        mode_byte = {
            TrackingFilterMode.INITIALIZING: 0x00,
            TrackingFilterMode.DUAL_TRACKING: 0x01,
            TrackingFilterMode.SINGLE_LEFT_COAST: 0x02,
            TrackingFilterMode.SINGLE_RIGHT_COAST: 0x03,
            TrackingFilterMode.COASTING_PREDICTION: 0x04,
            TrackingFilterMode.DIVERGED_RESET: 0x05,
        }.get(estimate.filter_mode, 0xFF)

        payload = [
            (w_int >> 8) & 0xFF, w_int & 0xFF,
            (theta_int >> 8) & 0xFF, theta_int & 0xFF,
            outlier_byte & 0xFF, mode_byte & 0xFF,
            0x00, 0x00
        ]
        return {
            "arbitration_id": 0x162,
            "dlc": 8,
            "payload_hex": "".join(f"{b:02X}" for b in payload),
            "payload_bytes": payload,
            "timestamp": estimate.timestamp_utc,
        }


def run_temporal_smoother_diagnostics(verbose: bool = True) -> bool:
    """
    Validate 8-state EKF update math, motion compensation, parallel constraint
    enforcement, 2-sigma Mahalanobis outlier rejection, curvature dynamics,
    and Stanley steering law coupling.
    """
    if verbose:
        logger.info("Executing LaneTemporalSmoother diagnostics...")

    smoother = LaneTemporalSmoother()
    exporter = LaneTemporalTelemetryExporter()
    dynamics_analyzer = LaneCurvatureDynamicsAnalyzer(max_lateral_accel_comfort_mps2=2.0)
    stanley_coupler = StanleyTrajectorySteeringCoupler(nominal_gain_k=0.85)
    cov_monitor = ExtendedKalmanFilterCovarianceMonitor()
    can_publisher = LaneSmootherCANPublisher()

    # 1. Test nominal straight lane update
    left_m = (0.0, 0.0, -1.85)
    right_m = (0.0, 0.0, 1.85)
    res_nominal = smoother.update(
        left_poly_meas=left_m,
        right_poly_meas=right_m,
        perception_quality=0.95,
        dt_s=0.0333,
        ego_vx_mps=20.0,
        ego_yaw_rate_radps=0.0,
    )
    assert res_nominal.outlier_rejected is False
    assert abs(res_nominal.lane_width_m - 3.70) < 0.05
    assert len(res_nominal.left_uncertainty_ribbon) == 6

    # Verify covariance health
    health = cov_monitor.check_covariance_health(res_nominal.covariance_matrix)
    assert health["is_healthy"] is True, f"Covariance unhealthy: {health}"

    # 2. Test hallucination from visible left line
    res_single = smoother.update(
        left_poly_meas=left_m,
        right_poly_meas=None,
        perception_quality=0.80,
        dt_s=0.0333,
        ego_vx_mps=20.0,
    )
    assert res_single.filter_mode == TrackingFilterMode.SINGLE_LEFT_COAST
    assert abs(res_single.lane_width_m - 3.70) < 0.10

    # 3. Test Mahalanobis outlier rejection on huge position jump
    huge_jump_left = (0.0, 0.0, -4.50)  # > 2.6m jump
    res_outlier = smoother.update(
        left_poly_meas=huge_jump_left,
        right_poly_meas=right_m,
        perception_quality=0.90,
        dt_s=0.0333,
    )
    assert res_outlier.outlier_rejected is True, "Expected Mahalanobis / 2-sigma gating to reject jump"

    # 4. Test Curvature Dynamics Analyzer on a curved highway exit ramp
    curved_poly = (0.0012, 0.05, -1.85)  # Radius ~ 416m
    curve_metrics = dynamics_analyzer.analyze_curvature(
        poly_coeffs=curved_poly,
        eval_distance_m=15.0,
        current_speed_mps=25.0,  # 90 km/h
    )
    assert curve_metrics["radius_of_curvature_m"] < 1000.0
    assert "advisory_speed_kph" in curve_metrics

    # 5. Test Stanley Steering Law synthesis
    steering = stanley_coupler.compute_steering_command(
        estimate=res_nominal,
        ego_speed_mps=22.0,
        lookahead_distance_m=5.0,
    )
    assert "commanded_steer_angle_deg" in steering
    assert abs(steering["lateral_error_m"]) < 0.15

    # 6. Test CAN bus payload encoding
    can_left = can_publisher.encode_left_boundary_frame(res_nominal)
    can_geom = can_publisher.encode_lane_geometry_frame(res_nominal)
    assert can_left["arbitration_id"] == 0x160
    assert len(can_left["payload_bytes"]) == 8
    assert can_geom["arbitration_id"] == 0x162

    # Export telemetry check
    telemetry = exporter.to_uncertainty_ribbon_dict(res_nominal)
    assert "left_band" in telemetry and len(telemetry["left_band"]) == 6

    if verbose:
        logger.info("All LaneTemporalSmoother diagnostics passed successfully!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_temporal_smoother_diagnostics(verbose=True)
    bench_res = run_temporal_smoother_stress_benchmark(num_frames=500)
    print("=" * 80)
    print("APEX-Vision 8-State EKF Temporal Smoother Stress Benchmark Results")
    print(f"Total Frames Tested: {bench_res.total_frames}")
    print(f"Mean Latency: {bench_res.mean_latency_ms:.3f} ms")
    print(f"Max Latency: {bench_res.max_latency_ms:.3f} ms")
    print(f"Outlier Rejections: {bench_res.outlier_rejections}")
    print(f"Mean Tracked Width: {bench_res.mean_lane_width_m:.4f} m (Target: 3.70m)")
    print(f"Lane Width Std Dev: {bench_res.std_lane_width_m:.4f} m")
    print(f"Dual Tracking Frames: {bench_res.dual_tracking_count}")
    print(f"Single Coast Frames: {bench_res.single_coast_count}")
    print(f"EKF Stability Validation Passed: {bench_res.passed_ekf_stability_test}")
    print("=" * 80)

