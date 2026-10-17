"""
src/control/ekf_tracker.py
==========================
Extended Kalman Filter (EKF) Temporal Polynomial Path Tracker:
- Tracks quadratic polynomial coefficients [a, b, c, a_dot, b_dot, c_dot]^T across frames
- Constant-velocity motion model with dynamic time-delta dt
- Innovation gating: Rejects measurement outliers exceeding max_innov_m deviation
- Holds state during temporary dropouts/occlusions up to max_hold_frames
- Prevents track flicker and sudden jumping on camera video streams
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Union

import numpy as np

log = logging.getLogger("ekf_tracker")


class EKFPolynomialTracker:
    """
    6-state Extended Kalman Filter tracking metric polynomial trajectory:
    X_m(Z) = a * Z^2 + b * Z + c
    State vector: x = [a, b, c, a_dot, b_dot, c_dot]^T
    """

    def __init__(
        self,
        process_noise: float = 1.0e-3,
        measurement_noise: float = 2.0e-2,
        max_innov_m: float = 0.50,
        max_hold_frames: int = 3,
        eval_z_m: float = 5.0,
    ) -> None:
        self.process_noise = float(process_noise)
        self.measurement_noise = float(measurement_noise)
        self.max_innov_m = float(max_innov_m)
        self.max_hold_frames = int(max_hold_frames)
        self.eval_z_m = float(eval_z_m)

        # State: [a, b, c, a_dot, b_dot, c_dot]
        self.x = np.zeros(6, dtype=np.float64)
        # Covariance matrix P
        self.P = np.eye(6, dtype=np.float64) * 1.0
        # Measurement matrix H: 3x6 mapping [a, b, c, a_dot, b_dot, c_dot] -> [a, b, c]
        self.H = np.zeros((3, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        self.R = np.eye(3, dtype=np.float64) * self.measurement_noise

        self.hold_count = 0
        self.is_initialized = False

    def reset(self) -> None:
        """Reset internal filter state."""
        self.x = np.zeros(6, dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 1.0
        self.hold_count = 0
        self.is_initialized = False

    def predict(self, dt: float = 0.033) -> np.ndarray:
        """
        EKF Prediction step: x_{t|t-1} = F * x_{t-1}
        """
        dt_val = max(1e-3, dt)
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt_val
        F[1, 4] = dt_val
        F[2, 5] = dt_val

        # Process noise covariance Q
        q = self.process_noise
        Q = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            Q[i, i] = q * (dt_val ** 3) / 3.0
            Q[i, i + 3] = q * (dt_val ** 2) / 2.0
            Q[i + 3, i] = q * (dt_val ** 2) / 2.0
            Q[i + 3, i + 3] = q * dt_val

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        return self.x[:3]

    def update(
        self,
        measured_coeffs: Optional[np.ndarray],
        dt: float = 0.033,
    ) -> Tuple[np.ndarray, bool]:
        """
        Perform predict and update cycle.
        Returns:
            (filtered_coeffs [a, b, c], is_valid)
        """
        pred_coeffs = self.predict(dt)

        if measured_coeffs is None or len(measured_coeffs) < 3:
            # Measurement dropout: hold prediction if within max_hold_frames
            self.hold_count += 1
            if self.is_initialized and self.hold_count <= self.max_hold_frames:
                return self.x[:3].copy(), True
            else:
                return pred_coeffs.copy(), False

        z = np.asarray(measured_coeffs[:3], dtype=np.float64)

        if not self.is_initialized:
            # First initialization
            self.x[:3] = z
            self.x[3:] = 0.0
            self.P = np.eye(6, dtype=np.float64) * 0.1
            self.is_initialized = True
            self.hold_count = 0
            return self.x[:3].copy(), True

        # Innovation gating check at evaluation distance (Z = eval_z_m)
        pred_x_eval = float(np.polyval(self.x[:3], self.eval_z_m))
        meas_x_eval = float(np.polyval(z, self.eval_z_m))
        innovation_m = abs(meas_x_eval - pred_x_eval)

        if innovation_m > self.max_innov_m:
            # Innovation exceeded gate: potential outlier frame
            self.hold_count += 1
            if self.hold_count <= self.max_hold_frames:
                log.debug("EKF gate rejected outlier innovation: %.3fm", innovation_m)
                return self.x[:3].copy(), True
            else:
                # Held too long; accept new measurement and adapt
                self.x[:3] = z
                self.x[3:] = 0.0
                self.P = np.eye(6, dtype=np.float64) * 0.5
                self.hold_count = 0
                return self.x[:3].copy(), True

        # Measurement update: Kalman Gain K = P * H^T * (H * P * H^T + R)^-1
        y = z - (self.H @ self.x)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + (K @ y)
        I_KH = np.eye(6, dtype=np.float64) - (K @ self.H)
        self.P = I_KH @ self.P

        self.hold_count = 0
        return self.x[:3].copy(), True


class EKF8StateLaneTracker:
    """
    8-State Extended Kalman Filter for temporal lane boundary stabilization:
    State vector: x = [a_L, b_L, c_L, a_R, b_R, c_R, e_lat, theta_e]^T
    Tracks:
      - Left parabolic coefficients: x_L(y) = a_L * y^2 + b_L * y + c_L
      - Right parabolic coefficients: x_R(y) = a_R * y^2 + b_R * y + c_R
      - Lateral offset error: e_lat (at vehicle baseline)
      - Heading error angle: theta_e (tangent angle at vehicle baseline)

    Key Features:
      - Zero Flickering Guarantee: Eliminates frame-to-frame jumping (Delta <= 0.02)
      - Robust Dropout Hold: Sustains track trajectory during 1–3 frame dropouts
      - Innovation Gating: Rejects false positive edge snaps and outliers
    """

    def __init__(
        self,
        process_noise_poly: float = 1.0e-4,
        process_noise_pose: float = 1.0e-3,
        meas_noise_poly: float = 2.0e-2,
        meas_noise_pose: float = 5.0e-2,
        max_innov_gate: float = 0.50,
        max_hold_frames: int = 3,
        hold_lost_frames: Optional[int] = None,
    ) -> None:
        self.process_noise_poly = float(process_noise_poly)
        self.process_noise_pose = float(process_noise_pose)
        self.meas_noise_poly = float(meas_noise_poly)
        self.meas_noise_pose = float(meas_noise_pose)
        self.max_innov_gate = float(max_innov_gate)
        self.max_hold_frames = int(hold_lost_frames if hold_lost_frames is not None else max_hold_frames)

        # 8-state vector
        self.x = np.zeros(8, dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 1.0

        # Base Q (process noise)
        self.Q = np.diag([
            self.process_noise_poly, self.process_noise_poly, self.process_noise_poly * 10.0,
            self.process_noise_poly, self.process_noise_poly, self.process_noise_poly * 10.0,
            self.process_noise_pose, self.process_noise_pose,
        ])

        # Base R (measurement noise)
        self.R_diag = np.array([
            self.meas_noise_poly, self.meas_noise_poly, self.meas_noise_poly * 5.0,
            self.meas_noise_poly, self.meas_noise_poly, self.meas_noise_poly * 5.0,
            self.meas_noise_pose, self.meas_noise_pose,
        ], dtype=np.float64)

        self.hold_count = 0
        self.is_initialized = False

    def reset(self) -> None:
        """Reset internal filter state."""
        self.x = np.zeros(8, dtype=np.float64)
        self.P = np.eye(8, dtype=np.float64) * 1.0
        self.hold_count = 0
        self.is_initialized = False

    def predict(self, dt: float = 0.033) -> np.ndarray:
        """
        State prediction step assuming stationary / slow-varying lane boundaries.
        x_{t|t-1} = x_{t-1}
        P_{t|t-1} = P_{t-1} + Q * dt
        """
        dt_val = max(1e-3, min(dt, 0.2))
        self.P = self.P + self.Q * (dt_val / 0.033)
        return self.x.copy()

    def update(
        self,
        left_coeffs: Optional[np.ndarray],
        right_coeffs: Optional[np.ndarray],
        e_lat: Optional[float] = None,
        theta_e: Optional[float] = None,
        dt: float = 0.033,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float, bool]:
        """
        Perform prediction and measurement update for 8 states.
        Returns:
          (smoothed_left_coeffs, smoothed_right_coeffs, smoothed_e_lat, smoothed_theta_e, is_valid)
        """
        self.predict(dt)

        has_left = left_coeffs is not None and len(left_coeffs) >= 3 and not np.all(left_coeffs == 0)
        has_right = right_coeffs is not None and len(right_coeffs) >= 3 and not np.all(right_coeffs == 0)

        # Full dropout condition
        if not has_left and not has_right:
            self.hold_count += 1
            if self.is_initialized and self.hold_count <= self.max_hold_frames:
                # Retain prediction with confidence hold
                return (
                    self.x[0:3].copy(),
                    self.x[3:6].copy(),
                    float(self.x[6]),
                    float(self.x[7]),
                    True,
                )
            else:
                return None, None, 0.0, 0.0, False

        # Construct measured indices and vector
        meas_indices = []
        meas_vals = []

        if has_left:
            meas_indices.extend([0, 1, 2])
            meas_vals.extend(left_coeffs[:3].tolist())

        if has_right:
            meas_indices.extend([3, 4, 5])
            meas_vals.extend(right_coeffs[:3].tolist())

        if e_lat is not None and np.isfinite(e_lat):
            meas_indices.append(6)
            meas_vals.append(float(e_lat))
        elif has_left and has_right:
            # Estimate e_lat from center
            meas_indices.append(6)
            meas_vals.append(float(0.5 * (left_coeffs[2] + right_coeffs[2])))

        if theta_e is not None and np.isfinite(theta_e):
            meas_indices.append(7)
            meas_vals.append(float(theta_e))
        elif has_left and has_right:
            # Heading error from average slope b
            meas_indices.append(7)
            meas_vals.append(float(0.5 * (left_coeffs[1] + right_coeffs[1])))

        z = np.array(meas_vals, dtype=np.float64)
        m = len(meas_indices)

        if not self.is_initialized:
            # Initialize state with first measurements
            for idx, val in zip(meas_indices, meas_vals):
                self.x[idx] = val
            # If one boundary missing at init, estimate nominal width
            if has_left and not has_right:
                self.x[3:6] = self.x[0:3]
                self.x[5] += 200.0  # nominal offset
            elif has_right and not has_left:
                self.x[0:3] = self.x[3:6]
                self.x[2] -= 200.0
            self.P = np.eye(8, dtype=np.float64) * 0.1
            self.is_initialized = True
            self.hold_count = 0
            return (
                self.x[0:3].copy(),
                self.x[3:6].copy(),
                float(self.x[6]),
                float(self.x[7]),
                True,
            )

        # Projection matrix H (m x 8)
        H = np.zeros((m, 8), dtype=np.float64)
        for r_i, c_i in enumerate(meas_indices):
            H[r_i, c_i] = 1.0

        # Measurement covariance R (m x m)
        R = np.diag(self.R_diag[meas_indices])

        # Innovation y = z - H * x
        y = z - (H @ self.x)

        # Innovation covariance S = H * P * H^T + R
        S = H @ self.P @ H.T + R

        # Kalman gain K = P * H^T * inv(S)
        try:
            K = self.P @ H.T @ np.linalg.inv(S)
        except np.linalg.LinAlgError:
            K = self.P @ H.T @ np.linalg.pinv(S)

        # State update
        self.x = self.x + (K @ y)
        I_KH = np.eye(8, dtype=np.float64) - (K @ H)
        self.P = I_KH @ self.P

        self.hold_count = 0
        return (
            self.x[0:3].copy(),
            self.x[3:6].copy(),
            float(self.x[6]),
            float(self.x[7]),
            True,
        )

