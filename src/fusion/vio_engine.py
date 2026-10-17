"""
src/fusion/vio_engine.py
========================
Visual-Inertial Odometry (VIO) & GNSS-Denied Dead Reckoning Engine.
Estimates frame-to-frame ego-motion [delta_x, delta_y, delta_yaw] using Pyramidal
Lucas-Kanade optical flow, Shi-Tomasi feature tracking, and IMU metric scale integration.

Latency Budget: <= 5.0 ms per frame on CPU / Jetson Orin Nano.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class EgoMotionMeasurement:
    """Inter-frame visual-inertial ego-motion displacement and rotation."""
    dx: float = 0.0              # Longitudinal displacement in forward vehicle axis (meters)
    dy: float = 0.0              # Lateral displacement in left vehicle axis (meters)
    dz: float = 0.0              # Vertical displacement in vehicle Z axis (meters)
    droll: float = 0.0           # Relative roll increment (radians)
    dpitch: float = 0.0          # Relative pitch increment (radians)
    dyaw: float = 0.0            # Relative heading increment (radians)
    inlier_ratio: float = 1.0    # Ratio of RANSAC inliers to total tracked features [0.0, 1.0]
    num_tracked: int = 0         # Number of active tracked corner features
    valid: bool = False          # True if visual tracking converged with sufficient inliers
    processing_ms: float = 0.0   # Time elapsed in VIO processing (ms)
    timestamp: float = 0.0

    @property
    def dx_m(self) -> float:
        return self.dx

    @property
    def dy_m(self) -> float:
        return self.dy

    @property
    def dyaw_rad(self) -> float:
        return self.dyaw


@dataclass
class VIOConfig:
    """Configuration parameters for Lucas-Kanade feature tracker and RANSAC."""
    max_features: int = 200
    min_feature_inliers: int = 20
    quality_level: float = 0.015
    min_distance_px: float = 12.0
    win_size: Tuple[int, int] = (21, 21)
    max_pyramid_level: int = 2
    ransac_threshold_px: float = 2.0
    focal_length_px: float = 500.0
    principal_point: Tuple[float, float] = (320.0, 180.0)


class VisualInertialOdometryEngine:
    """
    High-Speed Monocular Visual-Inertial Odometry Engine.
    Tracks ground plane corner features and computes calibrated vehicle ego-motion.
    """

    def __init__(self, config: Optional[VIOConfig] = None) -> None:
        self.config = config or VIOConfig()
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pts: Optional[np.ndarray] = None
        self.last_timestamp: float = time.time()

        # Camera intrinsic matrix K
        fx = self.config.focal_length_px
        fy = self.config.focal_length_px
        cx, cy = self.config.principal_point
        self.K = np.array([
            [fx,  0.0, cx],
            [0.0, fy,  cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        # LK Optical Flow criteria
        self.lk_params = dict(
            winSize=self.config.win_size,
            maxLevel=self.config.max_pyramid_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )

        # Feature detector parameters
        self.feature_params = dict(
            maxCorners=self.config.max_features,
            qualityLevel=self.config.quality_level,
            minDistance=self.config.min_distance_px,
            blockSize=7
        )

        # Cumulative path history for dashboard HUD trajectory trail
        self.trajectory_history: List[Tuple[float, float]] = [(0.0, 0.0)]
        self.accum_x: float = 0.0
        self.accum_y: float = 0.0
        self.accum_yaw: float = 0.0

    def reset(self) -> None:
        """Clear previous frame state and re-initialize tracker."""
        self.prev_gray = None
        self.prev_pts = None
        self.trajectory_history = [(0.0, 0.0)]
        self.accum_x = 0.0
        self.accum_y = 0.0
        self.accum_yaw = 0.0

    def process_frame(
        self,
        frame_bgr_or_gray: np.ndarray,
        speed_mps: Optional[float] = None,
        imu_yaw_rate: Optional[float] = None,
        dt: Optional[float] = None,
    ) -> EgoMotionMeasurement:
        """
        Process incoming camera frame and compute relative vehicle motion.
        Executes in <= 5 ms.
        """
        t_start = time.perf_counter()
        now = time.time()
        if dt is None:
            dt = max(1e-3, min(now - self.last_timestamp, 0.20))
        self.last_timestamp = now

        # Convert to grayscale if necessary
        if frame_bgr_or_gray.ndim == 3:
            gray = cv2.cvtColor(frame_bgr_or_gray, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr_or_gray

        h, w = gray.shape[:2]

        # First frame or re-initialization
        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.config.min_feature_inliers:
            # Detect corner features primarily in the lower 65% road/ground region
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(0.35 * h):, :] = 255
            pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray
            self.prev_pts = pts
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return EgoMotionMeasurement(
                valid=False,
                num_tracked=len(pts) if pts is not None else 0,
                processing_ms=t_ms,
                timestamp=now,
            )

        # ── 1. Pyramidal Lucas-Kanade Optical Flow ─────────────────────────
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        if curr_pts is None or status is None:
            self.prev_gray = gray
            self.prev_pts = None
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return EgoMotionMeasurement(valid=False, processing_ms=t_ms, timestamp=now)

        # Filter points that were successfully tracked
        good_mask = (status.flatten() == 1)
        p0 = self.prev_pts[good_mask]
        p1 = curr_pts[good_mask]

        if len(p0) < self.config.min_feature_inliers:
            # Feature starvation -> re-detect
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(0.35 * h):, :] = 255
            self.prev_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            self.prev_gray = gray
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return EgoMotionMeasurement(valid=False, num_tracked=len(p0), processing_ms=t_ms, timestamp=now)

        # ── 2. Estimate 2D Rigid / Affine Ground Motion via RANSAC ──────────
        # Computes rotation angle d_psi and 2D pixel shift [dx_px, dy_px]
        M, inlier_mask = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC, ransacReprojThreshold=self.config.ransac_threshold_px
        )

        if M is None or inlier_mask is None:
            self.prev_gray = gray
            self.prev_pts = p1.reshape(-1, 1, 2)
            t_ms = (time.perf_counter() - t_start) * 1000.0
            return EgoMotionMeasurement(valid=False, num_tracked=len(p0), processing_ms=t_ms, timestamp=now)

        inlier_count = int(np.sum(inlier_mask))
        inlier_ratio = float(inlier_count / max(1, len(p0)))

        # Extract rotation and translation from Affine matrix [s*R | T]
        # M = [[cos*s, -sin*s, tx], [sin*s, cos*s, ty]]
        cos_s = M[0, 0]
        sin_s = M[1, 0]
        tx_px = M[0, 2]
        ty_px = M[1, 2]

        d_yaw = math.atan2(-sin_s, cos_s)  # Image yaw rotation

        # Use IMU gyro yaw rate if provided to enhance visual yaw measurement
        if imu_yaw_rate is not None and abs(imu_yaw_rate) > 1e-4:
            # Complementary filter: 70% IMU gyro integration, 30% visual flow
            d_yaw_imu = float(imu_yaw_rate * dt)
            d_yaw = 0.70 * d_yaw_imu + 0.30 * d_yaw

        # ── 3. Metric Scale Resolution ─────────────────────────────────────
        # In a forward-facing camera, forward motion causes downward/outward flow (ty_px > 0).
        # We scale the longitudinal distance using either vehicle speed or normalized focal length.
        if speed_mps is not None and speed_mps > 0.05:
            delta_forward_m = float(speed_mps * dt)
        else:
            # Metric conversion based on average optical expansion / pitch shift
            # ty_px ~ f * dZ / Z. With typical road plane distance Z ~ 8m, f ~ 500px:
            # dZ ~ ty_px * (8m / 500px) ~ ty_px * 0.016m
            scale_px_to_m = 0.018
            delta_forward_m = float(max(0.0, ty_px * scale_px_to_m))

        # Lateral displacement in vehicle frame (dx_px / f * Z)
        delta_lateral_m = float(-tx_px * 0.012)

        # ── 4. Accumulate Trajectory History for HUD Trail ─────────────────
        self.accum_yaw += d_yaw
        cos_y = math.cos(self.accum_yaw)
        sin_y = math.sin(self.accum_yaw)

        # Move in local world coordinates
        self.accum_x += delta_forward_m * cos_y - delta_lateral_m * sin_y
        self.accum_y += delta_forward_m * sin_y + delta_lateral_m * cos_y

        # Maintain last 60 trajectory waypoints (2.0s trail at 30 Hz)
        self.trajectory_history.append((float(self.accum_x), float(self.accum_y)))
        if len(self.trajectory_history) > 60:
            self.trajectory_history.pop(0)

        # ── 5. Replenish Features If Inliers Drop Below Safe Threshold ──────
        if inlier_count < int(self.config.max_features * 0.5):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[int(0.35 * h):, :] = 255
            new_pts = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
            if new_pts is not None:
                self.prev_pts = new_pts
            else:
                self.prev_pts = p1.reshape(-1, 1, 2)
        else:
            self.prev_pts = p1.reshape(-1, 1, 2)

        self.prev_gray = gray
        t_ms = (time.perf_counter() - t_start) * 1000.0

        return EgoMotionMeasurement(
            dx=delta_forward_m,
            dy=delta_lateral_m,
            dz=0.0,
            droll=0.0,
            dpitch=0.0,
            dyaw=d_yaw,
            inlier_ratio=inlier_ratio,
            num_tracked=inlier_count,
            valid=bool(inlier_count >= self.config.min_feature_inliers),
            processing_ms=t_ms,
            timestamp=now,
        )

    def get_trajectory_trail(self) -> List[Tuple[float, float]]:
        """Return 2D local coordinate waypoints of recent vehicle trajectory trail."""
        return list(self.trajectory_history)
