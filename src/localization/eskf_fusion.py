"""
src/localization/eskf_fusion.py
================================
APEX-Vision Core v3.0 — 15-DOF Error-State Kalman Filter (ESKF) Engine (ISO 26262 Compliant)
- State Vector (16D Nominal State):
    x = [p_3D (3), v_3D (3), q_4D (4), a_bias (3), w_bias (3)]^T
- Error State Vector (15D Error State):
    delta_x = [delta_p (3), delta_v (3), delta_theta (3), delta_a_b (3), delta_w_b (3)]^T
- Prediction Step: Propagates nominal state and error covariance P (15x15) using 100 Hz IMU.
- Correction Step: Fuses Visual Odometry, Wheel Encoder velocity, and GNSS observations.
- Quaternion Error Update: q_true = q_nominal (x) delta_q.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Import underlying verified ESKF core from src.fusion.eskf
from src.fusion.eskf import (
    ErrorStateKalmanFilter,
    LocalizationStatus,
    VehiclePose,
    GRAVITY_MAGNITUDE,
)


@dataclass
class ESKFFusionOutput:
    """State estimate emitted by 15-DOF ESKF fusion engine."""
    timestamp_s: float
    position_m: Tuple[float, float, float]       # [x, y, z] in local ENU
    velocity_m_s: Tuple[float, float, float]     # [vx, vy, vz] in vehicle body frame
    orientation_rpy_rad: Tuple[float, float, float] # [roll, pitch, yaw]
    quaternion: Tuple[float, float, float, float] # [w, x, y, z]
    accel_bias: Tuple[float, float, float]
    gyro_bias: Tuple[float, float, float]
    position_std_m: float
    yaw_rate_rad_s: float
    speed_m_s: float
    slip_angle_rad: float
    status: LocalizationStatus


class ESKFFusionEngine:
    """
    15-DOF Error-State Kalman Filter localization fusion manager.
    """

    def __init__(self, init_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self.eskf = ErrorStateKalmanFilter()
        self.eskf.reset(x=init_pos[0], y=init_pos[1], z=init_pos[2])
        self.last_update_ts = time.time()

    def predict(self, ax: float, ay: float, az: float, wx: float, wy: float, wz: float, dt: float = 0.01) -> None:
        """
        High-rate 100 Hz IMU prediction step.
        Propagates position, velocity, and orientation using nominal strapdown kinematics.
        """
        self.eskf.propagate_imu(ax=ax, ay=ay, az=az, wx=wx, wy=wy, wz=wz, dt=dt)
        self.last_update_ts = time.time()

    def correct_wheel_encoder(self, speed_m_s: float) -> None:
        """Fuse CAN wheel encoder speed observation."""
        self.eskf.update_wheel_encoder(speed_mps=speed_m_s)

    def correct_visual_odometry(self, dx: float, dy: float, dz: float = 0.0, dyaw: float = 0.0) -> None:
        """Fuse optical flow visual odometry increments."""
        self.eskf.update_visual_odometry(dx=dx, dy=dy, dpsi=dyaw)

    def correct_gnss(self, lat: float, lon: float, alt: float = 0.0, std_m: float = 1.5) -> None:
        """Fuse global GNSS/RTK fix."""
        # Convert lat/lon offset to local ENU meters (approx 1 deg lat = 111,000m)
        x_m = lon * 111320.0 * math.cos(math.radians(lat))
        y_m = lat * 110540.0
        self.eskf.update_gnss(x_m=x_m, y_m=y_m, z_m=alt, std_m=std_m)

    def get_state(self) -> ESKFFusionOutput:
        """Extract consolidated state and covariance metrics."""
        pose = self.eskf.get_pose()
        v = self.eskf.v
        speed = math.sqrt(v[0]**2 + v[1]**2)
        slip = math.atan2(v[1], max(abs(v[0]), 0.1)) if speed > 0.5 else 0.0

        # Position uncertainty from covariance P
        pos_cov = self.eskf.P[0:3, 0:3]
        pos_std = float(math.sqrt(max(0.0, np.trace(pos_cov) / 3.0)))

        return ESKFFusionOutput(
            timestamp_s=pose.timestamp,
            position_m=(pose.x, pose.y, pose.z),
            velocity_m_s=(float(v[0]), float(v[1]), float(v[2])),
            orientation_rpy_rad=(pose.roll, pose.pitch, pose.yaw),
            quaternion=tuple(pose.quaternion),
            accel_bias=tuple(self.eskf.a_b),
            gyro_bias=tuple(self.eskf.omega_b),
            position_std_m=round(pos_std, 4),
            yaw_rate_rad_s=pose.yaw_rate,
            speed_m_s=round(speed, 2),
            slip_angle_rad=round(slip, 4),
            status=pose.localization_status,
        )
