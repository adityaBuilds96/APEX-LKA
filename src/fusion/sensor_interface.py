"""
src/fusion/sensor_interface.py
==============================
Hardware Sensor Driver & CAN Wheel Encoder Interface Manager.
Provides non-blocking, thread-safe asynchronous ingestion of:
- 6-Axis IMU (MPU9250 / BNO055 via Serial / I2C)
- GNSS / RTK (NEO-M8N / ZED-F9P via NMEA / UBX)
- Dual Wheel Encoders via Arduino Node 2 CAN Frame 0x301

Includes vehicle dynamic bicycle model slip angle (beta) calculation.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class IMUReading:
    """6-Axis Inertial Measurement Unit sample."""
    ax: float = 0.0              # Body X acceleration (m/s^2)
    ay: float = 0.0              # Body Y acceleration (m/s^2)
    az: float = 9.80665          # Body Z acceleration (m/s^2)
    gx: float = 0.0              # Body X angular rate / roll rate (rad/s)
    gy: float = 0.0              # Body Y angular rate / pitch rate (rad/s)
    gz: float = 0.0              # Body Z angular rate / yaw rate (rad/s)
    temp_c: float = 25.0         # Sensor temperature (deg C)
    timestamp: float = 0.0

    @property
    def accel(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az], dtype=np.float64)

    @property
    def gyro(self) -> np.ndarray:
        return np.array([self.gx, self.gy, self.gz], dtype=np.float64)


@dataclass
class GNSSReading:
    """Global Navigation Satellite System position fix."""
    latitude: float = 0.0        # Degrees
    longitude: float = 0.0       # Degrees
    altitude: float = 0.0        # Altitude above mean sea level (meters)
    hdop: float = 1.0            # Horizontal Dilution of Precision
    fix_quality: int = 1         # 0=Invalid, 1=GPS SPS, 2=DGPS, 4=RTK Fixed, 5=RTK Float
    num_satellites: int = 12
    ground_speed_mps: float = 0.0
    track_heading_deg: float = 0.0
    timestamp: float = 0.0

    @property
    def is_rtk(self) -> bool:
        return self.fix_quality == 4


@dataclass
class WheelEncoderReading:
    """Dual rear wheel encoder linear velocity and tick counters."""
    v_left: float = 0.0          # Left rear wheel linear velocity (m/s)
    v_right: float = 0.0         # Right rear wheel linear velocity (m/s)
    ticks_left: int = 0          # Cumulative raw encoder ticks
    ticks_right: int = 0         # Cumulative raw encoder ticks
    timestamp: float = 0.0

    @property
    def average_speed(self) -> float:
        return (self.v_left + self.v_right) * 0.5


@dataclass
class VehicleKinematics:
    """Bicycle kinematic parameters for SAE BAJA autonomous chassis."""
    wheelbase_m: float = 1.60            # L = lf + lr
    dist_cg_to_front_m: float = 0.75     # lf
    dist_cg_to_rear_m: float = 0.85      # lr
    track_width_m: float = 1.25          # Tw
    understeer_gradient: float = 0.002   # Kus (rad/(m/s^2))
    steering_ratio: float = 12.5         # Steering wheel to road wheel ratio


class SensorInterfaceManager:
    """
    Non-blocking, thread-safe manager for hardware and simulated sensor streams.
    Fuses IMU, GNSS, CAN 0x301 wheel ticks, and computes dynamic vehicle slip angle.
    """

    def __init__(
        self,
        kinematics: Optional[VehicleKinematics] = None,
        sim_mode: bool = True,
    ) -> None:
        self.kinematics = kinematics or VehicleKinematics()
        self.sim_mode = sim_mode
        self._lock = threading.Lock()

        # Latest readings
        self._latest_imu = IMUReading(timestamp=time.time())
        self._latest_gnss = GNSSReading(timestamp=time.time())
        self._latest_wheels = WheelEncoderReading(timestamp=time.time())
        self._current_steering_deg: float = 0.0

        # Background stream thread control
        self._running = False
        self._sim_thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start asynchronous background ingestion threads."""
        self._running = True
        if self.sim_mode:
            self._sim_thread = threading.Thread(
                target=self._sim_loop, name="SensorSimWorker", daemon=True
            )
            self._sim_thread.start()

    def stop(self) -> None:
        """Clean shutdown of sensor threads."""
        self._running = False
        if self._sim_thread and self._sim_thread.is_alive():
            self._sim_thread.join(timeout=0.5)

    def set_steering_angle(self, steering_deg: float) -> None:
        """Update current measured steering angle (road wheel degrees)."""
        with self._lock:
            self._current_steering_deg = float(steering_deg)

    def update_imu(self, ax: float, ay: float, az: float, gx: float, gy: float, gz: float) -> None:
        """Thread-safe update for external or hardware IMU drivers."""
        with self._lock:
            self._latest_imu = IMUReading(
                ax=ax, ay=ay, az=az, gx=gx, gy=gy, gz=gz, timestamp=time.time()
            )

    def update_gnss(
        self,
        lat: float,
        lon: float,
        alt: float,
        fix_quality: int = 1,
        hdop: float = 1.0,
    ) -> None:
        """Thread-safe update for GNSS NMEA/UBX parser."""
        with self._lock:
            self._latest_gnss = GNSSReading(
                latitude=lat,
                longitude=lon,
                altitude=alt,
                fix_quality=fix_quality,
                hdop=hdop,
                timestamp=time.time(),
            )

    def update_wheel_encoders(self, v_left: float, v_right: float) -> None:
        """Thread-safe update from CAN ID 0x301 feedback."""
        with self._lock:
            self._latest_wheels = WheelEncoderReading(
                v_left=v_left, v_right=v_right, timestamp=time.time()
            )

    def get_latest_imu(self) -> IMUReading:
        with self._lock:
            return self._latest_imu

    def get_latest_gnss(self) -> GNSSReading:
        with self._lock:
            return self._latest_gnss

    def get_latest_wheel_encoders(self) -> WheelEncoderReading:
        with self._lock:
            return self._latest_wheels

    # ═══════════════════════════════════════════════════════════════════════
    # Dynamic Vehicle Slip Angle (Beta) Calculation
    # ═══════════════════════════════════════════════════════════════════════

    def compute_slip_angle_deg(
        self,
        speed_mps: Optional[float] = None,
        steering_deg: Optional[float] = None,
        yaw_rate_radps: Optional[float] = None,
    ) -> float:
        """
        Computes dynamic vehicle slip angle beta (degrees) at the center of gravity:
        Using kinematic bicycle model with understeer gradient correction:
            beta = arctan(v_y / v_x) approx (lr / L) * delta_steer / (1 + Kus * vx^2)
        """
        with self._lock:
            v_x = speed_mps if speed_mps is not None else self._latest_wheels.average_speed
            steer = steering_deg if steering_deg is not None else self._current_steering_deg
            wz = yaw_rate_radps if yaw_rate_radps is not None else self._latest_imu.gz

        if v_x < 0.30:
            return 0.0

        # Physical constants
        L = self.kinematics.wheelbase_m
        lr = self.kinematics.dist_cg_to_rear_m
        Kus = self.kinematics.understeer_gradient

        # Steering angle in radians
        delta_rad = math.radians(steer)

        # Kinematic slip angle at CG
        denominator = 1.0 + Kus * (v_x ** 2)
        beta_rad = (lr / L) * (delta_rad / max(0.1, denominator))

        # Dynamic cross-check using lateral acceleration / yaw rate if available
        # a_y = v_x * (yaw_rate + d_beta/dt). In steady-state: beta ~ lr * wz / v_x
        if abs(wz) > 0.05:
            beta_dyn_rad = (lr * wz) / max(0.5, v_x)
            # Blend kinematic and dynamic estimates (70% kinematic, 30% dynamic)
            beta_rad = 0.70 * beta_rad + 0.30 * beta_dyn_rad

        beta_deg = math.degrees(beta_rad)
        return float(np.clip(beta_deg, -25.0, 25.0))

    # ═══════════════════════════════════════════════════════════════════════
    # Simulation Generator Loop (For PC CI and Offline Replay)
    # ═══════════════════════════════════════════════════════════════════════

    def _sim_loop(self) -> None:
        """Simulates realistic 100 Hz IMU, 50 Hz Wheel, and 10 Hz GNSS streams."""
        t_prev = time.time()
        sim_time = 0.0

        while self._running:
            time.sleep(0.01)  # 100 Hz step
            now = time.time()
            dt = now - t_prev
            t_prev = now
            sim_time += dt

            # Simulated smooth vehicle driving dynamics (5.0 m/s with gentle curve)
            v_sim = 5.0 + 0.5 * math.sin(sim_time * 0.2)
            steer_sim = 3.0 * math.sin(sim_time * 0.1)  # degrees
            wz_sim = math.radians(steer_sim) * (v_sim / self.kinematics.wheelbase_m)
            ay_sim = v_sim * wz_sim

            with self._lock:
                self._current_steering_deg = steer_sim
                # 100 Hz IMU
                self._latest_imu = IMUReading(
                    ax=float(0.1 * math.cos(sim_time)),
                    ay=float(ay_sim),
                    az=9.80665,
                    gx=0.0,
                    gy=0.0,
                    gz=float(wz_sim),
                    timestamp=now,
                )

                # 50 Hz Wheels
                diff_v = wz_sim * (self.kinematics.track_width_m * 0.5)
                self._latest_wheels = WheelEncoderReading(
                    v_left=float(v_sim - diff_v),
                    v_right=float(v_sim + diff_v),
                    timestamp=now,
                )

                # 10 Hz GNSS
                if int(sim_time * 100) % 10 == 0:
                    self._latest_gnss = GNSSReading(
                        latitude=37.7749 + (v_sim * sim_time) / 111111.0,
                        longitude=-122.4194,
                        altitude=15.0,
                        fix_quality=4,  # RTK Fixed
                        hdop=0.8,
                        timestamp=now,
                    )
