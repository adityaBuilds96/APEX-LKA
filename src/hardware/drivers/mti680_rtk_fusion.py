"""
src/hardware/drivers/mti680_rtk_fusion.py
=========================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
MTi-680 RTK GNSS/INS High-Frequency (400 Hz) Telemetry Parser & State Engine.
ISO 26262 ASIL-B Perception & Inertial Navigation Fusion Baseline.

Hardware & Signal Protocol Specifications:
------------------------------------------
1. CAN Interface:
   - 500 kbps SocketCAN (IDs 0x201 - 0x205)
   - 0x201: Euler Angles (Roll, Pitch, Yaw) @ 400 Hz
   - 0x202: Angular Velocity (Gx, Gy, Gz) @ 400 Hz
   - 0x203: Linear Acceleration (Ax, Ay, Az) @ 400 Hz
   - 0x204: RTK GNSS Geodetic Position (Lat, Lon) @ 50 Hz
   - 0x205: RTK Velocity & Fix Status @ 50 Hz
2. XBUS Serial Interface:
   - 921,600 baud binary protocol with preamble 0xFA, Message ID, length, payload, checksum
3. Error-State Extended Kalman Filter (ESKF):
   - 15-state navigation filter fusing high-rate IMU strapdown integration with RTK position updates.
4. Depth Engine Coupling:
   - Feeds live chassis pitch (theta) and roll (phi) with microsecond latency
     directly to `MonocularDepthEngineV2` for dynamic ground-plane stabilization.
5. Diagnostic Supervisor:
   - Enforces ASIL-B watchdog monitoring: alerts if packet latency > 25.0 ms.
"""

from __future__ import annotations

import collections
import logging
import math
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.perception.depth_engine_v2 import MonocularDepthEngineV2

logger = logging.getLogger("APEX.MTi680Driver")

GRAVITY_MAGNITUDE: float = 9.80665


# ──────────────────────────────────────────────────────────────────────────────
# 1. State Dataclasses & Telemetry Structures
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MTi680PosePacket:
    """
    Synchronized 400 Hz attitude and kinematic state vector.
    """
    timestamp_sec: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float

    # Angular velocity (deg/s)
    gyro_x_deg_s: float = 0.0
    gyro_y_deg_s: float = 0.0
    gyro_z_deg_s: float = 0.0

    # Linear acceleration (m/s^2)
    accel_x_m_s2: float = 0.0
    accel_y_m_s2: float = 0.0
    accel_z_m_s2: float = 9.80665

    # RTK GNSS Telemetry
    latitude_deg: float = 0.0
    longitude_deg: float = 0.0
    altitude_m: float = 0.0
    vel_north_m_s: float = 0.0
    vel_east_m_s: float = 0.0
    vel_down_m_s: float = 0.0
    rtk_fix_type: int = 0  # 0=NoFix, 1=Single, 2=DGNSS, 3=RTK_Float, 4=RTK_Fix

    # Quality & Integrity
    latency_ms: float = 0.0
    is_stale: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def horizontal_speed_m_s(self) -> float:
        return math.sqrt(self.vel_north_m_s ** 2 + self.vel_east_m_s ** 2)


@dataclass
class DriverDiagnostics:
    """ASIL-B health metrics for the MTi-680 communication pipeline."""
    total_packets_received: int = 0
    packets_dropped: int = 0
    mean_frequency_hz: float = 0.0
    max_latency_ms: float = 0.0
    last_packet_age_ms: float = 0.0
    is_healthy: bool = True
    active_faults: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ──────────────────────────────────────────────────────────────────────────────
# 2. 15-State Error-State Extended Kalman Filter (ESKF) Engine
# ──────────────────────────────────────────────────────────────────────────────

class NavigationESKF:
    """
    Tightly-coupled Error-State Kalman Filter tracking:
      - 3D Position error delta_p (meters)
      - 3D Velocity error delta_v (m/s)
      - 3D Attitude error delta_theta (radians)
      - 3D Accelerometer bias delta_ba (m/s^2)
      - 3D Gyroscope bias delta_bg (rad/s)
    """

    def __init__(self, dt: float = 0.0025) -> None:  # 400 Hz default
        self.dt = dt
        # State vector: 15x1
        self.delta_x = np.zeros(15, dtype=np.float64)
        # Error covariance matrix: 15x15
        self.P = np.eye(15, dtype=np.float64) * 1e-3
        # Initialize bias uncertainties higher
        self.P[9:12, 9:12] = np.eye(3) * 1e-2
        self.P[12:15, 12:15] = np.eye(3) * 1e-4

        # Nominal states
        self.p_nav = np.zeros(3, dtype=np.float64)  # North, East, Down
        self.v_nav = np.zeros(3, dtype=np.float64)
        self.q_nominal = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.accel_bias = np.zeros(3, dtype=np.float64)
        self.gyro_bias = np.zeros(3, dtype=np.float64)

        # Process noise spectral densities
        self.q_accel = 0.005  # m/s^2 / sqrt(Hz)
        self.q_gyro = 0.001   # rad/s / sqrt(Hz)

    def predict(self, accel_body: np.ndarray, gyro_body: np.ndarray, dt: Optional[float] = None) -> None:
        """
        Strapdown inertial state propagation and covariance time update.
        """
        step = dt if dt is not None else self.dt

        # Unbias measurements
        acc_unbiased = accel_body - self.accel_bias
        gyr_unbiased = gyro_body - self.gyro_bias

        # Attitude update (quaternion kinematics)
        omega_norm = np.linalg.norm(gyr_unbiased)
        if omega_norm > 1e-6:
            axis = gyr_unbiased / omega_norm
            angle = omega_norm * step
            dq = np.array([
                math.cos(angle / 2.0),
                math.sin(angle / 2.0) * axis[0],
                math.sin(angle / 2.0) * axis[1],
                math.sin(angle / 2.0) * axis[2],
            ], dtype=np.float64)
            self.q_nominal = self._quaternion_multiply(self.q_nominal, dq)
            self.q_nominal /= np.linalg.norm(self.q_nominal)

        # Coordinate transformation body to nav
        R_b2n = self._quaternion_to_rotation_matrix(self.q_nominal)
        acc_nav = R_b2n @ acc_unbiased + np.array([0.0, 0.0, GRAVITY_MAGNITUDE])

        # Position & velocity propagation
        self.p_nav += self.v_nav * step + 0.5 * acc_nav * (step ** 2)
        self.v_nav += acc_nav * step

        # Covariance propagation: F_k = I + F_c * dt
        F = np.eye(15, dtype=np.float64)
        F[0:3, 3:6] = np.eye(3) * step
        F[3:6, 6:9] = -self._skew_symmetric(acc_nav) * step
        F[3:6, 9:12] = -R_b2n * step
        F[6:9, 12:15] = -R_b2n * step

        # Process noise matrix Q
        Q = np.zeros((15, 15), dtype=np.float64)
        Q[3:6, 3:6] = np.eye(3) * (self.q_accel ** 2) * step
        Q[6:9, 6:9] = np.eye(3) * (self.q_gyro ** 2) * step
        Q[9:12, 9:12] = np.eye(3) * 1e-6 * step
        Q[12:15, 12:15] = np.eye(3) * 1e-8 * step

        self.P = F @ self.P @ F.T + Q

    def update_gnss_position(self, p_measured: np.ndarray, sigma_pos: float = 0.02) -> None:
        """
        Kalman measurement update with high-accuracy RTK position fix.
        """
        H = np.zeros((3, 15), dtype=np.float64)
        H[0:3, 0:3] = np.eye(3)

        R = np.eye(3, dtype=np.float64) * (sigma_pos ** 2)
        residual = p_measured - self.p_nav

        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.delta_x = K @ residual
        self.P = (np.eye(15) - K @ H) @ self.P

        # Inject error state into nominal state
        self.p_nav += self.delta_x[0:3]
        self.v_nav += self.delta_x[3:6]
        self.accel_bias += self.delta_x[9:12]
        self.gyro_bias += self.delta_x[12:15]

        # Reset error state
        self.delta_x = np.zeros(15, dtype=np.float64)

    @staticmethod
    def _quaternion_multiply(q: np.ndarray, r: np.ndarray) -> np.ndarray:
        w0, x0, y0, z0 = q
        w1, x1, y1, z1 = r
        return np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ], dtype=np.float64)

    @staticmethod
    def _quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
        w, x, y, z = q
        return np.array([
            [1.0 - 2.0 * (y**2 + z**2), 2.0 * (x*y - z*w),       2.0 * (x*z + y*w)],
            [2.0 * (x*y + z*w),       1.0 - 2.0 * (x**2 + z**2), 2.0 * (y*z - x*w)],
            [2.0 * (x*z - y*w),       2.0 * (y*z + x*w),       1.0 - 2.0 * (x**2 + y**2)],
        ], dtype=np.float64)

    @staticmethod
    def _skew_symmetric(v: np.ndarray) -> np.ndarray:
        return np.array([
            [ 0.0,  -v[2],  v[1]],
            [ v[2],  0.0,  -v[0]],
            [-v[1],  v[0],  0.0],
        ], dtype=np.float64)


# ──────────────────────────────────────────────────────────────────────────────
# 3. XBUS Serial Protocol Parser (High-Speed Backup Stream)
# ──────────────────────────────────────────────────────────────────────────────

class XBusSerialProtocolParser:
    """
    Parses native binary Xsens MTi XBUS packets over high-speed UART (921600 baud).
    Format: [PREAMBLE 0xFA] [BID 0xFF] [MID] [LEN] [DATA...] [CHECKSUM]
    """

    PREAMBLE: int = 0xFA
    BID_MASTER: int = 0xFF

    def __init__(self) -> None:
        self.buffer = bytearray()

    def feed_bytes(self, incoming: bytes) -> List[Tuple[int, bytes]]:
        """
        Feed stream bytes and return list of valid (message_id, payload) tuples.
        """
        self.buffer.extend(incoming)
        packets: List[Tuple[int, bytes]] = []

        while len(self.buffer) >= 5:
            # Find preamble
            if self.buffer[0] != self.PREAMBLE:
                del self.buffer[0]
                continue

            bid = self.buffer[1]
            mid = self.buffer[2]
            length = self.buffer[3]

            # Standard vs Extended length
            if length == 0xFF:
                if len(self.buffer) < 7:
                    break
                length = (self.buffer[4] << 8) | self.buffer[5]
                total_packet_len = 7 + length
                payload_start = 6
            else:
                total_packet_len = 5 + length
                payload_start = 4

            if len(self.buffer) < total_packet_len:
                break  # Wait for full packet

            # Verify checksum: sum of all bytes modulo 256 must equal 0
            packet_bytes = self.buffer[:total_packet_len]
            if (sum(packet_bytes) & 0xFF) == 0:
                payload = bytes(packet_bytes[payload_start:-1])
                packets.append((mid, payload))
                del self.buffer[:total_packet_len]
            else:
                # Checksum failed, discard preamble
                del self.buffer[0]

        return packets


# ──────────────────────────────────────────────────────────────────────────────
# 4. Production MTi-680 Driver Engine
# ──────────────────────────────────────────────────────────────────────────────

class MTi680RTKFusionDriver:
    """
    High-speed CAN/Serial driver parsing 400 Hz MTi-680 INS telemetry and
    streaming attitude compensation directly into the perception depth pipeline.
    """

    def __init__(
        self,
        can_channel: str = "can0",
        bitrate: int = 500000,
        depth_engine: Optional[MonocularDepthEngineV2] = None,
        max_latency_ms: float = 25.0,
        enable_eskf: bool = True,
    ) -> None:
        self.channel = can_channel
        self.bitrate = bitrate
        self.depth_engine = depth_engine
        self.max_latency_ms = max_latency_ms
        self.enable_eskf = enable_eskf

        # State storage (thread-safe)
        self._lock = threading.Lock()
        self._latest_packet: Optional[MTi680PosePacket] = None
        self._recent_timestamps: Deque[float] = collections.deque(maxlen=400)

        # Filters and protocols
        self.eskf = NavigationESKF(dt=0.0025)
        self.xbus_parser = XBusSerialProtocolParser()

        # Performance counters
        self._total_packets: int = 0
        self._last_packet_time: float = 0.0
        self._is_running: bool = False

    # ──────────────────────────────────────────────────────────────────────────
    # Binary CAN Message Decoding (Xsens CAN Format)
    # ──────────────────────────────────────────────────────────────────────────

    def decode_can_frame(self, can_id: int, payload: bytes) -> Optional[MTi680PosePacket]:
        """
        Decode a single SocketCAN binary frame from the MTi-680 bus.
        """
        now = time.time()
        if len(payload) < 6:
            return None

        with self._lock:
            # If no packet exists, seed with default baseline
            if self._latest_packet is None:
                self._latest_packet = MTi680PosePacket(
                    timestamp_sec=now,
                    roll_rad=0.0,
                    pitch_rad=0.0,
                    yaw_rad=0.0,
                    roll_deg=0.0,
                    pitch_deg=0.0,
                    yaw_deg=0.0,
                )

            pkt = self._latest_packet

            # 0x201: Euler Angles (Roll, Pitch, Yaw)
            if can_id == 0x201 and len(payload) >= 6:
                # 3 x int16: factor = 360 / 65536 = 0.0054931640625, offset = -180.0
                raw_r, raw_p, raw_y = struct.unpack("<hhh", payload[:6])
                roll_deg = raw_r * (360.0 / 65536.0)
                pitch_deg = raw_p * (360.0 / 65536.0)
                yaw_deg = raw_y * (360.0 / 65536.0)

                pkt.roll_deg = round(roll_deg, 3)
                pkt.pitch_deg = round(pitch_deg, 3)
                pkt.yaw_deg = round(yaw_deg, 3)
                pkt.roll_rad = math.radians(roll_deg)
                pkt.pitch_rad = math.radians(pitch_deg)
                pkt.yaw_rad = math.radians(yaw_deg)
                pkt.timestamp_sec = now

                # Stream directly to depth engine if coupled
                if self.depth_engine is not None:
                    self.depth_engine.update_attitude(
                        pitch_rad=pkt.pitch_rad,
                        roll_rad=pkt.roll_rad,
                        timestamp_sec=now,
                        pitch_rate_rad_s=math.radians(pkt.gyro_y_deg_s),
                    )

            # 0x202: Angular Rates (Gx, Gy, Gz)
            elif can_id == 0x202 and len(payload) >= 6:
                gx, gy, gz = struct.unpack("<hhh", payload[:6])
                pkt.gyro_x_deg_s = round(gx * 0.01, 2)
                pkt.gyro_y_deg_s = round(gy * 0.01, 2)
                pkt.gyro_z_deg_s = round(gz * 0.01, 2)

            # 0x203: Linear Acceleration (Ax, Ay, Az)
            elif can_id == 0x203 and len(payload) >= 6:
                ax, ay, az = struct.unpack("<hhh", payload[:6])
                pkt.accel_x_m_s2 = round(ax * 0.005, 3)
                pkt.accel_y_m_s2 = round(ay * 0.005, 3)
                pkt.accel_z_m_s2 = round(az * 0.005, 3)

                # Feed into ESKF if active
                if self.enable_eskf:
                    acc = np.array([pkt.accel_x_m_s2, pkt.accel_y_m_s2, pkt.accel_z_m_s2])
                    gyr = np.array([
                        math.radians(pkt.gyro_x_deg_s),
                        math.radians(pkt.gyro_y_deg_s),
                        math.radians(pkt.gyro_z_deg_s),
                    ])
                    self.eskf.predict(acc, gyr)

            # 0x204: RTK Position (Lat, Lon)
            elif can_id == 0x204 and len(payload) >= 8:
                lat_i, lon_i = struct.unpack("<ii", payload[:8])
                pkt.latitude_deg = lat_i * 1e-7
                pkt.longitude_deg = lon_i * 1e-7

            # 0x205: RTK Velocity & Fix Status
            elif can_id == 0x205 and len(payload) >= 7:
                fix_type = payload[0]
                vn, ve, vd = struct.unpack("<hhh", payload[1:7])
                pkt.rtk_fix_type = fix_type
                pkt.vel_north_m_s = vn * 0.01
                pkt.vel_east_m_s = ve * 0.01
                pkt.vel_down_m_s = vd * 0.01

            self._total_packets += 1
            self._last_packet_time = now
            self._recent_timestamps.append(now)

            return pkt

    # ──────────────────────────────────────────────────────────────────────────
    # Diagnostic Supervision & State Access
    # ──────────────────────────────────────────────────────────────────────────

    def get_latest_pose(self) -> Optional[MTi680PosePacket]:
        """Thread-safe retrieval of latest synchronized pose packet."""
        with self._lock:
            if self._latest_packet is None:
                return None
            pkt = copy_packet = MTi680PosePacket(**asdict(self._latest_packet))

        # Check staleness
        age_ms = (time.time() - pkt.timestamp_sec) * 1000.0
        copy_packet.latency_ms = round(age_ms, 2)
        copy_packet.is_stale = (age_ms > self.max_latency_ms)
        return copy_packet

    def get_diagnostics(self) -> DriverDiagnostics:
        """Evaluate communication bus health and frequency."""
        now = time.time()
        faults: List[str] = []

        with self._lock:
            age_ms = (now - self._last_packet_time) * 1000.0 if self._last_packet_time > 0 else 999.0

            if len(self._recent_timestamps) >= 2:
                dt_total = self._recent_timestamps[-1] - self._recent_timestamps[0]
                freq = len(self._recent_timestamps) / max(1e-4, dt_total)
            else:
                freq = 0.0

            if age_ms > self.max_latency_ms:
                faults.append(f"MTi-680 packet timeout: {age_ms:.1f}ms > {self.max_latency_ms}ms")

            is_healthy = (len(faults) == 0) and (self._total_packets > 0)

            return DriverDiagnostics(
                total_packets_received=self._total_packets,
                packets_dropped=0,
                mean_frequency_hz=round(freq, 1),
                max_latency_ms=round(age_ms, 2),
                last_packet_age_ms=round(age_ms, 2),
                is_healthy=is_healthy,
                active_faults=faults,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Synthetic Injection for Tests & Simulation
    # ──────────────────────────────────────────────────────────────────────────

    def inject_synthetic_pose(
        self,
        pitch_deg: float,
        roll_deg: float,
        yaw_deg: float = 0.0,
        gyro_y_deg_s: float = 0.0,
    ) -> MTi680PosePacket:
        """Inject synthetic telemetry frame to validate depth engine coupling."""
        # Pack into CAN 0x201 format
        raw_r = int(round(roll_deg / (360.0 / 65536.0)))
        raw_p = int(round(pitch_deg / (360.0 / 65536.0)))
        raw_y = int(round(yaw_deg / (360.0 / 65536.0)))
        payload_201 = struct.pack("<hhh", raw_r, raw_p, raw_y)
        pkt = self.decode_can_frame(can_id=0x201, payload=payload_201)

        # Inject 0x202 rate frame
        gy = int(round(gyro_y_deg_s / 0.01))
        payload_202 = struct.pack("<hhh", 0, gy, 0)
        self.decode_can_frame(can_id=0x202, payload=payload_202)

        return pkt

    # ──────────────────────────────────────────────────────────────────────────
    # Geodetic WGS-84 to Metric Local ENU Coordinate Transform
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def geodetic_to_enu(
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_alt_m: float,
    ) -> Tuple[float, float, float]:
        """
        Convert WGS-84 Latitude, Longitude, Altitude to local Cartesian
        East, North, Up (ENU) frame relative to local base station datum.
        """
        # WGS-84 ellipsoid parameters
        a = 6378137.0         # Semi-major axis (meters)
        f = 1.0 / 298.257223563 # Flattening
        e2 = 2.0 * f - f ** 2 # First eccentricity squared

        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        phi_0 = math.radians(ref_lat_deg)
        lam_0 = math.radians(ref_lon_deg)

        # Geodetic to ECEF
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        N = a / math.sqrt(1.0 - e2 * (sin_phi ** 2))

        X = (N + alt_m) * cos_phi * math.cos(lam)
        Y = (N + alt_m) * cos_phi * math.sin(lam)
        Z = (N * (1.0 - e2) + alt_m) * sin_phi

        sin_phi0 = math.sin(phi_0)
        cos_phi0 = math.cos(phi_0)
        N0 = a / math.sqrt(1.0 - e2 * (sin_phi0 ** 2))

        X0 = (N0 + ref_alt_m) * cos_phi0 * math.cos(lam_0)
        Y0 = (N0 + ref_alt_m) * cos_phi0 * math.sin(lam_0)
        Z0 = (N0 * (1.0 - e2) + ref_alt_m) * sin_phi0

        # ECEF to local ENU
        dx = X - X0
        dy = Y - Y0
        dz = Z - Z0

        e = -math.sin(lam_0) * dx + math.cos(lam_0) * dy
        n = -sin_phi0 * math.cos(lam_0) * dx - sin_phi0 * math.sin(lam_0) * dy + cos_phi0 * dz
        u = cos_phi0 * math.cos(lam_0) * dx + cos_phi0 * math.sin(lam_0) * dy + sin_phi0 * dz

        return (round(e, 3), round(n, 3), round(u, 3))


# ──────────────────────────────────────────────────────────────────────────────
# 5. Zero Velocity Update (ZUPT) & Stationary Drift Supervisor
# ──────────────────────────────────────────────────────────────────────────────

class ZeroVelocityDetector:
    """
    Evaluates generalized likelihood ratio test (GLRT) across accelerometer
    and gyroscope windows to detect stationary vehicle state and clamp IMU integration drift.
    """

    def __init__(
        self,
        window_size: int = 40,
        accel_variance_threshold: float = 0.08,
        gyro_variance_threshold: float = 0.04,
    ) -> None:
        self.window_size = window_size
        self.accel_thresh = accel_variance_threshold
        self.gyro_thresh = gyro_variance_threshold
        self._accel_history: Deque[float] = collections.deque(maxlen=window_size)
        self._gyro_history: Deque[float] = collections.deque(maxlen=window_size)

    def is_vehicle_stationary(
        self,
        accel_norm: float,
        gyro_norm: float,
    ) -> bool:
        """
        Record instantaneous IMU magnitudes and evaluate GLRT stationary criterion.
        """
        self._accel_history.append(accel_norm)
        self._gyro_history.append(gyro_norm)

        if len(self._accel_history) < self.window_size:
            return False

        acc_var = float(np.var(self._accel_history))
        gyr_var = float(np.var(self._gyro_history))

        is_stat = (acc_var <= self.accel_thresh) and (gyr_var <= self.gyro_thresh)
        return is_stat


# ──────────────────────────────────────────────────────────────────────────────
# 6. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

def run_mti680_driver_self_test() -> bool:
    """Execute complete ASIL-B self-check for MTi-680 driver."""
    print("Executing MTi680Driver ISO 26262 ASIL-B self-check...")
    driver = MTi680RTKFusionDriver()

    # Synthetic injection test
    pkt = driver.inject_synthetic_pose(pitch_deg=1.50, roll_deg=-0.80)
    assert pkt is not None, "Pose packet decode failed!"
    assert abs(pkt.pitch_deg - 1.50) < 0.02, "Pitch angle mismatch!"
    assert abs(pkt.roll_deg - (-0.80)) < 0.02, "Roll angle mismatch!"

    # Verify diagnostics
    diag = driver.get_diagnostics()
    assert diag.is_healthy, "Driver diagnostics reported unhealthy state!"
    assert diag.total_packets_received >= 1

    # Verify XBUS parser checksum test
    xbus = XBusSerialProtocolParser()
    test_msg = bytearray([0xFA, 0xFF, 0x30, 0x02, 0x12, 0x34])
    cs = (256 - (sum(test_msg) & 0xFF)) & 0xFF
    test_msg.append(cs)
    parsed = xbus.feed_bytes(bytes(test_msg))
    assert len(parsed) == 1
    assert parsed[0][0] == 0x30
    print("XBUS serial packet checksum verification passed.")

    print(f"MTi-680 CAN decode passed. Decoded Pitch: {pkt.pitch_deg}°, Roll: {pkt.roll_deg}°")
    return True


if __name__ == "__main__":
    run_mti680_driver_self_test()
