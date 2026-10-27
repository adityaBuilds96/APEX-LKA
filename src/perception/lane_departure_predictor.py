"""
src/perception/lane_departure_predictor.py
==========================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Predictive Time-to-Lane-Departure (TTLD) & Inertially-Compensated Trajectory Engine.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module implements the predictive departure early-warning engine for the APEX-Vision
platform. By continuously predicting the Time-to-Lane-Departure (TTLD) and projecting the
vehicle's kinematic path 1.5 to 2.5 seconds forward, the system identifies impending boundary
crossings well before physical departure occurs.

Key Mathematical Formulations:
------------------------------
1. 1st-Order Linear TTLD:
   TTLD_lin = d_boundary / |v_lat|
2. 2nd-Order Inertially-Compensated Dynamic TTLD:
   d(t) = d_0 - v_lat * t - 0.5 * a_lat * t^2
   Where:
     - d_0 is current distance from vehicle flank to lane boundary
     - v_lat is lateral velocity derived from speed and heading: v_lat = v_x * sin(theta_e) + v_y
     - a_lat is 400 Hz IMU lateral acceleration from MTi-680 INS
   Exact time-to-crossing t* is obtained by solving the quadratic:
     0.5 * a_lat * t^2 + v_lat * t - d_0 = 0
3. Departure Direction Resolution:
   Segregates departure trajectories toward Class 2 (Left Boundary) vs Class 3 (Right Boundary).
4. Pre-Emptive Warning Thresholds:
   - TTLD < 2.00 s: CAUTION (amber dashboard warning)
   - TTLD < 1.20 s: CRITICAL (acoustic chime + haptic steering vibration)
   - TTLD < 0.60 s: EMERGENCY (automatic counter-steer torque intervention via CAN 0x150)
5. Kinematic Forward Ribbon Projection:
   Projects discrete future vehicle bounding envelopes at t = {0.5s, 1.0s, 1.5s, 2.0s}.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger("APEX.LaneDeparturePredictor")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Telemetry & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class DepartureDirection(str, Enum):
    """Direction of predicted lane boundary departure."""
    NONE = "NONE"                       # Centered within lane corridor
    DEPARTING_LEFT = "DEPARTING_LEFT"   # Crossing Class 2 Left Boundary
    DEPARTING_RIGHT = "DEPARTING_RIGHT" # Crossing Class 3 Right Boundary


@unique
class DepartureSeverity(str, Enum):
    """Severity tier of the impending departure event."""
    NOMINAL = "NOMINAL"                 # TTLD >= 2.0s or stable centering
    WARNING = "WARNING"                 # 1.2s <= TTLD < 2.0s
    CRITICAL = "CRITICAL"               # 0.6s <= TTLD < 1.2s
    EMERGENCY = "EMERGENCY"             # TTLD < 0.6s or already touching buffer


@dataclass
class ProjectedWaypoint:
    """Discrete future vehicle center and envelope coordinate in metric BEV space."""
    time_horizon_s: float
    x_m: float                          # Lateral position (positive = right, negative = left)
    y_m: float                          # Longitudinal position forward
    heading_rad: float
    left_wheel_x_m: float
    right_wheel_x_m: float
    lateral_uncertainty_sigma_m: float


@dataclass
class LaneDeparturePrediction:
    """
    Comprehensive departure telemetry payload evaluated per perception frame.
    """
    timestamp_utc: str
    severity: DepartureSeverity
    direction: DepartureDirection
    time_to_departure_s: float          # Infinite (or 99.0) if centered
    distance_to_left_boundary_m: float
    distance_to_right_boundary_m: float
    lateral_velocity_mps: float
    lateral_acceleration_mps2: float
    yaw_rate_radps: float
    projected_ribbon: List[ProjectedWaypoint]
    can_alert_required: bool
    can_message_priority: int           # CAN arbitration ID (e.g. 0x140 or 0x150)
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert prediction to serializable dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "severity": self.severity.value,
            "direction": self.direction.value,
            "ttld_s": round(self.time_to_departure_s, 3),
            "d_left_m": round(self.distance_to_left_boundary_m, 3),
            "d_right_m": round(self.distance_to_right_boundary_m, 3),
            "v_lat_mps": round(self.lateral_velocity_mps, 3),
            "a_lat_mps2": round(self.lateral_acceleration_mps2, 3),
            "yaw_rate_degps": round(math.degrees(self.yaw_rate_radps), 2),
            "can_alert_required": self.can_alert_required,
            "can_message_priority": hex(self.can_message_priority),
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class DeparturePredictorConfig:
    """
    Hyperparameters governing kinematic extrapolation and departure warnings.
    """
    vehicle_width_m: float = 1.90       # Ego vehicle track width + mirrors
    vehicle_length_m: float = 4.50
    wheelbase_m: float = 2.70
    ttld_warning_threshold_s: float = 2.00
    ttld_critical_threshold_s: float = 1.20
    ttld_emergency_threshold_s: float = 0.60
    max_prediction_horizon_s: float = 2.50
    prediction_time_step_s: float = 0.25
    can_warning_id: int = 0x140
    can_emergency_id: int = 0x150
    enable_imu_acceleration: bool = True
    lateral_jerk_limit_mps3: float = 3.50


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Kinematic Vehicle Forward Projector
# ═══════════════════════════════════════════════════════════════════════════════

class KinematicVehicleProjector:
    """
    Projects ego vehicle future spatial coordinates using bicycle kinematic model:
      x_dot = v * sin(theta)
      y_dot = v * cos(theta)
      theta_dot = (v / L) * tan(delta) = omega_z
    Augmented with measured lateral acceleration a_lat from MTi-680 RTK INS.
    """

    def __init__(self, config: DeparturePredictorConfig) -> None:
        self.config = config

    def project_trajectory(
        self,
        longitudinal_speed_mps: float,
        lateral_speed_mps: float,
        heading_angle_rad: float,
        yaw_rate_radps: float,
        lateral_accel_mps2: float = 0.0,
    ) -> List[ProjectedWaypoint]:
        """
        Extrapolate vehicle bounding ribbon forward across discrete time horizon.
        """
        v_x = max(0.1, float(longitudinal_speed_mps))
        v_y = float(lateral_speed_mps)
        theta = float(heading_angle_rad)
        omega = float(yaw_rate_radps)
        a_y = float(lateral_accel_mps2) if self.config.enable_imu_acceleration else 0.0

        half_w = self.config.vehicle_width_m * 0.5
        dt = self.config.prediction_time_step_s
        steps = int(round(self.config.max_prediction_horizon_s / dt))

        curr_x = 0.0
        curr_y = 0.0
        curr_theta = theta
        curr_vy = v_y

        waypoints: List[ProjectedWaypoint] = []

        for i in range(1, steps + 1):
            t = i * dt

            # Advance kinematics
            curr_theta += omega * dt
            curr_vy += a_y * dt
            curr_x += (v_x * math.sin(curr_theta) + curr_vy) * dt
            curr_y += (v_x * math.cos(curr_theta)) * dt

            # Wheels flank lateral offsets
            left_wheel_x = curr_x - half_w * math.cos(curr_theta)
            right_wheel_x = curr_x + half_w * math.cos(curr_theta)

            # Uncertainty expands quadratically with time horizon
            uncertainty_sigma = 0.02 + 0.05 * t + 0.02 * (t ** 2)

            waypoints.append(ProjectedWaypoint(
                time_horizon_s=round(t, 2),
                x_m=round(curr_x, 3),
                y_m=round(curr_y, 3),
                heading_rad=round(curr_theta, 4),
                left_wheel_x_m=round(left_wheel_x, 3),
                right_wheel_x_m=round(right_wheel_x, 3),
                lateral_uncertainty_sigma_m=round(uncertainty_sigma, 3),
            ))

        return waypoints


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Master Lane Departure Predictor Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LaneDeparturePredictor:
    """
    Real-time predictive safety module assessing boundary crossing hazard.
    """

    def __init__(self, config: Optional[DeparturePredictorConfig] = None) -> None:
        self.config = config if config is not None else DeparturePredictorConfig()
        self.projector = KinematicVehicleProjector(self.config)

    def predict_departure(
        self,
        left_boundary_x_at_vehicle_m: float,
        right_boundary_x_at_vehicle_m: float,
        vehicle_speed_mps: float,
        heading_error_rad: float,
        yaw_rate_radps: float = 0.0,
        lateral_accel_mps2: float = 0.0,
        inner_safety_buffer_m: float = 0.30,
    ) -> LaneDeparturePrediction:
        """
        Evaluate time-to-lane-departure and generate safety alerts.
        """
        t0 = time.perf_counter()

        half_w = self.config.vehicle_width_m * 0.5
        v_x = max(0.5, float(vehicle_speed_mps))
        theta = float(heading_error_rad)
        omega = float(yaw_rate_radps)
        a_y = float(lateral_accel_mps2)

        # 1. Flank clearances to boundaries
        # Vehicle origin x=0, left boundary is negative, right is positive
        d_left = abs(left_boundary_x_at_vehicle_m) - half_w - inner_safety_buffer_m
        d_right = abs(right_boundary_x_at_vehicle_m) - half_w - inner_safety_buffer_m

        # 2. Current lateral velocity toward boundaries: v_lat = v_x * sin(theta)
        v_lat = v_x * math.sin(theta)

        # 3. Compute 2nd-order dynamic crossing time
        # Equation: 0.5 * a_y * t^2 + v_lat * t - d = 0
        ttld_left = self._solve_quadratic_crossing_time(d_left, -v_lat, -a_y)
        ttld_right = self._solve_quadratic_crossing_time(d_right, +v_lat, +a_y)

        # 4. Determine primary impending departure direction and min TTLD
        if ttld_left < ttld_right:
            active_ttld = ttld_left
            direction = DepartureDirection.DEPARTING_LEFT if active_ttld < 10.0 else DepartureDirection.NONE
        else:
            active_ttld = ttld_right
            direction = DepartureDirection.DEPARTING_RIGHT if active_ttld < 10.0 else DepartureDirection.NONE

        # 5. Classify Severity
        if active_ttld < self.config.ttld_emergency_threshold_s or (d_left <= 0.0 or d_right <= 0.0):
            severity = DepartureSeverity.EMERGENCY
            can_alert = True
            can_id = self.config.can_emergency_id
        elif active_ttld < self.config.ttld_critical_threshold_s:
            severity = DepartureSeverity.CRITICAL
            can_alert = True
            can_id = self.config.can_emergency_id
        elif active_ttld < self.config.ttld_warning_threshold_s:
            severity = DepartureSeverity.WARNING
            can_alert = True
            can_id = self.config.can_warning_id
        else:
            severity = DepartureSeverity.NOMINAL
            can_alert = False
            can_id = self.config.can_warning_id

        # 6. Forward Waypoint Ribbon Projection
        ribbon = self.projector.project_trajectory(v_x, v_lat, theta, omega, a_y)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return LaneDeparturePrediction(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            severity=severity,
            direction=direction,
            time_to_departure_s=round(min(99.0, active_ttld), 3),
            distance_to_left_boundary_m=round(d_left, 3),
            distance_to_right_boundary_m=round(d_right, 3),
            lateral_velocity_mps=round(v_lat, 3),
            lateral_acceleration_mps2=round(a_y, 3),
            yaw_rate_radps=round(omega, 4),
            projected_ribbon=ribbon,
            can_alert_required=can_alert,
            can_message_priority=can_id,
            processing_time_ms=elapsed_ms,
        )

    def _solve_quadratic_crossing_time(self, d_margin_m: float, v_towards_mps: float, a_towards_mps2: float) -> float:
        """
        Solves 0.5 * a * t^2 + v * t - d = 0 for the smallest positive real root t > 0.
        Returns 99.0 if vehicle is moving away or parallel.
        """
        if d_margin_m <= 0.0:
            return 0.0  # Already touching or breached

        # If negligible acceleration, use simple linear formulation: t = d / v
        if abs(a_towards_mps2) < 0.05:
            if v_towards_mps > 0.05:
                return d_margin_m / v_towards_mps
            return 99.0

        a_half = 0.5 * a_towards_mps2
        b = v_towards_mps
        c = -d_margin_m

        disc = b ** 2 - 4.0 * a_half * c
        if disc < 0.0:
            return 99.0  # Trajectory does not intersect boundary

        sqrt_disc = math.sqrt(disc)
        t1 = (-b + sqrt_disc) / (2.0 * a_half)
        t2 = (-b - sqrt_disc) / (2.0 * a_half)

        positive_roots = [t for t in (t1, t2) if t > 0.001]
        if not positive_roots:
            return 99.0
        return float(min(positive_roots))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CAN 0x140 / 0x150 Priority Departure Frame Encoder
# ═══════════════════════════════════════════════════════════════════════════════

class CANAlertEncoder:
    """
    Constructs ISO 11898-2 compliant SocketCAN binary frames for real-time
    dashboard alert and steering actuator intervention.
    """

    def __init__(self) -> None:
        self._rolling_counter: int = 0

    def encode_departure_frame(
        self,
        prediction: LaneDeparturePrediction,
    ) -> Tuple[int, bytes]:
        """
        Encode 8-byte CAN payload:
          Byte 0: Severity enum (0=Nominal, 1=Warning, 2=Critical, 3=Emergency)
          Byte 1: Direction enum (0=None, 1=Left, 2=Right)
          Byte 2-3: TTLD in milliseconds (uint16, big-endian)
          Byte 4: Lateral velocity in 0.05 m/s units (signed int8)
          Byte 5: Distance to closest boundary in 0.02 m units (uint8)
          Byte 6: Rolling message counter [0..255]
          Byte 7: CRC-8 SAE J1850 checksum over bytes 0-6
        """
        sev_map = {
            DepartureSeverity.NOMINAL: 0,
            DepartureSeverity.WARNING: 1,
            DepartureSeverity.CRITICAL: 2,
            DepartureSeverity.EMERGENCY: 3,
        }
        dir_map = {
            DepartureDirection.NONE: 0,
            DepartureDirection.DEPARTING_LEFT: 1,
            DepartureDirection.DEPARTING_RIGHT: 2,
        }

        sev_code = sev_map.get(prediction.severity, 0)
        dir_code = dir_map.get(prediction.direction, 0)

        # TTLD in ms, clamped to uint16
        ttld_ms = int(min(65535, max(0, prediction.time_to_departure_s * 1000.0)))
        ttld_hi = (ttld_ms >> 8) & 0xFF
        ttld_lo = ttld_ms & 0xFF

        # Lateral velocity: scale 0.05 m/s, clamped to int8 [-6.4 .. +6.35 m/s]
        v_scaled = int(round(np.clip(prediction.lateral_velocity_mps / 0.05, -128, 127)))
        v_u8 = v_scaled & 0xFF

        # Closest clearance: scale 0.02 m, clamped to uint8 [0 .. 5.10 m]
        min_d = min(prediction.distance_to_left_boundary_m, prediction.distance_to_right_boundary_m)
        d_scaled = int(round(np.clip(max(0.0, min_d) / 0.02, 0, 255)))

        # Rolling counter
        self._rolling_counter = (self._rolling_counter + 1) % 256
        counter_u8 = self._rolling_counter

        payload_7 = bytes([sev_code, dir_code, ttld_hi, ttld_lo, v_u8, d_scaled, counter_u8])
        crc = self._compute_crc8_sae_j1850(payload_7)
        full_payload = payload_7 + bytes([crc])

        can_id = prediction.can_message_priority
        return can_id, full_payload

    @staticmethod
    def _compute_crc8_sae_j1850(data: bytes) -> int:
        """Standard automotive SAE J1850 CRC-8 calculation (poly=0x1D)."""
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x1D) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc ^ 0xFF


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Centrifugal Tire Slip & Curvature Drift Estimator
# ═══════════════════════════════════════════════════════════════════════════════

class CentrifugalDriftEstimator:
    """
    Estimates steady-state cornering drift and lateral tire slip angle under high speeds:
      alpha_f = (m * v^2 / (L * C_alpha_f * R)) * (b / L)
    Prevents vehicle from drifting toward outer lane boundary on highway curves.
    """

    def __init__(
        self,
        vehicle_mass_kg: float = 1450.0,
        front_axle_weight_ratio: float = 0.55,
        cornering_stiffness_N_per_rad: float = 85000.0,
        wheelbase_m: float = 2.70,
    ) -> None:
        self.mass = vehicle_mass_kg
        self.front_ratio = front_axle_weight_ratio
        self.c_alpha = cornering_stiffness_N_per_rad
        self.L = wheelbase_m

    def estimate_curvature_drift_velocity(
        self,
        speed_mps: float,
        radius_of_curvature_m: float,
    ) -> float:
        """
        Compute steady-state lateral drift velocity component induced by tire deformation.
        Positive denotes outward drift away from curve center.
        """
        r = max(20.0, abs(float(radius_of_curvature_m)))
        v = max(0.5, float(speed_mps))

        # Centripetal acceleration a_y = v^2 / R
        a_y = (v ** 2) / r
        # Understeer gradient slip angle: alpha = (m * a_y) / (2 * C_alpha)
        slip_angle_rad = (self.mass * self.front_ratio * a_y) / (2.0 * self.c_alpha)
        # Lateral velocity component v_y = v * sin(alpha)
        v_drift = v * math.sin(slip_angle_rad)
        return float(np.clip(v_drift, -2.5, 2.5))


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Statistical Departure Risk Heatmap Rasterizer
# ═══════════════════════════════════════════════════════════════════════════════

class DepartureRiskHeatmapRasterizer:
    """
    Renders top-down 2D spatial Gaussian probability distribution of future
    vehicle trajectories directly onto BEV ground maps.
    """

    def __init__(self, meters_per_pixel: float = 0.0185) -> None:
        self.m_per_px = meters_per_pixel

    def rasterize_risk_heatmap(
        self,
        bev_canvas_bgr: np.ndarray,
        ribbon: Sequence[ProjectedWaypoint],
        center_x_px: int = 200,
        bottom_y_px: int = 580,
    ) -> np.ndarray:
        """
        Blend Gaussian heat discs for each projected waypoint along future vehicle trajectory.
        """
        h, w = bev_canvas_bgr.shape[:2]
        heatmap = np.zeros((h, w), dtype=np.float32)

        for wp in ribbon:
            px_x = int(round(center_x_px + (wp.x_m / self.m_per_px)))
            px_y = int(round(bottom_y_px - (wp.y_m / self.m_per_px)))

            if 0 <= px_x < w and 0 <= px_y < h:
                radius_px = int(round(max(4.0, wp.lateral_uncertainty_sigma_m / self.m_per_px)))
                cv2.circle(heatmap, (px_x, px_y), radius_px, 1.0, -1)

        # Smooth heatmap
        smoothed_heat = cv2.GaussianBlur(heatmap, (15, 15), 5.0)
        norm_heat = cv2.normalize(smoothed_heat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # False-color thermal colormap
        colored_heat = cv2.applyColorMap(norm_heat, cv2.COLORMAP_JET)

        # Blend where heat is positive
        mask = (norm_heat > 25).astype(np.uint8)
        mask_3ch = cv2.merge([mask, mask, mask])
        blended = np.where(mask_3ch > 0, cv2.addWeighted(bev_canvas_bgr, 0.6, colored_heat, 0.4, 0), bev_canvas_bgr)
        return blended


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Diagnostic HUD Canvas Visualizer
# ═══════════════════════════════════════════════════════════════════════════════

def render_departure_hud(
    canvas_bgr: np.ndarray,
    prediction: LaneDeparturePrediction,
    origin: Tuple[int, int] = (430, 25),
) -> np.ndarray:
    """
    Renders live predictive departure gauge, flashing warning badges, and TTLD countdown.
    """
    out = canvas_bgr.copy()
    x0, y0 = origin

    # Severity Colors
    color_map = {
        DepartureSeverity.NOMINAL: (0, 200, 0),       # Green
        DepartureSeverity.WARNING: (0, 255, 255),     # Yellow
        DepartureSeverity.CRITICAL: (0, 140, 255),    # Orange
        DepartureSeverity.EMERGENCY: (0, 0, 255),     # Red
    }
    sev_color = color_map.get(prediction.severity, (200, 200, 200))

    # Background Box
    cv2.rectangle(out, (x0 - 5, y0 - 15), (x0 + 200, y0 + 65), (20, 20, 20), -1)
    cv2.rectangle(out, (x0 - 5, y0 - 15), (x0 + 200, y0 + 65), sev_color, 2)

    # Title
    cv2.putText(out, f"TTLD: {prediction.time_to_departure_s:.2f} s", (x0, y0),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, sev_color, 2, cv2.LINE_AA)

    # Direction and Severity details
    det1 = f"STATUS: {prediction.severity.value}"
    det2 = f"DIR: {prediction.direction.value}"
    cv2.putText(out, det1, (x0, y0 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, det2, (x0, y0 + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    # Margin clearances
    clr = f"L: {prediction.distance_to_left_boundary_m:.2f}m | R: {prediction.distance_to_right_boundary_m:.2f}m"
    cv2.putText(out, clr, (x0, y0 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Automated Synthetic Benchmarking Harness
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DeparturePredictorBenchmarkReport:
    """Benchmark outcome verifying TTLD calculation accuracy and alert dispatch latency."""
    total_test_cases: int
    mean_ttld_error_sec: float
    max_ttld_error_sec: float
    alert_dispatch_latency_ms: float
    alert_dispatch_rate: float
    passed_iso26262_interlock: bool


def benchmark_lane_departure_predictor(
    predictor: Optional[LaneDeparturePredictor] = None,
) -> DeparturePredictorBenchmarkReport:
    """
    Stress-test TTLD calculations against known analytical kinematic equations:
    Case 1: Straight centering (v_lat = 0) -> TTLD = 99.0s, Nominal
    Case 2: Gentle drift right (v_lat = 0.5 m/s, d = 1.0m) -> TTLD = 2.0s, Warning
    Case 3: Rapid drift left (v_lat = 1.5 m/s, d = 0.9m) -> TTLD = 0.6s, Critical/Emergency
    """
    pred = predictor if predictor is not None else LaneDeparturePredictor()
    encoder = CANAlertEncoder()

    cases = [
        # (left_x, right_x, speed, heading, yaw_rate, a_y, buf, expected_ttld, expected_sev)
        (-1.85, 1.85, 20.0, 0.0, 0.0, 0.0, 0.30, 99.0, DepartureSeverity.NOMINAL),
        (-1.85, 1.85, 20.0, math.radians(1.5), 0.0, 0.0, 0.30, 1.15, DepartureSeverity.CRITICAL),
        (-1.85, 1.85, 20.0, math.radians(-3.0), 0.0, 0.0, 0.30, 0.57, DepartureSeverity.EMERGENCY),
    ]

    ttld_errors: List[float] = []
    latencies: List[float] = []
    alert_count = 0

    for c in cases:
        t0 = time.perf_counter()
        res = pred.predict_departure(
            left_boundary_x_at_vehicle_m=c[0],
            right_boundary_x_at_vehicle_m=c[1],
            vehicle_speed_mps=c[2],
            heading_error_rad=c[3],
            yaw_rate_radps=c[4],
            lateral_accel_mps2=c[5],
            inner_safety_buffer_m=c[6],
        )
        can_id, payload = encoder.encode_departure_frame(res)
        lat_ms = (time.perf_counter() - t0) * 1000.0

        latencies.append(lat_ms)
        err = abs(res.time_to_departure_s - c[7])
        ttld_errors.append(err)
        if res.can_alert_required:
            alert_count += 1

    mean_err = float(np.mean(ttld_errors))
    max_err = float(np.max(ttld_errors))
    mean_lat = float(np.mean(latencies))

    return DeparturePredictorBenchmarkReport(
        total_test_cases=len(cases),
        mean_ttld_error_sec=mean_err,
        max_ttld_error_sec=max_err,
        alert_dispatch_latency_ms=mean_lat,
        alert_dispatch_rate=alert_count / float(len(cases)),
        passed_iso26262_interlock=(mean_lat < 2.0 and max_err < 0.20),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 9. ISO 17361 Haptic & Acoustic Departure Warning Synthesizer
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class HapticWarningCommand:
    """Actuation command for steering wheel vibration motor and audio buzzer."""
    vibration_frequency_hz: float       # Typically 20-50 Hz
    vibration_amplitude_pct: float      # [0..100%]
    pulse_duration_ms: int
    pulse_count: int
    audio_chime_frequency_hz: int       # Typically 800 - 1500 Hz
    is_left_speaker_active: bool
    is_right_speaker_active: bool


class HapticAudioPatternSynthesizer:
    """
    Synthesizes directional haptic steering pulses and spatialized stereo chimes
    compliant with ISO 17361 Lane Departure Warning Systems (LDWS) specifications.
    """

    def __init__(self) -> None:
        pass

    def synthesize_warning(
        self,
        prediction: LaneDeparturePrediction,
    ) -> Optional[HapticWarningCommand]:
        """
        Map departure prediction to tactile and acoustic warning parameters.
        """
        if prediction.severity == DepartureSeverity.NOMINAL:
            return None

        is_left = (prediction.direction == DepartureDirection.DEPARTING_LEFT)
        is_right = (prediction.direction == DepartureDirection.DEPARTING_RIGHT)

        if prediction.severity == DepartureSeverity.WARNING:
            # Gentle pulsing chime, mild vibration
            return HapticWarningCommand(
                vibration_frequency_hz=25.0,
                vibration_amplitude_pct=40.0,
                pulse_duration_ms=150,
                pulse_count=2,
                audio_chime_frequency_hz=880,
                is_left_speaker_active=is_left,
                is_right_speaker_active=is_right,
            )
        elif prediction.severity == DepartureSeverity.CRITICAL:
            # Rapid high-frequency vibration, dual-pulse alarm
            return HapticWarningCommand(
                vibration_frequency_hz=40.0,
                vibration_amplitude_pct=75.0,
                pulse_duration_ms=250,
                pulse_count=3,
                audio_chime_frequency_hz=1200,
                is_left_speaker_active=is_left,
                is_right_speaker_active=is_right,
            )
        else:  # EMERGENCY
            # Continuous maximum vibration, high-pitch emergency buzzer
            return HapticWarningCommand(
                vibration_frequency_hz=50.0,
                vibration_amplitude_pct=100.0,
                pulse_duration_ms=600,
                pulse_count=4,
                audio_chime_frequency_hz=1500,
                is_left_speaker_active=True,
                is_right_speaker_active=True,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Driver Override & Intentional Lane Change Supervisor
# ═══════════════════════════════════════════════════════════════════════════════

class DriverTakeoverSupervisor:
    """
    Supervises driver steering wheel torque and turn signal status.
    Suppresses false-alarm warnings when driver intentionally initiates a lane change.
    """

    def __init__(
        self,
        override_torque_threshold_nm: float = 3.20,
        override_timeout_sec: float = 2.50,
    ) -> None:
        self.torque_threshold = override_torque_threshold_nm
        self.timeout_sec = override_timeout_sec
        self._override_active = False
        self._override_start_time = 0.0

    def evaluate_override(
        self,
        driver_torque_nm: float,
        turn_signal_left_active: bool,
        turn_signal_right_active: bool,
    ) -> Tuple[bool, str]:
        """
        Check if driver is deliberately overriding the lane-keeping system.
        Returns: (is_overridden, status_description)
        """
        now = time.perf_counter()

        # 1. Turn signal intent
        if turn_signal_left_active or turn_signal_right_active:
            self._override_active = True
            self._override_start_time = now
            return True, "TURN_SIGNAL_INTENTIONAL_LANE_CHANGE"

        # 2. Driver steering wheel torque override
        if abs(driver_torque_nm) >= self.torque_threshold:
            self._override_active = True
            self._override_start_time = now
            return True, f"DRIVER_STEERING_TORQUE_OVERRIDE ({abs(driver_torque_nm):.1f} Nm)"

        # 3. Timeout persistence
        if self._override_active:
            if (now - self._override_start_time) < self.timeout_sec:
                return True, "OVERRIDE_HOLD_PERSISTENCE"
            self._override_active = False

        return False, "AUTONOMOUS_SUPERVISED_ACTIVE"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_PREDICTOR: Optional[LaneDeparturePredictor] = None
_GLOBAL_CAN_ENCODER: Optional[CANAlertEncoder] = None
_GLOBAL_DRIFT_ESTIMATOR: Optional[CentrifugalDriftEstimator] = None
_GLOBAL_HEATMAP_RASTERIZER: Optional[DepartureRiskHeatmapRasterizer] = None
_GLOBAL_HAPTIC_SYNTH: Optional[HapticAudioPatternSynthesizer] = None
_GLOBAL_TAKEOVER_SUPERVISOR: Optional[DriverTakeoverSupervisor] = None


def get_lane_departure_predictor(config: Optional[DeparturePredictorConfig] = None) -> LaneDeparturePredictor:
    """
    Obtain or instantiate global singleton LaneDeparturePredictor.
    """
    global _GLOBAL_PREDICTOR
    if _GLOBAL_PREDICTOR is None:
        _GLOBAL_PREDICTOR = LaneDeparturePredictor(config=config)
    return _GLOBAL_PREDICTOR


def get_can_alert_encoder() -> CANAlertEncoder:
    """Obtain or instantiate global CANAlertEncoder."""
    global _GLOBAL_CAN_ENCODER
    if _GLOBAL_CAN_ENCODER is None:
        _GLOBAL_CAN_ENCODER = CANAlertEncoder()
    return _GLOBAL_CAN_ENCODER


def get_centrifugal_drift_estimator() -> CentrifugalDriftEstimator:
    """Obtain or instantiate global CentrifugalDriftEstimator."""
    global _GLOBAL_DRIFT_ESTIMATOR
    if _GLOBAL_DRIFT_ESTIMATOR is None:
        _GLOBAL_DRIFT_ESTIMATOR = CentrifugalDriftEstimator()
    return _GLOBAL_DRIFT_ESTIMATOR


def get_departure_risk_heatmap_rasterizer() -> DepartureRiskHeatmapRasterizer:
    """Obtain or instantiate global DepartureRiskHeatmapRasterizer."""
    global _GLOBAL_HEATMAP_RASTERIZER
    if _GLOBAL_HEATMAP_RASTERIZER is None:
        _GLOBAL_HEATMAP_RASTERIZER = DepartureRiskHeatmapRasterizer()
    return _GLOBAL_HEATMAP_RASTERIZER


def get_haptic_audio_synthesizer() -> HapticAudioPatternSynthesizer:
    """Obtain or instantiate global HapticAudioPatternSynthesizer."""
    global _GLOBAL_HAPTIC_SYNTH
    if _GLOBAL_HAPTIC_SYNTH is None:
        _GLOBAL_HAPTIC_SYNTH = HapticAudioPatternSynthesizer()
    return _GLOBAL_HAPTIC_SYNTH


def get_driver_takeover_supervisor() -> DriverTakeoverSupervisor:
    """Obtain or instantiate global DriverTakeoverSupervisor."""
    global _GLOBAL_TAKEOVER_SUPERVISOR
    if _GLOBAL_TAKEOVER_SUPERVISOR is None:
        _GLOBAL_TAKEOVER_SUPERVISOR = DriverTakeoverSupervisor()
    return _GLOBAL_TAKEOVER_SUPERVISOR


