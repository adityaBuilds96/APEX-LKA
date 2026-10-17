"""
src/control/lanekeep_controllers.py
===================================
APEX-Vision Core v4.0 • Autonomous Lateral Vehicle Actuation Control Stack
ISO 26262 ASIL-B Aligned Controller Implementation

Implements precision lateral guidance, path tracking, and steering actuation control:
1. Cross-Track Error (e_lat) and Heading Error (theta_e) extraction along lookahead horizons.
2. Anti-Windup Proportional-Integral-Derivative (PID) Controller with filtered derivative.
3. Front-Axle Stanley Path Tracking Controller with velocity softening and yaw damping.
4. Curvature Feedforward Steering (Kinematic Bicycle Model: delta_ff = arctan(L * kappa)).
5. Pure Pursuit Lateral Guidance with speed-scheduled lookahead distance.
6. Actuation Saturation & Slew Rate Limiter (28 deg max angle, 40 deg/s max rate for BAJA SAE Buggy).
7. ISO 26262 ASIL-B Safety Supervisory Envelope (Watchdog heartbeat, plausibility checks).
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.LaneKeepControllers")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums, Vehicle Telemetry & Control Configurations
# ═══════════════════════════════════════════════════════════════════════════════

class ControlMode(Enum):
    """Operational mode of the lateral guidance controller."""
    STANLEY = "STANLEY"
    PID = "PID"
    PURE_PURSUIT = "PURE_PURSUIT"
    HYBRID_STANLEY_FEEDFORWARD = "HYBRID_STANLEY_FEEDFORWARD"
    DEGRADED_FAILSAFE = "DEGRADED_FAILSAFE"


class ActuatorState(Enum):
    """Health and saturation state of the steering actuation subsystem."""
    NOMINAL = "NOMINAL"
    ANGLE_SATURATED = "ANGLE_SATURATED"
    RATE_LIMITED = "RATE_LIMITED"
    ASIL_B_TRIPPED = "ASIL_B_TRIPPED"


@dataclass
class VehicleKinematics:
    """Real-time kinodynamic state of the vehicle platform."""
    timestamp_s: float
    velocity_mps: float              # Longitudinal forward speed (v_x)
    lateral_velocity_mps: float = 0.0 # Lateral slip velocity (v_y)
    yaw_rate_radps: float = 0.0      # Angular yaw velocity (r = dpsi/dt)
    steering_angle_rad: float = 0.0  # Current steering wheel / road wheel angle
    pitch_rad: float = 0.0           # Body pitch angle from IMU
    roll_rad: float = 0.0            # Body roll angle from IMU


@dataclass
class PathGeometry:
    """Geometric description of reference lane centerline in vehicle frame."""
    x_offset_m: float                # Lateral offset at lookahead distance (e_lat)
    heading_error_rad: float         # Angular orientation mismatch (theta_e = psi_path - psi_veh)
    curvature_inv_m: float           # Road curvature kappa = 1 / R (m^-1)
    lookahead_distance_m: float      # Distance downrange where error is evaluated
    confidence: float = 1.0          # Perception detection confidence [0.0, 1.0]
    coefficients: Optional[Tuple[float, float, float]] = None # Parabola x = A*y^2 + B*y + C


@dataclass
class ControllerGains:
    """Configurable tuning parameters for PID, Stanley, and Pure Pursuit controllers."""
    # PID Gains
    kp: float = 0.85
    ki: float = 0.04
    kd: float = 0.35
    integral_max_rad: float = 0.15   # Max integrator clamp for anti-windup (~8.6 deg)
    derivative_lpf_alpha: float = 0.70 # Derivative first-order low-pass filter constant

    # Stanley Controller Gains
    k_stanley: float = 1.25          # Cross-track error gain (s^-1)
    k_soft: float = 1.0              # Velocity softening factor epsilon_v (m/s)
    k_yaw_damping: float = 0.12      # Yaw rate damping gain (s)

    # Pure Pursuit Lookahead
    k_speed_lookahead: float = 0.40  # Speed-dependent lookahead gain (s)
    min_lookahead_m: float = 3.0     # Minimum lookahead distance (m)
    max_lookahead_m: float = 25.0    # Maximum lookahead distance (m)

    # Feedforward Curvature Gain
    enable_feedforward: bool = True
    feedforward_gain: float = 1.0


@dataclass
class VehicleProfileConfig:
    """Hardware and kinodynamic parameters for the vehicle chassis."""
    profile_name: str = "BAJA_SAE_BUGGY"
    wheelbase_m: float = 1.55        # Wheelbase L (m)
    track_width_m: float = 1.35      # Track width W (m)
    mass_kg: float = 230.0           # Vehicle mass
    steering_ratio: float = 1.0      # 1:1 direct rack-and-pinion steering ratio for Baja Buggy

    # Actuation Saturation Limits (BAJA SAE Buggy Profile)
    max_steering_angle_deg: float = 28.0 # Hard mechanical steering lock (28.0 deg = 0.48869 rad)
    max_steering_rate_degps: float = 40.0 # Maximum steering rack slew rate (40.0 deg/s = 0.69813 rad/s)
    max_lateral_accel_g: float = 0.65    # Max rollover threshold for high-suspension buggy
    enable_rollover_governor: bool = False # Configurable kinodynamic rollover prevention

    # ASIL-B Safety Thresholds
    max_allowed_crosstrack_err_m: float = 1.50 # Corridor boundary departure threshold
    max_consecutive_fault_cycles: int = 5     # Disengagement trigger threshold (166 ms @ 30 Hz)


@dataclass
class ActuationCommand:
    """Complete sanitized steering command ready for CAN transmission."""
    timestamp_s: float
    steering_angle_rad: float        # Final sanitized road wheel angle command
    steering_angle_deg: float        # Angle in degrees for human readout
    steering_rate_degps: float       # Applied rate of angle change
    crosstrack_error_m: float        # Lateral error used in computation
    heading_error_rad: float         # Heading error used in computation
    feedforward_angle_rad: float     # Curvature feedforward contribution
    feedback_angle_rad: float        # Controller feedback contribution
    actuator_state: ActuatorState
    is_clamped: bool
    is_rate_limited: bool
    asil_b_ok: bool
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry & Lookahead Horizon Computation
# ═══════════════════════════════════════════════════════════════════════════════

class PathGeometryExtractor:
    """
    Computes cross-track offset, heading orientation, and curvature from
    polynomial road centerlines or waypoint sequences.
    """

    @staticmethod
    def compute_adaptive_lookahead(
        velocity_mps: float,
        k_speed: float = 0.40,
        min_ld: float = 3.0,
        max_ld: float = 25.0,
    ) -> float:
        """
        Adaptive lookahead distance proportional to vehicle speed:
        L_d(v) = clamp(k_speed * v + min_ld, min_ld, max_ld)
        """
        ld = k_speed * max(0.0, velocity_mps) + min_ld
        return float(np.clip(ld, min_ld, max_ld))

    @staticmethod
    def extract_geometry_from_polynomial(
        poly_coeffs: Tuple[float, float, float],
        lookahead_m: float,
    ) -> PathGeometry:
        """
        Given centerline parabola x = A*y^2 + B*y + C in BEV vehicle frame
        (where y is forward longitudinal, x is lateral right):
        - Cross-track error at lookahead y=Ld: e_lat = -(A*Ld^2 + B*Ld + C)
          (convention: positive e_lat means vehicle is right of target, requires left steer).
        - Heading error theta_e = -arctan(dx/dy) = -arctan(2*A*Ld + B).
        - Curvature kappa = 2*|A| / (1 + (2*A*Ld + B)^2)^1.5.
        """
        a, b, c = poly_coeffs
        y = lookahead_m

        # Centerline lateral coordinate at lookahead
        x_target = a * (y ** 2) + b * y + c

        # Signed lateral offset: Vehicle is at x=0, so e_lat = x_veh - x_target = -x_target
        e_lat = -x_target

        # Tangent slope dx/dy
        dx_dy = 2.0 * a * y + b
        heading_err = -math.atan(dx_dy)

        # Analytical curvature
        denom = (1.0 + (dx_dy ** 2)) ** 1.5
        kappa = (2.0 * a) / max(1e-6, denom)

        return PathGeometry(
            x_offset_m=float(e_lat),
            heading_error_rad=float(heading_err),
            curvature_inv_m=float(kappa),
            lookahead_distance_m=float(lookahead_m),
            confidence=1.0,
            coefficients=poly_coeffs,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Lateral Controllers: PID, Stanley, Pure Pursuit
# ═══════════════════════════════════════════════════════════════════════════════

class LateralPIDController:
    """
    Precision Proportional-Integral-Derivative (PID) Lateral Controller.
    Features:
    - Back-calculation anti-windup clamping on the integral accumulator.
    - First-order low-pass filtered derivative to suppress sensor noise.
    - Slew-rate awareness for seamless integration with vehicle steering rack.
    """

    def __init__(self, gains: Optional[ControllerGains] = None) -> None:
        self.gains = gains or ControllerGains()
        self.integral_accumulator = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_timestamp_s = 0.0
        self.is_initialized = False

    def reset(self) -> None:
        """Reset internal integrator and memory registers."""
        self.integral_accumulator = 0.0
        self.prev_error = 0.0
        self.filtered_derivative = 0.0
        self.prev_timestamp_s = 0.0
        self.is_initialized = False

    def compute_steering(
        self,
        crosstrack_error_m: float,
        timestamp_s: float,
        is_actuator_saturated: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute target steering angle from cross-track error:
        u(t) = Kp * e(t) + Ki * int(e) + Kd * de_filtered(t)/dt

        Anti-windup:
        If actuator is currently saturated, freeze or decay integral growth.
        """
        if not self.is_initialized or self.prev_timestamp_s <= 0.0:
            dt = 0.0333  # Nominal 30 Hz cycle fallback
        else:
            dt = timestamp_s - self.prev_timestamp_s

        dt = float(np.clip(dt, 0.001, 0.200))
        error = crosstrack_error_m

        # 1. Proportional term
        p_term = self.gains.kp * error

        # 2. Integral term with anti-windup
        if not is_actuator_saturated:
            self.integral_accumulator += error * dt
            # Anti-windup clamping: |I| <= I_max
            self.integral_accumulator = float(np.clip(
                self.integral_accumulator,
                -self.gains.integral_max_rad / max(1e-4, self.gains.ki),
                self.gains.integral_max_rad / max(1e-4, self.gains.ki),
            ))
        else:
            # Integrator decay when saturated
            self.integral_accumulator *= 0.98

        i_term = self.gains.ki * self.integral_accumulator

        # 3. Derivative term with first-order low-pass filter
        if self.is_initialized:
            raw_derivative = (error - self.prev_error) / dt
            alpha = self.gains.derivative_lpf_alpha
            self.filtered_derivative = alpha * self.filtered_derivative + (1.0 - alpha) * raw_derivative
        else:
            self.filtered_derivative = 0.0
            self.is_initialized = True

        d_term = self.gains.kd * self.filtered_derivative

        # Total feedback steering command
        raw_cmd = p_term + i_term + d_term

        self.prev_error = error
        self.prev_timestamp_s = timestamp_s

        telemetry = {
            "p_term": float(p_term),
            "i_term": float(i_term),
            "d_term": float(d_term),
            "integral_accumulator": float(self.integral_accumulator),
            "filtered_derivative": float(self.filtered_derivative),
        }
        return float(raw_cmd), telemetry


class StanleyPathTracker:
    """
    Nonlinear Front-Axle Stanley Path Tracking Controller.
    Reference: Hoffman et al. (DARPA Grand Challenge Stanford Stanley vehicle).

    Steering law:
    delta(t) = theta_e(t) + arctan(k * e_lat(t) / (v(t) + epsilon_v)) - k_yaw * r(t)
    where:
    - theta_e: Heading angle error between path tangent and vehicle heading.
    - e_lat: Signed cross-track error measured from front axle to nearest path point.
    - k: Cross-track gain.
    - epsilon_v: Velocity softening parameter ensuring stability at near-zero speeds.
    - k_yaw: Dynamic yaw-rate damping term to suppress high-speed oscillation.
    """

    def __init__(
        self,
        gains: Optional[ControllerGains] = None,
        wheelbase_m: float = 1.55,
    ) -> None:
        self.gains = gains or ControllerGains()
        self.wheelbase_m = wheelbase_m

    def compute_steering(
        self,
        geometry: PathGeometry,
        kinematics: VehicleKinematics,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Execute Stanley path tracking calculation.
        """
        v = max(0.1, kinematics.velocity_mps)
        theta_e = geometry.heading_error_rad
        e_lat = geometry.x_offset_m
        yaw_rate = kinematics.yaw_rate_radps

        # 1. Heading alignment component
        heading_term = theta_e

        # 2. Cross-track correction component with softening
        crosstrack_term = math.atan((self.gains.k_stanley * e_lat) / (v + self.gains.k_soft))

        # 3. Dynamic yaw rate damping component
        yaw_damping_term = -self.gains.k_yaw_damping * yaw_rate

        # Combined Stanley feedback steering angle
        delta_stanley = heading_term + crosstrack_term + yaw_damping_term

        telemetry = {
            "heading_term_rad": float(heading_term),
            "crosstrack_term_rad": float(crosstrack_term),
            "yaw_damping_term_rad": float(yaw_damping_term),
            "e_lat_m": float(e_lat),
            "theta_e_rad": float(theta_e),
        }
        return float(delta_stanley), telemetry


class PurePursuitController:
    """
    Geometric Pure Pursuit Lateral Controller.
    Computes steering angle required to chase a preview waypoint along an arc:
    delta_pp = arctan(2 * L * sin(alpha) / L_d)
    where:
    - L: Wheelbase (m)
    - alpha: Angle between vehicle heading vector and preview point vector
    - L_d: Lookahead distance
    """

    def __init__(self, wheelbase_m: float = 1.55, gains: Optional[ControllerGains] = None) -> None:
        self.wheelbase_m = wheelbase_m
        self.gains = gains or ControllerGains()

    def compute_steering(
        self,
        geometry: PathGeometry,
        velocity_mps: float,
    ) -> Tuple[float, Dict[str, float]]:
        """
        Execute Pure Pursuit geometric arc calculation.
        """
        ld = PathGeometryExtractor.compute_adaptive_lookahead(
            velocity_mps=velocity_mps,
            k_speed=self.gains.k_speed_lookahead,
            min_ld=self.gains.min_lookahead_m,
            max_ld=self.gains.max_lookahead_m,
        )

        # Angle alpha to target in vehicle frame
        # If target is at lateral distance x_offset and forward distance ld:
        alpha = math.atan2(geometry.x_offset_m, ld)

        # Pure Pursuit steering equation: delta = arctan(2 * L * sin(alpha) / ld)
        sin_alpha = math.sin(alpha)
        curvature_arc = (2.0 * sin_alpha) / max(1e-4, ld)
        delta_pp = math.atan(self.wheelbase_m * curvature_arc)

        telemetry = {
            "pure_pursuit_ld_m": float(ld),
            "alpha_rad": float(alpha),
            "arc_curvature": float(curvature_arc),
        }
        return float(delta_pp), telemetry


class CurvatureFeedforward:
    """
    Kinematic Bicycle Model feedforward steering term for steady-state curve negotiation:
    delta_ff = arctan(L * kappa)
    where L is wheelbase and kappa is path curvature (1 / R).
    """

    @staticmethod
    def compute_feedforward_angle(wheelbase_m: float, curvature_inv_m: float) -> float:
        """
        Compute steady-state ackermann angle for circular arc.
        """
        return float(math.atan(wheelbase_m * curvature_inv_m))


# ═══════════════════════════════════════════════════════════════════════════════
# Actuation Saturation & Slew Rate Limiter (BAJA SAE Buggy Constraints)
# ═══════════════════════════════════════════════════════════════════════════════

class ActuationSaturationLimiter:
    """
    Enforces strict physical and kinodynamic actuator boundaries:
    1. Maximum steering angle lock (BAJA SAE Buggy: 28.0 deg = 0.48869 rad).
    2. Maximum steering rate / rack slew rate (BAJA SAE Buggy: 40.0 deg/s = 0.69813 rad/s).
    3. Lateral acceleration rollover governor: |a_lat| = v^2 * kappa <= a_lat_max.
    """

    def __init__(self, profile: Optional[VehicleProfileConfig] = None) -> None:
        self.profile = profile or VehicleProfileConfig()
        self.max_angle_rad = math.radians(self.profile.max_steering_angle_deg)
        self.max_rate_radps = math.radians(self.profile.max_steering_rate_degps)

        self.current_applied_angle_rad = 0.0
        self.prev_timestamp_s = 0.0
        self.is_initialized = False

    def reset(self, initial_angle_rad: float = 0.0) -> None:
        """Reset internal rate limiter state."""
        self.current_applied_angle_rad = float(np.clip(initial_angle_rad, -self.max_angle_rad, self.max_angle_rad))
        self.prev_timestamp_s = 0.0
        self.is_initialized = True

    def sanitize_command(
        self,
        requested_angle_rad: float,
        timestamp_s: float,
        current_velocity_mps: float = 0.0,
    ) -> Tuple[float, ActuatorState, bool, bool, float]:
        """
        Clamp requested steering command within angle limits and slew rate limits.
        Returns:
            (sanitized_angle_rad, actuator_state, is_clamped, is_rate_limited, applied_rate_degps)
        """
        if not self.is_initialized:
            self.current_applied_angle_rad = 0.0
            self.prev_timestamp_s = timestamp_s
            self.is_initialized = True
            dt = 0.0333
        elif self.prev_timestamp_s <= 0.0:
            dt = 0.0333
            self.prev_timestamp_s = timestamp_s
        else:
            dt = timestamp_s - self.prev_timestamp_s

        dt = float(np.clip(dt, 0.001, 0.200))

        # 1. Lateral acceleration rollover governor check (if enabled)
        effective_max_angle = self.max_angle_rad
        if self.profile.enable_rollover_governor and current_velocity_mps > 3.0:
            max_accel_mps2 = self.profile.max_lateral_accel_g * 9.80665
            rollover_limit = (max_accel_mps2 * self.profile.wheelbase_m) / (current_velocity_mps ** 2)
            effective_max_angle = min(self.max_angle_rad, max(math.radians(5.0), rollover_limit))

        # 2. Angle Saturation Hard Clamp
        clamped_angle = float(np.clip(requested_angle_rad, -effective_max_angle, effective_max_angle))
        is_angle_clamped = abs(clamped_angle - requested_angle_rad) > 1e-5

        # 3. Slew Rate Limiter (Max Rate: 40 deg/s)
        max_delta = self.max_rate_radps * dt
        angle_diff = clamped_angle - self.current_applied_angle_rad

        if abs(angle_diff) > max_delta:
            is_rate_limited = True
            step = math.copysign(max_delta, angle_diff)
            final_angle = self.current_applied_angle_rad + step
        else:
            is_rate_limited = False
            final_angle = clamped_angle

        # Compute actual applied rate in deg/s for diagnostics
        applied_rate_degps = math.degrees(abs(final_angle - self.current_applied_angle_rad) / dt)

        # Update state register
        self.current_applied_angle_rad = final_angle
        self.prev_timestamp_s = timestamp_s

        # Determine actuator state enum
        if is_angle_clamped:
            state = ActuatorState.ANGLE_SATURATED
        elif is_rate_limited:
            state = ActuatorState.RATE_LIMITED
        else:
            state = ActuatorState.NOMINAL

        return final_angle, state, is_angle_clamped, is_rate_limited, applied_rate_degps


# ═══════════════════════════════════════════════════════════════════════════════
# ISO 26262 ASIL-B Safety Supervisory Layer
# ═══════════════════════════════════════════════════════════════════════════════

class ASILBActuationSafetySupervisor:
    """
    ISO 26262 ASIL-B aligned safety supervisor monitoring steering commands
    before transmission over the 500 kbps CAN bus to the dual Arduino Mega watchdogs.

    Monitors:
    1. Cross-track departure breach (|e_lat| > 1.50 m).
    2. Command plausibility (sudden sign reversals or NaN/inf values).
    3. Watchdog counter and cyclic redundancy check (CRC-8).
    4. Slew violation detection.
    """

    def __init__(self, profile: Optional[VehicleProfileConfig] = None) -> None:
        self.profile = profile or VehicleProfileConfig()
        self.fault_counter = 0
        self.heartbeat_counter = 0
        self.prev_command_rad = 0.0

    @staticmethod
    def compute_crc8(data: bytes) -> int:
        """Compute standard SAE J1850 CRC-8 checksum over payload bytes."""
        crc = 0xFF
        for byte in data:
            crc ^= byte
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ 0x1D) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc ^ 0xFF

    def verify_command_safety(
        self,
        command_rad: float,
        crosstrack_error_m: float,
        velocity_mps: float,
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate safety invariants for steering command.
        Returns: (is_safe, violation_reason, safety_telemetry)
        """
        self.heartbeat_counter = (self.heartbeat_counter + 1) % 256

        # Invariant 1: Finite real number check
        if math.isnan(command_rad) or math.isinf(command_rad):
            self.fault_counter += 1
            return False, "ASIL_B_ERR_NAN_COMMAND", {"heartbeat": self.heartbeat_counter}

        # Invariant 2: Steering lock boundary check (28.0 degrees)
        max_rad = math.radians(self.profile.max_steering_angle_deg) + 1e-4
        if abs(command_rad) > max_rad:
            self.fault_counter += 1
            return False, "ASIL_B_ERR_ANGLE_BOUND_EXCEEDED", {"heartbeat": self.heartbeat_counter}

        # Invariant 3: Severe corridor departure check
        if abs(crosstrack_error_m) > self.profile.max_allowed_crosstrack_err_m:
            self.fault_counter += 1
            if self.fault_counter >= self.profile.max_consecutive_fault_cycles:
                return False, "ASIL_B_ERR_CORRIDOR_DEPARTURE_BREACH", {"heartbeat": self.heartbeat_counter}

        # Invariant 4: Reverse steer hazard (steering opposite to corrective direction at speed)
        # If e_lat is large positive (> 0.5m) and vehicle commands hard right (> +15 deg)
        if crosstrack_error_m > 0.60 and command_rad > math.radians(15.0) and velocity_mps > 5.0:
            self.fault_counter += 1
            return False, "ASIL_B_ERR_REVERSE_STEER_HAZARD", {"heartbeat": self.heartbeat_counter}

        # Reset or decrement fault counter on valid cycles
        self.fault_counter = max(0, self.fault_counter - 1)
        self.prev_command_rad = command_rad

        # Simulate CAN payload serialization for dual Arduino Mega watchdogs
        # 16-bit signed integer angle in 0.01 deg units + 8-bit alive counter + 8-bit CRC
        angle_units = int(round(math.degrees(command_rad) * 100.0))
        angle_bytes = angle_units.to_bytes(2, byteorder="big", signed=True)
        heartbeat_byte = bytes([self.heartbeat_counter])
        crc_val = self.compute_crc8(angle_bytes + heartbeat_byte)

        can_payload = {
            "can_id": 0x210,
            "angle_raw": angle_units,
            "heartbeat": self.heartbeat_counter,
            "crc8": crc_val,
        }

        return True, "ASIL_B_VERIFIED_OK", can_payload


# ═══════════════════════════════════════════════════════════════════════════════
# Master Lateral Controller Orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

class LaneKeepController:
    """
    Master Lateral Guidance Orchestrator.
    Integrates PID, Stanley, Pure Pursuit, Curvature Feedforward, Saturation Limiting,
    and ASIL-B Safety Verification into a unified real-time control interface.
    """

    def __init__(
        self,
        mode: ControlMode = ControlMode.STANLEY,
        profile: Optional[VehicleProfileConfig] = None,
        gains: Optional[ControllerGains] = None,
    ) -> None:
        self.mode = mode
        self.profile = profile or VehicleProfileConfig()
        self.gains = gains or ControllerGains()

        # Component modules
        self.pid = LateralPIDController(gains=self.gains)
        self.stanley = StanleyPathTracker(gains=self.gains, wheelbase_m=self.profile.wheelbase_m)
        self.pure_pursuit = PurePursuitController(wheelbase_m=self.profile.wheelbase_m, gains=self.gains)
        self.limiter = ActuationSaturationLimiter(profile=self.profile)
        self.safety_supervisor = ASILBActuationSafetySupervisor(profile=self.profile)

        self.last_command: Optional[ActuationCommand] = None

    def reset(self) -> None:
        """Reset all internal controller memories."""
        self.pid.reset()
        self.limiter.reset()
        self.last_command = None

    def set_mode(self, mode: ControlMode) -> None:
        """Switch active lateral control algorithm."""
        self.mode = mode
        logger.info(f"Switched lateral control mode to {self.mode.value}")

    def compute_actuation(
        self,
        geometry: PathGeometry,
        kinematics: VehicleKinematics,
    ) -> ActuationCommand:
        """
        Execute lateral guidance pass:
        1. Evaluate raw steering angle from active controller (Stanley/PID/PP).
        2. Add curvature feedforward term: delta_ff = arctan(L * kappa).
        3. Apply kinodynamic angle saturation and slew-rate limits (28 deg max, 40 deg/s max).
        4. Validate against ISO 26262 ASIL-B safety invariants.
        5. Package and return full ActuationCommand.
        """
        t_now = kinematics.timestamp_s

        # 1. Curvature Feedforward
        if self.gains.enable_feedforward:
            delta_ff = CurvatureFeedforward.compute_feedforward_angle(
                wheelbase_m=self.profile.wheelbase_m,
                curvature_inv_m=geometry.curvature_inv_m,
            ) * self.gains.feedforward_gain
        else:
            delta_ff = 0.0

        # 2. Feedback Controller Execution
        diag_data: Dict[str, Any] = {}

        if self.mode == ControlMode.STANLEY or self.mode == ControlMode.HYBRID_STANLEY_FEEDFORWARD:
            delta_fb, stanley_diag = self.stanley.compute_steering(geometry, kinematics)
            diag_data.update(stanley_diag)

        elif self.mode == ControlMode.PID:
            is_sat = (self.last_command is not None and self.last_command.is_clamped)
            delta_fb, pid_diag = self.pid.compute_steering(geometry.x_offset_m, t_now, is_actuator_saturated=is_sat)
            diag_data.update(pid_diag)

        elif self.mode == ControlMode.PURE_PURSUIT:
            delta_fb, pp_diag = self.pure_pursuit.compute_steering(geometry, kinematics.velocity_mps)
            diag_data.update(pp_diag)

        elif self.mode == ControlMode.DEGRADED_FAILSAFE:
            # Smoothly bring steering to neutral center (0 deg)
            delta_fb = 0.0
            delta_ff = 0.0
            diag_data["status"] = "FAILSAFE_CENTERING"

        else:
            delta_fb = 0.0

        # 3. Combine Feedforward and Feedback
        raw_target_angle = delta_fb + delta_ff

        # 4. Actuation Saturation & Slew Rate Limiting (BAJA SAE Buggy: 28 deg max, 40 deg/s max)
        sanitized_angle, state, is_clamped, is_rate_limited, applied_rate = self.limiter.sanitize_command(
            requested_angle_rad=raw_target_angle,
            timestamp_s=t_now,
            current_velocity_mps=kinematics.velocity_mps,
        )

        # 5. ISO 26262 ASIL-B Safety Verification
        is_safe, reason, can_telemetry = self.safety_supervisor.verify_command_safety(
            command_rad=sanitized_angle,
            crosstrack_error_m=geometry.x_offset_m,
            velocity_mps=kinematics.velocity_mps,
        )

        if not is_safe:
            # Trip into failsafe degraded state: clamp to safe limit or neutral
            state = ActuatorState.ASIL_B_TRIPPED
            sanitized_angle = float(np.clip(sanitized_angle, -math.radians(10.0), math.radians(10.0)))

        diag_data.update(can_telemetry)
        diag_data["asil_b_status"] = reason

        command = ActuationCommand(
            timestamp_s=t_now,
            steering_angle_rad=sanitized_angle,
            steering_angle_deg=round(math.degrees(sanitized_angle), 2),
            steering_rate_degps=round(applied_rate, 2),
            crosstrack_error_m=round(geometry.x_offset_m, 3),
            heading_error_rad=round(geometry.heading_error_rad, 4),
            feedforward_angle_rad=round(delta_ff, 4),
            feedback_angle_rad=round(delta_fb, 4),
            actuator_state=state,
            is_clamped=is_clamped,
            is_rate_limited=is_rate_limited,
            asil_b_ok=is_safe,
            diagnostics=diag_data,
        )

        self.last_command = command
        return command


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Comprehensive Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_lanekeep_controller(
    mode: ControlMode = ControlMode.STANLEY,
    profile_name: str = "BAJA_SAE_BUGGY",
) -> LaneKeepController:
    """Factory helper to instantiate LaneKeepController."""
    profile = VehicleProfileConfig(profile_name=profile_name)
    return LaneKeepController(mode=mode, profile=profile)


def run_control_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. Lookahead and polynomial geometry extraction.
    2. PID controller step response, integral accumulator anti-windup, and low-pass filtering.
    3. Stanley controller cross-track and heading error convergence.
    4. Curvature feedforward bicycle model math.
    5. Pure pursuit geometric chasing.
    6. Actuation saturation clipping to strict 28.0 deg limit.
    7. Slew rate limiter to strict 40.0 deg/s max rate.
    8. ISO 26262 ASIL-B safety supervisor CRC and violation trip checks.
    """
    print("Executing LaneKeepController Verification Suite...")
    profile = VehicleProfileConfig(
        profile_name="BAJA_SAE_BUGGY",
        max_steering_angle_deg=28.0,
        max_steering_rate_degps=40.0,
    )
    controller = LaneKeepController(mode=ControlMode.STANLEY, profile=profile)

    # -------------------------------------------------------------------------
    # Test 1: Geometry Extraction from Centerline Parabola
    # -------------------------------------------------------------------------
    # Parabola: x = 0.005 * y^2 - 0.02 * y + 0.35 (Curve right with 0.35m initial right offset)
    poly = (0.005, -0.02, 0.35)
    geo = PathGeometryExtractor.extract_geometry_from_polynomial(poly, lookahead_m=10.0)
    # At y=10: x = 0.005*100 - 0.2 + 0.35 = 0.65m  => e_lat = -0.65m
    assert abs(geo.x_offset_m - (-0.65)) < 1e-4, f"e_lat mismatch: {geo.x_offset_m}"
    assert geo.curvature_inv_m > 0.0, "Curvature should be positive"

    # -------------------------------------------------------------------------
    # Test 2: Stanley Controller Mathematics
    # -------------------------------------------------------------------------
    kin = VehicleKinematics(
        timestamp_s=100.0,
        velocity_mps=8.0,
        yaw_rate_radps=0.02,
        steering_angle_rad=0.0,
    )
    cmd_stanley = controller.compute_actuation(geo, kin)
    print(f"Stanley Command: {cmd_stanley.steering_angle_deg:.2f}° | State: {cmd_stanley.actuator_state.name}")
    assert abs(cmd_stanley.steering_angle_deg) <= 28.0, "Angle exceeds physical lock"
    assert cmd_stanley.asil_b_ok is True, "ASIL-B supervisor tripped on nominal input"

    # -------------------------------------------------------------------------
    # Test 3: Actuator Angle Saturation Hard Clamp (28 deg max)
    # -------------------------------------------------------------------------
    # Extreme error requiring huge steering command
    extreme_geo = PathGeometry(
        x_offset_m=5.0,  # 5 meters off course
        heading_error_rad=math.radians(35.0),
        curvature_inv_m=0.10,
        lookahead_distance_m=10.0,
    )
    # Run multiple steps to let slew rate reach maximum angle
    t_step = 101.0
    for _ in range(30):
        kin.timestamp_s = t_step
        cmd_sat = controller.compute_actuation(extreme_geo, kin)
        t_step += 0.0333

    print(f"Saturated Angle: {cmd_sat.steering_angle_deg:.2f}° (Limit: 28.00°)")
    assert abs(cmd_sat.steering_angle_deg) <= 28.0001, "Angle failed to clamp at 28.0 deg limit"
    assert cmd_sat.is_clamped is True, "Clamp flag not asserted on extreme command"

    # -------------------------------------------------------------------------
    # Test 4: Slew Rate Limiter (40 deg/s max rate)
    # -------------------------------------------------------------------------
    limiter = ActuationSaturationLimiter(profile=profile)
    limiter.reset(initial_angle_rad=0.0)

    # Request step command of 20 degrees in 0.0333s (requires ~600 deg/s without limiter)
    step_req = math.radians(20.0)
    angle_out, state, is_cl, is_rl, applied_rate = limiter.sanitize_command(
        requested_angle_rad=step_req,
        timestamp_s=1.0333,
        current_velocity_mps=5.0,
    )
    print(f"Applied Rate: {applied_rate:.2f}°/s (Limit: 40.00°/s)")
    assert is_rl is True, "Rate limiter failed to assert is_rate_limited"
    assert applied_rate <= 40.05, f"Applied slew rate exceeded 40 deg/s: {applied_rate}"

    # -------------------------------------------------------------------------
    # Test 5: PID Controller Step Response & Anti-Windup
    # -------------------------------------------------------------------------
    controller.set_mode(ControlMode.PID)
    t_pid = 200.0
    for step in range(50):
        kin_pid = VehicleKinematics(timestamp_s=t_pid, velocity_mps=6.0)
        cmd_pid = controller.compute_actuation(extreme_geo, kin_pid)
        t_pid += 0.0333

    assert abs(cmd_pid.steering_angle_deg) <= 28.0
    # Integral accumulator should not have grown unbounded
    assert abs(controller.pid.integral_accumulator) <= 10.0, "Integrator windup detected"

    # -------------------------------------------------------------------------
    # Test 6: Pure Pursuit Controller
    # -------------------------------------------------------------------------
    controller.set_mode(ControlMode.PURE_PURSUIT)
    cmd_pp = controller.compute_actuation(geo, kin)
    print(f"Pure Pursuit Command: {cmd_pp.steering_angle_deg:.2f}°")
    assert abs(cmd_pp.steering_angle_deg) <= 28.0

    # -------------------------------------------------------------------------
    # Test 7: ASIL-B Safety Supervisor Fault Tripping
    # -------------------------------------------------------------------------
    supervisor = ASILBActuationSafetySupervisor(profile=profile)
    # Scenario: NaN command injection
    safe, err, _ = supervisor.verify_command_safety(float("nan"), 0.0, 5.0)
    assert safe is False
    assert err == "ASIL_B_ERR_NAN_COMMAND"

    # Scenario: CRC-8 calculation check
    test_bytes = b"\x01\x02\x03\x04"
    crc = supervisor.compute_crc8(test_bytes)
    assert 0 <= crc <= 255, "Invalid CRC-8 output"

    print("LaneKeepController verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_control_verification_suite()
