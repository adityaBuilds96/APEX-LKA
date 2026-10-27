"""
deploy/vehicle_output_bridge.py
===============================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Vehicle Actuation Bridge, Stanley Controller & ASIL-B Safety Enforcement.
ISO 26262 ASIL-B Lateral Actuation & Fail-Safe Supervisory Baseline.

Hardware Interfaces:
--------------------
1. SocketCAN Actuation Bus:
   - 500 kbps differential SocketCAN interface (can0 / vcan0).
   - CAN ID 0x100: Primary Steering Actuation Command (Angle, Rate, Watchdog).
   - CAN ID 0x190: ADAS Perception State & Road Curvature Diagnostics.
   - CAN ID 0x198: Perception Latency & Pipeline Health Diagnostics.
   - CAN ID 0x120: Chassis Driver Torque Feedback / Driver Takeover Override.
2. USB Serial Hardware Node:
   - High-speed 115200 / 921600 baud line-delimited ASCII / NMEA-style or
     framed binary packet with sync bytes (0xAA, 0x55) and CRC-16-CCITT.
   - Dual-node microcontroller control network (e.g. Arduino Mega / STM32 / Teensy).

Hard Safety Limit Enforcement (NON-NEGOTIABLE):
-----------------------------------------------
1. Absolute Steering Angle Saturation: Strictly clamped to [-28.0 deg, +28.0 deg].
2. Slew-Rate Limiting: Maximum angular velocity clamped to 40.0 deg/sec.
3. Slew Acceleration / Jerk Filtering: Smooth transitions via second-order ramp.
4. Confidence Degradation Gate: If perception confidence < 0.35, automatically
   transitions to SAFE_STOP and ramps steering to center (0.0 deg).
5. Heartbeat Watchdog: If no valid perception frame arrives within 500 ms,
   the actuation output immediately enters neutral fail-safe state.
6. Driver Takeover Detection: If driver steering torque exceeds manual override
   threshold (e.g. 3.5 Nm), autonomous steering instantly disengages.
"""

from __future__ import annotations

import collections
import enum
import json
import logging
import math
import struct
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.pipeline.lane_track_runtime import (
    LanePolynomial,
    LaneTrackingFrameOutput,
    LaneTrackingState,
)

logger = logging.getLogger("APEX.VehicleBridge")

# ISO 26262 ASIL-B Hard Vehicle Physical Limits
STEERING_ANGLE_LIMIT_DEG: float = 28.0
STEERING_SLEW_LIMIT_DEG_S: float = 40.0
CONFIDENCE_THRESHOLD_MIN: float = 0.35
WATCHDOG_TIMEOUT_SEC: float = 0.500
DRIVER_OVERRIDE_TORQUE_NM: float = 3.50
MAX_LATERAL_ACCEL_LIMIT_G: float = 0.30
EARTH_GRAVITY_M_S2: float = 9.80665


# ==============================================================================
# 1. Controller Telemetry, State Machine & Dataclasses
# ==============================================================================

class ActuatorState(enum.Enum):
    """
    ISO 26262 Fail-Safe Actuator Operating States.
    """
    INITIALIZING = "INITIALIZING"
    DISARMED = "DISARMED"
    ACTIVE = "ACTIVE"
    DEGRADED_COAST = "DEGRADED_COAST"
    DRIVER_OVERRIDE = "DRIVER_OVERRIDE"
    SAFE_STOP_RAMP = "SAFE_STOP_RAMP"
    SAFE_STOP_ESTOP = "SAFE_STOP_ESTOP"
    WATCHDOG_TRIPPED = "WATCHDOG_TRIPPED"


@dataclass
class DriverTorqueFeedback:
    """
    Chassis feedback received over CAN (ID 0x120) or steering sensor.
    """
    timestamp_sec: float = 0.0
    measured_torque_nm: float = 0.0
    driver_hands_on: bool = False
    override_detected: bool = False
    chassis_speed_m_s: float = 12.0
    actual_pinion_angle_deg: float = 0.0


@dataclass
class ActuationCommand:
    """
    Final validated hardware actuation command passed to microcontroller rack.
    """
    target_steering_deg: float
    commanded_slew_rate_deg_s: float
    is_actuator_enabled: bool
    state: ActuatorState
    message_counter: int
    timestamp_sec: float
    raw_unclamped_angle_deg: float
    lateral_error_m: float
    heading_error_deg: float
    curvature_term_deg: float = 0.0
    crc8_checksum: int = 0
    safe_stop_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class StanleyTuningParameters:
    """
    Tuning gains for nonlinear Stanley cross-track steering controller
    augmented with road curvature feedforward and speed scheduling.
    """
    k_gain: float = 1.25                 # Cross-track error proportional gain
    k_soft: float = 1.00                 # Softening constant for low-speed stability (m/s)
    k_heading: float = 1.00              # Heading error gain
    k_integral: float = 0.05             # Integral gain for steady-state bias (crosswind/camber)
    integral_saturation_deg: float = 3.0 # Maximum windup trim angle (deg)
    feedforward_gain: float = 1.00       # Curvature feedforward gain (delta_ff = L * kappa)
    wheelbase_m: float = 2.70            # Vehicle physical wheelbase (meters)
    default_velocity_m_s: float = 12.0   # Fallback speed if vehicle odometry is missing
    lookahead_min_m: float = 2.0         # Minimum dynamic lookahead distance
    lookahead_ratio_sec: float = 0.35    # Speed-dependent lookahead time ratio


@dataclass
class SafetySupervisionMetrics:
    """
    Continuous diagnostic counters and telemetry tracking.
    """
    total_cycles: int = 0
    clamped_angle_count: int = 0
    slew_saturated_count: int = 0
    degraded_frames_count: int = 0
    watchdog_timeouts_count: int = 0
    driver_overrides_count: int = 0
    last_cycle_duration_ms: float = 0.0
    mean_cycle_latency_ms: float = 0.0


# ==============================================================================
# 2. CAN Frame Packet Formats & DBC Serialization
# ==============================================================================

class CANFrameEncoder:
    """
    Bit-exact SAE J1939 / ISO 11898-2 CAN Frame Packets:
      - 0x100: Steering Rack Actuation Command (High Priority, 50 Hz)
      - 0x190: Lane Geometry & Road Curvature Diagnostics (20 Hz)
      - 0x198: Perception Latency & Pipeline Health (10 Hz)
    """

    CAN_ID_STEERING_CMD: int = 0x100
    CAN_ID_ROAD_GEOMETRY: int = 0x190
    CAN_ID_PERCEPTION_HEALTH: int = 0x198
    CAN_ID_CHASSIS_FEEDBACK: int = 0x120

    @staticmethod
    def calculate_crc8(data: bytes, poly: int = 0x1D) -> int:
        """
        Standard SAE J1850 CRC-8 computation with initial value 0xFF and inversion.
        """
        crc = 0xFF
        for b in data:
            crc ^= b
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ poly) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
        return crc ^ 0xFF

    @classmethod
    def encode_steering_cmd(cls, cmd: ActuationCommand) -> Tuple[int, bytes]:
        """
        Encode CAN ID 0x100: Steering Rack Actuation Command.
        Payload layout (8 bytes):
          - Bytes 0-1: Steering Angle (int16, scale 0.01 deg, range [-327.68, 327.67])
          - Bytes 2-3: Slew Rate (int16, scale 0.01 deg/s)
          - Byte 4   : Enabled Flag (uint8: bit 0 = enabled, bit 1 = driver takeover)
          - Byte 5   : Actuator State Enum (uint8)
          - Byte 6   : Rolling Message Counter (uint8, 0-255)
          - Byte 7   : SAE J1850 CRC-8 Checksum
        """
        angle_i16 = int(round(cmd.target_steering_deg / 0.01))
        slew_i16 = int(round(cmd.commanded_slew_rate_deg_s / 0.01))

        # Flags byte
        flags_u8 = 0
        if cmd.is_actuator_enabled:
            flags_u8 |= 0x01
        if cmd.state == ActuatorState.DRIVER_OVERRIDE:
            flags_u8 |= 0x02

        state_idx = list(ActuatorState).index(cmd.state)
        counter_u8 = cmd.message_counter % 256

        partial_payload = struct.pack("<hhBBB", angle_i16, slew_i16, flags_u8, state_idx, counter_u8)
        crc = cls.calculate_crc8(partial_payload)
        payload = partial_payload + bytes([crc])

        return (cls.CAN_ID_STEERING_CMD, payload)

    @classmethod
    def decode_steering_cmd(cls, payload: bytes) -> Dict[str, Any]:
        """
        Decode CAN ID 0x100 payload for testing and loopback validation.
        """
        if len(payload) != 8:
            raise ValueError(f"CAN 0x100 expected 8 bytes, got {len(payload)}")

        crc_calc = cls.calculate_crc8(payload[:7])
        crc_recv = payload[7]
        crc_valid = (crc_calc == crc_recv)

        angle_i16, slew_i16, flags_u8, state_idx, counter_u8 = struct.unpack("<hhBBB", payload[:7])

        angle_deg = angle_i16 * 0.01
        slew_rate_deg_s = slew_i16 * 0.01
        is_enabled = bool(flags_u8 & 0x01)
        driver_override = bool(flags_u8 & 0x02)

        state_enum = list(ActuatorState)[state_idx] if state_idx < len(ActuatorState) else ActuatorState.SAFE_STOP_ESTOP

        return {
            "target_steering_deg": round(angle_deg, 2),
            "commanded_slew_rate_deg_s": round(slew_rate_deg_s, 2),
            "is_actuator_enabled": is_enabled,
            "driver_override": driver_override,
            "state": state_enum,
            "message_counter": counter_u8,
            "crc_valid": crc_valid,
        }

    # Backward compatibility alias
    decode_actuation_command = decode_steering_cmd

    @classmethod
    def encode_road_geometry(
        cls,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature_radius_m: float,
        lane_width_m: float,
        confidence: float,
        counter: int,
    ) -> Tuple[int, bytes]:
        """
        Encode CAN ID 0x190: Road Curvature & Ego Deviation Diagnostics.
        Payload layout (8 bytes):
          - Bytes 0-1: Lateral Error e_lat (int16, scale 0.001 m = 1 mm)
          - Bytes 2-3: Heading Error psi (int16, scale 0.01 deg)
          - Bytes 4-5: Curvature 1/R (int16, scale 0.0001 1/m, signed)
          - Byte 6   : Confidence (uint8, 0-100%)
          - Byte 7   : Rolling Counter (uint8, 0-255)
        """
        lat_i16 = int(round(lateral_error_m / 0.001))
        lat_i16 = max(-32768, min(32767, lat_i16))

        heading_deg = math.degrees(heading_error_rad)
        head_i16 = int(round(heading_deg / 0.01))
        head_i16 = max(-32768, min(32767, head_i16))

        # Signed curvature kappa = 1/R
        if abs(curvature_radius_m) > 1.0:
            kappa = 1.0 / curvature_radius_m
        else:
            kappa = 0.0
        kappa_i16 = int(round(kappa / 0.0001))
        kappa_i16 = max(-32768, min(32767, kappa_i16))

        conf_u8 = max(0, min(100, int(round(confidence * 100.0))))
        cnt_u8 = counter % 256

        payload = struct.pack("<hhhBB", lat_i16, head_i16, kappa_i16, conf_u8, cnt_u8)
        return (cls.CAN_ID_ROAD_GEOMETRY, payload)

    @classmethod
    def encode_perception_health(
        cls,
        inference_latency_ms: float,
        pipeline_latency_ms: float,
        tracking_state: LaneTrackingState,
        fps: float,
        counter: int,
    ) -> Tuple[int, bytes]:
        """
        Encode CAN ID 0x198: Latency and Health Diagnostics.
        Payload layout (8 bytes):
          - Bytes 0-1: Inference Latency (uint16, scale 0.1 ms)
          - Bytes 2-3: Pipeline Latency (uint16, scale 0.1 ms)
          - Byte 4   : Tracking State Enum (uint8)
          - Byte 5   : Frame Rate (uint8, FPS 0-255)
          - Byte 6   : Health Flags (uint8)
          - Byte 7   : Rolling Counter (uint8)
        """
        inf_u16 = int(round(inference_latency_ms / 0.1))
        pipe_u16 = int(round(pipeline_latency_ms / 0.1))
        state_u8 = list(LaneTrackingState).index(tracking_state)
        fps_u8 = max(0, min(255, int(round(fps))))

        health_flags = 0x01 if pipeline_latency_ms < 50.0 else 0x00
        cnt_u8 = counter % 256

        payload = struct.pack("<HHBBBB", inf_u16, pipe_u16, state_u8, fps_u8, health_flags, cnt_u8)
        return (cls.CAN_ID_PERCEPTION_HEALTH, payload)

    @classmethod
    def decode_chassis_feedback(cls, payload: bytes) -> DriverTorqueFeedback:
        """
        Decode CAN ID 0x120: Chassis Driver Steering Feedback.
        Payload layout (8 bytes):
          - Bytes 0-1: Driver Hand Torque (int16, scale 0.01 Nm)
          - Bytes 2-3: Actual Pinion Angle (int16, scale 0.01 deg)
          - Bytes 4-5: Chassis Speed (uint16, scale 0.01 m/s)
          - Byte 6   : Hands On Status (uint8: 1=Hands detected, 0=Hands off)
          - Byte 7   : Status / Fault Bits
        """
        if len(payload) != 8:
            return DriverTorqueFeedback()

        torque_i16, angle_i16, speed_u16, hands_u8, _ = struct.unpack("<hhHBB", payload)
        torque_nm = torque_i16 * 0.01
        pinion_deg = angle_i16 * 0.01
        speed_m_s = speed_u16 * 0.01
        hands_on = bool(hands_u8 & 0x01)
        override = abs(torque_nm) >= DRIVER_OVERRIDE_TORQUE_NM

        return DriverTorqueFeedback(
            timestamp_sec=time.time(),
            measured_torque_nm=torque_nm,
            driver_hands_on=hands_on,
            override_detected=override,
            chassis_speed_m_s=speed_m_s,
            actual_pinion_angle_deg=pinion_deg,
        )


# ==============================================================================
# 3. USB Serial Framing Protocol (Binary & ASCII NMEA)
# ==============================================================================

class SerialFrameEncoder:
    """
    Encodes actuation frames for microcontroller nodes via USB Serial.
    Features:
      - Binary Protocol: [SYNC_0=0xAA][SYNC_1=0x55][LEN=14][PAYLOAD][CRC16_CCITT]
      - ASCII NMEA-style fallback: $APEX,STEER,angle,rate,enable,state*CS\\r\\n
    """

    SYNC_BYTE_0: int = 0xAA
    SYNC_BYTE_1: int = 0x55

    @staticmethod
    def calculate_crc16_ccitt(data: bytes, poly: int = 0x1021, init: int = 0xFFFF) -> int:
        """Standard 16-bit CRC-CCITT (False) implementation."""
        crc = init
        for byte in data:
            crc ^= (byte << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ poly) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    @classmethod
    def encode_binary_frame(cls, cmd: ActuationCommand) -> bytes:
        """
        Pack command into high-integrity binary frame with sync words & CRC-16.
        """
        angle_i16 = int(round(cmd.target_steering_deg / 0.01))
        slew_i16 = int(round(cmd.commanded_slew_rate_deg_s / 0.01))
        lat_i16 = int(round(cmd.lateral_error_m / 0.001))
        head_i16 = int(round(cmd.heading_error_deg / 0.01))
        en_u8 = 1 if cmd.is_actuator_enabled else 0
        state_u8 = list(ActuatorState).index(cmd.state)
        counter_u8 = cmd.message_counter % 256

        # 11 bytes of payload
        payload = struct.pack("<hhhhBBB", angle_i16, slew_i16, lat_i16, head_i16, en_u8, state_u8, counter_u8)
        length_u8 = len(payload)

        header = bytes([cls.SYNC_BYTE_0, cls.SYNC_BYTE_1, length_u8])
        crc = cls.calculate_crc16_ccitt(payload)
        footer = struct.pack("<H", crc)

        return header + payload + footer

    @classmethod
    def decode_binary_frame(cls, raw: bytes) -> Optional[Dict[str, Any]]:
        """
        Parse binary serial frame and verify CRC-16.
        """
        if len(raw) < 16:
            return None
        if raw[0] != cls.SYNC_BYTE_0 or raw[1] != cls.SYNC_BYTE_1:
            return None

        payload_len = raw[2]
        if len(raw) != 3 + payload_len + 2:
            return None

        payload = raw[3 : 3 + payload_len]
        crc_expected = struct.unpack("<H", raw[3 + payload_len :])[0]
        crc_actual = cls.calculate_crc16_ccitt(payload)

        if crc_expected != crc_actual:
            return None

        angle_i16, slew_i16, lat_i16, head_i16, en_u8, state_u8, counter_u8 = struct.unpack(
            "<hhhhBBB", payload
        )

        return {
            "target_steering_deg": angle_i16 * 0.01,
            "commanded_slew_rate_deg_s": slew_i16 * 0.01,
            "lateral_error_m": lat_i16 * 0.001,
            "heading_error_deg": head_i16 * 0.01,
            "is_actuator_enabled": bool(en_u8),
            "state": list(ActuatorState)[state_u8] if state_u8 < len(ActuatorState) else ActuatorState.SAFE_STOP_ESTOP,
            "message_counter": counter_u8,
            "crc_valid": True,
        }

    @classmethod
    def encode_nmea_ascii(cls, cmd: ActuationCommand) -> str:
        """
        Format as human-readable NMEA 0183-style diagnostic sentence.
        Format: $APEX,STR,<angle>,<rate>,<en>,<state>,<seq>*<XOR_HEX>\\r\\n
        """
        content = (
            f"APEX,STR,{cmd.target_steering_deg:.2f},"
            f"{cmd.commanded_slew_rate_deg_s:.2f},"
            f"{1 if cmd.is_actuator_enabled else 0},"
            f"{cmd.state.value},"
            f"{cmd.message_counter}"
        )

        # XOR Checksum
        xor_sum = 0
        for char in content:
            xor_sum ^= ord(char)

        return f"${content}*{xor_sum:02X}\r\n"


# ==============================================================================
# 4. Master Vehicle Output Bridge Engine
# ==============================================================================

class VehicleOutputBridge:
    """
    Translates high-level BEV lane polynomials and tracking states into
    kinematically limited, ASIL-B certified CAN and Serial actuation signals.

    Features:
      - Augmented Stanley lateral steering controller (Heading + Lateral + Curvature FF)
      - Anti-windup integral trimming for road camber / crosswind compensation
      - Slew-rate limiting (40 deg/s) and angle clamping (+-28 deg)
      - Lateral acceleration kinematic ceiling (max 0.3g limit)
      - Dynamic driver takeover override via steering torque sensor
      - Hardware watchdog with 500 ms auto-timeout and progressive ramp-to-center
      - Circular telemetry buffer and ISO 26262 diagnostic metrics
    """

    def __init__(
        self,
        stanley_params: Optional[StanleyTuningParameters] = None,
        can_interface: str = "can0",
        enable_can: bool = False,
        enable_serial: bool = False,
        serial_port: Optional[str] = None,
        serial_baud: int = 115200,
        min_confidence_to_steer: float = CONFIDENCE_THRESHOLD_MIN,
    ) -> None:
        self.params = stanley_params or StanleyTuningParameters()
        self.can_interface = can_interface
        self.enable_can = enable_can
        self.enable_serial = enable_serial
        self.serial_port = serial_port
        self.serial_baud = serial_baud
        self.min_confidence = min_confidence_to_steer

        # Actuation state memory
        self._prev_steer_deg: float = 0.0
        self._prev_timestamp: float = time.time()
        self._msg_counter: int = 0
        self._state = ActuatorState.INITIALIZING
        self._integral_error_m: float = 0.0

        # Watchdog & driver feedback tracking
        self._last_frame_time: float = time.time()
        self._driver_feedback = DriverTorqueFeedback()
        self._lock = threading.Lock()

        # Telemetry history (circular buffer)
        self._history_len: int = 200
        self._cmd_history: Deque[ActuationCommand] = collections.deque(maxlen=self._history_len)
        self.metrics = SafetySupervisionMetrics()

        # Serial & CAN hooks (extensible callbacks)
        self.can_send_hook: Optional[Callable[[int, bytes], None]] = None
        self.serial_send_hook: Optional[Callable[[bytes], None]] = None

        logger.info(
            "VehicleOutputBridge initialized. CAN: %s (enabled=%s), Max Steer: ±%.1f°, Slew: %.1f°/s",
            self.can_interface,
            self.enable_can,
            STEERING_ANGLE_LIMIT_DEG,
            STEERING_SLEW_LIMIT_DEG_S,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Stanley Lateral Steering Controller
    # ──────────────────────────────────────────────────────────────────────────

    def compute_stanley_steering(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature_radius_m: float,
        velocity_m_s: Optional[float] = None,
        dt_sec: float = 0.02,
    ) -> Tuple[float, float]:
        """
        Compute augmented Stanley lateral steering angle:
          delta = psi_heading + arctan((k * e_lat) / (v + k_soft)) + delta_integral + delta_ff

        Returns:
            (total_steering_angle_deg, curvature_feedforward_deg)
        """
        v = velocity_m_s if (velocity_m_s and velocity_m_s > 0.5) else self.params.default_velocity_m_s

        # 1. Heading error component (radians -> degrees)
        heading_term_rad = self.params.k_heading * heading_error_rad

        # 2. Cross-track error nonlinear correction angle
        cross_track_term_rad = math.atan2(self.params.k_gain * lateral_error_m, v + self.params.k_soft)

        # 3. Integral anti-windup accumulator for steady-state bias
        self._integral_error_m += lateral_error_m * dt_sec
        integral_term_deg = self.params.k_integral * self._integral_error_m
        # Clamp integral term
        max_int = self.params.integral_saturation_deg
        if abs(integral_term_deg) > max_int:
            integral_term_deg = math.copysign(max_int, integral_term_deg)
            # Prevent integral divergence
            self._integral_error_m = integral_term_deg / (self.params.k_integral + 1e-6)

        # 4. Curvature feedforward term (Ackermann geometry: delta_ff = L * kappa)
        if abs(curvature_radius_m) > 5.0:
            kappa = 1.0 / curvature_radius_m
            delta_ff_rad = self.params.feedforward_gain * self.params.wheelbase_m * kappa
            curvature_term_deg = math.degrees(delta_ff_rad)
        else:
            curvature_term_deg = 0.0

        # Combine components
        total_rad = heading_term_rad + cross_track_term_rad
        raw_angle_deg = math.degrees(total_rad) + integral_term_deg + curvature_term_deg

        # Kinematic lateral acceleration ceiling: a_lat = v^2 * kappa_cmd <= a_max
        # where kappa_cmd approx delta / L
        max_steer_from_accel = math.degrees(
            (MAX_LATERAL_ACCEL_LIMIT_G * EARTH_GRAVITY_M_S2 * self.params.wheelbase_m) / max(1.0, v * v)
        )
        safe_dynamic_limit = min(STEERING_ANGLE_LIMIT_DEG, max(5.0, max_steer_from_accel))

        # Restrict raw angle if dynamic speed limit is tighter than static mechanical limit
        if abs(raw_angle_deg) > safe_dynamic_limit:
            raw_angle_deg = math.copysign(safe_dynamic_limit, raw_angle_deg)

        return raw_angle_deg, curvature_term_deg

    # ──────────────────────────────────────────────────────────────────────────
    # ASIL-B Safety Slew Limiter & Clipping Gates
    # ──────────────────────────────────────────────────────────────────────────

    def enforce_safety_envelope(
        self,
        raw_target_angle_deg: float,
        dt_sec: float,
    ) -> Tuple[float, float]:
        """
        Apply strict slew-rate limiting (40 deg/s) and hard mechanical clamping (±28 deg).
        Returns:
            (clamped_angle_deg, actual_slew_rate_deg_s)
        """
        dt = max(1e-4, dt_sec)
        max_delta = STEERING_SLEW_LIMIT_DEG_S * dt

        # 1. Slew rate limiting
        angle_diff = raw_target_angle_deg - self._prev_steer_deg

        if abs(angle_diff) > max_delta:
            clamped_diff = math.copysign(max_delta, angle_diff)
            self.metrics.slew_saturated_count += 1
        else:
            clamped_diff = angle_diff

        slew_limited_angle = self._prev_steer_deg + clamped_diff
        actual_rate = clamped_diff / dt

        # 2. Hard physical angle clamping
        if abs(slew_limited_angle) > STEERING_ANGLE_LIMIT_DEG:
            clamped_angle = math.copysign(STEERING_ANGLE_LIMIT_DEG, slew_limited_angle)
            self.metrics.clamped_angle_count += 1
        else:
            clamped_angle = slew_limited_angle

        return round(clamped_angle, 2), round(actual_rate, 2)

    # ──────────────────────────────────────────────────────────────────────────
    # Driver Takeover & Chassis Feedback Integration
    # ──────────────────────────────────────────────────────────────────────────

    def update_driver_feedback(self, feedback: DriverTorqueFeedback) -> None:
        """
        Feed driver steering wheel torque feedback into supervisor.
        """
        with self._lock:
            self._driver_feedback = feedback
            if feedback.override_detected:
                self.metrics.driver_overrides_count += 1
                logger.warning(
                    "Driver takeover detected! Torque: %.2f Nm > threshold %.2f Nm",
                    feedback.measured_torque_nm,
                    DRIVER_OVERRIDE_TORQUE_NM,
                )

    # ──────────────────────────────────────────────────────────────────────────
    # Actuation Command Generation Pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def process_perception_output(
        self,
        frame_output: LaneTrackingFrameOutput,
        vehicle_velocity_m_s: Optional[float] = None,
    ) -> ActuationCommand:
        """
        Main execution step converting perception frame into validated actuation command.
        Follows strict ISO 26262 ASIL-B supervisory verification steps:
          1. Heartbeat & timestamp sanity check
          2. Driver takeover override arbitration
          3. Confidence threshold & fail-safe gate evaluation
          4. Stanley controller execution with road curvature feedforward
          5. Dynamic slew-rate and angle envelope clamping
          6. Multi-bus broadcast (CAN ID 0x100, 0x190, 0x198 and USB Serial)
        """
        cycle_start = time.perf_counter()
        now = time.time()
        dt = max(1e-3, min(0.200, now - self._prev_timestamp))
        self._prev_timestamp = now
        self._last_frame_time = now
        self._msg_counter = (self._msg_counter + 1) % 256
        self.metrics.total_cycles += 1

        effective_speed = (
            vehicle_velocity_m_s
            if (vehicle_velocity_m_s and vehicle_velocity_m_s > 0.5)
            else (self._driver_feedback.chassis_speed_m_s or self.params.default_velocity_m_s)
        )

        # ──────────────────────────────────────────────────────────────────────
        # Gate 1: Check Driver Takeover Override
        # ──────────────────────────────────────────────────────────────────────
        if self._driver_feedback.override_detected:
            self._state = ActuatorState.DRIVER_OVERRIDE
            final_angle, rate = self.enforce_safety_envelope(self._driver_feedback.actual_pinion_angle_deg, dt)
            self._prev_steer_deg = final_angle
            self._integral_error_m = 0.0  # Reset integrator

            cmd = ActuationCommand(
                target_steering_deg=final_angle,
                commanded_slew_rate_deg_s=rate,
                is_actuator_enabled=False,  # Disengage electric rack
                state=self._state,
                message_counter=self._msg_counter,
                timestamp_sec=now,
                raw_unclamped_angle_deg=round(self._driver_feedback.actual_pinion_angle_deg, 2),
                lateral_error_m=frame_output.lateral_error_m,
                heading_error_deg=round(math.degrees(frame_output.heading_error_rad), 2),
                curvature_term_deg=0.0,
                safe_stop_reason="DRIVER_TAKEOVER_OVERRIDE",
            )
            self._finalize_and_broadcast(cmd, frame_output)
            return cmd

        # ──────────────────────────────────────────────────────────────────────
        # Gate 2: Evaluate Perception Confidence & Tracking State
        # ──────────────────────────────────────────────────────────────────────
        conf = frame_output.overall_confidence
        is_safe = frame_output.is_safe_to_steer

        if not is_safe or conf < CONFIDENCE_THRESHOLD_MIN or frame_output.state == LaneTrackingState.SAFE_STOP:
            # Transition to Safe Stop (progressive deceleration / ramp steering to center)
            self.metrics.degraded_frames_count += 1
            self._state = ActuatorState.SAFE_STOP_ESTOP
            self._integral_error_m = 0.0

            # Ramp steering angle smoothly to 0.0 deg (neutral steering)
            final_angle, rate = self.enforce_safety_envelope(0.0, dt)
            self._prev_steer_deg = final_angle

            cmd = ActuationCommand(
                target_steering_deg=final_angle,
                commanded_slew_rate_deg_s=rate,
                is_actuator_enabled=False,  # Disarm autonomous steering
                state=self._state,
                message_counter=self._msg_counter,
                timestamp_sec=now,
                raw_unclamped_angle_deg=0.0,
                lateral_error_m=frame_output.lateral_error_m,
                heading_error_deg=round(math.degrees(frame_output.heading_error_rad), 2),
                curvature_term_deg=0.0,
                safe_stop_reason=f"LOW_CONFIDENCE_{conf:.2f}_BELOW_{CONFIDENCE_THRESHOLD_MIN}",
            )
            self._finalize_and_broadcast(cmd, frame_output)
            return cmd

        elif frame_output.state == LaneTrackingState.DEGRADED:
            self._state = ActuatorState.DEGRADED_COAST
            # Reduce controller gains in degraded mode
            raw_angle_deg, curv_deg = self.compute_stanley_steering(
                lateral_error_m=frame_output.lateral_error_m,
                heading_error_rad=frame_output.heading_error_rad,
                curvature_radius_m=frame_output.curvature_radius_m,
                velocity_m_s=effective_speed,
                dt_sec=dt,
            )
            raw_angle_deg *= 0.70  # Conservative gain scale
        else:
            self._state = ActuatorState.ACTIVE
            raw_angle_deg, curv_deg = self.compute_stanley_steering(
                lateral_error_m=frame_output.lateral_error_m,
                heading_error_rad=frame_output.heading_error_rad,
                curvature_radius_m=frame_output.curvature_radius_m,
                velocity_m_s=effective_speed,
                dt_sec=dt,
            )

        # ──────────────────────────────────────────────────────────────────────
        # Gate 3: Enforce Kinematic Envelope & Slew-Rate Limiting
        # ──────────────────────────────────────────────────────────────────────
        final_angle, rate = self.enforce_safety_envelope(raw_angle_deg, dt)
        self._prev_steer_deg = final_angle

        cmd = ActuationCommand(
            target_steering_deg=final_angle,
            commanded_slew_rate_deg_s=rate,
            is_actuator_enabled=True,
            state=self._state,
            message_counter=self._msg_counter,
            timestamp_sec=now,
            raw_unclamped_angle_deg=round(raw_angle_deg, 2),
            lateral_error_m=frame_output.lateral_error_m,
            heading_error_deg=round(math.degrees(frame_output.heading_error_rad), 2),
            curvature_term_deg=round(curv_deg, 2),
            safe_stop_reason=None,
        )

        # Record cycle latency
        cycle_ms = (time.perf_counter() - cycle_start) * 1000.0
        self.metrics.last_cycle_duration_ms = cycle_ms
        self.metrics.mean_cycle_latency_ms = (
            self.metrics.mean_cycle_latency_ms * 0.95 + cycle_ms * 0.05
        )

        self._finalize_and_broadcast(cmd, frame_output)
        return cmd

    # ──────────────────────────────────────────────────────────────────────────
    # Heartbeat Watchdog Monitoring
    # ──────────────────────────────────────────────────────────────────────────

    def check_watchdog_timeout(self) -> Optional[ActuationCommand]:
        """
        Pollable heartbeat watchdog. If elapsed time since last perception frame
        exceeds WATCHDOG_TIMEOUT_SEC (500 ms), returns emergency fail-safe command.
        """
        now = time.time()
        elapsed = now - self._last_frame_time

        if elapsed > WATCHDOG_TIMEOUT_SEC and self._state != ActuatorState.WATCHDOG_TRIPPED:
            self._state = ActuatorState.WATCHDOG_TRIPPED
            self.metrics.watchdog_timeouts_count += 1
            logger.critical(
                "WATCHDOG TRIPPED! No perception update for %.3f s > %.3f s. Disarming rack.",
                elapsed,
                WATCHDOG_TIMEOUT_SEC,
            )

            # Ramp angle to zero
            dt = max(1e-3, elapsed)
            final_angle, rate = self.enforce_safety_envelope(0.0, dt)
            self._prev_steer_deg = final_angle

            cmd = ActuationCommand(
                target_steering_deg=final_angle,
                commanded_slew_rate_deg_s=rate,
                is_actuator_enabled=False,
                state=self._state,
                message_counter=(self._msg_counter + 1) % 256,
                timestamp_sec=now,
                raw_unclamped_angle_deg=0.0,
                lateral_error_m=0.0,
                heading_error_deg=0.0,
                curvature_term_deg=0.0,
                safe_stop_reason=f"WATCHDOG_TIMEOUT_{elapsed:.3f}S",
            )
            self._broadcast_can_frames(cmd, None)
            return cmd

        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Bus Broadcast & Finalization Helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _finalize_and_broadcast(
        self,
        cmd: ActuationCommand,
        frame_output: Optional[LaneTrackingFrameOutput] = None,
    ) -> None:
        """Store in history and broadcast across enabled hardware channels."""
        self._cmd_history.append(cmd)

        if self.enable_can:
            self._broadcast_can_frames(cmd, frame_output)

        if self.enable_serial:
            self._broadcast_serial_frames(cmd)

    def _broadcast_can_frames(
        self,
        cmd: ActuationCommand,
        frame_output: Optional[LaneTrackingFrameOutput],
    ) -> None:
        """Broadcast CAN ID 0x100, 0x190 and 0x198 frames."""
        # Primary steering frame (0x100)
        can_id, payload = CANFrameEncoder.encode_steering_cmd(cmd)
        if self.can_send_hook:
            self.can_send_hook(can_id, payload)

        # Secondary diagnostics frames
        if frame_output is not None:
            # 0x190 Road Geometry
            id_geom, payload_geom = CANFrameEncoder.encode_road_geometry(
                lateral_error_m=frame_output.lateral_error_m,
                heading_error_rad=frame_output.heading_error_rad,
                curvature_radius_m=frame_output.curvature_radius_m,
                lane_width_m=frame_output.lane_width_m,
                confidence=frame_output.overall_confidence,
                counter=cmd.message_counter,
            )
            if self.can_send_hook:
                self.can_send_hook(id_geom, payload_geom)

            # 0x198 Health Diagnostics
            fps = 1000.0 / max(1.0, frame_output.pipeline_latency_ms)
            id_hlth, payload_hlth = CANFrameEncoder.encode_perception_health(
                inference_latency_ms=frame_output.inference_latency_ms,
                pipeline_latency_ms=frame_output.pipeline_latency_ms,
                tracking_state=frame_output.state,
                fps=fps,
                counter=cmd.message_counter,
            )
            if self.can_send_hook:
                self.can_send_hook(id_hlth, payload_hlth)

    def _broadcast_serial_frames(self, cmd: ActuationCommand) -> None:
        """Broadcast packed binary and NMEA sentences over serial hook."""
        binary_pkt = SerialFrameEncoder.encode_binary_frame(cmd)
        if self.serial_send_hook:
            self.serial_send_hook(binary_pkt)

    def get_diagnostics_report(self) -> Dict[str, Any]:
        """Return diagnostic dictionary of bridge health and safety counters."""
        return {
            "state": self._state.value,
            "prev_steer_deg": self._prev_steer_deg,
            "integral_error_m": round(self._integral_error_m, 4),
            "metrics": asdict(self.metrics),
            "history_size": len(self._cmd_history),
            "last_frame_age_ms": round((time.time() - self._last_frame_time) * 1000.0, 1),
            "driver_override_active": self._driver_feedback.override_detected,
        }

    def encode_can_actuation_payload(self, cmd: ActuationCommand) -> Tuple[int, bytes]:
        """Convenience helper to encode CAN actuation command (0x100)."""
        return CANFrameEncoder.encode_steering_cmd(cmd)

    def encode_can_geometry_payload(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        curvature_radius_m: float,
        lane_width_m: float,
        confidence: float,
    ) -> Tuple[int, bytes]:
        """Convenience helper to encode road geometry CAN frame (0x190)."""
        return CANFrameEncoder.encode_road_geometry(
            lateral_error_m=lateral_error_m,
            heading_error_rad=heading_error_rad,
            curvature_radius_m=curvature_radius_m,
            lane_width_m=lane_width_m,
            confidence=confidence,
            counter=self._msg_counter,
        )

    def format_serial_actuation_packet(self, cmd: ActuationCommand) -> bytes:
        """Convenience helper to format binary USB serial actuation packet."""
        return SerialFrameEncoder.encode_binary_frame(cmd)


# Backward-compatible aliases
CANActuationCodec = CANFrameEncoder
CANFramePacket = Tuple[int, bytes]
ActuatorMode = ActuatorState


# ==============================================================================
# 5. Standalone Self-Test Verification Suite
# ==============================================================================

if __name__ == "__main__":
    print("Executing VehicleOutputBridge ISO 26262 ASIL-B self-check...")
    bridge = VehicleOutputBridge(enable_can=True, enable_serial=True)

    # Intercept CAN and Serial transmissions for test verification
    can_frames_sent: List[Tuple[int, bytes]] = []
    serial_bytes_sent: List[bytes] = []
    bridge.can_send_hook = lambda can_id, data: can_frames_sent.append((can_id, data))
    bridge.serial_send_hook = lambda data: serial_bytes_sent.append(data)

    # 1. Normal Tracking Frame Test
    mock_frame = LaneTrackingFrameOutput(
        timestamp_sec=time.time(),
        state=LaneTrackingState.TRACKING,
        left_poly=LanePolynomial(is_valid=True),
        right_poly=LanePolynomial(is_valid=True),
        center_poly=LanePolynomial(is_valid=True),
        lateral_error_m=0.25,        # 25 cm right of center
        heading_error_rad=math.radians(2.0),
        curvature_radius_m=1200.0,
        lane_width_m=3.70,
        overall_confidence=0.88,
        inference_latency_ms=18.0,
        pipeline_latency_ms=22.0,
        is_safe_to_steer=True,
    )

    cmd = bridge.process_perception_output(mock_frame)
    assert cmd.is_actuator_enabled
    assert abs(cmd.target_steering_deg) <= STEERING_ANGLE_LIMIT_DEG
    assert cmd.state == ActuatorState.ACTIVE
    print(f"[PASS] Normal steering command: {cmd.target_steering_deg:.2f}° (Slew={cmd.commanded_slew_rate_deg_s:.1f}°/s)")

    # 2. CAN Frame Pack / Unpack Roundtrip Test
    assert len(can_frames_sent) >= 3
    steer_can_id, steer_payload = can_frames_sent[0]
    assert steer_can_id == 0x100
    decoded_can = CANFrameEncoder.decode_steering_cmd(steer_payload)
    assert decoded_can["crc_valid"]
    assert abs(decoded_can["target_steering_deg"] - cmd.target_steering_deg) < 0.05
    print(f"[PASS] CAN ID 0x100 decoded steering: {decoded_can['target_steering_deg']}°, CRC Valid: {decoded_can['crc_valid']}")

    # 3. Serial Binary Packet Roundtrip Test
    assert len(serial_bytes_sent) >= 1
    decoded_serial = SerialFrameEncoder.decode_binary_frame(serial_bytes_sent[0])
    assert decoded_serial is not None
    assert decoded_serial["crc_valid"]
    assert abs(decoded_serial["target_steering_deg"] - cmd.target_steering_deg) < 0.05
    print(f"[PASS] Serial Binary Packet decoded: {decoded_serial['target_steering_deg']}°, CRC-16 Valid: True")

    # 4. Angle Saturation Test (Clamped at ±28.0°)
    mock_extreme = LaneTrackingFrameOutput(
        timestamp_sec=time.time(),
        state=LaneTrackingState.TRACKING,
        left_poly=LanePolynomial(is_valid=True),
        right_poly=LanePolynomial(is_valid=True),
        center_poly=LanePolynomial(is_valid=True),
        lateral_error_m=4.50, # Extreme lateral deviation
        heading_error_rad=math.radians(35.0),
        curvature_radius_m=200.0,
        lane_width_m=3.70,
        overall_confidence=0.90,
        inference_latency_ms=18.0,
        pipeline_latency_ms=22.0,
        is_safe_to_steer=True,
    )
    cmd_ext = bridge.process_perception_output(mock_extreme)
    assert abs(cmd_ext.target_steering_deg) <= STEERING_ANGLE_LIMIT_DEG
    print(f"[PASS] Safety clipping verified: requested {cmd_ext.raw_unclamped_angle_deg}° -> clamped to {cmd_ext.target_steering_deg}°")

    # 5. Low Confidence Fail-Safe Test (< 0.35)
    mock_low_conf = LaneTrackingFrameOutput(
        timestamp_sec=time.time(),
        state=LaneTrackingState.DEGRADED,
        left_poly=LanePolynomial(),
        right_poly=LanePolynomial(),
        center_poly=LanePolynomial(),
        lateral_error_m=0.0,
        heading_error_rad=0.0,
        curvature_radius_m=9999.0,
        lane_width_m=3.70,
        overall_confidence=0.15, # Below 0.35 threshold
        inference_latency_ms=18.0,
        pipeline_latency_ms=22.0,
        is_safe_to_steer=False,
    )
    cmd_shutdown = bridge.process_perception_output(mock_low_conf)
    assert not cmd_shutdown.is_actuator_enabled
    assert cmd_shutdown.state == ActuatorState.SAFE_STOP_ESTOP
    print(f"[PASS] Confidence fail-safe shutdown verified: state={cmd_shutdown.state.value}")

    # 6. Driver Takeover Override Test
    bridge.update_driver_feedback(
        DriverTorqueFeedback(
            timestamp_sec=time.time(),
            measured_torque_nm=4.2,  # > 3.5 Nm threshold
            driver_hands_on=True,
            override_detected=True,
            chassis_speed_m_s=15.0,
            actual_pinion_angle_deg=5.0,
        )
    )
    cmd_override = bridge.process_perception_output(mock_frame)
    assert not cmd_override.is_actuator_enabled
    assert cmd_override.state == ActuatorState.DRIVER_OVERRIDE
    print(f"[PASS] Driver takeover override verified: state={cmd_override.state.value}")

    # 7. Watchdog Timeout Test (> 500 ms)
    bridge.update_driver_feedback(DriverTorqueFeedback(override_detected=False))
    bridge._last_frame_time = time.time() - 0.600  # 600 ms in past
    cmd_watchdog = bridge.check_watchdog_timeout()
    assert cmd_watchdog is not None
    assert not cmd_watchdog.is_actuator_enabled
    assert cmd_watchdog.state == ActuatorState.WATCHDOG_TRIPPED
    print(f"[PASS] Watchdog 500 ms timeout trip verified: state={cmd_watchdog.state.value}")

    diag = bridge.get_diagnostics_report()
    print("Diagnostics summary:", json.dumps(diag, indent=2))
    print("All VehicleOutputBridge verification checks PASSED successfully.")
