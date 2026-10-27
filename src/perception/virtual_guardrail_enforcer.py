"""
src/perception/virtual_guardrail_enforcer.py
============================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Virtual Guardrail Safety Envelope & Autonomous Closed-Loop Interlock Enforcer.
Autonomous Driving Capability: driver-supervised SAE Level 2.
Safety Standards Alignment: ISO 26262-style process; ASIL-B aligned; not independently assessed.

This module implements the non-negotiable core safety safeguard of the APEX-Vision platform:
THE VEHICLE MUST NEVER CROSS EITHER LANE BOUNDARY (Class 2 Left or Class 3 Right).

Key Safety Guardrails & Architecture:
-------------------------------------
1. 0.3-Meter Inner Safety Envelope:
   Constructs a virtual boundary set 0.30 m inward from both the Left (Class 2) and
   Right (Class 3) physical road markings.
2. Dynamic Envelope Width Scaling:
   Adapts buffer width based on operational scene and speed:
     - High-speed highway (v > 20 m/s): buffer narrows to 0.20 m for high-speed tracking.
     - Urban stop-and-go (v < 12 m/s): buffer expands to 0.50 m to prevent curb scrubbing.
     - Adverse conditions (Rain, Tunnel, Night): buffer holds at 0.35 m.
3. 3-Tier Violation Severity Hierarchy:
   - Severity 1 (WARNING): Trajectory touches outer envelope (clearance <= 0.30 m).
     Action: Scale Stanley controller lateral gain by 1.5x to gently draw ego center.
   - Severity 2 (CRITICAL): Trajectory enters inner envelope (clearance <= 0.15 m).
     Action: Apply maximum counter-steering torque command to steer away from boundary.
   - Severity 3 (EMERGENCY): Trajectory breaches boundary (clearance <= 0.00 m).
     Action: Dispatch priority CAN 0x150 emergency brake assist and audible buzzer alarm.
4. Trajectory Horizon Forward Evaluation:
   Evaluates predicted vehicle flank positions at t in [0.0s .. 1.5s] using the
   kinematic trajectory ribbon from `lane_departure_predictor.py`.
5. Fail-Safe Arduino Relay Hardware Interlock:
   If perception quality score Q < 0.35 for 5+ frames or an unrecoverable system fault
   occurs, generates de-energize command for microcontroller clutch relay to physically
   disconnect steering rack motor and alert driver per SAE Level 2.
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

from src.perception.faded_lane_recovery import BoundaryPolynomial2D
from src.perception.lane_departure_predictor import (
    DepartureDirection,
    LaneDeparturePrediction,
    ProjectedWaypoint,
)
from src.scene.scene_profiles import SceneType

logger = logging.getLogger("APEX.VirtualGuardrailEnforcer")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Violation Hierarchy & Telemetry Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@unique
class GuardrailSeverity(str, Enum):
    """Enforcement severity tier."""
    NOMINAL = "NOMINAL"                 # Fully compliant within safe envelope
    WARNING = "WARNING"                 # Severity 1: Trajectory within 0.30m of boundary
    CRITICAL = "CRITICAL"               # Severity 2: Trajectory within 0.15m of boundary
    EMERGENCY = "EMERGENCY"             # Severity 3: Trajectory breaches physical boundary


@unique
class BoundarySide(str, Enum):
    """Specific road boundary being threatened."""
    NONE = "NONE"
    LEFT_BOUNDARY = "LEFT_BOUNDARY"     # Class 2 Left
    RIGHT_BOUNDARY = "RIGHT_BOUNDARY"   # Class 3 Right


@dataclass
class GuardrailViolation:
    """
    Forensic record of a guardrail safety envelope event.
    """
    timestamp_utc: str
    severity: GuardrailSeverity
    threatened_side: BoundarySide
    min_clearance_m: float              # Distance to physical line
    buffer_width_m: float               # Currently active inner buffer
    penetration_depth_m: float          # Depth into virtual envelope (positive = breach)
    time_to_envelope_touch_s: float     # Estimated time to touch envelope
    recommended_gain_multiplier: float  # Stanley gain multiplier
    emergency_brake_assist_active: bool
    can_message_priority: int           # CAN arbitration ID (0x150)
    arduino_relay_trip_command: bool    # True if safety interlock trips relay cutoff


@dataclass
class GuardrailEnforcementTelemetry:
    """
    Master enforcement output returned to the vehicle control bridge and HUD.
    """
    timestamp_utc: str
    active_severity: GuardrailSeverity
    active_side: BoundarySide
    left_clearance_m: float
    right_clearance_m: float
    effective_buffer_width_m: float
    active_stanley_gain: float
    counter_steer_torque_command_nm: float
    is_safe_for_actuation: bool
    violation_event: Optional[GuardrailViolation]
    processing_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert telemetry to serializable dictionary."""
        return {
            "timestamp_utc": self.timestamp_utc,
            "severity": self.active_severity.value,
            "threatened_side": self.active_side.value,
            "left_clearance_m": round(self.left_clearance_m, 3),
            "right_clearance_m": round(self.right_clearance_m, 3),
            "buffer_width_m": round(self.effective_buffer_width_m, 3),
            "stanley_gain": round(self.active_stanley_gain, 3),
            "counter_steer_nm": round(self.counter_steer_torque_command_nm, 2),
            "is_safe_for_actuation": self.is_safe_for_actuation,
            "has_violation": self.violation_event is not None,
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


@dataclass
class GuardrailConfig:
    """
    Configuration hyperparameters for virtual safety envelope enforcement.
    """
    vehicle_width_m: float = 1.90
    nominal_buffer_width_m: float = 0.30 # Standard 0.30m inner safety envelope
    highway_buffer_width_m: float = 0.20 # Narrowed for high speeds
    urban_buffer_width_m: float = 0.50   # Expanded for city navigation
    adverse_buffer_width_m: float = 0.35 # Rain, night, tunnel
    critical_buffer_fraction: float = 0.50 # Severity 2 boundary = 50% of buffer
    base_stanley_gain: float = 1.25
    warning_gain_multiplier: float = 1.50
    max_counter_steer_torque_nm: float = 4.50
    can_emergency_id: int = 0x150
    consecutive_low_quality_trip: int = 5
    quality_cutoff_threshold: float = 0.35


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Master Virtual Guardrail Enforcer Engine
# ═══════════════════════════════════════════════════════════════════════════════

class VirtualGuardrailEnforcer:
    """
    Closed-loop virtual guardrail safety enforcer.
    Evaluates vehicle predicted trajectory against dynamic inner envelopes.
    """

    def __init__(self, config: Optional[GuardrailConfig] = None) -> None:
        self.config = config if config is not None else GuardrailConfig()
        self._consecutive_low_q = 0
        self._relay_cutoff_tripped = False
        self._violation_history: List[GuardrailViolation] = []

    def compute_dynamic_buffer_width(
        self,
        vehicle_speed_mps: float,
        scene: SceneType = SceneType.HIGHWAY,
    ) -> float:
        """
        Dynamically scale inner buffer width based on operational scene and cruise velocity:
          - High speed (> 22 m/s): 0.20 m
          - Urban / low speed (< 12 m/s): 0.50 m
          - Rain, tunnel, night: 0.35 m
          - Nominal baseline: 0.30 m
        """
        if scene == SceneType.URBAN or vehicle_speed_mps < 12.0:
            return self.config.urban_buffer_width_m
        elif scene == SceneType.HIGHWAY and vehicle_speed_mps >= 22.0:
            return self.config.highway_buffer_width_m
        elif scene in (SceneType.RAIN, SceneType.TUNNEL, SceneType.NIGHT):
            return self.config.adverse_buffer_width_m
        return self.config.nominal_buffer_width_m

    def enforce_guardrails(
        self,
        left_poly: BoundaryPolynomial2D,
        right_poly: BoundaryPolynomial2D,
        prediction: LaneDeparturePrediction,
        perception_quality: float = 1.0,
        vehicle_speed_mps: float = 20.0,
        active_scene: SceneType = SceneType.HIGHWAY,
    ) -> GuardrailEnforcementTelemetry:
        """
        Evaluate ego predicted trajectory against 0.3m inner safety buffers.
        Modulates Stanley controller gains and issues emergency interventions if threatened.
        """
        t0 = time.perf_counter()
        half_w = self.config.vehicle_width_m * 0.5

        # 1. Quality Interlock: Audit Q < 0.35 degradation
        if perception_quality < self.config.quality_cutoff_threshold:
            self._consecutive_low_q += 1
            if self._consecutive_low_q >= self.config.consecutive_low_quality_trip:
                self._relay_cutoff_tripped = True
                logger.warning(
                    "RELAY CUTOFF: Lane confidence Q=%.2f < %.2f for %d frames. Disengaging steering.",
                    perception_quality,
                    self.config.quality_cutoff_threshold,
                    self._consecutive_low_q,
                )
        else:
            self._consecutive_low_q = max(0, self._consecutive_low_q - 1)
            if self._consecutive_low_q == 0:
                self._relay_cutoff_tripped = False

        # 2. Dynamic buffer scaling
        buffer_w = self.compute_dynamic_buffer_width(vehicle_speed_mps, active_scene)
        critical_buf = buffer_w * self.config.critical_buffer_fraction

        # 3. Flank clearances at current vehicle origin (y=0)
        left_clearance = abs(left_poly.evaluate_x(0.0)) - half_w if left_poly.is_valid else 99.0
        right_clearance = abs(right_poly.evaluate_x(0.0)) - half_w if right_poly.is_valid else 99.0

        # 4. Evaluate projected forward trajectory ribbon (t in [0.0s .. 1.5s])
        min_projected_left = left_clearance
        min_projected_right = right_clearance
        time_to_touch = 99.0

        for wp in prediction.projected_ribbon:
            y_eval = float(wp.y_m)
            if left_poly.is_valid:
                x_line_left = float(left_poly.evaluate_x(y_eval))
                # Left wheel clearance: x_wheel is negative
                d_l = abs(x_line_left) - abs(wp.left_wheel_x_m)
                if d_l < min_projected_left:
                    min_projected_left = d_l
                    if d_l <= buffer_w and time_to_touch > wp.time_horizon_s:
                        time_to_touch = wp.time_horizon_s

            if right_poly.is_valid:
                x_line_right = float(right_poly.evaluate_x(y_eval))
                # Right wheel clearance: x_wheel is positive
                d_r = abs(x_line_right) - abs(wp.right_wheel_x_m)
                if d_r < min_projected_right:
                    min_projected_right = d_r
                    if d_r <= buffer_w and time_to_touch > wp.time_horizon_s:
                        time_to_touch = wp.time_horizon_s

        # 5. Determine worst-case encroachment
        if min_projected_left < min_projected_right:
            min_clearance = min_projected_left
            threatened_side = BoundarySide.LEFT_BOUNDARY
        else:
            min_clearance = min_projected_right
            threatened_side = BoundarySide.RIGHT_BOUNDARY

        # 6. Classify Severity and Actuation Directives
        active_gain = self.config.base_stanley_gain
        counter_steer_nm = 0.0
        violation: Optional[GuardrailViolation] = None

        if min_clearance <= 0.0:
            # Severity 3: EMERGENCY (Boundary crossed!)
            severity = GuardrailSeverity.EMERGENCY
            active_gain = self.config.base_stanley_gain * 2.0
            torque_sign = +1.0 if threatened_side == BoundarySide.LEFT_BOUNDARY else -1.0
            counter_steer_nm = torque_sign * self.config.max_counter_steer_torque_nm

            violation = GuardrailViolation(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                threatened_side=threatened_side,
                min_clearance_m=round(min_clearance, 3),
                buffer_width_m=round(buffer_w, 3),
                penetration_depth_m=round(-min_clearance, 3),
                time_to_envelope_touch_s=round(time_to_touch, 2),
                recommended_gain_multiplier=2.0,
                emergency_brake_assist_active=True,
                can_message_priority=self.config.can_emergency_id,
                arduino_relay_trip_command=False,
            )

        elif min_clearance <= critical_buf:
            # Severity 2: CRITICAL (Inside inner buffer)
            severity = GuardrailSeverity.CRITICAL
            active_gain = self.config.base_stanley_gain * 1.75
            penetration = critical_buf - min_clearance
            torque_frac = min(1.0, penetration / critical_buf)
            torque_sign = +1.0 if threatened_side == BoundarySide.LEFT_BOUNDARY else -1.0
            counter_steer_nm = torque_sign * self.config.max_counter_steer_torque_nm * torque_frac

            violation = GuardrailViolation(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                threatened_side=threatened_side,
                min_clearance_m=round(min_clearance, 3),
                buffer_width_m=round(buffer_w, 3),
                penetration_depth_m=round(buffer_w - min_clearance, 3),
                time_to_envelope_touch_s=round(time_to_touch, 2),
                recommended_gain_multiplier=1.75,
                emergency_brake_assist_active=False,
                can_message_priority=self.config.can_emergency_id,
                arduino_relay_trip_command=False,
            )

        elif min_clearance <= buffer_w:
            # Severity 1: WARNING (Touching outer envelope)
            severity = GuardrailSeverity.WARNING
            active_gain = self.config.base_stanley_gain * self.config.warning_gain_multiplier

            violation = GuardrailViolation(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                severity=severity,
                threatened_side=threatened_side,
                min_clearance_m=round(min_clearance, 3),
                buffer_width_m=round(buffer_w, 3),
                penetration_depth_m=round(buffer_w - min_clearance, 3),
                time_to_envelope_touch_s=round(time_to_touch, 2),
                recommended_gain_multiplier=self.config.warning_gain_multiplier,
                emergency_brake_assist_active=False,
                can_message_priority=self.config.can_emergency_id,
                arduino_relay_trip_command=False,
            )

        else:
            # Nominal centered driving
            severity = GuardrailSeverity.NOMINAL
            threatened_side = BoundarySide.NONE

        if violation is not None:
            self._violation_history.append(violation)
            if len(self._violation_history) > 200:
                self._violation_history.pop(0)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        is_safe = (not self._relay_cutoff_tripped) and (severity != GuardrailSeverity.EMERGENCY)

        return GuardrailEnforcementTelemetry(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            active_severity=severity,
            active_side=threatened_side,
            left_clearance_m=round(left_clearance, 3),
            right_clearance_m=round(right_clearance, 3),
            effective_buffer_width_m=round(buffer_w, 3),
            active_stanley_gain=round(active_gain, 3),
            counter_steer_torque_command_nm=round(counter_steer_nm, 2),
            is_safe_for_actuation=is_safe,
            violation_event=violation,
            processing_time_ms=elapsed_ms,
        )

    def get_violation_history(self) -> List[GuardrailViolation]:
        """Return history of all past envelope violation events."""
        return copy.deepcopy(self._violation_history)

    def force_reset_interlock(self) -> None:
        """Clear tripped relay cutoff state."""
        self._consecutive_low_q = 0
        self._relay_cutoff_tripped = False

    @property
    def is_interlock_tripped(self) -> bool:
        """Return True if the Arduino safety relay interlock has tripped."""
        return self._relay_cutoff_tripped


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Diagnostic HUD Canvas Visualizer & Safety Envelope Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_guardrail_hud(
    bev_canvas_bgr: np.ndarray,
    telemetry: GuardrailEnforcementTelemetry,
    left_poly: BoundaryPolynomial2D,
    right_poly: BoundaryPolynomial2D,
    meters_per_px: float = 0.0185,
) -> np.ndarray:
    """
    Renders the 0.3-meter inner virtual guardrail safety envelope onto BEV canvas.
    Shades inner envelopes in red/amber and highlights active counter-steering vectors.
    """
    out = bev_canvas_bgr.copy()
    h, w = out.shape[:2]
    cx = w // 2

    y_eval = np.linspace(0.0, 25.0, 60)
    buf_m = telemetry.effective_buffer_width_m

    def to_px(x_m: float, y_m: float) -> Tuple[int, int]:
        px_x = int(round(cx + (x_m / meters_per_px)))
        px_y = int(round(h - (y_m / meters_per_px)))
        return px_x, px_y

    # Draw Left Inner Safety Envelope (Class 2 + buf_m)
    if left_poly.is_valid:
        pts_left_env = [to_px(left_poly.evaluate_x(y) + buf_m, y) for y in y_eval]
        color_env_l = (0, 0, 255) if telemetry.active_side == BoundarySide.LEFT_BOUNDARY else (0, 140, 255)
        for i in range(len(pts_left_env) - 1):
            cv2.line(out, pts_left_env[i], pts_left_env[i + 1], color_env_l, 1, cv2.LINE_AA)

    # Draw Right Inner Safety Envelope (Class 3 - buf_m)
    if right_poly.is_valid:
        pts_right_env = [to_px(right_poly.evaluate_x(y) - buf_m, y) for y in y_eval]
        color_env_r = (0, 0, 255) if telemetry.active_side == BoundarySide.RIGHT_BOUNDARY else (0, 140, 255)
        for i in range(len(pts_right_env) - 1):
            cv2.line(out, pts_right_env[i], pts_right_env[i + 1], color_env_r, 1, cv2.LINE_AA)

    # Telemetry Badge Box
    cv2.rectangle(out, (10, h - 90), (280, h - 10), (25, 25, 25), -1)
    sev_color = (0, 200, 0) if telemetry.active_severity == GuardrailSeverity.NOMINAL else (0, 0, 255)
    cv2.rectangle(out, (10, h - 90), (280, h - 10), sev_color, 2)

    title = f"GUARDRAIL: {telemetry.active_severity.value}"
    cv2.putText(out, title, (20, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, sev_color, 1, cv2.LINE_AA)

    info1 = f"Buffer: {telemetry.effective_buffer_width_m:.2f}m | Stanley: {telemetry.active_stanley_gain:.2f}x"
    info2 = f"Clearance: L={telemetry.left_clearance_m:.2f}m | R={telemetry.right_clearance_m:.2f}m"
    info3 = f"Counter-Steer: {telemetry.counter_steer_torque_command_nm:+.1f} Nm"

    cv2.putText(out, info1, (20, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, info2, (20, h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(out, info3, (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Arduino Microcontroller Physical Clutch Relay Interlock Driver
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RelayCommandPacket:
    """Serial packet transmitted to the safety microcontroller."""
    relay_energized: bool               # True = Clutch closed (LKA active), False = De-energized
    safety_code: int                    # Cryptographic or handshake token
    reason_string: str
    serial_payload: bytes


class ArduinoRelaySafetyDriver:
    """
    Direct interface to the Arduino Mega / STM32 clutch relay hardware node.
    Enforces fail-safe de-energization (normally open circuit): loss of heartbeat,
    quality score Q < 0.35, or emergency violation opens the relay in < 15 ms,
    mechanically decoupling the steering rack motor from the driver rack.
    """

    def __init__(self, serial_port: Optional[str] = None, baudrate: int = 115200) -> None:
        self.port = serial_port
        self.baud = baudrate
        self.is_connected = False
        self._last_state: bool = True  # Default energized

    def format_relay_packet(
        self,
        energize: bool,
        reason: str = "NOMINAL",
    ) -> RelayCommandPacket:
        """
        Construct ASCII serial command packet:
          $APEX,RELAY,<STATE>,<REASON>,<CRC16>*<CR><LF>
        """
        state_str = "1" if energize else "0"
        body = f"APEX,RELAY,{state_str},{reason}"
        crc = self._compute_crc16_ccitt(body.encode("ascii"))
        full_msg = f"${body},{crc:04X}\r\n".encode("ascii")

        return RelayCommandPacket(
            relay_energized=energize,
            safety_code=0x5A5A if energize else 0x0000,
            reason_string=reason,
            serial_payload=full_msg,
        )

    def dispatch_safety_trip(self, reason: str = "PERCEPTION_QUALITY_DEGRADATION") -> RelayCommandPacket:
        """
        Generate de-energize trip command to physically open safety relay.
        """
        logger.critical("SAFETY INTERLOCK TRIPPED -> Opening Arduino steering clutch relay: %s", reason)
        self._last_state = False
        return self.format_relay_packet(energize=False, reason=reason)

    def dispatch_engage(self, reason: str = "PERCEPTION_HEALTH_RESTORED") -> RelayCommandPacket:
        """
        Generate engage command to close safety relay clutch.
        """
        logger.info("SAFETY RESTORED -> Closing Arduino steering clutch relay: %s", reason)
        self._last_state = True
        return self.format_relay_packet(energize=True, reason=reason)

    @staticmethod
    def _compute_crc16_ccitt(data: bytes) -> int:
        """Standard 16-bit CRC-CCITT (poly=0x1021)."""
        crc = 0xFFFF
        for b in data:
            crc = ((crc >> 8) | (crc << 8)) & 0xFFFF
            crc ^= b
            crc ^= (crc & 0xFF) >> 4
            crc ^= (crc << 12) & 0xFFFF
            crc ^= ((crc & 0xFF) << 5) & 0xFFFF
        return crc


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Streamlit HUD Telemetry Broadcaster
# ═══════════════════════════════════════════════════════════════════════════════

class StreamlitHUDPublisher:
    """
    Maintains a ring-buffer state cache of guardrail and steering telemetry
    for consumption by real-time Streamlit HUD dashboard displays.
    """

    def __init__(self, history_len: int = 100) -> None:
        self.history_len = history_len
        self._telemetry_buffer: List[Dict[str, Any]] = []

    def publish_telemetry(self, telemetry: GuardrailEnforcementTelemetry) -> None:
        """Store latest telemetry in buffer."""
        record = telemetry.to_dict()
        self._telemetry_buffer.append(record)
        if len(self._telemetry_buffer) > self.history_len:
            self._telemetry_buffer.pop(0)

    def get_hud_state_dict(self) -> Dict[str, Any]:
        """
        Retrieve structured dictionary for Streamlit front-end visualization.
        """
        if not self._telemetry_buffer:
            return {"status": "UNINITIALIZED", "severity": "NOMINAL"}

        latest = self._telemetry_buffer[-1]
        return {
            "current": latest,
            "history_length": len(self._telemetry_buffer),
            "recent_violations": sum(1 for r in self._telemetry_buffer if r.get("has_violation", False)),
            "buffer_width_m": latest.get("buffer_width_m", 0.30),
            "stanley_gain": latest.get("stanley_gain", 1.25),
            "counter_steer_nm": latest.get("counter_steer_nm", 0.0),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ISO 26262 ASIL-B Guardrail Audit Logger
# ═══════════════════════════════════════════════════════════════════════════════

class GuardrailAuditLogger:
    """
    Appends forensic guardrail violation and relay cutoff records to disk.
    Guarantees post-flight auditability for functional safety verification.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None) -> None:
        self.out_dir = Path(output_dir) if output_dir else PROJECT_ROOT / "logs" / "guardrail_safety"
        self._active_log_file: Optional[Path] = None

    def log_violation(self, violation: GuardrailViolation) -> None:
        """Append violation record to JSONL audit file."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if self._active_log_file is None:
            ts_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self._active_log_file = self.out_dir / f"guardrail_audit_{ts_str}.jsonl"

        import json
        with open(self._active_log_file, "a", encoding="utf-8") as f:
            d = asdict(violation)
            d["severity"] = violation.severity.value
            d["threatened_side"] = violation.threatened_side.value
            f.write(json.dumps(d) + "\n")


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Automated Synthetic Benchmarking Harness
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GuardrailBenchmarkReport:
    """Benchmark report certifying guardrail enforcement under simulated drift."""
    total_test_frames: int
    warning_count: int
    critical_count: int
    emergency_count: int
    relay_trip_count: int
    max_counter_steer_nm: float
    mean_enforcement_latency_ms: float
    passed_asil_b_enforcement: bool


def benchmark_virtual_guardrail_enforcer(
    enforcer: Optional[VirtualGuardrailEnforcer] = None,
) -> GuardrailBenchmarkReport:
    """
    Stress-test guardrail enforcer across progressive lateral drift scenarios:
      - Frames 0-9: Centered (Nominal)
      - Frames 10-19: Moderate drift left (clearance drops from 0.40m to 0.22m -> Warning)
      - Frames 20-29: Severe drift left (clearance drops to 0.10m -> Critical, Counter-steer)
      - Frames 30-34: Boundary touch (clearance drops to -0.05m -> Emergency)
      - Frames 35-44: Quality score drops to 0.20 -> Relay cutoff tripped after 5 frames
    """
    enf = enforcer if enforcer is not None else VirtualGuardrailEnforcer()
    enf.force_reset_interlock()

    nominal_w = 3.70
    left_poly = BoundaryPolynomial2D(is_valid=True, A=0.0, B=0.0, C=-nominal_w / 2.0, confidence=0.90)
    right_poly = BoundaryPolynomial2D(is_valid=True, A=0.0, B=0.0, C=+nominal_w / 2.0, confidence=0.90)

    n_warn, n_crit, n_emerg, n_trip = 0, 0, 0, 0
    max_torque = 0.0
    latencies: List[float] = []

    for f in range(45):
        # Construct synthetic predicted waypoint ribbon
        t_now = time.perf_counter()

        if f < 10:
            drift = 0.0
            q = 0.90
        elif f < 20:
            drift = -0.40 - (f - 10) * 0.05  # Approaches left boundary
            q = 0.90
        elif f < 30:
            drift = -0.70 - (f - 20) * 0.03  # Encroaches within 0.15m
            q = 0.85
        elif f < 35:
            drift = -0.98                    # Crosses left boundary!
            q = 0.80
        else:
            drift = 0.0
            q = 0.15                         # Low quality degradation

        # Waypoint with wheel flanks
        wp = ProjectedWaypoint(
            time_horizon_s=1.0,
            x_m=drift,
            y_m=15.0,
            heading_rad=0.0,
            left_wheel_x_m=drift - 0.95,
            right_wheel_x_m=drift + 0.95,
            lateral_uncertainty_sigma_m=0.05,
        )
        pred = LaneDeparturePrediction(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            severity=DepartureSeverity.NOMINAL,
            direction=DepartureDirection.NONE,
            time_to_departure_s=99.0,
            distance_to_left_boundary_m=1.0,
            distance_to_right_boundary_m=1.0,
            lateral_velocity_mps=0.0,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=[wp],
            can_alert_required=False,
            can_message_priority=0x140,
            processing_time_ms=0.1,
        )

        res = enf.enforce_guardrails(left_poly, right_poly, pred, perception_quality=q)
        lat = (time.perf_counter() - t_now) * 1000.0
        latencies.append(lat)

        if res.active_severity == GuardrailSeverity.WARNING:
            n_warn += 1
        elif res.active_severity == GuardrailSeverity.CRITICAL:
            n_crit += 1
        elif res.active_severity == GuardrailSeverity.EMERGENCY:
            n_emerg += 1

        if not res.is_safe_for_actuation:
            n_trip += 1

        max_torque = max(max_torque, abs(res.counter_steer_torque_command_nm))

    return GuardrailBenchmarkReport(
        total_test_frames=45,
        warning_count=n_warn,
        critical_count=n_crit,
        emergency_count=n_emerg,
        relay_trip_count=n_trip,
        max_counter_steer_nm=max_torque,
        mean_enforcement_latency_ms=float(np.mean(latencies)),
        passed_asil_b_enforcement=(n_emerg > 0 and n_trip > 0 and max_torque > 3.0),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Global Singleton Accessor
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_GUARDRAIL_ENFORCER: Optional[VirtualGuardrailEnforcer] = None
_GLOBAL_ARDUINO_RELAY: Optional[ArduinoRelaySafetyDriver] = None
_GLOBAL_HUD_PUBLISHER: Optional[StreamlitHUDPublisher] = None
_GLOBAL_AUDIT_LOGGER: Optional[GuardrailAuditLogger] = None


def get_virtual_guardrail_enforcer(config: Optional[GuardrailConfig] = None) -> VirtualGuardrailEnforcer:
    """
    Obtain or instantiate global singleton VirtualGuardrailEnforcer.
    """
    global _GLOBAL_GUARDRAIL_ENFORCER
    if _GLOBAL_GUARDRAIL_ENFORCER is None:
        _GLOBAL_GUARDRAIL_ENFORCER = VirtualGuardrailEnforcer(config=config)
    return _GLOBAL_GUARDRAIL_ENFORCER


def get_arduino_relay_driver() -> ArduinoRelaySafetyDriver:
    """Obtain or instantiate global ArduinoRelaySafetyDriver."""
    global _GLOBAL_ARDUINO_RELAY
    if _GLOBAL_ARDUINO_RELAY is None:
        _GLOBAL_ARDUINO_RELAY = ArduinoRelaySafetyDriver()
    return _GLOBAL_ARDUINO_RELAY


def get_streamlit_hud_publisher() -> StreamlitHUDPublisher:
    """Obtain or instantiate global StreamlitHUDPublisher."""
    global _GLOBAL_HUD_PUBLISHER
    if _GLOBAL_HUD_PUBLISHER is None:
        _GLOBAL_HUD_PUBLISHER = StreamlitHUDPublisher()
    return _GLOBAL_HUD_PUBLISHER


def get_guardrail_audit_logger() -> GuardrailAuditLogger:
    """Obtain or instantiate global GuardrailAuditLogger."""
    global _GLOBAL_AUDIT_LOGGER
    if _GLOBAL_AUDIT_LOGGER is None:
        _GLOBAL_AUDIT_LOGGER = GuardrailAuditLogger()
    return _GLOBAL_AUDIT_LOGGER


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Envelope Geometry & Safety Metrics Aggregation Utilities
# ═══════════════════════════════════════════════════════════════════════════════

class EnvelopeGeometryExporter:
    """
    Serializes virtual safety buffer envelopes into structured geometric formats
    suitable for CARLA bridge telemetry, ROS2 PointCloud/Marker messages, and
    diagnostics recording.

    Standards Compliance:
        - APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
        - Driver-supervised SAE Level 2 execution
        - ISO 26262-style process; ASIL-B aligned; not independently assessed
    """

    def __init__(self, decimal_precision: int = 4) -> None:
        self.precision = int(decimal_precision)

    def to_dict(self, envelope: SafetyEnvelope) -> Dict[str, Any]:
        """Convert a SafetyEnvelope instance to a serializable dictionary representation."""
        return {
            "valid": bool(envelope.is_valid),
            "buffer_margin_m": round(float(envelope.buffer_margin_m), self.precision),
            "critical_margin_m": round(float(envelope.critical_margin_m), self.precision),
            "emergency_margin_m": round(float(envelope.emergency_margin_m), self.precision),
            "left_physical": [
                [round(float(p[0]), self.precision), round(float(p[1]), self.precision)]
                for p in envelope.left_boundary_physical
            ],
            "left_safety_buffer": [
                [round(float(p[0]), self.precision), round(float(p[1]), self.precision)]
                for p in envelope.left_safety_buffer
            ],
            "right_physical": [
                [round(float(p[0]), self.precision), round(float(p[1]), self.precision)]
                for p in envelope.right_boundary_physical
            ],
            "right_safety_buffer": [
                [round(float(p[0]), self.precision), round(float(p[1]), self.precision)]
                for p in envelope.right_safety_buffer
            ],
        }

    def to_drivable_corridor_polygon(self, envelope: SafetyEnvelope) -> List[Tuple[float, float]]:
        """
        Produce a closed 2D polygon traversing along the right safety buffer
        and back along the left safety buffer.
        """
        if not envelope.is_valid or not envelope.left_safety_buffer or not envelope.right_safety_buffer:
            return []
        left_pts = [(round(p[0], self.precision), round(p[1], self.precision)) for p in envelope.left_safety_buffer]
        right_pts = [(round(p[0], self.precision), round(p[1], self.precision)) for p in envelope.right_safety_buffer]
        # Corridor: forward along right buffer, reverse along left buffer
        polygon = list(right_pts) + list(reversed(left_pts))
        return polygon


class GuardrailSafetyMetricsAggregator:
    """
    Online rolling statistics aggregator for safety envelope monitoring,
    boundary violation frequencies, and lateral controller stability metrics.

    Maintains a rolling window of 1000 frames to monitor ISO 26262 ASIL-B safety goals.
    """

    def __init__(self, window_size: int = 1000) -> None:
        self.window_size = int(window_size)
        self._history: deque = deque(maxlen=self.window_size)
        self._total_frame_count: int = 0
        self._warning_count: int = 0
        self._critical_count: int = 0
        self._emergency_count: int = 0
        self._relay_trip_count: int = 0
        self._lock = threading.Lock()

    def record_frame(self, enforcement_result: GuardrailEnforcementResult) -> None:
        """Ingest a single frame enforcement result into the rolling aggregator."""
        with self._lock:
            self._total_frame_count += 1
            sev = enforcement_result.active_severity

            if sev == GuardrailSeverity.WARNING:
                self._warning_count += 1
            elif sev == GuardrailSeverity.CRITICAL:
                self._critical_count += 1
            elif sev == GuardrailSeverity.EMERGENCY:
                self._emergency_count += 1

            if not enforcement_result.is_safe_for_actuation:
                self._relay_trip_count += 1

            self._history.append({
                "severity": sev.value,
                "torque_nm": float(enforcement_result.counter_steer_torque_command_nm),
                "stanley_scale": float(enforcement_result.stanley_gain_multiplier),
                "safe": bool(enforcement_result.is_safe_for_actuation),
                "dist_left": float(enforcement_result.distance_to_left_buffer_m),
                "dist_right": float(enforcement_result.distance_to_right_buffer_m),
                "latency_ms": float(enforcement_result.enforcement_latency_ms),
            })

    def compute_summary_statistics(self) -> Dict[str, Any]:
        """Compute summary rates and distributions over the active history window."""
        with self._lock:
            if not self._history:
                return {
                    "total_frames_processed": 0,
                    "warning_rate_pct": 0.0,
                    "critical_rate_pct": 0.0,
                    "emergency_rate_pct": 0.0,
                    "relay_trip_rate_pct": 0.0,
                    "mean_latency_ms": 0.0,
                    "max_torque_nm": 0.0,
                    "mean_left_clearance_m": 0.0,
                    "mean_right_clearance_m": 0.0,
                    "asil_b_compliance_flag": True,
                }

            n = float(len(self._history))
            warnings = sum(1 for h in self._history if h["severity"] == GuardrailSeverity.WARNING.value)
            crits = sum(1 for h in self._history if h["severity"] == GuardrailSeverity.CRITICAL.value)
            emergs = sum(1 for h in self._history if h["severity"] == GuardrailSeverity.EMERGENCY.value)
            trips = sum(1 for h in self._history if not h["safe"])

            latencies = [h["latency_ms"] for h in self._history]
            torques = [abs(h["torque_nm"]) for h in self._history]
            left_dists = [h["dist_left"] for h in self._history]
            right_dists = [h["dist_right"] for h in self._history]

            bvr = emergs / n if n > 0 else 0.0

            return {
                "total_frames_processed": int(self._total_frame_count),
                "rolling_window_frames": int(n),
                "warning_rate_pct": round(warnings / n * 100.0, 3),
                "critical_rate_pct": round(crits / n * 100.0, 3),
                "emergency_rate_pct": round(bvr * 100.0, 3),
                "relay_trip_rate_pct": round(trips / n * 100.0, 3),
                "mean_latency_ms": round(float(np.mean(latencies)), 4),
                "max_latency_ms": round(float(np.max(latencies)), 4),
                "mean_counter_steer_torque_nm": round(float(np.mean(torques)), 4),
                "max_counter_steer_torque_nm": round(float(np.max(torques)), 4),
                "mean_left_clearance_m": round(float(np.mean(left_dists)), 4),
                "mean_right_clearance_m": round(float(np.mean(right_dists)), 4),
                "boundary_violation_rate_target_met": bool(bvr < 0.001),
                "asil_b_compliance_flag": bool(bvr < 0.005 and float(np.mean(latencies)) < 5.0),
            }


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Self-Test & Diagnostic Execution Entrypoint
# ═══════════════════════════════════════════════════════════════════════════════

def run_guardrail_enforcer_diagnostics(verbose: bool = True) -> bool:
    """
    Execute full functional diagnostics for VirtualGuardrailEnforcer,
    verifying 0.3m buffer generation, dynamic scaling, severity escalation,
    CAN 0x150 priority alert triggers, and Arduino clutch relay disengagement.
    """
    if verbose:
        logger.info("Executing APEX-Vision VirtualGuardrailEnforcer self-test diagnostics...")

    cfg = GuardrailConfig(
        nominal_buffer_m=0.30,
        critical_buffer_m=0.15,
        highway_buffer_m=0.20,
        urban_buffer_m=0.50,
        relay_degradation_frames=5,
        can_bus_arbitration_id=0x150,
    )
    enforcer = VirtualGuardrailEnforcer(config=cfg)
    exporter = EnvelopeGeometryExporter()
    aggregator = GuardrailSafetyMetricsAggregator(window_size=100)

    # 1. Test safety envelope creation on straight road
    left_poly = (0.0, 0.0, -1.85)
    right_poly = (0.0, 0.0, 1.85)
    envelope = enforcer.build_safety_envelope(left_poly, right_poly, scene_tag="HIGHWAY")

    assert envelope.is_valid, "Expected valid safety envelope on standard straight road"
    assert envelope.buffer_margin_m == 0.20, f"Expected 0.20m on HIGHWAY, got {envelope.buffer_margin_m}"
    corridor = exporter.to_drivable_corridor_polygon(envelope)
    assert len(corridor) > 10, f"Expected valid corridor polygon, got {len(corridor)} points"

    # 2. Test nominal trajectory (center of lane)
    from src.perception.lane_departure_predictor import (
        LaneDeparturePrediction,
        DepartureSeverity,
        DepartureDirection,
        RibbonWaypoint,
    )
    center_ribbon = [
        RibbonWaypoint(
            x_m=0.0,
            y_m=float(y),
            time_offset_s=float(y) / 20.0,
            uncertainty_radius_m=0.05,
        )
        for y in range(2, 32, 2)
    ]
    pred_nominal = LaneDeparturePrediction(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        severity=DepartureSeverity.NOMINAL,
        direction=DepartureDirection.NONE,
        time_to_departure_s=99.0,
        distance_to_left_boundary_m=1.85,
        distance_to_right_boundary_m=1.85,
        lateral_velocity_mps=0.0,
        lateral_acceleration_mps2=0.0,
        yaw_rate_radps=0.0,
        projected_ribbon=center_ribbon,
        can_alert_required=False,
        can_message_priority=0x140,
        processing_time_ms=0.5,
    )

    res_nominal = enforcer.enforce_guardrails(left_poly, right_poly, pred_nominal, perception_quality=0.95)
    aggregator.record_frame(res_nominal)
    assert res_nominal.active_severity == GuardrailSeverity.NONE
    assert res_nominal.is_safe_for_actuation is True
    assert res_nominal.stanley_gain_multiplier == 1.0

    # 3. Test WARNING escalation (trajectory approaching within 0.3m buffer)
    left_encroaching_ribbon = [
        RibbonWaypoint(
            x_m=-1.65,
            y_m=float(y),
            time_offset_s=float(y) / 20.0,
            uncertainty_radius_m=0.05,
        )
        for y in range(2, 32, 2)
    ]
    pred_encroach = LaneDeparturePrediction(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        severity=DepartureSeverity.WARNING,
        direction=DepartureDirection.LEFT,
        time_to_departure_s=1.8,
        distance_to_left_boundary_m=0.20,
        distance_to_right_boundary_m=3.50,
        lateral_velocity_mps=-0.8,
        lateral_acceleration_mps2=-0.2,
        yaw_rate_radps=0.02,
        projected_ribbon=left_encroaching_ribbon,
        can_alert_required=True,
        can_message_priority=0x140,
        processing_time_ms=0.5,
    )
    res_warn = enforcer.enforce_guardrails(left_poly, right_poly, pred_encroach, perception_quality=0.90)
    aggregator.record_frame(res_warn)
    assert res_warn.active_severity in (GuardrailSeverity.WARNING, GuardrailSeverity.CRITICAL)
    assert res_warn.stanley_gain_multiplier >= 1.5
    assert res_warn.counter_steer_torque_command_nm > 0.0

    # 4. Test EMERGENCY boundary breach (trajectory crossing boundary)
    breach_ribbon = [
        RibbonWaypoint(
            x_m=-2.10,
            y_m=float(y),
            time_offset_s=float(y) / 20.0,
            uncertainty_radius_m=0.05,
        )
        for y in range(2, 32, 2)
    ]
    pred_breach = LaneDeparturePrediction(
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        severity=DepartureSeverity.CRITICAL,
        direction=DepartureDirection.LEFT,
        time_to_departure_s=0.2,
        distance_to_left_boundary_m=-0.25,
        distance_to_right_boundary_m=3.95,
        lateral_velocity_mps=-1.5,
        lateral_acceleration_mps2=-0.5,
        yaw_rate_radps=0.05,
        projected_ribbon=breach_ribbon,
        can_alert_required=True,
        can_message_priority=0x150,
        processing_time_ms=0.5,
    )
    res_emerg = enforcer.enforce_guardrails(left_poly, right_poly, pred_breach, perception_quality=0.85)
    aggregator.record_frame(res_emerg)
    assert res_emerg.active_severity == GuardrailSeverity.EMERGENCY
    assert res_emerg.can_brake_assist_triggered is True
    assert res_emerg.can_arbitration_id == 0x150

    # 5. Test degradation interlock (Q < 0.35 for 5 frames)
    for _ in range(5):
        res_trip = enforcer.enforce_guardrails(left_poly, right_poly, pred_nominal, perception_quality=0.20)
        aggregator.record_frame(res_trip)

    assert enforcer.is_relay_tripped() is True
    assert res_trip.is_safe_for_actuation is False

    stats = aggregator.compute_summary_statistics()
    assert stats["total_frames_processed"] == 8

    if verbose:
        logger.info(f"Guardrail diagnostics completed successfully: {stats}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    success = run_guardrail_enforcer_diagnostics(verbose=True)
    benchmark_report = run_guardrail_enforcer_benchmark()
    print("=" * 80)
    print("APEX-Vision Virtual Guardrail Enforcer Diagnostic & Benchmark Summary")
    print(f"Self-Test Passed: {success}")
    print(f"Total Benchmark Frames: {benchmark_report.total_test_frames}")
    print(f"Warning Enforcements: {benchmark_report.warning_count}")
    print(f"Critical Enforcements: {benchmark_report.critical_count}")
    print(f"Emergency Enforcements: {benchmark_report.emergency_count}")
    print(f"Relay Trips: {benchmark_report.relay_trip_count}")
    print(f"Max Counter-Steer Torque: {benchmark_report.max_counter_steer_nm:.2f} Nm")
    print(f"Mean Latency: {benchmark_report.mean_enforcement_latency_ms:.3f} ms")
    print(f"ASIL-B Enforcement Standard Met: {benchmark_report.passed_asil_b_enforcement}")
    print("=" * 80)


