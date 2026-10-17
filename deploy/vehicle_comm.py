"""
deploy/vehicle_comm.py
======================
High-speed, fault-tolerant Vehicle Communication Layer for Jetson Orin Nano:
- Dual-path routing: Primary CAN Bus (500 kbps) with seamless Serial fallback
- Non-blocking asynchronous I/O with background transceiver threads
- Compact bit-packed struct serialization matching deploy/protocol_spec.md
- Auto-discovery and dynamic reconnection for 2x Arduino Mega 2560 nodes
- Full loopback simulation mode for testing without hardware peripherals
"""

from __future__ import annotations

import collections
import glob
import json
import logging
import platform
import struct
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Optional communication libraries
try:
    import can
    HAS_CAN = True
except ImportError:
    HAS_CAN = False

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

log = logging.getLogger("apex_comm")


# ═══════════════════════════════════════════════════════════════════════════
# Protocol Constants & Enums
# ═══════════════════════════════════════════════════════════════════════════

CAN_ID_HEARTBEAT    = 0x050
CAN_ID_CMD_NODE1    = 0x100
CAN_ID_THREAT_ALERT = 0x150
CAN_ID_SPATIAL_ALERT= 0x160
CAN_ID_TRAJECTORY_ALERT = 0x170
CAN_ID_CMD_NODE2    = 0x200
CAN_ID_FEEDBACK_N1  = 0x300
CAN_ID_FEEDBACK_N2  = 0x301

MODE_MANUAL         = 0
MODE_AUTO           = 1
MODE_EMERGENCY_STOP = 2

MODE_STR_TO_INT = {
    "MANUAL": MODE_MANUAL,
    "AUTO": MODE_AUTO,
    "EMERGENCY_STOP": MODE_EMERGENCY_STOP,
}

MODE_INT_TO_STR = {v: k for k, v in MODE_STR_TO_INT.items()}


@dataclass
class Node1Telemetry:
    """Feedback telemetry received from Arduino Node 1 (Steering & Actuator)."""
    node_id: int = 1
    actual_steering_deg: float = 0.0
    motor_current_amps: float = 0.0
    override_active: bool = False
    fault_flags: int = 0
    uptime_ms: int = 0
    timestamp: float = 0.0


@dataclass
class Node2Telemetry:
    """Feedback telemetry received from Arduino Node 2 (Throttle, Brake & Speed)."""
    node_id: int = 2
    wheel_speed_mps: float = 0.0
    brake_pressure_psi: float = 0.0
    override_active: bool = False
    fault_flags: int = 0
    uptime_ms: int = 0
    timestamp: float = 0.0


@dataclass
class ThreatAlertTelemetry:
    """Obstacle Threat Alert telemetry transmitted/received via CAN ID 0x150."""
    distance_mm: int = 0
    class_enum: int = 0
    risk_score: int = 0
    emergency_stop: bool = False
    seq: int = 0
    timestamp: float = 0.0


# 3D Occupancy Threat Levels
THREAT_LEVEL_CLEAR          = 0
THREAT_LEVEL_CAUTION        = 1
THREAT_LEVEL_WARNING        = 2
THREAT_LEVEL_CRITICAL_STOP  = 3

THREAT_LEVEL_STR_TO_INT = {
    "CLEAR": THREAT_LEVEL_CLEAR,
    "CAUTION": THREAT_LEVEL_CAUTION,
    "WARNING": THREAT_LEVEL_WARNING,
    "CRITICAL_STOP": THREAT_LEVEL_CRITICAL_STOP,
}
THREAT_LEVEL_INT_TO_STR = {v: k for k, v in THREAT_LEVEL_STR_TO_INT.items()}


@dataclass
class SpatialAlertTelemetry:
    """3D Obstacle Spatial Alert telemetry transmitted/received via CAN ID 0x160."""
    x_mm: int = 0
    z_mm: int = 65535
    tti_ms: int = 65535
    class_enum: int = 0
    threat_level: int = THREAT_LEVEL_CLEAR
@dataclass
class TrajectoryAlertTelemetry:
    """Trajectory Collision Alert telemetry transmitted/received via CAN ID 0x170."""
    track_id: int = 0
    tti_ms: int = 65535
    cross_x_mm: int = 0
    risk_level: int = 0
    action_flag: int = 0
    timestamp: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Serial Interface Layer
# ═══════════════════════════════════════════════════════════════════════════

class SerialInterface:
    """
    Non-blocking line-delimited JSON Serial communicator supporting
    auto-port scanning, node discovery, and automatic reconnection.
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 115200,
        node_id: int = 1,
        auto_scan: bool = True,
    ) -> None:
        self.port = port
        self.baudrate = baudrate
        self.node_id = node_id
        self.auto_scan = auto_scan

        self.ser: Optional[serial.Serial] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.is_connected = False

        self._rx_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._last_reconnect = 0.0

        if HAS_SERIAL:
            self._init_connection()

    def _init_connection(self) -> bool:
        """Attempt port connection with optional auto-scanning."""
        if not HAS_SERIAL:
            return False

        ports_to_try = []
        if self.port:
            ports_to_try.append(self.port)

        if self.auto_scan:
            try:
                available = [p.device for p in serial.tools.list_ports.comports()]
                for p in available:
                    if p not in ports_to_try:
                        ports_to_try.append(p)
            except Exception:
                pass

        for p in ports_to_try:
            try:
                s = serial.Serial(p, self.baudrate, timeout=0.05, write_timeout=0.05)
                self.ser = s
                self.port = p
                self.is_connected = True
                log.info("SerialInterface (Node %d) connected to %s @ %d baud", self.node_id, p, self.baudrate)
                self._start_reader()
                return True
            except Exception as e:
                log.debug("Serial connection attempt failed on %s: %s", p, e)

        self.is_connected = False
        return False

    def _start_reader(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._reader_loop,
            name=f"SerialReader_Node{self.node_id}",
            daemon=True,
        )
        self._thread.start()

    def _reader_loop(self) -> None:
        """Background thread consuming incoming newline-delimited JSON packets."""
        while self._running:
            if not self.is_connected or self.ser is None or not self.ser.is_open:
                now = time.time()
                if now - self._last_reconnect >= 4.0:
                    self._last_reconnect = now
                    self._init_connection()
                time.sleep(0.1)
                continue

            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if not line:
                    continue

                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    for cb in self._rx_callbacks:
                        cb(data)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception as e:
                log.debug("Serial read exception on Node %d: %s", self.node_id, e)
                self.is_connected = False
                try:
                    if self.ser:
                        self.ser.close()
                except Exception:
                    pass
                time.sleep(0.1)

    def register_callback(self, cb: Callable[[Dict[str, Any]], None]) -> None:
        self._rx_callbacks.append(cb)

    def send_json(self, payload: Dict[str, Any]) -> bool:
        """Transmit a JSON packet terminated by newline."""
        if not self.is_connected or self.ser is None or not self.ser.is_open:
            return False

        try:
            line = (json.dumps(payload) + "\n").encode("utf-8")
            with self._lock:
                self.ser.write(line)
                self.ser.flush()
            return True
        except Exception as e:
            log.warning("Serial send failed on Node %d: %s", self.node_id, e)
            self.is_connected = False
            return False

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        with self._lock:
            if self.ser and self.ser.is_open:
                try:
                    self.ser.close()
                except Exception:
                    pass
        self.is_connected = False


# ═══════════════════════════════════════════════════════════════════════════
# CAN Bus Interface Layer
# ═══════════════════════════════════════════════════════════════════════════

class CANInterface:
    """
    SocketCAN interface with background frame listener and struct packing.
    """

    def __init__(
        self,
        channel: str = "can0",
        bustype: str = "socketcan",
        bitrate: int = 500000,
        sim_mode: bool = False,
    ) -> None:
        self.channel = channel
        self.bustype = bustype
        self.bitrate = bitrate
        self.sim_mode = sim_mode

        self.bus: Optional[can.BusABC] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.is_connected = False
        self._last_reconnect = 0.0

        self._rx_callbacks: Dict[int, List[Callable[[int, bytes], None]]] = collections.defaultdict(list)

        if HAS_CAN and not self.sim_mode:
            self._init_bus()
        elif self.sim_mode:
            self.is_connected = True
            log.info("CANInterface initialized in Simulation Loopback mode.")

    def _init_bus(self) -> bool:
        if not HAS_CAN:
            return False

        try:
            # On Linux, SocketCAN is standard. On other OS or fallback, use virtual
            if platform.system() != "Linux" and self.bustype == "socketcan":
                log.info("Non-Linux OS detected; switching CAN bustype to 'virtual'")
                self.bustype = "virtual"

            self.bus = can.interface.Bus(
                channel=self.channel,
                bustype=self.bustype,
                bitrate=self.bitrate,
            )
            self.is_connected = True
            log.info("SocketCAN initialized on %s (%s @ %d bps)", self.channel, self.bustype, self.bitrate)
            self._start_listener()
            return True
        except Exception as e:
            log.warning("SocketCAN bus initialization failed on %s: %s", self.channel, e)
            self.is_connected = False
            return False

    def _start_listener(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._listener_loop, name="CANListenerThread", daemon=True)
        self._thread.start()

    def _listener_loop(self) -> None:
        """Asynchronous CAN frame receive loop with ID dispatching."""
        while self._running:
            if not self.is_connected or self.bus is None:
                now = time.time()
                if now - self._last_reconnect >= 4.0:
                    self._last_reconnect = now
                    self._init_bus()
                time.sleep(0.1)
                continue

            try:
                msg = self.bus.recv(timeout=0.05)
                if msg is not None:
                    cbs = self._rx_callbacks.get(msg.arbitration_id, [])
                    for cb in cbs:
                        cb(msg.arbitration_id, msg.data)
            except Exception as e:
                log.debug("CAN recv exception: %s", e)
                time.sleep(0.01)

    def register_callback(self, can_id: int, cb: Callable[[int, bytes], None]) -> None:
        self._rx_callbacks[can_id].append(cb)

    def send_frame(self, can_id: int, data: bytes) -> bool:
        """Send a standard 11-bit CAN data frame."""
        if self.sim_mode:
            # Deliver to local callbacks in simulation mode
            for cb in self._rx_callbacks.get(can_id, []):
                cb(can_id, data)
            return True

        if not self.is_connected or self.bus is None:
            return False

        try:
            msg = can.Message(
                arbitration_id=can_id,
                data=data,
                is_extended_id=False,
            )
            self.bus.send(msg, timeout=0.01)
            return True
        except Exception as e:
            log.warning("CAN transmission error on ID 0x%03X: %s", can_id, e)
            return False

    @staticmethod
    def pack_node1_cmd(
        steering_deg: float,
        offset_mm_or_trim: float,
        confidence: float,
        mode_val: int,
        ts_ms: int,
        endian: str = "<",
    ) -> bytes:
        """Bit-pack 8-byte CAN ID 0x100 control frame."""
        steer_s = int(np.clip(steering_deg * 100, -32768, 32767))
        trim_s = int(np.clip(offset_mm_or_trim, -32768, 32767))
        conf_s = int(np.clip(confidence * 255, 0, 255))
        return struct.pack(f"{endian}hhBBH", steer_s, trim_s, conf_s, mode_val, ts_ms % 65536)

    @staticmethod
    def pack_node2_cmd(
        cmd_a: float,
        cmd_b: float,
        mode_val: int,
        flags: int,
        ts_ms: int,
        endian: str = "<",
    ) -> bytes:
        """Bit-pack 8-byte CAN ID 0x200 control frame."""
        a_s = int(np.clip(cmd_a * 100, -32768, 32767))
        b_s = int(np.clip(cmd_b * 100, -32768, 32767))
        return struct.pack(f"{endian}hhBBH", a_s, b_s, mode_val, flags, ts_ms % 65536)

    @staticmethod
    def pack_heartbeat(
        alive_token: int = 0xAA,
        health_code: int = 0,
        sequence_id: int = 0,
        endian: str = "<",
    ) -> bytes:
        """Bit-pack 4-byte CAN ID 0x050 heartbeat frame."""
        return struct.pack(f"{endian}BBH", alive_token, health_code, sequence_id % 65536)

    def close(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.bus:
            try:
                self.bus.shutdown()
            except Exception:
                pass
        self.is_connected = False


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 Section 5: Electromechanical Actuation Mapper
# ═══════════════════════════════════════════════════════════════════════════

class ActuationMapper:
    """
    Calibrated software steering angle to physical actuator count mapper.
    Maps delta_deg in [-30, +30] to counts with non-linear Ackermann LUT,
    direction sign, mechanical zero center offset, deadband, and rate limiting.
    """

    def __init__(self, config_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None) -> None:
        self.config = self._load_config(config_or_path)
        self.direction_sign = int(self.config.get("direction_sign", 1))
        self.center_offset_counts = int(self.config.get("center_offset_counts", 0))
        self.counts_per_deg = float(self.config.get("counts_per_deg", 12.5))
        self.deadband_counts = int(self.config.get("deadband_counts", 2))
        self.max_rate_counts_per_tick = int(self.config.get("max_rate_counts_per_tick", 25))
        self.use_counts_in_can = bool(self.config.get("use_counts_in_can", True))
        self.counts_max = int(self.config.get("counts_max", 450))

        raw_lut = self.config.get("nonlinear_lut", [
            [-30.0, -410],
            [-20.0, -260],
            [-10.0, -125],
            [  0.0,    0],
            [ 10.0,  128],
            [ 20.0,  272],
            [ 30.0,  430],
        ])
        self.lut_degs = np.array([float(pt[0]) for pt in raw_lut], dtype=np.float64)
        self.lut_counts = np.array([float(pt[1]) for pt in raw_lut], dtype=np.float64)

        self._prev_counts: Optional[int] = None

    def _load_config(self, config_or_path: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config_or_path, dict):
            return config_or_path
        if config_or_path is not None:
            p = Path(config_or_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}

        default_p = PROJECT_ROOT / "configs" / "actuation_profile.yaml"
        if default_p.exists():
            with open(default_p, "r") as f:
                return yaml.safe_load(f) or {}
        return {}

    def reset(self) -> None:
        self._prev_counts = None

    def deg_to_counts(self, delta_deg: float) -> int:
        """Convert steering angle in degrees to physical actuator counts."""
        # 1. Interpolate from LUT with linear extrapolation beyond endpoints
        if delta_deg > self.lut_degs[-1]:
            raw_counts = self.lut_counts[-1] + (delta_deg - self.lut_degs[-1]) * self.counts_per_deg
        elif delta_deg < self.lut_degs[0]:
            raw_counts = self.lut_counts[0] + (delta_deg - self.lut_degs[0]) * self.counts_per_deg
        else:
            raw_counts = float(np.interp(delta_deg, self.lut_degs, self.lut_counts))

        # 2. Apply direction sign and center trim offset
        c = self.direction_sign * raw_counts + self.center_offset_counts

        # 3. Apply Deadband
        if abs(c) <= self.deadband_counts:
            c = 0.0

        target_counts = int(round(c))

        # 4. Apply Rate Limiting
        if self._prev_counts is not None and self.max_rate_counts_per_tick > 0:
            diff = target_counts - self._prev_counts
            clamped_diff = int(np.clip(diff, -self.max_rate_counts_per_tick, self.max_rate_counts_per_tick))
            final_counts = self._prev_counts + clamped_diff
        else:
            final_counts = target_counts

        # 5. Clamp to hard physical counts ceiling
        saturated_counts = int(np.clip(final_counts, -self.counts_max, self.counts_max))
        self._prev_counts = saturated_counts
        return saturated_counts

    def counts_to_deg(self, counts: int) -> float:
        """Inverse mapping from actuator counts back to steering degrees."""
        # 1. Reverse center offset
        c = float(counts) - self.center_offset_counts

        # 2. Reverse direction sign
        c_unscaled = c * self.direction_sign

        # 3. Inverse interpolation on LUT
        delta_deg = float(np.interp(c_unscaled, self.lut_counts, self.lut_degs))
        return round(delta_deg, 2)


# ═══════════════════════════════════════════════════════════════════════════
# Master Vehicle Communicator
# ═══════════════════════════════════════════════════════════════════════════

class VehicleCommunicator:
    """
    Unified, multi-node vehicle communication manager with automatic
    failover between CAN Bus and USB Serial, and thread-safe telemetry caches.
    """

    def __init__(
        self,
        can_channel: str = "can0",
        can_bitrate: int = 500000,
        node1_serial_port: Optional[str] = None,
        node2_serial_port: Optional[str] = None,
        baudrate: int = 115200,
        enable_can: bool = True,
        enable_serial: bool = True,
        sim_mode: bool = False,
        actuation_profile: Optional[Union[str, Path, Dict[str, Any]]] = None,
    ) -> None:
        self.sim_mode = sim_mode
        self.enable_can = enable_can
        self.enable_serial = enable_serial

        # Actuation mapping layer
        self.actuation_mapper = ActuationMapper(actuation_profile)

        # Failover state tracking
        self.active_channel = "CAN" if enable_can else "SERIAL"
        self.consecutive_can_errors = 0
        self.can_error_threshold = 5

        # Initialize CAN
        self.can_if = CANInterface(
            channel=can_channel,
            bitrate=can_bitrate,
            sim_mode=sim_mode,
        ) if enable_can else None

        # Initialize Serial nodes
        self.serial_n1 = SerialInterface(
            port=node1_serial_port,
            baudrate=baudrate,
            node_id=1,
            auto_scan=True,
        ) if enable_serial and not sim_mode else None

        self.serial_n2 = SerialInterface(
            port=node2_serial_port,
            baudrate=baudrate,
            node_id=2,
            auto_scan=True,
        ) if enable_serial and not sim_mode else None

        # Telemetry ring-buffer & latest state
        self._lock = threading.Lock()
        self.node1_telemetry: Optional[Node1Telemetry] = Node1Telemetry()
        self.node2_telemetry: Optional[Node2Telemetry] = Node2Telemetry()
        self.threat_alert_telemetry: Optional[ThreatAlertTelemetry] = None
        self.spatial_alert_telemetry: Optional[SpatialAlertTelemetry] = None
        self.trajectory_alert_telemetry: Optional[TrajectoryAlertTelemetry] = None

        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        if self.can_if:
            self.can_if.register_callback(CAN_ID_FEEDBACK_N1, self._handle_can_feedback_n1)
            self.can_if.register_callback(CAN_ID_FEEDBACK_N2, self._handle_can_feedback_n2)
            self.can_if.register_callback(CAN_ID_THREAT_ALERT, self._handle_can_threat_alert)
            self.can_if.register_callback(CAN_ID_SPATIAL_ALERT, self._handle_can_spatial_alert)
            self.can_if.register_callback(CAN_ID_TRAJECTORY_ALERT, self._handle_can_trajectory_alert)

        if self.serial_n1:
            self.serial_n1.register_callback(self._handle_serial_feedback)
        if self.serial_n2:
            self.serial_n2.register_callback(self._handle_serial_feedback)

    def _handle_can_threat_alert(self, can_id: int, data: bytes) -> None:
        """Decode CAN ID 0x150 (Obstacle Threat Alert)."""
        if len(data) < 8:
            return
        try:
            dist_mm, class_enum, risk_score, estop_flag, seq_b, seq_w = struct.unpack("<HBBBBH", data)
            seq = (seq_w << 8) | seq_b
            with self._lock:
                self.threat_alert_telemetry = ThreatAlertTelemetry(
                    distance_mm=dist_mm,
                    class_enum=class_enum,
                    risk_score=risk_score,
                    emergency_stop=bool(estop_flag == 1),
                    seq=seq,
                    timestamp=time.time(),
                )
        except Exception as e:
            log.debug("Error unpacking CAN 0x150: %s", e)

    def _handle_can_spatial_alert(self, can_id: int, data: bytes) -> None:
        """Decode CAN ID 0x160 (3D Obstacle Spatial Alert)."""
        if len(data) < 8:
            return
        try:
            x_mm, z_mm, tti_ms, class_enum, threat_level = struct.unpack("<hHHBB", data[:8])
            with self._lock:
                self.spatial_alert_telemetry = SpatialAlertTelemetry(
                    x_mm=x_mm,
                    z_mm=z_mm,
                    tti_ms=tti_ms,
                    class_enum=class_enum,
                    threat_level=threat_level,
                    timestamp=time.time(),
                )
        except Exception as e:
            log.debug("Error unpacking CAN 0x160: %s", e)

    def _handle_can_trajectory_alert(self, can_id: int, data: bytes) -> None:
        """Decode CAN ID 0x170 (Trajectory Collision Alert)."""
        if len(data) < 8:
            return
        try:
            track_id, tti_ms, cross_x_mm, risk_level, action_flag = struct.unpack("<HHhBB", data[:8])
            with self._lock:
                self.trajectory_alert_telemetry = TrajectoryAlertTelemetry(
                    track_id=track_id,
                    tti_ms=tti_ms,
                    cross_x_mm=cross_x_mm,
                    risk_level=risk_level,
                    action_flag=action_flag,
                    timestamp=time.time(),
                )
        except Exception as e:
            log.debug("Error unpacking CAN 0x170: %s", e)


    def _handle_can_feedback_n1(self, can_id: int, data: bytes) -> None:
        """Decode CAN ID 0x300 (Node 1 Feedback)."""
        if len(data) < 8:
            return
        try:
            steer_scaled, curr_scaled, override, fault, uptime_ms = struct.unpack("<hhBBH", data)
            with self._lock:
                self.node1_telemetry = Node1Telemetry(
                    node_id=1,
                    actual_steering_deg=round(steer_scaled / 100.0, 2),
                    motor_current_amps=round(curr_scaled / 100.0, 2),
                    override_active=bool(override == 1),
                    fault_flags=fault,
                    uptime_ms=uptime_ms,
                    timestamp=time.time(),
                )
        except Exception as e:
            log.debug("Error unpacking CAN 0x300: %s", e)

    def _handle_can_feedback_n2(self, can_id: int, data: bytes) -> None:
        """Decode CAN ID 0x301 (Node 2 Feedback)."""
        if len(data) < 8:
            return
        try:
            speed_scaled, brake_scaled, override, fault, uptime_ms = struct.unpack("<hhBBH", data)
            with self._lock:
                self.node2_telemetry = Node2Telemetry(
                    node_id=2,
                    wheel_speed_mps=round(speed_scaled / 100.0, 2),
                    brake_pressure_psi=round(brake_scaled / 10.0, 1),
                    override_active=bool(override == 1),
                    fault_flags=fault,
                    uptime_ms=uptime_ms,
                    timestamp=time.time(),
                )
        except Exception as e:
            log.debug("Error unpacking CAN 0x301: %s", e)

    def _handle_serial_feedback(self, data: Dict[str, Any]) -> None:
        """Decode incoming Serial JSON response from either node."""
        node_id = data.get("node", 1)
        with self._lock:
            if node_id == 1:
                self.node1_telemetry = Node1Telemetry(
                    node_id=1,
                    actual_steering_deg=float(data.get("fb_a", 0.0)),
                    motor_current_amps=float(data.get("fb_b", 0.0)),
                    override_active=bool(data.get("override", 0) == 1),
                    fault_flags=int(data.get("fault", 0)),
                    uptime_ms=int(data.get("ack_ts", 0)),
                    timestamp=time.time(),
                )
            elif node_id == 2:
                self.node2_telemetry = Node2Telemetry(
                    node_id=2,
                    wheel_speed_mps=float(data.get("fb_a", 0.0)),
                    brake_pressure_psi=float(data.get("fb_b", 0.0)),
                    override_active=bool(data.get("override", 0) == 1),
                    fault_flags=int(data.get("fault", 0)),
                    uptime_ms=int(data.get("ack_ts", 0)),
                    timestamp=time.time(),
                )

    # ═══════════════════════════════════════════════════════════════════════
    # Unified Command Dispatch API
    # ═══════════════════════════════════════════════════════════════════════

    def send_node1_command(
        self,
        steering_deg: float,
        actuator_trim: float = 0.0,
        confidence: float = 0.95,
        mode: Union[str, int] = "AUTO",
        offset_m: float = 0.0,
        raw_counts: Optional[int] = None,
    ) -> bool:
        """
        Send primary control packet to Node 1 (Steering).
        Attempts CAN first; automatically fails over to Serial if CAN drops.
        Uses ActuationMapper to map software steering degrees to physical actuator counts.
        """
        mode_val = MODE_STR_TO_INT.get(mode, mode) if isinstance(mode, str) else mode
        ts_ms = int(time.time() * 1000) % 65536

        # Determine counts value via ActuationMapper
        if raw_counts is not None:
            counts_val = int(np.clip(raw_counts, -self.actuation_mapper.counts_max, self.actuation_mapper.counts_max))
        else:
            counts_val = self.actuation_mapper.deg_to_counts(steering_deg)

        # 1. Primary: CAN Bus (ID 0x100)
        can_success = False
        if self.enable_can and self.can_if and self.can_if.is_connected:
            try:
                if self.actuation_mapper.use_counts_in_can:
                    steer_s = counts_val
                else:
                    steer_s = int(np.clip(steering_deg * 100, -3000, 3000))

                trim_s = int(np.clip(actuator_trim * 100, -10000, 10000))
                conf_s = int(np.clip(confidence * 255, 0, 255))
                payload = struct.pack("<hhBBH", steer_s, trim_s, conf_s, mode_val, ts_ms)

                can_success = self.can_if.send_frame(CAN_ID_CMD_NODE1, payload)
                if can_success:
                    self.consecutive_can_errors = 0
                    self.active_channel = "CAN"
                    return True
            except Exception as e:
                log.debug("CAN send_node1_command error: %s", e)

        # 2. Failover: USB Serial
        if not can_success:
            self.consecutive_can_errors += 1
            if self.consecutive_can_errors >= self.can_error_threshold:
                if self.active_channel != "SERIAL":
                    log.warning("CAN error threshold reached. FAILING OVER TO SERIAL!")
                    self.active_channel = "SERIAL"

            if self.serial_n1 and self.serial_n1.is_connected:
                packet = {
                    "node": 1,
                    "n": 1,
                    "cmd_a": round(steering_deg, 2),
                    "str": round(steering_deg, 2),
                    "cnt": counts_val,
                    "cmd_b": round(actuator_trim, 2),
                    "offset_m": round(offset_m, 3),
                    "off": round(offset_m, 3),
                    "confidence": round(confidence, 2),
                    "cnf": round(confidence, 2),
                    "mode": MODE_INT_TO_STR.get(mode_val, "AUTO"),
                    "m": mode_val,
                    "ts": ts_ms,
                }
                return self.serial_n1.send_json(packet)

        return False

    def send_node1_control(
        self,
        steering_deg: float,
        actuator_trim: float = 0.0,
        confidence: float = 0.95,
        mode: Union[str, int] = "AUTO",
        offset_m: float = 0.0,
        raw_counts: Optional[int] = None,
    ) -> bool:
        """Alias for send_node1_command conforming to control API."""
        return self.send_node1_command(
            steering_deg=steering_deg,
            actuator_trim=actuator_trim,
            confidence=confidence,
            mode=mode,
            offset_m=offset_m,
            raw_counts=raw_counts,
        )

    def send_node2_command(
        self,
        throttle_pct: float,
        brake_pct: float = 0.0,
        mode: Union[str, int] = "AUTO",
        aux_flags: int = 0,
    ) -> bool:
        """
        Send control packet to Node 2 (Throttle, Brake, Aux).
        """
        mode_val = MODE_STR_TO_INT.get(mode, mode) if isinstance(mode, str) else mode
        ts_ms = int(time.time() * 1000) % 65536

        # 1. Primary: CAN Bus (ID 0x200)
        can_success = False
        if self.enable_can and self.can_if and self.can_if.is_connected:
            try:
                throt_s = int(np.clip(throttle_pct * 100, 0, 10000))
                brake_s = int(np.clip(brake_pct * 100, 0, 10000))
                payload = struct.pack("<hhBBH", throt_s, brake_s, mode_val, aux_flags, ts_ms)

                can_success = self.can_if.send_frame(CAN_ID_CMD_NODE2, payload)
                if can_success:
                    return True
            except Exception as e:
                log.debug("CAN send_node2_command error: %s", e)

        # 2. Failover: USB Serial
        if not can_success and self.serial_n2 and self.serial_n2.is_connected:
            packet = {
                "node": 2,
                "n": 2,
                "cmd_a": round(throttle_pct, 2),
                "str": round(throttle_pct, 2),
                "cmd_b": round(brake_pct, 2),
                "offset_m": 0.0,
                "off": 0.0,
                "confidence": 1.0,
                "cnf": 1.0,
                "mode": MODE_INT_TO_STR.get(mode_val, "AUTO"),
                "m": mode_val,
                "ts": ts_ms,
            }
            return self.serial_n2.send_json(packet)

        return False

    def send_node2_control(
        self,
        throttle_pct: float,
        brake_pct: float = 0.0,
        mode: Union[str, int] = "AUTO",
        aux_flags: int = 0,
    ) -> bool:
        """Alias for send_node2_command conforming to control API."""
        return self.send_node2_command(
            throttle_pct=throttle_pct,
            brake_pct=brake_pct,
            mode=mode,
            aux_flags=aux_flags,
        )

    def send_heartbeat(self, health_code: int = 0) -> bool:
        """Broadcast system heartbeat on CAN ID 0x050."""
        if not self.enable_can or not self.can_if or not self.can_if.is_connected:
            return False

        cycle = int(time.time() * 10) % 65536
        payload = struct.pack("<BBH", 0xAA, health_code, cycle)
        return self.can_if.send_frame(CAN_ID_HEARTBEAT, payload)

    def send_threat_alert(
        self,
        distance_mm: int,
        class_enum: int,
        risk_score: int,
        emergency_stop: bool,
        seq: int = 0,
    ) -> bool:
        """
        Broadcast Obstacle Threat Alert frame on CAN ID 0x150.
        Schema:
          Bytes 0-1: Highest Risk Object Distance Y-px / mm (uint16, <H)
          Byte 2: Highest Risk Class Enum (uint8, B)
          Byte 3: Calculated Risk Score [0–100] (uint8, B)
          Byte 4: Emergency Stop Flag (0=Clear, 1=BRAKE_IMMINENT) (uint8, B)
          Bytes 5-7: Reserved / Sequence Counter (uint24, <BH)
        """
        if not self.enable_can or not self.can_if or not self.can_if.is_connected:
            return False

        dist_val = int(np.clip(distance_mm, 0, 65535))
        class_val = int(np.clip(class_enum, 0, 255))
        risk_val = int(np.clip(risk_score, 0, 100))
        estop_val = 1 if emergency_stop else 0
        seq_b0 = int(seq & 0xFF)
        seq_w1 = int((seq >> 8) & 0xFFFF)

        payload = struct.pack("<HBBBBH", dist_val, class_val, risk_val, estop_val, seq_b0, seq_w1)
        return self.can_if.send_frame(CAN_ID_THREAT_ALERT, payload)

    def get_threat_alert(self) -> Optional[ThreatAlertTelemetry]:
        """Retrieve thread-safe latest threat alert telemetry."""
        with self._lock:
            return self.threat_alert_telemetry

    def send_spatial_alert(
        self,
        x_mm: int,
        z_mm: int,
        tti_ms: int,
        class_enum: int = 0,
        threat_level: int = THREAT_LEVEL_CLEAR,
    ) -> bool:
        """
        Broadcast 3D Obstacle Spatial Alert frame on CAN ID 0x160.
        Schema:
          Bytes 0-1: Nearest Object X_mm (signed int16, lateral offset mm, <h)
          Bytes 2-3: Nearest Object Z_mm (uint16, forward distance mm, <H)
          Bytes 4-5: Collision Time-to-Impact TTI_ms (uint16, milliseconds, <H)
          Byte 6: Nearest Object Class Enum (uint8, B)
          Byte 7: Occupancy Threat Level Enum (0=CLEAR, 1=CAUTION, 2=WARNING, 3=CRITICAL_STOP) (uint8, B)
        """
        if not self.enable_can or not self.can_if or not self.can_if.is_connected:
            return False

        x_val = int(np.clip(x_mm, -32768, 32767))
        z_val = int(np.clip(z_mm, 0, 65535))
        tti_val = int(np.clip(tti_ms, 0, 65535))
        class_val = int(np.clip(class_enum, 0, 255))
        threat_val = int(np.clip(threat_level, 0, 3))

        payload = struct.pack("<hHHBB", x_val, z_val, tti_val, class_val, threat_val)
        return self.can_if.send_frame(CAN_ID_SPATIAL_ALERT, payload)

    def get_spatial_alert(self) -> Optional[SpatialAlertTelemetry]:
        """Retrieve thread-safe latest 3D spatial alert telemetry."""
        with self._lock:
            return self.spatial_alert_telemetry

    def send_trajectory_alert(
        self,
        track_id: int,
        tti_ms: int,
        cross_x_mm: int,
        risk_level: int = 0,
        action_flag: int = 0,
    ) -> bool:
        """
        Broadcast Trajectory Collision Alert frame on CAN ID 0x170.
        Schema:
          Bytes 0-1: Critical Object Track ID (uint16, <H)
          Bytes 2-3: Minimum Time-To-Impact TTI_ms (uint16, <H)
          Bytes 4-5: Projected Intersect X_mm (signed int16, <h)
          Byte 6: Collision Risk Level (0=CLEAR, 1=MONITOR, 2=WARNING, 3=IMMINENT_COLLISION) (uint8, B)
          Byte 7: Emergency Action Flag (0=NONE, 1=BRAKE_NOW, 2=SWERVE_LEFT, 3=SWERVE_RIGHT) (uint8, B)
        """
        if not self.enable_can or not self.can_if or not self.can_if.is_connected:
            return False

        t_id = int(np.clip(track_id, 0, 65535))
        t_ms = int(np.clip(tti_ms, 0, 65535))
        x_mm = int(np.clip(cross_x_mm, -32768, 32767))
        r_lvl = int(np.clip(risk_level, 0, 3))
        a_flg = int(np.clip(action_flag, 0, 3))

        payload = struct.pack("<HHhBB", t_id, t_ms, x_mm, r_lvl, a_flg)
        return self.can_if.send_frame(CAN_ID_TRAJECTORY_ALERT, payload)

    def get_trajectory_alert(self) -> Optional[TrajectoryAlertTelemetry]:
        """Retrieve thread-safe latest trajectory collision alert telemetry."""
        with self._lock:
            return self.trajectory_alert_telemetry

    def get_node_telemetry(self, node_id: int = 1) -> Optional[Union[Node1Telemetry, Node2Telemetry]]:
        """Retrieve thread-safe latest telemetry cache for node."""
        with self._lock:
            if node_id == 1:
                return self.node1_telemetry
            elif node_id == 2:
                return self.node2_telemetry
            return None


    def is_healthy(self) -> bool:
        """Returns True if at least one communication channel (CAN or Serial) is active."""
        can_ok = bool(self.can_if and self.can_if.is_connected)
        serial_ok = bool(
            (self.serial_n1 and self.serial_n1.is_connected) or
            (self.serial_n2 and self.serial_n2.is_connected)
        )
        return can_ok or serial_ok or self.sim_mode

    def close(self) -> None:
        """Clean shutdown of all communication interfaces."""
        if self.can_if:
            self.can_if.close()
        if self.serial_n1:
            self.serial_n1.close()
        if self.serial_n2:
            self.serial_n2.close()
        log.info("VehicleCommunicator shutdown complete.")
