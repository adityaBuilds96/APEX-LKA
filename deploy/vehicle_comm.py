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
    ) -> None:
        self.sim_mode = sim_mode
        self.enable_can = enable_can
        self.enable_serial = enable_serial

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

        self._setup_callbacks()

    def _setup_callbacks(self) -> None:
        if self.can_if:
            self.can_if.register_callback(CAN_ID_FEEDBACK_N1, self._handle_can_feedback_n1)
            self.can_if.register_callback(CAN_ID_FEEDBACK_N2, self._handle_can_feedback_n2)

        if self.serial_n1:
            self.serial_n1.register_callback(self._handle_serial_feedback)
        if self.serial_n2:
            self.serial_n2.register_callback(self._handle_serial_feedback)

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
    ) -> bool:
        """
        Send primary control packet to Node 1 (Steering).
        Attempts CAN first; automatically fails over to Serial if CAN drops.
        """
        mode_val = MODE_STR_TO_INT.get(mode, mode) if isinstance(mode, str) else mode
        ts_ms = int(time.time() * 1000) % 65536

        # 1. Primary: CAN Bus (ID 0x100)
        can_success = False
        if self.enable_can and self.can_if and self.can_if.is_connected:
            try:
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
                    "cmd_a": round(steering_deg, 2),
                    "cmd_b": round(actuator_trim, 2),
                    "offset_m": round(offset_m, 3),
                    "confidence": round(confidence, 2),
                    "mode": MODE_INT_TO_STR.get(mode_val, "AUTO"),
                    "ts": ts_ms,
                }
                return self.serial_n1.send_json(packet)

        return False

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
                "cmd_a": round(throttle_pct, 2),
                "cmd_b": round(brake_pct, 2),
                "offset_m": 0.0,
                "confidence": 1.0,
                "mode": MODE_INT_TO_STR.get(mode_val, "AUTO"),
                "ts": ts_ms,
            }
            return self.serial_n2.send_json(packet)

        return False

    def send_heartbeat(self, health_code: int = 0) -> bool:
        """Broadcast system heartbeat on CAN ID 0x050."""
        if not self.enable_can or not self.can_if or not self.can_if.is_connected:
            return False

        cycle = int(time.time() * 10) % 65536
        payload = struct.pack("<BBH", 0xAA, health_code, cycle)
        return self.can_if.send_frame(CAN_ID_HEARTBEAT, payload)

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
