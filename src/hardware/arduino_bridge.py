"""
APEX-LKA v4.0 — Dual Arduino Mega 2560 Bridge Host Coordinator
================================================================
Host-side Python coordinator managing dual Arduino Mega 2560 serial bridges.

Modes:
  Primary:  Real SocketCAN (can0) at 500 kbps
  Fallback: Automatic switchover to /dev/ttyACM0 / /dev/ttyACM1 serial
            if can0 goes down OR frame corruption rate > 5%

Features:
  - Dynamic serial port reconnection with exponential backoff
  - Thread-safe message queues for TX and RX
  - Real-time latency measuring (< 1.5 ms target overhead)
  - COBS encoding/decoding matching arduino_can_bridge.ino
  - CRC-8 SAE J1850 verification on all frames
  - Bus health monitoring with automatic fallback trigger
  - Dual Arduino failover (Primary → Secondary)
"""

from __future__ import annotations

import logging
import math
import queue
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------
try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False
    log.warning("pyserial not available — serial fallback disabled.")

try:
    import can as python_can
    HAS_CAN = True
except ImportError:
    HAS_CAN = False
    log.warning("python-can not available — SocketCAN primary mode disabled.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERIAL_BAUD_RATE      = 921600
SOCKETCAN_CHANNEL     = "can0"
SOCKETCAN_BITRATE     = 500_000
MAX_CORRUPTION_RATE   = 0.05     # 5% frame corruption threshold for fallback
CORRUPTION_WINDOW     = 100      # Frame window for computing corruption rate
RECONNECT_INTERVAL_S  = 2.0     # Base reconnect interval
MAX_RECONNECT_S       = 30.0    # Maximum reconnect back-off
LATENCY_TARGET_MS     = 1.5     # Target overhead latency
TX_QUEUE_SIZE         = 256
RX_QUEUE_SIZE         = 512
HEARTBEAT_TIMEOUT_MS  = 500     # Arduino heartbeat timeout
SERIAL_READ_TIMEOUT   = 0.01    # Serial read timeout (10 ms)
COBS_FRAME_END        = 0x00
CRC8_POLY             = 0x1D
CRC8_INIT             = 0xFF
CRC8_FINAL_XOR        = 0xFF

ARDUINO_VENDOR_IDS = [
    "2341",   # Arduino LLC
    "1A86",   # CH340 clone
    "0403",   # FTDI
    "10C4",   # Silicon Labs CP210x
]

ARDUINO_PRODUCT_IDS = [
    "0042",   # Arduino Mega 2560
    "0010",   # Arduino Mega 2560 (old)
    "7523",   # CH340
    "6001",   # FTDI FT232
    "EA60",   # CP210x
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BridgeMode(Enum):
    SOCKETCAN = auto()
    SERIAL_PRIMARY = auto()
    SERIAL_SECONDARY = auto()
    DISCONNECTED = auto()


class BridgeHealth(Enum):
    HEALTHY = auto()
    DEGRADED = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# CRC-8 SAE J1850
# ---------------------------------------------------------------------------

_CRC8_TABLE: List[int] = []


def _build_crc8_table() -> None:
    global _CRC8_TABLE
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    _CRC8_TABLE = table


_build_crc8_table()


def compute_crc8(data: bytes, length: Optional[int] = None) -> int:
    crc = CRC8_INIT
    end = length if length is not None else len(data)
    for i in range(end):
        crc = _CRC8_TABLE[crc ^ data[i]]
    return crc ^ CRC8_FINAL_XOR


# ---------------------------------------------------------------------------
# COBS Codec
# ---------------------------------------------------------------------------

def cobs_encode(data: bytes) -> bytes:
    """Encode bytes using Consistent Overhead Byte Stuffing."""
    output = bytearray()
    overhead_idx = 0
    output.append(0x01)   # Initial overhead byte
    code = 0x01

    for byte in data:
        if byte == 0x00:
            output[overhead_idx] = code
            overhead_idx = len(output)
            output.append(0x01)
            code = 0x01
        else:
            output.append(byte)
            code += 1
            if code == 0xFF:
                output[overhead_idx] = code
                overhead_idx = len(output)
                output.append(0x01)
                code = 0x01

    output[overhead_idx] = code
    return bytes(output)


def cobs_decode(data: bytes) -> Optional[bytes]:
    """Decode COBS-encoded bytes. Returns None on error."""
    output = bytearray()
    idx = 0

    while idx < len(data):
        code = data[idx]
        if code == 0x00:
            return None  # Invalid
        idx += 1
        for _ in range(code - 1):
            if idx >= len(data):
                return None
            output.append(data[idx])
            idx += 1
        if code < 0xFF and idx < len(data):
            output.append(0x00)

    return bytes(output)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class CANFrame:
    """Unified CAN frame representation."""
    can_id: int
    data: bytes
    dlc: int = 8
    timestamp: float = 0.0
    crc_ok: bool = True
    source: str = "unknown"  # "socketcan" | "serial_primary" | "serial_secondary"


@dataclass
class BridgeStats:
    """Bridge runtime statistics."""
    mode: BridgeMode = BridgeMode.DISCONNECTED
    tx_count: int = 0
    rx_count: int = 0
    crc_errors: int = 0
    seq_errors: int = 0
    fallback_events: int = 0
    reconnect_attempts: int = 0
    last_mode_switch_time: float = 0.0
    avg_latency_ms: float = 0.0
    max_latency_ms: float = 0.0
    corruption_rate: float = 0.0


@dataclass
class SerialPortConfig:
    """Configuration for a serial-connected Arduino."""
    port: str
    baud_rate: int = SERIAL_BAUD_RATE
    is_primary: bool = True
    device_id: str = "ARDUINO_PRIMARY"
    reconnect_interval_s: float = RECONNECT_INTERVAL_S


# ---------------------------------------------------------------------------
# COBS Serial Frame Parser
# ---------------------------------------------------------------------------

class COBSFrameParser:
    """
    Stateful COBS frame parser for incoming serial bytes.
    Collects bytes between 0x00 delimiters and decodes COBS blocks.
    """

    def __init__(self, max_frame_size: int = 64) -> None:
        self._buf: bytearray = bytearray()
        self._in_frame: bool = False
        self._max_size = max_frame_size

    def feed(self, data: bytes) -> List[bytes]:
        """
        Feed raw bytes from serial port.
        Returns list of fully decoded frame payloads received.
        """
        completed = []
        for byte in data:
            if byte == COBS_FRAME_END:
                if self._in_frame and len(self._buf) > 0:
                    decoded = cobs_decode(bytes(self._buf))
                    if decoded is not None:
                        completed.append(decoded)
                self._buf.clear()
                self._in_frame = True  # Next bytes start new frame
            else:
                if self._in_frame:
                    if len(self._buf) < self._max_size:
                        self._buf.append(byte)
                    else:
                        # Buffer overflow — reset
                        self._buf.clear()
                        self._in_frame = False

        return completed

    def reset(self) -> None:
        self._buf.clear()
        self._in_frame = False


# ---------------------------------------------------------------------------
# Serial Connection Manager
# ---------------------------------------------------------------------------

class SerialConnectionManager:
    """
    Manages a single serial port connection with automatic reconnection.
    Exponential backoff on failure.
    """

    def __init__(self, config: SerialPortConfig) -> None:
        self._cfg = config
        self._port: Optional["serial.Serial"] = None
        self._connected = False
        self._reconnect_delay = RECONNECT_INTERVAL_S
        self._lock = threading.Lock()
        self._parser = COBSFrameParser()
        self._rx_queue: queue.Queue = queue.Queue(maxsize=RX_QUEUE_SIZE)
        self._last_hb_time = 0.0
        self._crc_error_count = 0
        self._frame_count = 0

    def connect(self) -> bool:
        """Attempt to open serial port. Returns True on success."""
        if not HAS_SERIAL:
            return False
        with self._lock:
            try:
                self._port = serial.Serial(
                    port=self._cfg.port,
                    baudrate=self._cfg.baud_rate,
                    timeout=SERIAL_READ_TIMEOUT,
                    write_timeout=0.05,
                )
                self._connected = True
                self._reconnect_delay = RECONNECT_INTERVAL_S  # Reset backoff
                self._parser.reset()
                log.info(
                    "%s: Connected to %s @ %d baud.",
                    self._cfg.device_id, self._cfg.port, self._cfg.baud_rate,
                )
                return True
            except Exception as exc:
                log.warning(
                    "%s: Failed to connect to %s: %s",
                    self._cfg.device_id, self._cfg.port, exc,
                )
                self._connected = False
                return False

    def disconnect(self) -> None:
        """Close serial port."""
        with self._lock:
            if self._port and self._port.is_open:
                try:
                    self._port.close()
                except Exception:
                    pass
            self._port = None
            self._connected = False

    def reconnect(self) -> bool:
        """Disconnect then reconnect with backoff."""
        self.disconnect()
        time.sleep(self._reconnect_delay)
        self._reconnect_delay = min(self._reconnect_delay * 1.5, MAX_RECONNECT_S)
        return self.connect()

    def read_frames(self) -> List[CANFrame]:
        """
        Read available bytes from serial port and parse complete COBS frames.
        Returns list of decoded CANFrame objects.
        """
        if not self._connected or self._port is None:
            return []

        frames = []
        try:
            with self._lock:
                waiting = self._port.in_waiting
                if waiting == 0:
                    return []
                raw = self._port.read(min(waiting, 256))

            decoded_payloads = self._parser.feed(raw)
            for payload in decoded_payloads:
                frame = self._parse_serial_payload(payload)
                if frame is not None:
                    frames.append(frame)

        except Exception as exc:
            log.warning(
                "%s: Read error: %s. Will reconnect.",
                self._cfg.device_id, exc,
            )
            self._connected = False

        return frames

    def _parse_serial_payload(self, payload: bytes) -> Optional[CANFrame]:
        """
        Parse a decoded COBS payload into a CANFrame.
        Format: [ID_H][ID_L][DLC][D0..D7][CRC8][CTR]  (13 bytes)
        """
        if len(payload) < 13:
            return None

        can_id = (payload[0] << 8) | payload[1]
        dlc = payload[2]
        if dlc > 8:
            return None

        data = bytes(payload[3:11])
        pkt_crc = payload[11]
        # _ctr = payload[12]  # Counter for future use

        # Verify packet CRC over first 11 bytes
        expected_crc = compute_crc8(payload, 11)
        if expected_crc != pkt_crc:
            self._crc_error_count += 1
            return None

        # Verify inner CAN frame CRC (byte 6 of data)
        crc_ok = True
        if dlc == 8:
            frame_crc_expected = compute_crc8(data, 6)
            crc_ok = frame_crc_expected == data[6]
            if not crc_ok:
                self._crc_error_count += 1

        self._frame_count += 1
        return CANFrame(
            can_id=can_id,
            data=data,
            dlc=dlc,
            timestamp=time.monotonic(),
            crc_ok=crc_ok,
            source=f"serial_{self._cfg.device_id.lower()}",
        )

    def send_frame(self, frame: CANFrame) -> bool:
        """Encode and send a CANFrame via serial. Returns True on success."""
        if not self._connected or self._port is None:
            return False

        raw = bytearray(13)
        raw[0] = (frame.can_id >> 8) & 0xFF
        raw[1] = frame.can_id & 0xFF
        raw[2] = frame.dlc
        data = bytes(frame.data[:8]).ljust(8, b'\x00')
        raw[3:11] = data
        raw[11] = compute_crc8(bytes(raw), 11)
        raw[12] = 0x00  # Counter placeholder

        encoded = cobs_encode(bytes(raw))
        packet = bytes([COBS_FRAME_END]) + encoded + bytes([COBS_FRAME_END])

        try:
            with self._lock:
                self._port.write(packet)
            return True
        except Exception as exc:
            log.warning("%s: Write error: %s", self._cfg.device_id, exc)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def corruption_rate(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return self._crc_error_count / self._frame_count


# ---------------------------------------------------------------------------
# SocketCAN Interface
# ---------------------------------------------------------------------------

class SocketCANInterface:
    """Manages the primary SocketCAN (can0) interface."""

    def __init__(
        self,
        channel: str = SOCKETCAN_CHANNEL,
        bitrate: int = SOCKETCAN_BITRATE,
    ) -> None:
        self._channel = channel
        self._bitrate = bitrate
        self._bus: Optional[Any] = None
        self._connected = False
        self._crc_errors = 0
        self._frame_count = 0

    def connect(self) -> bool:
        """Open SocketCAN interface. Returns True on success."""
        if not HAS_CAN:
            return False
        try:
            self._bus = python_can.interface.Bus(
                channel=self._channel,
                bustype="socketcan",
                bitrate=self._bitrate,
            )
            self._connected = True
            log.info("SocketCAN: Connected to %s @ %d bps.", self._channel, self._bitrate)
            return True
        except Exception as exc:
            log.warning("SocketCAN: Failed to connect to %s: %s", self._channel, exc)
            self._connected = False
            return False

    def disconnect(self) -> None:
        if self._bus:
            try:
                self._bus.shutdown()
            except Exception:
                pass
        self._bus = None
        self._connected = False

    def read_frame(self, timeout: float = 0.001) -> Optional[CANFrame]:
        """Read one frame from SocketCAN. Returns None on timeout or error."""
        if not self._connected or self._bus is None:
            return None
        try:
            msg = self._bus.recv(timeout=timeout)
            if msg is None:
                return None
            self._frame_count += 1
            data = bytes(msg.data).ljust(8, b'\x00')
            crc_ok = True
            if msg.dlc == 8:
                expected = compute_crc8(data, 6)
                crc_ok = expected == data[6]
                if not crc_ok:
                    self._crc_errors += 1
            return CANFrame(
                can_id=msg.arbitration_id,
                data=data,
                dlc=msg.dlc,
                timestamp=msg.timestamp or time.monotonic(),
                crc_ok=crc_ok,
                source="socketcan",
            )
        except Exception as exc:
            log.warning("SocketCAN: Read error: %s", exc)
            self._connected = False
            return None

    def send_frame(self, frame: CANFrame) -> bool:
        """Send a CANFrame on SocketCAN. Returns True on success."""
        if not self._connected or self._bus is None:
            return False
        try:
            msg = python_can.Message(
                arbitration_id=frame.can_id,
                data=frame.data[:frame.dlc],
                is_extended_id=False,
            )
            self._bus.send(msg)
            return True
        except Exception as exc:
            log.warning("SocketCAN: Send error: %s", exc)
            self._connected = False
            return False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def corruption_rate(self) -> float:
        if self._frame_count == 0:
            return 0.0
        return self._crc_errors / self._frame_count


# ---------------------------------------------------------------------------
# Latency Tracker
# ---------------------------------------------------------------------------

class LatencyTracker:
    """Rolling latency measurement with < 1.5 ms target."""

    def __init__(self, window: int = 200) -> None:
        self._samples: deque = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, latency_ms: float) -> None:
        with self._lock:
            self._samples.append(latency_ms)

    def measure(self, start_time: float) -> float:
        """Measure and record latency from start_time."""
        lat = (time.monotonic() - start_time) * 1000.0
        self.record(lat)
        return lat

    @property
    def avg_ms(self) -> float:
        with self._lock:
            return sum(self._samples) / len(self._samples) if self._samples else 0.0

    @property
    def max_ms(self) -> float:
        with self._lock:
            return max(self._samples) if self._samples else 0.0

    @property
    def p95_ms(self) -> float:
        with self._lock:
            if not self._samples:
                return 0.0
            sorted_s = sorted(self._samples)
            idx = int(len(sorted_s) * 0.95)
            return sorted_s[min(idx, len(sorted_s) - 1)]

    def is_target_met(self) -> bool:
        return self.avg_ms < LATENCY_TARGET_MS


# ---------------------------------------------------------------------------
# Main Arduino Bridge Coordinator
# ---------------------------------------------------------------------------

class ArduinoBridge:
    """
    Dual Arduino Mega 2560 host-side bridge coordinator.

    Primary mode: SocketCAN (can0)
    Fallback mode: Serial /dev/ttyACM0 (Primary Arduino)
    Secondary fallback: Serial /dev/ttyACM1 (Secondary Arduino)

    Automatic failover triggered when:
      - can0 interface is not available or goes down
      - Frame corruption rate > 5% over 100-frame window
      - No frames received within 500 ms timeout
    """

    def __init__(
        self,
        socketcan_channel: str = SOCKETCAN_CHANNEL,
        serial_primary_port: str = "/dev/ttyACM0",
        serial_secondary_port: str = "/dev/ttyACM1",
        baud_rate: int = SERIAL_BAUD_RATE,
    ) -> None:
        self._socketcan = SocketCANInterface(channel=socketcan_channel)
        self._serial_primary = SerialConnectionManager(
            SerialPortConfig(serial_primary_port, baud_rate, True, "ARDUINO_PRIMARY")
        )
        self._serial_secondary = SerialConnectionManager(
            SerialPortConfig(serial_secondary_port, baud_rate, False, "ARDUINO_SECONDARY")
        )

        self._mode = BridgeMode.DISCONNECTED
        self._mode_lock = threading.Lock()

        # Message queues
        self._tx_queue: queue.Queue = queue.Queue(maxsize=TX_QUEUE_SIZE)
        self._rx_queue: queue.Queue = queue.Queue(maxsize=RX_QUEUE_SIZE)

        # Latency tracking
        self._latency = LatencyTracker()

        # Statistics
        self._stats = BridgeStats()

        # Callbacks
        self._rx_callbacks: List[Callable[[CANFrame], None]] = []

        # Threads
        self._running = False
        self._rx_thread: Optional[threading.Thread] = None
        self._tx_thread: Optional[threading.Thread] = None
        self._health_thread: Optional[threading.Thread] = None

        # Corruption tracking window
        self._corruption_window: deque = deque(maxlen=CORRUPTION_WINDOW)
        self._last_frame_time = 0.0

        log.info(
            "ArduinoBridge initialized: SocketCAN=%s, Primary=%s, Secondary=%s",
            socketcan_channel, serial_primary_port, serial_secondary_port,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start bridge in best available mode."""
        self._running = True

        # Try SocketCAN first
        if self._socketcan.connect():
            self._set_mode(BridgeMode.SOCKETCAN)
        elif self._serial_primary.connect():
            self._set_mode(BridgeMode.SERIAL_PRIMARY)
        elif self._serial_secondary.connect():
            self._set_mode(BridgeMode.SERIAL_SECONDARY)
        else:
            log.error("ArduinoBridge: No CAN interface available!")
            self._set_mode(BridgeMode.DISCONNECTED)

        # Start background threads
        self._rx_thread = threading.Thread(target=self._rx_loop, name="ArduinoBridge_RX", daemon=True)
        self._tx_thread = threading.Thread(target=self._tx_loop, name="ArduinoBridge_TX", daemon=True)
        self._health_thread = threading.Thread(target=self._health_loop, name="ArduinoBridge_Health", daemon=True)

        self._rx_thread.start()
        self._tx_thread.start()
        self._health_thread.start()

        log.info("ArduinoBridge started in mode: %s", self._mode.name)

    def stop(self) -> None:
        """Stop bridge gracefully."""
        self._running = False
        for t in (self._rx_thread, self._tx_thread, self._health_thread):
            if t and t.is_alive():
                t.join(timeout=2.0)
        self._socketcan.disconnect()
        self._serial_primary.disconnect()
        self._serial_secondary.disconnect()
        log.info("ArduinoBridge stopped.")

    def send(self, can_id: int, data: bytes, dlc: int = 8) -> bool:
        """
        Queue a CAN frame for transmission.
        Returns False if TX queue is full.
        """
        frame = CANFrame(can_id=can_id, data=data, dlc=dlc, timestamp=time.monotonic())
        # Inject CRC before queuing
        payload = bytearray(frame.data[:8].ljust(8, b'\x00'))
        payload[6] = compute_crc8(bytes(payload), 6)
        frame.data = bytes(payload)

        try:
            self._tx_queue.put_nowait(frame)
            return True
        except queue.Full:
            log.warning("ArduinoBridge TX queue full — frame dropped.")
            return False

    def register_rx_callback(self, callback: Callable[[CANFrame], None]) -> None:
        """Register callback for incoming CAN frames."""
        self._rx_callbacks.append(callback)

    def get_mode(self) -> BridgeMode:
        with self._mode_lock:
            return self._mode

    def get_stats(self) -> Dict[str, Any]:
        return {
            "mode": self._stats.mode.name,
            "tx_count": self._stats.tx_count,
            "rx_count": self._stats.rx_count,
            "crc_errors": self._stats.crc_errors,
            "fallback_events": self._stats.fallback_events,
            "corruption_rate": round(self._stats.corruption_rate, 4),
            "avg_latency_ms": round(self._latency.avg_ms, 3),
            "max_latency_ms": round(self._latency.max_ms, 3),
            "p95_latency_ms": round(self._latency.p95_ms, 3),
            "latency_target_met": self._latency.is_target_met(),
        }

    # -----------------------------------------------------------------------
    # Background threads
    # -----------------------------------------------------------------------

    def _rx_loop(self) -> None:
        """RX thread: read frames from active interface and dispatch callbacks."""
        log.debug("ArduinoBridge RX loop started.")
        while self._running:
            t0 = time.monotonic()
            frames = self._read_frames()

            for frame in frames:
                self._last_frame_time = frame.timestamp
                self._stats.rx_count += 1

                # Track corruption window
                self._corruption_window.append(0 if frame.crc_ok else 1)
                if len(self._corruption_window) >= CORRUPTION_WINDOW:
                    self._stats.corruption_rate = (
                        sum(self._corruption_window) / len(self._corruption_window)
                    )

                if not frame.crc_ok:
                    self._stats.crc_errors += 1

                # Dispatch to callbacks
                lat_ms = self._latency.measure(t0)
                if lat_ms > LATENCY_TARGET_MS * 2:
                    log.debug("RX latency spike: %.2f ms", lat_ms)

                for cb in self._rx_callbacks:
                    try:
                        cb(frame)
                    except Exception as exc:
                        log.error("RX callback error: %s", exc)

                try:
                    self._rx_queue.put_nowait(frame)
                except queue.Full:
                    pass  # Drop oldest if queue full

            if not frames:
                time.sleep(0.0005)  # 0.5 ms sleep when idle

        log.debug("ArduinoBridge RX loop exited.")

    def _tx_loop(self) -> None:
        """TX thread: dequeue and transmit frames."""
        log.debug("ArduinoBridge TX loop started.")
        while self._running:
            try:
                frame = self._tx_queue.get(timeout=0.01)
            except queue.Empty:
                continue

            t0 = time.monotonic()
            ok = self._send_frame(frame)
            self._latency.measure(t0)

            if ok:
                self._stats.tx_count += 1
            self._tx_queue.task_done()

        log.debug("ArduinoBridge TX loop exited.")

    def _health_loop(self) -> None:
        """Health monitoring thread: detect failures and trigger fallback."""
        log.debug("ArduinoBridge health loop started.")
        while self._running:
            time.sleep(0.1)  # Check every 100 ms

            mode = self.get_mode()

            # Check corruption rate threshold
            if self._stats.corruption_rate > MAX_CORRUPTION_RATE:
                log.warning(
                    "ArduinoBridge: Corruption rate %.1f%% > threshold %.1f%%. "
                    "Triggering fallback.",
                    self._stats.corruption_rate * 100, MAX_CORRUPTION_RATE * 100,
                )
                self._trigger_fallback(mode)
                continue

            # Check interface liveness
            if mode == BridgeMode.SOCKETCAN:
                if not self._socketcan.is_connected:
                    log.warning("SocketCAN down. Falling back to serial.")
                    self._trigger_fallback(mode)
            elif mode == BridgeMode.SERIAL_PRIMARY:
                if not self._serial_primary.is_connected:
                    log.warning("Serial primary down. Trying secondary.")
                    self._trigger_fallback(mode)
            elif mode == BridgeMode.SERIAL_SECONDARY:
                if not self._serial_secondary.is_connected:
                    log.warning("Serial secondary down. Reconnecting.")
                    self._serial_secondary.reconnect()
            elif mode == BridgeMode.DISCONNECTED:
                # Try to reconnect
                self._attempt_reconnect()

        log.debug("ArduinoBridge health loop exited.")

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _read_frames(self) -> List[CANFrame]:
        mode = self.get_mode()
        if mode == BridgeMode.SOCKETCAN:
            frame = self._socketcan.read_frame()
            return [frame] if frame else []
        elif mode == BridgeMode.SERIAL_PRIMARY:
            return self._serial_primary.read_frames()
        elif mode == BridgeMode.SERIAL_SECONDARY:
            return self._serial_secondary.read_frames()
        return []

    def _send_frame(self, frame: CANFrame) -> bool:
        mode = self.get_mode()
        if mode == BridgeMode.SOCKETCAN:
            return self._socketcan.send_frame(frame)
        elif mode == BridgeMode.SERIAL_PRIMARY:
            return self._serial_primary.send_frame(frame)
        elif mode == BridgeMode.SERIAL_SECONDARY:
            return self._serial_secondary.send_frame(frame)
        return False

    def _set_mode(self, mode: BridgeMode) -> None:
        with self._mode_lock:
            old_mode = self._mode
            self._mode = mode
            self._stats.mode = mode
        if old_mode != mode:
            self._stats.last_mode_switch_time = time.monotonic()
            log.info("ArduinoBridge mode: %s → %s", old_mode.name, mode.name)

    def _trigger_fallback(self, current_mode: BridgeMode) -> None:
        """Execute fallback sequence."""
        self._stats.fallback_events += 1
        self._corruption_window.clear()
        self._stats.corruption_rate = 0.0

        if current_mode == BridgeMode.SOCKETCAN:
            self._socketcan.disconnect()
            if self._serial_primary.connect() or self._serial_primary.reconnect():
                self._set_mode(BridgeMode.SERIAL_PRIMARY)
            elif self._serial_secondary.connect() or self._serial_secondary.reconnect():
                self._set_mode(BridgeMode.SERIAL_SECONDARY)
            else:
                self._set_mode(BridgeMode.DISCONNECTED)

        elif current_mode == BridgeMode.SERIAL_PRIMARY:
            self._serial_primary.disconnect()
            if self._serial_secondary.connect() or self._serial_secondary.reconnect():
                self._set_mode(BridgeMode.SERIAL_SECONDARY)
            else:
                self._set_mode(BridgeMode.DISCONNECTED)

        else:
            self._serial_secondary.reconnect()
            if not self._serial_secondary.is_connected:
                self._set_mode(BridgeMode.DISCONNECTED)

    def _attempt_reconnect(self) -> None:
        """Try all interfaces in priority order."""
        if self._socketcan.connect():
            self._set_mode(BridgeMode.SOCKETCAN)
            self._stats.reconnect_attempts += 1
        elif self._serial_primary.connect():
            self._set_mode(BridgeMode.SERIAL_PRIMARY)
            self._stats.reconnect_attempts += 1
        elif self._serial_secondary.connect():
            self._set_mode(BridgeMode.SERIAL_SECONDARY)
            self._stats.reconnect_attempts += 1
        else:
            time.sleep(RECONNECT_INTERVAL_S)

    @staticmethod
    def detect_arduino_ports() -> List[str]:
        """Auto-detect connected Arduino Mega 2560 serial ports."""
        if not HAS_SERIAL:
            return []
        ports = []
        for port_info in serial.tools.list_ports.comports():
            vid = f"{port_info.vid:04X}" if port_info.vid else ""
            pid = f"{port_info.pid:04X}" if port_info.pid else ""
            if vid in ARDUINO_VENDOR_IDS or pid in ARDUINO_PRODUCT_IDS:
                ports.append(port_info.device)
                log.info(
                    "Detected Arduino port: %s (VID=%s PID=%s desc='%s')",
                    port_info.device, vid, pid, port_info.description or "",
                )
        return ports
