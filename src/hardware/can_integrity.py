"""
APEX-LKA v4.0 — CAN Hardware Framing Integrity Engine
======================================================
High-performance binary CAN encoder/decoder engine with:
  - Lookup-table driven CRC-8 SAE J1850 (poly=0x1D) — O(1) per byte
  - Per-ID sequence counter state machines with anomaly classification:
      VALID, OUT_OF_ORDER, FRAME_DROP, RESET
  - Struct-based binary payload packing/unpacking with NaN/Inf guarding
  - Signal boundary clamping against DBC-defined min/max
  - Bus line status tracking: message age, bus load, frame loss rate
  - Safe fallback flags when message age > 100 ms

Designed for real-time operation on NVIDIA Jetson Orin NX.
Thread-safe where noted; individual encode/decode ops are GIL-safe.
"""

from __future__ import annotations

import logging
import math
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

__all__ = [
    "CRC8Engine",
    "SequenceVerifier",
    "SequenceStatus",
    "CANSignalSpec",
    "CANMessageSpec",
    "CANPayloadCodec",
    "LineStatusTracker",
    "CANIntegrityEngine",
    "APEX_MESSAGE_SPECS",
    "HeartbeatFrame",
    "SteeringFrame",
    "PriorityRiskFrame",
    "SpatialStateFrame",
    "TrajectoryPolyFrame",
    "VehicleKinematicsFrame",
    "PlanControlFrame",
    "MTi680OrientFrame",
    "MTi680VelFrame",
    "RadarHeaderFrame",
    "RadarTrackFrame",
    "compute_crc8",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRC8_POLYNOMIAL   = 0x1D     # SAE J1850
CRC8_INIT_VALUE   = 0xFF
CRC8_FINAL_XOR    = 0xFF
COUNTER_MODULUS   = 16       # 4-bit counter
STALE_THRESHOLD_S = 0.100    # 100 ms message age threshold
FRAME_LOSS_WINDOW = 1.0      # Moving window size for frame loss calculation (seconds)
BUS_SPEED_BPS     = 500_000  # 500 kbps
BITS_PER_FRAME    = 111      # Nominal bits per standard CAN frame (including overhead)


# ---------------------------------------------------------------------------
# CRC-8 Engine (SAE J1850, poly=0x1D)
# ---------------------------------------------------------------------------

class CRC8Engine:
    """
    Lookup-table driven CRC-8 computation.
    Polynomial: 0x1D (SAE J1850).
    Init: 0xFF, Final XOR: 0xFF.

    O(n) where each byte lookup is O(1), no bit-level loop.
    """

    _TABLE: List[int] = []
    _TABLE_LOCK = threading.Lock()

    @classmethod
    def _build_table(cls) -> None:
        """Build 256-entry lookup table (lazy, thread-safe)."""
        if cls._TABLE:
            return
        with cls._TABLE_LOCK:
            if cls._TABLE:
                return
            table = []
            for i in range(256):
                crc = i
                for _ in range(8):
                    if crc & 0x80:
                        crc = ((crc << 1) ^ CRC8_POLYNOMIAL) & 0xFF
                    else:
                        crc = (crc << 1) & 0xFF
                table.append(crc)
            cls._TABLE = table
            log.debug("CRC-8 SAE J1850 lookup table built (256 entries).")

    def __init__(self) -> None:
        self._build_table()

    def compute(self, data: bytes, length: Optional[int] = None) -> int:
        """
        Compute CRC-8 SAE J1850 over data[0:length].
        If length is None, processes all bytes.
        Returns 8-bit CRC.
        """
        crc = CRC8_INIT_VALUE
        table = self._TABLE
        end = length if length is not None else len(data)
        for i in range(end):
            crc = table[crc ^ data[i]]
        return crc ^ CRC8_FINAL_XOR

    def compute_frame_crc(self, payload: bytes) -> int:
        """
        Compute CRC-8 over the first 6 bytes of an 8-byte CAN frame.
        Byte 6 is the CRC field, byte 7 is the counter nibble.
        """
        assert len(payload) == 8, f"Expected 8-byte frame, got {len(payload)}"
        return self.compute(payload, length=6)

    def verify_frame(self, payload: bytes) -> bool:
        """
        Verify frame CRC. Byte 6 must match CRC-8 over bytes 0-5.
        """
        expected = self.compute_frame_crc(payload)
        actual = payload[6]
        if expected != actual:
            log.debug("CRC mismatch: expected=0x%02X, actual=0x%02X", expected, actual)
            return False
        return True

    def inject_crc(self, payload: bytearray) -> None:
        """
        Compute and inject CRC-8 into byte 6 of a mutable payload bytearray.
        Byte 7 (Counter nibble) must already be set before calling.
        """
        payload[6] = self.compute(bytes(payload), length=6)

    @property
    def table(self) -> List[int]:
        """Expose lookup table for testing/auditing."""
        return list(self._TABLE)


_default_crc8_engine: Optional[CRC8Engine] = None


def compute_crc8(data: bytes, length: Optional[int] = None) -> int:
    """Compute CRC-8 SAE J1850 using the default module-level engine."""
    global _default_crc8_engine
    if _default_crc8_engine is None:
        _default_crc8_engine = CRC8Engine()
    return _default_crc8_engine.compute(data, length)


# ---------------------------------------------------------------------------
# Sequence Counter State Machine
# ---------------------------------------------------------------------------

class SequenceStatus(IntEnum):
    VALID       = 0   # Expected next counter
    OUT_OF_ORDER = 1  # Repeated or small jump within window
    FRAME_DROP  = 2   # Gap > 1 frame detected
    RESET       = 3   # Counter reset to 0 safely


@dataclass
class SequenceState:
    """Per-message-ID rolling sequence counter state."""
    can_id: int
    last_counter: int = -1        # -1 = never received
    total_frames: int = 0
    dropped_frames: int = 0
    out_of_order_frames: int = 0
    reset_events: int = 0
    last_timestamp: float = 0.0


class SequenceVerifier:
    """
    Per-ID rolling sequence counter state machine.

    4-bit counter (0-15) wrapping:
      VALID       : C_new == (C_old + 1) % 16
      OUT_OF_ORDER: C_new == C_old (repeat) or small backward jump
      FRAME_DROP  : gap > 1 and gap < 15 (not a reset)
      RESET       : C_new == 0 and C_old != 15 (safe reset detected)
    """

    def __init__(self) -> None:
        self._states: Dict[int, SequenceState] = {}
        self._lock = threading.Lock()

    def get_state(self, can_id: int) -> SequenceState:
        """Get or create per-ID state."""
        with self._lock:
            if can_id not in self._states:
                self._states[can_id] = SequenceState(can_id=can_id)
            return self._states[can_id]

    def verify(self, can_id: int, new_counter: int, timestamp: float = 0.0) -> SequenceStatus:
        """
        Verify new_counter for the given CAN ID.
        Returns SequenceStatus and updates internal state.
        """
        with self._lock:
            if can_id not in self._states:
                self._states[can_id] = SequenceState(can_id=can_id)
            state = self._states[can_id]

            # First ever frame
            if state.last_counter == -1:
                state.last_counter = new_counter
                state.total_frames = 1
                state.last_timestamp = timestamp
                return SequenceStatus.VALID

            state.total_frames += 1
            expected = (state.last_counter + 1) % COUNTER_MODULUS
            gap = (new_counter - state.last_counter) % COUNTER_MODULUS

            status: SequenceStatus

            if new_counter == expected:
                # Normal: exactly next counter
                status = SequenceStatus.VALID

            elif gap == 0:
                # Repeat: same counter received twice
                state.out_of_order_frames += 1
                status = SequenceStatus.OUT_OF_ORDER

            elif new_counter == 0 and state.last_counter not in (14, 15):
                # Counter jumped back to 0 unexpectedly → node reset
                state.reset_events += 1
                state.dropped_frames += max(0, gap - 1)
                status = SequenceStatus.RESET

            elif 1 < gap < COUNTER_MODULUS // 2:
                # Forward jump > 1 → dropped frames
                missed = gap - 1
                state.dropped_frames += missed
                log.warning(
                    "CAN 0x%03X: Frame drop detected. Gap=%d, missed=%d frames.",
                    can_id, gap, missed,
                )
                status = SequenceStatus.FRAME_DROP

            else:
                # Backward jump within window → out of order
                state.out_of_order_frames += 1
                status = SequenceStatus.OUT_OF_ORDER

            state.last_counter = new_counter
            state.last_timestamp = timestamp
            return status

    def reset_id(self, can_id: int) -> None:
        """Reset state for a specific CAN ID (e.g., after node reboot)."""
        with self._lock:
            self._states.pop(can_id, None)
        log.info("CAN 0x%03X: Sequence state reset.", can_id)

    def get_drop_rate(self, can_id: int) -> float:
        """Return frame drop rate [0.0, 1.0] for given CAN ID."""
        with self._lock:
            state = self._states.get(can_id)
            if state is None or state.total_frames == 0:
                return 0.0
            return state.dropped_frames / state.total_frames

    def get_all_stats(self) -> Dict[int, Dict[str, Any]]:
        """Return statistics for all tracked IDs."""
        with self._lock:
            return {
                can_id: {
                    "total_frames": s.total_frames,
                    "dropped_frames": s.dropped_frames,
                    "out_of_order": s.out_of_order_frames,
                    "resets": s.reset_events,
                    "drop_rate": s.dropped_frames / max(1, s.total_frames),
                    "last_counter": s.last_counter,
                    "last_ts": s.last_timestamp,
                }
                for can_id, s in self._states.items()
            }


# ---------------------------------------------------------------------------
# Signal & Message Specs (DBC-mirrored)
# ---------------------------------------------------------------------------

@dataclass
class CANSignalSpec:
    """Signal specification matching DBC SG_ definition."""
    name: str
    start_bit: int       # Little-endian start bit (LSB position)
    length: int          # Bit length
    factor: float        # Scale
    offset: float        # Offset
    min_val: float       # Physical minimum (for clamping)
    max_val: float       # Physical maximum (for clamping)
    unit: str = ""
    is_signed: bool = False

    def encode(self, physical: float) -> int:
        """Physical value → raw integer, with clamping."""
        if math.isnan(physical) or math.isinf(physical):
            log.warning("Signal '%s': NaN/Inf input replaced with min_val.", self.name)
            physical = self.min_val
        physical = max(self.min_val, min(self.max_val, physical))
        raw = (physical - self.offset) / self.factor
        return int(round(raw))

    def decode(self, raw: int) -> float:
        """Raw integer → physical value with clamping."""
        physical = raw * self.factor + self.offset
        return max(self.min_val, min(self.max_val, physical))

    @property
    def max_raw(self) -> int:
        return (1 << self.length) - 1


@dataclass
class CANMessageSpec:
    """Message specification matching DBC BO_ definition."""
    can_id: int
    name: str
    dlc: int
    signals: List[CANSignalSpec]
    cycle_ms: int = 33

    def get_signal(self, name: str) -> Optional[CANSignalSpec]:
        for s in self.signals:
            if s.name == name:
                return s
        return None


# ---------------------------------------------------------------------------
# APEX v4 Message Specifications (canonical, matches apex_v4.dbc)
# ---------------------------------------------------------------------------

APEX_MESSAGE_SPECS: Dict[int, CANMessageSpec] = {
    0x050: CANMessageSpec(0x050, "Heartbeat", 8, cycle_ms=10, signals=[
        CANSignalSpec("NodeID",          0,  8, 1.0,   0.0, 0, 255),
        CANSignalSpec("SystemState",     8,  8, 1.0,   0.0, 0, 5),
        CANSignalSpec("ErrorFlags",     16, 16, 1.0,   0.0, 0, 65535),
        CANSignalSpec("WatchdogCounter",32,  8, 1.0,   0.0, 0, 255),
        CANSignalSpec("Reserved0",      40,  8, 1.0,   0.0, 0, 255),
        CANSignalSpec("CRC8",           48,  8, 1.0,   0.0, 0, 255),
        CANSignalSpec("Counter",        56,  4, 1.0,   0.0, 0, 15),
    ]),
    0x100: CANMessageSpec(0x100, "Steering_Primary", 8, cycle_ms=33, signals=[
        CANSignalSpec("TargetSteeringAngle",  0, 16, 0.001, -28.0, -28.0,  28.0, "deg"),
        CANSignalSpec("AngularVelocityLimit", 16, 12, 0.1,   0.0,   0.0, 200.0, "deg/s"),
        CANSignalSpec("TorqueCommand",        28, 12, 0.01, -10.0, -10.0,  10.0, "Nm"),
        CANSignalSpec("SteeringMode",         40,  4, 1.0,   0.0,   0.0,   4.0),
        CANSignalSpec("Reserved1",            44,  4, 1.0,   0.0,   0.0,  15.0),
        CANSignalSpec("CRC8",                 48,  8, 1.0,   0.0,   0.0, 255.0),
        CANSignalSpec("Counter",              56,  4, 1.0,   0.0,   0.0,  15.0),
        CANSignalSpec("Reserved2",            60,  4, 1.0,   0.0,   0.0,  15.0),
    ]),
    0x150: CANMessageSpec(0x150, "Priority_Risk", 8, cycle_ms=33, signals=[
        CANSignalSpec("DynamicRiskLevel",   0,  8, 1.0,   0.0,   0.0, 255.0),
        CANSignalSpec("TTC",                8, 16, 0.001, 0.0,   0.0,  10.0, "s"),
        CANSignalSpec("EmergencyBrakeFlag",24,  1, 1.0,   0.0,   0.0,   1.0),
        CANSignalSpec("RiskZoneID",        25,  4, 1.0,   0.0,   0.0,  15.0),
        CANSignalSpec("Reserved3",         29,  3, 1.0,   0.0,   0.0,   7.0),
        CANSignalSpec("RiskConfidence",    32,  8, 0.392, 0.0,   0.0, 100.0, "%"),
        CANSignalSpec("Reserved4",         40,  8, 1.0,   0.0,   0.0, 255.0),
        CANSignalSpec("CRC8",              48,  8, 1.0,   0.0,   0.0, 255.0),
        CANSignalSpec("Counter",           56,  4, 1.0,   0.0,   0.0,  15.0),
        CANSignalSpec("Reserved5",         60,  4, 1.0,   0.0,   0.0,  15.0),
    ]),
    0x160: CANMessageSpec(0x160, "Spatial_State", 8, cycle_ms=33, signals=[
        CANSignalSpec("HeadingError",   0, 16, 0.01,   -180.0, -180.0, 180.0, "deg"),
        CANSignalSpec("LateralError",  16, 16, 0.001,  -16.0,  -16.0,  16.0, "m"),
        CANSignalSpec("PathCurvature", 32, 12, 0.0001, -0.2,   -0.2,    0.2, "1/m"),
        CANSignalSpec("Reserved6",     44,  4, 1.0,    0.0,    0.0,   15.0),
        CANSignalSpec("CRC8",          48,  8, 1.0,    0.0,    0.0,  255.0),
        CANSignalSpec("Counter",       56,  4, 1.0,    0.0,    0.0,   15.0),
        CANSignalSpec("Reserved7",     60,  4, 1.0,    0.0,    0.0,   15.0),
    ]),
    0x170: CANMessageSpec(0x170, "Trajectory_Poly", 8, cycle_ms=33, signals=[
        CANSignalSpec("PolyCoeffA",  0, 10, 0.001, -0.5, -0.5, 0.5),
        CANSignalSpec("PolyCoeffB", 10, 10, 0.001, -0.5, -0.5, 0.5),
        CANSignalSpec("PolyCoeffC", 20, 10, 0.001, -0.5, -0.5, 0.5),
        CANSignalSpec("PolyCoeffD", 30, 10, 0.01,  -5.0, -5.0, 5.0, "m"),
        CANSignalSpec("Reserved8",  40,  8, 1.0,    0.0,  0.0, 255.0),
        CANSignalSpec("CRC8",       48,  8, 1.0,    0.0,  0.0, 255.0),
        CANSignalSpec("Counter",    56,  4, 1.0,    0.0,  0.0,  15.0),
        CANSignalSpec("Reserved9",  60,  4, 1.0,    0.0,  0.0,  15.0),
    ]),
    0x180: CANMessageSpec(0x180, "Vehicle_Kinematics", 8, cycle_ms=10, signals=[
        CANSignalSpec("WheelSpeed",         0, 16, 0.01,  0.0,   0.0,   250.0, "km/h"),
        CANSignalSpec("YawRate",           16, 12, 0.01, -20.48, -20.48, 20.47, "deg/s"),
        CANSignalSpec("LongitudinalAccel", 28, 10, 0.01, -5.12,  -5.12,  5.11, "m/s2"),
        CANSignalSpec("LateralAccel",      38, 10, 0.01, -5.12,  -5.12,  5.11, "m/s2"),
        CANSignalSpec("CRC8",              48,  8, 1.0,   0.0,    0.0,  255.0),
        CANSignalSpec("Counter",           56,  4, 1.0,   0.0,    0.0,   15.0),
        CANSignalSpec("KinFlags",          60,  4, 1.0,   0.0,    0.0,   15.0),
    ]),
    0x190: CANMessageSpec(0x190, "Plan_Control_Status", 8, cycle_ms=33, signals=[
        CANSignalSpec("ControllerMode",  0,  4, 1.0, 0.0,  0.0,       4.0),
        CANSignalSpec("HealthBitmask",   4, 20, 1.0, 0.0,  0.0, 1048575.0),
        CANSignalSpec("FailsafeLevel",  24,  4, 1.0, 0.0,  0.0,       4.0),
        CANSignalSpec("LoopLatencyUs",  28, 16, 1.0, 0.0,  0.0,   65535.0, "us"),
        CANSignalSpec("ReservedA",      44,  4, 1.0, 0.0,  0.0,      15.0),
        CANSignalSpec("CRC8",           48,  8, 1.0, 0.0,  0.0,     255.0),
        CANSignalSpec("Counter",        56,  4, 1.0, 0.0,  0.0,      15.0),
        CANSignalSpec("ReservedB",      60,  4, 1.0, 0.0,  0.0,      15.0),
    ]),
    0x320: CANMessageSpec(0x320, "MTi680_Orientation", 8, cycle_ms=10, signals=[
        CANSignalSpec("Roll",       0, 16, 0.01, -180.0, -180.0, 180.0, "deg"),
        CANSignalSpec("Pitch",     16, 16, 0.01,  -90.0,  -90.0,  90.0, "deg"),
        CANSignalSpec("Yaw",       32, 16, 0.01,    0.0,    0.0, 360.0, "deg"),
        CANSignalSpec("CRC8",      48,  8, 1.0,     0.0,    0.0, 255.0),
        CANSignalSpec("Counter",   56,  4, 1.0,     0.0,    0.0,  15.0),
        CANSignalSpec("InsStatus", 60,  4, 1.0,     0.0,    0.0,   5.0),
    ]),
    0x321: CANMessageSpec(0x321, "MTi680_Velocity", 8, cycle_ms=10, signals=[
        CANSignalSpec("VelocityEast",  0, 16, 0.01, -327.68, -327.68, 327.67, "m/s"),
        CANSignalSpec("VelocityNorth",16, 16, 0.01, -327.68, -327.68, 327.67, "m/s"),
        CANSignalSpec("VelocityDown", 32, 12, 0.01, -20.48,  -20.48,  20.47, "m/s"),
        CANSignalSpec("GnssFixType",  44,  4, 1.0,    0.0,     0.0,    6.0),
        CANSignalSpec("CRC8",         48,  8, 1.0,    0.0,     0.0,  255.0),
        CANSignalSpec("Counter",      56,  4, 1.0,    0.0,     0.0,   15.0),
        CANSignalSpec("VelFlags",     60,  4, 1.0,    0.0,     0.0,   15.0),
    ]),
    0x400: CANMessageSpec(0x400, "Starkenn_Radar_Header", 8, cycle_ms=50, signals=[
        CANSignalSpec("TargetCount",  0,  5, 1.0, 0.0,  0.0,    8.0),
        CANSignalSpec("BusLoad",      5,  7, 1.0, 0.0,  0.0,  100.0, "%"),
        CANSignalSpec("StatusFlags", 12, 12, 1.0, 0.0,  0.0, 4095.0),
        CANSignalSpec("RadarMode",   24,  4, 1.0, 0.0,  0.0,    4.0),
        CANSignalSpec("SensorTemp",  28,  8, 1.0,-40.0,-40.0,   90.0, "degC"),
        CANSignalSpec("FrameID",     36, 12, 1.0, 0.0,  0.0, 4095.0),
        CANSignalSpec("CRC8",        48,  8, 1.0, 0.0,  0.0,  255.0),
        CANSignalSpec("Counter",     56,  4, 1.0, 0.0,  0.0,   15.0),
        CANSignalSpec("ReservedC",   60,  4, 1.0, 0.0,  0.0,   15.0),
    ]),
}

# Build radar track specs for 0x401-0x408
_TRACK_SIGNALS = lambda: [
    CANSignalSpec("TrackID",   0,  4, 1.0,   0.0, 0.0,   15.0),
    CANSignalSpec("RangeX",    4, 12, 0.05, -50.0,-50.0, 154.75, "m"),
    CANSignalSpec("RangeY",   16, 12, 0.05, -25.0,-25.0, 179.75, "m"),
    CANSignalSpec("RelVelX",  28, 10, 0.05, -25.6,-25.6,  25.55, "m/s"),
    CANSignalSpec("RelVelY",  38,  8, 0.05, -6.4, -6.4,    6.35, "m/s"),
    CANSignalSpec("RCS_Flag", 46,  1, 1.0,   0.0,  0.0,    1.0),
    CANSignalSpec("ConfFlag", 47,  1, 1.0,   0.0,  0.0,    1.0),
    CANSignalSpec("CRC8",     48,  8, 1.0,   0.0,  0.0,  255.0),
    CANSignalSpec("Counter",  56,  4, 1.0,   0.0,  0.0,   15.0),
    CANSignalSpec("TrackSt",  60,  4, 1.0,   0.0,  0.0,   15.0),
]

for _track_idx in range(1, 9):
    _can_id = 0x400 + _track_idx
    APEX_MESSAGE_SPECS[_can_id] = CANMessageSpec(
        _can_id, f"Starkenn_Radar_Track{_track_idx}", 8,
        cycle_ms=50, signals=_TRACK_SIGNALS()
    )


# ---------------------------------------------------------------------------
# Payload Codec (pack/unpack struct-based binary serialization)
# ---------------------------------------------------------------------------

def _extract_bits_le(data: bytes, start_bit: int, length: int) -> int:
    """
    Extract little-endian bit field from CAN payload bytes.
    start_bit: bit position of LSB.
    length: number of bits to extract.
    """
    raw = int.from_bytes(data, byteorder="little")
    mask = (1 << length) - 1
    return (raw >> start_bit) & mask


def _inject_bits_le(data: bytearray, start_bit: int, length: int, value: int) -> None:
    """
    Inject little-endian bit field into CAN payload bytearray.
    """
    raw = int.from_bytes(data, byteorder="little")
    mask = ((1 << length) - 1) << start_bit
    raw = (raw & ~mask) | ((value << start_bit) & mask)
    packed = raw.to_bytes(8, byteorder="little")
    data[:] = packed


class CANPayloadCodec:
    """
    Struct-based binary serializer/deserializer for APEX CAN payloads.
    Uses little-endian bit extraction with explicit NaN/Inf guarding.
    """

    def __init__(self, crc_engine: Optional[CRC8Engine] = None) -> None:
        self._crc = crc_engine or CRC8Engine()

    def encode(
        self,
        spec: CANMessageSpec,
        values: Dict[str, float],
        counter: int,
    ) -> bytes:
        """
        Encode a dict of {signal_name: physical_value} into 8-byte CAN payload.
        NaN/Inf inputs are replaced with signal minimum.
        CRC-8 is computed and injected at byte 6.
        Counter nibble (0-15) is injected at bit 56.
        """
        payload = bytearray(spec.dlc)

        for sig in spec.signals:
            if sig.name in ("CRC8", "Counter"):
                continue
            phys = values.get(sig.name, sig.offset)  # default to offset (zero physical)
            raw = sig.encode(phys)
            raw = max(0, min(sig.max_raw, raw))  # clamp to bit width
            _inject_bits_le(payload, sig.start_bit, sig.length, raw)

        # Inject counter
        counter_val = counter % COUNTER_MODULUS
        _inject_bits_le(payload, 56, 4, counter_val)

        # Inject CRC over bytes 0-5
        self._crc.inject_crc(payload)

        return bytes(payload)

    def decode(
        self,
        spec: CANMessageSpec,
        payload: bytes,
        verify_crc: bool = True,
        verify_counter: bool = False,
        expected_counter: int = -1,
    ) -> Tuple[Dict[str, float], bool, bool]:
        """
        Decode 8-byte CAN payload to {signal_name: physical_value}.

        Returns:
            (values_dict, crc_ok, counter_ok)
            Values are clamped to declared [min, max] range.
        """
        if len(payload) != spec.dlc:
            raise ValueError(
                f"Payload length {len(payload)} != DLC {spec.dlc} "
                f"for {spec.name} (0x{spec.can_id:03X})"
            )

        crc_ok = True
        if verify_crc:
            crc_ok = self._crc.verify_frame(payload)

        values: Dict[str, float] = {}
        for sig in spec.signals:
            raw = _extract_bits_le(payload, sig.start_bit, sig.length)
            phys = sig.decode(raw)
            # NaN/Inf guard on decoded value (should never occur from integer decode)
            if math.isnan(phys) or math.isinf(phys):
                phys = sig.min_val
            values[sig.name] = phys

        counter_ok = True
        if verify_counter and expected_counter >= 0:
            actual_ctr = int(values.get("Counter", -1))
            counter_ok = actual_ctr == (expected_counter % COUNTER_MODULUS)

        return values, crc_ok, counter_ok

    def decode_with_spec_lookup(
        self,
        can_id: int,
        payload: bytes,
    ) -> Optional[Tuple[str, Dict[str, float], bool]]:
        """
        Convenience decode using global APEX_MESSAGE_SPECS lookup.
        Returns (message_name, values_dict, crc_ok) or None if ID unknown.
        """
        spec = APEX_MESSAGE_SPECS.get(can_id)
        if spec is None:
            return None
        values, crc_ok, _ = self.decode(spec, payload)
        return spec.name, values, crc_ok


# ---------------------------------------------------------------------------
# Typed Frame Containers
# ---------------------------------------------------------------------------

@dataclass
class HeartbeatFrame:
    node_id: int = 0
    system_state: int = 0
    error_flags: int = 0
    watchdog_counter: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class SteeringFrame:
    target_angle_deg: float = 0.0
    angular_velocity_limit: float = 0.0
    torque_command_nm: float = 0.0
    steering_mode: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class PriorityRiskFrame:
    risk_level: int = 0
    ttc_s: float = 10.0
    emergency_brake: bool = False
    risk_zone_id: int = 0
    risk_confidence: float = 0.0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class SpatialStateFrame:
    heading_error_deg: float = 0.0
    lateral_error_m: float = 0.0
    path_curvature_inv_m: float = 0.0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class TrajectoryPolyFrame:
    coeff_a: float = 0.0
    coeff_b: float = 0.0
    coeff_c: float = 0.0
    coeff_d_m: float = 0.0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class VehicleKinematicsFrame:
    wheel_speed_kmh: float = 0.0
    yaw_rate_dps: float = 0.0
    long_accel_mss: float = 0.0
    lat_accel_mss: float = 0.0
    kin_flags: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class PlanControlFrame:
    controller_mode: int = 0
    health_bitmask: int = 0
    failsafe_level: int = 0
    loop_latency_us: float = 0.0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class MTi680OrientFrame:
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    orient_status: int = 0
    ins_status: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class MTi680VelFrame:
    vel_east_mps: float = 0.0
    vel_north_mps: float = 0.0
    vel_down_mps: float = 0.0
    gnss_fix_type: int = 0
    vel_flags: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class RadarHeaderFrame:
    target_count: int = 0
    bus_load_pct: float = 0.0
    status_flags: int = 0
    radar_mode: int = 0
    sensor_temp_c: float = 0.0
    frame_id: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


@dataclass
class RadarTrackFrame:
    track_idx: int = 1          # 1-8
    track_id: int = 0
    range_x_m: float = 0.0
    range_y_m: float = 0.0
    rel_vel_x_mps: float = 0.0
    rel_vel_y_mps: float = 0.0
    rcs_above_threshold: bool = False
    confidence_high: bool = False
    track_status: int = 0
    crc8: int = 0
    counter: int = 0
    timestamp: float = 0.0
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID


# ---------------------------------------------------------------------------
# Line Status Tracker
# ---------------------------------------------------------------------------

@dataclass
class MessageLineStatus:
    """Per-message-ID line health status."""
    can_id: int
    name: str
    last_rx_time: float = 0.0
    rx_count: int = 0
    rx_timestamps: deque = field(default_factory=lambda: deque(maxlen=1000))
    is_stale: bool = True
    stale_flag_triggered: bool = False
    bus_load_estimate: float = 0.0
    loss_rate_1s: float = 0.0


class LineStatusTracker:
    """
    Tracks per-message health metrics:
      - Message age (time since last RX)
      - 1-second moving window frame count
      - Bus load estimate
      - Stale flag (>100ms without update)
    """

    def __init__(self, stale_threshold_s: float = STALE_THRESHOLD_S) -> None:
        self._statuses: Dict[int, MessageLineStatus] = {}
        self._stale_threshold = stale_threshold_s
        self._lock = threading.Lock()

    def register(self, spec: CANMessageSpec) -> None:
        """Register a message spec for tracking."""
        with self._lock:
            self._statuses[spec.can_id] = MessageLineStatus(
                can_id=spec.can_id, name=spec.name
            )

    def record_rx(self, can_id: int, timestamp: Optional[float] = None) -> None:
        """Record reception of a message."""
        ts = timestamp if timestamp is not None else time.monotonic()
        with self._lock:
            if can_id not in self._statuses:
                self._statuses[can_id] = MessageLineStatus(can_id=can_id, name=f"0x{can_id:03X}")
            status = self._statuses[can_id]
            status.last_rx_time = ts
            status.rx_count += 1
            status.rx_timestamps.append(ts)
            status.is_stale = False
            status.stale_flag_triggered = False

    def update_all(self, now: Optional[float] = None) -> List[int]:
        """
        Update stale flags for all tracked messages.
        Returns list of CAN IDs that just became stale.
        """
        now = now if now is not None else time.monotonic()
        newly_stale = []

        with self._lock:
            for can_id, status in self._statuses.items():
                age = now - status.last_rx_time if status.last_rx_time > 0 else float("inf")
                was_stale = status.is_stale

                if age > self._stale_threshold:
                    status.is_stale = True
                    if not was_stale:
                        newly_stale.append(can_id)
                        status.stale_flag_triggered = True
                        log.warning(
                            "CAN 0x%03X '%s': STALE — age=%.3f s > threshold %.3f s",
                            can_id, status.name, age, self._stale_threshold,
                        )
                else:
                    status.is_stale = False

                # Compute 1-second frame rate
                window_start = now - FRAME_LOSS_WINDOW
                frames_in_window = sum(1 for t in status.rx_timestamps if t >= window_start)
                status.loss_rate_1s = frames_in_window

                # Estimate bus load contribution (frames/s × bits_per_frame / bus_bps)
                spec = APEX_MESSAGE_SPECS.get(can_id)
                expected_hz = (1000.0 / spec.cycle_ms) if spec else 30.0
                status.bus_load_estimate = (expected_hz * BITS_PER_FRAME / BUS_SPEED_BPS) * 100.0

        return newly_stale

    def get_age(self, can_id: int, now: Optional[float] = None) -> float:
        """Return message age in seconds."""
        now = now if now is not None else time.monotonic()
        with self._lock:
            status = self._statuses.get(can_id)
            if status is None or status.last_rx_time == 0:
                return float("inf")
            return now - status.last_rx_time

    def is_stale(self, can_id: int) -> bool:
        with self._lock:
            return self._statuses.get(can_id, MessageLineStatus(0, "")).is_stale

    def get_all_status(self) -> Dict[int, Dict[str, Any]]:
        """Return snapshot of all line statuses."""
        now = time.monotonic()
        with self._lock:
            return {
                can_id: {
                    "name": s.name,
                    "age_s": round(now - s.last_rx_time, 4) if s.last_rx_time > 0 else None,
                    "is_stale": s.is_stale,
                    "rx_count": s.rx_count,
                    "bus_load_pct": round(s.bus_load_estimate, 2),
                    "frames_last_1s": s.loss_rate_1s,
                }
                for can_id, s in self._statuses.items()
            }


# ---------------------------------------------------------------------------
# Main Integrity Engine
# ---------------------------------------------------------------------------

class CANIntegrityEngine:
    """
    Top-level CAN hardware framing integrity engine for APEX-LKA v4.0.

    Integrates:
      - CRC-8 SAE J1850 verification (lookup-table O(1) per byte)
      - Per-ID sequence counter state machines
      - Payload decode with signal boundary clamping
      - Line status tracking with 100ms stale threshold

    Thread-safe for concurrent publish/subscribe patterns.
    """

    def __init__(self) -> None:
        self.crc = CRC8Engine()
        self.seq = SequenceVerifier()
        self.codec = CANPayloadCodec(self.crc)
        self.line = LineStatusTracker()
        self._lock = threading.Lock()
        self._callbacks: Dict[int, List[Callable]] = {}

        # Register all APEX messages for line tracking
        for spec in APEX_MESSAGE_SPECS.values():
            self.line.register(spec)

        log.info(
            "CANIntegrityEngine initialized: %d message specs, CRC-8 poly=0x%02X",
            len(APEX_MESSAGE_SPECS), CRC8_POLYNOMIAL,
        )

    def process_frame(
        self,
        can_id: int,
        payload: bytes,
        timestamp: Optional[float] = None,
    ) -> Tuple[Optional[Dict[str, float]], bool, SequenceStatus]:
        """
        Process an incoming CAN frame end-to-end:
          1. Verify CRC-8
          2. Decode payload to physical values
          3. Verify sequence counter
          4. Update line status
          5. Fire registered callbacks

        Returns:
            (decoded_values, crc_ok, seq_status)
            decoded_values is None if spec not found.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        spec = APEX_MESSAGE_SPECS.get(can_id)

        if spec is None:
            log.debug("CAN 0x%03X: Unknown message ID, skipping.", can_id)
            return None, False, SequenceStatus.VALID

        # CRC check
        crc_ok = self.crc.verify_frame(payload)
        if not crc_ok:
            log.warning("CAN 0x%03X '%s': CRC-8 FAIL", can_id, spec.name)

        # Decode
        values, _, _ = self.codec.decode(spec, payload, verify_crc=False)

        # Sequence check
        counter_raw = int(values.get("Counter", 0))
        seq_status = self.seq.verify(can_id, counter_raw, timestamp=ts)

        if seq_status == SequenceStatus.FRAME_DROP:
            log.warning("CAN 0x%03X '%s': FRAME DROP (counter=%d)", can_id, spec.name, counter_raw)
        elif seq_status == SequenceStatus.OUT_OF_ORDER:
            log.debug("CAN 0x%03X '%s': OUT_OF_ORDER (counter=%d)", can_id, spec.name, counter_raw)
        elif seq_status == SequenceStatus.RESET:
            log.info("CAN 0x%03X '%s': RESET detected.", can_id, spec.name)

        # Line tracking
        self.line.record_rx(can_id, timestamp=ts)

        # Callbacks
        with self._lock:
            cbs = self._callbacks.get(can_id, [])
        for cb in cbs:
            try:
                cb(can_id, values, crc_ok, seq_status, ts)
            except Exception as exc:
                log.error("Callback error for CAN 0x%03X: %s", can_id, exc)

        return values, crc_ok, seq_status

    def encode_frame(
        self,
        can_id: int,
        values: Dict[str, float],
        counter: int,
    ) -> Optional[bytes]:
        """
        Encode physical values to 8-byte CAN payload with CRC injection.
        Returns None if CAN ID not in spec.
        """
        spec = APEX_MESSAGE_SPECS.get(can_id)
        if spec is None:
            log.error("CAN 0x%03X: Cannot encode — unknown message ID.", can_id)
            return None
        return self.codec.encode(spec, values, counter)

    def subscribe(self, can_id: int, callback: Callable) -> None:
        """Register a callback for a specific CAN ID."""
        with self._lock:
            if can_id not in self._callbacks:
                self._callbacks[can_id] = []
            self._callbacks[can_id].append(callback)
        log.debug("CAN 0x%03X: Callback registered.", can_id)

    def unsubscribe(self, can_id: int, callback: Callable) -> None:
        """Remove a callback for a specific CAN ID."""
        with self._lock:
            if can_id in self._callbacks:
                try:
                    self._callbacks[can_id].remove(callback)
                except ValueError:
                    pass

    def get_health_report(self) -> Dict[str, Any]:
        """Return comprehensive health report of all CAN lines."""
        self.line.update_all()
        return {
            "line_status": self.line.get_all_status(),
            "sequence_stats": self.seq.get_all_stats(),
            "crc_table_size": len(self.crc.table),
        }
