#!/usr/bin/env python3
"""
APEX-LKA v4.0 — DBC Validation & Programmatic Generator
=========================================================
Programmatically validates, regenerates, and verifies the apex_v4.dbc CAN database.

Features:
  - Signal overlap detection (bit-level collision checking within each message)
  - Bit-ordering consistency enforcement (Little Endian canonical form)
  - Range limit verification (scale + offset math against physical limits)
  - CRC-8 coverage validation across all messages
  - Rolling counter field presence per message
  - Expected message cycle-time cross-reference
  - Full DBC regeneration from internal Python schema

Usage:
  python scripts/gen_dbc.py --validate --input configs/can/apex_v4.dbc
  python scripts/gen_dbc.py --generate --output configs/can/apex_v4_gen.dbc
  python scripts/gen_dbc.py --validate --generate --input configs/can/apex_v4.dbc
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
APEX_VERSION = "APEX_LKA_v4.0"
BUS_SPEED_BPS = 500_000  # 500 kbps
MAX_DLC = 8               # Standard CAN frame max DLC
CRC8_POLYNOMIAL = 0x1D   # SAE J1850 CRC-8

# Expected message cycle times in milliseconds (derived from Hz specs)
EXPECTED_CYCLE_MS: Dict[int, int] = {
    0x050: 10,   # Heartbeat 100 Hz
    0x100: 33,   # Steering_Primary 30 Hz
    0x150: 33,   # Priority_Risk 30 Hz
    0x160: 33,   # Spatial_State 30 Hz
    0x170: 33,   # Trajectory_Poly 30 Hz
    0x180: 10,   # Vehicle_Kinematics 100 Hz
    0x190: 33,   # Plan_Control_Status 30 Hz
    0x320: 10,   # MTi680_Orientation 100 Hz
    0x321: 10,   # MTi680_Velocity 100 Hz
    0x400: 50,   # Starkenn_Radar_Header 20 Hz
    0x401: 50,   # Starkenn_Radar_Track1
    0x402: 50,
    0x403: 50,
    0x404: 50,
    0x405: 50,
    0x406: 50,
    0x407: 50,
    0x408: 50,   # Starkenn_Radar_Track8
}

# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

@dataclass
class SignalDef:
    """Full signal definition matching DBC SG_ format."""
    name: str
    start_bit: int          # Little Endian start bit position (LSB)
    length: int             # Bit length
    byte_order: str         # 'little_endian' or 'big_endian'
    is_signed: bool
    factor: float
    offset: float
    min_val: float
    max_val: float
    unit: str
    receivers: List[str]
    comment: str = ""

    @property
    def end_bit(self) -> int:
        """Inclusive end bit (for overlap detection in little-endian signals)."""
        return self.start_bit + self.length - 1

    def physical_min(self) -> float:
        """Decode raw=0 → physical minimum."""
        return 0.0 * self.factor + self.offset

    def physical_max_raw(self) -> float:
        """Decode raw=max_raw → physical maximum."""
        max_raw = (1 << self.length) - 1
        return max_raw * self.factor + self.offset

    def raw_encode(self, physical: float) -> int:
        """Encode physical value to raw integer."""
        raw = (physical - self.offset) / self.factor
        return int(round(raw))

    def validate_range(self) -> List[str]:
        """Return list of range-violation error strings."""
        errors = []
        phys_min_calc = self.physical_min()
        phys_max_calc = self.physical_max_raw()
        # Allow 1% tolerance for floating point rounding
        tol = max(abs(self.min_val) * 0.01, 1e-6)
        if phys_min_calc > self.min_val + tol:
            errors.append(
                f"  Signal '{self.name}': computed physical min "
                f"{phys_min_calc:.6f} > declared min {self.min_val:.6f}"
            )
        if phys_max_calc < self.max_val - tol:
            errors.append(
                f"  Signal '{self.name}': computed physical max "
                f"{phys_max_calc:.6f} < declared max {self.max_val:.6f}"
            )
        return errors


@dataclass
class MessageDef:
    """Full CAN message definition."""
    can_id: int
    name: str
    dlc: int
    transmitter: str
    signals: List[SignalDef] = field(default_factory=list)
    comment: str = ""
    cycle_ms: int = 0

    def get_signal(self, name: str) -> Optional[SignalDef]:
        for s in self.signals:
            if s.name == name:
                return s
        return None

    def has_crc8(self) -> bool:
        return any(s.name == "CRC8" for s in self.signals)

    def has_counter(self) -> bool:
        return any(s.name == "Counter" for s in self.signals)


# ---------------------------------------------------------------------------
# CRC-8 SAE J1850 (Polynomial 0x1D) — lookup table
# ---------------------------------------------------------------------------

def _build_crc8_table(poly: int = 0x1D) -> List[int]:
    """Build 256-entry CRC-8 lookup table for SAE J1850 (poly=0x1D)."""
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ poly) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
        table.append(crc)
    return table


CRC8_TABLE = _build_crc8_table(CRC8_POLYNOMIAL)


def compute_crc8(data: bytes) -> int:
    """Compute CRC-8 SAE J1850 using precomputed lookup table (O(1) per byte)."""
    crc = 0xFF
    for byte in data:
        crc = CRC8_TABLE[crc ^ byte]
    return crc ^ 0xFF


# ---------------------------------------------------------------------------
# APEX v4 Schema Definition
# ---------------------------------------------------------------------------

NODES = ["JETSON_ORIN", "ARDUINO_PRIMARY", "ARDUINO_SECONDARY", "RADAR_STARKENN", "INS_MTI680"]

ALL_RECEIVERS = ["ARDUINO_PRIMARY", "ARDUINO_SECONDARY", "RADAR_STARKENN", "INS_MTI680"]
JETSON_RX = ["JETSON_ORIN"]
ARDUINO_RX = ["ARDUINO_PRIMARY", "ARDUINO_SECONDARY"]


def build_apex_schema() -> List[MessageDef]:
    """
    Construct the complete APEX-LKA v4.0 CAN message schema programmatically.
    Returns a list of MessageDef objects mirroring apex_v4.dbc exactly.
    """
    messages: List[MessageDef] = []

    # ------------------------------------------------------------------
    # 0x050 — Heartbeat (100 Hz)
    # ------------------------------------------------------------------
    hb = MessageDef(
        can_id=0x050, name="Heartbeat", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=10,
        comment="Heartbeat 100 Hz. All nodes must receive within 15ms.",
    )
    hb.signals = [
        SignalDef("NodeID",          0,  8, "little_endian", False, 1.0,   0.0, 0.0, 255.0, "", ALL_RECEIVERS,
                  "Node identifier: 1=Jetson, 2=Ard_Pri, 3=Ard_Sec, 4=Radar, 5=INS."),
        SignalDef("SystemState",     8,  8, "little_endian", False, 1.0,   0.0, 0.0, 5.0,   "", ALL_RECEIVERS,
                  "0=INIT, 1=STANDBY, 2=ACTIVE, 3=DEGRADED, 4=FAILSAFE, 5=SHUTDOWN."),
        SignalDef("ErrorFlags",     16, 16, "little_endian", False, 1.0,   0.0, 0.0, 65535.0, "", ALL_RECEIVERS,
                  "16-bit error bitmask."),
        SignalDef("WatchdogCounter",32,  8, "little_endian", False, 1.0,   0.0, 0.0, 255.0, "", ARDUINO_RX,
                  "Rolling watchdog counter. Stall > 200ms triggers relay reset."),
        SignalDef("Reserved0",      40,  8, "little_endian", False, 1.0,   0.0, 0.0, 255.0, "", [],
                  "Reserved for future use."),
        SignalDef("CRC8",           48,  8, "little_endian", False, 1.0,   0.0, 0.0, 255.0, "", ARDUINO_RX,
                  "SAE J1850 CRC-8 over bytes 0-5."),
        SignalDef("Counter",        56,  4, "little_endian", False, 1.0,   0.0, 0.0, 15.0,  "", ARDUINO_RX,
                  "4-bit rolling frame counter 0-15."),
    ]
    messages.append(hb)

    # ------------------------------------------------------------------
    # 0x100 — Steering_Primary (30 Hz)
    # ------------------------------------------------------------------
    st = MessageDef(
        can_id=0x100, name="Steering_Primary", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=33,
        comment="Steering primary command 30 Hz. TargetSteeringAngle +-28.0 deg.",
    )
    st.signals = [
        SignalDef("TargetSteeringAngle", 0, 16, "little_endian", False, 0.001, -28.0, -28.0, 28.0, "deg",
                  ARDUINO_RX, "Target steering angle -28.0 to +28.0 deg. Scale 0.001 deg/LSB."),
        SignalDef("AngularVelocityLimit",16, 12, "little_endian", False, 0.1, 0.0, 0.0, 200.0, "deg/s",
                  ["ARDUINO_PRIMARY"], "Max angular velocity 0-200 deg/s. Scale 0.1 deg/s/LSB."),
        SignalDef("TorqueCommand",       28, 12, "little_endian", False, 0.01, -10.0, -10.0, 10.0, "Nm",
                  ["ARDUINO_PRIMARY"], "EPS torque -10.0 to +10.0 Nm. Scale 0.01 Nm/LSB."),
        SignalDef("SteeringMode",        40,  4, "little_endian", False, 1.0,   0.0, 0.0, 4.0, "",
                  ["ARDUINO_PRIMARY"], "0=DISABLE, 1=ANGLE_CTRL, 2=TORQUE_CTRL, 3=EMERGENCY, 4=MANUAL."),
        SignalDef("Reserved1",           44,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
        SignalDef("CRC8",                48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",             56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("Reserved2",           60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(st)

    # ------------------------------------------------------------------
    # 0x150 — Priority_Risk (30 Hz)
    # ------------------------------------------------------------------
    risk = MessageDef(
        can_id=0x150, name="Priority_Risk", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=33,
        comment="Priority risk assessment 30 Hz. TTC 0-10 s.",
    )
    risk.signals = [
        SignalDef("DynamicRiskLevel",  0,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ARDUINO_RX, "Risk 0-255. 0=CLEAR, 255=CRITICAL_STOP."),
        SignalDef("TTC",               8, 16, "little_endian", False, 0.001, 0.0, 0.0, 10.0, "s",
                  ["ARDUINO_PRIMARY"], "Time-to-collision 0-10.0 s. Scale 0.001 s/LSB."),
        SignalDef("EmergencyBrakeFlag",24,  1, "little_endian", False, 1.0, 0.0, 0.0, 1.0, "",
                  ARDUINO_RX, "1=engage full braking and relay interlock."),
        SignalDef("RiskZoneID",        25,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "Risk zone identifier 0-15."),
        SignalDef("Reserved3",         29,  3, "little_endian", False, 1.0, 0.0, 0.0, 7.0, "", [], "Reserved."),
        SignalDef("RiskConfidence",    32,  8, "little_endian", False, 0.392, 0.0, 0.0, 100.0, "%",
                  ["ARDUINO_PRIMARY"], "Risk confidence 0-100%. Scale 0.392%/LSB."),
        SignalDef("Reserved4",         40,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "", [], "Reserved."),
        SignalDef("CRC8",              48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",           56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("Reserved5",         60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(risk)

    # ------------------------------------------------------------------
    # 0x160 — Spatial_State (30 Hz)
    # ------------------------------------------------------------------
    sp = MessageDef(
        can_id=0x160, name="Spatial_State", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=33,
        comment="Spatial state 30 Hz. HeadingError, LateralError, PathCurvature.",
    )
    sp.signals = [
        SignalDef("HeadingError",   0, 16, "little_endian", False, 0.01,  -180.0, -180.0, 180.0, "deg",
                  ["ARDUINO_PRIMARY"], "Heading error -180 to +180 deg. Scale 0.01 deg/LSB."),
        SignalDef("LateralError",  16, 16, "little_endian", False, 0.001, -16.0, -16.0, 16.0, "m",
                  ["ARDUINO_PRIMARY"], "Lateral deviation -16 to +16 m. Scale 0.001 m/LSB."),
        SignalDef("PathCurvature", 32, 12, "little_endian", False, 0.0001, -0.2, -0.2, 0.2, "1/m",
                  ["ARDUINO_PRIMARY"], "Path curvature -0.2 to +0.2 1/m. Scale 0.0001 1/m/LSB."),
        SignalDef("Reserved6",     44,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
        SignalDef("CRC8",          48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",       56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("Reserved7",     60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(sp)

    # ------------------------------------------------------------------
    # 0x170 — Trajectory_Poly (30 Hz)
    # ------------------------------------------------------------------
    poly = MessageDef(
        can_id=0x170, name="Trajectory_Poly", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=33,
        comment="Cubic trajectory polynomial coefficients A,B,C,D at 30 Hz.",
    )
    poly.signals = [
        SignalDef("PolyCoeffA",  0, 10, "little_endian", False, 0.001, -0.5, -0.5, 0.5, "",
                  ["ARDUINO_PRIMARY"], "Cubic poly coeff A (cubic term). Range -0.5 to +0.5."),
        SignalDef("PolyCoeffB", 10, 10, "little_endian", False, 0.001, -0.5, -0.5, 0.5, "",
                  ["ARDUINO_PRIMARY"], "Cubic poly coeff B (quadratic term). Range -0.5 to +0.5."),
        SignalDef("PolyCoeffC", 20, 10, "little_endian", False, 0.001, -0.5, -0.5, 0.5, "",
                  ["ARDUINO_PRIMARY"], "Cubic poly coeff C (linear term). Range -0.5 to +0.5."),
        SignalDef("PolyCoeffD", 30, 10, "little_endian", False, 0.01, -5.0, -5.0, 5.0, "m",
                  ["ARDUINO_PRIMARY"], "Cubic poly coeff D (offset). Range -5.0 to +5.0 m."),
        SignalDef("Reserved8",  40,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "", [], "Reserved."),
        SignalDef("CRC8",       48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",    56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("Reserved9",  60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(poly)

    # ------------------------------------------------------------------
    # 0x180 — Vehicle_Kinematics (100 Hz)
    # ------------------------------------------------------------------
    kin = MessageDef(
        can_id=0x180, name="Vehicle_Kinematics", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=10,
        comment="Vehicle kinematics 100 Hz. WheelSpeed, YawRate, Accel.",
    )
    kin.signals = [
        SignalDef("WheelSpeed",        0, 16, "little_endian", False, 0.01, 0.0, 0.0, 250.0, "km/h",
                  ARDUINO_RX, "Avg wheel speed 0-250 km/h. Scale 0.01 km/h/LSB."),
        SignalDef("YawRate",          16, 12, "little_endian", False, 0.01, -20.48, -20.48, 20.47, "deg/s",
                  ["ARDUINO_PRIMARY"], "Yaw rate -20.48 to +20.47 deg/s. Scale 0.01 deg/s/LSB."),
        SignalDef("LongitudinalAccel",28, 10, "little_endian", False, 0.01, -5.12, -5.12, 5.11, "m/s2",
                  ["ARDUINO_PRIMARY"], "Longitudinal accel -5.12 to +5.11 m/s2."),
        SignalDef("LateralAccel",     38, 10, "little_endian", False, 0.01, -5.12, -5.12, 5.11, "m/s2",
                  ["ARDUINO_PRIMARY"], "Lateral accel -5.12 to +5.11 m/s2."),
        SignalDef("CRC8",             48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",          56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("KinFlags",         60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "Kinematics validity flags bitmask."),
    ]
    messages.append(kin)

    # ------------------------------------------------------------------
    # 0x190 — Plan_Control_Status (30 Hz)
    # ------------------------------------------------------------------
    pcs = MessageDef(
        can_id=0x190, name="Plan_Control_Status", dlc=8, transmitter="JETSON_ORIN",
        cycle_ms=33,
        comment="Planning & control status 30 Hz. ControllerMode, HealthBitmask.",
    )
    pcs.signals = [
        SignalDef("ControllerMode",  0,  4, "little_endian", False, 1.0, 0.0, 0.0, 4.0, "",
                  ARDUINO_RX, "0=OFF, 1=STANDBY, 2=ACTIVE, 3=OVERRIDE, 4=FAILSAFE."),
        SignalDef("HealthBitmask",   4, 20, "little_endian", False, 1.0, 0.0, 0.0, 1048575.0, "",
                  ARDUINO_RX, "20-bit subsystem health. 0=OK, 1=FAULT per bit."),
        SignalDef("FailsafeLevel",  24,  4, "little_endian", False, 1.0, 0.0, 0.0, 4.0, "",
                  ARDUINO_RX, "0=NOMINAL, 1=WARN, 2=DEGRADE, 3=FAILSAFE, 4=SHUTDOWN."),
        SignalDef("LoopLatencyUs", 28, 16, "little_endian", False, 1.0, 0.0, 0.0, 65535.0, "us",
                  ["ARDUINO_PRIMARY"], "Main loop latency in microseconds."),
        SignalDef("ReservedA",     44,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
        SignalDef("CRC8",          48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  ["ARDUINO_PRIMARY"], "SAE J1850 CRC-8."),
        SignalDef("Counter",       56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  ["ARDUINO_PRIMARY"], "4-bit rolling counter."),
        SignalDef("ReservedB",     60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(pcs)

    # ------------------------------------------------------------------
    # 0x320 — MTi680_Orientation (100 Hz)
    # ------------------------------------------------------------------
    ori = MessageDef(
        can_id=0x320, name="MTi680_Orientation", dlc=8, transmitter="INS_MTI680",
        cycle_ms=10,
        comment="MTi-680 Euler orientation 100 Hz. Roll, Pitch, Yaw at 0.01 deg resolution.",
    )
    ori.signals = [
        SignalDef("Roll",       0, 16, "little_endian", False, 0.01, -180.0, -180.0, 180.0, "deg",
                  JETSON_RX, "Roll angle -180 to +180 deg. Scale 0.01 deg/LSB."),
        SignalDef("Pitch",     16, 16, "little_endian", False, 0.01, -90.0, -90.0, 90.0, "deg",
                  JETSON_RX, "Pitch angle -90 to +90 deg. Scale 0.01 deg/LSB."),
        SignalDef("Yaw",       32, 16, "little_endian", False, 0.01, 0.0, 0.0, 360.0, "deg",
                  JETSON_RX, "True-North heading 0-360 deg. Scale 0.01 deg/LSB."),
        SignalDef("CRC8",      48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  JETSON_RX, "SAE J1850 CRC-8."),
        SignalDef("Counter",   56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  JETSON_RX, "4-bit rolling counter."),
        SignalDef("InsStatus", 60,  4, "little_endian", False, 1.0, 0.0, 0.0, 5.0, "",
                  JETSON_RX, "0=INIT, 1=COARSE, 2=FINE, 3=RTK_FLOAT, 4=RTK_FIXED, 5=DEAD_RECK."),
    ]
    messages.append(ori)

    # ------------------------------------------------------------------
    # 0x321 — MTi680_Velocity (100 Hz)
    # ------------------------------------------------------------------
    vel = MessageDef(
        can_id=0x321, name="MTi680_Velocity", dlc=8, transmitter="INS_MTI680",
        cycle_ms=10,
        comment="MTi-680 ECEF velocity East/North/Down at 100 Hz.",
    )
    vel.signals = [
        SignalDef("VelocityEast",  0, 16, "little_endian", False, 0.01, -327.68, -327.68, 327.67, "m/s",
                  JETSON_RX, "East velocity -327.68 to +327.67 m/s. Scale 0.01 m/s/LSB."),
        SignalDef("VelocityNorth",16, 16, "little_endian", False, 0.01, -327.68, -327.68, 327.67, "m/s",
                  JETSON_RX, "North velocity -327.68 to +327.67 m/s. Scale 0.01 m/s/LSB."),
        SignalDef("VelocityDown", 32, 12, "little_endian", False, 0.01, -20.48, -20.48, 20.47, "m/s",
                  JETSON_RX, "Down velocity -20.48 to +20.47 m/s. Scale 0.01 m/s/LSB."),
        SignalDef("GnssFixType",  44,  4, "little_endian", False, 1.0, 0.0, 0.0, 6.0, "",
                  JETSON_RX, "0=NO_FIX .. 6=PPP."),
        SignalDef("CRC8",         48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  JETSON_RX, "SAE J1850 CRC-8."),
        SignalDef("Counter",      56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  JETSON_RX, "4-bit rolling counter."),
        SignalDef("VelFlags",     60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  JETSON_RX, "Velocity validity flags."),
    ]
    messages.append(vel)

    # ------------------------------------------------------------------
    # 0x400 — Starkenn_Radar_Header (20 Hz)
    # ------------------------------------------------------------------
    rh = MessageDef(
        can_id=0x400, name="Starkenn_Radar_Header", dlc=8, transmitter="RADAR_STARKENN",
        cycle_ms=50,
        comment="Starkenn 77GHz radar header 20 Hz. TargetCount, BusLoad, StatusFlags.",
    )
    rh.signals = [
        SignalDef("TargetCount", 0,  5, "little_endian", False, 1.0, 0.0, 0.0, 8.0, "",
                  JETSON_RX, "Number of valid tracks 0-8."),
        SignalDef("BusLoad",     5,  7, "little_endian", False, 1.0, 0.0, 0.0, 100.0, "%",
                  JETSON_RX, "CAN bus load 0-100%."),
        SignalDef("StatusFlags",12, 12, "little_endian", False, 1.0, 0.0, 0.0, 4095.0, "",
                  JETSON_RX, "Radar status bitmask."),
        SignalDef("RadarMode",  24,  4, "little_endian", False, 1.0, 0.0, 0.0, 4.0, "",
                  JETSON_RX, "0=STANDBY, 1=NEAR, 2=FAR, 3=COMBINED, 4=CALIBRATION."),
        SignalDef("SensorTemp", 28,  8, "little_endian", False, 1.0, -40.0, -40.0, 90.0, "degC",
                  JETSON_RX, "Radar board temperature."),
        SignalDef("FrameID",    36, 12, "little_endian", False, 1.0, 0.0, 0.0, 4095.0, "",
                  JETSON_RX, "Radar frame sequence ID."),
        SignalDef("CRC8",       48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                  JETSON_RX, "SAE J1850 CRC-8."),
        SignalDef("Counter",    56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                  JETSON_RX, "4-bit rolling counter."),
        SignalDef("ReservedC",  60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "", [], "Reserved."),
    ]
    messages.append(rh)

    # ------------------------------------------------------------------
    # 0x401 - 0x408 — Starkenn_Radar_Track 1-8 (20 Hz each)
    # ------------------------------------------------------------------
    for track_idx in range(1, 9):
        can_id = 0x400 + track_idx
        trk = MessageDef(
            can_id=can_id,
            name=f"Starkenn_Radar_Track{track_idx}",
            dlc=8,
            transmitter="RADAR_STARKENN",
            cycle_ms=50,
            comment=f"Starkenn radar track {track_idx} of 8. RangeX/Y, RelVelX/Y, RCS, Confidence.",
        )
        trk.signals = [
            SignalDef("TrackID",   0,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                      JETSON_RX, "Track object ID 0-15."),
            SignalDef("RangeX",    4, 12, "little_endian", False, 0.05, -50.0, -50.0, 154.75, "m",
                      JETSON_RX, "Longitudinal range -50.0 to +154.75 m. Scale 0.05 m/LSB."),
            SignalDef("RangeY",   16, 12, "little_endian", False, 0.05, -25.0, -25.0, 179.75, "m",
                      JETSON_RX, "Lateral range -25.0 to +179.75 m. Scale 0.05 m/LSB."),
            SignalDef("RelVelX",  28, 10, "little_endian", False, 0.05, -25.6, -25.6, 25.55, "m/s",
                      JETSON_RX, "Relative longitudinal velocity -25.6 to +25.55 m/s."),
            SignalDef("RelVelY",  38,  8, "little_endian", False, 0.05, -6.4, -6.4, 6.35, "m/s",
                      JETSON_RX, "Relative lateral velocity -6.4 to +6.35 m/s."),
            SignalDef("RCS_Flag", 46,  1, "little_endian", False, 1.0, 0.0, 0.0, 1.0, "",
                      JETSON_RX, "RCS above -15 dBm flag. 0=below, 1=above."),
            SignalDef("ConfFlag", 47,  1, "little_endian", False, 1.0, 0.0, 0.0, 1.0, "",
                      JETSON_RX, "Confidence above 60% flag. 0=low, 1=high."),
            SignalDef("CRC8",     48,  8, "little_endian", False, 1.0, 0.0, 0.0, 255.0, "",
                      JETSON_RX, "SAE J1850 CRC-8."),
            SignalDef("Counter",  56,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                      JETSON_RX, "4-bit rolling counter."),
            SignalDef("TrackSt",  60,  4, "little_endian", False, 1.0, 0.0, 0.0, 15.0, "",
                      JETSON_RX, "Track status flags."),
        ]
        messages.append(trk)

    return messages


# ---------------------------------------------------------------------------
# Validation Engine
# ---------------------------------------------------------------------------

class DBCValidator:
    """Validates the APEX-LKA v4 CAN schema for signal integrity."""

    def __init__(self, messages: List[MessageDef]):
        self.messages = messages
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.passed: List[str] = []

    def _check_signal_overlaps(self, msg: MessageDef) -> None:
        """Detect any bit-level collisions between signals in a message."""
        occupied: Dict[int, str] = {}
        for sig in msg.signals:
            for bit in range(sig.start_bit, sig.start_bit + sig.length):
                if bit in occupied:
                    self.errors.append(
                        f"MSG 0x{msg.can_id:03X} '{msg.name}': "
                        f"Signal '{sig.name}' OVERLAPS '{occupied[bit]}' at bit {bit}."
                    )
                else:
                    occupied[bit] = sig.name
        total_bits = sum(s.length for s in msg.signals)
        if total_bits <= msg.dlc * 8:
            self.passed.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': No signal overlaps detected "
                f"({total_bits}/{msg.dlc * 8} bits used)."
            )

    def _check_dlc_fit(self, msg: MessageDef) -> None:
        """Ensure all signals fit within the declared DLC."""
        for sig in msg.signals:
            end_byte = (sig.start_bit + sig.length - 1) // 8
            if end_byte >= msg.dlc:
                self.errors.append(
                    f"MSG 0x{msg.can_id:03X} '{msg.name}': "
                    f"Signal '{sig.name}' overflows DLC={msg.dlc} "
                    f"(end_byte={end_byte})."
                )

    def _check_range_math(self, msg: MessageDef) -> None:
        """Verify physical range can be represented given scale and bit width."""
        for sig in msg.signals:
            range_errors = sig.validate_range()
            if range_errors:
                self.warnings.extend(range_errors)
            else:
                self.passed.append(
                    f"MSG 0x{msg.can_id:03X} sig '{sig.name}': "
                    f"Range [{sig.min_val}, {sig.max_val}] {sig.unit} is representable."
                )

    def _check_crc_present(self, msg: MessageDef) -> None:
        """Verify each message has a CRC8 signal at byte 6 (bit 48)."""
        crc_sig = msg.get_signal("CRC8")
        if crc_sig is None:
            self.errors.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Missing CRC8 signal."
            )
        elif crc_sig.start_bit != 48 or crc_sig.length != 8:
            self.errors.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': CRC8 must be at bit 48, len=8 "
                f"(found bit {crc_sig.start_bit}, len {crc_sig.length})."
            )
        else:
            self.passed.append(f"MSG 0x{msg.can_id:03X} '{msg.name}': CRC8 signal OK at bit 48.")

    def _check_counter_present(self, msg: MessageDef) -> None:
        """Verify rolling Counter signal exists and is 4 bits wide."""
        ctr_sig = msg.get_signal("Counter")
        if ctr_sig is None:
            self.errors.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Missing Counter signal."
            )
        elif ctr_sig.length != 4:
            self.errors.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Counter must be 4 bits wide "
                f"(found {ctr_sig.length})."
            )
        else:
            self.passed.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Counter signal OK "
                f"({ctr_sig.length}-bit at bit {ctr_sig.start_bit})."
            )

    def _check_cycle_time(self, msg: MessageDef) -> None:
        """Cross-check cycle time against expected spec."""
        expected = EXPECTED_CYCLE_MS.get(msg.can_id)
        if expected is None:
            self.warnings.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': No expected cycle time in spec table."
            )
            return
        if msg.cycle_ms != expected:
            self.errors.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Cycle time {msg.cycle_ms} ms "
                f"does not match expected {expected} ms."
            )
        else:
            hz = 1000.0 / expected
            self.passed.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': Cycle time {msg.cycle_ms} ms "
                f"= {hz:.1f} Hz — OK."
            )

    def _check_id_uniqueness(self) -> None:
        """Ensure no two messages share the same CAN ID."""
        seen: Dict[int, str] = {}
        for msg in self.messages:
            if msg.can_id in seen:
                self.errors.append(
                    f"Duplicate CAN ID 0x{msg.can_id:03X}: "
                    f"'{msg.name}' conflicts with '{seen[msg.can_id]}'."
                )
            else:
                seen[msg.can_id] = msg.name
        self.passed.append(f"CAN ID uniqueness: {len(seen)} unique IDs, no collisions.")

    def _check_byte_ordering_consistency(self, msg: MessageDef) -> None:
        """Verify all signals use consistent little-endian ordering (canonical APEX form)."""
        mixed = [s for s in msg.signals if s.byte_order != "little_endian"]
        if mixed:
            for s in mixed:
                self.errors.append(
                    f"MSG 0x{msg.can_id:03X} '{msg.name}': "
                    f"Signal '{s.name}' uses '{s.byte_order}' — APEX requires little_endian."
                )
        else:
            self.passed.append(
                f"MSG 0x{msg.can_id:03X} '{msg.name}': All signals use little_endian. OK."
            )

    def run_all(self) -> bool:
        """Run all validation checks. Returns True if no errors found."""
        print("=" * 72)
        print(f"APEX-LKA v4.0 DBC Validation Engine  — {len(self.messages)} messages")
        print("=" * 72)

        self._check_id_uniqueness()

        for msg in self.messages:
            self._check_signal_overlaps(msg)
            self._check_dlc_fit(msg)
            self._check_range_math(msg)
            self._check_crc_present(msg)
            self._check_counter_present(msg)
            self._check_cycle_time(msg)
            self._check_byte_ordering_consistency(msg)

        # Print results
        print(f"\n[PASSED] {len(self.passed)} checks:")
        for p in self.passed:
            print(f"  [OK] {p}")

        if self.warnings:
            print(f"\n[WARNINGS] {len(self.warnings)} warnings:")
            for w in self.warnings:
                print(f"  [WARN] {w}")

        if self.errors:
            print(f"\n[ERRORS] {len(self.errors)} errors:")
            for e in self.errors:
                print(f"  [FAIL] {e}")
            print("\n[RESULT] VALIDATION FAILED")
            return False

        print(f"\n[RESULT] VALIDATION PASSED -- {len(self.passed)} checks, "
              f"{len(self.warnings)} warnings, 0 errors.")
        return True


# ---------------------------------------------------------------------------
# DBC Generator
# ---------------------------------------------------------------------------

class DBCGenerator:
    """Generates a DBC file from a list of MessageDef objects."""

    def __init__(self, messages: List[MessageDef], version: str = APEX_VERSION):
        self.messages = messages
        self.version = version

    def _signal_byte_order_char(self, sig: SignalDef) -> str:
        return "1" if sig.byte_order == "little_endian" else "0"

    def _signal_value_type(self, sig: SignalDef) -> str:
        return "-" if sig.is_signed else "+"

    def _format_signal(self, sig: SignalDef) -> str:
        byte_order = self._signal_byte_order_char(sig)
        value_type = self._signal_value_type(sig)
        receivers = ",".join(sig.receivers) if sig.receivers else "Vector__XXX"
        return (
            f" SG_ {sig.name} : {sig.start_bit}|{sig.length}@{byte_order}{value_type} "
            f"({sig.factor},{sig.offset}) [{sig.min_val}|{sig.max_val}] "
            f'"{sig.unit}" {receivers}'
        )

    def generate(self) -> str:
        """Generate complete DBC content as a string."""
        lines = []

        # Header
        lines.append(f'VERSION "{self.version}"')
        lines.append("")
        lines.append("NS_ :")
        for ns in ["NS_DESC_", "CM_", "BA_DEF_", "BA_", "VAL_", "CAT_DEF_", "CAT_",
                   "FILTER", "BA_DEF_DEF_", "EV_DATA_", "ENVVAR_DATA_", "SGTYPE_",
                   "SGTYPE_VAL_", "BA_DEF_SGTYPE_", "BA_DEF_SGTYPE_VAL_", "BA_DEF_REL_",
                   "BA_DEF_DEF_REL_", "BA_REL_", "BA_SGTYPE_REL_", "BU_SG_REL_",
                   "BU_EV_REL_", "BU_BO_REL_", "SG_MUL_VAL_"]:
            lines.append(f"    {ns}")
        lines.append("")
        lines.append(f"BS_ : {BUS_SPEED_BPS}")
        lines.append("")
        lines.append("BU_ : " + " ".join(NODES))
        lines.append("")

        # Messages
        for msg in self.messages:
            lines.append(f"BO_ {msg.can_id} {msg.name}: {msg.dlc} {msg.transmitter}")
            for sig in msg.signals:
                lines.append(self._format_signal(sig))
            lines.append("")

        # Comments
        lines.append('CM_ BU_ JETSON_ORIN "NVIDIA Jetson Orin NX primary compute node — APEX-LKA v4.0 at 500 kbps.";')
        lines.append('CM_ BU_ ARDUINO_PRIMARY "Arduino Mega 2560 #1 — Primary SPI MCP2515 CAN bridge.";')
        lines.append('CM_ BU_ ARDUINO_SECONDARY "Arduino Mega 2560 #2 — Redundant CAN bridge node.";')
        lines.append('CM_ BU_ RADAR_STARKENN "Starkenn Orion 77GHz FMCW Radar — 8 simultaneous tracks.";')
        lines.append('CM_ BU_ INS_MTI680 "Xsens MTi-680 GNSS/INS — Dual-antenna RTK GPS + IMU at 100 Hz.";')
        lines.append("")

        for msg in self.messages:
            if msg.comment:
                comment = msg.comment.replace('"', '\\"')
                lines.append(f'CM_ BO_ {msg.can_id} "{comment}";')
        lines.append("")

        for msg in self.messages:
            for sig in msg.signals:
                if sig.comment:
                    comment = sig.comment.replace('"', '\\"')
                    lines.append(f'CM_ SG_ {msg.can_id} {sig.name} "{comment}";')
        lines.append("")

        # Attributes
        lines.append('BA_DEF_ BO_ "GenMsgCycleTime" INT 0 10000;')
        lines.append('BA_DEF_ BO_ "GenMsgSendType" STRING;')
        lines.append('BA_DEF_ BU_ "NmStationAddress" INT 0 255;')
        lines.append('BA_DEF_ BU_ "ILUsed" STRING;')
        lines.append("")
        lines.append('BA_DEF_DEF_ "GenMsgCycleTime" 0;')
        lines.append('BA_DEF_DEF_ "GenMsgSendType" "NoMsgSendType";')
        lines.append('BA_DEF_DEF_ "NmStationAddress" 0;')
        lines.append('BA_DEF_DEF_ "ILUsed" "No";')
        lines.append("")

        for msg in self.messages:
            lines.append(f'BA_ "GenMsgCycleTime" BO_ {msg.can_id} {msg.cycle_ms};')
        lines.append("")

        for msg in self.messages:
            lines.append(f'BA_ "GenMsgSendType" BO_ {msg.can_id} "Cyclic";')
        lines.append("")

        for i, node in enumerate(NODES, 1):
            lines.append(f'BA_ "NmStationAddress" BU_ {node} {i};')
        for node in NODES:
            lines.append(f'BA_ "ILUsed" BU_ {node} "Yes";')
        lines.append("")

        # Value tables
        lines.append('VAL_ 80 SystemState 0 "INIT" 1 "STANDBY" 2 "ACTIVE" 3 "DEGRADED" 4 "FAILSAFE" 5 "SHUTDOWN" ;')
        lines.append('VAL_ 256 SteeringMode 0 "DISABLE" 1 "ANGLE_CTRL" 2 "TORQUE_CTRL" 3 "EMERGENCY" 4 "MANUAL" ;')
        lines.append('VAL_ 400 ControllerMode 0 "OFF" 1 "STANDBY" 2 "ACTIVE" 3 "OVERRIDE" 4 "FAILSAFE" ;')
        lines.append('VAL_ 400 FailsafeLevel 0 "NOMINAL" 1 "WARN" 2 "DEGRADE" 3 "FAILSAFE" 4 "SHUTDOWN" ;')
        lines.append('VAL_ 800 InsStatus 0 "INIT" 1 "COARSE_ALIGN" 2 "FINE_ALIGN" 3 "RTK_FLOAT" 4 "RTK_FIXED" 5 "DEAD_RECK" ;')
        lines.append('VAL_ 801 GnssFixType 0 "NO_FIX" 1 "SINGLE" 2 "SBAS" 3 "DGPS" 4 "RTK_FLOAT" 5 "RTK_FIXED" 6 "PPP" ;')
        lines.append('VAL_ 1024 RadarMode 0 "STANDBY" 1 "NEAR" 2 "FAR" 3 "COMBINED" 4 "CALIBRATION" ;')
        lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DBC File Parser (lightweight, for cross-validation against apex_v4.dbc)
# ---------------------------------------------------------------------------

def parse_dbc_message_ids(dbc_path: Path) -> Tuple[List[int], List[str]]:
    """
    Quick parse of a DBC file to extract CAN IDs and message names.
    Returns (can_ids, message_names).
    """
    can_ids = []
    names = []
    with open(dbc_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line.startswith("BO_ "):
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        can_ids.append(int(parts[1]))
                        names.append(parts[2].rstrip(":"))
                    except ValueError:
                        pass
    return can_ids, names


def cross_validate_dbc_file(dbc_path: Path, schema_messages: List[MessageDef]) -> bool:
    """
    Cross-validate a parsed DBC file against the programmatic schema.
    Checks that all schema CAN IDs are present in the file.
    """
    print(f"\n[CROSS-VALIDATE] Parsing DBC file: {dbc_path}")
    if not dbc_path.exists():
        print(f"  ERROR: File not found: {dbc_path}")
        return False

    file_ids, file_names = parse_dbc_message_ids(dbc_path)
    schema_ids = {m.can_id for m in schema_messages}

    print(f"  DBC file: {len(file_ids)} messages found.")
    print(f"  Schema:   {len(schema_ids)} messages expected.")

    # Check for expected IDs
    missing = schema_ids - set(file_ids)
    extra = set(file_ids) - schema_ids

    if missing:
        print(f"  ✗ MISSING IDs in DBC file: {[hex(i) for i in sorted(missing)]}")
    if extra:
        print(f"  ⚠ EXTRA IDs in DBC file (not in schema): {[hex(i) for i in sorted(extra)]}")

    # Check line count
    with open(dbc_path, "r", encoding="utf-8", errors="replace") as f:
        line_count = sum(1 for _ in f)

    print(f"  DBC file line count: {line_count}")
    if line_count < 250:
        print(f"  ⚠ WARNING: DBC file has {line_count} lines — minimum 250 expected.")
    else:
        print(f"  ✓ DBC file line count {line_count} >= 250 minimum.")

    # CRC-8 table sanity check
    known = compute_crc8(b"\x01\x02\x03\x04\x05\x06")
    print(f"  CRC-8 table check: CRC(0x01..0x06) = 0x{known:02X}")

    ok = len(missing) == 0
    if ok:
        print(f"  ✓ All {len(schema_ids)} schema CAN IDs present in DBC file.")
    return ok


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="APEX-LKA v4.0 DBC Validation & Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/gen_dbc.py --validate --input configs/can/apex_v4.dbc
              python scripts/gen_dbc.py --generate --output configs/can/apex_v4_gen.dbc
              python scripts/gen_dbc.py --validate --generate
        """),
    )
    parser.add_argument("--validate", action="store_true",
                        help="Run schema validation checks.")
    parser.add_argument("--generate", action="store_true",
                        help="Generate DBC file from Python schema.")
    parser.add_argument("--input", type=Path, default=Path("configs/can/apex_v4.dbc"),
                        help="Input DBC file to cross-validate (default: configs/can/apex_v4.dbc).")
    parser.add_argument("--output", type=Path, default=Path("configs/can/apex_v4_gen.dbc"),
                        help="Output path for generated DBC (default: configs/can/apex_v4_gen.dbc).")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output.")
    args = parser.parse_args()

    if not args.validate and not args.generate:
        parser.print_help()
        return 1

    # Build schema
    print(f"Building APEX-LKA v4.0 schema ({APEX_VERSION})...")
    schema = build_apex_schema()
    print(f"Schema: {len(schema)} messages defined.")

    exit_code = 0

    # Validate schema in-memory
    if args.validate:
        validator = DBCValidator(schema)
        ok = validator.run_all()
        if not ok:
            exit_code = 1
        # Cross-validate against file
        cross_ok = cross_validate_dbc_file(args.input, schema)
        if not cross_ok:
            exit_code = 1

    # Generate DBC file
    if args.generate:
        gen = DBCGenerator(schema)
        content = gen.generate()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
        gen_lines = content.count("\n") + 1
        print(f"\n[GENERATE] Written {gen_lines} lines to: {args.output}")
        if gen_lines < 350:
            print(f"  ⚠ Generated file has {gen_lines} lines — target is 350+.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
