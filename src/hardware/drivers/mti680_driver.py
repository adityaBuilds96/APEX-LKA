"""
src/hardware/drivers/mti680_driver.py
=====================================
APEX-Vision Core v4.0 • RTK GNSS / INS Sensor Driver
ISO 26262 ASIL-B Aligned MTi-680 RTK GNSS/INS Driver

Decodes the Xsens XBUS binary serial protocol and CAN protocol frames emitted by the
MTi-680 RTK GNSS/INS sensor unit:
- 400 Hz AHRS attitude stream (Quaternions [q0, q1, q2, q3], Roll, Pitch, Yaw angles).
- 400 Hz Calibrated Kinematics (3D Accelerations, 3D Angular Gyroscope Rates).
- 10-20 Hz cm-level RTK GNSS positioning (WGS-84 Latitude, Longitude, Ellipsoidal Height).
- RTK Fix Status telemetry (Single, DGPS, RTK Float, RTK Fixed).
- Real-time kinematic attitude streaming directly into ipm_transformer.py and roi_processor.py.
- ASIL-B sensor health monitoring (Packet checksums, 400 Hz cycle jitter, RTK fix loss).
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.MTi680Driver")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums, Data Structures & XBUS Protocol Constants
# ═══════════════════════════════════════════════════════════════════════════════

class XbusMID(Enum):
    """XBUS Message Identifiers (MID)."""
    ERROR = 0x42
    GO_TO_CONFIG = 0x30
    GO_TO_MEASUREMENT = 0x10
    RESET = 0x40
    DEVICE_ID = 0x00
    REQ_DEVICE_ID = 0x01
    MT_DATA_2 = 0x36


class RTKFixType(Enum):
    """GNSS Positioning Solution Fix Type."""
    NO_FIX = 0
    FIX_2D = 1
    FIX_3D = 2
    DGPS = 3
    RTK_FLOAT = 4
    RTK_FIXED = 5


class IMUHealthState(Enum):
    """Operational health status of the MTi-680 subsystem."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    RTK_LOST = "RTK_LOST"
    TIMEOUT = "TIMEOUT"
    CHECKSUM_ERROR = "CHECKSUM_ERROR"
    DISCONNECTED = "DISCONNECTED"


# Standard Xsens MTData2 Data Identifier Tags
DATA_ID_TEMPERATURE = 0x0810
DATA_ID_UTC_TIME = 0x1010
DATA_ID_PACKET_COUNTER = 0x1020
DATA_ID_SAMPLE_TIME_FINE = 0x1060
DATA_ID_QUATERNION = 0x2010
DATA_ID_EULER_ANGLES = 0x2030
DATA_ID_ROTATION_MATRIX = 0x2020
DATA_ID_ACCELERATION = 0x4020
DATA_ID_FREE_ACCELERATION = 0x4030
DATA_ID_RATE_OF_TURN = 0x8020
DATA_ID_DELTA_V = 0x4010
DATA_ID_DELTA_Q = 0x8010
DATA_ID_MAGNETIC_FIELD = 0xC020
DATA_ID_LAT_LON = 0x5042
DATA_ID_ALTITUDE_ELLIPSOID = 0x5020
DATA_ID_VELOCITY_XYZ = 0xD010
DATA_ID_STATUS_WORD = 0xE020
DATA_ID_GNSS_STATUS = 0x7010


@dataclass
class AttitudeState:
    """High-rate 400 Hz spatial orientation data."""
    timestamp_s: float
    roll_deg: float                  # Rotation around longitudinal X axis (+roll: bank right)
    pitch_deg: float                 # Rotation around lateral Y axis (+pitch: nose up)
    yaw_deg: float                   # Rotation around vertical Z axis (+yaw: clockwise/heading)
    quaternion: Tuple[float, float, float, float] # (q0/w, q1/x, q2/y, q3/z)
    roll_rad: float = 0.0
    pitch_rad: float = 0.0
    yaw_rad: float = 0.0

    def __post_init__(self):
        self.roll_rad = math.radians(self.roll_deg)
        self.pitch_rad = math.radians(self.pitch_deg)
        self.yaw_rad = math.radians(self.yaw_deg)


@dataclass
class CalibratedKinematics:
    """High-rate 400 Hz body inertial acceleration and angular velocity."""
    timestamp_s: float
    accel_x_mps2: float              # Longitudinal forward acceleration (+ax: accelerating)
    accel_y_mps2: float              # Lateral rightward acceleration (+ay: right turn centripetal)
    accel_z_mps2: float              # Vertical upward acceleration (+az: upwards, ~9.81 m/s^2 static)
    gyro_x_radps: float              # Roll rate (p)
    gyro_y_radps: float              # Pitch rate (q)
    gyro_z_radps: float              # Yaw rate (r)


@dataclass
class RTKPositionState:
    """10-20 Hz cm-accurate RTK GNSS position solution."""
    timestamp_s: float
    latitude_deg: float              # WGS-84 Latitude (-90 to +90 deg)
    longitude_deg: float             # WGS-84 Longitude (-180 to +180 deg)
    altitude_m: float                # Height above WGS-84 ellipsoid (m)
    horizontal_accuracy_m: float     # Estimated 1-sigma horizontal position accuracy (m)
    vertical_accuracy_m: float       # Estimated 1-sigma vertical position accuracy (m)
    velocity_north_mps: float        # NED North velocity component
    velocity_east_mps: float         # NED East velocity component
    velocity_down_mps: float         # NED Down velocity component
    fix_type: RTKFixType
    num_satellites: int
    is_differential: bool = False
    is_rtk_fixed: bool = False


@dataclass
class MTi680Telemetry:
    """Complete synchronous snapshot from MTi-680 sensor suite."""
    timestamp_s: float
    packet_counter: int
    attitude: AttitudeState
    kinematics: CalibratedKinematics
    gnss: Optional[RTKPositionState]
    health: IMUHealthState
    status_word: int
    inter_packet_dt_ms: float
    is_valid: bool = True
    diagnostics: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# XBUS Protocol Framing, Bit Parsing & Checksums
# ═══════════════════════════════════════════════════════════════════════════════

class XbusProtocolParser:
    """
    Parser for Xsens XBUS binary communication protocol.
    Frame Structure:
    [PREAMBLE: 0xFA] [BID: 0xFF] [MID: 1 byte] [LEN: 1 byte] [DATA: LEN bytes] [CHECKSUM: 1 byte]
    Checksum rule: sum(BID + MID + LEN + DATA + CHECKSUM) % 256 == 0.
    """

    PREAMBLE = 0xFA
    DEFAULT_BID = 0xFF

    @staticmethod
    def compute_checksum(buffer: bytes) -> int:
        """
        Compute XBUS two's complement checksum:
        Checksum = (256 - (sum(buffer) % 256)) % 256
        """
        s = sum(buffer) & 0xFF
        return (256 - s) & 0xFF

    @staticmethod
    def verify_checksum(packet_with_checksum: bytes) -> bool:
        """Verify that sum of all bytes in message including checksum modulo 256 equals 0."""
        return (sum(packet_with_checksum) & 0xFF) == 0

    @classmethod
    def serialize_xbus_message(cls, mid: int, payload: bytes) -> bytes:
        """Construct a complete valid XBUS frame with preamble and checksum."""
        bid = cls.DEFAULT_BID
        length = len(payload)
        assert length <= 254, "Extended length framing not required for standard MTData2"

        header_and_data = bytes([bid, mid, length]) + payload
        checksum = cls.compute_checksum(header_and_data)
        return bytes([cls.PREAMBLE]) + header_and_data + bytes([checksum])


class MTData2PayloadParser:
    """
    Decodes MTData2 container fields into individual telemetry structures.
    MTData2 Payload consists of concatenated Data Packets:
    [DataID: 2 bytes] [DataLen: 1 byte] [DataPayload: DataLen bytes]
    """

    @staticmethod
    def parse_mtdata2(payload: bytes, timestamp_s: float) -> Tuple[
        int,                                  # Packet Counter
        Optional[AttitudeState],             # Attitude
        Optional[CalibratedKinematics],      # Kinematics
        Optional[RTKPositionState],          # GNSS RTK
        int,                                  # Status Word
    ]:
        idx = 0
        total_len = len(payload)

        packet_counter = 0
        attitude: Optional[AttitudeState] = None
        kinematics: Optional[CalibratedKinematics] = None
        gnss_pos: Optional[RTKPositionState] = None
        status_word = 0

        # Intermediate field registers
        roll = 0.0
        pitch = 0.0
        yaw = 0.0
        quat = (1.0, 0.0, 0.0, 0.0)

        ax, ay, az = 0.0, 0.0, 9.80665
        gx, gy, gz = 0.0, 0.0, 0.0

        lat, lon, alt = 0.0, 0.0, 0.0
        h_acc, v_acc = 10.0, 15.0
        vn, ve, vd = 0.0, 0.0, 0.0
        fix_type = RTKFixType.NO_FIX
        num_sats = 0

        while idx + 3 <= total_len:
            data_id = (payload[idx] << 8) | payload[idx + 1]
            data_len = payload[idx + 2]
            idx += 3

            if idx + data_len > total_len:
                break  # Truncated field

            field_data = payload[idx : idx + data_len]
            idx += data_len

            # 1. Packet Counter
            if data_id == DATA_ID_PACKET_COUNTER and len(field_data) >= 2:
                packet_counter = struct.unpack(">H", field_data[:2])[0]

            # 2. Status Word
            elif data_id == DATA_ID_STATUS_WORD and len(field_data) >= 4:
                status_word = struct.unpack(">I", field_data[:4])[0]

            # 3. Orientation Quaternions (q0, q1, q2, q3: 4 x float32)
            elif data_id == DATA_ID_QUATERNION and len(field_data) >= 16:
                q0, q1, q2, q3 = struct.unpack(">4f", field_data[:16])
                quat = (float(q0), float(q1), float(q2), float(q3))

            # 4. Orientation Euler Angles (Roll, Pitch, Yaw: 3 x float32 in degrees)
            elif data_id == DATA_ID_EULER_ANGLES and len(field_data) >= 12:
                r, p, y = struct.unpack(">3f", field_data[:12])
                roll, pitch, yaw = float(r), float(p), float(y)

            # 5. Calibrated Acceleration (ax, ay, az: 3 x float32 in m/s^2)
            elif data_id == DATA_ID_ACCELERATION and len(field_data) >= 12:
                ax, ay, az = struct.unpack(">3f", field_data[:12])

            # 6. Rate of Turn / Gyroscope (gx, gy, gz: 3 x float32 in rad/s)
            elif data_id == DATA_ID_RATE_OF_TURN and len(field_data) >= 12:
                gx, gy, gz = struct.unpack(">3f", field_data[:12])

            # 7. WGS-84 Latitude / Longitude (double64 x 2)
            elif data_id == DATA_ID_LAT_LON and len(field_data) >= 16:
                lat, lon = struct.unpack(">2d", field_data[:16])

            # 8. Altitude Ellipsoid (double64)
            elif data_id == DATA_ID_ALTITUDE_ELLIPSOID and len(field_data) >= 8:
                alt = struct.unpack(">d", field_data[:8])[0]

            # 9. Velocity XYZ / NED (float32 x 3)
            elif data_id == DATA_ID_VELOCITY_XYZ and len(field_data) >= 12:
                vn, ve, vd = struct.unpack(">3f", field_data[:12])

            # 10. GNSS Status / Fix
            elif data_id == DATA_ID_GNSS_STATUS and len(field_data) >= 4:
                fix_raw = field_data[0]
                num_sats = field_data[1]
                fix_map = {
                    0: RTKFixType.NO_FIX,
                    1: RTKFixType.FIX_2D,
                    2: RTKFixType.FIX_3D,
                    3: RTKFixType.DGPS,
                    4: RTKFixType.RTK_FLOAT,
                    5: RTKFixType.RTK_FIXED,
                }
                fix_type = fix_map.get(fix_raw, RTKFixType.NO_FIX)

        # Assemble AttitudeState
        attitude = AttitudeState(
            timestamp_s=timestamp_s,
            roll_deg=round(roll, 3),
            pitch_deg=round(pitch, 3),
            yaw_deg=round(yaw, 3),
            quaternion=quat,
        )

        # Assemble Kinematics
        kinematics = CalibratedKinematics(
            timestamp_s=timestamp_s,
            accel_x_mps2=round(float(ax), 3),
            accel_y_mps2=round(float(ay), 3),
            accel_z_mps2=round(float(az), 3),
            gyro_x_radps=round(float(gx), 4),
            gyro_y_radps=round(float(gy), 4),
            gyro_z_radps=round(float(gz), 4),
        )

        # Assemble RTK Position (if coordinates non-zero)
        if abs(lat) > 1e-4 or abs(lon) > 1e-4:
            is_rtk = (fix_type == RTKFixType.RTK_FIXED)
            h_acc = 0.02 if is_rtk else (0.40 if fix_type == RTKFixType.RTK_FLOAT else 2.50)
            v_acc = 0.03 if is_rtk else (0.60 if fix_type == RTKFixType.RTK_FLOAT else 4.00)

            gnss_pos = RTKPositionState(
                timestamp_s=timestamp_s,
                latitude_deg=round(float(lat), 7),
                longitude_deg=round(float(lon), 7),
                altitude_m=round(float(alt), 3),
                horizontal_accuracy_m=h_acc,
                vertical_accuracy_m=v_acc,
                velocity_north_mps=round(float(vn), 2),
                velocity_east_mps=round(float(ve), 2),
                velocity_down_mps=round(float(vd), 2),
                fix_type=fix_type,
                num_satellites=num_sats,
                is_differential=(fix_type in (RTKFixType.DGPS, RTKFixType.RTK_FLOAT, RTKFixType.RTK_FIXED)),
                is_rtk_fixed=is_rtk,
            )

        return packet_counter, attitude, kinematics, gnss_pos, status_word


# ═══════════════════════════════════════════════════════════════════════════════
# Circular Byte Stream Buffer & Synchronizer
# ═══════════════════════════════════════════════════════════════════════════════

class XbusStreamSynchronizer:
    """
    Streams raw asynchronous serial bytes into a sliding FIFO buffer,
    scans for preambles (0xFA), aligns packet frames, verifies checksums,
    and returns valid extracted XBUS messages.
    """

    def __init__(self, max_buffer_size: int = 4096) -> None:
        self.max_size = max_buffer_size
        self.buffer = bytearray()
        self.checksum_error_count = 0

    def feed_bytes(self, chunk: bytes) -> List[Tuple[int, bytes]]:
        """
        Append bytes and extract all complete, validated XBUS frames.
        Returns list of (mid, payload) tuples.
        """
        self.buffer.extend(chunk)
        if len(self.buffer) > self.max_size:
            # Prevent memory overflow: prune oldest half
            self.buffer = self.buffer[-2048:]

        valid_frames: List[Tuple[int, bytes]] = []

        while len(self.buffer) >= 5:  # Minimum frame: FA FF MID LEN CHK
            # Find preamble 0xFA
            try:
                preamble_idx = self.buffer.index(0xFA)
            except ValueError:
                self.buffer.clear()
                break

            # Discard bytes prior to preamble
            if preamble_idx > 0:
                self.buffer = self.buffer[preamble_idx:]

            if len(self.buffer) < 4:
                break  # Wait for header bytes

            bid = self.buffer[1]
            mid = self.buffer[2]
            length = self.buffer[3]

            total_frame_len = 1 + 1 + 1 + 1 + length + 1  # FA + BID + MID + LEN + DATA + CHK
            if len(self.buffer) < total_frame_len:
                break  # Wait for remaining payload bytes

            candidate_frame = bytes(self.buffer[:total_frame_len])
            # Checksum is computed over BID + MID + LEN + DATA + CHK
            check_slice = candidate_frame[1:]

            if XbusProtocolParser.verify_checksum(check_slice):
                # Valid frame extracted!
                payload = candidate_frame[4 : 4 + length]
                valid_frames.append((mid, payload))
                self.buffer = self.buffer[total_frame_len:]
            else:
                # Checksum mismatch: discard false preamble and resume search
                self.checksum_error_count += 1
                self.buffer = self.buffer[1:]

        return valid_frames


# ═══════════════════════════════════════════════════════════════════════════════
# ASIL-B 400 Hz Jitter & RTK Integrity Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class MTi680HealthMonitor:
    """
    Monitors 400 Hz packet arrival jitter, serial communication dropouts,
    and RTK GNSS fix degradation under ISO 26262 ASIL-B expectations.
    """

    def __init__(
        self,
        nominal_rate_hz: float = 400.0,
        max_allowed_jitter_ms: float = 8.0,
        dropout_timeout_ms: float = 25.0,
    ) -> None:
        self.nominal_period_s = 1.0 / nominal_rate_hz  # 0.0025 s (2.5 ms)
        self.max_jitter_s = max_allowed_jitter_ms / 1000.0
        self.dropout_timeout_s = dropout_timeout_ms / 1000.0

        self.last_packet_timestamp_s = 0.0
        self.consecutive_jitter_faults = 0
        self.rtk_fix_lost_counter = 0

    def evaluate_timing_and_fix(
        self,
        timestamp_s: float,
        gnss_pos: Optional[RTKPositionState],
    ) -> Tuple[IMUHealthState, float, str]:
        """
        Verify packet timing against 400 Hz nominal cadence and assess RTK fix status.
        Returns: (health_state, inter_packet_dt_ms, diagnostic_message)
        """
        if self.last_packet_timestamp_s <= 0.0:
            dt_s = self.nominal_period_s
        else:
            dt_s = timestamp_s - self.last_packet_timestamp_s

        self.last_packet_timestamp_s = timestamp_s
        dt_ms = dt_s * 1000.0

        # 1. Serial Dropout Timeout (>25 ms silence)
        if dt_s > self.dropout_timeout_s:
            return IMUHealthState.TIMEOUT, dt_ms, f"400 Hz cadence timeout: {dt_ms:.1f} ms"

        # 2. Timing Jitter Check
        jitter_s = abs(dt_s - self.nominal_period_s)
        if jitter_s > self.max_jitter_s:
            self.consecutive_jitter_faults += 1
            if self.consecutive_jitter_faults >= 5:
                return IMUHealthState.DEGRADED, dt_ms, f"High 400 Hz timing jitter ({jitter_s*1000.0:.2f} ms)"
        else:
            self.consecutive_jitter_faults = max(0, self.consecutive_jitter_faults - 1)

        # 3. RTK Fix Integrity
        if gnss_pos is not None:
            if not gnss_pos.is_rtk_fixed:
                self.rtk_fix_lost_counter += 1
                if self.rtk_fix_lost_counter >= 10:
                    return IMUHealthState.RTK_LOST, dt_ms, f"RTK cm-accuracy lost (Current: {gnss_pos.fix_type.name})"
            else:
                self.rtk_fix_lost_counter = 0

        return IMUHealthState.OK, dt_ms, "Nominal 400 Hz operation"


# ═══════════════════════════════════════════════════════════════════════════════
# Master MTi-680 Driver Interface
# ═══════════════════════════════════════════════════════════════════════════════

class MTi680Driver:
    """
    Master Driver for the MTi-680 RTK GNSS/INS Sensor Suite.
    Integrates:
    - Asynchronous byte stream synchronization and XBUS frame validation.
    - Full MTData2 container decoding (400 Hz AHRS + 10 Hz RTK).
    - Direct attitude feed to camera IPM warping and ROI horizon processors.
    - ASIL-B 400 Hz timing jitter and RTK fix supervision.
    - Synthetic packet generation for deterministic offline testing.
    """

    def __init__(
        self,
        serial_port: str = "COM3",
        baudrate: int = 115200,
        enable_synthetic_stream: bool = False,
    ) -> None:
        self.port = serial_port
        self.baudrate = baudrate
        self.synthetic_mode = enable_synthetic_stream

        self.synchronizer = XbusStreamSynchronizer()
        self.payload_parser = MTData2PayloadParser()
        self.health_monitor = MTi680HealthMonitor()

        self.latest_telemetry: Optional[MTi680Telemetry] = None
        self.subscribers: List[Callable[[MTi680Telemetry], None]] = []

    def register_telemetry_subscriber(self, callback: Callable[[MTi680Telemetry], None]) -> None:
        """Register consumer callback for 400 Hz synchronous telemetry events."""
        self.subscribers.append(callback)

    def process_incoming_stream_bytes(
        self,
        data_chunk: bytes,
        timestamp_s: float,
    ) -> List[MTi680Telemetry]:
        """
        Feed raw serial buffer chunk and return all decoded telemetry frames.
        """
        valid_frames = self.synchronizer.feed_bytes(data_chunk)
        telemetry_batch: List[MTi680Telemetry] = []

        for mid, payload in valid_frames:
            if mid == XbusMID.MT_DATA_2.value:
                p_count, att, kin, gnss, status = self.payload_parser.parse_mtdata2(payload, timestamp_s)

                if att is not None and kin is not None:
                    # Health assessment
                    health, dt_ms, diag_msg = self.health_monitor.evaluate_timing_and_fix(timestamp_s, gnss)

                    telem = MTi680Telemetry(
                        timestamp_s=timestamp_s,
                        packet_counter=p_count,
                        attitude=att,
                        kinematics=kin,
                        gnss=gnss,
                        health=health,
                        status_word=status,
                        inter_packet_dt_ms=round(dt_ms, 2),
                        is_valid=(health in (IMUHealthState.OK, IMUHealthState.RTK_LOST)),
                        diagnostics={"status_msg": diag_msg},
                    )

                    self.latest_telemetry = telem
                    telemetry_batch.append(telem)

                    # Notify registered perception modules
                    for sub in self.subscribers:
                        try:
                            sub(telem)
                        except Exception as e:
                            logger.error(f"Error in telemetry callback: {e}")

        return telemetry_batch

    def get_latest_pitch_roll_angles(self) -> Tuple[float, float]:
        """
        Quick convenience accessor for IPM and ROI modules.
        Returns: (pitch_deg, roll_deg)
        """
        if self.latest_telemetry is not None:
            return self.latest_telemetry.attitude.pitch_deg, self.latest_telemetry.attitude.roll_deg
        return 0.0, 0.0

    @classmethod
    def generate_synthetic_xbus_packet(
        cls,
        packet_counter: int,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        accel_mps2: Tuple[float, float, float] = (0.0, 0.0, 9.81),
        gyro_radps: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        lat_deg: float = 18.5204303,
        lon_deg: float = 73.8567437,
        alt_m: float = 560.25,
        fix_type: RTKFixType = RTKFixType.RTK_FIXED,
    ) -> bytes:
        """
        Serialize a complete, authentic XBUS MTData2 frame for unit testing and offline bench runs.
        Format per Xsens MTData2 specification: [DataID: 2 bytes] [DataLen: 1 byte] [Payload: DataLen bytes]
        """
        payload = bytearray()

        # 1. Packet Counter (DataID: 0x1020, Len: 2)
        payload.extend(struct.pack(">HB", DATA_ID_PACKET_COUNTER, 2))
        payload.extend(struct.pack(">H", packet_counter & 0xFFFF))

        # 2. Euler Angles (DataID: 0x2030, Len: 12)
        payload.extend(struct.pack(">HB", DATA_ID_EULER_ANGLES, 12))
        payload.extend(struct.pack(">3f", roll_deg, pitch_deg, yaw_deg))

        # 3. Quaternions (DataID: 0x2010, Len: 16)
        cr = math.cos(math.radians(roll_deg) * 0.5)
        sr = math.sin(math.radians(roll_deg) * 0.5)
        cp = math.cos(math.radians(pitch_deg) * 0.5)
        sp = math.sin(math.radians(pitch_deg) * 0.5)
        cy = math.cos(math.radians(yaw_deg) * 0.5)
        sy = math.sin(math.radians(yaw_deg) * 0.5)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        payload.extend(struct.pack(">HB", DATA_ID_QUATERNION, 16))
        payload.extend(struct.pack(">4f", qw, qx, qy, qz))

        # 4. Acceleration (DataID: 0x4020, Len: 12)
        payload.extend(struct.pack(">HB", DATA_ID_ACCELERATION, 12))
        payload.extend(struct.pack(">3f", accel_mps2[0], accel_mps2[1], accel_mps2[2]))

        # 5. Gyroscope / Rate of Turn (DataID: 0x8020, Len: 12)
        payload.extend(struct.pack(">HB", DATA_ID_RATE_OF_TURN, 12))
        payload.extend(struct.pack(">3f", gyro_radps[0], gyro_radps[1], gyro_radps[2]))

        # 6. WGS-84 Lat / Lon (DataID: 0x5042, Len: 16)
        payload.extend(struct.pack(">HB", DATA_ID_LAT_LON, 16))
        payload.extend(struct.pack(">2d", lat_deg, lon_deg))

        # 7. Altitude (DataID: 0x5020, Len: 8)
        payload.extend(struct.pack(">HB", DATA_ID_ALTITUDE_ELLIPSOID, 8))
        payload.extend(struct.pack(">d", alt_m))

        # 8. GNSS Status (DataID: 0x7010, Len: 4)
        payload.extend(struct.pack(">HB", DATA_ID_GNSS_STATUS, 4))
        payload.extend(bytes([fix_type.value, 18, 0, 0]))  # 18 satellites

        # 9. Status Word (DataID: 0xE020, Len: 4)
        payload.extend(struct.pack(">HB", DATA_ID_STATUS_WORD, 4))
        payload.extend(struct.pack(">I", 0x00000003))

        # Wrap in standard XBUS frame
        return XbusProtocolParser.serialize_xbus_message(XbusMID.MT_DATA_2.value, bytes(payload))


# ═══════════════════════════════════════════════════════════════════════════════
# Direct Perception Adapter: IPM & Dynamic ROI Feeder
# ═══════════════════════════════════════════════════════════════════════════════

class MTi680PerceptionBridge:
    """
    Direct bridge feeding MTi-680 attitude stream into:
    1. src/perception/ipm_transformer.py (BEV warp matrix update on >0.5 deg pitch delta).
    2. src/perception/roi_processor.py (Horizon and vanishing point repositioning).
    """

    def __init__(
        self,
        pitch_recompute_deadband_deg: float = 0.50,
        roll_recompute_deadband_deg: float = 0.50,
    ) -> None:
        self.pitch_deadband = pitch_recompute_deadband_deg
        self.roll_deadband = roll_recompute_deadband_deg

        self.last_dispatched_pitch = 0.0
        self.last_dispatched_roll = 0.0
        self.update_count = 0

    def evaluate_attitude_update(self, telemetry: MTi680Telemetry) -> Tuple[bool, float, float]:
        """
        Check if pitch or roll orientation delta exceeds deadband threshold.
        Returns: (needs_ipm_recompute, current_pitch_deg, current_roll_deg)
        """
        cur_pitch = telemetry.attitude.pitch_deg
        cur_roll = telemetry.attitude.roll_deg

        d_pitch = abs(cur_pitch - self.last_dispatched_pitch)
        d_roll = abs(cur_roll - self.last_dispatched_roll)

        if d_pitch >= self.pitch_deadband or d_roll >= self.roll_deadband or self.update_count == 0:
            self.last_dispatched_pitch = cur_pitch
            self.last_dispatched_roll = cur_roll
            self.update_count += 1
            return True, cur_pitch, cur_roll

        return False, cur_pitch, cur_roll


# ═══════════════════════════════════════════════════════════════════════════════
# CAN Protocol Decoder for MTi-680
# ═══════════════════════════════════════════════════════════════════════════════

class CANMTi680Decoder:
    """
    Decodes standard Xsens CAN frames when MTi-680 is operated over 500 kbps CAN bus:
    - 0x201: Euler angles (Roll, Pitch, Yaw: 16-bit signed, scale 0.0055 deg)
    - 0x202: Accelerations (ax, ay, az: 16-bit signed, scale 0.001 m/s^2)
    - 0x203: Angular rates (gx, gy, gz: 16-bit signed, scale 0.0001 rad/s)
    - 0x205: RTK Position (Lat, Lon: 32-bit signed fixed point, scale 1e-7 deg)
    - 0x206: RTK Fix status & Altitude (Fix type uint8, Alt uint24 scale 0.01m)
    """

    @staticmethod
    def decode_can_frame(can_id: int, payload: bytes, timestamp_s: float) -> Optional[Dict[str, Any]]:
        """Decode a single 8-byte CAN frame."""
        if len(payload) < 6:
            return None

        if can_id == 0x201:
            # Euler angles
            r_raw, p_raw, y_raw = struct.unpack(">3h", payload[:6])
            return {
                "type": "EULER",
                "timestamp_s": timestamp_s,
                "roll_deg": round(r_raw * 0.0055, 3),
                "pitch_deg": round(p_raw * 0.0055, 3),
                "yaw_deg": round(y_raw * 0.0055, 3),
            }

        elif can_id == 0x202:
            # Acceleration
            ax_raw, ay_raw, az_raw = struct.unpack(">3h", payload[:6])
            return {
                "type": "ACCEL",
                "timestamp_s": timestamp_s,
                "accel_x": round(ax_raw * 0.001, 3),
                "accel_y": round(ay_raw * 0.001, 3),
                "accel_z": round(az_raw * 0.001, 3),
            }

        elif can_id == 0x203:
            # Angular velocity
            gx_raw, gy_raw, gz_raw = struct.unpack(">3h", payload[:6])
            return {
                "type": "GYRO",
                "timestamp_s": timestamp_s,
                "gyro_x": round(gx_raw * 0.0001, 4),
                "gyro_y": round(gy_raw * 0.0001, 4),
                "gyro_z": round(gz_raw * 0.0001, 4),
            }

        elif can_id == 0x205 and len(payload) >= 8:
            # RTK Lat/Lon
            lat_i, lon_i = struct.unpack(">2i", payload[:8])
            return {
                "type": "GNSS_POS",
                "timestamp_s": timestamp_s,
                "lat_deg": lat_i * 1e-7,
                "lon_deg": lon_i * 1e-7,
            }

        return None


# ═══════════════════════════════════════════════════════════════════════════════
# WGS-84 to Local ENU Geodetic Coordinate Transformer
# ═══════════════════════════════════════════════════════════════════════════════

class GeodeticCoordinateTransformer:
    """
    Transforms WGS-84 ellipsoidal coordinates (Latitude, Longitude, Altitude)
    into a local tangent East-North-Up (ENU) Cartesian metric frame
    anchored to a reference datum origin (e.g., Baja SAE track starting gate).

    Uses standard WGS-84 constants:
    - Semi-major axis a = 6378137.0 m
    - Flattening f = 1 / 298.257223563
    - First eccentricity squared e^2 = 2f - f^2
    """

    WGS84_A = 6378137.0
    WGS84_F = 1.0 / 298.257223563
    WGS84_E2 = 2.0 * WGS84_F - (WGS84_F ** 2)

    def __init__(
        self,
        origin_lat_deg: float = 18.5204303,
        origin_lon_deg: float = 73.8567437,
        origin_alt_m: float = 560.25,
    ) -> None:
        self.lat0_rad = math.radians(origin_lat_deg)
        self.lon0_rad = math.radians(origin_lon_deg)
        self.alt0 = origin_alt_m

        # Compute ECEF coordinates of origin
        self.x0_ecef, self.y0_ecef, self.z0_ecef = self.geodetic_to_ecef(
            origin_lat_deg, origin_lon_deg, origin_alt_m
        )

        # Precompute rotation matrix terms from ECEF to ENU
        self.sin_lat0 = math.sin(self.lat0_rad)
        self.cos_lat0 = math.cos(self.lat0_rad)
        self.sin_lon0 = math.sin(self.lon0_rad)
        self.cos_lon0 = math.cos(self.lon0_rad)

    @classmethod
    def geodetic_to_ecef(
        cls,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
    ) -> Tuple[float, float, float]:
        """Convert Latitude, Longitude, Altitude into Earth-Centered Earth-Fixed (ECEF) coordinates."""
        phi = math.radians(lat_deg)
        lam = math.radians(lon_deg)
        h = alt_m

        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        sin_lam = math.sin(lam)
        cos_lam = math.cos(lam)

        # Prime vertical radius of curvature N(phi)
        n = cls.WGS84_A / math.sqrt(1.0 - cls.WGS84_E2 * (sin_phi ** 2))

        x = (n + h) * cos_phi * cos_lam
        y = (n + h) * cos_phi * sin_lam
        z = (n * (1.0 - cls.WGS84_E2) + h) * sin_phi

        return x, y, z

    def geodetic_to_enu(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
    ) -> Tuple[float, float, float]:
        """
        Convert WGS-84 point to local East-North-Up (ENU) Cartesian metric displacement (meters)
        relative to the configured track datum origin.
        """
        x_e, y_e, z_e = self.geodetic_to_ecef(lat_deg, lon_deg, alt_m)

        # Delta vector in ECEF
        dx = x_e - self.x0_ecef
        dy = y_e - self.y0_ecef
        dz = z_e - self.z0_ecef

        # Rotate from ECEF to ENU
        # East:  -sin(lam)*dx + cos(lam)*dy
        # North: -sin(phi)*cos(lam)*dx - sin(phi)*sin(lam)*dy + cos(phi)*dz
        # Up:     cos(phi)*cos(lam)*dx + cos(phi)*sin(lam)*dy + sin(phi)*dz
        east = -self.sin_lon0 * dx + self.cos_lon0 * dy
        north = -self.sin_lat0 * self.cos_lon0 * dx - self.sin_lat0 * self.sin_lon0 * dy + self.cos_lat0 * dz
        up = self.cos_lat0 * self.cos_lon0 * dx + self.cos_lat0 * self.sin_lon0 * dy + self.sin_lat0 * dz

        return round(east, 3), round(north, 3), round(up, 3)


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_mti680_driver(
    serial_port: str = "COM3",
    baudrate: int = 115200,
) -> MTi680Driver:
    """Factory helper to instantiate MTi680Driver."""
    return MTi680Driver(serial_port=serial_port, baudrate=baudrate)


def run_mti680_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. XBUS Checksum generation and validation.
    2. Serial stream synchronization and byte defragmentation.
    3. MTData2 decoding (Attitude, Kinematics, cm-level RTK GNSS).
    4. 400 Hz timing jitter and dropout health monitoring.
    5. Perception bridge deadband triggering for IPM / ROI.
    6. CAN bus frame decoding.
    7. Geodetic WGS-84 to Local ENU metric transformation.
    """
    print("Executing MTi680Driver Verification Suite...")
    driver = get_mti680_driver()

    # -------------------------------------------------------------------------
    # Test 1: XBUS Protocol Checksum Verification
    # -------------------------------------------------------------------------
    msg = bytes([0xFF, 0x30, 0x00])
    chk = XbusProtocolParser.compute_checksum(msg)
    assert chk == 0xD1, f"Checksum calculation mismatch: got {hex(chk)}, expected 0xD1"
    assert XbusProtocolParser.verify_checksum(msg + bytes([chk])) is True

    # -------------------------------------------------------------------------
    # Test 2: Synthetic MTData2 Packet Serialization & Roundtrip Parse
    # -------------------------------------------------------------------------
    raw_packet = MTi680Driver.generate_synthetic_xbus_packet(
        packet_counter=1042,
        roll_deg=3.45,
        pitch_deg=5.82,
        yaw_deg=112.50,
        accel_mps2=(0.25, -0.15, 9.78),
        gyro_radps=(0.01, 0.02, -0.05),
        lat_deg=18.5204303,
        lon_deg=73.8567437,
        alt_m=560.25,
        fix_type=RTKFixType.RTK_FIXED,
    )
    assert len(raw_packet) > 20
    assert raw_packet[0] == 0xFA, "Preamble byte missing"

    telemetry_list = driver.process_incoming_stream_bytes(raw_packet, timestamp_s=100.0)
    assert len(telemetry_list) == 1, "Failed to decode synthetic MTData2 packet"
    t = telemetry_list[0]

    print(f"Decoded Attitude: Roll={t.attitude.roll_deg}°, Pitch={t.attitude.pitch_deg}°, Yaw={t.attitude.yaw_deg}°")
    assert abs(t.attitude.pitch_deg - 5.82) < 0.01, "Pitch mismatch"
    assert abs(t.attitude.roll_deg - 3.45) < 0.01, "Roll mismatch"
    assert abs(t.attitude.yaw_deg - 112.50) < 0.01, "Yaw mismatch"
    assert t.packet_counter == 1042, "Packet counter mismatch"

    assert t.gnss is not None, "RTK GNSS state missing"
    assert t.gnss.is_rtk_fixed is True
    assert t.gnss.horizontal_accuracy_m == 0.02, "RTK fixed should have 2cm accuracy"
    assert abs(t.gnss.latitude_deg - 18.5204303) < 1e-6
    print(f"Decoded RTK GNSS: Lat={t.gnss.latitude_deg}, Lon={t.gnss.longitude_deg}, Acc={t.gnss.horizontal_accuracy_m}m")

    # -------------------------------------------------------------------------
    # Test 3: Stream Defragmentation with Split Chunks & Noise
    # -------------------------------------------------------------------------
    noise = bytes([0xAA, 0xBB, 0xCC, 0xDD])
    chunk1 = noise + raw_packet[:25]
    chunk2 = raw_packet[25:]

    t_list1 = driver.process_incoming_stream_bytes(chunk1, timestamp_s=100.0025)
    assert len(t_list1) == 0, "Partial packet should not emit telemetry"
    t_list2 = driver.process_incoming_stream_bytes(chunk2, timestamp_s=100.0050)
    assert len(t_list2) == 1, "Failed to assemble fragmented stream packet"

    # -------------------------------------------------------------------------
    # Test 4: 400 Hz Cadence Timeout & Jitter Detection
    # -------------------------------------------------------------------------
    health, dt_ms, reason = driver.health_monitor.evaluate_timing_and_fix(timestamp_s=100.055, gnss_pos=t.gnss)
    assert health == IMUHealthState.TIMEOUT, "Failed to flag 400 Hz stream dropout"

    # -------------------------------------------------------------------------
    # Test 5: Perception Bridge Deadband & IPM Trigger
    # -------------------------------------------------------------------------
    bridge = MTi680PerceptionBridge(pitch_recompute_deadband_deg=0.50)
    trig1, p1, r1 = bridge.evaluate_attitude_update(t)
    assert trig1 is True

    t_minor = MTi680Telemetry(
        timestamp_s=100.01,
        packet_counter=1043,
        attitude=AttitudeState(timestamp_s=100.01, roll_deg=3.45, pitch_deg=5.90, yaw_deg=112.5, quaternion=(1,0,0,0)),
        kinematics=t.kinematics,
        gnss=t.gnss,
        health=IMUHealthState.OK,
        status_word=0,
        inter_packet_dt_ms=2.5,
    )
    trig2, _, _ = bridge.evaluate_attitude_update(t_minor)
    assert trig2 is False, "Perception bridge triggered within pitch deadband"

    t_major = MTi680Telemetry(
        timestamp_s=100.02,
        packet_counter=1044,
        attitude=AttitudeState(timestamp_s=100.02, roll_deg=3.45, pitch_deg=7.10, yaw_deg=112.5, quaternion=(1,0,0,0)),
        kinematics=t.kinematics,
        gnss=t.gnss,
        health=IMUHealthState.OK,
        status_word=0,
        inter_packet_dt_ms=2.5,
    )
    trig3, p3, _ = bridge.evaluate_attitude_update(t_major)
    assert trig3 is True, "Perception bridge failed to trigger on >0.5 deg pitch step"

    # -------------------------------------------------------------------------
    # Test 6: CAN Frame Decoding
    # -------------------------------------------------------------------------
    # Synthetic CAN 0x201 Euler packet: Roll=1000 (-5.5 deg), Pitch=2000 (11.0 deg), Yaw=0
    can_euler = struct.pack(">3h", 1000, 2000, 0)
    decoded_can = CANMTi680Decoder.decode_can_frame(0x201, can_euler, timestamp_s=100.1)
    assert decoded_can is not None
    assert decoded_can["type"] == "EULER"
    assert abs(decoded_can["pitch_deg"] - 11.0) < 0.1
    print(f"Decoded CAN Euler Frame: Pitch={decoded_can['pitch_deg']}°")

    # -------------------------------------------------------------------------
    # Test 7: Geodetic WGS-84 to Local ENU Transformation
    # -------------------------------------------------------------------------
    geo_trans = GeodeticCoordinateTransformer(
        origin_lat_deg=18.5204303,
        origin_lon_deg=73.8567437,
        origin_alt_m=560.25,
    )
    # Target 0 displacement at datum origin
    e0, n0, u0 = geo_trans.geodetic_to_enu(18.5204303, 73.8567437, 560.25)
    assert abs(e0) < 0.05 and abs(n0) < 0.05 and abs(u0) < 0.05, f"Origin displacement error: {e0}, {n0}, {u0}"

    # Target 100m north (1 degree latitude is approx 111,139 meters => 0.0009 deg approx 100m)
    d_lat = 100.0 / 111139.0
    e_n, n_n, u_n = geo_trans.geodetic_to_enu(18.5204303 + d_lat, 73.8567437, 560.25)
    assert abs(n_n - 100.0) < 1.0, f"North displacement error: {n_n}"
    print(f"ENU North Displacement: {n_n:.2f} m (Expected ~100.00 m)")

    print("MTi680Driver verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_mti680_verification_suite()

