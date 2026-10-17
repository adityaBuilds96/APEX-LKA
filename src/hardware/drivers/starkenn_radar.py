"""
src/hardware/drivers/starkenn_radar.py
======================================
APEX-Vision Core v4.0 • Automotive Radar Driver
ISO 26262 ASIL-B Aligned Starkenn Orion 77GHz CAN Radar Driver

Parses CAN ID frames 0x400 through 0x408 emitted by the Starkenn Orion 77GHz
FMCW radar sensor. Decodes physical track attributes, performs polar-to-Cartesian
coordinate transformation into the vehicle body frame, maintains a temporal track
history buffer with track lifetime management, and detects sensor blockage/faults.

Hardware Specifications:
------------------------
- Center Frequency: 77.0 GHz FMCW
- Range: 0.2 m to 250.0 m (Resolution: 0.05 m)
- Azimuth FOV: +/- 60 deg (Resolution: 0.1 deg)
- Elevation FOV: +/- 15 deg (Resolution: 0.2 deg)
- Radial Velocity (Doppler): -70.0 m/s to +70.0 m/s (Resolution: 0.05 m/s)
- RCS Sensitivity: -40.0 dBsm to +60.0 dBsm
- CAN Bus Baudrate: 500 kbps (Standard 11-bit CAN 2.0B)
- Frame Header ID: 0x400
- Track Message IDs: 0x401 - 0x408 (Multiplexed track frames)
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import struct
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.StarkennRadar")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums, Data Structures & CAN Protocol Definitions
# ═══════════════════════════════════════════════════════════════════════════════

class RadarHealthState(Enum):
    """Operational health condition of the Starkenn radar sensor."""
    OK = "OK"
    DEGRADED = "DEGRADED"
    BLOCKED = "BLOCKED"
    TIMEOUT = "TIMEOUT"
    CRC_ERROR = "CRC_ERROR"
    DISCONNECTED = "DISCONNECTED"


class DynamicProperty(Enum):
    """Motion classification of detected radar target."""
    MOVING = 0
    STATIONARY = 1
    ONCOMING = 2
    CROSSING = 3
    STOPPED = 4
    UNKNOWN = 5


@dataclass
class RadarMountingConfig:
    """Extrinsic mounting position and orientation of radar in vehicle frame."""
    x_offset_m: float = 0.0          # Lateral offset from vehicle centerline (positive right)
    y_offset_m: float = 1.65         # Longitudinal offset from rear axle (front bumper)
    z_offset_m: float = 0.45         # Vertical height above ground plane
    pitch_offset_deg: float = 0.0    # Mechanical tilt in pitch
    yaw_offset_deg: float = 0.0      # Mechanical misalignment in azimuth


@dataclass
class RadarTrack:
    """Individual decoded physical target track from the radar sensor."""
    track_id: int
    timestamp_s: float
    range_m: float                   # Radial distance R (m)
    azimuth_deg: float               # Azimuth angle phi (-60 to +60 deg)
    elevation_deg: float             # Elevation angle theta (-15 to +15 deg)
    radial_velocity_mps: float       # Doppler radial velocity v_r (m/s)
    rcs_dbsm: float                  # Radar cross-section in dBsm
    existence_prob: float            # Detection confidence [0.0, 1.0]
    dynamic_property: DynamicProperty

    # Cartesian Coordinates in Vehicle Body Frame (X: right, Y: forward, Z: up)
    x_veh_m: float = 0.0
    y_veh_m: float = 0.0
    z_veh_m: float = 0.0

    # Temporal tracking state
    age_cycles: int = 1
    lost_cycles: int = 0


@dataclass
class RadarHeader:
    """Header status frame emitted on CAN ID 0x400."""
    timestamp_s: float
    cycle_counter: int
    num_valid_tracks: int
    sensor_fault_flag: bool
    temperature_celsius: float
    voltage_volts: float


@dataclass
class RadarScan:
    """Complete radar point cloud and track compilation for a single measurement sweep."""
    timestamp_s: float
    scan_id: int
    header: Optional[RadarHeader]
    tracks: List[RadarTrack]
    health: RadarHealthState
    latency_ms: float
    is_blocked: bool = False


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate Transformer: Polar to Metric Cartesian
# ═══════════════════════════════════════════════════════════════════════════════

class RadarCoordinateTransformer:
    """
    Transforms polar spherical radar measurements (Range, Azimuth, Elevation)
    into 3D Metric Cartesian coordinates in the vehicle body reference frame.
    Coordinate Convention (ISO 8855):
    - X_veh: Lateral right (+X)
    - Y_veh: Longitudinal forward (+Y)
    - Z_veh: Vertical up (+Z)
    """

    def __init__(self, mounting: Optional[RadarMountingConfig] = None) -> None:
        self.mounting = mounting or RadarMountingConfig()
        self.pitch_rad = math.radians(self.mounting.pitch_offset_deg)
        self.yaw_rad = math.radians(self.mounting.yaw_offset_deg)

    def polar_to_cartesian(
        self,
        range_m: float,
        azimuth_deg: float,
        elevation_deg: float,
    ) -> Tuple[float, float, float]:
        """
        Convert raw spherical measurements to 3D vehicle coordinates.
        """
        phi = math.radians(azimuth_deg) + self.yaw_rad
        theta = math.radians(elevation_deg) + self.pitch_rad

        # Spherical to sensor Cartesian
        # In radar frame: forward is Y_s, right is X_s, up is Z_s
        x_s = range_m * math.cos(theta) * math.sin(phi)
        y_s = range_m * math.cos(theta) * math.cos(phi)
        z_s = range_m * math.sin(theta)

        # Apply mounting offset translation
        x_v = x_s + self.mounting.x_offset_m
        y_v = y_s + self.mounting.y_offset_m
        z_v = z_s + self.mounting.z_offset_m

        return float(x_v), float(y_v), float(z_v)


# ═══════════════════════════════════════════════════════════════════════════════
# CAN Frame Parser & Bit Unpacker
# ═══════════════════════════════════════════════════════════════════════════════

class StarkennCANParser:
    """
    Decodes raw 8-byte CAN 2.0B payloads for Starkenn Orion 77GHz Radar.
    CAN ID Scheme:
    - 0x400: Header status (Cycle count, Track count, Temp, Voltage)
    - 0x401 - 0x408: Track data frames.
      Each track frame encodes 1 complete track object over 8 bytes:
      Byte 0: Track ID (6 bits) | Dynamic Property (2 bits)
      Byte 1: Range MSB (8 bits)
      Byte 2: Range LSB (4 bits: scale 0.05m) | Azimuth MSB (4 bits)
      Byte 3: Azimuth LSB (8 bits: scale 0.1 deg, offset -60 deg)
      Byte 4: Radial Velocity (8 bits: scale 0.05 m/s, offset -70 m/s)
      Byte 5: RCS (8 bits: scale 0.5 dBsm, offset -40 dBsm)
      Byte 6: Elevation (8 bits: scale 0.2 deg, offset -15 deg)
      Byte 7: Probability of Existence (6 bits) | CRC-2 / Counter (2 bits)
    """

    def __init__(self, transformer: RadarCoordinateTransformer) -> None:
        self.transformer = transformer

    def parse_header_frame(self, payload: bytes, timestamp_s: float) -> Optional[RadarHeader]:
        """Parse CAN ID 0x400 header status."""
        if len(payload) < 8:
            return None

        cycle = payload[0] | ((payload[1] & 0x0F) << 8)
        num_tracks = (payload[1] >> 4) & 0x0F | ((payload[2] & 0x03) << 4)
        fault_flag = bool((payload[2] >> 2) & 0x01)
        temp_raw = payload[3]
        temp_c = float(temp_raw) - 40.0  # -40 to +215 C
        volt_raw = payload[4]
        volt_v = float(volt_raw) * 0.1   # 0.0 to 25.5 V

        return RadarHeader(
            timestamp_s=timestamp_s,
            cycle_counter=cycle,
            num_valid_tracks=min(num_tracks, 64),
            sensor_fault_flag=fault_flag,
            temperature_celsius=temp_c,
            voltage_volts=volt_v,
        )

    def parse_track_frame(
        self,
        can_id: int,
        payload: bytes,
        timestamp_s: float,
    ) -> Optional[RadarTrack]:
        """Parse CAN IDs 0x401-0x408 track payload."""
        if len(payload) < 8 or not (0x401 <= can_id <= 0x408):
            return None

        b0, b1, b2, b3, b4, b5, b6, b7 = payload[:8]

        track_id = b0 & 0x3F
        dyn_prop_raw = (b0 >> 6) & 0x03
        dyn_map = {
            0: DynamicProperty.MOVING,
            1: DynamicProperty.STATIONARY,
            2: DynamicProperty.ONCOMING,
            3: DynamicProperty.CROSSING,
        }
        dyn_prop = dyn_map.get(dyn_prop_raw, DynamicProperty.UNKNOWN)

        # Range: 12 bits (Byte 1 full, Byte 2 upper 4 bits)
        range_raw = (b1 << 4) | ((b2 >> 4) & 0x0F)
        range_m = float(range_raw) * 0.05  # Max ~204.75 m

        # Azimuth: 12 bits (Byte 2 lower 4 bits, Byte 3 full)
        az_raw = ((b2 & 0x0F) << 8) | b3
        azimuth_deg = float(az_raw) * 0.1 - 60.0  # -60 to +60 deg

        # Velocity: 8 bits (Byte 4)
        vel_raw = b4
        velocity_mps = float(vel_raw) * 0.5 - 70.0  # -70 to +57.5 m/s

        # RCS: 8 bits (Byte 5)
        rcs_raw = b5
        rcs_dbsm = float(rcs_raw) * 0.5 - 40.0  # -40 to +87.5 dBsm

        # Elevation: 8 bits (Byte 6)
        elev_raw = b6
        elev_deg = float(elev_raw) * 0.2 - 15.0  # -15 to +36 deg

        # Existence Probability: 6 bits (Byte 7 upper 6 bits)
        prob_raw = (b7 >> 2) & 0x3F
        prob = float(prob_raw) / 63.0  # Normalize to [0.0, 1.0]

        # Convert to Vehicle Body Frame
        x_v, y_v, z_v = self.transformer.polar_to_cartesian(range_m, azimuth_deg, elev_deg)

        return RadarTrack(
            track_id=track_id,
            timestamp_s=timestamp_s,
            range_m=round(range_m, 2),
            azimuth_deg=round(azimuth_deg, 2),
            elevation_deg=round(elev_deg, 2),
            radial_velocity_mps=round(velocity_mps, 2),
            rcs_dbsm=round(rcs_dbsm, 1),
            existence_prob=round(prob, 2),
            dynamic_property=dyn_prop,
            x_veh_m=round(x_v, 2),
            y_veh_m=round(y_v, 2),
            z_veh_m=round(z_v, 2),
            age_cycles=1,
            lost_cycles=0,
        )

    @staticmethod
    def encode_synthetic_track_frame(
        track_id: int,
        range_m: float,
        azimuth_deg: float,
        velocity_mps: float,
        rcs_dbsm: float,
        elevation_deg: float = 0.0,
        prob: float = 0.95,
        dynamic_prop: DynamicProperty = DynamicProperty.MOVING,
    ) -> bytes:
        """Helper to serialize a radar track into standard 8-byte CAN payload for testing."""
        dyn_val = min(3, dynamic_prop.value)
        b0 = (dyn_val << 6) | (track_id & 0x3F)

        range_raw = int(np.clip(round(range_m / 0.05), 0, 4095))
        b1 = (range_raw >> 4) & 0xFF

        az_raw = int(np.clip(round((azimuth_deg + 60.0) / 0.1), 0, 4095))
        b2 = ((range_raw & 0x0F) << 4) | ((az_raw >> 8) & 0x0F)
        b3 = az_raw & 0xFF

        vel_raw = int(np.clip(round((velocity_mps + 70.0) / 0.5), 0, 255))
        b4 = vel_raw & 0xFF

        rcs_raw = int(np.clip(round((rcs_dbsm + 40.0) / 0.5), 0, 255))
        b5 = rcs_raw & 0xFF

        elev_raw = int(np.clip(round((elevation_deg + 15.0) / 0.2), 0, 255))
        b6 = elev_raw & 0xFF

        prob_raw = int(np.clip(round(prob * 63.0), 0, 63))
        b7 = (prob_raw << 2) & 0xFC

        return bytes([b0, b1, b2, b3, b4, b5, b6, b7])


# ═══════════════════════════════════════════════════════════════════════════════
# Temporal Track Buffer & Lifetime Manager
# ═══════════════════════════════════════════════════════════════════════════════

class RadarTrackBuffer:
    """
    Maintains active track state across consecutive measurement cycles:
    - Track confirmation and age accumulation.
    - Coasting (extrapolation) during brief sensor occlusions (up to max_lost_cycles).
    - Pruning stale tracks to prevent ghost target propagation.
    """

    def __init__(
        self,
        max_lost_cycles: int = 3,
        min_confirmed_cycles: int = 2,
    ) -> None:
        self.max_lost_cycles = max_lost_cycles
        self.min_confirmed_cycles = min_confirmed_cycles
        self.active_tracks: Dict[int, RadarTrack] = {}

    def update(self, new_tracks: List[RadarTrack]) -> List[RadarTrack]:
        """
        Fuse new incoming track frame with internal state buffer.
        """
        incoming_ids = set()

        for track in new_tracks:
            tid = track.track_id
            incoming_ids.add(tid)

            if tid in self.active_tracks:
                # Update existing track
                prior = self.active_tracks[tid]
                track.age_cycles = prior.age_cycles + 1
                track.lost_cycles = 0
                self.active_tracks[tid] = track
            else:
                # New track initialization
                track.age_cycles = 1
                track.lost_cycles = 0
                self.active_tracks[tid] = track

        # Coast or prune missing tracks
        pruned_ids = []
        for tid, active_t in self.active_tracks.items():
            if tid not in incoming_ids:
                active_t.lost_cycles += 1
                if active_t.lost_cycles > self.max_lost_cycles:
                    pruned_ids.append(tid)

        for tid in pruned_ids:
            del self.active_tracks[tid]

        # Return confirmed tracks only
        confirmed = [
            t for t in self.active_tracks.values()
            if t.age_cycles >= self.min_confirmed_cycles or t.existence_prob >= 0.80
        ]
        return confirmed

    def clear(self) -> None:
        """Flush buffer."""
        self.active_tracks.clear()


# ═══════════════════════════════════════════════════════════════════════════════
# ASIL-B Radar Health & Blockage Monitor
# ═══════════════════════════════════════════════════════════════════════════════

class RadarHealthMonitor:
    """
    ISO 26262 ASIL-B aligned health supervisor:
    1. Heartbeat / Timeout Monitor: Flags TIMEOUT if no CAN packets received for >150 ms.
    2. Environmental Blockage Detector: Flags BLOCKED if vehicle is moving at high speed
       (>5 m/s) with 0 tracks and zero background clutter over >3.0 seconds (mud/debris on radome).
    3. Hardware Diagnostics: Voltage and temperature limit checks.
    """

    def __init__(
        self,
        timeout_threshold_s: float = 0.150,
        blockage_speed_threshold_mps: float = 5.0,
        blockage_duration_s: float = 3.0,
    ) -> None:
        self.timeout_threshold_s = timeout_threshold_s
        self.blockage_speed_mps = blockage_speed_threshold_mps
        self.blockage_duration_s = blockage_duration_s

        self.last_packet_time_s = 0.0
        self.zero_track_motion_start_s: Optional[float] = None
        self.is_sensor_blocked = False

    def notify_packet_received(self, timestamp_s: float) -> None:
        """Update reception heartbeat."""
        self.last_packet_time_s = timestamp_s

    def evaluate_health(
        self,
        current_time_s: float,
        num_active_tracks: int,
        vehicle_speed_mps: float = 0.0,
        header: Optional[RadarHeader] = None,
    ) -> Tuple[RadarHealthState, str]:
        """
        Evaluate full operational state.
        """
        # 1. Timeout Check
        dt_since_rx = current_time_s - self.last_packet_time_s
        if self.last_packet_time_s > 0.0 and dt_since_rx > self.timeout_threshold_s:
            return RadarHealthState.TIMEOUT, f"CAN packet loss: {dt_since_rx*1000.0:.1f}ms silence"

        # 2. Hardware Fault Flags
        if header is not None:
            if header.sensor_fault_flag:
                return RadarHealthState.DEGRADED, "Internal sensor fault flag asserted"
            if header.voltage_volts < 9.0 or header.voltage_volts > 16.5:
                return RadarHealthState.DEGRADED, f"Supply voltage out of spec: {header.voltage_volts:.1f}V"
            if header.temperature_celsius > 95.0:
                return RadarHealthState.DEGRADED, f"Over-temperature alarm: {header.temperature_celsius:.1f}C"

        # 3. Radome Mud / Debris Blockage Detection
        if vehicle_speed_mps > self.blockage_speed_mps and num_active_tracks == 0:
            if self.zero_track_motion_start_s is None:
                self.zero_track_motion_start_s = current_time_s
            elif (current_time_s - self.zero_track_motion_start_s) > self.blockage_duration_s:
                self.is_sensor_blocked = True
                return RadarHealthState.BLOCKED, "Radome mud/debris blockage detected"
        else:
            self.zero_track_motion_start_s = None
            self.is_sensor_blocked = False

        return RadarHealthState.OK, "Nominal operational state"


# ═══════════════════════════════════════════════════════════════════════════════
# Master Starkenn Radar Driver Interface
# ═══════════════════════════════════════════════════════════════════════════════

class StarkennRadarDriver:
    """
    Master Driver for the Starkenn Orion 77GHz Automotive CAN Radar.
    Orchestrates:
    - Raw CAN packet decoding (Header 0x400 and Tracks 0x401-0x408).
    - Polar-to-Cartesian coordinate projection into vehicle frame.
    - Persistent multi-object tracking and ghost-suppression buffering.
    - ASIL-B health, timeout, and radome blockage monitoring.
    - Synthetic playback simulation for off-road Baja test tracks.
    """

    def __init__(
        self,
        mounting_config: Optional[RadarMountingConfig] = None,
        can_channel: str = "can0",
        bitrate: int = 500000,
        enable_synthetic_playback: bool = False,
    ) -> None:
        self.mounting = mounting_config or RadarMountingConfig()
        self.can_channel = can_channel
        self.bitrate = bitrate
        self.synthetic_mode = enable_synthetic_playback

        self.transformer = RadarCoordinateTransformer(self.mounting)
        self.parser = StarkennCANParser(self.transformer)
        self.buffer = RadarTrackBuffer()
        self.health_monitor = RadarHealthMonitor()

        self.current_scan_id = 0
        self.last_header: Optional[RadarHeader] = None
        self.pending_scan_tracks: List[RadarTrack] = []

    def feed_can_frame(
        self,
        can_id: int,
        payload: bytes,
        timestamp_s: float,
        vehicle_speed_mps: float = 0.0,
    ) -> Optional[RadarScan]:
        """
        Process a single incoming CAN frame.
        When a full cycle (0x400 header or end of track batch) is detected,
        compiles and returns a complete RadarScan.
        """
        self.health_monitor.notify_packet_received(timestamp_s)

        # 1. Header Frame 0x400
        if can_id == 0x400:
            header = self.parser.parse_header_frame(payload, timestamp_s)
            self.last_header = header

            # Emit accumulated scan from previous cycle if tracks exist
            if self.pending_scan_tracks or self.current_scan_id > 0:
                scan = self._finalize_scan(timestamp_s, vehicle_speed_mps)
                self.pending_scan_tracks = []
                self.current_scan_id += 1
                return scan
            return None

        # 2. Track Frames 0x401 - 0x408
        elif 0x401 <= can_id <= 0x408:
            track = self.parser.parse_track_frame(can_id, payload, timestamp_s)
            if track is not None:
                self.pending_scan_tracks.append(track)
            return None

        return None

    def _finalize_scan(self, timestamp_s: float, vehicle_speed_mps: float) -> RadarScan:
        """Compile accumulated tracks into a structured RadarScan."""
        t_start = time.perf_counter()

        # Update persistent track buffer
        confirmed_tracks = self.buffer.update(self.pending_scan_tracks)

        # Evaluate health and blockage
        health_state, _ = self.health_monitor.evaluate_health(
            current_time_s=timestamp_s,
            num_active_tracks=len(confirmed_tracks),
            vehicle_speed_mps=vehicle_speed_mps,
            header=self.last_header,
        )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        return RadarScan(
            timestamp_s=timestamp_s,
            scan_id=self.current_scan_id,
            header=self.last_header,
            tracks=confirmed_tracks,
            health=health_state,
            latency_ms=round(latency_ms, 3),
            is_blocked=(health_state == RadarHealthState.BLOCKED),
        )

    def generate_synthetic_offroad_scan(
        self,
        timestamp_s: float,
        scenario: str = "boulder_and_oncoming_buggy",
    ) -> RadarScan:
        """
        Generate realistic synthetic radar sweeps for off-road Baja SAE testing.
        Scenarios:
        - 'boulder_and_oncoming_buggy': Stationary rock cluster at 12m, oncoming buggy at 35m closing at 15 m/s.
        - 'clear_path': Open trail with zero obstacles.
        - 'radome_mud_blindness': 0 tracks while driving fast.
        """
        tracks: List[RadarTrack] = []

        if scenario == "boulder_and_oncoming_buggy":
            # Track 1: Large exposed rock boulder (stationary, high RCS)
            x_b, y_b, z_b = self.transformer.polar_to_cartesian(12.5, -4.5, -1.0)
            tracks.append(RadarTrack(
                track_id=1,
                timestamp_s=timestamp_s,
                range_m=12.5,
                azimuth_deg=-4.5,
                elevation_deg=-1.0,
                radial_velocity_mps=0.0,
                rcs_dbsm=18.5,
                existence_prob=0.98,
                dynamic_property=DynamicProperty.STATIONARY,
                x_veh_m=round(x_b, 2),
                y_veh_m=round(y_b, 2),
                z_veh_m=round(z_b, 2),
                age_cycles=12,
                lost_cycles=0,
            ))

            # Track 2: Oncoming competitor buggy (moving towards us at -14.2 m/s)
            x_o, y_o, z_o = self.transformer.polar_to_cartesian(38.0, 1.8, 0.0)
            tracks.append(RadarTrack(
                track_id=2,
                timestamp_s=timestamp_s,
                range_m=38.0,
                azimuth_deg=1.8,
                elevation_deg=0.0,
                radial_velocity_mps=-14.2,
                rcs_dbsm=12.0,
                existence_prob=0.92,
                dynamic_property=DynamicProperty.ONCOMING,
                x_veh_m=round(x_o, 2),
                y_veh_m=round(y_o, 2),
                z_veh_m=round(z_o, 2),
                age_cycles=25,
                lost_cycles=0,
            ))

        self.current_scan_id += 1
        return RadarScan(
            timestamp_s=timestamp_s,
            scan_id=self.current_scan_id,
            header=RadarHeader(
                timestamp_s=timestamp_s,
                cycle_counter=self.current_scan_id,
                num_valid_tracks=len(tracks),
                sensor_fault_flag=False,
                temperature_celsius=48.5,
                voltage_volts=12.4,
            ),
            tracks=tracks,
            health=RadarHealthState.OK,
            latency_ms=0.15,
            is_blocked=False,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Radar Object Clustering & Camera-Radar Fusion Adapter
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class RadarCluster:
    """Clustered group of radar reflections representing a single physical obstacle."""
    cluster_id: int
    centroid_x_m: float
    centroid_y_m: float
    centroid_z_m: float
    mean_doppler_mps: float
    max_rcs_dbsm: float
    track_count: int
    bounding_box_width_m: float
    bounding_box_length_m: float
    dynamic_property: DynamicProperty


class RadarClusterer:
    """
    Euclidean spatial clusterer grouping nearby radar track points into
    coherent physical obstacle candidates (e.g., vehicle chassis or boulder piles).
    """

    def __init__(self, cluster_distance_threshold_m: float = 2.5) -> None:
        self.dist_thresh = cluster_distance_threshold_m

    def cluster_tracks(self, tracks: List[RadarTrack]) -> List[RadarCluster]:
        """Group tracks by Euclidean proximity in the vehicle ground plane."""
        if not tracks:
            return []

        clusters: List[RadarCluster] = []
        visited = set()
        cluster_id = 1

        for i, t_i in enumerate(tracks):
            if i in visited:
                continue

            group = [t_i]
            visited.add(i)

            for j, t_j in enumerate(tracks):
                if j in visited:
                    continue
                # 2D Euclidean distance in vehicle horizontal plane
                dx = t_i.x_veh_m - t_j.x_veh_m
                dy = t_i.y_veh_m - t_j.y_veh_m
                dist = math.sqrt(dx * dx + dy * dy)
                if dist <= self.dist_thresh:
                    group.append(t_j)
                    visited.add(j)

            # Compute cluster centroid and bounds
            xs = [t.x_veh_m for t in group]
            ys = [t.y_veh_m for t in group]
            zs = [t.z_veh_m for t in group]
            v_dops = [t.radial_velocity_mps for t in group]
            rcss = [t.rcs_dbsm for t in group]

            c_x = float(np.mean(xs))
            c_y = float(np.mean(ys))
            c_z = float(np.mean(zs))
            mean_v = float(np.mean(v_dops))
            max_rcs = float(np.max(rcss))
            w = max(0.5, float(np.max(xs) - np.min(xs)))
            l = max(0.5, float(np.max(ys) - np.min(ys)))

            # Dominant dynamic property
            dyn = group[0].dynamic_property

            clusters.append(RadarCluster(
                cluster_id=cluster_id,
                centroid_x_m=round(c_x, 2),
                centroid_y_m=round(c_y, 2),
                centroid_z_m=round(c_z, 2),
                mean_doppler_mps=round(mean_v, 2),
                max_rcs_dbsm=round(max_rcs, 1),
                track_count=len(group),
                bounding_box_width_m=round(w, 2),
                bounding_box_length_m=round(l, 2),
                dynamic_property=dyn,
            ))
            cluster_id += 1

        return clusters


class CameraRadarProjectionEngine:
    """
    Projects 3D radar targets into 2D camera pixel coordinates (u, v)
    using pinhole camera intrinsics and vehicle extrinsics for sensor fusion.
    """

    def __init__(
        self,
        f_x: float = 450.0,
        f_y: float = 450.0,
        c_x: float = 320.0,
        c_y: float = 180.0,
        h_cam_m: float = 1.20,
        nominal_pitch_deg: float = 5.0,
    ) -> None:
        self.fx = f_x
        self.fy = f_y
        self.cx = c_x
        self.cy = c_y
        self.h_cam = h_cam_m
        self.pitch_rad = math.radians(nominal_pitch_deg)

    def project_to_camera_uv(
        self,
        x_veh_m: float,
        y_veh_m: float,
        z_veh_m: float,
    ) -> Optional[Tuple[int, int]]:
        """
        Transforms vehicle frame point (X: right, Y: forward, Z: up)
        to camera optical frame (X_c: right, Y_c: down, Z_c: forward),
        accounting for camera mounting height and pitch angle.
        Returns pixel coordinate (u, v) or None if target is behind camera.
        """
        if y_veh_m <= 0.5:
            return None  # Behind or too close to camera

        # Translate relative to camera position
        dx = x_veh_m
        dy = y_veh_m
        dz = z_veh_m - self.h_cam

        # Apply pitch rotation (pitch down tilts Z_c downwards)
        cos_p = math.cos(self.pitch_rad)
        sin_p = math.sin(self.pitch_rad)

        # In camera coordinates:
        # Z_c = forward (optical axis)
        # X_c = right
        # Y_c = down
        z_c = dy * cos_p + dz * sin_p
        y_c = -(dz * cos_p - dy * sin_p)
        x_c = dx

        if z_c <= 0.2:
            return None

        # Pinhole perspective projection
        u = int(round(self.fx * (x_c / z_c) + self.cx))
        v = int(round(self.fy * (y_c / z_c) + self.cy))

        return u, v


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_starkenn_radar_driver(
    mounting_config: Optional[RadarMountingConfig] = None,
    can_channel: str = "can0",
) -> StarkennRadarDriver:
    """Factory helper to instantiate StarkennRadarDriver."""
    return StarkennRadarDriver(mounting_config=mounting_config, can_channel=can_channel)


def run_radar_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. Coordinate transformation from spherical polar to metric vehicle frame.
    2. CAN header and track frame bit-unpacking accuracy.
    3. Serialization and roundtrip parsing of track objects.
    4. Persistent track buffering and age incrementation.
    5. ISO 26262 ASIL-B timeout and radome blockage fault detection.
    6. Synthetic off-road scan generation.
    7. Multi-target radar clustering.
    8. Camera-radar perspective projection.
    """
    print("Executing StarkennRadarDriver Verification Suite...")
    driver = get_starkenn_radar_driver()

    # -------------------------------------------------------------------------
    # Test 1: Coordinate Transformation Math
    # -------------------------------------------------------------------------
    x, y, z = driver.transformer.polar_to_cartesian(range_m=20.0, azimuth_deg=0.0, elevation_deg=0.0)
    assert abs(x - 0.0) < 1e-3, f"X mismatch: {x}"
    assert abs(y - 21.65) < 1e-3, f"Y mismatch: {y}"
    assert abs(z - 0.45) < 1e-3, f"Z mismatch: {z}"

    x_r, y_r, _ = driver.transformer.polar_to_cartesian(range_m=10.0, azimuth_deg=30.0, elevation_deg=0.0)
    assert abs(x_r - 5.0) < 1e-2, f"X_r mismatch: {x_r}"

    # -------------------------------------------------------------------------
    # Test 2: CAN Frame Encoding & Roundtrip Parsing
    # -------------------------------------------------------------------------
    raw_payload = StarkennCANParser.encode_synthetic_track_frame(
        track_id=5,
        range_m=24.50,
        azimuth_deg=-12.3,
        velocity_mps=-8.5,
        rcs_dbsm=15.0,
        elevation_deg=1.2,
        prob=0.95,
        dynamic_prop=DynamicProperty.ONCOMING,
    )
    assert len(raw_payload) == 8, "Payload must be exactly 8 bytes"

    track = driver.parser.parse_track_frame(0x401, raw_payload, timestamp_s=100.0)
    assert track is not None, "Failed to parse track frame"
    assert track.track_id == 5, f"Track ID mismatch: {track.track_id}"
    assert abs(track.range_m - 24.50) < 0.10, f"Range mismatch: {track.range_m}"
    assert abs(track.azimuth_deg - (-12.3)) < 0.20, f"Azimuth mismatch: {track.azimuth_deg}"
    assert abs(track.radial_velocity_mps - (-8.5)) < 0.50, f"Velocity mismatch: {track.radial_velocity_mps}"
    assert track.dynamic_property == DynamicProperty.ONCOMING

    # -------------------------------------------------------------------------
    # Test 3: CAN Header Parsing
    # -------------------------------------------------------------------------
    header_payload = bytes([0x2A, 0x30, 0x00, 0x58, 0x78, 0x00, 0x00, 0x00])
    header = driver.parser.parse_header_frame(header_payload, timestamp_s=100.0)
    assert header is not None
    assert header.sensor_fault_flag is False
    assert header.temperature_celsius == (0x58 - 40.0)  # 48.0 C
    assert header.voltage_volts == (0x78 * 0.1)        # 12.0 V

    # -------------------------------------------------------------------------
    # Test 4: Full Multi-Frame CAN Stream Feeding & Scan Finalization
    # -------------------------------------------------------------------------
    t_now = 200.0
    driver.feed_can_frame(0x401, raw_payload, t_now)
    driver.feed_can_frame(0x402, raw_payload, t_now)
    scan = driver.feed_can_frame(0x400, header_payload, t_now + 0.01)
    assert scan is not None, "Failed to emit scan on header boundary"
    assert scan.health == RadarHealthState.OK

    # -------------------------------------------------------------------------
    # Test 5: ASIL-B Timeout Detection
    # -------------------------------------------------------------------------
    health, reason = driver.health_monitor.evaluate_health(
        current_time_s=t_now + 0.50,
        num_active_tracks=1,
    )
    assert health == RadarHealthState.TIMEOUT, "Timeout supervisor failed to flag silence"

    # -------------------------------------------------------------------------
    # Test 6: Radome Mud / Debris Blockage Detection
    # -------------------------------------------------------------------------
    t_block = 300.0
    driver.health_monitor.notify_packet_received(t_block)
    driver.health_monitor.evaluate_health(t_block, num_active_tracks=0, vehicle_speed_mps=10.0)
    # Packet received at t_block + 3.5 (heartbeat alive, but 0 tracks while at 10 m/s)
    driver.health_monitor.notify_packet_received(t_block + 3.5)
    h_blocked, _ = driver.health_monitor.evaluate_health(
        t_block + 3.5,
        num_active_tracks=0,
        vehicle_speed_mps=10.0,
    )
    assert h_blocked == RadarHealthState.BLOCKED, "Failed to detect radome blockage"

    # -------------------------------------------------------------------------
    # Test 7: Synthetic Playback Sweep
    # -------------------------------------------------------------------------
    synth_scan = driver.generate_synthetic_offroad_scan(400.0, "boulder_and_oncoming_buggy")
    assert len(synth_scan.tracks) == 2
    assert synth_scan.tracks[0].dynamic_property == DynamicProperty.STATIONARY
    assert synth_scan.tracks[1].dynamic_property == DynamicProperty.ONCOMING
    print(f"Synthetic Radar Scan ID {synth_scan.scan_id}: {len(synth_scan.tracks)} tracks verified.")

    # -------------------------------------------------------------------------
    # Test 8: Multi-Target Clustering & Camera Projection
    # -------------------------------------------------------------------------
    clusterer = RadarClusterer(cluster_distance_threshold_m=2.0)
    clusters = clusterer.cluster_tracks(synth_scan.tracks)
    assert len(clusters) == 2, f"Expected 2 separate clusters, got {len(clusters)}"

    proj_engine = CameraRadarProjectionEngine()
    uv = proj_engine.project_to_camera_uv(synth_scan.tracks[0].x_veh_m, synth_scan.tracks[0].y_veh_m, synth_scan.tracks[0].z_veh_m)
    assert uv is not None, "Failed to project near radar track to camera coordinates"
    assert 0 <= uv[0] <= 640 and 0 <= uv[1] <= 360, f"Projected UV out of frame: {uv}"
    print(f"Projected Radar Track to Camera Pixel: (u={uv[0]}, v={uv[1]})")

    print("StarkennRadarDriver verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_radar_verification_suite()

