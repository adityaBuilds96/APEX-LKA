"""
APEX-LKA v4.0 — Starkenn Orion 77GHz CAN Driver
=================================================
Decodes Starkenn radar CAN frames (0x400 - 0x408) with:
  - Thread-safe zero-copy ring buffer parsing
  - CRC-8 validation and rolling sequence tracking via can_integrity
  - RCS threshold filter (>-15 dBm flag) and confidence score filter (>60%)
  - RadarTrack object output compatible with TrackRadarFusion Engine

Starkenn Orion 77GHz radar specifications:
  - Frequency: 76-77 GHz FMCW
  - Range: 0.5 - 160 m (far mode)
  - Velocity resolution: 0.1 m/s
  - Angular resolution: ±0.1°
  - Max simultaneous tracks: 8
  - CAN bitrate: 500 kbps
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.hardware.can_integrity import (
    CANIntegrityEngine,
    APEX_MESSAGE_SPECS,
    SequenceStatus,
    CRC8Engine,
)

__all__ = [
    "RadarTrack",
    "RadarScanFrame",
    "StarkennCANDriver",
    "TrackStatus",
    "RadarDriverConfig",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RADAR_HEADER_ID = 0x400
RADAR_TRACK_BASE_ID = 0x401
RADAR_TRACK_COUNT = 8
RCS_THRESHOLD_DBM = -15.0        # Minimum RCS in dBm to pass filter
CONFIDENCE_THRESHOLD_PCT = 60.0  # Minimum confidence % to pass filter
RING_BUFFER_SIZE = 256           # Zero-copy ring buffer capacity
MAX_TRACK_AGE_S = 0.200          # 200 ms max track age before invalidation
MIN_RANGE_X_M = -50.0            # Physical range limits (mirror DBC)
MAX_RANGE_X_M = 154.75
MIN_RANGE_Y_M = -25.0
MAX_RANGE_Y_M = 179.75
RADAR_STALE_S = 0.150            # Radar stale threshold (3 frames @ 20 Hz)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class TrackStatus(IntEnum):
    INVALID      = 0
    NEW          = 1
    CONFIRMED    = 2
    COASTING     = 3
    LOST         = 4
    FILTERED_RCS = 5   # Filtered out: RCS below threshold
    FILTERED_CNF = 6   # Filtered out: Confidence below threshold


class RadarMode(IntEnum):
    STANDBY     = 0
    NEAR        = 1
    FAR         = 2
    COMBINED    = 3
    CALIBRATION = 4


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class RadarTrack:
    """
    Decoded and validated radar track object.
    Compatible with TrackRadarFusion Engine input format.
    """
    track_idx: int = 0           # Frame slot index 1-8
    track_id: int = 0            # Persistent track ID 0-15
    range_x_m: float = 0.0      # Longitudinal range (positive = forward)
    range_y_m: float = 0.0      # Lateral range (positive = left)
    rel_vel_x_mps: float = 0.0  # Relative longitudinal velocity
    rel_vel_y_mps: float = 0.0  # Relative lateral velocity
    rcs_above_threshold: bool = False    # RCS > -15 dBm
    confidence_high: bool = False        # Confidence > 60%
    track_status: TrackStatus = TrackStatus.INVALID
    timestamp: float = 0.0
    scan_frame_id: int = 0       # Radar scan sequence ID
    crc_ok: bool = True
    seq_status: SequenceStatus = SequenceStatus.VALID

    @property
    def is_valid_for_fusion(self) -> bool:
        """True if track passes all filters and is suitable for fusion engine."""
        return (
            self.rcs_above_threshold
            and self.confidence_high
            and self.track_status not in (TrackStatus.INVALID, TrackStatus.LOST,
                                          TrackStatus.FILTERED_RCS, TrackStatus.FILTERED_CNF)
            and self.crc_ok
            and MIN_RANGE_X_M <= self.range_x_m <= MAX_RANGE_X_M
            and MIN_RANGE_Y_M <= self.range_y_m <= MAX_RANGE_Y_M
        )

    @property
    def range_m(self) -> float:
        """Euclidean range to target."""
        return math.sqrt(self.range_x_m ** 2 + self.range_y_m ** 2)

    @property
    def azimuth_deg(self) -> float:
        """Azimuth angle from vehicle longitudinal axis in degrees."""
        return math.degrees(math.atan2(self.range_y_m, self.range_x_m))

    @property
    def closing_speed_mps(self) -> float:
        """Positive = approaching. Relative longitudinal velocity toward ego."""
        return -self.rel_vel_x_mps


@dataclass
class RadarScanFrame:
    """Complete radar scan: header + all tracks for one scan cycle."""
    target_count: int = 0
    bus_load_pct: float = 0.0
    status_flags: int = 0
    radar_mode: RadarMode = RadarMode.COMBINED
    sensor_temp_c: float = 25.0
    frame_id: int = 0
    crc_ok: bool = True
    counter: int = 0
    seq_status: SequenceStatus = SequenceStatus.VALID
    timestamp: float = 0.0
    tracks: List[RadarTrack] = field(default_factory=list)

    @property
    def valid_tracks(self) -> List[RadarTrack]:
        """Return only tracks that pass all quality filters."""
        return [t for t in self.tracks if t.is_valid_for_fusion]

    @property
    def nearest_threat(self) -> Optional[RadarTrack]:
        """Return the nearest valid forward track (min positive range_x)."""
        forward = [t for t in self.valid_tracks if t.range_x_m > 0]
        return min(forward, key=lambda t: t.range_x_m) if forward else None


@dataclass
class RadarDriverConfig:
    """Configuration for StarkennCANDriver."""
    rcs_threshold_dbm: float = RCS_THRESHOLD_DBM
    confidence_threshold_pct: float = CONFIDENCE_THRESHOLD_PCT
    ring_buffer_size: int = RING_BUFFER_SIZE
    max_track_age_s: float = MAX_TRACK_AGE_S
    stale_threshold_s: float = RADAR_STALE_S
    enable_rcs_filter: bool = True
    enable_confidence_filter: bool = True
    enable_range_filter: bool = True
    min_range_x_m: float = 0.5      # Don't process tracks closer than 0.5 m
    max_range_x_m: float = 150.0    # Far mode effective range


# ---------------------------------------------------------------------------
# Zero-Copy Ring Buffer
# ---------------------------------------------------------------------------

class ZeroCopyRingBuffer:
    """
    Thread-safe fixed-size ring buffer for incoming CAN frames.
    Uses preallocated bytearray slots for zero-copy semantics.
    """

    def __init__(self, capacity: int = RING_BUFFER_SIZE) -> None:
        self._capacity = capacity
        self._slots: List[Tuple[int, bytearray, float]] = [
            (0, bytearray(8), 0.0) for _ in range(capacity)
        ]
        self._head = 0   # Next write position
        self._tail = 0   # Next read position
        self._count = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)

    def push(self, can_id: int, payload: bytes, timestamp: float) -> bool:
        """
        Push a frame into the ring buffer. Returns False if buffer is full.
        Zero-copy: writes into pre-allocated bytearray slot.
        """
        with self._lock:
            if self._count >= self._capacity:
                # Overwrite oldest (drop oldest = best real-time behavior)
                self._tail = (self._tail + 1) % self._capacity
                self._count -= 1
                log.debug("Ring buffer overflow: oldest frame dropped.")

            slot_id, slot_buf, _ = self._slots[self._head]
            slot_buf[:len(payload)] = payload
            self._slots[self._head] = (can_id, slot_buf, timestamp)
            self._head = (self._head + 1) % self._capacity
            self._count += 1
            self._not_empty.notify_all()
            return True

    def pop(self, timeout: float = 0.01) -> Optional[Tuple[int, bytes, float]]:
        """
        Pop a frame from the ring buffer.
        Returns (can_id, payload_bytes, timestamp) or None on timeout.
        """
        with self._not_empty:
            if self._count == 0:
                self._not_empty.wait(timeout=timeout)
            if self._count == 0:
                return None

            can_id, slot_buf, timestamp = self._slots[self._tail]
            payload = bytes(slot_buf)  # Copy out for consumer
            self._tail = (self._tail + 1) % self._capacity
            self._count -= 1
            return can_id, payload, timestamp

    @property
    def size(self) -> int:
        with self._lock:
            return self._count

    @property
    def is_empty(self) -> bool:
        with self._lock:
            return self._count == 0


# ---------------------------------------------------------------------------
# Starkenn CAN Driver
# ---------------------------------------------------------------------------

class StarkennCANDriver:
    """
    Starkenn Orion 77GHz radar CAN decoder.

    Decodes frames 0x400 (header) and 0x401-0x408 (tracks).
    Integrates CRC-8 validation and sequence tracking from CANIntegrityEngine.
    Applies RCS and confidence filters before publishing RadarTrack objects.

    Thread-safe: frame ingestion and scan consumption can run from separate threads.
    """

    def __init__(
        self,
        integrity_engine: Optional[CANIntegrityEngine] = None,
        config: Optional[RadarDriverConfig] = None,
    ) -> None:
        self._engine = integrity_engine or CANIntegrityEngine()
        self._cfg = config or RadarDriverConfig()
        self._ring = ZeroCopyRingBuffer(self._cfg.ring_buffer_size)

        # Current scan assembly state
        self._current_scan: Optional[RadarScanFrame] = None
        self._scan_lock = threading.Lock()

        # Completed scan history (last N scans)
        self._scan_history: deque[RadarScanFrame] = deque(maxlen=50)
        self._scan_callbacks: List[Callable[[RadarScanFrame], None]] = []

        # Per-track state for aging
        self._track_last_seen: Dict[int, float] = {}

        # Statistics
        self._stats = {
            "total_header_frames": 0,
            "total_track_frames": 0,
            "crc_failures": 0,
            "seq_drops": 0,
            "filtered_rcs": 0,
            "filtered_confidence": 0,
            "completed_scans": 0,
        }
        self._stats_lock = threading.Lock()

        # Background parser thread
        self._running = False
        self._parser_thread: Optional[threading.Thread] = None

        # Subscribe to radar CAN IDs on the integrity engine
        self._engine.subscribe(RADAR_HEADER_ID, self._on_header_frame)
        for track_idx in range(1, RADAR_TRACK_COUNT + 1):
            self._engine.subscribe(RADAR_TRACK_BASE_ID + track_idx - 1, self._on_track_frame)

        log.info(
            "StarkennCANDriver initialized. RCS threshold=%.1f dBm, "
            "Confidence threshold=%.1f%%, RingBuffer=%d slots.",
            self._cfg.rcs_threshold_dbm,
            self._cfg.confidence_threshold_pct,
            self._cfg.ring_buffer_size,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start background frame parser thread."""
        if self._running:
            return
        self._running = True
        self._parser_thread = threading.Thread(
            target=self._parser_loop,
            name="StarkennCANDriver",
            daemon=True,
        )
        self._parser_thread.start()
        log.info("StarkennCANDriver parser thread started.")

    def stop(self) -> None:
        """Stop background parser thread gracefully."""
        self._running = False
        if self._parser_thread and self._parser_thread.is_alive():
            self._parser_thread.join(timeout=2.0)
        log.info("StarkennCANDriver stopped.")

    def ingest_frame(self, can_id: int, payload: bytes, timestamp: Optional[float] = None) -> None:
        """
        Ingest a raw CAN frame from the bus interface.
        Thread-safe. Returns immediately (non-blocking).
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        if can_id == RADAR_HEADER_ID or RADAR_TRACK_BASE_ID <= can_id <= RADAR_TRACK_BASE_ID + 7:
            self._ring.push(can_id, payload, ts)

    def get_latest_scan(self) -> Optional[RadarScanFrame]:
        """Return the most recently completed full scan frame."""
        with self._scan_lock:
            return self._scan_history[-1] if self._scan_history else None

    def get_latest_valid_tracks(self) -> List[RadarTrack]:
        """Return valid, filtered tracks from the most recent scan."""
        scan = self.get_latest_scan()
        return scan.valid_tracks if scan else []

    def register_scan_callback(self, callback: Callable[[RadarScanFrame], None]) -> None:
        """Register a callback called on each completed scan assembly."""
        self._scan_callbacks.append(callback)

    def get_statistics(self) -> Dict[str, Any]:
        """Return driver statistics snapshot."""
        with self._stats_lock:
            return dict(self._stats)

    def is_radar_healthy(self) -> bool:
        """Check if radar data is fresh and not stale."""
        age = self._engine.line.get_age(RADAR_HEADER_ID)
        return age < self._cfg.stale_threshold_s

    # -----------------------------------------------------------------------
    # Internal: CAN callbacks (registered with integrity engine)
    # -----------------------------------------------------------------------

    def _on_header_frame(
        self,
        can_id: int,
        values: Dict[str, float],
        crc_ok: bool,
        seq_status: SequenceStatus,
        timestamp: float,
    ) -> None:
        """Called by integrity engine on header frame decode."""
        with self._stats_lock:
            self._stats["total_header_frames"] += 1
            if not crc_ok:
                self._stats["crc_failures"] += 1
            if seq_status == SequenceStatus.FRAME_DROP:
                self._stats["seq_drops"] += 1

        scan = RadarScanFrame(
            target_count=int(values.get("TargetCount", 0)),
            bus_load_pct=values.get("BusLoad", 0.0),
            status_flags=int(values.get("StatusFlags", 0)),
            radar_mode=RadarMode(int(values.get("RadarMode", 3))),
            sensor_temp_c=values.get("SensorTemp", 25.0),
            frame_id=int(values.get("FrameID", 0)),
            crc_ok=crc_ok,
            counter=int(values.get("Counter", 0)),
            seq_status=seq_status,
            timestamp=timestamp,
            tracks=[],
        )

        with self._scan_lock:
            self._current_scan = scan

    def _on_track_frame(
        self,
        can_id: int,
        values: Dict[str, float],
        crc_ok: bool,
        seq_status: SequenceStatus,
        timestamp: float,
    ) -> None:
        """Called by integrity engine on track frame decode."""
        track_idx = can_id - RADAR_TRACK_BASE_ID + 1  # 1-8

        with self._stats_lock:
            self._stats["total_track_frames"] += 1
            if not crc_ok:
                self._stats["crc_failures"] += 1

        rcs_flag = bool(values.get("RCS_Flag", 0))
        conf_flag = bool(values.get("ConfFlag", 0))

        # Apply RCS filter
        if self._cfg.enable_rcs_filter and not rcs_flag:
            with self._stats_lock:
                self._stats["filtered_rcs"] += 1
            track_status = TrackStatus.FILTERED_RCS
        # Apply confidence filter
        elif self._cfg.enable_confidence_filter and not conf_flag:
            with self._stats_lock:
                self._stats["filtered_confidence"] += 1
            track_status = TrackStatus.FILTERED_CNF
        else:
            track_status = TrackStatus.CONFIRMED

        range_x = values.get("RangeX", 0.0)
        range_y = values.get("RangeY", 0.0)

        # Apply range filter
        if self._cfg.enable_range_filter:
            if range_x < self._cfg.min_range_x_m or range_x > self._cfg.max_range_x_m:
                track_status = TrackStatus.FILTERED_RCS  # Reuse invalid tag

        track_id = int(values.get("TrackID", 0))
        track = RadarTrack(
            track_idx=track_idx,
            track_id=track_id,
            range_x_m=range_x,
            range_y_m=range_y,
            rel_vel_x_mps=values.get("RelVelX", 0.0),
            rel_vel_y_mps=values.get("RelVelY", 0.0),
            rcs_above_threshold=rcs_flag,
            confidence_high=conf_flag,
            track_status=track_status,
            timestamp=timestamp,
            crc_ok=crc_ok,
            seq_status=seq_status,
        )

        # Update track age
        self._track_last_seen[track_id] = timestamp

        with self._scan_lock:
            if self._current_scan is not None:
                track.scan_frame_id = self._current_scan.frame_id
                self._current_scan.tracks.append(track)

                # If we have all expected tracks, finalize the scan
                expected = self._current_scan.target_count
                if len(self._current_scan.tracks) >= max(1, expected):
                    self._finalize_scan()

    def _finalize_scan(self) -> None:
        """Finalize current scan and push to history. Called under scan_lock."""
        scan = self._current_scan
        if scan is None:
            return

        self._scan_history.append(scan)
        self._current_scan = None

        with self._stats_lock:
            self._stats["completed_scans"] += 1

        valid_count = len(scan.valid_tracks)
        log.debug(
            "Scan complete: frame_id=%d, %d/%d valid tracks, mode=%s",
            scan.frame_id, valid_count, len(scan.tracks), scan.radar_mode.name,
        )

        # Fire callbacks (outside lock to prevent deadlock)
        for cb in self._scan_callbacks:
            try:
                cb(scan)
            except Exception as exc:
                log.error("Scan callback error: %s", exc)

    # -----------------------------------------------------------------------
    # Background parser loop
    # -----------------------------------------------------------------------

    def _parser_loop(self) -> None:
        """
        Background thread: drain ring buffer and process frames via integrity engine.
        Runs until stop() is called.
        """
        log.debug("StarkennCANDriver parser loop started.")
        while self._running:
            frame = self._ring.pop(timeout=0.005)
            if frame is None:
                continue
            can_id, payload, timestamp = frame
            try:
                self._engine.process_frame(can_id, payload, timestamp=timestamp)
            except Exception as exc:
                log.error("Frame processing error CAN 0x%03X: %s", can_id, exc)

        log.debug("StarkennCANDriver parser loop exited.")

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Return comprehensive driver diagnostic report."""
        stats = self.get_statistics()
        latest = self.get_latest_scan()
        return {
            "driver": "StarkennCANDriver",
            "healthy": self.is_radar_healthy(),
            "ring_buffer_utilization": self._ring.size / self._cfg.ring_buffer_size,
            "latest_scan": {
                "frame_id": latest.frame_id if latest else None,
                "target_count": latest.target_count if latest else 0,
                "valid_tracks": len(latest.valid_tracks) if latest else 0,
                "age_s": round(time.monotonic() - latest.timestamp, 3) if latest else None,
            },
            "statistics": stats,
            "config": {
                "rcs_threshold_dbm": self._cfg.rcs_threshold_dbm,
                "confidence_threshold_pct": self._cfg.confidence_threshold_pct,
            },
        }

    def decode_raw_track(self, can_id: int, payload: bytes, timestamp: float = 0.0) -> Optional[RadarTrack]:
        """
        Directly decode a single raw track frame (utility for testing).
        Does not use ring buffer; returns decoded RadarTrack immediately.
        """
        if not (RADAR_TRACK_BASE_ID <= can_id <= RADAR_TRACK_BASE_ID + 7):
            return None
        spec = APEX_MESSAGE_SPECS.get(can_id)
        if spec is None:
            return None

        crc_eng = CRC8Engine()
        crc_ok = crc_eng.verify_frame(payload)

        from src.hardware.can_integrity import CANPayloadCodec, _extract_bits_le
        codec = CANPayloadCodec(crc_eng)
        values, _, _ = codec.decode(spec, payload, verify_crc=False)

        track_idx = can_id - RADAR_TRACK_BASE_ID + 1
        rcs_flag = bool(values.get("RCS_Flag", 0))
        conf_flag = bool(values.get("ConfFlag", 0))

        status = TrackStatus.CONFIRMED
        if not rcs_flag:
            status = TrackStatus.FILTERED_RCS
        elif not conf_flag:
            status = TrackStatus.FILTERED_CNF

        return RadarTrack(
            track_idx=track_idx,
            track_id=int(values.get("TrackID", 0)),
            range_x_m=values.get("RangeX", 0.0),
            range_y_m=values.get("RangeY", 0.0),
            rel_vel_x_mps=values.get("RelVelX", 0.0),
            rel_vel_y_mps=values.get("RelVelY", 0.0),
            rcs_above_threshold=rcs_flag,
            confidence_high=conf_flag,
            track_status=status,
            timestamp=timestamp or time.monotonic(),
            crc_ok=crc_ok,
        )

    def get_track_ttc(self, track_id: int, ego_vel_x_mps: float = 0.0) -> Optional[float]:
        """
        Calculate Time-To-Collision (TTC) in seconds for a specific radar track.
        
        Args:
            track_id: Integer identifier of the track.
            ego_vel_x_mps: Host vehicle forward velocity (m/s).
            
        Returns:
            TTC in seconds if approaching, None if diverging or stationary.
        """
        with self._lock:
            track = self._tracks.get(track_id)
            if track is None or not track.is_valid:
                return None
            
            # Relative closing speed: negative rel_vel_x means target is approaching host
            closing_speed = -(track.rel_vel_x_mps)
            if closing_speed <= 0.1:  # Target is opening distance or static relative
                return None
            
            if track.range_x_m <= 0.0:
                return 0.0
                
            ttc = track.range_x_m / closing_speed
            return max(0.0, min(ttc, 99.9))

    def filter_tracks_by_spatial_roi(
        self,
        min_x_m: float = 0.5,
        max_x_m: float = 120.0,
        min_y_m: float = -4.5,
        max_y_m: float = 4.5,
    ) -> List[RadarTrack]:
        """
        Extract active confirmed tracks within an automotive region of interest (ROI).
        Typically set to the current lane corridor and immediate adjacent lanes.
        """
        active_tracks = self.get_active_tracks()
        roi_tracks = []
        for t in active_tracks:
            if min_x_m <= t.range_x_m <= max_x_m and min_y_m <= t.range_y_m <= max_y_m:
                roi_tracks.append(t)
        return roi_tracks

    def get_cluster_centroid(self) -> Optional[Tuple[float, float]]:
        """
        Compute weighted centroid (x_m, y_m) across all active confirmed radar tracks.
        Returns None if no confirmed tracks exist in the current scan.
        """
        active = self.get_active_tracks()
        if not active:
            return None
        
        sum_x = sum(t.range_x_m for t in active)
        sum_y = sum(t.range_y_m for t in active)
        n = len(active)
        return (sum_x / n, sum_y / n)

    def export_diagnostics(self) -> Dict[str, Any]:
        """
        Compile comprehensive real-time driver diagnostics and ring buffer telemetry.
        
        Returns:
            Dictionary containing packet counts, error metrics, dropped frames,
            buffer fill levels, and per-track health summary.
        """
        with self._lock:
            total_rx = self._total_frames_received
            crc_errs = self._crc_errors
            seq_errs = self._sequence_errors
            hdr_cnt = self._header_count
            act_cnt = len(self.get_active_tracks())
            
            drop_rate = 0.0
            if total_rx > 0:
                drop_rate = (crc_errs + seq_errs) / total_rx

            return {
                "total_frames_received": total_rx,
                "crc_errors": crc_errs,
                "sequence_errors": seq_errs,
                "error_rate_pct": round(drop_rate * 100.0, 2),
                "header_frames_count": hdr_cnt,
                "active_confirmed_tracks": act_cnt,
                "ring_buffer_capacity": RING_BUFFER_SIZE,
                "ring_buffer_current_fill": len(self._ring_buffer),
                "last_header_age_s": round(time.monotonic() - self._last_header_time, 3)
                if self._last_header_time > 0 else None,
                "rcs_filter_threshold_dbm": RCS_THRESHOLD_DBM,
                "confidence_filter_threshold_pct": CONFIDENCE_THRESHOLD_PCT,
            }

    def reset_driver(self) -> None:
        """
        Reset internal tracking tables, sequence counters, and ring buffer.
        Used on CAN bus reconnect or hardware power-cycle.
        """
        with self._lock:
            self._ring_buffer.clear()
            self._tracks.clear()
            self._header_info = None
            self._total_frames_received = 0
            self._crc_errors = 0
            self._sequence_errors = 0
            self._header_count = 0
            self._last_header_time = 0.0
            self._can_engine.reset()
            log.info("Starkenn CAN Driver reset completed.")
