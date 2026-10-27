"""
APEX-LKA v4.0 — Xsens MTi-680 GNSS/INS CAN Driver
====================================================
Decodes MTi-680 CAN telemetry (0x320 - 0x321) with:
  - 100 Hz Euler angles (Roll, Pitch, Yaw) at 0.01° resolution
  - 100 Hz velocity vector (East, North, Down) from RTK GNSS
  - RTK fix status bitfield decoding (NO_FIX → RTK_FIXED → PPP)
  - Pitch validation for real-time horizon calibration feed to DepthEngineV2
  - Dead-reckoning fallback with elevated variance estimation
  - GNSS lock loss recovery with dynamic status flag management

Xsens MTi-680 Specifications:
  - GNSS: Dual-antenna RTK, u-blox F9P chipset
  - IMU: MEMS 3-axis accel + gyro + magnetometer
  - Heading accuracy: 0.2° RMS (RTK fixed)
  - Position accuracy: 2 cm (RTK fixed)
  - Output rate: up to 400 Hz (IMU), 100 Hz (GNSS)
  - CAN interface: 500 kbps
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.hardware.can_integrity import (
    CANIntegrityEngine,
    APEX_MESSAGE_SPECS,
    SequenceStatus,
    CRC8Engine,
    CANPayloadCodec,
)

__all__ = [
    "MTi680State",
    "GnssFixType",
    "InsStatus",
    "DeadReckoningState",
    "HorizonCalibration",
    "MTi680CANDriver",
    "MTi680Config",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ORIENT_MSG_ID = 0x320
VELOCITY_MSG_ID = 0x321
PITCH_VALID_RANGE_DEG = (-15.0, 15.0)    # Physical pitch limits for calibration
ROLL_VALID_RANGE_DEG = (-30.0, 30.0)
YAW_VALID_RANGE_DEG = (0.0, 360.0)
GNSS_STALE_S = 0.150                      # 150 ms stale threshold (15 frames @ 100 Hz)
DEAD_RECK_VAR_MULTIPLIER = 10.0           # Variance inflation factor during dead-reckoning
MAX_VELOCITY_MPS = 50.0                   # Maximum plausible vehicle speed
RTK_FIXED_ACCURACY_M = 0.02              # RTK fixed position accuracy
RTK_FLOAT_ACCURACY_M = 0.20             # RTK float position accuracy
SINGLE_POINT_ACCURACY_M = 2.0           # Single-point fix accuracy
PITCH_HORIZON_SMOOTHING_ALPHA = 0.05    # EMA smoothing for horizon calibration
MIN_SAMPLES_FOR_HORIZON = 50            # Minimum samples before horizon is valid


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GnssFixType(IntEnum):
    NO_FIX    = 0
    SINGLE    = 1
    SBAS      = 2
    DGPS      = 3
    RTK_FLOAT = 4
    RTK_FIXED = 5
    PPP       = 6

    @property
    def accuracy_m(self) -> float:
        _acc = {
            0: float("inf"), 1: SINGLE_POINT_ACCURACY_M,
            2: 1.0, 3: 0.5, 4: RTK_FLOAT_ACCURACY_M,
            5: RTK_FIXED_ACCURACY_M, 6: 0.01,
        }
        return _acc.get(self.value, float("inf"))

    @property
    def is_rtk(self) -> bool:
        return self in (GnssFixType.RTK_FLOAT, GnssFixType.RTK_FIXED)

    @property
    def is_fixed(self) -> bool:
        return self == GnssFixType.RTK_FIXED


class InsStatus(IntEnum):
    INIT        = 0
    COARSE_ALIGN = 1
    FINE_ALIGN  = 2
    RTK_FLOAT   = 3
    RTK_FIXED   = 4
    DEAD_RECK   = 5

    @property
    def is_navigation_valid(self) -> bool:
        return self in (InsStatus.FINE_ALIGN, InsStatus.RTK_FLOAT,
                        InsStatus.RTK_FIXED, InsStatus.DEAD_RECK)

    @property
    def variance_scale(self) -> float:
        """Variance inflation factor for uncertainty propagation."""
        _scale = {
            0: 1e6, 1: 1000.0, 2: 10.0,
            3: 2.0, 4: 1.0, 5: DEAD_RECK_VAR_MULTIPLIER,
        }
        return _scale.get(self.value, 1000.0)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class MTi680State:
    """Complete MTi-680 navigation state snapshot."""
    # Orientation (Euler angles)
    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0
    orient_valid: bool = False
    orient_timestamp: float = 0.0
    orient_crc_ok: bool = True
    orient_seq_status: SequenceStatus = SequenceStatus.VALID

    # Velocity (ECEF components)
    vel_east_mps: float = 0.0
    vel_north_mps: float = 0.0
    vel_down_mps: float = 0.0
    vel_valid: bool = False
    vel_timestamp: float = 0.0
    vel_crc_ok: bool = True
    vel_seq_status: SequenceStatus = SequenceStatus.VALID

    # GNSS status
    gnss_fix_type: GnssFixType = GnssFixType.NO_FIX
    ins_status: InsStatus = InsStatus.INIT

    # Dead reckoning
    is_dead_reckoning: bool = False
    dr_duration_s: float = 0.0
    dr_start_time: float = 0.0

    # Derived quantities
    speed_mps: float = 0.0
    heading_deg: float = 0.0   # 2D horizontal heading

    @property
    def is_navigation_valid(self) -> bool:
        return self.ins_status.is_navigation_valid and self.orient_valid

    @property
    def position_variance_m2(self) -> float:
        acc = self.gnss_fix_type.accuracy_m
        return acc ** 2 * self.ins_status.variance_scale

    @property
    def yaw_rad(self) -> float:
        return math.radians(self.yaw_deg)

    @property
    def pitch_rad(self) -> float:
        return math.radians(self.pitch_deg)

    @property
    def roll_rad(self) -> float:
        return math.radians(self.roll_deg)


@dataclass
class DeadReckoningState:
    """State maintained during GNSS-denied dead reckoning."""
    is_active: bool = False
    start_time: float = 0.0
    last_valid_time: float = 0.0
    last_valid_fix: GnssFixType = GnssFixType.NO_FIX
    propagated_heading_deg: float = 0.0
    heading_drift_rate_dps: float = 0.05  # Estimated gyro drift
    position_uncertainty_m: float = 0.0   # Grows with time

    @property
    def duration_s(self) -> float:
        if not self.is_active:
            return 0.0
        return time.monotonic() - self.start_time

    def propagate(self, dt_s: float, yaw_rate_dps: float) -> None:
        """Propagate heading estimate during dead reckoning."""
        self.propagated_heading_deg += yaw_rate_dps * dt_s
        self.propagated_heading_deg %= 360.0
        # Uncertainty grows proportionally to time
        self.position_uncertainty_m += 0.1 * dt_s  # 10 cm/s drift assumption

    @property
    def variance_scale(self) -> float:
        """Variance scale increases with DR duration."""
        return 1.0 + self.duration_s * 2.0  # 1.0 → large over time


@dataclass
class HorizonCalibration:
    """
    Pitch-based horizon calibration state for DepthEngineV2.
    Uses exponential moving average over stationary/low-speed samples.
    """
    pitch_bias_deg: float = 0.0          # Estimated camera-road pitch bias
    pitch_variance: float = 1.0          # Current uncertainty (degrees^2)
    sample_count: int = 0
    last_update_time: float = 0.0
    is_calibrated: bool = False          # True after MIN_SAMPLES_FOR_HORIZON
    smoothing_alpha: float = PITCH_HORIZON_SMOOTHING_ALPHA
    _ema_pitch: float = 0.0
    _ema_initialized: bool = False

    def update(self, pitch_deg: float, speed_mps: float) -> bool:
        """
        Update horizon calibration with new pitch measurement.
        Only uses samples when vehicle speed is low (< 2 m/s) to avoid road pitch.
        Returns True if calibration was updated.
        """
        if abs(speed_mps) > 2.0:
            return False  # Skip dynamic samples

        if not self._ema_initialized:
            self._ema_pitch = pitch_deg
            self._ema_initialized = True
        else:
            self._ema_pitch = (
                self.smoothing_alpha * pitch_deg
                + (1.0 - self.smoothing_alpha) * self._ema_pitch
            )

        self.pitch_bias_deg = self._ema_pitch
        self.sample_count += 1
        self.last_update_time = time.monotonic()

        # Update variance using exponential decay toward small value
        target_variance = 0.01  # 0.1 deg std dev when converged
        self.pitch_variance = (
            self.smoothing_alpha * (pitch_deg - self._ema_pitch) ** 2
            + (1.0 - self.smoothing_alpha) * self.pitch_variance
        )
        self.pitch_variance = max(target_variance, self.pitch_variance)

        if self.sample_count >= MIN_SAMPLES_FOR_HORIZON:
            self.is_calibrated = True

        return True

    def get_horizon_offset_pixels(self, focal_length_px: float) -> float:
        """
        Convert pitch bias to horizon vertical offset in pixels.
        focal_length_px: camera focal length in pixels.
        """
        return focal_length_px * math.tan(math.radians(self.pitch_bias_deg))


@dataclass
class MTi680Config:
    """Configuration for MTi680CANDriver."""
    stale_threshold_s: float = GNSS_STALE_S
    enable_horizon_calibration: bool = True
    horizon_update_min_speed_mps: float = 2.0
    pitch_plausibility_range: Tuple[float, float] = PITCH_VALID_RANGE_DEG
    roll_plausibility_range: Tuple[float, float] = ROLL_VALID_RANGE_DEG
    dead_reckoning_max_duration_s: float = 30.0
    velocity_sanity_max_mps: float = MAX_VELOCITY_MPS
    enable_gnss_loss_recovery: bool = True


# ---------------------------------------------------------------------------
# MTi-680 CAN Driver
# ---------------------------------------------------------------------------

class MTi680CANDriver:
    """
    Xsens MTi-680 GNSS/INS CAN telemetry decoder.

    Handles:
      - 100 Hz Euler orientation (0x320)
      - 100 Hz velocity vector (0x321)
      - RTK fix status decoding and quality assessment
      - Dead reckoning fallback on GNSS lock loss
      - Pitch-based horizon calibration for DepthEngineV2

    Thread-safe for concurrent RX/publish patterns.
    """

    def __init__(
        self,
        integrity_engine: Optional[CANIntegrityEngine] = None,
        config: Optional[MTi680Config] = None,
    ) -> None:
        self._engine = integrity_engine or CANIntegrityEngine()
        self._cfg = config or MTi680Config()
        self._crc = CRC8Engine()
        self._codec = CANPayloadCodec(self._crc)

        # Navigation state
        self._state = MTi680State()
        self._dr_state = DeadReckoningState()
        self._horizon_cal = HorizonCalibration()
        self._state_lock = threading.RLock()

        # Previous fix type for transition detection
        self._prev_fix: GnssFixType = GnssFixType.NO_FIX
        self._prev_ins: InsStatus = InsStatus.INIT

        # Statistics
        self._stats: Dict[str, Any] = {
            "orient_frames": 0,
            "vel_frames": 0,
            "crc_failures": 0,
            "seq_drops": 0,
            "gnss_lock_losses": 0,
            "gnss_lock_recoveries": 0,
            "dr_events": 0,
            "horizon_updates": 0,
        }
        self._stats_lock = threading.Lock()

        # Callbacks
        self._state_callbacks: List[Callable[[MTi680State], None]] = []
        self._horizon_callbacks: List[Callable[[HorizonCalibration], None]] = []

        # Subscribe to MTi-680 CAN IDs
        self._engine.subscribe(ORIENT_MSG_ID, self._on_orient_frame)
        self._engine.subscribe(VELOCITY_MSG_ID, self._on_velocity_frame)

        log.info(
            "MTi680CANDriver initialized. Stale threshold=%.3f s, "
            "Horizon calibration=%s, Dead reckoning recovery=%s.",
            self._cfg.stale_threshold_s,
            "enabled" if self._cfg.enable_horizon_calibration else "disabled",
            "enabled" if self._cfg.enable_gnss_loss_recovery else "disabled",
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_state(self) -> MTi680State:
        """Return a snapshot of the current navigation state."""
        with self._state_lock:
            # Return a copy to prevent mutation races
            import copy
            return copy.copy(self._state)

    def get_horizon_calibration(self) -> HorizonCalibration:
        """Return current horizon calibration state for DepthEngineV2."""
        with self._state_lock:
            import copy
            return copy.copy(self._horizon_cal)

    def get_dead_reckoning_state(self) -> DeadReckoningState:
        """Return current dead reckoning state."""
        with self._state_lock:
            import copy
            return copy.copy(self._dr_state)

    def is_orientation_valid(self) -> bool:
        """True if orientation data is fresh and not stale."""
        age = self._engine.line.get_age(ORIENT_MSG_ID)
        return age < self._cfg.stale_threshold_s

    def is_velocity_valid(self) -> bool:
        """True if velocity data is fresh and not stale."""
        age = self._engine.line.get_age(VELOCITY_MSG_ID)
        return age < self._cfg.stale_threshold_s

    def is_ins_healthy(self) -> bool:
        """True if INS is providing valid navigation solution."""
        with self._state_lock:
            return (
                self._state.is_navigation_valid
                and self.is_orientation_valid()
                and self.is_velocity_valid()
            )

    def register_state_callback(self, callback: Callable[[MTi680State], None]) -> None:
        """Register callback called on every state update."""
        self._state_callbacks.append(callback)

    def register_horizon_callback(self, callback: Callable[[HorizonCalibration], None]) -> None:
        """Register callback called when horizon calibration is updated."""
        self._horizon_callbacks.append(callback)

    def ingest_frame(self, can_id: int, payload: bytes, timestamp: Optional[float] = None) -> None:
        """
        Ingest raw CAN frame from bus interface.
        Routes to integrity engine which fires registered callbacks.
        """
        ts = timestamp if timestamp is not None else time.monotonic()
        if can_id in (ORIENT_MSG_ID, VELOCITY_MSG_ID):
            self._engine.process_frame(can_id, payload, timestamp=ts)

    def get_statistics(self) -> Dict[str, Any]:
        """Return driver statistics."""
        with self._stats_lock:
            return dict(self._stats)

    def get_pitch_for_depth_engine(self) -> Tuple[float, float, bool]:
        """
        Return pitch bias for DepthEngineV2 horizon correction.
        Returns: (pitch_bias_deg, pitch_variance_deg2, is_calibrated)
        """
        with self._state_lock:
            return (
                self._horizon_cal.pitch_bias_deg,
                self._horizon_cal.pitch_variance,
                self._horizon_cal.is_calibrated,
            )

    # -----------------------------------------------------------------------
    # Internal: CAN callbacks
    # -----------------------------------------------------------------------

    def _on_orient_frame(
        self,
        can_id: int,
        values: Dict[str, float],
        crc_ok: bool,
        seq_status: SequenceStatus,
        timestamp: float,
    ) -> None:
        """Process decoded orientation frame from integrity engine."""
        with self._stats_lock:
            self._stats["orient_frames"] += 1
            if not crc_ok:
                self._stats["crc_failures"] += 1
            if seq_status == SequenceStatus.FRAME_DROP:
                self._stats["seq_drops"] += 1

        roll = values.get("Roll", 0.0)
        pitch = values.get("Pitch", 0.0)
        yaw = values.get("Yaw", 0.0)
        ins_raw = int(values.get("InsStatus", 0))
        ins_status = InsStatus(min(ins_raw, 5))

        # Plausibility checks
        roll_ok = self._cfg.roll_plausibility_range[0] <= roll <= self._cfg.roll_plausibility_range[1]
        pitch_ok = self._cfg.pitch_plausibility_range[0] <= pitch <= self._cfg.pitch_plausibility_range[1]

        if not roll_ok:
            log.warning("MTi-680: Roll=%.2f deg exceeds plausibility range. Clamping.", roll)
            roll = max(self._cfg.roll_plausibility_range[0],
                       min(self._cfg.roll_plausibility_range[1], roll))
        if not pitch_ok:
            log.warning("MTi-680: Pitch=%.2f deg exceeds plausibility range. Clamping.", pitch)
            pitch = max(self._cfg.pitch_plausibility_range[0],
                        min(self._cfg.pitch_plausibility_range[1], pitch))

        with self._state_lock:
            self._state.roll_deg = roll
            self._state.pitch_deg = pitch
            self._state.yaw_deg = yaw
            self._state.orient_valid = crc_ok
            self._state.orient_timestamp = timestamp
            self._state.orient_crc_ok = crc_ok
            self._state.orient_seq_status = seq_status
            self._state.ins_status = ins_status
            self._state.heading_deg = yaw

            # Update dead reckoning state based on INS status
            self._update_dead_reckoning(ins_status, timestamp)

            # Update horizon calibration
            if self._cfg.enable_horizon_calibration:
                speed = self._state.speed_mps
                updated = self._horizon_cal.update(pitch, speed)
                if updated:
                    with self._stats_lock:
                        self._stats["horizon_updates"] += 1

            state_copy = self._state

        # Fire callbacks
        for cb in self._state_callbacks:
            try:
                cb(state_copy)
            except Exception as exc:
                log.error("State callback error: %s", exc)

        if self._cfg.enable_horizon_calibration:
            with self._state_lock:
                cal_copy = self._horizon_cal
            for cb in self._horizon_callbacks:
                try:
                    cb(cal_copy)
                except Exception as exc:
                    log.error("Horizon callback error: %s", exc)

    def _on_velocity_frame(
        self,
        can_id: int,
        values: Dict[str, float],
        crc_ok: bool,
        seq_status: SequenceStatus,
        timestamp: float,
    ) -> None:
        """Process decoded velocity frame from integrity engine."""
        with self._stats_lock:
            self._stats["vel_frames"] += 1
            if not crc_ok:
                self._stats["crc_failures"] += 1

        vel_e = values.get("VelocityEast", 0.0)
        vel_n = values.get("VelocityNorth", 0.0)
        vel_d = values.get("VelocityDown", 0.0)
        fix_raw = int(values.get("GnssFixType", 0))
        gnss_fix = GnssFixType(min(fix_raw, 6))

        # Sanity check velocity magnitudes
        speed_3d = math.sqrt(vel_e**2 + vel_n**2 + vel_d**2)
        if speed_3d > self._cfg.velocity_sanity_max_mps:
            log.warning(
                "MTi-680: Velocity sanity check failed: speed=%.2f m/s > max=%.2f m/s. "
                "Zeroing velocity.", speed_3d, self._cfg.velocity_sanity_max_mps
            )
            vel_e = vel_n = vel_d = 0.0
            gnss_fix = GnssFixType.NO_FIX

        speed_2d = math.sqrt(vel_e**2 + vel_n**2)

        with self._state_lock:
            self._state.vel_east_mps = vel_e
            self._state.vel_north_mps = vel_n
            self._state.vel_down_mps = vel_d
            self._state.vel_valid = crc_ok
            self._state.vel_timestamp = timestamp
            self._state.vel_crc_ok = crc_ok
            self._state.vel_seq_status = seq_status
            self._state.gnss_fix_type = gnss_fix
            self._state.speed_mps = speed_2d

            # Detect GNSS lock transitions
            self._handle_gnss_transition(gnss_fix, timestamp)
            self._prev_fix = gnss_fix

    def _update_dead_reckoning(self, ins_status: InsStatus, timestamp: float) -> None:
        """Update dead reckoning state. Must be called under state_lock."""
        if ins_status == InsStatus.DEAD_RECK:
            if not self._dr_state.is_active:
                self._dr_state.is_active = True
                self._dr_state.start_time = timestamp
                self._dr_state.last_valid_fix = self._state.gnss_fix_type
                self._state.is_dead_reckoning = True
                self._state.dr_start_time = timestamp
                with self._stats_lock:
                    self._stats["dr_events"] += 1
                log.warning(
                    "MTi-680: Dead reckoning ACTIVATED at t=%.3f. "
                    "Last valid fix: %s", timestamp, self._dr_state.last_valid_fix.name,
                )
        else:
            if self._dr_state.is_active:
                dr_duration = timestamp - self._dr_state.start_time
                log.info(
                    "MTi-680: Dead reckoning DEACTIVATED after %.2f s. "
                    "INS status: %s", dr_duration, ins_status.name,
                )
                self._dr_state.is_active = False
                self._dr_state.position_uncertainty_m = 0.0
                self._state.is_dead_reckoning = False

        if self._dr_state.is_active:
            self._state.dr_duration_s = timestamp - self._dr_state.start_time
            # Check if DR has exceeded maximum duration
            if self._state.dr_duration_s > self._cfg.dead_reckoning_max_duration_s:
                log.error(
                    "MTi-680: Dead reckoning duration %.2f s exceeds maximum %.2f s. "
                    "Navigation DEGRADED.", self._state.dr_duration_s,
                    self._cfg.dead_reckoning_max_duration_s,
                )

    def _handle_gnss_transition(self, new_fix: GnssFixType, timestamp: float) -> None:
        """Handle GNSS fix type transitions. Must be called under state_lock."""
        if self._prev_fix != new_fix:
            if self._prev_fix.is_rtk and not new_fix.is_rtk:
                with self._stats_lock:
                    self._stats["gnss_lock_losses"] += 1
                log.warning(
                    "MTi-680: RTK lock LOST. %s → %s",
                    self._prev_fix.name, new_fix.name,
                )
            elif not self._prev_fix.is_rtk and new_fix.is_rtk:
                with self._stats_lock:
                    self._stats["gnss_lock_recoveries"] += 1
                log.info(
                    "MTi-680: RTK lock RECOVERED. %s → %s",
                    self._prev_fix.name, new_fix.name,
                )

    # -----------------------------------------------------------------------
    # Direct decode (for testing/utilities)
    # -----------------------------------------------------------------------

    def decode_orientation_raw(self, payload: bytes, timestamp: float = 0.0) -> Optional[Dict[str, float]]:
        """
        Directly decode a raw 0x320 orientation payload.
        Returns decoded physical values dict or None on CRC failure.
        """
        spec = APEX_MESSAGE_SPECS.get(ORIENT_MSG_ID)
        if spec is None:
            return None
        values, crc_ok, _ = self._codec.decode(spec, payload)
        if not crc_ok:
            log.warning("MTi-680 orientation decode: CRC failure.")
        return values

    def decode_velocity_raw(self, payload: bytes, timestamp: float = 0.0) -> Optional[Dict[str, float]]:
        """
        Directly decode a raw 0x321 velocity payload.
        Returns decoded physical values dict or None on CRC failure.
        """
        spec = APEX_MESSAGE_SPECS.get(VELOCITY_MSG_ID)
        if spec is None:
            return None
        values, crc_ok, _ = self._codec.decode(spec, payload)
        if not crc_ok:
            log.warning("MTi-680 velocity decode: CRC failure.")
        return values

    def get_diagnostic_report(self) -> Dict[str, Any]:
        """Return comprehensive driver diagnostic report."""
        with self._state_lock:
            state = self._state
            dr = self._dr_state
            cal = self._horizon_cal

        return {
            "driver": "MTi680CANDriver",
            "healthy": self.is_ins_healthy(),
            "orientation": {
                "roll_deg": round(state.roll_deg, 3),
                "pitch_deg": round(state.pitch_deg, 3),
                "yaw_deg": round(state.yaw_deg, 3),
                "valid": state.orient_valid,
                "age_s": round(self._engine.line.get_age(ORIENT_MSG_ID), 3),
            },
            "velocity": {
                "east_mps": round(state.vel_east_mps, 3),
                "north_mps": round(state.vel_north_mps, 3),
                "down_mps": round(state.vel_down_mps, 3),
                "speed_mps": round(state.speed_mps, 3),
                "valid": state.vel_valid,
                "age_s": round(self._engine.line.get_age(VELOCITY_MSG_ID), 3),
            },
            "gnss": {
                "fix_type": state.gnss_fix_type.name,
                "ins_status": state.ins_status.name,
                "accuracy_m": state.gnss_fix_type.accuracy_m,
                "position_variance_m2": round(state.position_variance_m2, 6),
            },
            "dead_reckoning": {
                "active": dr.is_active,
                "duration_s": round(dr.duration_s, 2) if dr.is_active else 0.0,
                "position_uncertainty_m": round(dr.position_uncertainty_m, 3),
            },
            "horizon_calibration": {
                "calibrated": cal.is_calibrated,
                "pitch_bias_deg": round(cal.pitch_bias_deg, 4),
                "pitch_variance": round(cal.pitch_variance, 6),
                "sample_count": cal.sample_count,
            },
            "statistics": self.get_statistics(),
        }
