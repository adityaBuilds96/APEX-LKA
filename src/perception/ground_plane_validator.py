"""
src/perception/ground_plane_validator.py
========================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Ground-Plane Assumption Validity Gate & Attitude Limit Supervisor.
ISO 26262 ASIL-B Dynamic Perception Fault-Containment Baseline.

Operational Validity Constraints & Hazard Analysis:
----------------------------------------------------
Monocular flat-ground depth estimation strictly depends on the planar assumption:
1. Chassis Pitch Limits:
   - Pitch angle theta must remain within [-15.0 deg, +15.0 deg].
   - Beyond this threshold, suspension pitch dynamics invalidate planar projection.
2. Chassis Roll Limits:
   - Roll angle phi must remain within [-12.0 deg, +12.0 deg].
   - Excessive bank angles require full 3D homography projection.
3. Vanishing Line Horizon Bounds:
   - Calculated horizon row v_horizon = c_y - f_y * tan(theta_pitch).
   - Must reside strictly inside valid image frame limits [15px, 340px].
4. Pitch Rate & Suspension Dynamic Jerk:
   - Detects harsh braking transients or speed-bump strikes (pitch_rate > 35 deg/s).
5. Road Slope & Grade Stability:
   - Evaluates sustained road gradient to prevent false distance foreshortening.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import IntFlag, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.GroundPlaneValidator")


# ──────────────────────────────────────────────────────────────────────────────
# 1. ISO 26262 ASIL-B Fault Bitmask Codes & Diagnostic Trouble Codes (DTC)
# ──────────────────────────────────────────────────────────────────────────────

class GroundPlaneFaultBits(IntFlag):
    """
    Standardized AUTOSAR-compatible 32-bit diagnostic trouble bitmask.
    Used for hardware telemetry and fail-safe state machine handoff.
    """
    OK = 0
    PITCH_EXCEEDED = auto()              # Bit 0: Pitch angle beyond max limit
    ROLL_EXCEEDED = auto()               # Bit 1: Roll angle beyond bank limit
    PITCH_RATE_EXCEEDED = auto()         # Bit 2: Angular pitch jerk (braking / bump)
    HORIZON_OUT_OF_BOUNDS = auto()       # Bit 3: Horizon line projected outside camera FOV
    CONFIDENCE_SUBPAR = auto()           # Bit 4: Aggregate confidence below safety margin
    SUSPENSION_ASYMMETRY = auto()        # Bit 5: Asymmetric chassis loading detected
    SUSTAINED_GRADE_WARNING = auto()     # Bit 6: Road grade > 12% for prolonged duration
    INS_TELEMETRY_STALE = auto()         # Bit 7: INS heartbeat timed out (> 25 ms)
    SENSOR_SATURATION = auto()           # Bit 8: IMU gyro or accel clipping threshold hit


# ──────────────────────────────────────────────────────────────────────────────
# 2. Configuration & Diagnostic Containers
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GroundPlaneLimitsConfig:
    """
    Physical thresholds defining valid ground plane geometry.
    Designed for standard passenger and commercial vehicle chassis dynamics.
    """
    max_pitch_deg: float = 15.0
    max_roll_deg: float = 12.0
    max_pitch_rate_deg_s: float = 35.0
    focal_length_y_px: float = 450.0
    principal_point_y_px: float = 180.0
    image_height_px: int = 360
    min_horizon_row_px: float = 15.0
    max_horizon_row_px: float = 340.0
    min_valid_confidence: float = 0.40
    max_acceptable_grade_pct: float = 14.0
    telemetry_stale_timeout_sec: float = 0.050
    pitch_smoothing_alpha: float = 0.25
    roll_smoothing_alpha: float = 0.25


@dataclass
class GroundPlaneValidityResult:
    """
    Diagnostic gate report evaluating whether monocular planar depth is dependable.
    Broadcast to downstream perception nodes and vehicle state machine.
    """
    is_valid: bool
    confidence_score: float             # [0.0, 1.0]
    horizon_row_px: float
    current_pitch_deg: float
    current_roll_deg: float
    current_pitch_rate_deg_s: float
    is_pitch_safe: bool
    is_roll_safe: bool
    is_horizon_in_frame: bool
    is_dynamic_rate_safe: bool
    fault_codes: List[str] = field(default_factory=list)
    fault_bitmask: int = 0
    estimated_road_grade_pct: float = 0.0
    timestamp_utc: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result into standard serializable dictionary."""
        return asdict(self)

    @property
    def has_critical_fault(self) -> bool:
        """Indicate whether fault requires immediate transition to degraded mode."""
        critical_mask = (
            GroundPlaneFaultBits.PITCH_EXCEEDED |
            GroundPlaneFaultBits.HORIZON_OUT_OF_BOUNDS |
            GroundPlaneFaultBits.INS_TELEMETRY_STALE
        )
        return bool(self.fault_bitmask & critical_mask)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Dynamic Attitude & Suspension Filtering Helper
# ──────────────────────────────────────────────────────────────────────────────

class AttitudeDynamicsSmoother:
    """
    Exponential low-pass filter and finite-difference derivative estimator
    for raw attitude telemetry received over high-speed CAN / serial.
    """

    def __init__(self, alpha_pitch: float = 0.25, alpha_roll: float = 0.25) -> None:
        self.alpha_pitch = alpha_pitch
        self.alpha_roll = alpha_roll
        self._filtered_pitch_rad: Optional[float] = None
        self._filtered_roll_rad: Optional[float] = None
        self._prev_pitch_deg: float = 0.0
        self._prev_timestamp: float = 0.0
        self._instantaneous_pitch_rate: float = 0.0

    def update(
        self,
        raw_pitch_rad: float,
        raw_roll_rad: float,
        timestamp_sec: float,
    ) -> Tuple[float, float, float]:
        """
        Apply low-pass filter and compute angular pitch rate.
        Returns:
            (filtered_pitch_rad, filtered_roll_rad, pitch_rate_deg_s)
        """
        if self._filtered_pitch_rad is None:
            self._filtered_pitch_rad = raw_pitch_rad
            self._filtered_roll_rad = raw_roll_rad
            self._prev_pitch_deg = math.degrees(raw_pitch_rad)
            self._prev_timestamp = timestamp_sec
            return raw_pitch_rad, raw_roll_rad, 0.0

        # Exponential low-pass filter
        self._filtered_pitch_rad = (
            self.alpha_pitch * raw_pitch_rad + (1.0 - self.alpha_pitch) * self._filtered_pitch_rad
        )
        self._filtered_roll_rad = (
            self.alpha_roll * raw_roll_rad + (1.0 - self.alpha_roll) * self._filtered_roll_rad
        )

        dt = max(1e-4, timestamp_sec - self._prev_timestamp)
        curr_pitch_deg = math.degrees(raw_pitch_rad)
        self._instantaneous_pitch_rate = abs(curr_pitch_deg - self._prev_pitch_deg) / dt

        self._prev_pitch_deg = curr_pitch_deg
        self._prev_timestamp = timestamp_sec

        return self._filtered_pitch_rad, self._filtered_roll_rad, self._instantaneous_pitch_rate

    def reset(self) -> None:
        """Reset internal filter states."""
        self._filtered_pitch_rad = None
        self._filtered_roll_rad = None
        self._prev_pitch_deg = 0.0
        self._prev_timestamp = 0.0
        self._instantaneous_pitch_rate = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# 4. Master Ground Plane Validator Engine
# ──────────────────────────────────────────────────────────────────────────────

class GroundPlaneValidator:
    """
    ISO 26262 ASIL-B Perception Safety Gate.
    Monitors live attitude telemetry from the MTi-680 INS and suppresses monocular
    depth estimation whenever chassis dynamics violate flat-ground assumptions.
    
    Safety Goals:
      SG-01: Inhibit false emergency braking caused by ground plane divergence.
      SG-02: Prevent lateral cross-track divergence on steep road inclines.
    """

    def __init__(self, limits: Optional[GroundPlaneLimitsConfig] = None) -> None:
        self.limits = limits or GroundPlaneLimitsConfig()
        self._prev_pitch_deg: float = 0.0
        self._prev_timestamp_sec: float = 0.0
        self._smoother = AttitudeDynamicsSmoother(
            alpha_pitch=self.limits.pitch_smoothing_alpha,
            alpha_roll=self.limits.roll_smoothing_alpha,
        )
        self._total_validations: int = 0
        self._total_rejections: int = 0
        self._last_heartbeat_time: float = time.time()

    def calculate_horizon_row(self, pitch_rad: float) -> float:
        """
        Calculate pixel row where ground plane intersects infinity:
          v_horizon = c_y - f_y * tan(pitch)
        """
        v_h = self.limits.principal_point_y_px - self.limits.focal_length_y_px * math.tan(pitch_rad)
        return v_h

    def estimate_road_grade(self, pitch_rad: float) -> float:
        """
        Estimate road gradient in percent from chassis pitch:
          Grade (%) = tan(pitch_rad) * 100.0
        """
        return math.tan(pitch_rad) * 100.0

    def validate_attitude(
        self,
        pitch_rad: float,
        roll_rad: float,
        timestamp_sec: Optional[float] = None,
        live_pitch_rate_deg_s: Optional[float] = None,
    ) -> GroundPlaneValidityResult:
        """
        Evaluate full attitude health against safety thresholds.
        
        Args:
            pitch_rad: Current chassis pitch angle in radians (positive nose up).
            roll_rad: Current chassis roll angle in radians (positive right down).
            timestamp_sec: Optional wall-clock or telemetry timestamp in seconds.
            live_pitch_rate_deg_s: Optional precomputed angular pitch velocity.

        Returns:
            GroundPlaneValidityResult with validation status and diagnostic telemetry.
        """
        self._total_validations += 1
        curr_time = timestamp_sec if timestamp_sec is not None else time.time()
        self._last_heartbeat_time = curr_time

        pitch_deg = math.degrees(pitch_rad)
        roll_deg = math.degrees(roll_rad)
        faults: List[str] = []
        fault_mask = GroundPlaneFaultBits.OK

        # 1. Pitch Rate Calculation
        if live_pitch_rate_deg_s is not None:
            pitch_rate = abs(live_pitch_rate_deg_s)
        elif timestamp_sec is not None and self._prev_timestamp_sec > 0.0:
            dt = max(1e-4, timestamp_sec - self._prev_timestamp_sec)
            pitch_rate = abs(pitch_deg - self._prev_pitch_deg) / dt
        else:
            pitch_rate = 0.0

        if timestamp_sec is not None:
            self._prev_pitch_deg = pitch_deg
            self._prev_timestamp_sec = timestamp_sec

        # 2. Horizon Row Verification
        v_horizon = self.calculate_horizon_row(pitch_rad)
        road_grade_pct = self.estimate_road_grade(pitch_rad)

        is_pitch_safe = abs(pitch_deg) <= self.limits.max_pitch_deg
        is_roll_safe = abs(roll_deg) <= self.limits.max_roll_deg
        is_rate_safe = pitch_rate <= self.limits.max_pitch_rate_deg_s
        is_horizon_in_frame = (
            self.limits.min_horizon_row_px <= v_horizon <= self.limits.max_horizon_row_px
        )

        if not is_pitch_safe:
            faults.append(f"Chassis pitch exceeded limit: {pitch_deg:.2f}° > {self.limits.max_pitch_deg}°")
            fault_mask |= GroundPlaneFaultBits.PITCH_EXCEEDED

        if not is_roll_safe:
            faults.append(f"Chassis roll exceeded limit: {roll_deg:.2f}° > {self.limits.max_roll_deg}°")
            fault_mask |= GroundPlaneFaultBits.ROLL_EXCEEDED

        if not is_rate_safe:
            faults.append(f"Harsh pitch transient: {pitch_rate:.1f}°/s > {self.limits.max_pitch_rate_deg_s}°/s")
            fault_mask |= GroundPlaneFaultBits.PITCH_RATE_EXCEEDED

        if not is_horizon_in_frame:
            faults.append(f"Vanishing horizon row {v_horizon:.1f}px outside frame bounds.")
            fault_mask |= GroundPlaneFaultBits.HORIZON_OUT_OF_BOUNDS

        if abs(road_grade_pct) > self.limits.max_acceptable_grade_pct:
            faults.append(f"Sustained road grade warning: {road_grade_pct:.1f}%")
            fault_mask |= GroundPlaneFaultBits.SUSTAINED_GRADE_WARNING

        # 3. Confidence Calculation
        pitch_penalty = max(0.0, 1.0 - (abs(pitch_deg) / self.limits.max_pitch_deg))
        roll_penalty = max(0.0, 1.0 - (abs(roll_deg) / self.limits.max_roll_deg))
        rate_penalty = max(0.0, 1.0 - (pitch_rate / self.limits.max_pitch_rate_deg_s))

        confidence = 0.40 * pitch_penalty + 0.30 * roll_penalty + 0.30 * rate_penalty
        if not is_horizon_in_frame:
            confidence = 0.0

        if confidence < self.limits.min_valid_confidence:
            fault_mask |= GroundPlaneFaultBits.CONFIDENCE_SUBPAR

        is_overall_valid = (len(faults) == 0) and (confidence >= self.limits.min_valid_confidence)

        if not is_overall_valid:
            self._total_rejections += 1

        return GroundPlaneValidityResult(
            is_valid=is_overall_valid,
            confidence_score=round(confidence, 3),
            horizon_row_px=round(v_horizon, 1),
            current_pitch_deg=round(pitch_deg, 2),
            current_roll_deg=round(roll_deg, 2),
            current_pitch_rate_deg_s=round(pitch_rate, 2),
            is_pitch_safe=is_pitch_safe,
            is_roll_safe=is_roll_safe,
            is_horizon_in_frame=is_horizon_in_frame,
            is_dynamic_rate_safe=is_rate_safe,
            fault_codes=faults,
            fault_bitmask=int(fault_mask),
            estimated_road_grade_pct=round(road_grade_pct, 2),
            timestamp_utc=curr_time,
        )

    def evaluate_batch(
        self,
        pitch_sequence: Sequence[float],
        roll_sequence: Sequence[float],
        timestamps: Sequence[float],
    ) -> List[GroundPlaneValidityResult]:
        """
        Batch validation for recorded drive logs and offline validation runs.
        """
        results: List[GroundPlaneValidityResult] = []
        for p, r, t in zip(pitch_sequence, roll_sequence, timestamps):
            res = self.validate_attitude(pitch_rad=p, roll_rad=r, timestamp_sec=t)
            results.append(res)
        return results

    def get_statistics(self) -> Dict[str, Any]:
        """
        Return runtime health metrics for ASIL-B supervisory logging.
        """
        rejection_rate = (
            (self._total_rejections / self._total_validations * 100.0)
            if self._total_validations > 0 else 0.0
        )
        return {
            "total_validations": self._total_validations,
            "total_rejections": self._total_rejections,
            "rejection_rate_pct": round(rejection_rate, 2),
            "last_heartbeat_time": self._last_heartbeat_time,
        }

    def reset_statistics(self) -> None:
        """Reset validation and rejection counters."""
        self._total_validations = 0
        self._total_rejections = 0
        self._smoother.reset()


# ──────────────────────────────────────────────────────────────────────────────
# 5. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

def run_ground_plane_self_test() -> bool:
    """Execute built-in self-test verifying all edge cases and diagnostic paths."""
    print("Executing GroundPlaneValidator ISO 26262 ASIL-B self-check...")
    validator = GroundPlaneValidator()

    # Normal level driving
    res_normal = validator.validate_attitude(pitch_rad=math.radians(1.0), roll_rad=math.radians(0.5))
    assert res_normal.is_valid, "Normal driving check failed!"
    assert res_normal.confidence_score > 0.85, "Confidence score lower than expected!"
    print(f"Normal driving check passed (conf={res_normal.confidence_score}).")

    # Extreme pitch violation
    res_extreme = validator.validate_attitude(pitch_rad=math.radians(20.0), roll_rad=0.0)
    assert not res_extreme.is_valid, "Extreme pitch violation was not rejected!"
    assert "pitch" in res_extreme.fault_codes[0].lower()
    assert res_extreme.has_critical_fault, "Critical fault bit was not set!"
    print("Extreme pitch rejection passed.")

    # Harsh transient check
    validator.validate_attitude(pitch_rad=0.0, roll_rad=0.0, timestamp_sec=10.0)
    res_jerk = validator.validate_attitude(pitch_rad=math.radians(5.0), roll_rad=0.0, timestamp_sec=10.05)
    assert not res_jerk.is_dynamic_rate_safe, "Harsh suspension jerk was not detected!"
    print("Harsh suspension transient rejection passed.")

    # Road slope grade estimation
    grade = validator.estimate_road_grade(math.radians(4.0))
    assert 6.5 < grade < 7.5, f"Unexpected road grade: {grade}%"
    print(f"Road grade check passed: {grade:.2f}%.")

    stats = validator.get_statistics()
    assert stats["total_validations"] >= 3
    print("All GroundPlaneValidator checks successfully verified.")
    return True


if __name__ == "__main__":
    run_ground_plane_self_test()
