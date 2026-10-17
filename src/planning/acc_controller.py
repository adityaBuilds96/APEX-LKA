"""
src/planning/acc_controller.py
==============================
Adaptive Cruise Control (ACC) & Dynamic Curvature Speed Profiler for APEX-LKA:
- Intelligent Driver Model (IDM) car-following control:
    a_acc = a_max * [ 1 - (v_ego / v_set)^4 - (d_target / d_actual)^2 ]
    d_target = d_standstill + t_gap * v_ego
- Kinematic curvature speed clamping on road bends:
    v_bend = sqrt( a_max,lat / (kappa + eps) )
- Dynamic jerk limiter: |da/dt| <= 2.0 m/s^3 to guarantee vehicle stability.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

log = logging.getLogger("apex_acc_controller")


@dataclass
class ACCConfig:
    """Configurable parameters for ACC and longitudinal motion profiling."""
    default_set_speed_mps: float = 4.17       # 15 km/h chassis cruising speed
    standstill_distance_m: float = 2.0        # d_standstill (m)
    time_gap_seconds: float = 1.5             # t_gap (s)
    max_accel_mpss: float = 1.5               # a_max (m/s^2)
    max_decel_mpss: float = 3.0               # a_decel_max (m/s^2)
    max_lateral_accel_mpss: float = 2.5       # a_max,lat comfort limit for curves (m/s^2)
    max_jerk_mps3: float = 2.0                # |da/dt| maximum longitudinal jerk (m/s^3)
    min_speed_mps: float = 0.0                # Minimum forward speed
    max_detection_horizon_m: float = 35.0     # Max horizon for active car-following


@dataclass
class ACCControlOutput:
    """Calculated longitudinal targets for powertrain and chassis controller."""
    target_speed_mps: float = 4.17
    commanded_accel_mpss: float = 0.0
    target_gap_m: float = 2.0
    actual_gap_m: Optional[float] = None
    gap_error_m: Optional[float] = None
    curvature_speed_limit_mps: float = 99.0
    is_lead_detected: bool = False
    jerk_limited: bool = False

    @property
    def acceleration_mpss(self) -> float:
        return self.commanded_accel_mpss

    @property
    def lead_vehicle_detected(self) -> bool:
        return self.is_lead_detected

    @property
    def distance_to_lead_m(self) -> Optional[float]:
        return self.actual_gap_m

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target_speed_mps": round(float(self.target_speed_mps), 2),
            "commanded_accel_mpss": round(float(self.commanded_accel_mpss), 2),
            "target_gap_m": round(float(self.target_gap_m), 2),
            "actual_gap_m": round(float(self.actual_gap_m), 2) if self.actual_gap_m is not None else None,
            "gap_error_m": round(float(self.gap_error_m), 2) if self.gap_error_m is not None else None,
            "curvature_speed_limit_mps": round(float(self.curvature_speed_limit_mps), 2),
            "is_lead_detected": self.is_lead_detected,
            "jerk_limited": self.jerk_limited,
        }


ACCOutput = ACCControlOutput


class AdaptiveCruiseController:
    """
    Intelligent Driver Model (IDM) longitudinal controller with curvature speed clamping
    and bounded jerk profile generation.
    """

    def __init__(self, config: Optional[ACCConfig] = None) -> None:
        self.config = config or ACCConfig()
        self._last_accel_cmd: float = 0.0

    def reset(self) -> None:
        """Reset internal integrator and jerk filter state."""
        self._last_accel_cmd = 0.0

    def compute_curvature_speed_limit(self, curvature: float) -> float:
        """
        Calculates safe negotiation speed on curved trajectories:
            v_bend = sqrt( a_max,lat / (|kappa| + eps) )
        """
        abs_kappa = abs(float(curvature))
        eps = 1e-4
        if abs_kappa < eps:
            return 99.0

        v_bend = math.sqrt(self.config.max_lateral_accel_mpss / (abs_kappa + eps))
        return float(v_bend)

    def compute_target_gap(self, ego_speed_mps: float) -> float:
        """Calculates speed-dependent target headway gap: d_target = d_0 + T * v."""
        v = max(0.0, float(ego_speed_mps))
        return float(self.config.standstill_distance_m + self.config.time_gap_seconds * v)

    def update(
        self,
        ego_speed_mps: float,
        v_set_mps: Optional[float] = None,
        lead_distance_m: Optional[float] = None,
        lead_speed_mps: Optional[float] = None,
        path_curvature: float = 0.0,
        dt: float = 0.033,
    ) -> ACCControlOutput:
        """
        Executes one cycle of ACC speed profiling.

        1. Computes curvature speed limit v_bend.
        2. Clamps desired speed v_desired = min(v_set, v_bend).
        3. Evaluates IDM acceleration formula based on actual vs target gap.
        4. Applies deceleration limits and jerk slew-rate constraints.
        5. Computes resulting target speed forward.
        """
        dt_clamped = max(1e-3, min(float(dt), 0.2))
        v_ego = max(0.0, float(ego_speed_mps))

        # Base set speed
        v_set = float(v_set_mps if v_set_mps is not None else self.config.default_set_speed_mps)

        # 1. Curvature speed clamp
        v_bend = self.compute_curvature_speed_limit(path_curvature)
        v_desired = min(v_set, v_bend)

        # 2. IDM Car-Following Acceleration
        d_target = self.compute_target_gap(v_ego)

        has_lead = (
            lead_distance_m is not None
            and lead_distance_m > 0.1
            and lead_distance_m <= self.config.max_detection_horizon_m
        )

        a_max = self.config.max_accel_mpss

        # Free-road acceleration component
        if v_desired > 0.1:
            ratio_v = min(2.5, v_ego / v_desired)
            free_accel = a_max * (1.0 - (ratio_v ** 4))
        else:
            free_accel = -self.config.max_decel_mpss

        if has_lead:
            d_act = max(0.5, float(lead_distance_m))
            # Relative speed term: closing velocity penalty
            v_rel = (v_ego - float(lead_speed_mps)) if lead_speed_mps is not None else 0.0
            s_star = d_target + max(0.0, (v_ego * v_rel) / (2.0 * math.sqrt(max(0.1, a_max * self.config.max_decel_mpss))))
            int_term = (s_star / d_act) ** 2

            raw_accel = free_accel - a_max * int_term
            gap_err = d_act - d_target
        else:
            raw_accel = free_accel
            gap_err = None

        # 3. Acceleration envelope clamping
        accel_clamped = float(np.clip(raw_accel, -self.config.max_decel_mpss, a_max))

        # 4. Jerk Slew-Rate Limiting: |da/dt| <= max_jerk
        max_delta_a = self.config.max_jerk_mps3 * dt_clamped
        accel_jerk_bounded = float(np.clip(
            accel_clamped,
            self._last_accel_cmd - max_delta_a,
            self._last_accel_cmd + max_delta_a,
        ))

        is_jerk_limited = abs(accel_jerk_bounded - accel_clamped) > 1e-3
        self._last_accel_cmd = accel_jerk_bounded

        # 5. Compute Target Speed
        v_target = max(self.config.min_speed_mps, v_ego + accel_jerk_bounded * dt_clamped)
        v_target = min(v_target, v_desired)

        return ACCControlOutput(
            target_speed_mps=float(v_target),
            commanded_accel_mpss=float(accel_jerk_bounded),
            target_gap_m=float(d_target),
            actual_gap_m=float(lead_distance_m) if has_lead else None,
            gap_error_m=float(gap_err) if gap_err is not None else None,
            curvature_speed_limit_mps=float(v_bend),
            is_lead_detected=has_lead,
            jerk_limited=is_jerk_limited,
        )
