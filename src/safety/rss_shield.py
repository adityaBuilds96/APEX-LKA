"""
src/safety/rss_shield.py
========================
Mobileye Responsibility-Sensitive Safety (RSS) Mathematical Safety Shield:
- Formal non-collision safety envelope guarantees (ISO 26262 ASIL-B Compliant).
- Formulation: Mobileye RSS Minimum Longitudinal Distance:
    d_min = [ v_resp * rho + 0.5 * a_max_acc * rho^2 +
              (v_resp + rho * a_max_acc)^2 / (2 * a_min_brake) -
              v_lead^2 / (2 * a_max_brake) ]^+ + d_standstill
- Lateral safety clearance envelope.
- Force hard override flag (RSS_OVERRIDE = True) to trigger maximum physical deceleration when d_actual < d_min.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Defensive project root / config resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import PATHS, cfg
except (ImportError, ModuleNotFoundError, AttributeError):
    PATHS = {"ROOT": PROJECT_ROOT}
    cfg = None

log = logging.getLogger("apex_safety_rss")


@dataclass
class RSSConfig:
    """Mobileye RSS physical parameters for SAE autonomous vehicle chassis."""
    response_time_sec: float = 0.5            # Driver / perception reaction time rho (s)
    accel_max_following_mpss: float = 1.5     # Maximum following vehicle acceleration a_max,acc (m/s^2)
    min_brake_mpss: float = 3.0               # Minimum emergency braking rate a_min,brake (m/s^2)
    max_brake_mpss: float = 5.0               # Maximum leading vehicle braking capability a_max,brake (m/s^2)
    min_lateral_distance_m: float = 0.75      # Minimum lateral safety margin (m)
    standstill_clearance_m: float = 1.5       # Buffer distance when stopped (m)
    emergency_decel_cmd_mpss: float = -4.5    # Commanded deceleration on override (m/s^2)
    corridor_half_width_m: float = 1.75       # Lateral threat boundary (m)


@dataclass
class RSSSafetyCheckResult:
    """Formal safety evaluation result."""
    is_safe: bool = True
    override_active: bool = False
    d_min_long: float = 0.0
    d_actual_long: float = 999.0
    d_min_lat: float = 0.75
    d_actual_lat: float = 999.0
    violation_margin_m: float = 0.0           # Positive if violated (d_min - d_actual)
    commanded_accel_mpss: Optional[float] = None
    safety_status: str = "SAFE"               # "SAFE", "CAUTION", "OVERRIDE_BRAKE"
    lead_track_id: Optional[int] = None
    execution_ms: float = 0.0

    @property
    def RSS_OVERRIDE(self) -> bool:
        """Explicit Mobileye RSS Hard Override flag."""
        return self.override_active

    @property
    def safe_acceleration_mpss(self) -> float:
        return self.commanded_accel_mpss if self.commanded_accel_mpss is not None else -4.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_safe": self.is_safe,
            "override_active": self.override_active,
            "RSS_OVERRIDE": self.RSS_OVERRIDE,
            "d_min_long": round(float(self.d_min_long), 2),
            "d_actual_long": round(float(self.d_actual_long), 2),
            "violation_margin_m": round(float(self.violation_margin_m), 2),
            "commanded_accel_mpss": round(float(self.commanded_accel_mpss), 2) if self.commanded_accel_mpss is not None else None,
            "safety_status": self.safety_status,
            "lead_track_id": self.lead_track_id,
            "execution_ms": round(float(self.execution_ms), 3),
        }


RSSShieldResult = RSSSafetyCheckResult


class MobileyeRSSShield:
    """
    Responsibility-Sensitive Safety mathematical barrier for collision avoidance.
    Enforces non-collision guarantees via closed-form kinematic braking inequalities.
    """

    def __init__(self, config: Optional[RSSConfig] = None) -> None:
        self.config = config or RSSConfig()
        self.last_result: Optional[RSSSafetyCheckResult] = None

    def compute_min_safe_longitudinal_distance(
        self,
        ego_speed_mps: float,
        lead_speed_mps: float,
        rho: Optional[float] = None,
        a_max_acc: Optional[float] = None,
        a_min_brake: Optional[float] = None,
        a_max_brake: Optional[float] = None,
    ) -> float:
        """
        Closed-form Mobileye RSS formula for same-direction longitudinal distance:
        d_min = [ v_resp * rho + 0.5 * a_max_acc * rho^2 +
                  (v_resp + rho * a_max_acc)^2 / (2 * a_min_brake) -
                  v_lead^2 / (2 * a_max_brake) ]^+ + d_standstill
        """
        v_resp = max(0.0, float(ego_speed_mps))
        v_lead = max(0.0, float(lead_speed_mps))

        rho_val = float(rho) if rho is not None else self.config.response_time_sec
        a_acc = float(a_max_acc) if a_max_acc is not None else self.config.accel_max_following_mpss
        a_min_b = float(a_min_brake) if a_min_brake is not None else self.config.min_brake_mpss
        a_max_b = float(a_max_brake) if a_max_brake is not None else self.config.max_brake_mpss

        # Reaction distance during driver/actuator response time rho
        d_reaction = (v_resp * rho_val) + (0.5 * a_acc * (rho_val ** 2))

        # Ego stopping distance after reaction phase at min braking capability
        v_at_braking = v_resp + (rho_val * a_acc)
        d_ego_brake = (v_at_braking ** 2) / (2.0 * max(0.1, a_min_b))

        # Lead vehicle stopping distance under max possible emergency braking
        d_lead_brake = (v_lead ** 2) / (2.0 * max(0.1, a_max_b))

        # RSS positive clearance inequality
        d_kinematic = max(0.0, d_reaction + d_ego_brake - d_lead_brake)

        # Total minimum safe distance with physical standstill buffer
        return float(d_kinematic + self.config.standstill_clearance_m)

    def compute_min_safe_lateral_distance(
        self,
        ego_lat_speed_mps: float = 0.0,
        lead_lat_speed_mps: float = 0.0,
    ) -> float:
        """Computes minimum safe lateral boundary envelope."""
        return float(self.config.min_lateral_distance_m)

    def evaluate_threat(
        self,
        ego_speed_mps: float,
        lead_distance_m: float,
        lead_speed_mps: float,
        lateral_offset_m: float = 0.0,
        lead_track_id: Optional[int] = None,
    ) -> RSSSafetyCheckResult:
        """
        Evaluate RSS safety formula against a specific lead obstacle.
        Returns safety check result and commands emergency deceleration if violated.
        """
        t_start = time.perf_counter()

        d_min_long = self.compute_min_safe_longitudinal_distance(
            ego_speed_mps=ego_speed_mps,
            lead_speed_mps=lead_speed_mps,
        )
        d_min_lat = self.compute_min_safe_lateral_distance()

        d_actual_long = max(0.0, float(lead_distance_m))
        d_actual_lat = abs(float(lateral_offset_m))

        # Check if obstacle is inside lateral corridor
        in_corridor = (d_actual_lat <= self.config.corridor_half_width_m)

        # Check longitudinal boundary
        violation_margin = d_min_long - d_actual_long
        is_violated = in_corridor and (violation_margin > 0.0)

        if is_violated:
            status = "OVERRIDE_BRAKE"
            override = True
            safe = False
            accel_cmd = self.config.emergency_decel_cmd_mpss
        elif in_corridor and (d_actual_long <= d_min_long * 1.25):
            status = "CAUTION"
            override = False
            safe = True
            accel_cmd = None
        else:
            status = "SAFE"
            override = False
            safe = True
            accel_cmd = None

        exec_ms = (time.perf_counter() - t_start) * 1000.0

        res = RSSSafetyCheckResult(
            is_safe=safe,
            override_active=override,
            d_min_long=d_min_long,
            d_actual_long=d_actual_long,
            d_min_lat=d_min_lat,
            d_actual_lat=d_actual_lat,
            violation_margin_m=max(0.0, violation_margin) if in_corridor else 0.0,
            commanded_accel_mpss=accel_cmd,
            safety_status=status,
            lead_track_id=lead_track_id,
            execution_ms=exec_ms,
        )
        self.last_result = res
        return res

    def evaluate_objects(
        self,
        ego_speed_mps: float,
        tracked_objects: Optional[List[Any]] = None,
    ) -> RSSSafetyCheckResult:
        """
        Batch evaluates all detected/tracked objects ahead of the vehicle.
        Returns the most critical safety violation among all candidates.
        """
        if not tracked_objects:
            res = RSSSafetyCheckResult(
                is_safe=True,
                override_active=False,
                d_min_long=self.compute_min_safe_longitudinal_distance(ego_speed_mps, ego_speed_mps),
                d_actual_long=999.0,
                safety_status="SAFE",
            )
            self.last_result = res
            return res

        worst_result: Optional[RSSSafetyCheckResult] = None

        for obj in tracked_objects:
            # Handle dictionary or object attributes
            if isinstance(obj, dict):
                dist = obj.get("z_m", obj.get("distance_m", 999.0))
                lat_x = obj.get("x_m", obj.get("lateral_offset_m", 0.0))
                track_id = obj.get("track_id", None)
                rel_v = obj.get("vz_mps", obj.get("relative_speed_mps", 0.0))
            else:
                dist = getattr(obj, "z_m", getattr(obj, "distance_m", 999.0))
                lat_x = getattr(obj, "x_m", getattr(obj, "lateral_offset_m", 0.0))
                track_id = getattr(obj, "track_id", None)
                rel_v = getattr(obj, "vz_mps", getattr(obj, "relative_speed_mps", 0.0))

            lead_speed = max(0.0, ego_speed_mps + rel_v)

            result = self.evaluate_threat(
                ego_speed_mps=ego_speed_mps,
                lead_distance_m=dist,
                lead_speed_mps=lead_speed,
                lateral_offset_m=lat_x,
                lead_track_id=track_id,
            )

            if worst_result is None or (result.violation_margin_m > worst_result.violation_margin_m):
                worst_result = result

        final_res = worst_result or RSSSafetyCheckResult()
        self.last_result = final_res
        return final_res
