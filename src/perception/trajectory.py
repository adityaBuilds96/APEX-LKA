"""
src/perception/trajectory.py
============================
Trajectory Prediction & Time-To-Impact (TTI) Collision Physics Engine for APEX-LKA:
- Projects future 3D positions (X_{t+k}, Z_{t+k}) at 0.5s, 1.0s, 2.0s, 3.0s horizons
  using Constant Acceleration (CA) and Constant Velocity motion dynamics.
- Computes exact Time-To-Impact (TTI) and Time-To-Cross-Path (TTCP).
- Evaluates collision risk levels (0=CLEAR, 1=MONITOR, 2=WARNING, 3=IMMINENT_COLLISION).
- Determines evasive emergency action flags (0=NONE, 1=BRAKE_NOW, 2=SWERVE_LEFT, 3=SWERVE_RIGHT).
- Generates TrajectoryReport ready for CAN ID 0x170 and HUD motion vector overlays.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from enum import IntEnum
from src.perception.tracker import TrackedObject

log = logging.getLogger("apex_trajectory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Collision Risk Levels (matches CAN ID 0x170 schema)
class CollisionRiskLevel(IntEnum):
    CLEAR              = 0
    MONITOR            = 1
    WARNING            = 2
    IMMINENT_COLLISION = 3

RISK_LEVEL_CLEAR              = CollisionRiskLevel.CLEAR.value
RISK_LEVEL_MONITOR            = CollisionRiskLevel.MONITOR.value
RISK_LEVEL_WARNING            = CollisionRiskLevel.WARNING.value
RISK_LEVEL_IMMINENT_COLLISION = CollisionRiskLevel.IMMINENT_COLLISION.value

# Evasive Action Flags (matches CAN ID 0x170 schema)
class EvasiveActionFlag(IntEnum):
    NONE         = 0
    BRAKE_NOW    = 1
    SWERVE_LEFT  = 2
    SWERVE_RIGHT = 3

ACTION_NONE         = EvasiveActionFlag.NONE.value
ACTION_BRAKE_NOW    = EvasiveActionFlag.BRAKE_NOW.value
ACTION_SWERVE_LEFT  = EvasiveActionFlag.SWERVE_LEFT.value
ACTION_SWERVE_RIGHT = EvasiveActionFlag.SWERVE_RIGHT.value


# ═══════════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PredictedWaypoint:
    """Predicted future 3D waypoint coordinate at delta-t."""
    dt_s: float
    x_m: float
    z_m: float

    @property
    def t_offset_s(self) -> float:
        return self.dt_s

    @property
    def x(self) -> float:
        return self.x_m

    @property
    def z(self) -> float:
        return self.z_m


@dataclass
class PredictedTrajectory:
    """Predicted path, kinematics, and collision risks for a tracked object."""
    track_id: int
    class_id: int = 0
    class_name: str = "vehicle"
    current_x: float = 0.0
    current_z: float = 0.0
    vx: float = 0.0
    vz: float = 0.0
    waypoints: List[PredictedWaypoint] = field(default_factory=list)
    tti_s: float = float("inf")
    ttcp_s: float = float("inf")
    risk_level: int = RISK_LEVEL_CLEAR
    action_flag: int = ACTION_NONE
    projected_cross_x_m: float = 0.0

    @property
    def velocity_vx(self) -> float:
        return self.vx

    @property
    def velocity_vz(self) -> float:
        return self.vz

    @property
    def collision_risk(self) -> CollisionRiskLevel:
        return CollisionRiskLevel(self.risk_level)

    @property
    def evasive_action(self) -> EvasiveActionFlag:
        return EvasiveActionFlag(self.action_flag)

    @property
    def tti_ms(self) -> int:
        if math.isinf(self.tti_s) or self.tti_s < 0:
            return 65535
        return int(min(65535, round(self.tti_s * 1000.0)))

    @property
    def projected_cross_x_mm(self) -> int:
        return int(np.clip(round(self.projected_cross_x_m * 1000.0), -32768, 32767))


@dataclass
class TrajectoryReport:
    """Comprehensive multi-object trajectory and collision safety report."""
    trajectories: Dict[int, PredictedTrajectory] = field(default_factory=dict)
    critical_track_id: Optional[int] = None
    min_tti_ms: int = 65535
    collision_warning_flag: bool = False
    emergency_brake_flag: bool = False
    risk_level: int = RISK_LEVEL_CLEAR
    action_flag: int = ACTION_NONE
    projected_cross_x_mm: int = 0
    prediction_ms: float = 0.0
    timestamp_s: float = 0.0

    @property
    def min_tti_s(self) -> float:
        if self.min_tti_ms >= 65535:
            return float("inf")
        return self.min_tti_ms / 1000.0

    @property
    def projected_cross_x_m(self) -> float:
        return self.projected_cross_x_mm / 1000.0

    @property
    def predicted_trajectories(self) -> List[PredictedTrajectory]:
        return list(self.trajectories.values())

    @property
    def max_risk_level(self) -> CollisionRiskLevel:
        return CollisionRiskLevel(self.risk_level)

    @property
    def recommended_action(self) -> EvasiveActionFlag:
        return EvasiveActionFlag(self.action_flag)

    @property
    def predicted_paths(self) -> Dict[int, List[Tuple[float, float]]]:
        """Convenience map of track_id -> List[(x_m, z_m)] for HUD overlay."""
        paths = {}
        for tid, traj in self.trajectories.items():
            paths[tid] = [(wp.x_m, wp.z_m) for wp in traj.waypoints]
        return paths

    def to_dict(self) -> Dict[str, Any]:
        return {
            "critical_track_id": self.critical_track_id,
            "min_tti_ms": self.min_tti_ms,
            "collision_warning": self.collision_warning_flag,
            "emergency_brake": self.emergency_brake_flag,
            "risk_level": self.risk_level,
            "action_flag": self.action_flag,
            "projected_cross_x_mm": self.projected_cross_x_mm,
            "prediction_ms": round(self.prediction_ms, 2),
            "num_tracks_projected": len(self.trajectories),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Trajectory Predictor Engine
# ═══════════════════════════════════════════════════════════════════════════

class TrajectoryPredictor:
    """
    Constant Acceleration / Constant Velocity Trajectory Predictor & Collision Engine.
    Evaluates closing physics, corridor cross paths, and calculates TTI.
    """

    def __init__(
        self,
        config_or_path: Optional[Union[str, Path, Dict[str, Any]]] = None,
        horizon_seconds: float = 3.0,
        tti_warning_threshold_s: float = 2.5,
        tti_critical_threshold_s: float = 1.0,
        lane_corridor_half_width_m: float = 1.0,
        horizon_s: Optional[float] = None,
        warning_tti_s: Optional[float] = None,
        critical_tti_s: Optional[float] = None,
        ego_lane_width_m: Optional[float] = None,
    ) -> None:
        cfg = self._load_config(config_or_path)
        h_s = horizon_s if horizon_s is not None else cfg.get("horizon_seconds", horizon_seconds)
        self.horizon_seconds = float(h_s)

        w_tti = warning_tti_s if warning_tti_s is not None else cfg.get("tti_warning_threshold_s", tti_warning_threshold_s)
        self.tti_warning_threshold_s = float(w_tti)

        c_tti = critical_tti_s if critical_tti_s is not None else cfg.get("tti_critical_threshold_s", tti_critical_threshold_s)
        self.tti_critical_threshold_s = float(c_tti)

        half_w = (ego_lane_width_m / 2.0) if ego_lane_width_m is not None else cfg.get("lane_corridor_half_width_m", lane_corridor_half_width_m)
        self.corridor_w = float(half_w)

        self.forecast_horizons = [0.5, 1.0, 2.0, 3.0]

    def _load_config(self, config_or_path: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config_or_path, dict):
            return config_or_path.get("trajectory_prediction", config_or_path)
        if config_or_path is not None:
            p = Path(config_or_path)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                with open(p, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
                    return data.get("trajectory_prediction", data)

        default_p = PROJECT_ROOT / "configs" / "project_config.yaml"
        if default_p.exists():
            with open(default_p, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("trajectory_prediction", data)
        return {}

    def predict(
        self,
        tracked_objects: List[TrackedObject],
        ego_speed_mps: float = 0.0,
        lane_bounds: Optional[Tuple[float, float]] = None,
        timestamp_s: Optional[float] = None,
    ) -> TrajectoryReport:
        """
        Compute future trajectories, TTI, and collision risks for all tracked objects.

        Parameters
        ----------
        tracked_objects : List[TrackedObject]
            Active tracks from APEXMultiObjectTracker.
        ego_speed_mps : float
            Current ego vehicle longitudinal speed (m/s).
        lane_bounds : Optional[Tuple[float, float]]
            Drivable corridor [x_left_m, x_right_m]. Default is [-1.0m, +1.0m].

        Returns
        -------
        TrajectoryReport
        """
        t0 = time.perf_counter()

        x_min_corridor = lane_bounds[0] if lane_bounds else -self.corridor_w
        x_max_corridor = lane_bounds[1] if lane_bounds else self.corridor_w

        trajectories: Dict[int, PredictedTrajectory] = {}

        min_tti_s = float("inf")
        crit_track_id: Optional[int] = None
        global_risk = RISK_LEVEL_CLEAR
        global_action = ACTION_NONE
        global_cross_x_mm = 0

        for obj in tracked_objects:
            x0 = obj.x_m
            z0 = obj.z_m
            vx = float(obj.velocity_3d[0])
            vz = float(obj.velocity_3d[2])

            # Waypoint forecasting: X(t + dt) = X0 + vx*dt, Z(t + dt) = Z0 + vz*dt
            waypoints: List[PredictedWaypoint] = []
            for dt in self.forecast_horizons:
                x_pred = x0 + vx * dt
                z_pred = z0 + vz * dt
                waypoints.append(PredictedWaypoint(dt_s=dt, x_m=x_pred, z_m=z_pred))

            # 1. Closing Speed Calculation
            # In camera frame, positive Z is forward. An approaching obstacle has vz < 0 (z decreasing).
            # Ego vehicle traveling forward at ego_speed_mps adds to closing rate.
            closing_speed = (-vz) + max(0.0, ego_speed_mps)

            # 2. Time-To-Impact (TTI)
            bumper_z = 0.0  # Front bumper position
            if z0 > bumper_z and closing_speed > 0.10:
                tti = (z0 - bumper_z) / closing_speed
            else:
                tti = float("inf")

            # 3. Time-To-Cross-Path (TTCP)
            # Time when lateral position X intersects ego corridor [x_min_corridor, x_max_corridor]
            ttcp = float("inf")
            projected_cross_x = x0

            in_corridor = (x_min_corridor <= x0 <= x_max_corridor)
            if in_corridor:
                ttcp = 0.0
                projected_cross_x = x0
            elif x0 > x_max_corridor and vx < -0.10:
                # Approaching from right to left into lane
                ttcp = (x0 - x_max_corridor) / (-vx)
                projected_cross_x = x0 + vx * min(tti if not math.isinf(tti) else 2.0, ttcp)
            elif x0 < x_min_corridor and vx > 0.10:
                # Approaching from left to right into lane
                ttcp = (x_min_corridor - x0) / vx
                projected_cross_x = x0 + vx * min(tti if not math.isinf(tti) else 2.0, ttcp)
            else:
                # Moving parallel or away
                projected_cross_x = x0 + vx * (tti if not math.isinf(tti) and tti < 3.0 else 1.0)

            # 4. Collision Risk Level & Evasive Action
            risk = RISK_LEVEL_CLEAR
            action = ACTION_NONE

            # Critical imminent collision
            if (tti <= self.tti_critical_threshold_s and ttcp <= 1.0) or (in_corridor and z0 <= 3.5 and closing_speed > 0):
                risk = RISK_LEVEL_IMMINENT_COLLISION
                action = ACTION_BRAKE_NOW
            elif tti <= self.tti_warning_threshold_s and ttcp <= (tti + 0.5):
                risk = RISK_LEVEL_WARNING
                if in_corridor:
                    action = ACTION_BRAKE_NOW
                elif x0 > x_max_corridor:
                    action = ACTION_SWERVE_LEFT  # Obstacle cutting in from right -> avoid left
                else:
                    action = ACTION_SWERVE_RIGHT # Obstacle cutting in from left -> avoid right
            elif tti <= self.horizon_seconds or (ttcp <= self.horizon_seconds and z0 <= 15.0):
                risk = RISK_LEVEL_MONITOR
                action = ACTION_NONE

            traj = PredictedTrajectory(
                track_id=obj.track_id,
                class_id=obj.class_id,
                class_name=obj.class_name,
                current_x=x0,
                current_z=z0,
                vx=vx,
                vz=vz,
                waypoints=waypoints,
                tti_s=tti,
                ttcp_s=ttcp,
                risk_level=risk,
                action_flag=action,
                projected_cross_x_m=projected_cross_x,
            )
            trajectories[obj.track_id] = traj

            # Track global critical event
            if tti < min_tti_s and risk >= RISK_LEVEL_WARNING:
                min_tti_s = tti
                crit_track_id = obj.track_id
                global_risk = max(global_risk, risk)
                global_action = action
                global_cross_x_mm = traj.projected_cross_x_mm
            elif crit_track_id is None and risk >= RISK_LEVEL_MONITOR:
                crit_track_id = obj.track_id
                min_tti_s = min(min_tti_s, tti)
                global_risk = max(global_risk, risk)
                global_cross_x_mm = traj.projected_cross_x_mm

        # Final report assembly
        min_tti_ms = int(min(65535, round(min_tti_s * 1000.0))) if not math.isinf(min_tti_s) else 65535
        collision_warn = (global_risk >= RISK_LEVEL_WARNING)
        emergency_brake = (global_risk == RISK_LEVEL_IMMINENT_COLLISION or global_action == ACTION_BRAKE_NOW)

        pred_ms = (time.perf_counter() - t0) * 1000.0

        return TrajectoryReport(
            trajectories=trajectories,
            critical_track_id=crit_track_id,
            min_tti_ms=min_tti_ms,
            collision_warning_flag=collision_warn,
            emergency_brake_flag=emergency_brake,
            risk_level=global_risk,
            action_flag=global_action,
            projected_cross_x_mm=global_cross_x_mm,
            prediction_ms=pred_ms,
        )
