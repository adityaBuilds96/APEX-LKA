"""
src/perception/trajectory_predictor.py
=======================================
APEX-Vision Core v3.0 — CTRV Trajectory Prediction Engine (ISO 26262 Compliant)
- Employs Constant Turn Rate and Velocity (CTRV) non-linear motion physics.
- Predicts spatial trajectory coordinates (X_pred, Z_pred) at:
    t + 1.0s, t + 2.0s, t + 3.0s horizons.
- Evaluates Ego-Lane Intersection Risk: detects when target trajectory crosses
  ego-vehicle driving corridor within safety horizons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class TrajectoryWaypoint:
    """Predicted waypoint at future timestamp dt."""
    horizon_s: float
    x_m: float
    z_m: float
    speed_m_s: float
    yaw_rad: float
    is_in_ego_lane: bool


@dataclass
class PredictedTrajectory:
    """Full multi-horizon trajectory forecast for dynamic entity."""
    track_id: int
    class_name: str
    current_x: float
    current_z: float
    current_vx: float
    current_vz: float
    waypoints: List[TrajectoryWaypoint] = field(default_factory=list)
    intersects_ego_lane: bool = False
    earliest_intersection_s: Optional[float] = None


class CTRVTrajectoryPredictor:
    """
    Constant Turn Rate and Velocity (CTRV) Trajectory Prediction Engine.
    """

    def __init__(
        self,
        prediction_horizons: Tuple[float, ...] = (1.0, 2.0, 3.0),
        lane_half_width_m: float = 1.75,
    ) -> None:
        self.horizons = sorted(prediction_horizons)
        self.lane_half_w = float(lane_half_width_m)

    def predict_ctrv(
        self,
        x0: float,
        z0: float,
        vx0: float,
        vz0: float,
        yaw_rate_rad_s: float = 0.0,
        dt: float = 1.0,
    ) -> Tuple[float, float, float, float]:
        """
        Closed-form CTRV kinematic integration.
        State: [x, z, v, heading, yaw_rate]
        Returns:
            (x_pred, z_pred, v_pred, heading_pred)
        """
        # Forward speed v and heading theta
        v = math.sqrt(vx0**2 + vz0**2)
        theta = math.atan2(vx0, max(vz0, 1e-4))  # heading in ground plane

        omega = yaw_rate_rad_s

        if abs(omega) < 1e-4:
            # Straight-line constant velocity limit
            x_pred = x0 + vx0 * dt
            z_pred = z0 + vz0 * dt
            theta_pred = theta
        else:
            # Non-linear circular arc integration
            # x(t) = x0 + (v/omega) * (cos(theta) - cos(theta + omega*dt))
            # z(t) = z0 + (v/omega) * (sin(theta + omega*dt) - sin(theta))
            theta_pred = theta + omega * dt
            x_pred = x0 + (v / omega) * (math.cos(theta) - math.cos(theta_pred))
            z_pred = z0 + (v / omega) * (math.sin(theta_pred) - math.sin(theta))

        return x_pred, z_pred, v, theta_pred

    def predict_trajectory(
        self,
        track_id: int,
        x: float = 0.0,
        z: float = 0.0,
        vx: float = 0.0,
        vz: float = 0.0,
        yaw_rate_rad_s: float = 0.0,
        class_name: str = "vehicle",
        **kwargs: Any,
    ) -> PredictedTrajectory:
        """
        Generates full trajectory forecast over t + 1.0s, t + 2.0s, t + 3.0s.
        """
        x_val = float(kwargs.get("x0", x))
        z_val = float(kwargs.get("z0", z))
        vx_val = float(kwargs.get("vx0", vx))
        vz_val = float(kwargs.get("vz0", vz))
        x, z, vx, vz = x_val, z_val, vx_val, vz_val
        waypoints = []
        intersects = False
        earliest_t = None

        for dt in self.horizons:
            x_p, z_p, v_p, yaw_p = self.predict_ctrv(x, z, vx, vz, yaw_rate_rad_s, dt=dt)
            in_lane = abs(x_p) <= self.lane_half_w and z_p >= 0.0

            if in_lane and not intersects:
                intersects = True
                earliest_t = dt

            waypoints.append(
                TrajectoryWaypoint(
                    horizon_s=dt,
                    x_m=round(x_p, 3),
                    z_m=round(z_p, 3),
                    speed_m_s=round(v_p, 2),
                    yaw_rad=round(yaw_p, 3),
                    is_in_ego_lane=in_lane,
                )
            )

        return PredictedTrajectory(
            track_id=track_id,
            class_name=class_name,
            current_x=round(x, 3),
            current_z=round(z, 3),
            current_vx=round(vx, 3),
            current_vz=round(vz, 3),
            waypoints=waypoints,
            intersects_ego_lane=intersects,
            earliest_intersection_s=earliest_t,
        )
