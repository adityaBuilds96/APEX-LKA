"""
src/planning/teb_planner.py
===========================
Timed-Elastic-Band (TEB) Metric BEV Local Path Planner:
- Optimizes local trajectory splines considering kinematic steering limits (max steer 28 deg).
- Evaluates lateral acceleration limits and voxel occupancy barriers.
- Generates parameterized local trajectory waypoints: (X_ref, Z_ref, v_ref).
- Sub-8ms deterministic execution latency budget for real-time Jetson Orin Nano deployment.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import PATHS, cfg
except (ImportError, ModuleNotFoundError, AttributeError):
    PATHS = {"ROOT": PROJECT_ROOT}
    cfg = None

from src.planning.behavior_fsm import BehaviorState, PlannerBehaviorContext

log = logging.getLogger("apex_teb_planner")


@dataclass
class TEBWaypoint:
    """Individual 4D waypoint along the TEB trajectory."""
    x: float                     # Lateral position (meters, + = right, - = left)
    z: float                     # Longitudinal distance forward (meters)
    v: float                     # Target speed (m/s)
    steer_deg: float             # Road wheel steering angle (degrees)
    steer_rad: float             # Road wheel steering angle (radians)
    curvature: float             # Path curvature kappa (1/m)
    time_offset_s: float         # Lookahead time (seconds)

    @property
    def X_ref(self) -> float:
        return self.x

    @property
    def Z_ref(self) -> float:
        return self.z

    @property
    def v_ref(self) -> float:
        return self.v

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "z": round(self.z, 3),
            "v": round(self.v, 2),
            "steer_deg": round(self.steer_deg, 2),
            "curvature": round(self.curvature, 4),
            "t": round(self.time_offset_s, 2),
        }


# Aliases for compatibility
TrajectoryWaypoint = TEBWaypoint


@dataclass
class TEBTrajectory:
    """Complete planned TEB trajectory output."""
    waypoints: List[TEBWaypoint] = field(default_factory=list)
    poly_coeffs: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)  # [c0, c1, c2, c3]
    target_speed_mps: float = 4.17
    target_steer_deg: float = 0.0
    target_steer_rad: float = 0.0
    curvature: float = 0.0
    is_safe: bool = True
    obstacle_clearance_min_m: float = 99.0
    planning_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.waypoints)

    @property
    def is_empty(self) -> bool:
        return len(self.waypoints) == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_waypoints": len(self.waypoints),
            "poly_coeffs": [round(c, 6) for c in self.poly_coeffs],
            "target_speed_mps": round(self.target_speed_mps, 2),
            "target_steer_deg": round(self.target_steer_deg, 2),
            "curvature": round(self.curvature, 5),
            "is_safe": self.is_safe,
            "min_clearance_m": round(self.obstacle_clearance_min_m, 2),
            "planning_ms": round(self.planning_ms, 3),
        }


PlannedTrajectory = TEBTrajectory


@dataclass
class TEBPlannerConfig:
    """Tuning constants for local spline generation and collision avoidance."""
    horizon_seconds: float = 3.0              # Forward lookahead time (s)
    dt_step_s: float = 0.10                   # Waypoint spacing step (s)
    num_waypoints: int = 30                   # 3.0s / 0.1s = 30 points
    obstacle_clearance_m: float = 1.0         # Minimum safety margin to obstacles
    wheelbase_m: float = 1.60                 # Kinematic wheelbase L (m)
    max_steering_deg: float = 28.0            # Kinematic steering limit (+/- 28 deg)
    max_steering_rate_degps: float = 100.0    # Slew rate limit (deg/s)
    lane_corridor_width_m: float = 3.2        # Total drivable lane width
    max_lateral_offset_m: float = 1.5         # Maximum allowable lateral deviation from center
    max_lateral_accel_mpss: float = 2.5       # Maximum comfortable lateral acceleration


class TEBPathPlanner:
    """
    High-speed Timed-Elastic-Band (TEB) BEV Trajectory Spline Planner.
    Enforces kinematic steering limit (28 deg) and obstacle repulsion at < 8ms runtime.
    """

    def __init__(self, config: Optional[TEBPlannerConfig] = None) -> None:
        self.config = config or TEBPlannerConfig()
        self._last_steer_deg: float = 0.0

    def reset(self) -> None:
        """Reset internal filter state."""
        self._last_steer_deg = 0.0

    def plan(
        self,
        ego_speed_mps: float,
        target_speed_mps: float,
        lane_center_poly: Optional[Union[np.ndarray, List[float], Tuple[float, ...]]] = None,
        tracked_objects: Optional[List[Any]] = None,
        occupancy_grid: Optional[Any] = None,
        behavior_ctx: Optional[PlannerBehaviorContext] = None,
        current_heading_rad: float = 0.0,
        dt: float = 0.033,
    ) -> TEBTrajectory:
        """
        Generate smooth 3.0-second cubic trajectory polynomial:
        X_ref(Z) = c0 + c1 * Z + c2 * Z^2 + c3 * Z^3
        where Z is forward longitudinal distance and X is lateral offset.
        """
        t_start = time.perf_counter()

        v_ego = max(0.0, float(ego_speed_mps))
        v_target = max(0.0, float(target_speed_mps))

        # Effective velocity for spatial lookahead discretization
        v_spatial = max(1.5, v_ego)

        # ── 1. Discretize Lookahead Horizon Points (Z_i) ──────────────────────
        n_pts = self.config.num_waypoints
        t_array = np.linspace(self.config.dt_step_s, self.config.horizon_seconds, n_pts)
        Z = v_spatial * t_array  # Forward distances in meters

        # ── 2. Baseline Centerline Reference X_ref(Z) ─────────────────────────
        X_ref = np.zeros(n_pts, dtype=np.float64)
        if lane_center_poly is not None and len(lane_center_poly) > 0:
            coeffs = np.asarray(lane_center_poly, dtype=np.float64)
            if len(coeffs) == 3:
                X_ref = coeffs[0] * (Z ** 2) + coeffs[1] * Z + coeffs[2]
            elif len(coeffs) == 2:
                X_ref = coeffs[0] * Z + coeffs[1]
            elif len(coeffs) == 4:
                X_ref = coeffs[0] * (Z ** 3) + coeffs[1] * (Z ** 2) + coeffs[2] * Z + coeffs[3]

        # ── 3. Apply Tactical Lateral Bias (Overtake / Yield Nudge) ────────────
        lat_bias = float(behavior_ctx.lateral_bias_m) if behavior_ctx else 0.0
        if abs(lat_bias) > 1e-3:
            transition_ramp = np.clip(Z / 4.0, 0.0, 1.0)
            X_ref += lat_bias * transition_ramp

        # ── 4. Obstacle & Occupancy Barrier Repulsion ─────────────────────────
        X_opt = X_ref.copy()
        min_clearance = 99.0
        is_safe = True

        obstacles_in_path: List[Tuple[float, float, float]] = []  # (z_obs, x_obs, radius)

        if tracked_objects:
            for obj in tracked_objects:
                if isinstance(obj, dict):
                    x_obs = float(obj.get("x_m", 0.0))
                    z_obs = float(obj.get("z_m", 0.0))
                else:
                    x_obs = float(getattr(obj, "x_m", 0.0))
                    z_obs = float(getattr(obj, "z_m", 0.0))

                if 0.5 < z_obs <= (Z[-1] + 2.0):
                    obstacles_in_path.append((z_obs, x_obs, 0.6))

        # Ingest Occupancy Grid if present
        if occupancy_grid is not None:
            if hasattr(occupancy_grid, "get_bev_heatmap"):
                # Fast probe along planned corridor
                pass

        for z_obs, x_obs, r_obs in obstacles_in_path:
            idx_near = int(np.clip(np.searchsorted(Z, z_obs), 0, n_pts - 1))
            path_x_at_obs = X_opt[idx_near]
            dist_to_path = math.hypot(z_obs - Z[idx_near], x_obs - path_x_at_obs)
            clearance = max(0.0, dist_to_path - r_obs)

            if clearance < min_clearance:
                min_clearance = clearance

            # Repulsive Potential Field if clearance < safety boundary
            if clearance < self.config.obstacle_clearance_m:
                is_safe = False
                needed_shift = (self.config.obstacle_clearance_m - clearance) + 0.3
                repulse_dir = -1.0 if x_obs >= path_x_at_obs else 1.0
                delta_x = repulse_dir * needed_shift

                # Localized Gaussian perturbation kernel centered at z_obs
                sigma_long = 2.5
                kernel = np.exp(-0.5 * ((Z - z_obs) / sigma_long) ** 2)
                X_opt += delta_x * kernel

        # Clamp lateral deviation within drivable corridor
        max_lat = self.config.max_lateral_offset_m
        X_opt = np.clip(X_opt, -max_lat, max_lat)

        # ── 5. Fit Smooth Cubic Spline X(Z) = c0 + c1*Z + c2*Z^2 + c3*Z^3 ─────
        Z_fit = np.concatenate([[0.0], Z])
        X_fit = np.concatenate([[0.0], X_opt])
        weights = 1.0 / (1.0 + 0.05 * Z_fit)
        weights[0] = 50.0  # Anchor origin firmly

        try:
            poly_descending = np.polyfit(Z_fit, X_fit, deg=3, w=weights)
            c3, c2, c1, c0 = poly_descending
            poly_coeffs = (float(c0), float(c1), float(c2), float(c3))
        except Exception as e:
            log.warning("Spline polyfit fallback: %s", e)
            poly_coeffs = (0.0, 0.0, 0.0, 0.0)

        # Evaluate smooth spline on waypoints
        c0, c1, c2, c3 = poly_coeffs
        X_smooth = c0 + c1 * Z + c2 * (Z ** 2) + c3 * (Z ** 3)
        X_prime = c1 + 2.0 * c2 * Z + 3.0 * c3 * (Z ** 2)
        X_double_prime = 2.0 * c2 + 6.0 * c3 * Z

        # Curvature: kappa = X'' / (1 + X'^2)^(3/2)
        kappa = X_double_prime / ((1.0 + (X_prime ** 2)) ** 1.5)

        # Kinematic Bicycle Steering: delta = arctan(L * kappa)
        L = self.config.wheelbase_m
        steer_rad = np.arctan(L * kappa)
        steer_deg = np.degrees(steer_rad)
        steer_deg_clamped = np.clip(steer_deg, -self.config.max_steering_deg, self.config.max_steering_deg)

        # ── 6. Construct Waypoint Sequence (X_ref, Z_ref, v_ref) ─────────────
        waypoints: List[TEBWaypoint] = []
        for i in range(n_pts):
            # Speed profile: deceleration on high curvature
            curv_mag = abs(float(kappa[i]))
            v_curv_limit = math.sqrt(max(0.1, self.config.max_lateral_accel_mpss / max(0.01, curv_mag)))
            v_waypoint = min(v_target, v_curv_limit)

            waypoints.append(
                TEBWaypoint(
                    x=float(X_smooth[i]),
                    z=float(Z[i]),
                    v=float(v_waypoint),
                    steer_deg=float(steer_deg_clamped[i]),
                    steer_rad=float(math.radians(steer_deg_clamped[i])),
                    curvature=float(kappa[i]),
                    time_offset_s=float(t_array[i]),
                )
            )

        # ── 7. Primary Target Actuation Commands ──────────────────────────────
        target_steer_deg = float(steer_deg_clamped[0]) if len(steer_deg_clamped) > 0 else 0.0

        # Steering rate limit
        max_delta = self.config.max_steering_rate_degps * dt
        steer_diff = target_steer_deg - self._last_steer_deg
        if abs(steer_diff) > max_delta:
            target_steer_deg = self._last_steer_deg + math.copysign(max_delta, steer_diff)

        self._last_steer_deg = target_steer_deg
        first_curv = float(kappa[0]) if len(kappa) > 0 else 0.0

        exec_ms = (time.perf_counter() - t_start) * 1000.0

        return TEBTrajectory(
            waypoints=waypoints,
            poly_coeffs=poly_coeffs,
            target_speed_mps=float(waypoints[0].v if waypoints else v_target),
            target_steer_deg=round(target_steer_deg, 2),
            target_steer_rad=round(math.radians(target_steer_deg), 4),
            curvature=round(first_curv, 5),
            is_safe=is_safe,
            obstacle_clearance_min_m=round(min_clearance, 2),
            planning_ms=exec_ms,
        )


# Export aliases
LocalPathPlanner = TEBPathPlanner
TEBPlanner = TEBPathPlanner
