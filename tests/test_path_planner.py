"""
tests/test_path_planner.py
==========================
Unit tests for Task 5.2: TEB / Metric BEV Local Path Planner.
"""

import math
import time
import pytest
import numpy as np

from src.planning.path_planner import (
    LocalPathPlanner,
    PathPlannerConfig,
    PlannedTrajectory,
    TrajectoryWaypoint,
)
from src.planning.behavior_fsm import BehaviorState, PlannerBehaviorContext
from src.perception.tracker import TrackedObject


def test_trajectory_waypoint_structure():
    """Verify planned trajectory generates exactly 30 waypoints over 3.0s horizon."""
    planner = LocalPathPlanner()
    traj = planner.plan(ego_speed_mps=3.0, target_speed_mps=4.17)

    assert len(traj) == 30
    assert len(traj.waypoints) == 30
    assert traj.waypoints[0].time_offset_s == pytest.approx(0.10, abs=1e-3)
    assert traj.waypoints[-1].time_offset_s == pytest.approx(3.0, abs=1e-3)
    # Forward distances must be strictly increasing
    xs = [wp.x for wp in traj.waypoints]
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))


def test_straight_road_centering():
    """Verify zero-curvature centerline produces straight trajectory with near-zero steering."""
    planner = LocalPathPlanner()
    # Lane center poly: Y(X) = 0.0
    traj = planner.plan(
        ego_speed_mps=3.0,
        target_speed_mps=4.0,
        lane_center_poly=[0.0, 0.0, 0.0],
    )

    assert abs(traj.target_steer_deg) < 1.0
    assert abs(traj.curvature) < 1e-3
    assert abs(traj.waypoints[-1].y) < 0.15


def test_curved_road_kinematic_steering():
    """Verify curved lane polynomial produces proportional curvature and steering angle."""
    planner = LocalPathPlanner()
    # Moderate left curve: Y(X) = -0.015 * X^2
    traj = planner.plan(
        ego_speed_mps=3.0,
        target_speed_mps=4.0,
        lane_center_poly=[-0.015, 0.0, 0.0],
    )

    # Waypoints should curve left (negative Y)
    assert traj.waypoints[-1].y < -0.5
    # Curvature should be negative
    assert traj.curvature < 0.0
    # Steering should be negative (left steer)
    assert traj.target_steer_deg < 0.0


def test_obstacle_repulsion_nudge():
    """Verify obstacle in path shifts trajectory away to maintain clearance."""
    planner = LocalPathPlanner()

    # Obstacle directly ahead at (x=0.0, z=5.0m)
    obs = TrackedObject(
        track_id=1,
        class_name="traffic_cone",
        bbox_3d=np.array([0.0, 0.0, 5.0]),
    )

    traj_clear = planner.plan(ego_speed_mps=3.0, target_speed_mps=3.0)
    traj_avoid = planner.plan(ego_speed_mps=3.0, target_speed_mps=3.0, tracked_objects=[obs])

    # Trajectory at 5.0m should be laterally shifted compared to clear road
    wp_clear_at_5 = min(traj_clear.waypoints, key=lambda w: abs(w.x - 5.0))
    wp_avoid_at_5 = min(traj_avoid.waypoints, key=lambda w: abs(w.x - 5.0))

    assert abs(wp_avoid_at_5.y - wp_clear_at_5.y) > 0.40


def test_tactical_lateral_bias_overtake():
    """Verify behavior context lateral bias applies smooth nudge across waypoints."""
    planner = LocalPathPlanner()
    b_ctx = PlannerBehaviorContext(
        state=BehaviorState.OVERTAKE_OBSTACLE,
        lateral_bias_m=-1.2,  # Left lane nudge
    )

    traj = planner.plan(ego_speed_mps=3.0, target_speed_mps=3.0, behavior_ctx=b_ctx)
    assert traj.waypoints[-1].y < -0.80  # Shifted left


def test_steering_slew_rate_limit():
    """Verify rapid steering changes are bounded by max_steering_rate (100 deg/s)."""
    planner = LocalPathPlanner()
    dt = 0.033

    # Step 1: straight
    planner.plan(ego_speed_mps=3.0, target_speed_mps=3.0, lane_center_poly=[0.0, 0.0, 0.0], dt=dt)

    # Step 2: sudden hard 25-degree curve demand
    traj_step2 = planner.plan(
        ego_speed_mps=3.0,
        target_speed_mps=3.0,
        lane_center_poly=[0.10, 0.0, 0.0],
        dt=dt,
    )

    max_allowed_delta = 100.0 * dt
    assert abs(traj_step2.target_steer_deg) <= (max_allowed_delta + 1e-2)


def test_planner_latency_benchmark():
    """Verify local path planner runs well within 8.0 ms cycle budget."""
    planner = LocalPathPlanner()
    obs = TrackedObject(track_id=1, bbox_3d=np.array([0.2, 0.0, 6.0]))

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        planner.plan(
            ego_speed_mps=3.5,
            target_speed_mps=4.17,
            lane_center_poly=[0.005, -0.01, 0.0],
            tracked_objects=[obs],
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p99 = float(np.percentile(latencies, 99))
    assert p99 < 8.0, f"Expected < 8.0 ms, got {p99:.2f} ms"
