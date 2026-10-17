"""
tests/test_trajectory.py
========================
Unit tests for Task 3.2: Trajectory Prediction & Collision Physics Engine:
1. Future waypoint forecasting (t + 0.5s, 1.0s, 2.0s, 3.0s)
2. Time-To-Impact (TTI) physics calculation
3. Time-To-Cross-Path (TTCP) lateral corridor intersection
4. Multi-level collision risk assessment (CLEAR, MONITOR, WARNING, IMMINENT_COLLISION)
5. Action flags (BRAKE_NOW, SWERVE_LEFT, SWERVE_RIGHT)
6. Trajectory report structure and latency
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.projection_3d import Object3D
from src.perception.tracker import TrackedObject
from src.perception.trajectory import (
    ACTION_BRAKE_NOW,
    ACTION_NONE,
    ACTION_SWERVE_LEFT,
    ACTION_SWERVE_RIGHT,
    RISK_LEVEL_CLEAR,
    RISK_LEVEL_IMMINENT_COLLISION,
    RISK_LEVEL_MONITOR,
    RISK_LEVEL_WARNING,
    TrajectoryPredictor,
    TrajectoryReport,
)


def make_mock_tracked_object(
    track_id: int = 1,
    x: float = 0.0,
    z: float = 15.0,
    vx: float = 0.0,
    vz: float = -5.0,
    class_name: str = "vehicle",
) -> TrackedObject:
    """Helper to create TrackedObject with specified position and velocity."""
    box_2d = np.array([250.0, 150.0, 350.0, 250.0], dtype=np.float32)
    obj3d = Object3D(
        x=x,
        y=0.0,
        z=z,
        width=1.8,
        height=1.5,
        length=4.2,
        class_id=0,
        class_name=class_name,
        confidence=0.95,
        box_2d=box_2d,
    )
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        bbox_2d=box_2d,
        bbox_3d=obj3d,
        velocity_3d=np.array([vx, 0.0, vz], dtype=np.float32),
        age_frames=5,
        state="Tracked",
        confidence=0.95,
    )


def test_trajectory_waypoint_forecasting():
    """Verify future waypoints at 0.5s, 1.0s, 2.0s, 3.0s follow motion dynamics."""
    predictor = TrajectoryPredictor()
    obj = make_mock_tracked_object(track_id=4, x=2.0, z=20.0, vx=-1.0, vz=-4.0)

    report = predictor.predict([obj], ego_speed_mps=0.0)
    assert 4 in report.trajectories

    traj = report.trajectories[4]
    assert len(traj.waypoints) == 4
    # dt=0.5s -> x = 2.0 - 0.5 = 1.5, z = 20.0 - 2.0 = 18.0
    wp0 = traj.waypoints[0]
    assert abs(wp0.dt_s - 0.5) < 1e-3
    assert abs(wp0.x_m - 1.5) < 1e-2
    assert abs(wp0.z_m - 18.0) < 1e-2

    # dt=3.0s -> x = 2.0 - 3.0 = -1.0, z = 20.0 - 12.0 = 8.0
    wp3 = traj.waypoints[3]
    assert abs(wp3.dt_s - 3.0) < 1e-3
    assert abs(wp3.x_m - (-1.0)) < 1e-2
    assert abs(wp3.z_m - 8.0) < 1e-2


def test_tti_physics_head_on_approaching():
    """Verify exact Time-To-Impact computation for approaching obstacle directly ahead."""
    predictor = TrajectoryPredictor()
    # Obstacle directly ahead at Z=10m, closing at 5 m/s (vz = -5 m/s, ego = 0 m/s)
    # TTI = 10 / 5 = 2.0 seconds (2000 ms)
    obj = make_mock_tracked_object(track_id=1, x=0.0, z=10.0, vx=0.0, vz=-5.0)

    report = predictor.predict([obj], ego_speed_mps=0.0)
    traj = report.trajectories[1]

    assert abs(traj.tti_s - 2.0) < 0.05
    assert abs(report.min_tti_ms - 2000) < 50
    assert traj.ttcp_s == 0.0  # Already in corridor
    assert report.collision_warning_flag is True
    assert traj.risk_level == RISK_LEVEL_WARNING
    assert traj.action_flag == ACTION_BRAKE_NOW


def test_imminent_collision_and_emergency_brake():
    """Verify critical imminent collision triggers RISK_LEVEL_IMMINENT_COLLISION and BRAKE_NOW."""
    predictor = TrajectoryPredictor()
    # Vehicle at Z=3.0m directly ahead closing fast
    obj = make_mock_tracked_object(track_id=2, x=0.0, z=3.0, vx=0.0, vz=-5.0)

    report = predictor.predict([obj], ego_speed_mps=2.0)
    assert report.risk_level == RISK_LEVEL_IMMINENT_COLLISION
    assert report.emergency_brake_flag is True
    assert report.action_flag == ACTION_BRAKE_NOW


def test_cross_path_cut_in_and_evasive_action():
    """Verify cut-in obstacle approaching from right side triggers TTCP and SWERVE_LEFT."""
    predictor = TrajectoryPredictor(lane_corridor_half_width_m=1.0)
    # Obstacle at X=2.5m (outside lane corridor), moving left at vx = -1.5 m/s
    # Corridor edge is at X=1.0m, distance to edge = 1.5m -> TTCP = 1.5 / 1.5 = 1.0s
    obj = make_mock_tracked_object(track_id=3, x=2.5, z=12.0, vx=-1.5, vz=-5.0)

    report = predictor.predict([obj], ego_speed_mps=1.0)
    traj = report.trajectories[3]

    assert abs(traj.ttcp_s - 1.0) < 0.05
    # TTI = 12.0 / (5.0 + 1.0) = 2.0s
    assert abs(traj.tti_s - 2.0) < 0.10
    # Cutting in from right -> action should be SWERVE_LEFT or BRAKE
    assert traj.action_flag in (ACTION_SWERVE_LEFT, ACTION_BRAKE_NOW)


def test_trajectory_predictor_latency():
    """Verify trajectory prediction completes in < 1.0 ms for 15 active tracks."""
    predictor = TrajectoryPredictor()
    objs = [
        make_mock_tracked_object(
            track_id=i,
            x=float(i - 7) * 0.5,
            z=15.0 + float(i),
            vx=-0.2 * (i % 3),
            vz=-3.0,
        )
        for i in range(15)
    ]

    t0 = time.perf_counter()
    report = predictor.predict(objs, ego_speed_mps=5.0)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert latency_ms < 1.0, f"Trajectory prediction too slow: {latency_ms:.2f} ms"
    assert len(report.trajectories) == 15
