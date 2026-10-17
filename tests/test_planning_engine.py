"""
tests/test_planning_engine.py
=============================
Integration test suite for APEX-LKA Phase 5 Section 5:
- Hierarchical Behavior FSM (6 states)
- Local Path Planner (30-waypoint cubic spline, TEB obstacle clearance, steering slew limit)
- Adaptive Cruise Control (IDM, curvature speed clamping, bounded jerk)
- Mobileye RSS Safety Shield (closed-form longitudinal clearance, override latency)
- CAN Frame 0x190 telemetry serialization and loopback
- End-to-end perception pipeline motion planning execution
"""

import math
import struct
import time
import numpy as np
import pytest

from src.planning.behavior_fsm import (
    BehaviorStateMachine,
    BehaviorState,
    BehaviorConfig,
    PlannerBehaviorContext,
)
from src.planning.path_planner import (
    LocalPathPlanner,
    PathPlannerConfig,
    PlannedTrajectory,
    TrajectoryWaypoint,
)
from src.planning.acc_controller import (
    AdaptiveCruiseController,
    ACCConfig,
    ACCControlOutput,
)
from src.planning.rss_shield import (
    MobileyeRSSShield,
    RSSConfig,
    RSSSafetyCheckResult,
)
from src.perception.tracker import TrackedObject
from src.perception.occupancy_grid import OccupancyGrid2D, CellState
from deploy.vehicle_comm import (
    CAN_ID_PLANNING_STATE,
    PlanningStateTelemetry,
    VehicleCommunicator,
)
from src.inference.pipeline import run_pipeline


# ── Test 1: Behavior FSM Initial State ───────────────────────────────────────
def test_behavior_fsm_initial_state():
    fsm = BehaviorStateMachine()
    ctx = fsm.update(ego_speed_mps=3.0, tracked_objects=[])
    assert ctx.state == BehaviorState.LANE_FOLLOWING
    assert ctx.target_speed_mps == pytest.approx(4.17, abs=0.1)
    assert not ctx.emergency_brake_active
    assert not ctx.obstacle_avoidance_active


# ── Test 2: Behavior FSM Transition to ACC ───────────────────────────────────
def test_behavior_fsm_transition_to_acc():
    fsm = BehaviorStateMachine()
    # Step once in lane following
    fsm.update(ego_speed_mps=3.5, tracked_objects=[])

    # Lead vehicle moving ahead at 8.0m with moderate speed (safe TTI > 1.8s)
    lead_obj = TrackedObject(
        track_id=1,
        class_name="car",
        bbox_2d=np.array([200, 100, 300, 200]),
        bbox_3d=np.array([0.0, 0.0, 8.0]),
        velocity_3d=np.array([0.0, 0.0, 2.5]),
    )
    # Advance multiple steps to exceed dwell time
    for _ in range(5):
        ctx = fsm.update(ego_speed_mps=3.5, tracked_objects=[lead_obj], dt=0.05)

    assert ctx.state == BehaviorState.ADAPTIVE_CRUISE
    assert ctx.target_speed_mps == pytest.approx(4.17, abs=0.1)


# ── Test 3: Behavior FSM Emergency Brake on RSS Override ────────────────────
def test_behavior_fsm_emergency_brake_on_rss_override():
    fsm = BehaviorStateMachine()
    ctx = fsm.update(ego_speed_mps=3.0, tracked_objects=[], rss_override=True)
    assert ctx.state == BehaviorState.EMERGENCY_BRAKE
    assert ctx.emergency_brake_active is True
    assert ctx.target_speed_mps == 0.0
    assert ctx.priority_score == 10.0


# ── Test 4: Behavior FSM Safe Stop on Lost Track Quality ─────────────────────
def test_behavior_fsm_safe_stop_on_lost_lanes():
    fsm = BehaviorStateMachine()
    # If lane is lost for multiple cycles (dwell threshold)
    for _ in range(15):
        ctx = fsm.update(
            ego_speed_mps=2.0,
            tracked_objects=[],
            lane_detected=False,  # Lost lane boundaries
            lane_retained=False,
            dt=0.1,
        )
    assert ctx.state == BehaviorState.SAFE_STOP
    assert ctx.target_speed_mps == 0.0


# ── Test 5: Behavior FSM Yield Obstacle ──────────────────────────────────────
def test_behavior_fsm_yield_obstacle():
    fsm = BehaviorStateMachine()
    # Centered static obstacle in corridor at 8m forward (< yield_distance_m 12m)
    obstacle = TrackedObject(
        track_id=7,
        class_name="obstacle",
        bbox_2d=np.array([220, 120, 280, 180]),
        bbox_3d=np.array([0.0, 0.0, 8.0]),
        velocity_3d=np.array([0.0, 0.0, 0.0]),
    )
    for _ in range(5):
        ctx = fsm.update(
            ego_speed_mps=2.0,
            tracked_objects=[obstacle],
            dt=0.05,
        )
    assert ctx.state == BehaviorState.YIELD_OBSTACLE
    assert ctx.obstacle_avoidance_active is True


# ── Test 6: Path Planner 30 Waypoints and 3.0s Horizon ────────────────────────
def test_path_planner_waypoint_count_and_horizon():
    planner = LocalPathPlanner()
    traj = planner.plan(ego_speed_mps=3.0, target_speed_mps=4.17)

    assert len(traj) == 30
    assert len(traj.waypoints) == 30
    assert traj.waypoints[0].time_offset_s == pytest.approx(0.10, abs=1e-3)
    assert traj.waypoints[-1].time_offset_s == pytest.approx(3.0, abs=1e-3)
    xs = [wp.x for wp in traj.waypoints]
    assert all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))
    assert traj.waypoints[-1].x > traj.waypoints[0].x


# ── Test 7: Path Planner Straight Corridor Centering ─────────────────────────
def test_path_planner_straight_corridor():
    planner = LocalPathPlanner()
    traj = planner.plan(
        ego_speed_mps=3.0,
        target_speed_mps=4.0,
        lane_center_poly=[0.0, 0.0, 0.0],
    )

    for wp in traj.waypoints:
        assert abs(wp.y) < 0.15
    assert abs(traj.curvature) < 1e-3
    assert abs(traj.target_steer_deg) < 1.0


# ── Test 8: Path Planner Curvature Following ─────────────────────────────────
def test_path_planner_curvature_following():
    planner = LocalPathPlanner()
    # Left curve: Y(X) = -0.015 * X^2
    traj = planner.plan(
        ego_speed_mps=3.0,
        target_speed_mps=4.0,
        lane_center_poly=[-0.015, 0.0, 0.0],
    )

    assert traj.curvature != 0.0
    assert abs(traj.target_steer_deg) > 1.0
    assert abs(traj.waypoints[-1].y) > 0.2


# ── Test 9: Path Planner Obstacle Repulsion ───────────────────────────────────
def test_path_planner_obstacle_repulsion():
    planner = LocalPathPlanner()

    # Create obstacle directly in corridor path at (x=0.0m lateral, z=5.0m forward)
    obs = TrackedObject(
        track_id=10,
        class_name="cone",
        bbox_2d=np.array([200, 100, 300, 200]),
        bbox_3d=np.array([0.0, 0.0, 5.0]),
        velocity_3d=np.array([0.0, 0.0, 0.0]),
    )

    traj_clear = planner.plan(ego_speed_mps=3.0, target_speed_mps=4.0, tracked_objects=[])
    traj_avoid = planner.plan(ego_speed_mps=3.0, target_speed_mps=4.0, tracked_objects=[obs])

    # Trajectory must deflect away from obstacle
    diff = max(abs(w_a.y - w_c.y) for w_a, w_c in zip(traj_avoid.waypoints, traj_clear.waypoints))
    assert diff > 0.05


# ── Test 10: Path Planner Steering Slew Rate Limiter ─────────────────────────
def test_path_planner_steering_slew_rate_limiter():
    config = PathPlannerConfig(max_steering_rate_degps=100.0)
    planner = LocalPathPlanner(config)

    # Initial plan straight
    planner.plan(ego_speed_mps=3.0, target_speed_mps=4.0, lane_center_poly=[0.0, 0.0, 0.0], dt=0.05)

    # Sharp curve requested
    sharp_poly = [-0.05, 0.0, 0.0]
    dt = 0.05  # 50 ms
    traj = planner.plan(ego_speed_mps=3.0, target_speed_mps=4.0, lane_center_poly=sharp_poly, dt=dt)

    max_delta = config.max_steering_rate_degps * dt  # 100 * 0.05 = 5.0 deg
    assert abs(traj.target_steer_deg) <= max_delta + 1e-4


# ── Test 11: ACC Free Road Acceleration ──────────────────────────────────────
def test_acc_controller_free_road_acceleration():
    acc = AdaptiveCruiseController()
    out = acc.update(ego_speed_mps=1.0, v_set_mps=4.17, lead_distance_m=None)
    assert out.commanded_accel_mpss > 0.0
    assert out.target_speed_mps > 1.0
    assert not out.is_lead_detected


# ── Test 12: ACC IDM Car-Following Deceleration ──────────────────────────────
def test_acc_controller_idm_car_following_deceleration():
    acc = AdaptiveCruiseController()
    # Ego at 3.0 m/s (target gap = 2.0 + 1.5 * 3.0 = 6.5m). Lead is at 3.5m (much closer).
    out = acc.update(
        ego_speed_mps=3.0,
        v_set_mps=4.17,
        lead_distance_m=3.5,
        lead_speed_mps=2.0,
        dt=0.05,
    )
    assert out.is_lead_detected is True
    assert out.commanded_accel_mpss < 0.0
    assert out.target_speed_mps < 3.0


# ── Test 13: ACC Curvature Speed Clamping ────────────────────────────────────
def test_acc_controller_curvature_speed_clamping():
    acc = AdaptiveCruiseController()
    # High curvature bend: kappa = 0.20 rad/m (radius 5m)
    # v_bend = sqrt(2.5 / 0.2) = 3.53 m/s < 4.17 m/s
    v_bend = acc.compute_curvature_speed_limit(curvature=0.20)
    assert v_bend < 4.17
    assert v_bend == pytest.approx(math.sqrt(2.5 / 0.20), rel=1e-2)

    out = acc.update(ego_speed_mps=4.0, v_set_mps=4.17, path_curvature=0.20)
    assert out.curvature_speed_limit_mps < 4.17
    assert out.target_speed_mps <= out.curvature_speed_limit_mps + 0.1


# ── Test 14: ACC Bounded Longitudinal Jerk ───────────────────────────────────
def test_acc_controller_bounded_jerk():
    acc = AdaptiveCruiseController()
    dt = 0.05
    # First step
    out1 = acc.update(ego_speed_mps=2.0, v_set_mps=4.17, dt=dt)
    a1 = out1.commanded_accel_mpss

    # Sudden request to brake hard
    out2 = acc.update(ego_speed_mps=2.0, v_set_mps=4.17, lead_distance_m=1.0, lead_speed_mps=0.0, dt=dt)
    a2 = out2.commanded_accel_mpss

    jerk = abs(a2 - a1) / dt
    assert jerk <= acc.config.max_jerk_mps3 + 1e-3


# ── Test 15: RSS Shield Safe Distance Speed Scaling ──────────────────────────
def test_rss_shield_safe_distance_scaling():
    shield = MobileyeRSSShield()
    d_low = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=2.0, lead_speed_mps=2.0)
    d_mid = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=4.17, lead_speed_mps=4.17)
    d_high = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=8.0, lead_speed_mps=8.0)

    assert d_low < d_mid < d_high
    assert d_low >= shield.config.standstill_clearance_m


# ── Test 16: RSS Shield Override Trigger and Latency ─────────────────────────
def test_rss_shield_override_trigger_and_latency():
    shield = MobileyeRSSShield()
    t0 = time.perf_counter()
    res = shield.evaluate_threat(
        ego_speed_mps=4.0,
        lead_distance_m=2.5,  # Unsafe close distance
        lead_speed_mps=0.0,
        lateral_offset_m=0.0,
    )
    lat_ms = (time.perf_counter() - t0) * 1000.0

    assert res.override_active is True
    assert res.is_safe is False
    assert res.commanded_accel_mpss == -4.5
    assert lat_ms < 1.0  # Must evaluate in < 1 ms


# ── Test 17: CAN 0x190 Encoding and Serialization ────────────────────────────
def test_can_0x190_encoding_and_roundtrip():
    comm = VehicleCommunicator(sim_mode=True)
    success = comm.send_planning_state(
        target_speed_mps=3.50,
        target_steer_deg=-12.45,
        behavior_state=1,  # ADAPTIVE_CRUISE
        rss_override=False,
        path_curvature=0.0125,
    )
    assert success is True

    telemetry = comm.get_planning_state()
    assert telemetry is not None
    assert telemetry.target_speed_mps == pytest.approx(3.50, abs=0.01)
    assert telemetry.target_steer_deg == pytest.approx(-12.45, abs=0.01)
    assert telemetry.behavior_state == 1
    assert telemetry.rss_override is False
    assert telemetry.path_curvature == pytest.approx(0.0125, abs=0.0002)

    # Direct struct byte validation: <HhBBh (8 bytes)
    raw = struct.pack("<HhBBh", 350, -1245, 1, 0, 125)
    assert len(raw) == 8
    v_s, d_s, fsm, rss, kap = struct.unpack("<HhBBh", raw)
    assert v_s == 350
    assert d_s == -1245
    assert fsm == 1
    assert rss == 0
    assert kap == 125


# ── Test 18: End-to-End Pipeline Motion Planning Integration ─────────────────
def test_pipeline_motion_planning_integration():
    frame = np.full((360, 640, 3), 70, dtype=np.uint8)
    # Paint lane boundaries
    frame[:, 200:205] = 255
    frame[:, 440:445] = 255

    res = run_pipeline(
        frame,
        backend="classical_cv",
        enable_planner=True,
    )

    assert res.planned_trajectory is not None
    assert len(res.planned_trajectory.waypoints) == 30
    assert res.behavior_context is not None
    assert res.behavior_context.state_name in ("LANE_FOLLOWING", "ADAPTIVE_CRUISE", "SAFE_STOP")
    assert res.acc_output is not None
    assert res.rss_result is not None
    assert "planning_ms" in res.timings
    assert res.timings["planning_ms"] <= 8.0
