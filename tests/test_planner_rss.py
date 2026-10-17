"""
tests/test_planner_rss.py
=========================
Phase 6 Section 5 Verification Suite:
1. Mobileye RSS Safety Shield triggers override when lead car decelerates hard.
2. Hierarchical Behavior FSM transitions under simulated obstacle injection.
3. TEB Trajectory Generator execution latency < 8ms and kinematic steering limits <= 28 deg.
4. CAN ID 0x190 Autonomous Plan & Control State Serialization.
"""

from __future__ import annotations

import math
import time
import numpy as np
import pytest

from src.safety.rss_shield import MobileyeRSSShield, RSSConfig, RSSSafetyCheckResult
from src.planning.behavior_fsm import BehaviorStateMachine, BehaviorState, BehaviorConfig
from src.planning.teb_planner import TEBPathPlanner, TEBPlannerConfig
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. Mobileye RSS Safety Shield Hard Deceleration Override
# ==============================================================================

def test_rss_safety_shield_override_on_hard_braking():
    """
    Verify that Mobileye RSS Safety Shield activates RSS_OVERRIDE when a leading car
    is closer than d_min or decelerates sharply.
    """
    shield = MobileyeRSSShield()

    # Scenario: Ego traveling at 20 m/s (~72 km/h), Lead vehicle traveling at 15 m/s (~54 km/h)
    v_ego = 20.0
    v_lead = 15.0

    d_min = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=v_ego, lead_speed_mps=v_lead)

    # Lead car is comfortably ahead at d_min + 10m
    safe_res = shield.evaluate_threat(ego_speed_mps=v_ego, lead_distance_m=d_min + 10.0, lead_speed_mps=v_lead)
    assert safe_res.is_safe is True
    assert safe_res.override_active is False
    assert safe_res.RSS_OVERRIDE is False

    # Lead car suddenly slams brakes (lead speed drops to 2 m/s, distance compresses to 5m)
    violated_res = shield.evaluate_threat(ego_speed_mps=v_ego, lead_distance_m=5.0, lead_speed_mps=2.0)
    assert violated_res.is_safe is False
    assert violated_res.override_active is True
    assert violated_res.RSS_OVERRIDE is True
    assert violated_res.safety_status == "OVERRIDE_BRAKE"
    assert violated_res.commanded_accel_mpss is not None and violated_res.commanded_accel_mpss < 0.0


# ==============================================================================
# 2. Hierarchical Behavior FSM State Transitions
# ==============================================================================

def test_behavior_fsm_transitions_on_obstacle_injection():
    """
    Verify FSM state transitions:
    LANE_KEEPING -> ADAPTIVE_CRUISE -> OBSTACLE_AVOIDANCE -> EMERGENCY_STOP.
    """
    fsm = BehaviorStateMachine()

    class MockTrackedObstacle:
        def __init__(self, x_m: float, z_m: float, vx: float = 0.0, vz: float = 0.0):
            self.x_m = x_m
            self.z_m = z_m
            self.velocity_3d = np.array([vx, 0.0, vz], dtype=np.float64)

    # 1. Clear corridor -> LANE_KEEPING / LANE_FOLLOWING
    for _ in range(6):
        ctx = fsm.update(ego_speed_mps=10.0, tracked_objects=[], rss_override=False)
    assert ctx.state in (BehaviorState.LANE_KEEPING, BehaviorState.LANE_FOLLOWING)

    # 2. Moving lead car ahead at 18m -> ADAPTIVE_CRUISE
    lead_car = MockTrackedObstacle(x_m=0.0, z_m=18.0, vz=1.0)
    for _ in range(6):
        ctx = fsm.update(ego_speed_mps=10.0, tracked_objects=[lead_car], rss_override=False)
    assert ctx.state == BehaviorState.ADAPTIVE_CRUISE

    # 3. Static obstacle at 6m shifted +0.6m -> OBSTACLE_AVOIDANCE (or OVERTAKE_OBSTACLE)
    static_obs = MockTrackedObstacle(x_m=0.6, z_m=6.0, vz=0.0)
    for _ in range(6):
        ctx = fsm.update(ego_speed_mps=5.0, tracked_objects=[static_obs], rss_override=False)
    assert ctx.state in (BehaviorState.OBSTACLE_AVOIDANCE, BehaviorState.OVERTAKE_OBSTACLE)

    # 4. Sudden RSS Override -> Immediate EMERGENCY_STOP / EMERGENCY_BRAKE (zero dwell delay)
    ctx = fsm.update(ego_speed_mps=5.0, tracked_objects=[static_obs], rss_override=True)
    assert ctx.state in (BehaviorState.EMERGENCY_STOP, BehaviorState.EMERGENCY_BRAKE)
    assert ctx.emergency_brake_active is True


# ==============================================================================
# 3. TEB Trajectory Generator Latency (< 8ms) & 28-deg Steering Limits
# ==============================================================================

def test_teb_trajectory_latency_and_steering_limits():
    """
    Verify TEB local path planner executes in < 8ms and enforces maximum steering limit of 28 degrees.
    """
    planner = TEBPathPlanner(TEBPlannerConfig(max_steering_deg=28.0))

    ego_speed = 12.0
    target_speed = 15.0
    # Sharp lane curvature to stress-test steering limit
    sharp_poly = (0.015, 0.25, 0.1)

    # Measure execution latency over 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        trajectory = planner.plan(
            ego_speed_mps=ego_speed,
            target_speed_mps=target_speed,
            lane_center_poly=sharp_poly,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = float(np.mean(latencies))
    assert avg_latency < 8.0, f"TEB planning latency {avg_latency:.2f}ms exceeds 8ms ceiling!"
    assert len(trajectory.waypoints) == 30

    # Verify steering limit compliance across all waypoints
    for wp in trajectory.waypoints:
        assert abs(wp.steer_deg) <= 28.001, f"Steering angle {wp.steer_deg} exceeds 28 deg limit!"
        assert wp.Z_ref > 0.0
        assert wp.v_ref <= target_speed


# ==============================================================================
# 4. CAN ID 0x190 Autonomous Plan & Control State Serialization
# ==============================================================================

def test_can_0x190_control_state_serialization():
    """Verify CAN ID 0x190 packing and unpacking."""
    can_mgr = CANManager(channel="vcan0")

    target_speed_cm_s = 1500     # 15.00 m/s
    target_steer_cdeg = -1250    # -12.50 deg
    fsm_state = int(BehaviorState.ADAPTIVE_CRUISE)
    rss_override = 1             # Active override
    curvature_x10000 = 85        # kappa = 0.0085 1/m

    payload = can_mgr.pack_control_state_frame(
        target_speed_cm_s=target_speed_cm_s,
        target_steer_cdeg=target_steer_cdeg,
        fsm_state=fsm_state,
        rss_override=rss_override,
        curvature_x10000=curvature_x10000,
    )

    assert len(payload) == 8, f"CAN 0x190 frame must be 8 bytes, got {len(payload)}"

    decoded = can_mgr.unpack_control_state_frame(payload)
    assert decoded["target_speed_cm_s"] == 1500
    assert decoded["target_speed_m_s"] == 15.0
    assert decoded["target_steer_cdeg"] == -1250
    assert decoded["target_steer_deg"] == -12.50
    assert decoded["fsm_state"] == int(BehaviorState.ADAPTIVE_CRUISE)
    assert decoded["rss_override"] == 1
    assert decoded["curvature_x10000"] == 85
    assert pytest.approx(decoded["curvature"], abs=1e-5) == 0.0085
