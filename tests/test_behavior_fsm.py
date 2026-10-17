"""
tests/test_behavior_fsm.py
==========================
Unit tests for Task 5.1: Hierarchical Behavior State Machine.
"""

import pytest
import numpy as np

from src.planning.behavior_fsm import (
    BehaviorStateMachine,
    BehaviorState,
    BehaviorConfig,
    PlannerBehaviorContext,
)
from src.perception.tracker import TrackedObject


def test_initial_state_lane_following():
    """Verify default initial state is LANE_FOLLOWING at configured speed limit."""
    fsm = BehaviorStateMachine()
    ctx = fsm.update(ego_speed_mps=3.5, tracked_objects=[])

    assert ctx.state == BehaviorState.LANE_FOLLOWING
    assert ctx.target_speed_mps == pytest.approx(4.17, rel=1e-2)
    assert not ctx.emergency_brake_active
    assert not ctx.obstacle_avoidance_active


def test_rss_override_triggers_instantaneous_emergency_brake():
    """Verify RSS shield override immediately transitions to EMERGENCY_BRAKE with zero dwell delay."""
    fsm = BehaviorStateMachine()
    ctx = fsm.update(ego_speed_mps=3.5, tracked_objects=[], rss_override=True)

    assert ctx.state == BehaviorState.EMERGENCY_BRAKE
    assert ctx.target_speed_mps == 0.0
    assert ctx.emergency_brake_active is True
    assert ctx.priority_score == 10.0


def test_critical_tti_triggers_emergency_brake():
    """Verify closing obstacle with critical TTI triggers emergency braking."""
    fsm = BehaviorStateMachine()

    # Lead vehicle 4.0m ahead, ego 4.0 m/s closing -> TTI = 1.0s (< 1.8s threshold)
    lead_obj = TrackedObject(
        track_id=1,
        class_name="vehicle",
        bbox_2d=np.array([200, 100, 300, 200]),
        bbox_3d=np.array([0.0, 0.0, 4.0]),
        velocity_3d=np.array([0.0, 0.0, 0.0]),
    )

    ctx = fsm.update(ego_speed_mps=4.0, tracked_objects=[lead_obj])
    assert ctx.state == BehaviorState.EMERGENCY_BRAKE
    assert ctx.emergency_brake_active is True


def test_moving_lead_vehicle_adaptive_cruise_transition():
    """Verify dynamic lead vehicle triggers ADAPTIVE_CRUISE state after dwell time."""
    config = BehaviorConfig(min_state_dwell_frames=2)
    fsm = BehaviorStateMachine(config=config)

    # Lead vehicle at 15.0m moving at 3.0 m/s
    lead_obj = TrackedObject(
        track_id=2,
        class_name="vehicle",
        bbox_2d=np.array([200, 100, 300, 200]),
        bbox_3d=np.array([0.1, 0.0, 15.0]),
        velocity_3d=np.array([0.0, 0.0, 3.0]),
    )

    # Frame 1 (dwell)
    fsm.update(ego_speed_mps=3.0, tracked_objects=[lead_obj])
    # Frame 2 (dwell met)
    ctx = fsm.update(ego_speed_mps=3.0, tracked_objects=[lead_obj])

    assert ctx.state == BehaviorState.ADAPTIVE_CRUISE
    assert ctx.lead_distance_m == pytest.approx(15.0, abs=0.2)


def test_static_obstacle_overtake_with_lateral_nudge():
    """Verify off-center static obstacle triggers OVERTAKE_OBSTACLE with opposite lateral bias."""
    config = BehaviorConfig(min_state_dwell_frames=1)
    fsm = BehaviorStateMachine(config=config)

    # Static obstacle at 5.0m ahead, shifted right by +0.6m
    obs = TrackedObject(
        track_id=3,
        class_name="traffic_cone",
        bbox_2d=np.array([250, 150, 320, 220]),
        bbox_3d=np.array([0.6, 0.0, 5.0]),
        velocity_3d=np.array([0.0, 0.0, 0.0]),
    )

    ctx = fsm.update(ego_speed_mps=2.0, tracked_objects=[obs])
    assert ctx.state == BehaviorState.OVERTAKE_OBSTACLE
    assert ctx.obstacle_avoidance_active is True
    # Should nudge left (- lateral bias)
    assert ctx.lateral_bias_m < 0.0


def test_safe_stop_behind_obstacle():
    """Verify near-zero speed behind obstacle triggers SAFE_STOP."""
    config = BehaviorConfig(min_state_dwell_frames=1)
    fsm = BehaviorStateMachine(config=config)

    obs = TrackedObject(
        track_id=4,
        class_name="barrier",
        bbox_2d=np.array([200, 150, 300, 250]),
        bbox_3d=np.array([0.0, 0.0, 2.2]),
        velocity_3d=np.array([0.0, 0.0, 0.0]),
    )

    ctx = fsm.update(ego_speed_mps=0.2, tracked_objects=[obs])
    assert ctx.state == BehaviorState.SAFE_STOP
    assert ctx.target_speed_mps == 0.0


def test_fsm_reset():
    """Verify reset() restores state to initial nominal tracking."""
    fsm = BehaviorStateMachine()
    fsm.update(ego_speed_mps=3.0, tracked_objects=[], rss_override=True)
    assert fsm.current_state == BehaviorState.EMERGENCY_BRAKE

    fsm.reset()
    assert fsm.current_state == BehaviorState.LANE_FOLLOWING
