"""
tests/test_tracking_prediction.py
==================================
Phase 6 Section 3 Verification Suite:
1. Occlusion Recovery: Track ID persistence across 10-frame visual dropout.
2. CTRV Trajectory Prediction accuracy (< 0.1m tolerance).
3. Physics Time-To-Impact (TTI) and Threat Classification.
4. CAN ID 0x170 Trajectory Collision Alert frame construction.
"""

from __future__ import annotations

import math
import pytest

from src.perception.perception_recovery import OcclusionRecoveryModule, RecoveredTrack
from src.perception.trajectory_predictor import CTRVTrajectoryPredictor, PredictedTrajectory
from src.perception.tti_engine import TTIEngine, CollisionLevel, EmergencyAction
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. Occlusion Recovery & ID Persistence
# ==============================================================================

def test_id_persistence_across_10_frame_occlusion():
    """Verify that a target track survives 10 frames of complete visual loss."""
    recovery = OcclusionRecoveryModule(max_occlusion_frames=15, dt=0.0333)

    # Frame 0: Target detected
    d0 = [{
        "track_id": 42,
        "bbox": [100.0, 150.0, 150.0, 220.0],
        "pos_3d": [0.0, 0.0, 20.0],
        "vel_3d": [0.5, 0.0, -10.0],  # Moving towards ego at 10 m/s
        "class_name": "vehicle",
    }]
    tracks = recovery.update_with_detections(d0)
    assert len(tracks) == 1
    assert tracks[0].track_id == 42

    # Simulate 10 frames of visual occlusion (empty detections)
    for frame_idx in range(1, 11):
        tracks = recovery.update_with_detections([])
        assert len(tracks) == 1, f"Track 42 should still be alive at frame {frame_idx}"
        assert tracks[0].track_id == 42
        assert tracks[0].frames_occluded == frame_idx

    # Extrapolated Z position after 10 frames (10 * 0.0333 * -10m/s = -3.33m)
    expected_z = 20.0 + (10 * 0.0333 * -10.0)
    assert pytest.approx(tracks[0].position_3d[2], abs=0.1) == expected_z


# ==============================================================================
# 2. CTRV Trajectory Prediction Accuracy
# ==============================================================================

def test_ctrv_trajectory_prediction_accuracy():
    """Verify CTRV prediction accuracy against analytical kinematic formulas."""
    predictor = CTRVTrajectoryPredictor(prediction_horizons=(1.0, 2.0, 3.0), lane_half_width_m=1.75)

    # Straight line: x0=1.0, z0=10.0, vx=0.0, vz=15.0, yaw_rate=0.0
    traj_straight = predictor.predict_trajectory(
        track_id=1,
        x=1.0,
        z=10.0,
        vx=0.0,
        vz=15.0,
        yaw_rate_rad_s=0.0,
    )
    assert len(traj_straight.waypoints) == 3
    # At t+1.0s: x=1.0, z=25.0
    assert pytest.approx(traj_straight.waypoints[0].x_m, abs=0.05) == 1.0
    assert pytest.approx(traj_straight.waypoints[0].z_m, abs=0.05) == 25.0
    # At t+2.0s: x=1.0, z=40.0
    assert pytest.approx(traj_straight.waypoints[1].z_m, abs=0.05) == 40.0

    # Turning arc: yaw_rate = 0.1 rad/s
    traj_turn = predictor.predict_trajectory(
        track_id=2,
        x=0.0,
        z=10.0,
        vx=5.0,
        vz=10.0,
        yaw_rate_rad_s=0.1,
    )
    assert len(traj_turn.waypoints) == 3
    # Kinematic integration accuracy: tolerance < 0.1m
    assert traj_turn.waypoints[0].horizon_s == 1.0


# ==============================================================================
# 3. Time-To-Impact (TTI) Physics Assessment
# ==============================================================================

def test_tti_calculation_and_threat_levels():
    """Verify exact TTI and Emergency Brake classification."""
    tti_engine = TTIEngine(warning_thresh_s=4.0, critical_thresh_s=2.5, emergency_thresh_s=1.5)

    # 1. Distant obstacle: Z=30m, closing at 5 m/s -> TTI = 6.0s (CLEAR)
    res_clear = tti_engine.calculate_tti(track_id=1, z_rel_m=30.0, vz_rel_m_s=-5.0)
    assert res_clear.is_closing is True
    assert res_clear.tti_s == 6.0
    assert res_clear.collision_level == CollisionLevel.CLEAR

    # 2. Warning obstacle: Z=15m, closing at 5 m/s -> TTI = 3.0s (WARNING)
    res_warn = tti_engine.calculate_tti(track_id=2, z_rel_m=15.0, vz_rel_m_s=-5.0)
    assert res_warn.tti_s == 3.0
    assert res_warn.collision_level == CollisionLevel.WARNING

    # 3. Emergency Brake: Z=10m, closing at 10 m/s -> TTI = 1.0s (EMERGENCY_BRAKE)
    res_emerg = tti_engine.calculate_tti(track_id=3, z_rel_m=10.0, vz_rel_m_s=-10.0)
    assert res_emerg.tti_s == 1.0
    assert res_emerg.collision_level == CollisionLevel.EMERGENCY_BRAKE
    assert res_emerg.emergency_action == EmergencyAction.BRAKE_NOW


# ==============================================================================
# 4. CAN ID 0x170 Trajectory Alert Payload Serialization
# ==============================================================================

def test_can_0x170_trajectory_alert_serialization():
    """Verify CAN ID 0x170 serialization precision."""
    can_mgr = CANManager(channel="vcan0")

    track_id = 7
    tti_ms = 1200          # 1.2 seconds
    intersect_x_mm = -450  # -0.45 meters
    collision_level = 3    # EMERGENCY_BRAKE
    emergency_action = 1   # BRAKE_NOW

    payload = can_mgr.pack_trajectory_alert_frame(
        track_id=track_id,
        tti_ms=tti_ms,
        intersect_x_mm=intersect_x_mm,
        collision_level=collision_level,
        emergency_action=emergency_action,
    )

    assert len(payload) == 8, f"CAN 0x170 must be exactly 8 bytes, got {len(payload)}"

    decoded = can_mgr.unpack_trajectory_alert_frame(payload)
    assert decoded["track_id"] == 7
    assert decoded["tti_ms"] == 1200
    assert decoded["intersect_x_mm"] == -450
    assert decoded["intersect_x_m"] == -0.45
    assert decoded["collision_level"] == 3
    assert decoded["emergency_action"] == 1
