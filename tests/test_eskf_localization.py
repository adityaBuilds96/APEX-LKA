"""
tests/test_eskf_localization.py
================================
Phase 6 Section 4 Verification Suite:
1. 15-DOF ESKF Error Covariance P convergence over 500 simulated steps.
2. 5-second Kinematic Dead-Reckoning Trajectory Drift (< 0.35m).
3. CAN ID 0x180 Kinematic Vehicle State Serialization.
4. Hardware Sensor Bridge (IMU, Encoder, GPS) parsing precision.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from src.hardware.sensor_bridge import HardwareSensorBridge
from src.localization.eskf_fusion import ESKFFusionEngine, LocalizationStatus
from src.control.lane_retention import KinematicLaneRetentionEngine, LaneHoldState
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. 15-DOF ESKF Error Covariance Convergence
# ==============================================================================

def test_eskf_error_covariance_convergence():
    """Verify that ESKF error covariance P(15x15) converges over 500 simulated steps."""
    engine = ESKFFusionEngine(init_pos=(0.0, 0.0, 0.0))

    initial_pos_trace = np.trace(engine.eskf.P[0:3, 0:3])

    # Simulate 500 steps (5 seconds at 100 Hz) with IMU + Wheel Encoder + GNSS
    dt = 0.01
    for i in range(500):
        # 100 Hz IMU propagation (driving forward at 10 m/s with small noise)
        engine.predict(ax=0.0, ay=0.0, az=9.80665, wx=0.0, wy=0.0, wz=0.0, dt=dt)

        # 50 Hz Wheel Encoder update
        if i % 2 == 0:
            engine.correct_wheel_encoder(speed_m_s=10.0)

        # 10 Hz GNSS update
        if i % 10 == 0:
            sim_x = 0.0
            sim_y = (i * dt) * 10.0  # forward motion
            engine.eskf.update_gnss(x_m=sim_x, y_m=sim_y, z_m=0.0, std_m=0.5)

    final_state = engine.get_state()
    final_pos_trace = np.trace(engine.eskf.P[0:3, 0:3])

    # Covariance trace must decrease and converge
    assert final_pos_trace < initial_pos_trace, "Covariance trace must converge with multi-sensor updates"
    assert final_state.position_std_m < 0.50, f"Position uncertainty {final_state.position_std_m}m should be < 0.5m"
    assert pytest.approx(final_state.speed_m_s, abs=1.0) == 10.0


# ==============================================================================
# 2. 5-Second Dead-Reckoning Trajectory Drift (< 0.35m)
# ==============================================================================

def test_5s_kinematic_dead_reckoning_drift():
    """
    Validate that 5-second kinematic lane retention hold keeps lateral drift
    below 0.35m under nominal forward highway driving.
    """
    retention_engine = KinematicLaneRetentionEngine(max_hold_seconds=5.0)

    # Initial locked visual lane: 0.0m offset, 0.0 rad heading
    retention_engine.update_visual_lane(offset_m=0.0, heading_rad=0.0)

    # Simulate 5.0 seconds of blind hold at 100 Hz (dt = 0.01, 500 steps)
    # Forward velocity vx = 20 m/s (72 km/h), small sensor gyro yaw rate = 0.001 rad/s
    start_time = 1000.0
    dt = 0.01
    vx = 20.0
    wz = 0.001  # slight curvature

    for step in range(500):
        current_time = start_time + (step + 1) * dt
        hold_state = retention_engine.extrapolate_hold(vx_m_s=vx, yaw_rate_rad_s=wz, dt=dt, now=current_time)

    assert hold_state.is_active is True
    assert pytest.approx(hold_state.hold_duration_s, abs=0.05) == 5.0
    # Drift must be strictly < 0.35 meters
    assert abs(hold_state.extrapolated_offset_m) < 0.35, (
        f"Dead-reckoning drift ({hold_state.extrapolated_offset_m:.3f}m) exceeds 0.35m ceiling!"
    )
    assert hold_state.confidence > 0.10


# ==============================================================================
# 3. CAN ID 0x180 Kinematic Vehicle State Format
# ==============================================================================

def test_can_0x180_kinematic_state_serialization():
    """Verify CAN ID 0x180 byte format and precision."""
    can_mgr = CANManager(channel="vcan0")

    speed_cm_s = 2000       # 20.00 m/s (72 km/h)
    yaw_rate_mrad_s = -85   # -0.085 rad/s
    slip_angle_mrad = 12    # +0.012 rad
    localization_status = 3 # GNSS_FIX

    payload = can_mgr.pack_kinematic_state_frame(
        speed_cm_s=speed_cm_s,
        yaw_rate_mrad_s=yaw_rate_mrad_s,
        slip_angle_mrad=slip_angle_mrad,
        localization_status=localization_status,
    )

    assert len(payload) == 8, f"CAN 0x180 must be exactly 8 bytes, got {len(payload)}"

    decoded = can_mgr.unpack_kinematic_state_frame(payload)
    assert decoded["speed_cm_s"] == 2000
    assert decoded["speed_m_s"] == 20.0
    assert decoded["yaw_rate_mrad_s"] == -85
    assert decoded["slip_angle_mrad"] == 12
    assert decoded["localization_status"] == 3


# ==============================================================================
# 4. Hardware Sensor Bridge (IMU & CAN Encoders)
# ==============================================================================

def test_hardware_sensor_bridge_parsers():
    """Verify IMU and CAN encoder parsing."""
    bridge = HardwareSensorBridge(wheel_radius_m=0.28, ticks_per_rev=2048)

    # 1. IMU Parsing
    imu = bridge.parse_raw_imu(ax=0.1, ay=-0.2, az=9.81, wx=0.01, wy=-0.02, wz=0.05)
    assert imu.accel_m_s2 == (0.1, -0.2, 9.81)
    assert imu.gyro_rad_s == (0.01, -0.02, 0.05)

    # 2. CAN Encoder Frame Parsing (CAN ID 0x300, wheel 0, 4096 ticks, delta 20ms)
    import struct
    payload = struct.pack(">BiHB", 0, 4096, 20, 0)
    enc = bridge.parse_encoder_can_frame(arbitration_id=0x300, payload=payload)
    assert enc.wheel_id == 0
    assert enc.tick_count == 4096
    # 4096 ticks = 2 revolutions = 2 * 2 * pi * 0.28 = 3.518m
    assert pytest.approx(enc.distance_m, abs=0.1) == 3.52
