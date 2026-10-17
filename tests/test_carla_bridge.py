"""
tests/test_carla_bridge.py
==========================
Unit tests for APEX-LKA CARLA Simulator Bridge Protocol.
Verifies synchronous stepping, virtual sensor retrieval, CAN emulation,
actuation injection, and deterministic mock execution.
"""

import time
import numpy as np
import pytest

from src.simulation.carla_bridge import (
    CARLABridge,
    Transform3D,
    IMUData,
    GNSSData,
    WheelEncoderData,
    VehicleState,
)


@pytest.fixture
def sim_bridge():
    """Provides a clean mock simulation bridge instance."""
    bridge = CARLABridge(mock_mode=True, synchronous=True)
    assert bridge.connect() is True
    bridge.spawn_ego_vehicle()
    bridge.setup_sensors(enable_depth=True, enable_semantic=True)
    yield bridge
    bridge.cleanup()


def test_carla_bridge_connection_and_mock_fallback(sim_bridge):
    """Verify bridge connects and initializes mock world cleanly."""
    assert sim_bridge.is_connected() is True
    assert sim_bridge.is_mock_world is True


def test_ego_vehicle_spawning_and_state(sim_bridge):
    """Verify ego vehicle spawns and reports valid kinematics."""
    state = sim_bridge.get_vehicle_state()
    assert isinstance(state, VehicleState)
    assert isinstance(state.transform, Transform3D)
    assert state.speed_mps >= 0.0
    assert len(state.velocity_mps) == 3


def test_synchronous_stepping_and_camera_frames(sim_bridge):
    """Verify tick() advances simulation and generates 640x360 RGB frames."""
    for _ in range(5):
        sim_bridge.tick()

    frame = sim_bridge.get_camera_frame()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (360, 640, 3)
    assert frame.dtype == np.uint8


def test_depth_and_semantic_sensor_retrieval(sim_bridge):
    """Verify depth and semantic segmentation sensors produce valid arrays."""
    sim_bridge.tick()
    depth = sim_bridge.get_depth_frame()
    semantic = sim_bridge.get_semantic_mask()

    assert depth is not None
    assert depth.shape == (360, 640)
    assert depth.dtype == np.float32

    assert semantic is not None
    assert semantic.shape == (360, 640)
    assert semantic.dtype == np.uint8


def test_imu_and_gnss_sensor_telemetry(sim_bridge):
    """Verify 6-DOF IMU and GNSS data streams."""
    sim_bridge.tick()
    imu = sim_bridge.get_imu_data()
    gnss = sim_bridge.get_gnss_data()

    assert isinstance(imu, IMUData)
    assert len(imu.accelerometer) == 3
    assert len(imu.gyroscope) == 3

    assert isinstance(gnss, GNSSData)
    assert 10.0 <= gnss.latitude <= 20.0
    assert 70.0 <= gnss.longitude <= 85.0
    assert gnss.altitude > 0.0


def test_wheel_encoder_tick_accumulation(sim_bridge):
    """Verify virtual wheel tick counts increment with vehicle motion."""
    sim_bridge.tick()
    enc1 = sim_bridge.get_wheel_encoder_data()
    assert isinstance(enc1, WheelEncoderData)
    ticks1 = enc1.left_ticks

    # Step forward with positive throttle
    sim_bridge.apply_control(steer=0.0, throttle=0.5, brake=0.0)
    for _ in range(10):
        sim_bridge.tick()

    enc2 = sim_bridge.get_wheel_encoder_data()
    assert enc2.left_ticks >= ticks1
    assert enc2.linear_speed_mps >= 0.0


def test_actuation_command_injection_and_clamping(sim_bridge):
    """Verify steering/throttle/brake inputs are clamped to physical bounds."""
    # Exceed bounds to test clipping
    sim_bridge.apply_control(steer=2.5, throttle=-0.5, brake=1.5)
    assert sim_bridge._sim_steer == 1.0
    assert sim_bridge._sim_throttle == 0.0
    assert sim_bridge._sim_brake == 1.0

    sim_bridge.apply_control(steer=-3.0, throttle=0.8, brake=0.2)
    assert sim_bridge._sim_steer == -1.0
    assert sim_bridge._sim_throttle == 0.8
    assert sim_bridge._sim_brake == 0.2


def test_carla_bridge_cleanup(sim_bridge):
    """Verify bridge cleanup safely resets handles without raising."""
    sim_bridge.cleanup()
    assert len(sim_bridge.actors) == 0
    assert sim_bridge.ego_vehicle is None
