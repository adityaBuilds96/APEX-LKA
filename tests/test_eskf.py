"""
tests/test_eskf.py
==================
Unit tests for 15-DOF Error-State Kalman Filter (ESKF) core.
Validates:
- 100 Hz IMU propagation latency <= 0.5 ms
- Quaternion normalization and SO(3) attitude kinematics
- GNSS / RTK position measurement update
- CAN wheel encoder update with non-holonomic velocity constraints
- VIO relative displacement & yaw innovation update
- Dead reckoning accuracy (< 0.15 m drift over 5.0s GNSS/vision blackout)
- Covariance positive-definiteness & Joseph-form regularization
"""

import math
import time
import pytest
import numpy as np

from src.fusion.eskf import (
    ErrorStateKalmanFilter15DOF,
    ESKFConfig,
    LocalizationStatus,
    VehiclePose,
    quat_normalize,
    quat_to_rot_matrix,
    rot_matrix_to_euler,
)


@pytest.fixture
def eskf():
    config = ESKFConfig(
        imu_accel_noise=0.05,
        imu_gyro_noise=0.005,
        accel_bias_noise=0.001,
        gyro_bias_noise=0.0001,
        gnss_pos_noise=0.5,
        rtk_pos_noise=0.02,
    )
    return ErrorStateKalmanFilter15DOF(config=config)


def test_eskf_initialization(eskf):
    """Test initial nominal state vector, covariance, and status."""
    assert eskf.status == LocalizationStatus.INIT
    assert np.allclose(eskf.p, np.zeros(3))
    assert np.allclose(eskf.v, np.zeros(3))
    assert np.isclose(np.linalg.norm(eskf.q), 1.0)
    assert eskf.P.shape == (15, 15)
    # Covariance diagonal must be strictly positive
    assert np.all(np.diag(eskf.P) > 0.0)


def test_propagation_latency(eskf):
    """Constraint check: 100 Hz IMU propagation step MUST execute in <= 0.5 ms."""
    accel = [0.0, 0.0, 9.80665]
    gyro = [0.0, 0.0, 0.0]
    dt = 0.01

    # Warmup
    for _ in range(20):
        eskf.predict(accel, gyro, dt=dt)

    t0 = time.perf_counter()
    iterations = 200
    for _ in range(iterations):
        eskf.predict(accel, gyro, dt=dt)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    assert elapsed_ms <= 0.5, f"Propagation step took {elapsed_ms:.4f} ms (budget <= 0.5 ms)"


def test_quaternion_normalization_and_rotation(eskf):
    """Verify quaternion maintains unit norm and integrates pure yaw rotation correctly."""
    yaw_rate = 0.1  # rad/s around body Z
    dt = 0.01

    for _ in range(100):  # 1.0 second of rotation
        eskf.predict(imu_accel=[0.0, 0.0, 9.80665], imu_gyro=[0.0, 0.0, yaw_rate], dt=dt)
        q_norm = np.linalg.norm(eskf.q)
        assert np.isclose(q_norm, 1.0, atol=1e-7)

    pose = eskf.get_vehicle_pose()
    assert np.isclose(pose.yaw, 0.10, atol=0.015)
    assert np.isclose(pose.roll, 0.0, atol=1e-3)
    assert np.isclose(pose.pitch, 0.0, atol=1e-3)


def test_gnss_update_local_and_rtk(eskf):
    """Test GNSS / RTK position measurement update and status transition."""
    # Predict forward slightly
    eskf.predict(imu_accel=[0.0, 0.0, 9.80665], imu_gyro=[0.0, 0.0, 0.0], dt=0.01)

    # Standard GNSS update in local meters (pulls estimate from 0 towards measurement)
    eskf.update_gnss(lat_or_x=5.0, lon_or_y=12.0, alt_or_z=0.0, is_local_xyz=True, is_rtk=False)
    assert eskf.status == LocalizationStatus.GNSS_FIX
    assert np.isclose(eskf.p[0], 5.0, atol=2.0)
    assert np.isclose(eskf.p[1], 12.0, atol=3.0)

    # 2nd GNSS update converges close to true position
    eskf.update_gnss(lat_or_x=5.0, lon_or_y=12.0, alt_or_z=0.0, is_local_xyz=True, is_rtk=False)
    assert np.isclose(eskf.p[0], 5.0, atol=0.7)
    assert np.isclose(eskf.p[1], 12.0, atol=1.5)

    # RTK update with low noise (0.02m) locks position
    eskf.update_gnss(lat_or_x=5.20, lon_or_y=12.10, alt_or_z=0.0, is_local_xyz=True, is_rtk=True)
    assert eskf.status == LocalizationStatus.RTK_LOCKED
    assert np.isclose(eskf.p[0], 5.20, atol=0.10)
    assert np.isclose(eskf.p[1], 12.10, atol=0.10)


def test_can_wheel_encoder_update(eskf):
    """Test wheel speed fusion and non-holonomic velocity constraint."""
    # Forward speed: 10 m/s
    eskf.update_wheel_encoders(v_left=10.0, v_right=10.0)
    pose = eskf.get_vehicle_pose()

    assert np.isclose(pose.v_forward, 10.0, atol=1.0)
    # Body lateral and vertical velocities should remain near zero (non-holonomic)
    R = quat_to_rot_matrix(eskf.q)
    v_body = R.T @ eskf.v
    assert abs(v_body[1]) < 0.25
    assert abs(v_body[2]) < 0.25


def test_visual_odometry_update(eskf):
    """Test fusing visual odometry displacement [dx, dy] and dpsi."""
    eskf.reset()
    eskf.update_visual_odometry(dx=1.5, dy=0.2, dpsi=0.05)
    pose = eskf.get_vehicle_pose()

    assert eskf.status == LocalizationStatus.VIO_ACTIVE
    assert np.isclose(pose.x, 1.5, atol=0.5)
    assert np.isclose(pose.y, 0.2, atol=0.5)
    assert np.isclose(pose.yaw, 0.05, atol=0.05)


def test_dead_reckoning_drift_during_5s_blackout(eskf):
    """
    Hard constraint: Position drift during 5-second GNSS + Vision blackout
    MUST remain < 0.15 m (15 cm) with calibrated zero-motion IMU.
    """
    eskf.reset(x=0.0, y=0.0, z=0.0, yaw=0.0)

    # 5.0 seconds at 100 Hz (500 steps, dt=0.01) with vehicle stationary
    for _ in range(500):
        eskf.predict(imu_accel=[0.0, 0.0, 9.80665], imu_gyro=[0.0, 0.0, 0.0], dt=0.01)
        # 50 Hz wheel encoder confirms 0 m/s speed every 2nd step
        if _ % 2 == 0:
            eskf.update_wheel_encoders(v_left=0.0, v_right=0.0)

    pose = eskf.get_vehicle_pose()
    pos_drift = math.sqrt(pose.x**2 + pose.y**2 + pose.z**2)
    assert pos_drift < 0.15, f"Blackout position drift was {pos_drift:.4f} m (must be < 0.15 m)"


def test_covariance_matrix_regularization(eskf):
    """Ensure covariance matrix remains symmetric and positive-definite after severe updates."""
    for i in range(50):
        eskf.predict(imu_accel=[0.1, -0.05, 9.80665], imu_gyro=[0.01, -0.01, 0.02], dt=0.01)
        if i % 5 == 0:
            eskf.update_wheel_encoders(v_left=5.0, v_right=5.0)
        if i % 10 == 0:
            eskf.update_gnss(lat_or_x=float(i * 0.05), lon_or_y=0.0, alt_or_z=0.0, is_local_xyz=True)

    # Check symmetry
    assert np.allclose(eskf.P, eskf.P.T, atol=1e-8)
    # Check eigenvalues are all positive (positive-definite)
    eigenvalues = np.linalg.eigvalsh(eskf.P)
    assert np.all(eigenvalues > 0.0)
