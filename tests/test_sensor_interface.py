"""
tests/test_sensor_interface.py
==============================
Unit tests for Sensor Interface Manager and Vehicle Kinematics.
Validates:
- Sensor reading dataclasses (IMU, GNSS, Wheel Encoders)
- Thread-safe hardware ingestion methods
- Dynamic vehicle side-slip angle (beta) calculation and non-linear understeer model
- Asynchronous simulation worker thread life-cycle
"""

import math
import time
import pytest
import numpy as np

from src.fusion.sensor_interface import (
    SensorInterfaceManager,
    IMUReading,
    GNSSReading,
    WheelEncoderReading,
    VehicleKinematics,
)


def test_sensor_dataclass_properties():
    """Verify dataclass property accessors."""
    imu = IMUReading(ax=0.1, ay=-0.2, az=9.81, gx=0.01, gy=0.02, gz=0.03)
    assert np.allclose(imu.accel, np.array([0.1, -0.2, 9.81]))
    assert np.allclose(imu.gyro, np.array([0.01, 0.02, 0.03]))

    gnss_standard = GNSSReading(fix_quality=1)
    gnss_rtk = GNSSReading(fix_quality=4)
    assert not gnss_standard.is_rtk
    assert gnss_rtk.is_rtk

    wheels = WheelEncoderReading(v_left=4.8, v_right=5.2)
    assert np.isclose(wheels.average_speed, 5.0)


def test_thread_safe_sensor_updates():
    """Test manual thread-safe ingestion into SensorInterfaceManager."""
    mgr = SensorInterfaceManager(sim_mode=False)

    mgr.update_imu(ax=0.5, ay=-0.1, az=9.8, gx=0.0, gy=0.0, gz=0.05)
    imu = mgr.get_latest_imu()
    assert np.isclose(imu.ax, 0.5)
    assert np.isclose(imu.gz, 0.05)

    mgr.update_gnss(lat=37.77, lon=-122.42, alt=10.0, fix_quality=4, hdop=0.7)
    gnss = mgr.get_latest_gnss()
    assert np.isclose(gnss.latitude, 37.77)
    assert gnss.is_rtk

    mgr.update_wheel_encoders(v_left=6.0, v_right=6.0)
    wheels = mgr.get_latest_wheel_encoders()
    assert np.isclose(wheels.average_speed, 6.0)


def test_slip_angle_calculation():
    """
    Test dynamic vehicle slip angle (beta) formula:
        beta = arctan(v_y / v_x) approx (lr / L) * delta / (1 + Kus * vx^2)
    """
    kin = VehicleKinematics(
        wheelbase_m=1.60,
        dist_cg_to_rear_m=0.85,
        understeer_gradient=0.002,
    )
    mgr = SensorInterfaceManager(kinematics=kin, sim_mode=False)

    # 1. Zero speed -> zero slip angle
    beta_stopped = mgr.compute_slip_angle_deg(speed_mps=0.1, steering_deg=10.0)
    assert beta_stopped == 0.0

    # 2. Straight driving (zero steer, zero yaw) -> zero slip angle
    beta_straight = mgr.compute_slip_angle_deg(speed_mps=10.0, steering_deg=0.0, yaw_rate_radps=0.0)
    assert beta_straight == 0.0

    # 3. Turning at low speed (kinematic bicycle model)
    # L=1.6, lr=0.85 -> ratio = 0.85 / 1.6 = 0.53125
    # For steer = 10 deg, Kus * 5^2 = 0.05 -> denom = 1.05
    # Expected beta approx 10 * 0.53125 / 1.05 ~ 5.06 deg
    beta_turn = mgr.compute_slip_angle_deg(speed_mps=5.0, steering_deg=10.0, yaw_rate_radps=0.0)
    assert 4.5 < beta_turn < 5.5

    # 4. Saturation check (-25 to +25 deg)
    beta_extreme = mgr.compute_slip_angle_deg(speed_mps=20.0, steering_deg=60.0)
    assert abs(beta_extreme) <= 25.0


def test_simulation_worker_thread():
    """Verify asynchronous background simulation worker generates periodic updates."""
    mgr = SensorInterfaceManager(sim_mode=True)
    mgr.start()
    time.sleep(0.08)  # Allow background worker to populate streams

    imu = mgr.get_latest_imu()
    wheels = mgr.get_latest_wheel_encoders()

    assert wheels.average_speed > 2.0
    assert imu.az > 8.0

    mgr.stop()
    assert not mgr._running
