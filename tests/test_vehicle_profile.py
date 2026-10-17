"""
tests/test_vehicle_profile.py
=============================
Unit tests for the Universal Vehicle Profile Registry and Configuration Dataclasses.
"""

from pathlib import Path
import pytest

from src.config.vehicle_profile import (
    VehicleProfileRegistry,
    VehicleProfile,
    KinematicsConfig,
    CameraExtrinsicsConfig,
    ActuationProfile,
    SafetyLimitsConfig,
)


def test_registry_lists_and_loads_default_profiles():
    registry = VehicleProfileRegistry()
    profiles = registry.list_available_profiles()
    assert "baja_2027" in profiles
    assert "campus_shuttle" in profiles
    assert "agv_warehouse" in profiles
    assert "passenger_car" in profiles


def test_baja_2027_profile_specs():
    registry = VehicleProfileRegistry()
    profile = registry.load_profile("baja_2027")
    assert profile.name == "baja_2027"
    assert profile.vehicle_type == "offroad_buggy"
    assert profile.kinematics.wheelbase_m == 1.45
    assert profile.kinematics.track_width_m == 1.25
    assert profile.kinematics.max_steer_angle_deg == 32.0
    assert profile.camera_extrinsics.mount_height_m == 1.25
    assert profile.camera_extrinsics.pitch_deg == -5.0
    assert profile.safety_limits.max_speed_mps == 6.94


def test_campus_shuttle_profile_specs():
    registry = VehicleProfileRegistry()
    profile = registry.load_profile("campus_shuttle")
    assert profile.name == "campus_shuttle"
    assert profile.vehicle_type == "passenger_shuttle"
    assert profile.kinematics.wheelbase_m == 2.60
    assert profile.camera_extrinsics.mount_height_m == 1.80
    assert profile.safety_limits.max_speed_mps == 4.17


def test_agv_warehouse_profile_specs():
    registry = VehicleProfileRegistry()
    profile = registry.load_profile("agv_warehouse")
    assert profile.name == "agv_warehouse"
    assert profile.vehicle_type == "logistics_agv"
    assert profile.kinematics.wheelbase_m == 1.10
    assert profile.actuation.direction_sign == -1
    assert profile.safety_limits.lateral_error_limit_m == 0.35


def test_passenger_car_profile_specs():
    registry = VehicleProfileRegistry()
    profile = registry.load_profile("passenger_car")
    assert profile.name == "passenger_car"
    assert profile.vehicle_type == "passenger_ev"
    assert profile.kinematics.wheelbase_m == 2.80
    assert profile.safety_limits.max_speed_mps == 22.22


def test_actuation_lut_interpolation_and_bounds():
    profile = ActuationProfile(
        pwm_min=1000,
        pwm_max=2000,
        pwm_center=1500,
        center_trim=0,
        direction_sign=1,
        lut_steering=[[-30.0, 1000.0], [0.0, 1500.0], [30.0, 2000.0]],
        lut_throttle=[[0.0, 1500.0], [1.0, 2000.0]],
        lut_brake=[[0.0, 1500.0], [1.0, 1000.0]],
    )

    # Steering test
    assert profile.map_steering_deg_to_pwm(0.0) == 1500
    assert profile.map_steering_deg_to_pwm(15.0) == 1750
    assert profile.map_steering_deg_to_pwm(-15.0) == 1250
    assert profile.map_steering_deg_to_pwm(50.0) == 2000  # Clamped to max
    assert profile.map_steering_deg_to_pwm(-50.0) == 1000 # Clamped to min

    # Throttle test
    assert profile.map_throttle_to_pwm(0.0) == 1500
    assert profile.map_throttle_to_pwm(0.5) == 1750
    assert profile.map_throttle_to_pwm(1.0) == 2000

    # Brake test
    assert profile.map_brake_to_pwm(0.0) == 1500
    assert profile.map_brake_to_pwm(0.5) == 1250
    assert profile.map_brake_to_pwm(1.0) == 1000


def test_actuation_direction_sign_inversion():
    inverted_profile = ActuationProfile(
        pwm_min=1000,
        pwm_max=2000,
        pwm_center=1500,
        direction_sign=-1,
        lut_steering=[[-30.0, 1000.0], [0.0, 1500.0], [30.0, 2000.0]],
    )
    # With direction_sign=-1, positive angle maps to negative LUT entry
    assert inverted_profile.map_steering_deg_to_pwm(15.0) == 1250
    assert inverted_profile.map_steering_deg_to_pwm(-15.0) == 1750


def test_hot_swapping_and_callbacks():
    registry = VehicleProfileRegistry()
    notified = []

    def on_switch(new_p: VehicleProfile):
        notified.append(new_p.name)

    registry.register_callback(on_switch)
    p1 = registry.switch_profile("campus_shuttle")
    assert registry.get_active_profile().name == "campus_shuttle"
    assert notified == ["campus_shuttle"]

    p2 = registry.switch_profile("agv_warehouse")
    assert registry.get_active_profile().name == "agv_warehouse"
    assert notified == ["campus_shuttle", "agv_warehouse"]

    registry.unregister_callback(on_switch)
    registry.switch_profile("baja_2027")
    assert len(notified) == 2  # Callback was detached


def test_profile_validation():
    valid_profile = VehicleProfile(
        name="test_valid",
        vehicle_type="cart",
        kinematics=KinematicsConfig(wheelbase_m=1.2, track_width_m=0.9),
        camera_extrinsics=CameraExtrinsicsConfig(mount_height_m=1.1),
        actuation=ActuationProfile(pwm_min=1000, pwm_max=2000),
        safety_limits=SafetyLimitsConfig(max_speed_mps=5.0),
    )
    is_valid, issues = VehicleProfileRegistry.validate_profile(valid_profile)
    assert is_valid is True
    assert len(issues) == 0

    invalid_profile = VehicleProfile(
        name="test_invalid",
        vehicle_type="cart",
        kinematics=KinematicsConfig(wheelbase_m=-1.0, track_width_m=0.0),
        camera_extrinsics=CameraExtrinsicsConfig(mount_height_m=-0.5),
        actuation=ActuationProfile(pwm_min=2000, pwm_max=1000),
        safety_limits=SafetyLimitsConfig(max_speed_mps=-5.0),
    )
    is_valid, issues = VehicleProfileRegistry.validate_profile(invalid_profile)
    assert is_valid is False
    assert len(issues) >= 4


def test_profile_serialization_roundtrip(tmp_path: Path):
    registry = VehicleProfileRegistry()
    profile = registry.load_profile("baja_2027")

    save_file = tmp_path / "custom_profile.yaml"
    profile.save_yaml(save_file)
    assert save_file.exists()

    loaded = registry.load_profile(save_file)
    assert loaded.name == profile.name
    assert loaded.kinematics.wheelbase_m == profile.kinematics.wheelbase_m
    assert loaded.safety_limits.max_speed_mps == profile.safety_limits.max_speed_mps


def test_missing_profile_raises_error():
    registry = VehicleProfileRegistry()
    with pytest.raises(FileNotFoundError):
        registry.load_profile("non_existent_vehicle_profile_xyz")
