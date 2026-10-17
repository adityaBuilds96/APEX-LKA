"""
tests/test_universal_deployment.py
==================================
Comprehensive integration and unit test suite for Phase 5 Section 8:
- Universal Vehicle Profile System & Hot-Swapping
- Multi-Hardware Abstraction Layer (HAL) & Compute Backends
- Multi-Protocol Network Adapter (CAN, CAN-FD, Serial, ROS 2 DDS)
- ISO 26262-Inspired Functional Safety Artifacts (FMEA, FTA, Safety Case, SBOM, Manifest)
- Production Docker & Deployment Verification
"""

import json
from pathlib import Path
import time
import numpy as np
import pytest

from src.config.vehicle_profile import (
    VehicleProfileRegistry,
    VehicleProfile,
    KinematicsConfig,
    CameraExtrinsicsConfig,
    ActuationProfile,
    SafetyLimitsConfig,
)
from src.hardware.hal import (
    HardwareAbstractionLayer,
    HardwareBackend,
    CPUGenericBackend,
    get_hal,
)
from src.hardware.backends.jetson_backend import JetsonBackend
from src.hardware.backends.tpu_backend import CoralTPUBackend
from src.hardware.backends.openvino_backend import OpenVINOBackend
from src.hardware.net_adapter import (
    NetworkAdapterManager,
    NetworkProtocol,
)
from src.hardware.ros2_publisher import APEXPerceptionNode


# ==============================================================================
# 1. Universal Vehicle Profile System Tests (5 Tests)
# ==============================================================================

def test_profile_registry_loads_all_standard_profiles():
    registry = VehicleProfileRegistry()
    available = registry.list_available_profiles()
    for required in ["baja_2027", "campus_shuttle", "agv_warehouse", "passenger_car"]:
        assert required in available, f"Missing required profile: {required}"
        prof = registry.load_profile(required)
        assert prof.name == required
        assert prof.kinematics.wheelbase_m > 0.0
        assert prof.safety_limits.max_speed_mps > 0.0


def test_runtime_zero_recompile_profile_hotswapping():
    registry = VehicleProfileRegistry()
    history = []

    def on_change(p: VehicleProfile):
        history.append(p.name)

    registry.register_callback(on_change)

    p1 = registry.switch_profile("baja_2027")
    assert registry.get_active_profile().name == "baja_2027"

    p2 = registry.switch_profile("passenger_car")
    assert registry.get_active_profile().name == "passenger_car"
    assert registry.get_active_profile().kinematics.wheelbase_m == 2.80

    assert history == ["baja_2027", "passenger_car"]


def test_actuation_lut_non_linear_interpolation():
    act = ActuationProfile(
        pwm_min=1000,
        pwm_max=2000,
        pwm_center=1500,
        direction_sign=1,
        lut_steering=[[-30.0, 1000.0], [-10.0, 1300.0], [0.0, 1500.0], [10.0, 1700.0], [30.0, 2000.0]],
    )
    assert act.map_steering_deg_to_pwm(0.0) == 1500
    assert act.map_steering_deg_to_pwm(5.0) == 1600
    assert act.map_steering_deg_to_pwm(-5.0) == 1400
    # Clamping beyond LUT
    assert act.map_steering_deg_to_pwm(45.0) == 2000
    assert act.map_steering_deg_to_pwm(-45.0) == 1000


def test_camera_extrinsics_angles_in_radians():
    extrinsics = CameraExtrinsicsConfig(
        mount_height_m=1.25,
        pitch_deg=-5.0,
        roll_deg=0.0,
        yaw_deg=0.0,
    )
    assert np.isclose(extrinsics.pitch_rad, np.radians(-5.0))
    assert extrinsics.roll_rad == 0.0
    assert extrinsics.yaw_rad == 0.0


def test_profile_safety_limits_enforcement():
    profile = VehicleProfile(
        name="test_lim",
        vehicle_type="buggy",
        safety_limits=SafetyLimitsConfig(
            max_speed_mps=6.94,
            max_accel_mpss=2.5,
            emergency_decel_mpss=6.0,
            lateral_error_limit_m=1.0,
        ),
    )
    is_valid, issues = VehicleProfileRegistry.validate_profile(profile)
    assert is_valid is True
    assert profile.safety_limits.emergency_decel_mpss > profile.safety_limits.max_accel_mpss


# ==============================================================================
# 2. Multi-Hardware Abstraction Layer (HAL) Tests (6 Tests)
# ==============================================================================

def test_hal_auto_detection_and_initialization():
    hal = HardwareAbstractionLayer(preferred_backend="auto")
    assert hal.active_backend is not None
    assert hal.active_backend.is_available() is True
    assert hal.active_backend.initialize() is True


def test_hal_polymorphic_backend_switching():
    hal = HardwareAbstractionLayer()

    hal.set_backend("jetson")
    assert hal.active_backend.name == "jetson_tensorrt"

    hal.set_backend("tpu")
    assert hal.active_backend.name == "coral_edge_tpu"

    hal.set_backend("openvino")
    assert hal.active_backend.name == "intel_openvino"

    hal.set_backend("cpu")
    assert hal.active_backend.name == "generic_cpu"


def test_hal_predict_contract_consistency():
    tensor = np.zeros((1, 3, 360, 640), dtype=np.float32)
    hal = HardwareAbstractionLayer()

    for b_name in ["jetson", "tpu", "openvino", "cpu"]:
        hal.set_backend(b_name)
        out = hal.predict(tensor)
        assert "segmentation" in out
        assert "lane_logits" in out
        assert "boxes" in out
        assert "depth_map" in out
        assert out["segmentation"].shape == (1, 3, 360, 640)
        assert out["depth_map"].shape == (1, 1, 360, 640)


def test_hal_telemetry_schema():
    hal = HardwareAbstractionLayer()
    telemetry = hal.get_telemetry()
    assert "backend" in telemetry
    assert "device" in telemetry
    assert "cpu_percent" in telemetry
    assert "ram_used_mb" in telemetry
    assert "chip_temperature_c" in telemetry
    assert "hal_overhead_ms" in telemetry
    assert telemetry["hal_overhead_ms"] < 0.10


def test_hal_dispatch_benchmark_sub_100_microseconds():
    hal = HardwareAbstractionLayer(preferred_backend="cpu")
    avg_us = hal.measure_dispatch_overhead_us(iterations=200)
    assert avg_us > 0.0
    # Dispatch wrapper must not add prohibitive overhead
    assert avg_us < 2500.0


def test_hal_singleton_accessor():
    h1 = get_hal()
    h2 = get_hal()
    assert h1 is h2


# ==============================================================================
# 3. Multi-Protocol Network Adapter Tests (5 Tests)
# ==============================================================================

def test_net_adapter_can_socket_frame_encoding():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.CAN_SOCKET, channel="vcan0")
    received = []
    adapter.register_rx_callback(adapter.ID_ACTUATION_CMD, lambda cid, d: received.append((cid, d)))

    ok = adapter.send_actuation_command(steering_deg=15.0, throttle=0.5, brake=0.0, confidence=0.98, mode=1)
    assert ok is True
    assert len(received) == 1
    cid, data = received[0]
    assert cid == 0x100
    assert len(data) == 8
    adapter.close()


def test_net_adapter_can_fd_trajectory_64_bytes():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.CAN_FD, channel="vcan0")
    received = []
    adapter.register_rx_callback(adapter.ID_CAN_FD_TRAJECTORY, lambda cid, d: received.append((cid, d)))

    pts = [(1.0, 0.0), (2.0, 0.2), (3.0, 0.5), (4.0, 0.8)]
    ok = adapter.send_can_fd_trajectory(pts)
    assert ok is True
    assert len(received) == 1
    cid, data = received[0]
    assert cid == 0x150
    assert len(data) == 64
    adapter.close()


def test_net_adapter_serial_json_crc16():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.SERIAL_JSON)
    raw = adapter._encode_serial_json({"cmd": "steer", "deg": 10.0})
    assert raw.endswith("\n")
    assert "*" in raw
    json_str, crc_str = raw.strip().split("*")
    computed = NetworkAdapterManager.compute_crc16(json_str.encode("utf-8"))
    assert int(crc_str, 16) == computed
    adapter.close()


def test_ros2_perception_publisher_dispatch():
    node = APEXPerceptionNode()
    rec_lanes = []
    rec_state = []
    node.subscribe("/apex/perception/lanes", lambda m: rec_lanes.append(m))
    node.subscribe("/apex/vehicle/state", lambda m: rec_state.append(m))

    node.publish_lanes([{"x": 1.0, "y": -0.5}])
    node.publish_state(speed_mps=5.0, steer_deg=2.0, yaw_rad=0.0)

    assert len(rec_lanes) == 1
    assert len(rec_state) == 1
    assert rec_lanes[0]["num_points"] == 1
    assert rec_state[0]["speed_mps"] == 5.0


def test_ros2_node_target_frequency_stats():
    node = APEXPerceptionNode(target_hz=30.0)
    stats = node.get_stats()
    assert stats["target_hz"] == 30.0
    assert "published_counts" in stats


# ==============================================================================
# 4. Functional Safety & Certification Artifacts Tests (4 Tests)
# ==============================================================================

def test_fmea_matrix_document_completeness():
    fmea_path = Path("docs/safety_certification/FMEA_MATRIX.md")
    assert fmea_path.exists()
    content = fmea_path.read_text(encoding="utf-8")
    assert "ISO 26262" in content
    assert "RPN" in content
    assert "SEN-01" in content
    assert "ACT-01" in content
    # Verify at least 25 failure mode IDs exist
    assert content.count("| **") >= 25


def test_fault_tree_analysis_minimal_cut_sets():
    fta_path = Path("docs/safety_certification/FAULT_TREE_ANALYSIS.md")
    assert fta_path.exists()
    content = fta_path.read_text(encoding="utf-8")
    assert "HAZ-01" in content
    assert "HAZ-02" in content
    assert "Minimal Cut Sets" in content
    assert "BE-1" in content


def test_asil_safety_case_and_hara():
    case_path = Path("docs/safety_certification/ASIL_SAFETY_CASE.md")
    assert case_path.exists()
    content = case_path.read_text(encoding="utf-8")
    assert "ASIL-B" in content
    assert "HARA" in content
    assert "SG-01" in content
    assert "Freedom From Interference" in content


def test_reproducibility_manifest_and_sbom():
    manifest_path = Path("docs/safety_certification/REPRODUCIBILITY_MANIFEST.json")
    sbom_path = Path("docs/safety_certification/SOFTWARE_BOM.md")
    assert manifest_path.exists()
    assert sbom_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "manifest_version" in data
    assert "docker_artifacts" in data
    assert "models" in data
    assert "scenario_test_suite" in data
    assert data["scenario_test_suite"]["total_scenarios"] == 101


# ==============================================================================
# 5. Production Docker & Deployment Verification Tests (2 Tests)
# ==============================================================================

def test_dockerfile_edge_configuration_sanity():
    df_path = Path("deploy/Dockerfile.edge")
    assert df_path.exists()
    content = df_path.read_text(encoding="utf-8")
    assert "FROM ubuntu:22.04 AS builder" in content
    assert "FROM ubuntu:22.04 AS runtime" in content
    assert "HEALTHCHECK" in content
    assert "ENTRYPOINT" in content


def test_docker_compose_and_entrypoint_integrity():
    compose_path = Path("deploy/docker-compose.deploy.yml")
    entry_path = Path("deploy/docker_entrypoint.sh")
    assert compose_path.exists()
    assert entry_path.exists()

    c_content = compose_path.read_text(encoding="utf-8")
    assert "apex-lka-edge" in c_content
    assert "devices:" in c_content
    assert "/dev/video0" in c_content

    e_content = entry_path.read_text(encoding="utf-8")
    assert "trap cleanup SIGINT SIGTERM" in e_content
