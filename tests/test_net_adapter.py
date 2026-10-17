"""
tests/test_net_adapter.py
=========================
Unit tests for the Multi-Protocol Network Adapter Manager and ROS 2 DDS Publisher.
"""

import time
import pytest

from src.hardware.net_adapter import NetworkAdapterManager, NetworkProtocol
from src.hardware.ros2_publisher import APEXPerceptionNode


def test_network_protocol_enum():
    assert NetworkProtocol.CAN_SOCKET.value == "can_socket"
    assert NetworkProtocol.CAN_FD.value == "can_fd"
    assert NetworkProtocol.SERIAL_JSON.value == "serial_json"
    assert NetworkProtocol.ROS2_DDS.value == "ros2_dds"


def test_can_socket_actuation_command():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.CAN_SOCKET, channel="vcan0")
    received_frames = []

    def rx_handler(can_id: int, data: bytes):
        received_frames.append((can_id, data))

    adapter.register_rx_callback(adapter.ID_ACTUATION_CMD, rx_handler)

    success = adapter.send_actuation_command(
        steering_deg=12.5,
        throttle=0.45,
        brake=0.0,
        confidence=0.95,
        mode=1,
    )
    assert success is True
    assert len(received_frames) == 1
    can_id, data = received_frames[0]
    assert can_id == 0x100
    assert len(data) == 8  # Exactly 8 bytes for CAN 2.0B

    adapter.close()


def test_can_fd_trajectory_waypoints_64_bytes():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.CAN_FD, channel="vcan0")
    received_frames = []

    def rx_handler(can_id: int, data: bytes):
        received_frames.append((can_id, data))

    adapter.register_rx_callback(adapter.ID_CAN_FD_TRAJECTORY, rx_handler)

    waypoints = [(0.5, 0.0), (1.0, 0.1), (2.0, 0.25), (3.0, 0.40), (5.0, 0.60)]
    success = adapter.send_can_fd_trajectory(waypoints)
    assert success is True
    assert len(received_frames) == 1
    can_id, data = received_frames[0]
    assert can_id == 0x150
    assert len(data) == 64  # CAN-FD standard 64-byte payload

    adapter.close()


def test_serial_json_crc16_framing():
    adapter = NetworkAdapterManager(protocol=NetworkProtocol.SERIAL_JSON)
    success = adapter.send_actuation_command(
        steering_deg=-15.0,
        throttle=0.2,
        brake=0.8,
        confidence=0.99,
        mode=2,
    )
    assert success is True

    # Test CRC calculation directly
    data = b'{"type":"ACTUATION_CMD","steer":15.0}'
    crc = NetworkAdapterManager.compute_crc16(data)
    assert isinstance(crc, int)
    assert 0 <= crc <= 0xFFFF

    framed = adapter._encode_serial_json({"test": 123})
    assert framed.endswith("\n")
    assert "*" in framed
    json_part, crc_part = framed.strip().split("*")
    assert len(crc_part) == 4  # 4-hex digit CRC16

    adapter.close()


def test_ros2_perception_node_topics():
    node = APEXPerceptionNode(namespace="apex")
    received = {}

    def lane_cb(msg):
        received["lanes"] = msg

    def obj_cb(msg):
        received["objects"] = msg

    def traj_cb(msg):
        received["trajectory"] = msg

    def state_cb(msg):
        received["state"] = msg

    node.subscribe("/apex/perception/lanes", lane_cb)
    node.subscribe("/apex/perception/objects_3d", obj_cb)
    node.subscribe("/apex/planning/trajectory", traj_cb)
    node.subscribe("/apex/vehicle/state", state_cb)

    # Publish samples
    node.publish_lanes([{"x": 1.0, "y": -0.8}, {"x": 2.0, "y": -0.85}])
    node.publish_objects_3d([{"class": "pedestrian", "x": 8.0, "y": 0.5, "z": 0.0}])
    node.publish_trajectory([{"x": 0.0, "y": 0.0}, {"x": 2.0, "y": 0.1}])
    node.publish_state(speed_mps=5.0, steer_deg=-2.5, yaw_rad=0.01)

    assert "lanes" in received
    assert received["lanes"]["num_points"] == 2
    assert "objects" in received
    assert received["objects"]["num_objects"] == 1
    assert "trajectory" in received
    assert "state" in received

    stats = node.get_stats()
    assert stats["published_counts"]["/apex/perception/lanes"] == 1
    assert stats["published_counts"]["/apex/vehicle/state"] == 1


def test_ros2_node_periodic_publisher_30hz():
    node = APEXPerceptionNode(namespace="apex", target_hz=30.0)
    count_received = 0

    def state_cb(msg):
        nonlocal count_received
        count_received += 1

    node.subscribe("/apex/vehicle/state", state_cb)
    node.start_periodic_publisher()

    # Sleep for ~150ms -> should publish approx 4-5 messages at 30 Hz
    time.sleep(0.15)
    node.stop_periodic_publisher()

    assert count_received >= 3
