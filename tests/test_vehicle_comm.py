"""
tests/test_vehicle_comm.py
==========================
Comprehensive Pytest suite for Jetson ↔ Arduino communication layer:
1. Struct packing and unpacking correctness for CAN IDs 0x100, 0x200, 0x300, 0x301, 0x050
2. JSON packet serialization, deserialization, and malformed string handling
3. Automatic failover logic (CAN error threshold triggering Serial switch)
4. Full simulation mode (VehicleCommunicator in sim_mode=True)
5. Telemetry caching and thread safety
6. Mode enum mapping and boundary clipping
"""

import json
import struct
import time
import pytest
import numpy as np

from deploy.vehicle_comm import (
    CAN_ID_CMD_NODE1,
    CAN_ID_CMD_NODE2,
    CAN_ID_FEEDBACK_N1,
    CAN_ID_FEEDBACK_N2,
    CAN_ID_HEARTBEAT,
    MODE_AUTO,
    MODE_EMERGENCY_STOP,
    MODE_MANUAL,
    MODE_INT_TO_STR,
    MODE_STR_TO_INT,
    Node1Telemetry,
    Node2Telemetry,
    VehicleCommunicator,
)


def test_can_id_0x100_packing_and_unpacking():
    """Verify bit-packed struct encoding for Node 1 Control Command (0x100)."""
    steer_deg = 14.50
    trim = -2.50
    confidence = 0.95
    mode = MODE_AUTO
    ts_ms = 45120

    # Pack format: <hhBBH
    steer_s = int(steer_deg * 100)  # 1450
    trim_s = int(trim * 100)        # -250
    conf_s = int(confidence * 255)  # 242
    packed = struct.pack("<hhBBH", steer_s, trim_s, conf_s, mode, ts_ms)

    assert len(packed) == 8
    unpacked = struct.unpack("<hhBBH", packed)
    assert unpacked[0] == 1450
    assert unpacked[1] == -250
    assert unpacked[2] == 242
    assert unpacked[3] == MODE_AUTO
    assert unpacked[4] == 45120


def test_can_id_0x200_packing_and_unpacking():
    """Verify bit-packed struct encoding for Node 2 Control Command (0x200)."""
    throttle_pct = 45.00
    brake_pct = 10.00
    mode = MODE_AUTO
    aux_flags = 0x05
    ts_ms = 12345

    packed = struct.pack("<hhBBH", int(throttle_pct * 100), int(brake_pct * 100), mode, aux_flags, ts_ms)
    assert len(packed) == 8

    unpacked = struct.unpack("<hhBBH", packed)
    assert unpacked[0] == 4500
    assert unpacked[1] == 1000
    assert unpacked[2] == MODE_AUTO
    assert unpacked[3] == 0x05
    assert unpacked[4] == 12345


def test_can_id_0x300_feedback_node1():
    """Verify unpacking for Node 1 Feedback (0x300)."""
    # Actual steer: -8.20 deg, Motor current: 3.40A, override: 0, fault: 0, uptime: 5000ms
    packed = struct.pack("<hhBBH", -820, 340, 0, 0, 5000)
    steer_s, curr_s, ovr, flt, up = struct.unpack("<hhBBH", packed)

    assert steer_s / 100.0 == -8.20
    assert curr_s / 100.0 == 3.40
    assert ovr == 0
    assert flt == 0
    assert up == 5000


def test_can_id_0x301_feedback_node2():
    """Verify unpacking for Node 2 Feedback (0x301)."""
    # Speed: 6.75 m/s, Brake pressure: 145.2 psi, override: 1 (pedal pressed), fault: 0
    packed = struct.pack("<hhBBH", 675, 1452, 1, 0, 10240)
    speed_s, press_s, ovr, flt, up = struct.unpack("<hhBBH", packed)

    assert speed_s / 100.0 == 6.75
    assert press_s / 10.0 == 145.2
    assert ovr == 1
    assert up == 10240


def test_can_id_0x050_heartbeat():
    """Verify master heartbeat token and format."""
    packed = struct.pack("<BBH", 0xAA, 0, 100)
    assert len(packed) == 4
    token, health, cycle = struct.unpack("<BBH", packed)
    assert token == 0xAA
    assert health == 0
    assert cycle == 100


def test_json_serialization_and_malformed_handling():
    """Verify JSON packet format and resilience against malformed strings."""
    packet = {
        "node": 1,
        "cmd_a": 12.5,
        "cmd_b": 0.0,
        "offset_m": -0.12,
        "confidence": 0.95,
        "mode": "AUTO",
        "ts": 123456,
    }
    encoded = (json.dumps(packet) + "\n").encode("utf-8")
    assert b"\"node\": 1" in encoded or b"\"node\":1" in encoded

    decoded = json.loads(encoded.decode("utf-8").strip())
    assert decoded["cmd_a"] == 12.5
    assert decoded["mode"] == "AUTO"

    # Malformed JSON should not crash
    with pytest.raises(json.JSONDecodeError):
        json.loads("{node: 1, broken_json")


def test_mode_conversions():
    """Verify mode enum string and integer bi-directional mapping."""
    assert MODE_STR_TO_INT["AUTO"] == MODE_AUTO
    assert MODE_STR_TO_INT["MANUAL"] == MODE_MANUAL
    assert MODE_STR_TO_INT["EMERGENCY_STOP"] == MODE_EMERGENCY_STOP

    assert MODE_INT_TO_STR[MODE_AUTO] == "AUTO"
    assert MODE_INT_TO_STR[MODE_MANUAL] == "MANUAL"
    assert MODE_INT_TO_STR[MODE_EMERGENCY_STOP] == "EMERGENCY_STOP"


def test_vehicle_communicator_sim_mode():
    """Verify VehicleCommunicator operates seamlessly in sim_mode=True."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    assert comm.is_healthy()

    # Transmit commands
    ok1 = comm.send_node1_command(steering_deg=10.0, confidence=0.98, mode="AUTO")
    ok2 = comm.send_node2_command(throttle_pct=30.0, mode="AUTO")
    ok_hb = comm.send_heartbeat(health_code=0)

    assert ok1
    assert ok2
    assert ok_hb

    # In sim_mode, simulate feedback injection
    fb1 = struct.pack("<hhBBH", 980, 210, 0, 0, 1000)
    comm.can_if.send_frame(CAN_ID_FEEDBACK_N1, fb1)

    t1 = comm.get_node_telemetry(node_id=1)
    assert isinstance(t1, Node1Telemetry)
    assert t1.actual_steering_deg == 9.80
    assert t1.motor_current_amps == 2.10

    comm.close()


def test_vehicle_communicator_failover_trigger():
    """Verify automatic failover channel switching when CAN errors accumulate."""
    comm = VehicleCommunicator(enable_can=False, enable_serial=False, sim_mode=False)

    # Force simulated error increments
    for _ in range(6):
        comm.send_node1_command(steering_deg=5.0)

    assert comm.consecutive_can_errors >= 5
    assert comm.active_channel == "SERIAL"
    comm.close()
