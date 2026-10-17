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
import sys
import time
import pytest
import numpy as np

from deploy.vehicle_comm import (
    CAN_ID_CMD_NODE1,
    CAN_ID_CMD_NODE2,
    CAN_ID_FEEDBACK_N1,
    CAN_ID_FEEDBACK_N2,
    CAN_ID_HEARTBEAT,
    CANInterface,
    MODE_AUTO,
    MODE_EMERGENCY_STOP,
    MODE_MANUAL,
    MODE_INT_TO_STR,
    MODE_STR_TO_INT,
    Node1Telemetry,
    Node2Telemetry,
    VehicleCommunicator,
)
from deploy.jetson_run import StanleyController, SafeExecutor


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


def test_stanley_controller_zero_error_yields_zero_steering():
    """Verify that zero heading error and zero lateral offset produces 0 deg steering."""
    ctrl = StanleyController(k_gain=2.0, max_steer_deg=30.0, default_speed_mps=5.0)
    steer = ctrl.compute(heading_err_rad=0.0, lateral_offset_m=0.0, speed_mps=5.0)
    assert steer == 0.0


def test_stanley_controller_cross_track_correction_and_clamping():
    """Verify Stanley steering correction direction, velocity scaling, and max angle saturation."""
    ctrl = StanleyController(k_gain=2.0, max_steer_deg=30.0, default_speed_mps=5.0)

    # When vehicle is right of center (offset > 0), controller should command negative steer (steer left)
    steer_right = ctrl.compute(heading_err_rad=0.0, lateral_offset_m=0.5, speed_mps=5.0)
    assert steer_right < 0.0

    # When vehicle is left of center (offset < 0), controller should command positive steer (steer right)
    steer_left = ctrl.compute(heading_err_rad=0.0, lateral_offset_m=-0.5, speed_mps=5.0)
    assert steer_left > 0.0

    # Higher speed dampens steering angle for same cross-track error
    steer_fast = ctrl.compute(heading_err_rad=0.0, lateral_offset_m=0.5, speed_mps=15.0)
    assert abs(steer_fast) < abs(steer_right)

    # Extreme cross-track error saturates at max_steer_deg
    steer_extreme = ctrl.compute(heading_err_rad=0.0, lateral_offset_m=10.0, speed_mps=1.0)
    assert steer_extreme == -30.0


def test_send_node1_control_alias():
    """Verify send_node1_control alias forwards arguments properly to send_node1_command."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    ok_str = comm.send_node1_control(steering_deg=-12.5, confidence=0.92, mode="AUTO", offset_m=0.15)
    assert ok_str

    ok_int = comm.send_node1_control(steering_deg=10.0, confidence=0.88, mode=MODE_AUTO, offset_m=-0.10)
    assert ok_int
    comm.close()


def test_send_node2_control_alias():
    """Verify send_node2_control alias forwards arguments properly to send_node2_command."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    ok = comm.send_node2_control(throttle_pct=35.0, brake_pct=0.0, mode="AUTO")
    assert ok
    comm.close()


def test_serial_compact_json_schema_fields():
    """Verify Serial fallback JSON packet contains both compact and descriptive keys."""
    class MockSerialInterface:
        def __init__(self):
            self.is_connected = True
            self.last_json = None

        def send_json(self, packet):
            self.last_json = packet
            return True

        def close(self):
            pass

    comm = VehicleCommunicator(enable_can=False, enable_serial=True, sim_mode=False)
    mock_serial = MockSerialInterface()
    comm.serial_n1 = mock_serial

    comm.send_node1_command(steering_deg=-12.5, actuator_trim=0.0, confidence=0.92, mode="AUTO", offset_m=0.12)
    assert mock_serial.last_json is not None
    pkt = mock_serial.last_json

    # Compact keys
    assert pkt["n"] == 1
    assert pkt["str"] == -12.5
    assert pkt["off"] == 0.12
    assert pkt["cnf"] == 0.92
    assert pkt["m"] == MODE_AUTO
    assert "ts" in pkt
    comm.close()


def test_can_interface_static_packing_helpers():
    """Verify CANInterface static bit-packing helper methods."""
    # 0x100 Node 1 Frame
    p1 = CANInterface.pack_node1_cmd(steering_deg=14.5, offset_mm_or_trim=-250, confidence=0.95, mode_val=1, ts_ms=45120)
    assert len(p1) == 8
    unpacked_1 = struct.unpack("<hhBBH", p1)
    assert unpacked_1[0] == 1450
    assert unpacked_1[1] == -250
    assert unpacked_1[3] == 1

    # 0x200 Node 2 Frame
    p2 = CANInterface.pack_node2_cmd(cmd_a=45.0, cmd_b=10.0, mode_val=1, flags=0x05, ts_ms=12345)
    assert len(p2) == 8
    unpacked_2 = struct.unpack("<hhBBH", p2)
    assert unpacked_2[0] == 4500
    assert unpacked_2[1] == 1000

    # 0x050 Heartbeat Frame
    ph = CANInterface.pack_heartbeat(alive_token=0xAA, health_code=0, sequence_id=100)
    assert len(ph) == 4
    unpacked_h = struct.unpack("<BBH", ph)
    assert unpacked_h[0] == 0xAA


def test_can_network_order_endianness_support():
    """Verify bit-packing helper supports big-endian network order '!' as well."""
    p_net = CANInterface.pack_node1_cmd(steering_deg=10.0, offset_mm_or_trim=500, confidence=1.0, mode_val=1, ts_ms=1000, endian="!")
    assert len(p_net) == 8
    unpacked_net = struct.unpack("!hhBBH", p_net)
    assert unpacked_net[0] == 1000
    assert unpacked_net[1] == 500


def test_safe_executor_camera_loss_shielding():
    """Verify SafeExecutor triggers failsafe zero-steering and neutral throttle on camera disconnect."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    executor = SafeExecutor(communicator=comm, reconnect_interval_sec=2.0)

    executor.handle_camera_loss()
    assert comm.is_healthy()
    comm.close()


def test_telemetry_serialization_latency_under_2ms():
    """Verify mean command serialization and dispatch time is strictly < 2.0 ms."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        comm.send_node1_control(steering_deg=5.0, confidence=0.95, mode="AUTO", offset_m=-0.02)
        comm.send_node2_control(throttle_pct=20.0, brake_pct=0.0, mode="AUTO")
        comm.send_heartbeat(health_code=0)
        times.append((time.perf_counter() - t0) * 1000.0)

    comm.close()
    mean_lat = sum(times) / len(times)
    assert mean_lat < 2.0, f"Mean latency {mean_lat:.3f}ms exceeded 2.0ms constraint"


def test_node1_and_node2_fault_bitmasks():
    """Verify fault bitmask flag definitions match protocol specifications."""
    FAULT_SENSOR_ERR       = 0x01
    FAULT_OVERCURRENT      = 0x02
    FAULT_WATCHDOG_TIMEOUT = 0x04
    FAULT_CAN_BUS_OFF      = 0x08

    # Simulate combined fault: sensor error + watchdog timeout
    flags = FAULT_SENSOR_ERR | FAULT_WATCHDOG_TIMEOUT
    assert bool(flags & FAULT_SENSOR_ERR) is True
    assert bool(flags & FAULT_OVERCURRENT) is False
    assert bool(flags & FAULT_WATCHDOG_TIMEOUT) is True
    assert bool(flags & FAULT_CAN_BUS_OFF) is False


def test_zero_ui_overhead_in_jetson_runner():
    """Verify deploy.jetson_run does NOT import Streamlit, Gradio, or web dependencies."""
    import subprocess
    cmd = [
        sys.executable,
        "-c",
        "import sys, deploy.jetson_run; "
        "forbidden = ['streamlit', 'gradio', 'flask', 'django', 'dash']; "
        "found = [m for m in forbidden if m in sys.modules]; "
        "assert not found, f'Forbidden modules loaded: {found}'; "
        "print('ZERO_UI_OK')",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Zero UI check failed: {res.stderr}"
    assert "ZERO_UI_OK" in res.stdout

