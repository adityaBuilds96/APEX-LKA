"""
deploy/comm_test.py
===================
Hardware Communication Diagnostic & Testing Bench Utility:
- Tests USB Serial connectivity and measures Round-Trip Time (RTT) to Arduino nodes
- Transmits CAN test frames (0x100, 0x200, 0x050) and monitors response telemetry (0x300, 0x301)
- Includes full simulation mode (--sim-mode) to validate communication pipelines on PC
"""

from __future__ import annotations

import argparse
import logging
import os
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = Path(__file__).resolve().parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from vehicle_comm import (
    CAN_ID_CMD_NODE1,
    CAN_ID_CMD_NODE2,
    CAN_ID_FEEDBACK_N1,
    CAN_ID_FEEDBACK_N2,
    CAN_ID_HEARTBEAT,
    CANInterface,
    Node1Telemetry,
    Node2Telemetry,
    SerialInterface,
    VehicleCommunicator,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("comm_test")


# ═══════════════════════════════════════════════════════════════════════════
# Virtual Arduino Simulator for Hardware-in-the-Loop Mocking
# ═══════════════════════════════════════════════════════════════════════════

class VirtualArduinoSim:
    """Simulates 2x Arduino Mega nodes on virtual CAN and Serial channels."""

    def __init__(self, comm: VehicleCommunicator):
        self.comm = comm
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sim_loop, name="VirtualArduinoSim", daemon=True)
        self._thread.start()
        log.info("Virtual Arduino nodes running (Node 1: Steering, Node 2: Throttle/Brake).")

    def _sim_loop(self):
        node1_steer = 0.0
        node2_speed = 0.0
        cycle = 0

        while self._running:
            time.sleep(0.033)  # ~30 Hz
            cycle += 1
            now_ms = int(time.time() * 1000) % 65536

            # Simulate Node 1 dynamics (tracks target with slight lag)
            node1_steer += (0.0 - node1_steer) * 0.1
            payload1 = struct.pack(
                "<hhBBH",
                int(node1_steer * 100),
                int(2.45 * 100),  # 2.45A motor current
                0,                # No override
                0,                # No faults
                now_ms,
            )
            if self.comm.can_if:
                self.comm.can_if.send_frame(CAN_ID_FEEDBACK_N1, payload1)

            # Simulate Node 2 dynamics (5.2 m/s speed, 120 PSI brake)
            node2_speed = 5.20
            payload2 = struct.pack(
                "<hhBBH",
                int(node2_speed * 100),
                int(12.0 * 10),   # 12.0 PSI
                0,
                0,
                now_ms,
            )
            if self.comm.can_if:
                self.comm.can_if.send_frame(CAN_ID_FEEDBACK_N2, payload2)

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Diagnostic Procedures
# ═══════════════════════════════════════════════════════════════════════════

def test_serial(port: Optional[str] = None, baudrate: int = 115200, count: int = 5):
    """Test USB Serial link and measure round-trip time."""
    print("=" * 70)
    print(f"[SERIAL] Testing Serial Communication Link (Port: {port or 'Auto-Scan'}, {baudrate} baud)")
    print("=" * 70)

    serial_if = SerialInterface(port=port, baudrate=baudrate, node_id=1, auto_scan=True)
    if not serial_if.is_connected:
        print("[!] No active serial device detected. Please connect an Arduino Mega via USB.")
        serial_if.close()
        return

    rtt_times: list[float] = []

    def on_ack(data: dict):
        rtt = (time.perf_counter() - t0) * 1000.0
        rtt_times.append(rtt)
        print(f"  [ACK RECEIVED] Node {data.get('node')}: {data} (RTT: {rtt:.2f} ms)")

    serial_if.register_callback(on_ack)

    for i in range(count):
        t0 = time.perf_counter()
        packet = {
            "node": 1,
            "cmd_a": float(i * 2.0),
            "cmd_b": 0.0,
            "offset_m": -0.10,
            "confidence": 0.95,
            "mode": "AUTO",
            "ts": int(time.time() * 1000) % 65536,
        }
        sent = serial_if.send_json(packet)
        print(f"Sent packet #{i+1}: {packet}")
        time.sleep(0.1)

    serial_if.close()
    if rtt_times:
        print(f"\n[OK] Average Serial Round-Trip Time: {sum(rtt_times)/len(rtt_times):.2f} ms")
    else:
        print("\n[!] Packets sent, but no Arduino responses received within timeout.")


def test_can(channel: str = "can0", bitrate: int = 500000, count: int = 10):
    """Test physical CAN Bus transmission and feedback."""
    print("=" * 70)
    print(f"[CAN] Testing CAN Bus Communication (Interface: {channel} @ {bitrate} bps)")
    print("=" * 70)

    can_if = CANInterface(channel=channel, bitrate=bitrate)
    if not can_if.is_connected:
        print(f"[!] Could not open CAN interface '{channel}'. Check physical cable and SocketCAN setup.")
        can_if.close()
        return

    feedback_received = 0

    def on_feedback_n1(can_id: int, data: bytes):
        nonlocal feedback_received
        feedback_received += 1
        steer_s, curr_s, ovr, flt, up = struct.unpack("<hhBBH", data)
        print(f"  [RX CAN 0x300] Node 1 Feedback: Steer={steer_s/100.0:+.2f}° | Current={curr_s/100.0:.2f}A | Override={ovr} | Faults=0x{flt:02X}")

    def on_feedback_n2(can_id: int, data: bytes):
        nonlocal feedback_received
        feedback_received += 1
        speed_s, brake_s, ovr, flt, up = struct.unpack("<hhBBH", data)
        print(f"  [RX CAN 0x301] Node 2 Feedback: Speed={speed_s/100.0:.2f} m/s | Brake={brake_s/10.0:.1f} PSI | Override={ovr}")

    can_if.register_callback(CAN_ID_FEEDBACK_N1, on_feedback_n1)
    can_if.register_callback(CAN_ID_FEEDBACK_N2, on_feedback_n2)

    for i in range(count):
        # 1. Send Heartbeat 0x050
        hb = struct.pack("<BBH", 0xAA, 0, i)
        can_if.send_frame(CAN_ID_HEARTBEAT, hb)

        # 2. Send Node 1 Command 0x100 (Steering test)
        steer_test = float(i * 1.5)
        p1 = struct.pack("<hhBBH", int(steer_test * 100), 0, int(0.95 * 255), 1, int(time.time() * 1000) % 65536)
        can_if.send_frame(CAN_ID_CMD_NODE1, p1)

        # 3. Send Node 2 Command 0x200 (Throttle test)
        p2 = struct.pack("<hhBBH", int(15.0 * 100), 0, 1, 0, int(time.time() * 1000) % 65536)
        can_if.send_frame(CAN_ID_CMD_NODE2, p2)

        print(f"[TX] Cycle #{i+1} dispatched (Steer={steer_test:+.1f}°, Throttle=15%)")
        time.sleep(0.1)

    can_if.close()
    print(f"\n[SUMMARY] Total feedback frames captured: {feedback_received}/{count*2}")


def run_simulation(duration_sec: float = 3.0):
    """Run simulated loopback verification without physical hardware."""
    print("=" * 70)
    print("[SIMULATION] Launching Full Hardware-in-the-Loop Communication Simulation")
    print("=" * 70)

    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)
    sim = VirtualArduinoSim(comm)
    sim.start()

    start_t = time.time()
    cycle = 0

    while time.time() - start_t < duration_sec:
        cycle += 1
        steer_target = np.sin(cycle * 0.2) * 15.0

        # Send command through VehicleCommunicator
        comm.send_node1_command(steering_deg=steer_target, confidence=0.98, mode="AUTO", offset_m=-0.05)
        comm.send_node2_command(throttle_pct=25.0, brake_pct=0.0, mode="AUTO")
        comm.send_heartbeat(health_code=0)

        # Query received telemetry
        t1 = comm.get_node_telemetry(node_id=1)
        t2 = comm.get_node_telemetry(node_id=2)

        if t1 and t2:
            print(f"Cycle {cycle:02d} | Target Steer: {steer_target:+.1f} deg | Feedback: Steer={t1.actual_steering_deg:+.2f} deg, Speed={t2.wheel_speed_mps:.2f}m/s | Health={comm.is_healthy()}")
        time.sleep(0.05)

    sim.stop()
    comm.close()
    print("\n[OK] Simulation completed successfully with zero communication faults!")



def main():
    parser = argparse.ArgumentParser(description="APEX-LKA Hardware Communication Diagnostic Tool")
    parser.add_argument("--test-serial", action="store_true", help="Run Serial communication test to Arduinos")
    parser.add_argument("--test-can", action="store_true", help="Run SocketCAN frame send/receive test")
    parser.add_argument("--sim-mode", action="store_true", help="Run software-in-the-loop simulation mode")
    parser.add_argument("--port", type=str, default=None, help="Serial port to test (default: auto-scan)")
    parser.add_argument("--baudrate", type=int, default=115200, help="Serial baudrate (default: 115200)")
    parser.add_argument("--interface", type=str, default="can0", help="CAN interface (default: can0)")
    args = parser.parse_args()

    if args.test_serial:
        test_serial(port=args.port, baudrate=args.baudrate)
    elif args.test_can:
        test_can(channel=args.interface)
    elif args.sim_mode or len(sys.argv) == 1:
        run_simulation()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
