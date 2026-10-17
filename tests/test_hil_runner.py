"""
tests/test_hil_runner.py
========================
Unit tests for Task 6.1: Hardware-In-The-Loop (HIL) Bench Testing:
1. Scenario manifest loading and structure validation
2. Latency budget compliance on synthetic test clips
3. CAN 0x100 / 0x200 encoding assertions against actuation profile
4. End-to-end execution of HILBenchRunner on synthetic scenario
"""

import json
from pathlib import Path
import sys
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.hil_runner import HILBenchRunner, HILSyntheticPatternGenerator
from deploy.vehicle_comm import VehicleCommunicator, ActuationMapper


def test_scenario_manifest_loader():
    """Verify HIL sequence manifest contains all required scenarios and parameters."""
    manifest_path = PROJECT_ROOT / "deploy" / "hil_scenarios" / "hil_sequence_manifest.yaml"
    assert manifest_path.exists(), f"Missing manifest: {manifest_path}"

    with open(manifest_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert "scenarios" in data
    assert len(data["scenarios"]) >= 4

    required_keys = ["id", "name", "synthetic_pattern", "duration_frames", "target_fps", "latency_budget_p99_ms"]
    for scen in data["scenarios"]:
        for k in required_keys:
            assert k in scen, f"Scenario {scen.get('id')} missing key '{k}'"
        assert scen["target_fps"] >= 30.0
        assert scen["latency_budget_p99_ms"] <= 40.0


def test_synthetic_pattern_generator():
    """Verify synthetic pattern generator produces valid BGR frames."""
    patterns = ["straight", "left_curve", "right_curve", "broken_paint"]
    for pat in patterns:
        img = HILSyntheticPatternGenerator.generate(pat, frame_idx=10, width=640, height=360)
        assert img is not None
        assert img.shape == (360, 640, 3)
        assert img.dtype.name == "uint8"


def test_can_actuation_encoding_assertions():
    """Verify CAN frame bit-packing matches ActuationMapper values."""
    import struct
    mapper = ActuationMapper()
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)

    steer_deg = 12.5
    counts = mapper.deg_to_counts(steer_deg)
    assert counts > 0

    success = comm.send_node1_control(
        steering_deg=steer_deg,
        confidence=0.92,
        mode="AUTO",
        offset_m=0.15,
        raw_counts=counts,
    )
    assert success is True

    # Test Node 1 feedback decoding (CAN ID 0x300)
    steer_scaled = int(steer_deg * 100)
    curr_scaled = 250  # 2.50 A
    override = 0
    fault = 0
    uptime_ms = 1200
    data_0x300 = struct.pack("<hhBBH", steer_scaled, curr_scaled, override, fault, uptime_ms)
    comm.can_if.send_frame(0x300, data_0x300)

    t1 = comm.get_node_telemetry(node_id=1)
    assert t1 is not None
    assert abs(t1.actual_steering_deg - steer_deg) < 0.1
    assert t1.motor_current_amps == 2.50
    assert t1.override_active is False

    comm.close()



def test_hil_single_scenario_latency_budget(tmp_path):
    """Verify single scenario execution meets latency and zero-exception criteria."""
    manifest_file = tmp_path / "test_manifest.yaml"
    manifest_data = {
        "version": "1.0.0",
        "scenarios": [
            {
                "id": "TEST-01",
                "name": "Synthetic Straight Test",
                "synthetic_pattern": "straight",
                "duration_frames": 15,
                "target_fps": 30.0,
                "latency_budget_p99_ms": 40.0,
                "latency_budget_p50_ms": 18.0,
                "steering_range_deg": [-5.0, 5.0],
            }
        ]
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_data, f)

    runner = HILBenchRunner(manifest_path=manifest_file)
    try:
        report = runner.run_all_scenarios()
        assert report["overall_pass"] is True
        assert report["total_scenarios"] == 1
        scen_res = report["scenarios"][0]
        assert scen_res["exceptions_count"] == 0
        assert scen_res["watchdog_trips"] == 0
        assert scen_res["latency_p99_ms"] <= 40.0
    finally:
        runner.close()
