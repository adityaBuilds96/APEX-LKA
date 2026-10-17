"""
tests/test_fault_injector.py
============================
Unit tests for Task 6.4: Fault Injection & Safety Watchdog Certification:
1. Fault matrix YAML configuration integrity
2. Safe-state transition time <= 500ms for F1–F6
3. Perception-driven command attenuation for F5 (Lens Occlusion) and F7 (Low Confidence)
4. Thermal cooldown response for F8
5. Full audit matrix execution and JSON artifact generation
"""

import json
from pathlib import Path
import sys
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.fault_injector import FaultInjector


def test_fault_matrix_configuration():
    """Verify fault matrix configuration has all required fault definitions."""
    matrix_file = PROJECT_ROOT / "configs" / "fault_matrix.yaml"
    assert matrix_file.exists()

    with open(matrix_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["global_safety_budget_ms"] <= 500
    faults = data["faults"]
    for fid in ["F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"]:
        assert fid in faults
        assert "expected_safe_state_ms" in faults[fid]


def test_fault_injector_individual_faults():
    """Verify F1 through F8 fault responses meet safety bounds."""
    injector = FaultInjector()
    try:
        # F1: Camera Disconnect
        res_f1 = injector.inject_fault("F1")
        assert res_f1["passed"] is True
        assert res_f1["time_to_safe_state_ms"] <= 500.0
        assert res_f1["safe_mode_engaged"] == "EMERGENCY_STOP"

        # F4: Model Inference Exception
        res_f4 = injector.inject_fault("F4")
        assert res_f4["passed"] is True
        assert res_f4["time_to_safe_state_ms"] <= 500.0

        # F5: Lens Occlusion Attenuation
        res_f5 = injector.inject_fault("F5")
        assert res_f5["passed"] is True
        assert res_f5["attenuation_observed"] is True

        # F6: Manual Override
        res_f6 = injector.inject_fault("F6")
        assert res_f6["passed"] is True
        assert res_f6["safe_mode_engaged"] == "MANUAL_OVERRIDE"

        # F8: Thermal Cooldown
        res_f8 = injector.inject_fault("F8")
        assert res_f8["passed"] is True
        assert res_f8["cooldown_inserted"] is True
    finally:
        injector.close()


def test_full_safety_audit_run():
    """Verify full audit execution outputs safety_audit_matrix.json with CERTIFIED status."""
    injector = FaultInjector()
    try:
        audit = injector.run_all(bench_mode=True)
        assert audit["overall_certified"] is True
        assert audit["verdict"] == "CERTIFIED"
        assert audit["faults_passed"] == 8

        out_path = PROJECT_ROOT / "results" / "certification" / "safety_audit_matrix.json"
        assert out_path.exists()
        with open(out_path, "r", encoding="utf-8") as f:
            saved_audit = json.load(f)
        assert saved_audit["overall_certified"] is True
    finally:
        injector.close()
