"""
tests/test_actuation_mapping.py
===============================
Unit tests for Task 5.4: Electromechanical Actuation Mapping:
1. LUT monotonicity verification
2. Deadband correctness around 0 deg (center)
3. Direction sign inversion (+1 vs -1)
4. Actuator rate limiter (max_rate_counts_per_tick)
5. Physical saturation clamping (counts_max)
6. CAN ID 0x100 bit-packed encoding round-trip (deg -> counts -> bytes -> counts -> deg)
7. CLI execution with --dry-run
"""

import math
from pathlib import Path
import struct
import subprocess
import sys
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.vehicle_comm import (
    ActuationMapper,
    VehicleCommunicator,
    CAN_ID_CMD_NODE1,
    MODE_AUTO,
)
from deploy.calibrate_actuation import validate_lut_monotonicity, calibrate_actuation_profile


def test_lut_monotonicity_validation():
    """Verify LUT monotonicity checker accepts valid sequences and rejects inversions."""
    # Valid monotonic LUT
    valid_lut = [
        [-30.0, -410],
        [-20.0, -260],
        [-10.0, -125],
        [  0.0,    0],
        [ 10.0,  128],
        [ 20.0,  272],
        [ 30.0,  430],
    ]
    is_valid, issues = validate_lut_monotonicity(valid_lut)
    assert is_valid is True
    assert len(issues) == 0

    # Non-monotonic degrees
    bad_deg_lut = [
        [-30.0, -410],
        [-10.0, -260],
        [-20.0, -125],  # Inverted order
        [  0.0,    0],
    ]
    is_valid, issues = validate_lut_monotonicity(bad_deg_lut)
    assert is_valid is False

    # Non-monotonic counts
    bad_count_lut = [
        [-30.0, -410],
        [-20.0, -100],
        [-10.0, -150],  # Inverted count
        [  0.0,    0],
    ]
    is_valid, issues = validate_lut_monotonicity(bad_count_lut)
    assert is_valid is False


def test_deadband_around_zero():
    """Verify small steering angles inside deadband evaluate to 0 counts."""
    mapper = ActuationMapper()
    # At deadband_counts = 2, with slope 12.5 counts/deg,
    # 0.10 deg gives ~1.25 counts -> inside deadband -> 0 counts
    counts_zero = mapper.deg_to_counts(0.0)
    assert counts_zero == 0

    mapper.reset()
    counts_small = mapper.deg_to_counts(0.10)
    assert counts_small == 0

    mapper.reset()
    counts_neg_small = mapper.deg_to_counts(-0.10)
    assert counts_neg_small == 0

    mapper.reset()
    # 1.0 deg gives ~12.5 counts -> well outside deadband
    counts_active = mapper.deg_to_counts(1.0)
    assert counts_active > 0


def test_direction_sign_inversion():
    """Verify setting direction_sign = -1 inverts the mapped count sign."""
    cfg_pos = {"direction_sign": 1, "deadband_counts": 0, "max_rate_counts_per_tick": 1000}
    cfg_neg = {"direction_sign": -1, "deadband_counts": 0, "max_rate_counts_per_tick": 1000}

    map_pos = ActuationMapper(cfg_pos)
    map_neg = ActuationMapper(cfg_neg)

    c_pos = map_pos.deg_to_counts(15.0)
    c_neg = map_neg.deg_to_counts(15.0)

    assert c_pos > 0
    assert c_neg < 0
    assert c_pos == -c_neg


def test_actuator_rate_limiter():
    """Verify max_rate_counts_per_tick smoothly limits sudden step changes."""
    cfg = {"max_rate_counts_per_tick": 25, "deadband_counts": 0}
    mapper = ActuationMapper(cfg)

    # First step: start at 0
    c0 = mapper.deg_to_counts(0.0)
    assert c0 == 0

    # Step to 20 deg (nominally ~272 counts). Slew limit = +25 counts/tick
    c1 = mapper.deg_to_counts(20.0)
    assert c1 == 25

    # Second tick: next +25 counts
    c2 = mapper.deg_to_counts(20.0)
    assert c2 == 50


def test_actuator_saturation():
    """Verify physical actuator output is clamped strictly to [-counts_max, +counts_max]."""
    cfg = {"counts_max": 450, "max_rate_counts_per_tick": 1000, "deadband_counts": 0}
    mapper = ActuationMapper(cfg)

    # Command extreme 50 deg steering
    c = mapper.deg_to_counts(50.0)
    assert c == 450

    mapper.reset()
    c_neg = mapper.deg_to_counts(-50.0)
    assert c_neg == -450


def test_can_encoding_roundtrip():
    """Verify CAN 0x100 encoding round-trip: deg -> counts -> struct bytes -> counts -> deg."""
    mapper = ActuationMapper({"max_rate_counts_per_tick": 1000, "deadband_counts": 0})
    test_angles = [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0]

    for orig_deg in test_angles:
        mapper.reset()
        counts = mapper.deg_to_counts(orig_deg)

        # Pack into CAN 0x100 struct: <hhBBH
        payload = struct.pack("<hhBBH", counts, 0, 240, MODE_AUTO, 12345)
        assert len(payload) == 8

        # Unpack from CAN 0x100
        unpacked_counts, _, _, mode, _ = struct.unpack("<hhBBH", payload)
        assert unpacked_counts == counts

        # Invert mapping back to degrees
        recovered_deg = mapper.counts_to_deg(unpacked_counts)
        assert math.isclose(orig_deg, recovered_deg, abs_tol=0.5), f"Failed for {orig_deg}° -> {recovered_deg}°"


def test_cli_dry_run():
    """Verify CLI execution: python deploy/calibrate_actuation.py --dry-run."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "deploy" / "calibrate_actuation.py"),
        "--dry-run",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"CLI command failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Calibrated Actuation Parameters:" in out
    assert "[DRY-RUN]" in out
