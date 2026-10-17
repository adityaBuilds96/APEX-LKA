"""
tests/test_calibration_extrinsics.py
====================================
Unit tests for Task 5.1: Camera Extrinsic Calibration:
1. Synthetic extrinsics round-trip (pitch -> analytic horizon computation)
2. YAML serialization & deserialization round-trip
3. Validation bounds enforcement (negative height, pitch > 60 deg)
4. Roll angle sanity warning (> 5 deg)
5. Driving vanishing point calibration
6. Headless CLI execution with --dry-run and --synthetic
"""

import math
from pathlib import Path
import subprocess
import sys
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CameraExtrinsics
from deploy.calibrate_extrinsics import (
    compute_analytic_horizon,
    calibrate_synthetic,
    calibrate_from_vanishing_point,
    run_calibration,
)


def test_synthetic_extrinsics_roundtrip():
    """Verify analytic horizon calculation: y_horizon = cy - fy * tan(pitch)."""
    cy = 180.0
    fy = 500.0

    # 1. Zero pitch -> horizon exactly at image center (cy)
    pitch_zero = 0.0
    y_h0 = compute_analytic_horizon(cy, fy, pitch_zero)
    assert y_h0 == 180

    # 2. Down-positive pitch of 5 deg (~0.087266 rad)
    # y = 180 - 500 * tan(5 deg) = 180 - 500 * 0.08748866 = 180 - 43.74 = 136
    pitch_5deg = math.radians(5.0)
    y_h5 = compute_analytic_horizon(cy, fy, pitch_5deg)
    assert y_h5 == 136

    # 3. Down-positive pitch of 10 deg (~0.17453 rad)
    # y = 180 - 500 * tan(10 deg) = 180 - 500 * 0.17632698 = 180 - 88.16 = 92
    pitch_10deg = math.radians(10.0)
    y_h10 = compute_analytic_horizon(cy, fy, pitch_10deg)
    assert y_h10 == 92


def test_camera_extrinsics_dataclass_methods():
    """Verify CameraExtrinsics dataclass computation and validation."""
    ext = CameraExtrinsics(
        mount_height_m=1.20,
        pitch_rad=math.radians(5.0),
        roll_rad=0.0,
        yaw_rad=0.0,
    )
    y_h = ext.compute_horizon_y(cy=180.0, fy=500.0)
    assert y_h == 136

    is_valid, issues = ext.validate(image_height=360, cy=180.0, fy=500.0)
    assert is_valid is True
    assert len(issues) == 0


def test_yaml_persistence_roundtrip(tmp_path):
    """Verify saving and loading camera extrinsics YAML reproduces identical parameters."""
    cfg_file = tmp_path / "camera_extrinsics.yaml"
    orig = CameraExtrinsics(
        mount_height_m=1.35,
        pitch_rad=0.087266,
        roll_rad=0.012,
        yaw_rad=-0.005,
        x_offset_m=0.02,
        z_offset_m=0.45,
        y_horizon_px=136,
        calibration_method="unit_test",
        timestamp="2026-10-02T19:30:00Z",
    )
    orig.save_yaml(cfg_file)
    assert cfg_file.exists()

    loaded = CameraExtrinsics.from_yaml(cfg_file)
    assert math.isclose(loaded.mount_height_m, orig.mount_height_m, rel_tol=1e-5)
    assert math.isclose(loaded.pitch_rad, orig.pitch_rad, rel_tol=1e-5)
    assert math.isclose(loaded.roll_rad, orig.roll_rad, rel_tol=1e-5)
    assert math.isclose(loaded.yaw_rad, orig.yaw_rad, rel_tol=1e-5)
    assert math.isclose(loaded.x_offset_m, orig.x_offset_m, rel_tol=1e-5)
    assert math.isclose(loaded.z_offset_m, orig.z_offset_m, rel_tol=1e-5)
    assert loaded.y_horizon_px == orig.y_horizon_px
    assert loaded.calibration_method == "unit_test"


def test_rejects_impossible_values():
    """Verify validator rejects negative mount height and extreme pitch angles > 60 deg."""
    # Negative height
    ext_neg_h = CameraExtrinsics(mount_height_m=-0.5, pitch_rad=0.0)
    is_valid, issues = ext_neg_h.validate()
    assert is_valid is False
    assert any("mount_height_m must be > 0" in i for i in issues)

    # Zero height
    ext_zero_h = CameraExtrinsics(mount_height_m=0.0, pitch_rad=0.0)
    is_valid, issues = ext_zero_h.validate()
    assert is_valid is False

    # Pitch > 60 degrees (1.0472 rad)
    ext_extreme_pitch = CameraExtrinsics(mount_height_m=1.2, pitch_rad=math.radians(65.0))
    is_valid, issues = ext_extreme_pitch.validate()
    assert is_valid is False
    assert any("pitch_rad exceeds 60 deg ceiling" in i for i in issues)


def test_roll_angle_warning():
    """Verify roll angle >= 5 degrees issues a warning but does not mark configuration as fatal invalid."""
    ext_roll = CameraExtrinsics(
        mount_height_m=1.2,
        pitch_rad=math.radians(5.0),
        roll_rad=math.radians(6.5),  # > 5 deg
    )
    is_valid, issues = ext_roll.validate()
    assert is_valid is True  # Warning only, not fatal
    assert any("Warning: absolute roll_rad exceeds 5 deg" in i for i in issues)


def test_calibrate_from_vanishing_point_synthetic():
    """Verify vanishing point calculation accurately extracts pitch from converging lines."""
    fx, fy, cx, cy = 500.0, 500.0, 320.0, 180.0
    # Two straight road lines converging at (vx=320, vy=136.26)
    # cy - vy = 180 - 136.26 = 43.74 -> pitch = atan(43.74 / 500) = 5.0 deg
    lines = [
        (120.0, 360.0, 320.0, 136.26),
        (520.0, 360.0, 320.0, 136.26),
    ]
    ext, success = calibrate_from_vanishing_point(lines, fx, fy, cx, cy, mount_height_m=1.20)
    assert success is True
    assert math.isclose(math.degrees(ext.pitch_rad), 5.0, abs_tol=0.2)
    assert ext.y_horizon_px == 136


def test_cli_dry_run_synthetic():
    """Verify CLI execution: python deploy/calibrate_extrinsics.py --dry-run --synthetic."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "deploy" / "calibrate_extrinsics.py"),
        "--dry-run",
        "--synthetic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"CLI command failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Calibrated Extrinsics:" in out
    assert "[DRY-RUN]" in out
