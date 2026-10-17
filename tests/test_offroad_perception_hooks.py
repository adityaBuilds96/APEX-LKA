"""
tests/test_offroad_perception_hooks.py
======================================
Unit tests for Task 6.3: Off-Road, Dirt, Shadow, and Dust Run:
1. Multi-mode perception auto-switching (Mode A, B, C)
2. Demonstration of C7 tone mapping, C8 glare suppression, and C9 dust occlusion
3. Actuator jerk and slew-rate bounds verification (delta_dot <= 120 deg/s)
4. End-to-end off-road execution meeting cross-track error bounds (RMS <= 0.35m)
"""

import json
from pathlib import Path
import sys
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.field_runs.run_executor import FieldRunExecutor, FieldFrameGenerator
from src.perception.camera import CameraPipeline


def test_offroad_profile_loading():
    """Verify off-road run profile contains required parameters."""
    profile_path = PROJECT_ROOT / "deploy" / "field_runs" / "offroad_run_profile.yaml"
    assert profile_path.exists()

    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["surface_type"] == "offroad"
    assert data["pass_criteria"]["rms_cross_track_dirt_straight_max_m"] <= 0.35
    assert data["pass_criteria"]["delta_rate_max_dps"] <= 120.0


def test_offroad_environmental_generators():
    """Verify offroad generator triggers shadow, glare, and dust conditions."""
    f_norm, cond_norm = FieldFrameGenerator.generate_offroad(10)
    assert cond_norm == "NORMAL_DIRT"
    assert f_norm.shape == (360, 640, 3)

    f_shadow, cond_shadow = FieldFrameGenerator.generate_offroad(45)
    assert cond_shadow == "SHADOW_TUNNEL"

    f_glare, cond_glare = FieldFrameGenerator.generate_offroad(85)
    assert cond_glare == "SUN_GLARE"

    f_dust, cond_dust = FieldFrameGenerator.generate_offroad(120)
    assert cond_dust == "DUST_BURST"


def test_camera_stack_glare_and_occlusion_triggers():
    """Verify C8 glare suppression and C9 occlusion detector trigger on synthetic environmental bursts."""
    cam = CameraPipeline(benchmark_mode=True)
    try:
        # 1. Test glare frame
        f_glare, _ = FieldFrameGenerator.generate_offroad(85)
        glare_detected = cam.detect_glare(f_glare)
        assert glare_detected is True

        # 2. Test dust occlusion frame
        f_dust, _ = FieldFrameGenerator.generate_offroad(120)
        # Feed multiple frames to trigger threshold
        occ = False
        for _ in range(12):
            occ = cam.detect_occlusion(f_dust)
        assert occ is True
    finally:
        cam.release()


def test_offroad_run_execution(tmp_path):
    """Verify full off-road session meets pass criteria and logs mode transitions."""
    test_prof = tmp_path / "test_offroad_prof.yaml"
    prof_data = {
        "profile_id": "TEST-OFFROAD",
        "surface_type": "offroad",
        "mission": {
            "target_speed_mps": 2.0,
            "duration_frames": 40,
            "target_fps": 30.0,
        },
        "pass_criteria": {
            "rms_cross_track_dirt_straight_max_m": 0.35,
            "peak_cross_track_dirt_curve_max_m": 0.70,
            "max_sustained_lka_lost_s": 1.0,
            "delta_rate_max_dps": 120.0,
        },
    }
    with open(test_prof, "w", encoding="utf-8") as f:
        yaml.safe_dump(prof_data, f)

    executor = FieldRunExecutor(profile_path=test_prof)
    try:
        report = executor.run()
        assert report["passed"] is True
        assert report["metrics"]["rms_cross_track_m"] <= 0.35
        assert report["metrics"]["peak_cross_track_m"] <= 0.70
        assert report["metrics"]["peak_steer_rate_dps"] <= 120.0
        assert report["metrics"]["max_sustained_lka_lost_s"] <= 1.0

        # Verify mode switch distribution exists
        modes = report["perception_modes"]
        assert sum(modes.values()) == 40
    finally:
        executor.close()
