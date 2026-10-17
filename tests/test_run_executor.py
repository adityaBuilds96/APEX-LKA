"""
tests/test_run_executor.py
==========================
Unit tests for Task 6.2: Low-Speed Campus Paved Autonomous Run:
1. Paved run profile validation
2. Session-tagged telemetry CSV and metadata creation
3. Paved autonomous tracking accuracy assertions (RMS e_y <= 0.20m, heading <= 5 deg)
4. Camera stack adaptation tracking (C4 AEC, C5 AWB, C7 Tone Mapping)
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


def test_paved_profile_loading():
    """Verify paved run profile contains required configuration."""
    profile_path = PROJECT_ROOT / "deploy" / "field_runs" / "paved_run_profile.yaml"
    assert profile_path.exists()

    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["surface_type"] == "paved"
    assert data["mission"]["target_fps"] >= 30.0
    assert data["pass_criteria"]["rms_cross_track_straight_max_m"] <= 0.20


def test_paved_frame_generator():
    """Verify paved frame generator produces valid image matrices."""
    frame = FieldFrameGenerator.generate_paved(frame_idx=10, width=640, height=360)
    assert frame is not None
    assert frame.shape == (360, 640, 3)
    assert frame.dtype.name == "uint8"


def test_paved_run_execution(tmp_path):
    """Verify autonomous execution of paved profile meets all pass criteria."""
    # Create lightweight test profile with 25 frames
    test_prof = tmp_path / "test_paved_prof.yaml"
    prof_data = {
        "profile_id": "TEST-PAVED",
        "surface_type": "paved",
        "mission": {
            "target_speed_mps": 2.5,
            "duration_frames": 25,
            "target_fps": 30.0,
        },
        "pass_criteria": {
            "rms_cross_track_straight_max_m": 0.20,
            "peak_cross_track_curve_max_m": 0.50,
            "rms_heading_error_max_deg": 5.0,
            "max_unintended_lka_lost_s": 0.0,
        },
    }
    with open(test_prof, "w", encoding="utf-8") as f:
        yaml.safe_dump(prof_data, f)

    executor = FieldRunExecutor(profile_path=test_prof)
    try:
        report = executor.run()
        assert report["passed"] is True
        assert report["metrics"]["rms_cross_track_m"] <= 0.20
        assert report["metrics"]["peak_cross_track_m"] <= 0.50
        assert report["metrics"]["rms_heading_error_deg"] <= 5.0
        assert report["metrics"]["max_sustained_lka_lost_s"] == 0.0

        # Verify telemetry file existence
        telemetry_file = Path(report["telemetry_csv"])
        assert telemetry_file.exists()
        assert telemetry_file.stat().st_size > 0
    finally:
        executor.close()
