"""
tests/test_carla_simulation.py
==============================
Phase 6 Section 7 Verification Suite:
1. Golden Scenario Runner executes Scenarios 01-04 and detects safety violations in crash runs.
2. Monte Carlo Fuzzing Engine runs 100 iterations and computes safety margin statistics.
3. Synthetic Data Auto-Generator exports paired images, 2D/3D boxes, and lane polynomials to data/processed/synthetic.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from src.simulation.scenario_runner import GoldenScenarioRunner, ScenarioTestReport
from src.simulation.fuzzer import MonteCarloFuzzEngine, MonteCarloFuzzReport
from src.simulation.synthetic_data_gen import SyntheticDataAutoGenerator, SyntheticSampleMetadata


# ==============================================================================
# 1. Golden Scenario Runner & Safety Violation Detection
# ==============================================================================

def test_scenario_runner_detects_safety_violation():
    """
    Verify scenario runner correctly catches collisions and safety margin breaches.
    """
    runner = GoldenScenarioRunner()

    # Nominal run should pass with emergency RSS braking
    nominal_report = runner.run_scenario("SCENARIO_01", sim_duration_s=2.0)
    assert nominal_report.scenario_id == "SCENARIO_01"
    assert nominal_report.collision_occurred is False
    assert nominal_report.passed is True
    assert nominal_report.min_safety_margin_m > 0.5

    # Forced failure scenario simulating an unmitigated crash
    crash_report = runner.run_scenario("SCENARIO_01", sim_duration_s=2.0, force_failure=True)
    assert crash_report.passed is False
    assert crash_report.collision_occurred is True
    assert len(crash_report.failure_reasons) > 0


# ==============================================================================
# 2. Monte Carlo Fuzzing Engine (100 Iterations)
# ==============================================================================

def test_monte_carlo_fuzz_campaign_100_runs():
    """
    Verify Monte Carlo fuzzer executes 100 continuous iterations and compiles
    safety margin distributions.
    """
    fuzzer = MonteCarloFuzzEngine(random_seed=123)

    report = fuzzer.run_fuzz_campaign(num_iterations=100, base_speed_kmh=45.0)

    assert report.total_iterations == 100
    assert report.passed_count + report.failed_count == 100
    assert report.pass_rate_pct >= 0.0
    assert report.mean_margin_m > 0.0
    assert report.min_margin_m <= report.mean_margin_m <= 100.0
    assert len(report.iterations) == 100


# ==============================================================================
# 3. Synthetic Data Generator Export Schema
# ==============================================================================

def test_synthetic_data_generator_schema(tmp_path):
    """
    Validate synthetic data generator exports valid structural schema:
    paired RGB images, 2D/3D bounding boxes, and lane polynomials.
    """
    gen = SyntheticDataAutoGenerator(output_dir=tmp_path / "synthetic")

    meta = gen.generate_and_export_sample(frame_idx=1, width=640, height=360)

    assert isinstance(meta, SyntheticSampleMetadata)
    assert meta.sample_id == "synth_000001"

    # Verify image exists on disk
    img_path = tmp_path / "synthetic" / "images" / meta.image_file
    assert img_path.exists()
    assert img_path.stat().st_size > 500

    # Verify metadata JSON exists and matches schema
    lbl_path = tmp_path / "synthetic" / "labels" / f"{meta.sample_id}.json"
    assert lbl_path.exists()

    data = json.loads(lbl_path.read_text(encoding="utf-8"))
    assert data["sample_id"] == "synth_000001"
    assert "boxes_2d" in data and len(data["boxes_2d"]) > 0
    assert "boxes_3d" in data and len(data["boxes_3d"]) > 0
    assert "lane_polynomials" in data
    assert "center_lane" in data["lane_polynomials"]
    assert len(data["lane_polynomials"]["center_lane"]) == 3
