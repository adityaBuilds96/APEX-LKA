"""
tests/test_fuzz_tester.py
=========================
Unit tests for APEX-LKA Monte Carlo Fuzz Testing Framework.
Verifies stochastic perturbations, statistical distributions,
reproducibility guarantee, and JSON reporting.
"""

from pathlib import Path
import json
import pytest

from src.simulation.fuzz_tester import (
    MonteCarloFuzzTester,
    FuzzReport,
)
from src.simulation.scenario_executor import ScenarioExecutor


@pytest.fixture
def fuzzer():
    """Provides a clean MonteCarloFuzzTester in mock simulation mode."""
    executor = ScenarioExecutor(mock_mode=True)
    return MonteCarloFuzzTester(executor=executor, default_seed=42)


def test_monte_carlo_fuzz_execution(fuzzer):
    """Verify fuzz tester runs iterations and calculates statistical metrics."""
    scen_path = Path("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml")
    assert scen_path.exists()

    report = fuzzer.fuzz_scenario(scen_path, num_iterations=10, seed=42)
    assert isinstance(report, FuzzReport)
    assert report.total_iterations == 10
    assert report.passed_count + report.failed_count == 10
    assert 0.0 <= report.collision_probability <= 1.0
    assert "mean_m" in report.min_distance_stats
    assert "p05_s" in report.min_ttc_stats


def test_monte_carlo_determinism_reproducibility(fuzzer):
    """Verify identical random seeds produce identical pass rates and statistics."""
    scen_path = Path("scenarios/golden_test_suite/cutin_01_aggressive_left.yaml")

    rep1 = fuzzer.fuzz_scenario(scen_path, num_iterations=5, seed=1234)
    rep2 = fuzzer.fuzz_scenario(scen_path, num_iterations=5, seed=1234)

    assert rep1.pass_rate_pct == rep2.pass_rate_pct
    assert rep1.collision_probability == rep2.collision_probability
    assert rep1.min_distance_stats["mean_m"] == rep2.min_distance_stats["mean_m"]


def test_export_fuzz_report_json(fuzzer, tmp_path):
    """Verify exporting fuzz report to JSON file on disk."""
    scen_path = Path("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml")
    report = fuzzer.fuzz_scenario(scen_path, num_iterations=5, seed=77)

    out_file = tmp_path / "fuzz_report.json"
    fuzzer.export_report_json(report, out_file)
    assert out_file.exists()

    with open(out_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data["total_iterations"] == 5
    assert "min_distance_stats" in data
