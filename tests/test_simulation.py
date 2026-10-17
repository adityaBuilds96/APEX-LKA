"""
tests/test_simulation.py
========================
End-to-End Integration Suite for APEX-LKA Phase 5 Section 7:
Simulation, Digital Twin, Golden Scenarios, Synthetic Generation,
Monte Carlo Fuzz Testing, and CI Regression.
"""

from pathlib import Path
import json
import pytest

from src.simulation.carla_bridge import CARLABridge
from src.simulation.digital_twin import DigitalTwinLoader
from src.simulation.scenario_executor import ScenarioExecutor, ScenarioResult
from src.simulation.synthetic_generator import SyntheticDataGenerator
from src.simulation.fuzz_tester import MonteCarloFuzzTester
from src.simulation.regression_runner import SimulationRegressionRunner


@pytest.fixture
def sim_stack(tmp_path):
    """Provides a fully wired simulation stack in mock mode."""
    bridge = CARLABridge(mock_mode=True, synchronous=True)
    dt = DigitalTwinLoader(maps_dir="simulation/maps")
    executor = ScenarioExecutor(bridge=bridge, digital_twin=dt, mock_mode=True)
    fuzzer = MonteCarloFuzzTester(executor=executor, default_seed=42)
    synth = SyntheticDataGenerator(output_dir=tmp_path / "synth_data", digital_twin=dt)
    regression = SimulationRegressionRunner(
        scenario_dir="scenarios/golden_test_suite",
        executor=executor,
        min_pass_rate_pct=95.0,
    )
    yield {
        "bridge": bridge,
        "dt": dt,
        "executor": executor,
        "fuzzer": fuzzer,
        "synth": synth,
        "regression": regression,
        "tmp_path": tmp_path,
    }
    bridge.cleanup()


def test_e2e_carla_bridge_lifecycle(sim_stack):
    """Verify CARLA bridge full lifecycle from connect to sensor acquisition."""
    bridge = sim_stack["bridge"]
    assert bridge.connect() is True
    bridge.spawn_ego_vehicle()
    bridge.setup_sensors()
    bridge.tick()

    frame = bridge.get_camera_frame()
    assert frame is not None
    assert frame.shape == (360, 640, 3)

    state = bridge.get_vehicle_state()
    assert state.speed_mps >= 0.0


def test_e2e_digital_twin_and_weather_variants(sim_stack):
    """Verify loading both digital twin tracks and applying weather presets."""
    dt = sim_stack["dt"]
    campus = dt.load_map("college_campus")
    baja = dt.load_map("baja_track_2027")

    assert campus.roads_count >= 1
    assert baja.total_length_m > 100.0

    for w in ("clear_noon", "rain_moderate", "fog_dense", "dust_cloud_heavy"):
        params = dt.set_weather(w)
        assert isinstance(params, dict)


def test_e2e_golden_scenario_execution(sim_stack):
    """Verify executing multiple scenario categories with pass/fail checks."""
    executor = sim_stack["executor"]

    # Pedestrian
    res1 = executor.run_scenario("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml", max_duration_s=4.0)
    assert res1.passed is True
    assert res1.collision_occurred is False

    # Vehicle Cutin
    res2 = executor.run_scenario("scenarios/golden_test_suite/cutin_01_aggressive_left.yaml", max_duration_s=4.0)
    assert res2.passed is True

    # Track Obstacle
    res3 = executor.run_scenario("scenarios/golden_test_suite/obs_01_tire_in_lane.yaml", max_duration_s=4.0)
    assert res3.passed is True


def test_e2e_synthetic_dataset_generation(sim_stack):
    """Verify synthetic dataset batch generation and manifest serialization."""
    synth = sim_stack["synth"]
    manifest = synth.generate_batch(num_frames=15, weather="clear_noon")
    assert manifest["total_samples"] == 15
    assert manifest["target_met"] is True
    assert (synth.output_dir / "dataset_manifest.json").exists()


def test_e2e_monte_carlo_fuzz_testing(sim_stack):
    """Verify Monte Carlo fuzz testing generates statistical safety metrics."""
    fuzzer = sim_stack["fuzzer"]
    report = fuzzer.fuzz_scenario(
        "scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml",
        num_iterations=8,
        seed=101,
    )
    assert report.total_iterations == 8
    assert report.pass_rate_pct >= 85.0
    assert 0.0 <= report.collision_probability <= 0.15


def test_e2e_ci_regression_runner_and_html_report(sim_stack):
    """Verify regression runner executes golden suite, enforces 95% pass rate, and builds HTML report."""
    runner = sim_stack["regression"]
    html_out = sim_stack["tmp_path"] / "test_report.html"

    # Run small category subset for fast integration assertion
    summary = runner.run_all(max_duration_per_scenario_s=2.0, category_filter="ped_0")
    assert summary["total_scenarios"] > 0
    assert summary["pass_rate_pct"] >= 95.0
    assert summary["threshold_met"] is True

    report_path = runner.generate_html_report(summary, output_path=html_out)
    assert report_path.exists()
    assert report_path.stat().st_size > 1000
