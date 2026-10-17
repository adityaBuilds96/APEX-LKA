"""
tests/test_scenario_executor.py
===============================
Unit tests for APEX-LKA Golden Scenario Executor.
Verifies YAML parsing, actor trajectory stepping, safety assertion evaluation,
and pass/fail classification.
"""

from pathlib import Path
import pytest
import yaml

from src.simulation.scenario_executor import (
    ScenarioExecutor,
    ScenarioDefinition,
    ScenarioResult,
    ActorConfig,
)
from src.simulation.carla_bridge import CARLABridge
from src.simulation.digital_twin import DigitalTwinLoader


@pytest.fixture
def executor():
    """Provides an isolated ScenarioExecutor in mock simulation mode."""
    bridge = CARLABridge(mock_mode=True, synchronous=True)
    dt = DigitalTwinLoader(maps_dir="simulation/maps")
    return ScenarioExecutor(bridge=bridge, digital_twin=dt, mock_mode=True)


def test_scenario_manifest_structure_and_completeness():
    """Verify scenarios/golden_test_suite/manifest.yaml contains 101 scenarios across 6 categories."""
    manifest_path = Path("scenarios/golden_test_suite/manifest.yaml")
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    assert doc["total_scenarios"] == 101
    assert len(doc["scenarios"]) == 101

    cats = doc["categories"]
    assert cats["PEDESTRIAN_INTERACTIONS"] == 20
    assert cats["VEHICLE_CUTINS"] == 20
    assert cats["LANE_CLOSURES"] == 15
    assert cats["WEATHER_SCENARIOS"] == 15
    assert cats["TRACK_OBSTACLES"] == 15
    assert cats["EMERGENCY_EVENTS"] == 16


def test_load_scenario_yaml_parsing(executor):
    """Verify loading and parsing of a Golden Scenario YAML file."""
    scen_path = Path("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml")
    assert scen_path.exists()

    scen = executor.load_scenario(scen_path)
    assert isinstance(scen, ScenarioDefinition)
    assert scen.scenario_id == "PED_01"
    assert scen.category == "PEDESTRIAN_INTERACTION"
    assert scen.map_name == "college_campus"
    assert scen.weather == "clear_noon"
    assert len(scen.actors) == 1
    assert scen.actors[0].behavior == "cross"


def test_execute_pedestrian_crossing_success(executor):
    """Verify vehicle safely detects and brakes for pedestrian crossing ahead."""
    scen_path = Path("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml")
    res = executor.run_scenario(scen_path, max_duration_s=5.0)

    assert isinstance(res, ScenarioResult)
    assert res.scenario_id == "PED_01"
    assert res.steps_count > 0
    assert res.collision_occurred is False
    assert res.passed is True
    assert res.emergency_brake_activated is True


def test_execute_vehicle_cutin_success(executor):
    """Verify vehicle reacts to cutting in vehicle without collision."""
    scen_path = Path("scenarios/golden_test_suite/cutin_01_aggressive_left.yaml")
    res = executor.run_scenario(scen_path, max_duration_s=5.0)

    assert isinstance(res, ScenarioResult)
    assert res.collision_occurred is False
    assert res.passed is True
    assert res.min_distance_to_actor_m > 0.0


def test_execute_track_obstacle_success(executor):
    """Verify vehicle stops or avoids tire obstacle on highway."""
    scen_path = Path("scenarios/golden_test_suite/obs_01_tire_in_lane.yaml")
    res = executor.run_scenario(scen_path, max_duration_s=5.0)

    assert isinstance(res, ScenarioResult)
    assert res.collision_occurred is False
    assert res.passed is True


def test_scenario_failure_on_unreasonable_safety_threshold(executor):
    """Verify that an impossible safety threshold correctly triggers a test failure."""
    scen_path = Path("scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml")
    scen = executor.load_scenario(scen_path)

    # Set impossible minimum distance (e.g. 100 meters)
    scen.success_criteria["min_distance_to_pedestrian_m"] = 100.0
    res = executor.run_scenario(scen, max_duration_s=4.0)

    assert res.passed is False
    assert any("Minimum distance violation" in reason for reason in res.failure_reasons)
