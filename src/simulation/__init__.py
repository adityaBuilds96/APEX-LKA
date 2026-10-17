"""
src/simulation/__init__.py
==========================
APEX-LKA Phase 5 Section 7: Simulation, Digital Twin & Validation Framework.
"""

from src.simulation.carla_bridge import (
    CARLABridge,
    Transform3D,
    IMUData,
    GNSSData,
    WheelEncoderData,
    VehicleState,
)
from src.simulation.digital_twin import (
    DigitalTwinLoader,
    DigitalTwinMap,
    RoadWaypoint,
)
from src.simulation.scenario_executor import (
    ScenarioExecutor,
    ScenarioDefinition,
    ScenarioResult,
    ActorConfig,
)
from src.simulation.synthetic_generator import (
    SyntheticDataGenerator,
    SyntheticSample,
    BoundingBox3D,
)
from src.simulation.fuzz_tester import (
    MonteCarloFuzzTester,
    FuzzReport,
    FuzzIterationResult,
)
from src.simulation.regression_runner import (
    SimulationRegressionRunner,
)

__all__ = [
    "CARLABridge",
    "Transform3D",
    "IMUData",
    "GNSSData",
    "WheelEncoderData",
    "VehicleState",
    "DigitalTwinLoader",
    "DigitalTwinMap",
    "RoadWaypoint",
    "ScenarioExecutor",
    "ScenarioDefinition",
    "ScenarioResult",
    "ActorConfig",
    "SyntheticDataGenerator",
    "SyntheticSample",
    "BoundingBox3D",
    "MonteCarloFuzzTester",
    "FuzzReport",
    "FuzzIterationResult",
    "SimulationRegressionRunner",
]
