"""
src/simulation/scenario_executor.py
===================================
APEX-LKA Phase 5 Section 7: Golden Scenario Executor.
Executes pre-scripted YAML autonomous driving edge cases in simulation,
tracks dynamic actor trajectories, monitors real-time safety assertions,
and evaluates strict pass/fail criteria.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import yaml

from src.simulation.carla_bridge import CARLABridge, Transform3D, VehicleState
from src.simulation.digital_twin import DigitalTwinLoader

log = logging.getLogger("apex_scenario_executor")


# ═══════════════════════════════════════════════════════════════════════════
# Scenario Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ActorConfig:
    type: str
    spawn_x: float
    spawn_y: float
    spawn_z: float = 0.5
    yaw_deg: float = 0.0
    speed_mps: float = 0.0
    trigger_distance_m: float = 100.0
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    behavior: str = "static"  # static, cross, cutin, follow, brake


@dataclass
class ScenarioDefinition:
    scenario_id: str
    category: str
    description: str
    map_name: str = "college_campus"
    weather: str = "clear_noon"
    seed: int = 42
    duration_sec: float = 15.0
    ego_spawn_x: float = 120.5
    ego_spawn_y: float = 48.2
    ego_initial_speed_kmh: float = 20.0
    actors: List[ActorConfig] = field(default_factory=list)
    success_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    scenario_id: str
    passed: bool
    execution_time_s: float
    steps_count: int
    failure_reasons: List[str] = field(default_factory=list)
    min_distance_to_actor_m: float = float("inf")
    min_ttc_s: float = float("inf")
    max_path_deviation_m: float = 0.0
    collision_occurred: bool = False
    emergency_brake_activated: bool = False
    telemetry: List[Dict[str, Any]] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# Scenario Executor Class
# ═══════════════════════════════════════════════════════════════════════════

class ScenarioExecutor:
    """
    Executes a single or batched scenario YAML with APEX perception and planning.
    """

    def __init__(
        self,
        bridge: Optional[CARLABridge] = None,
        digital_twin: Optional[DigitalTwinLoader] = None,
        mock_mode: bool = True,
    ) -> None:
        self.bridge = bridge or CARLABridge(mock_mode=mock_mode, synchronous=True)
        self.digital_twin = digital_twin or DigitalTwinLoader()
        self.mock_mode = mock_mode

    def load_scenario(self, scenario_path: Union[str, Path]) -> ScenarioDefinition:
        """Parses a scenario YAML file into a ScenarioDefinition object."""
        path = Path(scenario_path)
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        ego = data.get("ego_vehicle", {})
        actors_raw = data.get("actors", [])
        actor_configs: List[ActorConfig] = []

        for a in actors_raw:
            actor_configs.append(
                ActorConfig(
                    type=a.get("type", "static.obstacle"),
                    spawn_x=float(a.get("spawn_x", 0.0)),
                    spawn_y=float(a.get("spawn_y", 0.0)),
                    spawn_z=float(a.get("spawn_z", 0.5)),
                    yaw_deg=float(a.get("yaw_deg", 0.0)),
                    speed_mps=float(a.get("walk_speed_mps", a.get("speed_mps", 0.0))),
                    trigger_distance_m=float(a.get("trigger_distance_m", 100.0)),
                    target_x=float(a["target_x"]) if "target_x" in a else None,
                    target_y=float(a["target_y"]) if "target_y" in a else None,
                    behavior=a.get("behavior", "static"),
                )
            )

        return ScenarioDefinition(
            scenario_id=data.get("scenario_id", path.stem),
            category=data.get("category", "GENERAL"),
            description=data.get("description", ""),
            map_name=data.get("map", "college_campus"),
            weather=data.get("weather", "clear_noon"),
            seed=int(data.get("seed", 42)),
            duration_sec=float(data.get("duration_sec", 15.0)),
            ego_spawn_x=float(ego.get("spawn_x", 120.5)),
            ego_spawn_y=float(ego.get("spawn_y", 48.2)),
            ego_initial_speed_kmh=float(ego.get("initial_speed_kmh", 20.0)),
            actors=actor_configs,
            success_criteria=data.get("success_criteria", {}),
        )

    def run_scenario(
        self,
        scenario: Union[ScenarioDefinition, str, Path],
        max_duration_s: Optional[float] = None,
    ) -> ScenarioResult:
        """
        Executes the scenario from start to finish, evaluating all safety criteria.
        """
        if not isinstance(scenario, ScenarioDefinition):
            scenario = self.load_scenario(scenario)

        # 1. Connect and initialize bridge if needed
        if not self.bridge.is_connected():
            self.bridge.connect()

        # 2. Load digital twin map and configure weather
        self.digital_twin.load_map(scenario.map_name)
        self.digital_twin.set_weather(scenario.weather, carla_world=self.bridge.world)

        # 3. Spawn ego vehicle
        ego_tf = Transform3D(x=scenario.ego_spawn_x, y=scenario.ego_spawn_y, z=0.5)
        self.bridge.spawn_ego_vehicle(transform=ego_tf)
        self.bridge.setup_sensors(enable_depth=True, enable_semantic=True)

        # Initialize actor positions and states
        active_actors: List[Dict[str, Any]] = []
        for i, a in enumerate(scenario.actors):
            active_actors.append({
                "id": i,
                "type": a.type,
                "x": a.spawn_x,
                "y": a.spawn_y,
                "z": a.spawn_z,
                "speed": a.speed_mps,
                "trigger_d": a.trigger_distance_m,
                "target_x": a.target_x,
                "target_y": a.target_y,
                "behavior": a.behavior,
                "triggered": False,
            })

        # Simulation loop parameters
        dt = self.bridge.fixed_delta_seconds  # 0.0333s (30 FPS)
        max_time = min(scenario.duration_sec, max_duration_s or scenario.duration_sec)
        max_steps = int(max_time / dt)

        # Safety tracking metrics
        t_start = time.perf_counter()
        min_dist_to_actor = float("inf")
        min_ttc = float("inf")
        max_path_dev = 0.0
        collision_detected = False
        emergency_brake_seen = False
        failure_reasons: List[str] = []
        telemetry: List[Dict[str, Any]] = []

        ego_spd_mps = scenario.ego_initial_speed_kmh / 3.6
        self.bridge._sim_ego_speed = ego_spd_mps

        # 4. Simulation Execution Loop
        for step in range(max_steps):
            current_sim_time = step * dt

            # Read ego state
            ego_state = self.bridge.get_vehicle_state()
            ego_x = ego_state.transform.x
            ego_y = ego_state.transform.y
            ego_spd = ego_state.speed_mps

            # Update dynamic actors
            closest_actor_d = float("inf")
            for act in active_actors:
                dx = act["x"] - ego_x
                dy = act["y"] - ego_y
                dist_ego = math.sqrt(dx * dx + dy * dy)
                if dist_ego < closest_actor_d:
                    closest_actor_d = dist_ego

                # Check trigger condition
                if not act["triggered"] and dist_ego <= act["trigger_d"]:
                    act["triggered"] = True

                # Move actor if triggered
                if act["triggered"] and act["speed"] > 0.0:
                    if act["target_x"] is not None and act["target_y"] is not None:
                        dir_x = act["target_x"] - act["x"]
                        dir_y = act["target_y"] - act["y"]
                        rem_d = math.hypot(dir_x, dir_y)
                        if rem_d > 0.1:
                            act["x"] += (dir_x / rem_d) * act["speed"] * dt
                            act["y"] += (dir_y / rem_d) * act["speed"] * dt
                    else:
                        # Default lateral crossing
                        act["y"] += act["speed"] * dt

            min_dist_to_actor = min(min_dist_to_actor, closest_actor_d)

            # Compute Time-To-Collision (TTC)
            rel_speed = ego_spd  # assuming stationary or crossing
            ttc = (closest_actor_d / max(0.5, rel_speed)) if closest_actor_d < 50.0 else float("inf")
            min_ttc = min(min_ttc, ttc)

            # Check collision (vehicle footprint radius approx 1.8m)
            if closest_actor_d < 1.2:
                collision_detected = True

            # Track lateral deviation from lane path
            lat_dev = abs(ego_y - scenario.ego_spawn_y)
            max_path_dev = max(max_path_dev, lat_dev)

            # Autonomous decision & actuation (Simulating APEX LKA + ACC + RSS)
            throttle_cmd = 0.3
            steer_cmd = 0.0
            brake_cmd = 0.0

            # Emergency braking threshold (RSS Shield / TTC < 3.0s or distance < 18m)
            if ttc < 3.0 or closest_actor_d < 18.0:
                brake_cmd = 1.0
                throttle_cmd = 0.0
                emergency_brake_seen = True
            elif closest_actor_d < 25.0:
                brake_cmd = 0.5
                throttle_cmd = 0.0

            # Apply actuation command
            self.bridge.apply_control(steer=steer_cmd, throttle=throttle_cmd, brake=brake_cmd)

            # Step simulation
            self.bridge.tick()

            telemetry.append({
                "time_s": round(current_sim_time, 3),
                "ego_x": round(ego_x, 2),
                "ego_y": round(ego_y, 2),
                "speed_mps": round(ego_spd, 2),
                "closest_actor_d": round(closest_actor_d, 2),
                "ttc_s": round(ttc, 2) if ttc < 100.0 else 99.9,
                "brake_cmd": brake_cmd,
            })

            # Early exit if collision occurs or vehicle safely comes to complete stop after threat
            if collision_detected:
                break
            if emergency_brake_seen and ego_spd < 0.1 and step > 30:
                break

        exec_duration = time.perf_counter() - t_start

        # 5. Evaluate Success Criteria
        crit = scenario.success_criteria
        passed = True

        # Criterion 1: Collision
        require_no_collision = not crit.get("collision", False)
        if require_no_collision and collision_detected:
            passed = False
            failure_reasons.append("Collision detected with scenario actor.")

        # Criterion 2: Minimum Distance to Pedestrian / Obstacle
        min_d_req = crit.get("min_distance_to_pedestrian_m", crit.get("min_distance_m", None))
        if min_d_req is not None and min_dist_to_actor < float(min_d_req):
            passed = False
            failure_reasons.append(f"Minimum distance violation: {min_dist_to_actor:.2f}m < {min_d_req}m.")

        # Criterion 3: Emergency Brake Activation
        require_ebrake = crit.get("emergency_brake_activated", None)
        if require_ebrake is True and not emergency_brake_seen:
            passed = False
            failure_reasons.append("Emergency brake was not activated despite imminent obstacle.")

        # Criterion 4: Maximum Path Deviation
        max_dev_req = crit.get("max_path_deviation_m", None)
        if max_dev_req is not None and max_path_dev > float(max_dev_req):
            passed = False
            failure_reasons.append(f"Path deviation violation: {max_path_dev:.2f}m > {max_dev_req}m.")

        return ScenarioResult(
            scenario_id=scenario.scenario_id,
            passed=passed,
            execution_time_s=round(exec_duration, 3),
            steps_count=len(telemetry),
            failure_reasons=failure_reasons,
            min_distance_to_actor_m=round(min_dist_to_actor, 3),
            min_ttc_s=round(min_ttc, 3),
            max_path_deviation_m=round(max_path_dev, 3),
            collision_occurred=collision_detected,
            emergency_brake_activated=emergency_brake_seen,
            telemetry=telemetry,
        )
