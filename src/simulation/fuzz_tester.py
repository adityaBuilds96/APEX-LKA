"""
src/simulation/fuzz_tester.py
=============================
APEX-LKA Phase 5 Section 7: Monte Carlo Fuzz Testing Engine.
Stochastically perturbs autonomous driving scenario parameters (weather, lighting,
actor spawn coordinates, approach velocities, sensor noise) over 100-1,000+ iterations,
computes statistical failure distributions, and maps critical safety boundaries.
"""

from __future__ import annotations

import copy
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.simulation.scenario_executor import (
    ScenarioExecutor,
    ScenarioDefinition,
    ScenarioResult,
    ActorConfig,
)

log = logging.getLogger("apex_fuzz_tester")


# ═══════════════════════════════════════════════════════════════════════════
# Fuzz Telemetry & Statistics
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class FuzzIterationResult:
    iteration: int
    seed: int
    passed: bool
    collision: bool
    min_distance_m: float
    min_ttc_s: float
    max_path_deviation_m: float
    perturbed_params: Dict[str, Any]
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class FuzzReport:
    scenario_id: str
    total_iterations: int
    passed_count: int
    failed_count: int
    pass_rate_pct: float
    collision_probability: float
    deviation_failure_probability: float
    min_distance_stats: Dict[str, float]  # min, mean, median, p05
    min_ttc_stats: Dict[str, float]        # min, mean, median, p05
    critical_failure_cases: List[Dict[str, Any]]
    execution_duration_s: float


# ═══════════════════════════════════════════════════════════════════════════
# Monte Carlo Fuzz Testing Engine
# ═══════════════════════════════════════════════════════════════════════════

class MonteCarloFuzzTester:
    """
    Automated stochastic fuzz testing engine evaluating probabilistic safety limits.
    """

    WEATHER_OPTIONS = [
        "clear_noon",
        "cloudy_afternoon",
        "rain_moderate",
        "fog_dense",
        "dusk_golden",
        "night_streetlights",
        "night_pitch_black",
        "dust_cloud_heavy",
    ]

    def __init__(
        self,
        executor: Optional[ScenarioExecutor] = None,
        default_seed: int = 42,
    ) -> None:
        self.executor = executor or ScenarioExecutor(mock_mode=True)
        self.default_seed = default_seed
        self.iteration_results: List[FuzzIterationResult] = []

    def fuzz_scenario(
        self,
        base_scenario: Union[ScenarioDefinition, str, Path],
        num_iterations: int = 100,
        seed: Optional[int] = None,
        max_duration_per_run_s: float = 4.0,
    ) -> FuzzReport:
        """
        Runs Monte Carlo stochastic perturbation loop on base scenario.
        """
        if not isinstance(base_scenario, ScenarioDefinition):
            base_scenario = self.executor.load_scenario(base_scenario)

        run_seed = seed if seed is not None else self.default_seed
        rng = np.random.RandomState(run_seed)

        log.info(
            "Starting Monte Carlo Fuzz Testing on '%s' (%d iterations, seed=%d)...",
            base_scenario.scenario_id,
            num_iterations,
            run_seed,
        )

        t_start = time.perf_counter()
        self.iteration_results.clear()

        for it in range(num_iterations):
            # Generate perturbed scenario instance
            perturbed, param_log = self._perturb_scenario(base_scenario, rng, it)

            # Execute scenario
            res = self.executor.run_scenario(perturbed, max_duration_s=max_duration_per_run_s)

            it_res = FuzzIterationResult(
                iteration=it,
                seed=run_seed + it,
                passed=res.passed,
                collision=res.collision_occurred,
                min_distance_m=res.min_distance_to_actor_m,
                min_ttc_s=res.min_ttc_s,
                max_path_deviation_m=res.max_path_deviation_m,
                perturbed_params=param_log,
                failure_reasons=res.failure_reasons,
            )
            self.iteration_results.append(it_res)

        exec_time = time.perf_counter() - t_start
        report = self._compute_statistics(base_scenario.scenario_id, exec_time)

        log.info(
            "Monte Carlo Fuzz complete for '%s': Pass Rate=%.1f%%, P(collision)=%.3f in %.2fs",
            base_scenario.scenario_id,
            report.pass_rate_pct,
            report.collision_probability,
            exec_time,
        )
        return report

    def _perturb_scenario(
        self,
        base: ScenarioDefinition,
        rng: np.random.RandomState,
        iteration: int,
    ) -> Tuple[ScenarioDefinition, Dict[str, Any]]:
        """
        Randomly perturbs environmental weather, actor spawn, and approach velocities.
        """
        scen = copy.deepcopy(base)
        scen.scenario_id = f"{base.scenario_id}_fuzz_{iteration:03d}"
        param_log: Dict[str, Any] = {}

        # 1. Weather & Lighting Perturbation
        if rng.uniform(0.0, 1.0) < 0.40:
            scen.weather = str(rng.choice(self.WEATHER_OPTIONS))
            param_log["weather"] = scen.weather

        # 2. Ego Initial Speed Perturbation (± 15%)
        speed_delta = float(rng.uniform(-0.15, 0.15))
        scen.ego_initial_speed_kmh = max(5.0, base.ego_initial_speed_kmh * (1.0 + speed_delta))
        param_log["ego_speed_kmh"] = round(scen.ego_initial_speed_kmh, 2)

        # 3. Dynamic Actors Perturbations
        for act in scen.actors:
            # Spawn distance jitter (± 2.5m)
            dx = float(rng.uniform(-2.5, 2.5))
            dy = float(rng.uniform(-0.8, 0.8))
            act.spawn_x += dx
            act.spawn_y += dy

            # Speed jitter (± 20%)
            if act.speed_mps > 0.0:
                act.speed_mps = max(0.5, act.speed_mps * float(rng.uniform(0.8, 1.25)))

            # Trigger distance jitter (± 2.0m)
            act.trigger_distance_m = max(3.0, act.trigger_distance_m + float(rng.uniform(-2.0, 2.0)))

            param_log["actor_spawn_dx"] = round(dx, 2)
            param_log["actor_spawn_dy"] = round(dy, 2)
            param_log["actor_speed"] = round(act.speed_mps, 2)

        return scen, param_log

    def _compute_statistics(self, scenario_id: str, exec_duration_s: float) -> FuzzReport:
        """Computes statistical metrics over all Monte Carlo iterations."""
        total = len(self.iteration_results)
        if total == 0:
            return FuzzReport(
                scenario_id=scenario_id,
                total_iterations=0,
                passed_count=0,
                failed_count=0,
                pass_rate_pct=0.0,
                collision_probability=0.0,
                deviation_failure_probability=0.0,
                min_distance_stats={},
                min_ttc_stats={},
                critical_failure_cases=[],
                execution_duration_s=round(exec_duration_s, 2),
            )

        passed = sum(1 for r in self.iteration_results if r.passed)
        failed = total - passed
        collisions = sum(1 for r in self.iteration_results if r.collision)
        deviations = sum(1 for r in self.iteration_results if any("deviation" in f.lower() for f in r.failure_reasons))

        dist_vals = [r.min_distance_m for r in self.iteration_results if not math.isinf(r.min_distance_m)]
        ttc_vals = [r.min_ttc_s for r in self.iteration_results if not math.isinf(r.min_ttc_s) and r.min_ttc_s < 50.0]

        d_stats = {
            "min_m": round(float(np.min(dist_vals)), 3) if dist_vals else 0.0,
            "mean_m": round(float(np.mean(dist_vals)), 3) if dist_vals else 0.0,
            "median_m": round(float(np.median(dist_vals)), 3) if dist_vals else 0.0,
            "p05_m": round(float(np.percentile(dist_vals, 5)), 3) if dist_vals else 0.0,
        }

        ttc_stats = {
            "min_s": round(float(np.min(ttc_vals)), 3) if ttc_vals else 0.0,
            "mean_s": round(float(np.mean(ttc_vals)), 3) if ttc_vals else 0.0,
            "median_s": round(float(np.median(ttc_vals)), 3) if ttc_vals else 0.0,
            "p05_s": round(float(np.percentile(ttc_vals, 5)), 3) if ttc_vals else 0.0,
        }

        critical_cases = [
            {
                "iteration": r.iteration,
                "reasons": r.failure_reasons,
                "min_d_m": r.min_distance_m,
                "min_ttc_s": r.min_ttc_s,
                "params": r.perturbed_params,
            }
            for r in self.iteration_results
            if not r.passed
        ][:10]

        return FuzzReport(
            scenario_id=scenario_id,
            total_iterations=total,
            passed_count=passed,
            failed_count=failed,
            pass_rate_pct=round((passed / total) * 100.0, 2),
            collision_probability=round(collisions / total, 4),
            deviation_failure_probability=round(deviations / total, 4),
            min_distance_stats=d_stats,
            min_ttc_stats=ttc_stats,
            critical_failure_cases=critical_cases,
            execution_duration_s=round(exec_duration_s, 2),
        )

    def export_report_json(self, report: FuzzReport, output_path: Union[str, Path]) -> Path:
        """Exports fuzz testing results to JSON format."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        doc = {
            "scenario_id": report.scenario_id,
            "total_iterations": report.total_iterations,
            "passed": report.passed_count,
            "failed": report.failed_count,
            "pass_rate_pct": report.pass_rate_pct,
            "collision_probability": report.collision_probability,
            "deviation_failure_probability": report.deviation_failure_probability,
            "min_distance_stats": report.min_distance_stats,
            "min_ttc_stats": report.min_ttc_stats,
            "critical_failures": report.critical_failure_cases,
            "duration_sec": report.execution_duration_s,
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2)
        return out
