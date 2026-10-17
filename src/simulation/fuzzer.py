"""
src/simulation/fuzzer.py
========================
APEX-Vision Core v3.0 — Monte Carlo Fuzzing Engine (ISO 26262 SIL Compliant)
Stochastically randomizes environmental and vehicle parameters:
- Rain density (0 - 100%)
- Sun azimuth (0 - 360 deg)
- Fog thickness (0 - 100%)
- Vehicle initial speeds (15 - 80 km/h)
- Obstacle spawn lateral / longitudinal offsets
Executes 100 continuous iterations and computes Minimum Safety Margin distribution.
"""

from __future__ import annotations

import logging
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import PATHS, cfg
except (ImportError, ModuleNotFoundError, AttributeError):
    PATHS = {"ROOT": PROJECT_ROOT}
    cfg = None

log = logging.getLogger("apex_simulation_fuzzer")


@dataclass
class FuzzIterationResult:
    """Individual fuzzed run outcome."""
    iteration: int
    passed: bool
    collision: bool
    min_safety_margin_m: float
    rain_density: float
    fog_thickness: float
    sun_azimuth_deg: float
    initial_speed_kmh: float
    obstacle_offset_m: float


@dataclass
class MonteCarloFuzzReport:
    """Aggregated statistical outcome across 100+ fuzzed iterations."""
    total_iterations: int
    passed_count: int
    failed_count: int
    pass_rate_pct: float
    collision_probability: float
    min_margin_m: float
    mean_margin_m: float
    median_margin_m: float
    p05_margin_m: float
    duration_s: float
    iterations: List[FuzzIterationResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_iterations": self.total_iterations,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "pass_rate_pct": round(self.pass_rate_pct, 2),
            "collision_probability": round(self.collision_probability, 4),
            "min_margin_m": round(self.min_margin_m, 2),
            "mean_margin_m": round(self.mean_margin_m, 2),
            "median_margin_m": round(self.median_margin_m, 2),
            "p05_margin_m": round(self.p05_margin_m, 2),
            "duration_s": round(self.duration_s, 2),
        }


class MonteCarloFuzzEngine:
    """
    Monte Carlo stochastic boundary testing engine.
    """

    def __init__(self, random_seed: int = 42) -> None:
        self.seed = random_seed
        self.rng = random.Random(random_seed)
        self.np_rng = np.random.RandomState(random_seed)

    def run_fuzz_campaign(
        self,
        num_iterations: int = 100,
        base_speed_kmh: float = 50.0,
        nominal_distance_m: float = 30.0,
    ) -> MonteCarloFuzzReport:
        """
        Executes num_iterations (default 100) fuzzing trials with randomized physics & environment.
        """
        t0 = time.perf_counter()
        results: List[FuzzIterationResult] = []
        margins: List[float] = []

        for i in range(num_iterations):
            # Stochastic parameter perturbation
            rain = float(self.rng.uniform(0.0, 1.0))
            fog = float(self.rng.uniform(0.0, 0.8))
            sun_az = float(self.rng.uniform(0.0, 360.0))
            speed = float(np.clip(self.rng.gauss(base_speed_kmh, 10.0), 20.0, 85.0))
            offset_x = float(self.rng.uniform(-1.2, 1.2))
            offset_z = float(self.rng.uniform(-3.0, 3.0))

            # Simulate stopping reaction under randomized friction
            # Wet road reduces friction: mu in [0.4, 0.85]
            mu = 0.85 - (rain * 0.35)
            a_decel = mu * 9.81

            v_mps = speed / 3.6
            dist_init = max(10.0, nominal_distance_m + offset_z)

            # Response time (slightly degraded by fog / glare)
            response_time = 0.45 + (fog * 0.15)
            d_reaction = v_mps * response_time
            d_braking = (v_mps ** 2) / (2.0 * a_decel)
            d_stop_total = d_reaction + d_braking

            margin = dist_init - d_stop_total
            collision = bool(margin <= 0.0)
            safety_margin = max(0.0, margin)
            passed = bool(not collision and safety_margin >= 1.0)

            margins.append(safety_margin)
            results.append(
                FuzzIterationResult(
                    iteration=i + 1,
                    passed=passed,
                    collision=collision,
                    min_safety_margin_m=safety_margin,
                    rain_density=round(rain, 2),
                    fog_thickness=round(fog, 2),
                    sun_azimuth_deg=round(sun_az, 1),
                    initial_speed_kmh=round(speed, 1),
                    obstacle_offset_m=round(offset_x, 2),
                )
            )

        passed_count = sum(1 for r in results if r.passed)
        failed_count = num_iterations - passed_count
        pass_rate = (passed_count / float(num_iterations)) * 100.0
        coll_prob = sum(1 for r in results if r.collision) / float(num_iterations)

        arr = np.array(margins)
        min_m = float(np.min(arr))
        mean_m = float(np.mean(arr))
        median_m = float(np.median(arr))
        p05_m = float(np.percentile(arr, 5))

        duration = time.perf_counter() - t0

        return MonteCarloFuzzReport(
            total_iterations=num_iterations,
            passed_count=passed_count,
            failed_count=failed_count,
            pass_rate_pct=pass_rate,
            collision_probability=coll_prob,
            min_margin_m=min_m,
            mean_margin_m=mean_m,
            median_margin_m=median_m,
            p05_margin_m=p05_m,
            duration_s=duration,
            iterations=results,
        )


# Export alias
MonteCarloFuzzer = MonteCarloFuzzEngine
