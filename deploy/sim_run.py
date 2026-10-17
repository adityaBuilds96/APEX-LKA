"""
deploy/sim_run.py
=================
APEX-LKA Phase 5 Section 7: Simulation, Digital Twin & Validation Runner.
Provides the primary CLI entry point for CARLA simulation execution,
Golden Scenario testing, synthetic dataset generation, and CI regressions.

Usage Examples:
- Run a single scenario:
    python deploy/sim_run.py --map college_campus --scenario scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml

- Run full Golden Scenario CI regression:
    python deploy/sim_run.py --regression --min-pass-rate 95.0

- Run Monte Carlo fuzz testing:
    python deploy/sim_run.py --fuzz --scenario scenarios/golden_test_suite/cutin_01_aggressive_left.yaml --fuzz-iterations 50

- Generate synthetic training dataset:
    python deploy/sim_run.py --generate-synthetic --num-frames 100 --weather rain_moderate
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.carla_bridge import CARLABridge
from src.simulation.digital_twin import DigitalTwinLoader
from src.simulation.scenario_executor import ScenarioExecutor
from src.simulation.synthetic_generator import SyntheticDataGenerator
from src.simulation.fuzz_tester import MonteCarloFuzzTester
from src.simulation.regression_runner import SimulationRegressionRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("apex_sim_runner")


def main():
    parser = argparse.ArgumentParser(description="APEX-LKA Autonomous Simulation & Digital Twin Runner")
    parser.add_argument("--map", type=str, default="college_campus", help="Digital twin map (college_campus, baja_track_2027)")
    parser.add_argument("--scenario", type=str, default=None, help="Path or name of scenario YAML file")
    parser.add_argument("--weather", type=str, default="clear_noon", help="Environmental weather variant")
    parser.add_argument("--regression", action="store_true", help="Execute 100+ Golden Scenario CI regression suite")
    parser.add_argument("--min-pass-rate", type=float, default=95.0, help="Minimum pass-rate threshold percentage (default 95.0)")
    parser.add_argument("--report-html", type=str, default="results/regression_report.html", help="HTML report output path")
    parser.add_argument("--report-json", type=str, default="results/regression_summary.json", help="JSON summary output path")
    parser.add_argument("--fuzz", action="store_true", help="Run Monte Carlo fuzz testing on target scenario")
    parser.add_argument("--fuzz-iterations", type=int, default=50, help="Number of Monte Carlo fuzz iterations")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic multimodal dataset")
    parser.add_argument("--num-frames", type=int, default=100, help="Number of synthetic frames to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic_dataset", help="Output directory for synthetic data")
    parser.add_argument("--mock", action="store_true", default=False, help="Force mock simulation without CARLA server")

    args = parser.parse_args()

    # 1. Full Golden Scenario Regression Suite Mode
    if args.regression:
        log.info("Executing APEX Golden Scenario Regression Suite...")
        bridge = CARLABridge(mock_mode=args.mock, synchronous=True)
        dt = DigitalTwinLoader(maps_dir="simulation/maps")
        executor = ScenarioExecutor(bridge=bridge, digital_twin=dt, mock_mode=args.mock)
        runner = SimulationRegressionRunner(
            scenario_dir="scenarios/golden_test_suite",
            executor=executor,
            min_pass_rate_pct=args.min_pass_rate,
        )

        summary = runner.run_all(max_duration_per_scenario_s=3.0)
        runner.generate_html_report(summary, output_path=args.report_html)

        # Save JSON summary
        json_out = Path(args.report_json)
        json_out.parent.mkdir(parents=True, exist_ok=True)
        with open(json_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        print("\n" + "=" * 60)
        print("APEX-LKA SIMULATION REGRESSION AUDIT RESULTS")
        print("=" * 60)
        print(f"  Total Scenarios: {summary['total_scenarios']}")
        print(f"  Passed:          {summary['passed']}")
        print(f"  Failed:          {summary['failed']}")
        print(f"  Pass Rate:       {summary['pass_rate_pct']:.1f}% (Threshold: >= {summary['threshold_pct']}%)")
        print(f"  Duration:        {summary['execution_duration_sec']:.1f} s")
        print(f"  HTML Report:     {args.report_html}")
        print("=" * 60)

        if not summary["threshold_met"]:
            log.error("REGRESSION BLOCKED: Pass rate %.1f%% below threshold %.1f%%!", summary['pass_rate_pct'], summary['threshold_pct'])
            sys.exit(1)
        else:
            log.info("REGRESSION PASSED: Autonomous software certified for merge/deployment.")
            sys.exit(0)

    # 2. Synthetic Dataset Generation Mode
    if args.generate_synthetic:
        log.info("Running Synthetic Dataset Generator (%d frames, weather=%s)...", args.num_frames, args.weather)
        dt = DigitalTwinLoader(maps_dir="simulation/maps")
        gen = SyntheticDataGenerator(
            output_dir=args.output_dir,
            digital_twin=dt,
        )
        manifest = gen.generate_batch(num_frames=args.num_frames, weather=args.weather)
        print("\n" + "=" * 60)
        print("SYNTHETIC DATASET GENERATION COMPLETE")
        print("=" * 60)
        print(f"  Frames Generated:  {manifest['total_samples']}")
        print(f"  Generation Rate:   {manifest['throughput_frames_per_hour']:.0f} frames/hour (Target >= 10,000)")
        print(f"  Target Met:        {manifest['target_met']}")
        print(f"  Output Directory:  {args.output_dir}")
        print("=" * 60)
        sys.exit(0)

    # 3. Monte Carlo Fuzz Testing Mode
    if args.fuzz:
        scen_target = args.scenario or "scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml"
        scen_path = Path(scen_target)
        if not scen_path.exists():
            scen_path = Path("scenarios/golden_test_suite") / scen_target
        if not scen_path.exists():
            log.error("Scenario file '%s' not found.", scen_target)
            sys.exit(1)

        fuzzer = MonteCarloFuzzTester(default_seed=42)
        report = fuzzer.fuzz_scenario(scen_path, num_iterations=args.fuzz_iterations)
        report_path = Path("results/fuzz_report.json")
        fuzzer.export_report_json(report, report_path)

        print("\n" + "=" * 60)
        print("MONTE CARLO FUZZ TESTING RESULTS")
        print("=" * 60)
        print(f"  Scenario:          {report.scenario_id}")
        print(f"  Iterations:        {report.total_iterations}")
        print(f"  Pass Rate:         {report.pass_rate_pct:.1f}%")
        print(f"  P(collision):      {report.collision_probability:.4f}")
        print(f"  Min Distance (p5): {report.min_distance_stats.get('p05_m', 0.0):.2f} m")
        print(f"  Min TTC (p5):      {report.min_ttc_stats.get('p05_s', 0.0):.2f} s")
        print(f"  Report Export:     {report_path}")
        print("=" * 60)
        sys.exit(0)

    # 4. Single Scenario Execution Mode
    scen_target = args.scenario or "scenarios/golden_test_suite/ped_01_jaywalk_clear.yaml"
    scen_path = Path(scen_target)
    if not scen_path.exists():
        scen_path = Path("scenarios/golden_test_suite") / scen_target
    if not scen_path.exists():
        log.error("Scenario file '%s' not found.", scen_target)
        sys.exit(1)

    bridge = CARLABridge(mock_mode=args.mock, synchronous=True)
    dt = DigitalTwinLoader(maps_dir="simulation/maps")
    executor = ScenarioExecutor(bridge=bridge, digital_twin=dt, mock_mode=args.mock)

    log.info("Executing scenario: %s on map: %s...", scen_path.name, args.map)
    result = executor.run_scenario(scen_path)

    print("\n" + "=" * 60)
    print("SCENARIO EXECUTION RESULT")
    print("=" * 60)
    print(f"  Scenario ID:    {result.scenario_id}")
    print(f"  Passed:         {'YES [PASS]' if result.passed else 'NO [FAIL]'}")
    print(f"  Min Distance:   {result.min_distance_to_actor_m:.2f} m")
    print(f"  Min TTC:        {result.min_ttc_s:.2f} s")
    print(f"  Max Path Dev:   {result.max_path_deviation_m:.2f} m")
    print(f"  Collision:      {result.collision_occurred}")
    print(f"  Emergency Stop: {result.emergency_brake_activated}")
    if result.failure_reasons:
        print(f"  Failures:       {', '.join(result.failure_reasons)}")
    print("=" * 60)

    bridge.cleanup()
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
