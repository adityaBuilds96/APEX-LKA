# APEX-LKA Hardware-In-The-Loop (HIL) Test Scenarios

This directory contains deterministic scenario manifests and test clip definitions for stationary bench validation of the APEX-LKA autonomous driving stack on the NVIDIA Jetson Orin Nano and dual Arduino Mega 2560 controllers.

## Scenario Manifest Format (`hil_sequence_manifest.yaml`)

Each scenario defines:
- `id`: Unique identifier for the scenario.
- `name`: Human-readable title.
- `clip_path`: Relative path to video stream or synthetic generator identifier.
- `expected_behavior`: Target maneuver (`straight_centering`, `left_curve_tracking`, `right_curve_tracking`, `broken_paint`).
- `duration_frames`: Total frame count for the benchmark slice (default: 60 frames = 2.0s @ 30 FPS).
- `target_fps`: Minimum sustained throughput requirement (30 FPS).
- `latency_budget_ms`: Hard p99 latency upper bound (40.0 ms).
- `steering_range_deg`: Acceptable steering angle bounds for the maneuver.

## Execution

Execute the full HIL sequence via:
```bash
python deploy/hil_runner.py --manifest deploy/hil_scenarios/hil_sequence_manifest.yaml
```

The runner exercises the complete perception -> BEV -> EKF -> Controller -> CAN/Serial pipeline, checks latency bounds, verifies CAN frame transmissions, and outputs `results/benchmarks/hil_test_report.json`.
