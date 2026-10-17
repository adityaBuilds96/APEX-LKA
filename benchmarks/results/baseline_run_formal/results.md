# APEX-Vision Core v4.0 • Empirical Timing Baseline Report

> **Normative Process Notice**: Developed following an ISO 26262-style process; ASIL-B-aligned; not independently assessed. Target Operational Domain: Driver-supervised SAE Level 2.

## 1. System Execution Environment

- **Timestamp (UTC)**: `2026-10-17T06:03:05.784738+00:00`
- **Hardware Target Classification**: **`NON-TARGET`**
- **Host OS / Architecture**: `Windows 10 (64bit)`
- **CPU Model / Cores**: `AMD64 Family 25 Model 68 Stepping 1, AuthenticAMD (None physical / 16 logical)`
- **RAM (Total / Available)**: `23.69 GB / 7.42 GB`
- **PyTorch / Device**: `v2.12.0+cpu (CUDA Available: False)`
- **Git Commit / Dirty**: `5a4211bc955018fff656462626c120e9662da52d` (Dirty: `True`)

## 2. Overall Pipeline Latency Breakdown (All Frames)

| Pipeline Stage | Samples (n) | Mean (ms) | Std (ms) | Min (ms) | p50 (ms) | p90 (ms) | p99 (ms) | p99.9 (ms) | Max (ms) | >33.3ms | >20ms |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `preprocess` | 1000 | 0.33 | 0.06 | 0.23 | 0.31 | 0.41 | 0.49 | 0.88 | 0.88 | 0 | 0 |
| `HydraNet` | 1000 | 0.47 | 0.07 | 0.43 | 0.45 | 0.53 | 0.83 | 0.94 | 0.94 | 0 | 0 |
| `depth` | 1000 | 0.40 | 0.08 | 0.30 | 0.38 | 0.50 | 0.67 | 0.82 | 0.82 | 0 | 0 |
| `voxel` | 1000 | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 | 0.06 | 0.06 | 0 | 0 |
| `tracker` | 1000 | 0.02 | 0.02 | 0.01 | 0.01 | 0.07 | 0.09 | 0.14 | 0.14 | 0 | 0 |
| `prediction_TTI` | 1000 | 0.05 | 0.01 | 0.04 | 0.05 | 0.06 | 0.08 | 0.12 | 0.12 | 0 | 0 |
| `ESKF` | 1000 | 0.06 | 0.02 | 0.03 | 0.06 | 0.08 | 0.10 | 0.22 | 0.22 | 0 | 0 |
| `planner` | 1000 | 0.03 | 0.01 | 0.02 | 0.02 | 0.03 | 0.05 | 0.09 | 0.09 | 0 | 0 |
| `RSS` | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0 | 0 |
| `controller` | 1000 | 0.01 | 0.00 | 0.01 | 0.01 | 0.02 | 0.03 | 0.05 | 0.05 | 0 | 0 |
| `CAN_encode` | 1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 | 0.01 | 0 | 0 |
| `end_to_end` | 1000 | 1.41 | 0.18 | 1.17 | 1.37 | 1.65 | 2.04 | 2.29 | 2.29 | 0 | 0 |

## 3. Keyframe vs Non-Keyframe Disaggregated Analysis

| Pipeline Stage | Keyframe p50 (ms) | Keyframe p99 (ms) | Non-Keyframe p50 (ms) | Non-Keyframe p99 (ms) | Keyframe Delta (ms) |
|---|---|---|---|---|---|
| `HydraNet` | 0.45 | 0.79 | 0.45 | 0.83 | -0.04 |
| `depth` | 0.39 | 0.66 | 0.38 | 0.67 | -0.01 |
| `voxel` | 0.01 | 0.03 | 0.01 | 0.03 | +0.00 |
| `tracker` | 0.07 | 0.13 | 0.01 | 0.02 | +0.11 |
| `prediction_TTI` | 0.04 | 0.09 | 0.05 | 0.08 | +0.01 |
| `end_to_end` | 1.40 | 2.22 | 1.37 | 2.02 | +0.19 |

## 4. Run-to-Run Spread & Thermal Stability (3 Repeats)

- **p50 Spread**: `0.014 ms` (`1.05%`)
- **p99 Spread**: `0.161 ms` (`8.34%`)

## 5. Explicit Component Sum vs End-to-End Latency Verification

- **Sum of Individual Component p50s**: `1.32 ms`
- **Measured End-to-End p50**: `1.37 ms`
- **Sum of Individual Component p99s**: `2.37 ms`
- **Measured End-to-End p99**: `2.04 ms`
- **Pipeline Scheduling / Orchestration Overhead**: `0.05 ms`

## 6. Unmeasured Domains Notice

The following physical and hardware stages are not measurable in pure software emulation and are tagged `[UNMEASURED]`:

- `[UNMEASURED]` Camera exposure & ISP read-out latency (typically 10-25 ms hardware-dependent)
- `[UNMEASURED]` Physical CAN bus transceiver & wire propagation delay (typically 0.2-1.2 ms at 500 kbps)
- `[UNMEASURED]` Electronic power steering (EPS) motor actuation lag & mechanical deadband (typically 30-80 ms)
- `[UNMEASURED]` Electronic brake booster pressure rise latency (typically 40-120 ms)
- `[UNMEASURED]` Full photon-to-torque (photon-to-CAN + CAN-to-actuator) physical closed loop
- `[UNMEASURED]` Extended thermal soak endurance under enclosed automotive under-hood temperatures (55-85 deg C)
