# APEX-Vision Core v4.0 • Baseline Timing & Latency Profiling Harness

## 1. Executive Summary & Purpose

The APEX Baseline Benchmark Harness (`benchmarks/run_baseline.py`) provides an empirical, zero-fabrication timing evaluation suite designed for autonomous vehicle perception, planning, and control pipelines. Developed following an ISO 26262-style process (ASIL-B-aligned; not independently assessed) for driver-supervised SAE Level 2 operation, this harness measures latency across 11 distinct pipeline stages and end-to-end execution.

Key engineering capabilities include:
- **Nanosecond-Precision Monotonic Clock**: Utilizes Python `time.perf_counter_ns()` to eliminate clock drift and floating-point quantization errors.
- **Strict Hardware Synchronization**: Enforces `torch.cuda.synchronize()` barriers before and after all CUDA execution blocks to guarantee asynchronous GPU queues are completely flushed prior to timestamp capture.
- **Keyframe-Disaggregated Statistics**: Segregates full-inference keyframes (obstacle detections, deep feature extractions) from lightweight tracking/interpolation frames using the detector's keyframe flag, eliminating misleading average distortions.
- **Conservative Safety Percentiles**: Enforces NumPy `method='higher'` for percentile calculations ($p_{50}, p_{90}, p_{99}, p_{99.9}$) to strictly avoid underestimating tail latencies under ISO 26262 worst-case execution time (WCET) guidelines.
- **Run-to-Run Thermal Spread Analysis**: Executes multiple consecutive process runs to detect thermal throttling, memory leaks, and CPU scheduling variations.

---

## 2. Directory Structure & Artifacts

```
benchmarks/
├── capture_env.py              # System environment & hardware classifier (env.json)
├── run_baseline.py             # 10,000-frame timing harness orchestrator
├── README.md                   # This execution guide
├── results/                    # Output directory for timestamped benchmark runs
│   └── run_<UTC_TIMESTAMP>/
│       ├── env.json            # Captured host/Jetson environment metadata
│       ├── raw_latencies.csv   # Per-frame nanosecond-derived millisecond latencies
│       ├── results.json        # Machine-readable per-stage statistics and spread
│       └── results.md          # Markdown summary report with audit tables
└── tests/
    └── test_stats.py           # Unit tests validating percentile mathematics
```

---

## 3. Quick-Start Execution Commands

### 3.1 Single-Command Host / CI Execution (Fast Mode: 500 Frames)
To perform a rapid sanity check on an x86_64 development workstation:
```bash
python benchmarks/run_baseline.py --frames 500 --warmup 50 --repeat 1 --device auto
```

### 3.2 Full Production Baseline Audit (10,000 Frames, 3 Repeats)
To generate the formal Phase 7 baseline dataset on target hardware:
```bash
python benchmarks/run_baseline.py \
    --profile configs/profiles/passenger_car.yaml \
    --frames 10000 \
    --warmup 300 \
    --repeat 3 \
    --device auto \
    --gc on \
    --out benchmarks/results/formal_baseline
```

### 3.3 BAJA SAE Buggy Profile Execution (Off-Road Dynamics)
```bash
python benchmarks/run_baseline.py \
    --profile configs/profiles/baja_sae.yaml \
    --frames 5000 \
    --warmup 200 \
    --device auto
```

### 3.4 Validating Statistical Rigor
Verify the statistical percentile calculations against analytical Uniform, Normal, and Exponential distributions:
```bash
pytest benchmarks/tests/test_stats.py -v
```

---

## 4. Command-Line Options Reference

| CLI Flag | Type | Default | Description |
|---|---|---|---|
| `--profile` | Path | `configs/profiles/passenger_car.yaml` | Path to vehicle YAML dynamics and control limits profile |
| `--frames` | Integer | `10000` | Number of measurement frames to process after warmup completes |
| `--warmup` | Integer | `300` | Warmup iterations to prime instruction cache, PyTorch JIT, and GPU pipelines |
| `--seed` | Integer | `1234` | Deterministic pseudo-random seed for synthetic frame generation |
| `--source` | String | `synthetic` | Ingestion mode: `synthetic`, `dir`, or `video` |
| `--input` | Path | `None` | Optional image directory or video file path when not using synthetic |
| `--out` | Path | `benchmarks/results/run_<timestamp>` | Output directory path for generated benchmark artifacts |
| `--device` | String | `auto` | Execution compute backend: `auto`, `cuda`, or `cpu` |
| `--gc` | String | `on` | Garbage collection mode: `on` (collect between runs) or `off` (disable in loop) |
| `--repeat` | Integer | `3` | Number of sequential benchmark runs to measure run-to-run spread |

---

## 5. Pipeline Stages Measured

The timing harness wraps each of the following 11 functional stages individually, followed by a separate wall-clock end-to-end measurement:

1. **`preprocess`**: Resolution downsampling to 640x360, color space normalization, CLAHE enhancement, bilateral filtering.
2. **`HydraNet`**: Multi-task convolutional backbone inference for lane segmentation, free space, and vehicle bounding boxes.
3. **`depth`**: Monocular depth estimation producing pixel-level metric distance maps ($Z \in [0.5, 80.0]\,\text{m}$).
4. **`voxel`**: 2D/3D occupancy grid rasterization mapping drivable space and obstacle clusters ($200 \times 400$ BEV cells).
5. **`tracker`**: Multi-object ByteTrack / Kalman filter state association and track ID continuity management.
6. **`prediction_TTI`**: 1-3 second motion trajectory extrapolation and Time-To-Impact collision risk calculation.
7. **`ESKF`**: 15-state Error-State Extended Kalman Filter fusing IMU angular rates, wheel odometry, and visual odometry.
8. **`planner`**: Hierarchical polynomial motion planner computing target waypoints and curvature feeds.
9. **`RSS`**: Responsibility-Sensitive Safety mathematical shield computing minimum longitudinal and lateral safety envelopes.
10. **`controller`**: Stanley geometric steering controller + PID longitudinal acceleration controller with anti-windup clamping.
11. **`CAN_encode`**: Automotive CAN 2.0B frame serialization (IDs `0x150`, `0x160`, `0x170`, `0x180`, `0x400-0x408`).
12. **`end_to_end`**: Total wall-clock time from raw image ingestion to completed CAN packet transmission readiness.

---

## 6. Statistical Rigor & Tail Latency Rules

In automotive safety-critical systems governed by ISO 26262, the mean latency is an insufficient metric because control stability and actuator deadline misses are dictated by the tail percentiles ($p_{99}, p_{99.9}, \text{max}$).

### 6.1 Percentile Calculation Method
Standard linear percentile interpolation (`method='linear'`) computes weighted averages of adjacent rank order statistics, which can underestimate peak latencies. This harness mandates NumPy `method='higher'`:
$$\text{Index} = \lceil p \cdot (n - 1) \rceil$$
This guarantees that the reported percentile is always an observed sample equal to or greater than the exact mathematical quantile, providing a conservative safety bound.

### 6.2 Keyframe Disaggregation
Deep neural networks utilize conditional execution where full object detection heads execute on keyframes while non-keyframes rely on optical flow or lightweight temporal tracking. Combining these into a single distribution produces bimodal artifacts. The harness reports:
- Keyframe $p_{50}$ and $p_{99}$
- Non-Keyframe $p_{50}$ and $p_{99}$
- Keyframe Delta ($\Delta = p_{99}^{\text{key}} - p_{99}^{\text{non-key}}$)

---

## 7. Hardware Classification & Unmeasured Domains

### 7.1 Target Hardware Rules
The capture script (`benchmarks/capture_env.py`) automatically assesses the running host. The label **`TARGET`** is assigned **strictly and exclusively** when executing on physical NVIDIA Jetson Orin Nano hardware:
- Running on any x86_64 host (Intel / AMD) is strictly tagged **`NON-TARGET`**.
- Running in a virtual machine, WSL, or non-Tegra ARM platform is strictly tagged **`NON-TARGET`**.

### 7.2 Unmeasured Domains Disclaimer
In accordance with zero-fabrication principles, the following physical latencies cannot be captured in a software-in-the-loop (SIL) timing harness and are formally classified as `[UNMEASURED]`:
1. **Camera Sensor Exposure & ISP Processing**: Hardware photon integration and image signal processor pipeline (typically 10–25 ms).
2. **Physical CAN Bus Transceiver Latency**: Microcontroller SPI transfer and physical CAN bus arbitration at 500 kbps (typically 0.2–1.2 ms).
3. **Electronic Power Steering (EPS) Motor Delay**: Closed-loop torque motor response, gear backlash, and steering column inertia (typically 30–80 ms).
4. **Electronic Brake Booster Response**: Hydraulic pressure buildup and brake caliper contact time (typically 40–120 ms).
5. **Complete Closed-Loop Photon-to-Torque Latency**: End-to-end physical latency from optical photon arrival to mechanical road-wheel torque.
6. **Thermal Soak Endurance**: Extended multi-hour performance under ambient automotive under-hood operating temperatures (55°C–85°C).

---

## 8. Continuous Integration & Quality Assurance

To ensure timing harness integrity across git commits:
- Run statistical unit tests: `pytest benchmarks/tests/test_stats.py -v`
- Inspect environment capture: `python benchmarks/capture_env.py`
- Validate results JSON against schema: Ensure all 11 stages and end-to-end fields exist.

---

## 9. Detailed Stage Arithmetic & Theoretical Execution Bounds

Understanding the computational demands of each pipeline stage is essential for identifying bottlenecks:

| Stage Name | Computational Complexity | Primary Hardware Resource | Memory Bandwidth Traffic |
|---|---|---|---|
| `preprocess` | $\mathcal{O}(W \times H)$ | CPU Vector (SIMD) / GPU VIC | High (Image read/write) |
| `HydraNet` | $\mathcal{O}(\sum k^2 c_{\text{in}} c_{\text{out}} W H)$ | NVIDIA Tensor Core (FP16/INT8) | Moderate (Activation streaming) |
| `depth` | $\mathcal{O}(W \times H \times D)$ | GPU Shader Cores / DLA | High (Depth disparity map) |
| `voxel` | $\mathcal{O}(N_{\text{voxels}} + N_{\text{pts}})$ | CPU / GPU Shared Memory | Low-Moderate (2D BEV raster) |
| `tracker` | $\mathcal{O}(M^3 + N_{\text{tracks}} \cdot d^3)$ | CPU ALU (Lapjv / Kalman) | Minimal (State matrices) |
| `prediction_TTI` | $\mathcal{O}(N_{\text{tracks}} \times N_{\text{steps}})$ | CPU Vector FPU | Minimal (Kinematic trajectories) |
| `ESKF` | $\mathcal{O}(S^3)$ where $S=15$ | CPU Low-Latency Core | Minimal (Matrix inversion) |
| `planner` | $\mathcal{O}(K \times N_{\text{traj}})$ | CPU FPU | Minimal (Polynomial roots) |
| `RSS` | $\mathcal{O}(N_{\text{obstacles}})$ | CPU Safety Core | Minimal (Envelope inequalities) |
| `controller` | $\mathcal{O}(1)$ | CPU Real-Time Core | Negligible (Single-iteration PID/Stanley) |
| `CAN_encode` | $\mathcal{O}(1)$ | CPU Byte-Packer | Negligible (Bitwise manipulation) |

---

## 10. Thermal Throttling & Soak Test Protocol

On embedded automotive platforms such as the NVIDIA Jetson Orin Nano, sustained high-load execution causes chip die temperatures to elevate, eventually inducing hardware thermal throttling. When thermal throttling triggers:
1. GPU clock frequencies drop dynamically from 625 MHz down to 300 MHz or lower.
2. CPU frequencies are throttled across all 6 Carmel/Cortex-A78AE cores.
3. Pipeline latencies exhibit severe tail dilation ($p_{99}$ and $p_{99.9}$ increase by 40%–120%).

### 10.1 Jetson Clock Pinning Protocol
Before initiating formal 10,000-frame baseline runs or 30-minute soak benchmarks on target hardware, lock the power profile and fan speed:
```bash
# Set Orin Nano to maximum 15W operational power mode
sudo nvpmodel -m 0

# Pin clocks to maximum rated frequencies
sudo jetson_clocks

# Set fan to maximum speed to ensure thermal saturation plateau
sudo jetson_clocks --fan
```

### 10.2 Continuous Thermal Soak Command
To execute a 30-minute endurance soak test recording frame deadline compliance:
```bash
python benchmarks/run_baseline.py \
    --profile configs/profiles/passenger_car.yaml \
    --frames 54000 \
    --warmup 600 \
    --repeat 1 \
    --device auto \
    --out benchmarks/results/soak_30min
```

---

## 11. Troubleshooting Common Profiling Pitfalls

1. **Windows System Timer Resolution**: By default, Windows thread scheduler resolution is 15.6 ms. The harness uses `time.perf_counter_ns()`, which reads the hardware High-Precision Event Timer (HPET) or CPU Time Stamp Counter (TSC), bypassing thread quantum quantization.
2. **PyTorch Asynchronous Execution**: PyTorch CUDA calls are asynchronous by design. Omitting `torch.cuda.synchronize()` results in capturing only the kernel enqueue time (microsecond scale) rather than true execution time. The harness strictly synchronizes timers.
3. **Garbage Collection (GC) Jitter**: Python's cyclic GC can trigger mid-pipeline, causing spurious 5–15 ms latency spikes. Pass `--gc off` to measure pure algorithmic latency without GC interference, or `--gc on` to evaluate full runtime behavior.
4. **Host Thermal Divergence**: On non-target developer laptops, thermal dissipation limitations will induce CPU throttling across repeated runs (observed in test suites where run times expand by up to 40%). Always isolate thermal states between runs.

---

## 12. Reproduction Checklist for ISO 26262 ASIL-B Audit

When archiving benchmark evidence for an automotive safety review:
- [ ] Execute `python benchmarks/capture_env.py` and confirm `env.json` records git commit SHA and dirty status.
- [ ] Confirm `target_hardware` field in `env.json` matches physical deployment environment (`TARGET` on Jetson, `NON-TARGET` otherwise).
- [ ] Run `pytest benchmarks/tests/test_stats.py -v` and attach output verifying all statistical formulas.
- [ ] Execute `python benchmarks/run_baseline.py --frames 10000 --repeat 3` to capture `results.json` and `raw_latencies.csv`.
- [ ] Verify that individual component sum vs end-to-end latency discrepancy is documented in `results.md`.
- [ ] Ensure all unmeasured physical hardware stages remain tagged `[UNMEASURED]`.

*Document Version: 4.0.0-phase7*  
*Classification: Automotive Perception & Control Timing Reference*  
*Standard Alignment: ISO 26262 ASIL-B Aligned (Process-Only)*

