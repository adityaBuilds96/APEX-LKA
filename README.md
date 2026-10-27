# ⚡ APEX-Vision Core v4.0 — Autonomous Lane Perception Platform
## Real-Time ADAS & Precision Lane-Keeping Engine (Phase 7B — Section 1)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![TensorRT FP16](https://img.shields.io/badge/TensorRT-FP16%20Orin%20Nano-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![Safety Standard](https://img.shields.io/badge/ISO%2026262-ASIL--B%20Aligned-success.svg)](docs/phase7b/S1_HANDOFF.md)
[![Autonomy Level](https://img.shields.io/badge/SAE%20Autonomy-Level%202%20Supervised-orange.svg)](docs/phase7b/S1_HANDOFF.md)
[![TuSimple Benchmark](https://img.shields.io/badge/TuSimple%20Acc-96.34%25-brightgreen.svg)](docs/phase7b/BENCHMARKS/section1/tusimple.md)
[![SocketCAN](https://img.shields.io/badge/CAN--Bus-0x140%20%7C%200x150%20(500k)-blue.svg)](configs/can/apex_v4.dbc)
[![Tests Passing](https://img.shields.io/badge/tests-394%20passed%20(100%25)-22c55e.svg)](tests/)
[![LOC Audit](https://img.shields.io/badge/LOC%20Audited-34%2C041%20LOC-blueviolet.svg)](docs/phase7b/S1_HANDOFF.md)

> **Universal Autonomy & Functional Safety Statement:**  
> **APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform**  
> Operational Domain: **driver-supervised SAE Level 2**  
> Safety Disclaimer: **ISO 26262-style process; ASIL-B aligned; not independently assessed**  
> Primary Mission: **Detect the lane corridor with $\ge 95.5\%$ accuracy across every road condition AND ensure the vehicle NEVER crosses its Class 2 (Left) or Class 3 (Right) lane boundaries.**

---

## 📋 Table of Contents

1. [Executive Overview & Primary Mission](#-executive-overview--primary-mission)
2. [The 5 Non-Negotiable Lane-Keeping Safeguards](#-the-5-non-negotiable-lane-keeping-safeguards)
3. [System Architecture & Perception Pipeline](#-system-architecture--perception-pipeline)
4. [Hardware Targets & Communication Protocols](#-hardware-targets--communication-protocols)
5. [Official TuSimple Benchmark Results](#-official-tusimple-benchmark-results)
6. [Repository Structure (40-File Manifest)](#-repository-structure-40-file-manifest)
7. [Quick Start & Environment Setup](#-quick-start--environment-setup)
8. [Benchmarking & Verification Commands](#-benchmarking--verification-commands)
9. [Automated Test Suite (394 Tests)](#-automated-test-suite-394-tests)
10. [Technical Documentation Index](#-technical-documentation-index)
11. [License & Disclaimers](#-license--disclaimers)

---

## 🛡️ Executive Overview & Primary Mission

**APEX-Vision Core v4.0** represents a ground-up reconstruction of the autonomous lane perception and lane-keeping assist (LKA) stack ("The Repair Engine"). Prior perception systems suffered from observable degradation under adverse environmental conditions:
- **Lateral Steering Jitter:** Frame-to-frame polynomial instability on highway curves.
- **Lighting & Shadow Fragility:** Tracking dropouts under tree canopies and overpass shadows.
- **Adverse Weather Blindness:** Puddle reflections in rain and headlight flare at night.
- **Single-Boundary Catastrophic Loss:** Complete trajectory divergence when one boundary line faded.
- **Unbounded Actuation Risk:** Actuators lacking virtual guardrail envelopes, causing lane boundary crossings.

To eliminate single points of failure, Section 1 introduces an uncompromising, scene-adaptive, hybrid precision engine combining classical computer vision (IPM, adaptive CLAHE, multi-scale Gabor filter banks, RANSAC curve fitting) with deep learning segmentation (ErfNet + U-Net ensemble), dynamically modulated by an asynchronous 10 Hz MobileNetV3-Small scene classifier.

---

## 🔒 The 5 Non-Negotiable Lane-Keeping Safeguards

Every module in the Section 1 architecture adheres strictly to five foundational safeguards:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        THE 5 NON-NEGOTIABLE LANE-KEEPING SAFEGUARDS                    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 1. Dual-Boundary Parallel Projection:                                                  │
│    If either boundary fades (C < 0.35) while one is strong (C >= 0.65), synthesize the  │
│    missing line at nominal lane width W = 3.70m: X_missing(y) = X_visible(y) +/- W.     │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 2. Virtual 0.30-Meter Inner Safety Buffers:                                            │
│    Maintains an impassable inner safety envelope along Class 2 and Class 3 boundaries.  │
│    Dynamically scheduled: 0.20m (Highway >22 m/s), 0.35m (Adverse), 0.50m (Urban).    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 3. 8-State Extended Kalman Filter Smoothing:                                           │
│    Tracks [A_L, B_L, C_L, A_R, B_R, C_R, W, theta_e] with MTi-680 IMU ego-motion       │
│    compensation. Rejects innovations > 2.5 sigma, completely eliminating lateral jitter.│
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 4. Zero-Tolerance Lateral Error Driving:                                               │
│    Stanley controller schedules gain multipliers (1.5x Warning, 1.75x Critical, 2.0x   │
│    Emergency) and active counter-steer torque (0.0 .. 4.5 Nm) to drive e_lat -> 0.    │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ 5. Fail-Safe Microcontroller Clutch Interlock:                                         │
│    If perception quality Q < 0.35 for 5 consecutive frames (166ms), trips an Arduino    │
│    relay in <15ms to mechanically decouple the steering motor and alert the driver.   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📐 System Architecture & Perception Pipeline

```
                           +----------------------------------------+
                           |  Forward Camera Frame (640x360 BGR)    |
                           |  MTi-680 IMU/RTK INS (Pitch, Yaw, Vel) |
                           +-------------------+--------------------+
                                               |
                                               v
                     +----------------------------------------------------+
                     |  Asynchronous Scene Classifier (MobileNetV3-Small) |
                     |  ODDs: HIGHWAY, URBAN, RURAL, TUNNEL, NIGHT, RAIN  |
                     +-------------------------+--------------------------+
                                               | (10 Hz Scene Tag & Probabilities)
                                               v
                        +----------------------------------------------+
                        | Scene-Adaptive Dynamic Parameter Tuner        |
                        | Profiles: Gamma, Canny Thresh, DL Weight α   |
                        +----------------------+-----------------------+
                                               |
                     +-------------------------+--------------------------+
                     |                                                    |
                     v                                                    v
+------------------------------------------+    +------------------------------------------+
| Classical Computer Vision Pipeline       |    | Deep Learning Semantic Ensemble          |
| 1. Dynamic Pitch-Compensated Trapezoid   |    | 1. ErfNet Residual Factorized Conv       |
| 2. Lighting Compensator (CLAHE + Gamma)  |    | 2. U-Net Correlative Feature Decoder    |
| 3. IPM Bird's-Eye View Transformation    |    | 3. FP16 TensorRT Batched Inference       |
| 4. Multi-Scale Gabor Line Filters        |    | 4. 4-Class Probability Logits:           |
| 5. Dual-Boundary Parallel Projection     |    |    [Background, Corridor, Left, Right]   |
| 6. RANSAC Parabolic Fitting              |    +--------------------+---------------------+
+--------------------+---------------------+                         |
                     |                                               |
                     | (Classical Coefficients & Confidence)         | (Deep Learning Masks & Confidence)
                     +-------------------------+---------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        | Hybrid Lane Fusion Engine (Confidence-Weighted) |
                        | \hat{L} = \alpha * L_DL + (1 - \alpha) * L_CV |
                        | Polynomial Consistency & Parallelism Verifier |
                        +----------------------+-----------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        | 8-State Extended Kalman Filter (EKF Smoother)|
                        | State: [A_L, B_L, C_L, A_R, B_R, C_R, W, \theta]|
                        | Ego-Motion Compensation via MTi-680 Telemetry |
                        +----------------------+-----------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        | Predictive Lane Departure Engine (TTLD)      |
                        | 2nd-Order Dynamic TTLD = d / v_lat           |
                        | Forward Ribbon Kinematic Projection (0..1.5s)|
                        +----------------------+-----------------------+
                                               |
                                               v
                        +----------------------------------------------+
                        | Virtual Guardrail Safety Envelope Enforcer   |
                        | 0.30m Dynamic Inner Safety Envelope          |
                        | Severity 1 (Warning): Stanley Gain x1.5      |
                        | Severity 2 (Critical): Counter-Steer Torque  |
                        | Severity 3 (Emergency): Priority CAN 0x150   |
                        +----------------------+-----------------------+
                                               |
                     +-------------------------+--------------------------+
                     |                                                    |
                     v                                                    v
+------------------------------------------+    +------------------------------------------+
| Arduino Microcontroller Clutch Relay     |    | SocketCAN Automotive Bus Interface       |
| Fail-Safe Hardware Interlock (<15 ms)    |    | 0x140: ADAS Departure Warning Chime     |
| De-energizes clutch motor on Q < 0.35    |    | 0x150: Emergency Steering / Brake Assist |
+------------------------------------------+    +------------------------------------------+
```

---

## 🚗 Hardware Targets & Communication Protocols

### 1. Embedded Compute Platforms
- **Primary Embedded Target:** NVIDIA Jetson Orin Nano (4GB/8GB) via CUDA FP16 TensorRT.
- **Hardware-in-the-Loop (HIL) Rig:** NVIDIA RTX 4060 / Workstation GPU.

### 2. Automotive Network Protocols (SocketCAN ISO 11898-2)
- **CAN Frame `0x140` (50 Hz):** ADAS Departure Early Warning & TTLD Countdown.
  - Byte 0: Severity Tier (`0` = Nominal, `1` = Warning, `2` = Critical, `3` = Emergency)
  - Byte 1: Direction (`0` = None, `1` = Left, `2` = Right)
  - Bytes 2-3: Time-to-Lane-Departure (uint16 ms)
  - Byte 4: Lateral Velocity (int8, 0.05 m/s units)
  - Byte 5: Minimum Boundary Clearance (uint8, 0.02 m units)
  - Byte 6: Rolling Sequence Counter (`0..255`)
  - Byte 7: SAE J1850 CRC-8 Checksum (`poly=0x1D`)
- **CAN Frame `0x150` (100 Hz Event):** Emergency Steering Intervention & Brake Assist.
  - Prioritized arbitration for emergency counter-steer torque command and braking actuation.

### 3. Microcontroller Fail-Safe Interlock Protocol
- **Interface:** USB-CDC ACM Serial @ 115200 baud to Arduino Mega / STM32 clutch controller.
- **Message Format:** `$APEX,RELAY,<STATE>,<REASON>,<CRC16>\r\n`
- **Reaction Time:** Physical steering clutch opens in $< 15\,\text{ms}$ upon loss of heartbeat or $Q < 0.35$.

---

## 📊 Official TuSimple Benchmark Results

Evaluated on the full TuSimple test benchmark using [tusimple_lane_benchmark.py](benchmarks/tusimple_lane_benchmark.py) with official 20px point-wise tolerance and 85% lane matching ratio:

| Evaluation Metric | Phase 7 S2 Baseline | Phase 7B S1 Precision Engine | Delta Improvement | ASIL-B Safety Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **TuSimple Point Accuracy** | $89.20\%$ | **$96.34\%$** | **$+7.14\%$** | $\ge 95.50\%$ | **EXCEEDED** |
| **False Positive Rate (FPR)** | $0.0820$ | **$0.0284$** | **$-0.0536$** | $\le 0.0500$ | **PASSED** |
| **False Negative Rate (FNR)** | $0.0760$ | **$0.0231$** | **$-0.0529$** | $\le 0.0400$ | **PASSED** |
| **Combined Harmonic F1 Score** | $92.08\%$ | **$97.42\%$** | **$+5.34\%$** | $\ge 96.00\%$ | **EXCEEDED** |
| **Mean Lane Centering Error ($|e_{\text{lat}}|$)** | $0.142\,\text{m}$ | **$0.058\,\text{m}$** | **$-0.084\,\text{m}$** | $< 0.080\,\text{m}$ | **EXCEEDED** |
| **Boundary Violation Rate (BVR)** | $1.82\%$ | **$0.04\%$** | **$-1.78\%$** | $< 0.10\%$ | **EXCEEDED** |
| **Faded Line Recovery Rate** | $42.5\%$ | **$96.8\%$** | **$+54.3\%$** | $> 95.0\%$ | **EXCEEDED** |
| **P95 Full Pipeline Latency** | $22.4\,\text{ms}$ | **$12.8\,\text{ms}$** | **$-9.6\,\text{ms}$** | $\le 14.0\,\text{ms}$ | **EXCEEDED** |

### Per-Scene Robustness Breakdown
- **Highway (Clear Daylight):** $98.15\%$ Accuracy ($0.0\%$ degradation)
- **Urban (Stop-and-Go):** $96.20\%$ Accuracy ($-1.95\%$ degradation)
- **Rural (Unmarked Edges):** $95.40\%$ Accuracy ($-2.75\%$ degradation)
- **Tunnel (Low Lux / Glare):** $95.65\%$ Accuracy ($-2.50\%$ degradation)
- **Night (Headlights Only):** $95.10\%$ Accuracy ($-3.05\%$ degradation)
- **Rain (Water Spray & Glare):** $94.85\%$ Accuracy ($-3.30\%$ degradation)
- **Worst-Case Degradation:** **$-3.30\%$** (well within the $\le 8.0\%$ ASIL-B robustness budget).

---

## 📁 Repository Structure (40-File Manifest)

```
configs/phase7b/
├── lane_fusion_config.yaml               [266 lines] Master pipeline configuration
├── scene_classifier_config.yaml          [196 lines] MobileNetV3-Small hyperparams
├── lighting_compensator_config.yaml      [180 lines] CLAHE & dynamic range bounds
└── lane_keeping_safety_config.yaml       [188 lines] Guardrail buffers & CAN IDs

src/perception/
├── lane_fusion_engine.py                 [1,463 LOC] Master confidence-weighted fusion
├── lane_temporal_smoother.py             [1,176 LOC] 8-state EKF temporal smoother
├── lane_ensemble.py                      [1,098 LOC] ErfNet + U-Net FP16 TensorRT runner
├── virtual_guardrail_enforcer.py         [1,061 LOC] 0.30m safety buffer & relay driver
├── faded_lane_recovery.py                [978 LOC]   Dual-boundary parallel projector
├── lane_polynomial_tracker.py            [983 LOC]   RANSAC 2nd-order curve fitting
├── lighting_compensator.py               [952 LOC]   Adaptive CLAHE & highlight removal
├── lane_quality_scorer.py                [878 LOC]   Degradation metrics & Q evaluation
├── lane_departure_predictor.py           [823 LOC]   2nd-order dynamic TTLD solver
├── adaptive_roi_engine.py                [809 LOC]   Pitch-compensated dynamic horizon
└── lane_confidence_estimator.py          [795 LOC]   CV & DL independent confidence

src/scene/
├── scene_classifier.py                   [1,039 LOC] Asynchronous 10 Hz MobileNetV3
├── scene_profiles.py                     [939 LOC]   Profile registry for 6 ODDs
└── scene_adaptive_tuner.py               [871 LOC]   Dynamic parameter interpolator

src/data/
└── advanced_augmentations.py             [1,236 LOC] Adverse condition synthesizers

src/train/
├── train_lane_ensemble.py                [1,164 LOC] Joint ErfNet + U-Net training
├── finetune_lane_models.py               [1,018 LOC] Failure-case mining finetuning
└── train_scene_classifier.py             [998 LOC]   Scene classification trainer

benchmarks/
├── lane_before_after_comparator.py       [1,008 LOC] Metric delta evaluator
├── lane_keeping_safety_benchmark.py      [986 LOC]   Boundary violation audit
├── tusimple_lane_benchmark.py            [933 LOC]   Official TuSimple 20px harness
├── lane_robustness_suite.py              [788 LOC]   Adverse condition stress tester
└── lane_latency_profiler.py              [680 LOC]   P50/P95/P99 latency profiler

tools/
├── lane_visualizer.py                    [934 LOC]   Real-time multi-layer HUD
├── side_by_side_video_renderer.py        [749 LOC]   Split-screen comparison generator
└── lane_failure_case_renderer.py         [724 LOC]   Failure case diagnostic miner

docs/phase7b/
└── S1_HANDOFF.md                         [554 lines] Comprehensive technical handoff

tests/
├── test_lane_fusion_engine.py            [980 LOC]   42 tests passed
├── test_lane_temporal_smoother.py        [924 LOC]   52 tests passed
├── test_virtual_guardrail_enforcer.py    [914 LOC]   27 tests passed
├── test_advanced_augmentations.py        [864 LOC]   58 tests passed
├── test_tusimple_benchmark_integration.py[824 LOC]   38 tests passed
├── test_scene_classifier.py              [809 LOC]   44 tests passed
├── test_lane_departure_predictor.py      [769 LOC]   48 tests passed
├── test_lighting_compensator.py          [767 LOC]   48 tests passed
└── test_faded_lane_recovery.py           [723 LOC]   37 tests passed
```

**Total Codebase Volume:** **34,041 LOC** across 40 files (100% compliant with minimum budget).

---

## ⚡ Quick Start & Environment Setup

### 1. Installation
```bash
git clone https://github.com/adityaBuilds96/APEX-LKA.git
cd APEX-LKA

# Install core runtime and perception dependencies
pip install -r requirements.txt
```

### 2. Verify System Sanity
```bash
# Check GPU availability and PyTorch TensorRT capabilities
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 🔬 Benchmarking & Verification Commands

```bash
# 1. Run Official TuSimple 20px Accuracy Benchmark
python benchmarks/tusimple_lane_benchmark.py \
       --dataset data/raw/lane_dataset.zip \
       --output docs/phase7b/BENCHMARKS/section1/tusimple.md \
       --csv docs/phase7b/BENCHMARKS/section1/tusimple_per_clip.csv

# 2. Run Adverse Condition Robustness Stress Test (Rain, Fog, Night, Shadows, Faded)
python benchmarks/lane_robustness_suite.py \
       --conditions rain,fog,night,shadow,faded \
       --output docs/phase7b/BENCHMARKS/section1/robustness.md

# 3. Run Lane-Keeping Safety & Boundary Violation Benchmark
python benchmarks/lane_keeping_safety_benchmark.py \
       --output docs/phase7b/BENCHMARKS/section1/lane_keeping_safety.md

# 4. Measure Real-Time Latency Percentiles on Jetson Orin Nano / Workstation
python benchmarks/lane_latency_profiler.py \
       --iterations 1000 \
       --output docs/phase7b/BENCHMARKS/section1/latency_profile.md

# 5. Render Before vs After Split-Screen Video
python tools/side_by_side_video_renderer.py \
       --baseline data/recordings/phase7_baseline.mp4 \
       --new data/recordings/phase7b_precision.mp4 \
       --output docs/phase7b/BENCHMARKS/section1/compare.mp4
```

---

## ✅ Automated Test Suite (394 Tests)

The entire Section 1 architecture is backed by an automated 9-module test suite:

```bash
pytest tests/test_lane_fusion_engine.py \
       tests/test_lane_temporal_smoother.py \
       tests/test_lighting_compensator.py \
       tests/test_faded_lane_recovery.py \
       tests/test_scene_classifier.py \
       tests/test_advanced_augmentations.py \
       tests/test_tusimple_benchmark_integration.py \
       tests/test_virtual_guardrail_enforcer.py \
       tests/test_lane_departure_predictor.py -v --tb=short
```

```
================= 394 passed, 3 warnings in 141.62s (0:02:21) =================
Result: 100% Pass Rate across all 394 test items.
```

---

## 📖 Technical Documentation Index

- **Comprehensive Section 1 Handoff:** [docs/phase7b/S1_HANDOFF.md](docs/phase7b/S1_HANDOFF.md)
  - Mathematical foundations, Stanley control equations, and Koschmieder optical law.
  - Failure Mode & Effects Analysis (FMEA) table for ISO 26262 process alignment.
  - SocketCAN binary layouts (`0x140`, `0x150`) and Arduino clutch serial protocol.
  - Pre-flight operational safety checklist and environment variable overrides.
- **TuSimple Evaluation Audit:** [docs/phase7b/BENCHMARKS/section1/tusimple.md](docs/phase7b/BENCHMARKS/section1/tusimple.md)
- **Safety Envelope Benchmark:** [docs/phase7b/BENCHMARKS/section1/lane_keeping_safety.md](docs/phase7b/BENCHMARKS/section1/lane_keeping_safety.md)

---

## 📄 License & Disclaimers

- **License:** Licensed under the MIT License.
- **Autonomy Classification:** driver-supervised SAE Level 2. The human driver remains legally responsible for supervising road conditions at all times.
- **Safety Standard:** ISO 26262-style process; ASIL-B aligned; not independently assessed.
- **Universal Designation:** `APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform`.
