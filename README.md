# ⚡ APEX LKA — Autonomous Lane Perception & Driver Assistance Workstation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5c3ee8.svg)](https://opencv.org/)
[![ONNX Runtime](https://img.shields.io/badge/ONNX-Opset%2017-005ced.svg)](https://onnxruntime.ai/)
[![TensorRT](https://img.shields.io/badge/TensorRT-Jetson%20Orin-76b900.svg)](https://developer.nvidia.com/tensorrt)
[![CAN Bus](https://img.shields.io/badge/CAN%20Bus-500%20kbps-orange.svg)](deploy/protocol_spec.md)
[![Test Suite](https://img.shields.io/badge/tests-233%20passed-22c55e.svg)](tests/)

> **Autonomous Vehicle Perception & ADAS Engineering Platform:**  
> A complete, never-crash autonomous perception stack connecting an NVIDIA Jetson Orin Nano with 2× Arduino Mega 2560 microcontrollers over automotive CAN Bus and redundant Serial. Spans raw data recording, classical auto-annotation, 9-check QA auditing, LaneSegNet deep learning training with AMP on RTX 4060, ONNX/TensorRT edge export, and real-time lateral vehicle control.

---

## 📋 Table of Contents

1. [System Overview & Architecture](#-system-overview--architecture)
2. [Phase 3 Core Innovations](#-phase-3-core-innovations)
3. [The 7-Stage Workflow](#-the-7-stage-workflow)
4. [Quick Start & Setup](#-quick-start--setup)
5. [CLI Operations Guide (`run.py`)](#-cli-operations-guide-runpy)
6. [Hardware & Jetson Deployment](#-hardware--jetson-deployment)
7. [6-Level Graceful Degradation Stack](#-6-level-graceful-degradation-stack)
8. [Performance Benchmarks](#-performance-benchmarks)
9. [Documentation Directory](#-documentation-directory)
10. [Verification & Test Suite](#-verification--test-suite)
11. [License & Acknowledgments](#-license--acknowledgments)

---

## 🛡️ System Overview & Architecture

APEX-LKA is designed for SAE India BAJA off-road vehicle competitions, university research, and commercial autonomous development. The stack operates symmetrically across:
- **Workstation Training:** High-throughput Automatic Mixed Precision (AMP) training on NVIDIA RTX 4060 (8GB VRAM).
- **Vehicle Edge Deployment:** Ultra-low latency (< 30ms budget, 45+ FPS) inference on NVIDIA Jetson Orin Nano (4GB/8GB) via TensorRT and ONNX Runtime.
- **Microcontroller Actuation:** Dual-node Arduino Mega 2560 control network for steering rack actuation, slew-rate limiting, throttle/brake safety watchdogs, and optical wheel speed feedback.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        NVIDIA JETSON ORIN NANO                         │
│  ┌───────────────────────┐  ┌──────────────────────┐  ┌─────────────┐  │
│  │ CSI / USB Camera      │  │ TensorRT / ONNX Engine│  │ LKA Engine  │  │
│  │ 640x360 @ 30 FPS      │──│ LaneSegNet (4-Class) │──│ PD Steering │  │
│  └───────────────────────┘  └──────────────────────┘  └──────┬──────┘  │
│                                                              │         │
│  ┌────────────────────────────────────────────────────────┐  │         │
│  │ VehicleCommunicator (Automatic CAN -> Serial Failover)  │◄─┘         │
│  └───────────────────────────┬────────────────────────────┘            │
└──────────────────────────────┼─────────────────────────────────────────┘
                               │
       ┌───────────────────────┴───────────────────────┐
       │ CAN Bus (500 kbps, MCP2515 SPI) & USB Serial  │
       ▼                                               ▼
┌──────────────────────────────┐        ┌──────────────────────────────┐
│  Arduino Mega Node 1         │        │  Arduino Mega Node 2         │
│  - Steering Rack Servo Driver│        │  - Throttle & Brake PWM      │
│  - Slew Limiter (120°/s)     │        │  - Optical Wheel Speed Pulse │
│  - 500ms Neutral Watchdog    │        │  - Hydraulic Pressure Sensor │
│  - Manual Override ISR       │        │  - 500ms Safety Cutoff       │
└──────────────────────────────┘        └──────────────────────────────┘
```

---

## 🚀 Phase 3 Core Innovations

1. **Universal Path Perception (LaneSegNet):** Custom MobileNetV3-Small FPN with Squeeze-and-Excitation (SE) channel attention and spatial attention heads. Works seamlessly across painted freeways, unpainted campus roads, and off-road dirt trails without guardrail lock-on.
2. **Never-Crash 6-Level Graceful Degradation:** Automatic hierarchy stepping down from GPU TensorRT $\to$ GPU ONNX $\to$ GPU PyTorch $\to$ CPU ONNX $\to$ Classical CV Baseline $\to$ Last Known Good Cache. Recovers automatically every 60s.
3. **Automotive CAN Bus Protocol:** Bit-packed 500 kbps differential messaging (Little-Endian, IDs `0x050`, `0x100`, `0x200`, `0x300`, `0x301`) with microsecond latency and line-delimited JSON Serial fallback.
4. **Autonomous Supervisor Watchdog:** Independent monitor pinging every 500ms. Restarts unresponsive processes within 3s and enforces safe-mode emergency stop if $\ge 3$ crashes occur in 60s.
5. **Continuous Memory & Thermal Protection:** Detects monotonic memory leaks, automatically flushes PyTorch CUDA caches, and dynamically throttles frame rates ($>80^\circ\text{C} \to 15\,\text{FPS}$, $>85^\circ\text{C} \to 10\,\text{FPS}$, $>90^\circ\text{C} \to \text{ESTOP}$).

---

## 🔄 The 7-Stage Workflow

| Stage | Operation | Command | Primary Output |
| :---: | :--- | :--- | :--- |
| **1** | **Record** | `python run.py record --duration 60` | Synchronous JPEGs & `telemetry.csv` |
| **2** | **Ingest** | `python run.py ingest --zip data.zip` | Quarantined & deduplicated frames |
| **3** | **Annotate & QA** | `python run.py auto-annotate && python run.py quality-check` | 4-class pseudo-masks & QA audit |
| **4** | **Split** | `python run.py split` | Sequence-preserving Train/Val/Test splits |
| **5** | **Train** | `python run.py train --epochs 50` | Bulletproof AMP checkpoint (`best_model.pth`) |
| **6** | **Evaluate** | `python run.py evaluate --export-report` | Forensic mIoU, confusion matrix, HTML report |
| **7** | **Export & Deploy**| `python run.py export --package` | ONNX, TensorRT, and edge bundle (`.zip`) |

Detailed walkthrough: [docs/WORKFLOW.md](docs/WORKFLOW.md).

---

## ⚡ Quick Start & Setup

### 1. Installation
```bash
git clone https://github.com/adityaBuilds96/APEX-LKA.git
cd APEX-LKA

# Install full development requirements
pip install -r requirements.txt
```

### 2. Verify System Sanity
```bash
# Verify environment dependencies and GPU availability
python run.py env

# Run pre-flight health diagnostics
python -m src.utils.health_check
```

### 3. Launch Interactive Workstation
```bash
python run.py dashboard
```
Opens the Streamlit workstation at `http://localhost:8501` featuring live camera inference, drag-and-drop image analysis, video stream processing, and interactive session replay.

---

## 💻 CLI Operations Guide (`run.py`)

```bash
# System status & environment
python run.py env

# Record live camera session
python run.py record --duration 30 --tag "track_test"

# Ingest dataset archive or folder
python run.py ingest --zip data/road_dataset.zip
python run.py collect --video data/raw_videos/drive.mp4 --fps 5

# Auto-annotation and quality assurance
python run.py auto-annotate --confidence-threshold 0.70
python run.py quality-check --export --strict
python run.py split

# Training (AMP enabled, crash recovery active)
python run.py train --epochs 50 --batch-size 8 --lr 0.001
python run.py train --resume

# Evaluation & Reporting
python run.py evaluate --split test --export-report

# Universal Model Export
python run.py export --output models/exported/best_model.onnx
python run.py export --fp16
python run.py export --tensorrt
python run.py export --package

# Hardware speed benchmark across providers
python run.py benchmark
```

---

## 🏎️ Hardware & Jetson Deployment

Complete hardware wiring, pinouts, and systemd service instructions:
- [docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md) — Jetson Orin Nano setup, systemd watchdog, and CLI runner.
- [arduino/WIRING_GUIDE.md](arduino/WIRING_GUIDE.md) — MCP2515 SPI connections, $120\,\Omega$ bus termination, and star grounding.
- [deploy/protocol_spec.md](deploy/protocol_spec.md) — Bit-packed CAN frame layouts and Serial JSON schemas.

---

## 📊 Performance Benchmarks

Detailed report available in [results/benchmarks/phase3_benchmarks.md](results/benchmarks/phase3_benchmarks.md).

| Metric | Target | Measured Result | Status |
| :--- | :--- | :--- | :--- |
| **RTX 4060 Training Epoch Time (300 frames)** | $< 60\,\text{s}$ | **$21.4\,\text{s}$** | PASSED |
| **RTX 4060 Peak VRAM Allocation** | $< 6.0\,\text{GB}$ | **$2.84\,\text{GB}$** | PASSED |
| **RTX 4060 PyTorch Inference Throughput** | $> 100\,\text{FPS}$ | **$128.2\,\text{FPS}$** | PASSED |
| **Jetson Orin Nano TensorRT FP16 Throughput** | $> 40\,\text{FPS}$ | **$70.9\,\text{FPS}$** | PASSED |
| **ONNX FP16 Model Size** | $< 10.0\,\text{MB}$ | **$7.91\,\text{MB}$** | PASSED |
| **CAN Bus Transfer Time (8-byte Frame)** | $< 5.0\,\text{ms}$ | **$0.24\,\text{ms}$** | PASSED |

---

## 📚 Documentation Directory

- [docs/WORKFLOW.md](docs/WORKFLOW.md): Step-by-step guide from zero to deployed model.
- [docs/JETSON_DEPLOYMENT.md](docs/JETSON_DEPLOYMENT.md): NVIDIA Jetson Orin Nano and dual-node Arduino configuration.
- [docs/UNIVERSAL_MODEL.md](docs/UNIVERSAL_MODEL.md): Road types, dataset composition, and transfer learning fine-tuning.
- [arduino/WIRING_GUIDE.md](arduino/WIRING_GUIDE.md): Physical wiring, SPI pinouts, and bus termination rules.
- [deploy/protocol_spec.md](deploy/protocol_spec.md): Complete CAN Bus and Serial line protocol specification.
- [src/inference/README.md](src/inference/README.md): Concurrency model, thread safety audit, and mutex contracts.

---

## ✅ Verification & Test Suite

The platform includes **233 automated unit tests** covering all modules:

```bash
pytest tests/ -v --tb=short
```

```
============================== test session starts ===============================
rootdir: C:\Users\chavh\OneDrive\LKA.MAIN
collected 233 items

tests/test_auto_annotator.py ......                                         [  2%]
tests/test_batch_processor.py ...                                           [  3%]
tests/test_camera_stream.py ...                                             [  5%]
tests/test_config.py ...                                                    [  6%]
tests/test_dataset.py .......                                               [  9%]
tests/test_dataset_ingestion.py .......                                     [ 12%]
tests/test_e2e_workflow.py .                                                [ 12%]
tests/test_evaluate.py ........                                             [ 16%]
tests/test_export.py ........                                               [ 19%]
tests/test_inference_pipeline.py ................                           [ 26%]
tests/test_jetson_deploy.py .......                                         [ 29%]
tests/test_lka_state.py ......                                              [ 32%]
tests/test_losses.py .........                                              [ 36%]
tests/test_model.py ........                                                [ 39%]
tests/test_quality_scorer.py .......                                        [ 42%]
tests/test_robustness.py .............                                      [ 47%]
tests/test_trainer.py ..........                                            [ 51%]
tests/test_training.py .........                                            [ 55%]
tests/test_universal_detection.py ...................................       [ 70%]
tests/test_upload_pipeline.py ...                                           [ 72%]
tests/test_vehicle_comm.py .........                                        [ 75%]
tests/test_video_to_frames.py ...........................................   [100%]

============================= 233 passed in 48.20s ==============================
```

---

## 📄 License & Acknowledgments

- Built for SAE India BAJA autonomous competition vehicle prototyping.
- Core architecture: **LaneSegNet** (MobileNetV3 FPN with Spatial & SE Attention).
- Licensed under the MIT License.
