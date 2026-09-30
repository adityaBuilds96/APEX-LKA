# APEX-LKA Inference Architecture & Thread Safety Specification

## Overview

The APEX-LKA inference subsystem is designed for low-latency, real-time automotive perception operating under strict non-blocking execution budgets (< 30ms cycle latency). To guarantee zero race conditions, deadlocks, or UI freezing across Windows, Linux, and Jetson Orin Nano, the architecture strictly segregates execution contexts across threads and processes.

---

## 1. Concurrency Model

```
                    ┌─────────────────────────┐
                    │   Camera Capture Thread │
                    │   (Native 30-60 FPS)    │
                    └───────────┬─────────────┘
                                │ Latest frame (overwriting)
                                ▼
                    ┌─────────────────────────┐
                    │    LatestFrameBuffer    │ (Depth=1, Mutex Protected)
                    └───────────┬─────────────┘
                                │ Newest unread frame
                                ▼
                    ┌─────────────────────────┐
                    │ Inference Worker Thread │ (Decoupled Background Loop)
                    │  - Graceful Degradation │
                    │  - MemoryGuard & GC     │
                    │  - LKA State Tracker    │
                    └───────────┬─────────────┘
                                │ Latest perception & telemetry
                                ▼
                    ┌─────────────────────────┐
                    │   LatestResultBuffer    │ (Depth=1, Mutex Protected)
                    └───────────┬─────────────┘
                                │ Atomic read
                                ▼
                    ┌─────────────────────────┐
                    │   Presentation Layer    │ (Streamlit / OpenCV CLI / CAN)
                    └─────────────────────────┘
```

---

## 2. Shared State Synchronization Audit

All inter-thread data exchanges occur through designated thread-safe synchronization primitives:

| Component | State Protected | Synchronization Primitive | Contention Policy |
| :--- | :--- | :--- | :--- |
| `LatestFrameBuffer` | Latest BGR numpy frame, frame ID, dropped count | `threading.Lock` | Drop unread stale frames; reader never blocks writer |
| `LatestResultBuffer`| `InferenceResult`, `LKATelemetry` | `threading.Lock` | Non-blocking read; returns cached copy if busy |
| `BoundedHistoryBuffer` | Sliding telemetry history (depth $\le 100$) | `threading.Lock` | Fixed ring-buffer capacity; prevents memory leaks |
| `SafeExecutor` | Failure counters, last valid result | Atomic variables | Thread-safe failure shielding across calls |
| `MemoryGuard` | Memory snapshots, cleanup events | `threading.Lock` | Thread-safe sample registration and buffer clearing |
| `CrashLogger` | Forensic log ring-buffer, model state | `threading.Lock` | Safe concurrent exception capture |

---

## 3. Strict Model Weight Immutability

- **Zero In-Memory Model Updates During Inference:** All neural network weights (`LaneSegNet`, ONNX Session, TensorRT Engine) are placed in **evaluation-only mode** (`model.eval()`).
- **No Gradient Computation:** All PyTorch inference passes are executed inside an explicit `with torch.no_grad():` scope, preventing computation graph accumulation and memory leaks.
- **Process Isolation for Training:** Model training (`src/training/train.py`) runs in a **completely separate OS process** (never in an inference worker thread). This prevents Python GIL contention, avoids CUDA context corruption, and guarantees uninterrupted perception during background model updates.

---

## 4. Graceful Degradation Stack

The inference pipeline employs a 6-tier hierarchical fail-safe:
1. **Level 1 (TensorRT .plan on GPU):** High-speed INT8/FP16 engine on Jetson Orin Nano.
2. **Level 2 (ONNX Runtime on GPU):** CUDA / TensorRT execution providers on NVIDIA GPUs.
3. **Level 3 (PyTorch on GPU):** Native PyTorch inference via CUDA.
4. **Level 4 (ONNX Runtime on CPU):** Multi-threaded CPU execution provider fallback (5–15 FPS).
5. **Level 5 (Classical CV Baseline):** Pure OpenCV morphological edge & Hough transform pipeline.
6. **Level 6 (Last Known Good Temporal Cache):** Maintains previous lane geometry and emits `LANE_LOST` warning on frame drops.
