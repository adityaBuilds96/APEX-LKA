# APEX-LKA End-to-End Workflow Guide

This document outlines the complete operational lifecycle of the APEX-LKA system from an empty repository to hardware deployment on an autonomous vehicle.

---

## The 7-Stage Pipeline Lifecycle

```
[1. RECORD] ──> [2. INGEST] ──> [3. ANNOTATE & QA] ──> [4. TRAIN] ──> [5. EVALUATE] ──> [6. EXPORT] ──> [7. DEPLOY]
   Live Cam         ZIP / Video      Auto-Mask & 9-Check    RTX 4060 AMP     Full Test Metrics      ONNX & TRT      Jetson & Arduinos
```

---

## Stage 1: Data Recording & Ingestion

### Live Vehicle Camera Recording
To record test runs directly from a vehicle webcam or CSI camera:
```bash
# Capture a 60-second test session with telemetry
python run.py record --duration 60 --tag "campus_loop_01" --device-id 0
```
This writes raw JPEG frames and synchronized `telemetry.csv` into `data/recordings/session_YYYYMMDD_HHMMSS/`.

### Video & Batch Ingestion
To extract frames from recorded dashcam videos:
```bash
# Extract frames at 5 FPS with automatic SSIM deduplication
python run.py collect --video data/raw_videos/drive01.mp4 --fps 5

# Ingest an external dataset ZIP or directory
python run.py ingest --zip data/road_dataset.zip
```

---

## Stage 2: Classical Auto-Annotation & QA Audit

APEX-LKA features a zero-touch annotation pipeline that converts raw images into 4-class segmentation masks:
- Class 0: Background
- Class 1: Drivable Road Surface
- Class 2: Left Lane Marking
- Class 3: Right Lane Marking

```bash
# Generate pseudo-masks for all unannotated frames
python run.py auto-annotate --confidence-threshold 0.70

# Run 9-check quality assurance audit (verifies geometry, continuity, and class IDs)
python run.py quality-check --export
```

### Partitioning
Split the validated dataset into training, validation, and test splits:
```bash
python run.py split
```
This distributes data according to `project_config.yaml` (`70% train`, `15% val`, `15% test`) while preserving continuous driving sequences.

---

## Stage 3: Bulletproof Model Training (RTX 4060)

Train **LaneSegNet** with Automatic Mixed Precision (`torch.cuda.amp`), Cosine Annealing, Exponential Moving Average (EMA), and layer unfreezing:

```bash
# Standard 50-epoch training run
python run.py train --epochs 50 --batch-size 8 --lr 0.001

# Resume from latest checkpoint after interruption
python run.py train --resume
```

### Crash Protection & Memory Guarantees
- **OOM Shield:** If GPU memory is exhausted, the trainer automatically halves batch size, increases gradient accumulation, and continues.
- **SIGINT Handler:** Pressing `Ctrl+C` saves emergency checkpoint `interrupted_checkpoint.pt`.
- **Thermal Guard:** Pauses training if GPU temperature exceeds 85°C.

---

## Stage 4: Comprehensive Evaluation & Diagnostics

Evaluate model performance on the held-out test dataset:
```bash
python run.py evaluate --split test --export-report
```
Generates:
- Confusion matrix and per-class IoU/Dice/F1 metrics.
- Latency statistics (mean, p50, p95, p99 FPS).
- Side-by-side visual galleries in `results/reports/`.

---

## Stage 5: Universal Edge Model Export

Export the PyTorch model to hardware-optimized runtime artifacts:
```bash
# 1. Standard ONNX FP32 (Opset 17)
python run.py export --output models/exported/best_model.onnx

# 2. Edge-Optimized FP16 (50% storage reduction)
python run.py export --fp16

# 3. Jetson Orin Nano TensorRT Engine (.plan)
python run.py export --tensorrt

# 4. Generate Complete Edge Deployment Bundle (.zip)
python run.py export --package
```

---

## Stage 6: Edge Deployment & Hardware Integration

Deploy to the NVIDIA Jetson Orin Nano with 2× Arduino Mega microcontrollers:
```bash
# 1. Unpack deployment bundle on Jetson
unzip apex_deployment_package.zip -d ~/APEX-LKA
cd ~/APEX-LKA

# 2. Run standalone zero-UI perception runner
python deploy/jetson_run.py --config deploy/deploy_config.yaml

# 3. Enable autonomous supervisor watchdog service
sudo cp deploy/apex-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-watchdog.service
```

---

## Troubleshooting & FAQ

| Problem | Root Cause | Solution |
| :--- | :--- | :--- |
| `CUDA out of memory` during training | Batch size too large for VRAM | Trainer automatically falls back, or use `--batch-size 4` |
| `Camera 0 could not be opened` | Device busy or permissions | Check `ls /dev/video*` or grant dialout permissions |
| Guardrail lock-on in Classical CV | High-contrast metal edges | Adaptive ROI and asphalt flood-fill are enabled by default |
| CAN bus timeout | MCP2515 crystal frequency mismatch | Verify `16MHz` vs `8MHz` in `arduino/node_1/node_1.ino` |
