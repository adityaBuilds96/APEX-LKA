# Lane Keep Assist (LKA) — Portfolio ML/ADAS Project

> **A genuine end-to-end Computer Vision pipeline: data collection → annotation → training → evaluation → live inference → steering recommendation.**

---

## Table of Contents

1. [Project Objective](#1-project-objective)
2. [Architecture Decision — Why Semantic Segmentation?](#2-architecture-decision)
3. [System Architecture Diagram](#3-system-architecture-diagram)
4. [Hardware & Software Requirements](#4-requirements)
5. [Project Structure](#5-project-structure)
6. [Quick Start](#6-quick-start)
7. [Stage 1 — Data Collection](#7-stage-1--data-collection)
8. [Stage 2 — Annotation Guide](#8-stage-2--annotation-guide)
9. [Stage 3 — Training](#9-stage-3--training)
10. [Stage 4 — Evaluation](#10-stage-4--evaluation)
11. [Stage 5 — Live Inference](#11-stage-5--live-inference)
12. [Lane Geometry & Steering Logic](#12-lane-geometry--steering-logic)
13. [Experiment Tracking](#13-experiment-tracking)
14. [Limitations](#14-limitations)
15. [Future Improvements](#15-future-improvements)

---

## 1. Project Objective

Build a computer-vision prototype that:

| # | Capability |
|---|---|
| 1 | Detects lane markings in road images/video |
| 2 | Identifies left and right lane boundaries |
| 3 | Estimates lane center |
| 4 | Estimates camera/vehicle lateral position |
| 5 | Calculates lateral offset (signed) |
| 6 | Detects left/right drift |
| 7 | Produces a steering correction recommendation |
| 8 | Displays everything visually in real time |
| 9 | Reports confidence and quality metrics |
| 10 | Logs all inference results for analysis |

> ⚠️ **Safety disclaimer**: This system outputs **steering recommendations only**. It is **not connected to any vehicle, motor, or steering actuator**. It is a research prototype.

---

## 2. Architecture Decision — Why Semantic Segmentation?

### Approach Comparison

| Approach | Pros | Cons | Suitable for LKA? |
|---|---|---|---|
| **A. Classical CV** (Canny + Hough) | No training data needed, fast | Fails on curves, shadows, faded markings, rain | Baseline only |
| **B. Object Detection** (YOLO) | Detects objects as boxes | Lane markings are thin, curved lines — boxes are semantically wrong | No |
| **C. Semantic Segmentation** | Pixel-level lane mask, handles curves | Needs annotated masks | ✅ **Selected** |
| **D. Instance Segmentation** | Separates individual lane instances | Over-engineering for 2-lane scenario | Optional future upgrade |

### Final Choice: **Lightweight Semantic Segmentation**

**Why:**
- Lane markings are **pixel-level spatial structures** — the model must know *exactly where* each pixel belongs to left vs. right lane.
- Bounding boxes (object detection) fundamentally cannot represent a curved line.
- Semantic segmentation gives per-pixel class probabilities that feed directly into the lane geometry estimator.
- A custom lightweight encoder-decoder with a MobileNetV3 backbone achieves real-time FPS on CPU.

**Classes:**

| Value | Class | Colour in mask |
|---|---|---|
| 0 | Background (road, sky, vehicles) | Black (0,0,0) |
| 1 | Left lane marking | Red (128,0,0) → value=1 |
| 2 | Right lane marking | Green (0,128,0) → value=2 |

---

## 3. System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       INPUT SOURCES                             │
│   Webcam / Video file / Single image                            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    PREPROCESSING                                │
│   Resize → Normalize → ROI crop → Augment (train only)         │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              LANE DETECTION MODEL (Semantic Seg)                │
│   LaneSegNet: MobileNetV3 backbone + lightweight decoder         │
│   Output: 3-class probability map [H × W × 3]                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LANE GEOMETRY MODULE                         │
│   Extract left/right lane pixels → polynomial curve fit         │
│   → lane center → lateral offset calculation                    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEERING CONTROLLER (PD)                       │
│   steering_cmd = Kp × error + Kd × d(error)/dt                 │
│   → STEER LEFT / STEER RIGHT / KEEP CENTER / LOW CONFIDENCE    │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT / DASHBOARD                           │
│   Visual overlay + Streamlit dashboard + CSV log                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Requirements

### Hardware
- **CPU**: Any modern laptop (Intel/AMD). Training without GPU is slow but works.
- **GPU**: NVIDIA GPU with CUDA ≥ 11.7 strongly recommended for training.
- **RAM**: 8 GB minimum; 16 GB recommended.
- **Storage**: 5–20 GB for video, frames, and model checkpoints.
- **Camera**: Dashcam, phone camera, or webcam mounted facing the road.

### Software
- Python 3.10+
- See `requirements.txt` for full list

---

## 5. Project Structure

```
lane_keep_assist/
│
├── data/
│   ├── raw_videos/          ← Place your road videos here
│   ├── raw_frames/          ← Auto-extracted frames + metadata.csv
│   ├── annotated/
│   │   ├── images/          ← Copy curated frames here for annotation
│   │   └── masks/           ← Segmentation masks (output from labeling tool)
│   ├── train/images+masks/  ← Auto-generated by dataset_splitter.py
│   ├── val/images+masks/
│   └── test/images+masks/
│
├── annotations/             ← Reserved for future label exports (CVAT, etc.)
│
├── models/
│   ├── checkpoints/         ← Training checkpoints (epoch_N.pth)
│   └── exported/            ← Best model for inference (best_model.pth)
│
├── notebooks/               ← Jupyter notebooks for exploration
│
├── src/
│   ├── config.py            ← Centralized config loader
│   ├── data_collection/
│   │   ├── video_to_frames.py   ← STAGE 1: Extract frames
│   │   ├── inspect_dataset.py   ← Inspect current dataset
│   │   ├── dataset_stats.py     ← Detailed statistics
│   │   └── dataset_splitter.py  ← Split into train/val/test
│   ├── preprocessing/       ← Resize, normalize, augment
│   ├── training/            ← Model, loss, training loop
│   ├── inference/           ← Live inference engine
│   ├── lane_geometry/       ← Curve fitting, offset, steering
│   └── visualization/       ← Overlay drawing utilities
│
├── configs/
│   └── project_config.yaml  ← ALL tunable parameters
│
├── logs/
│   ├── training/            ← TensorBoard logs + experiment CSV
│   └── inference/           ← Real-time inference CSV logs
│
├── results/
│   ├── plots/               ← Training curves, evaluation charts
│   └── metrics/             ← JSON metric files
│
├── dashboard/
│   └── app.py               ← Streamlit dashboard
│
├── tests/                   ← Unit and integration tests
│
├── requirements.txt
├── run.py                   ← Master entry point
└── README.md
```

---

## 6. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Check your environment
python run.py env

# 3. Collect data — place video in data/raw_videos/ then:
python run.py collect --fps 5

# 4. Inspect extracted frames
python run.py inspect

# 5. Annotate (see Stage 2 below)

# 6. Split dataset
python run.py split

# 7. Train
python run.py train

# 8. Evaluate
python run.py evaluate

# 9. Live inference
python run.py infer --source webcam
```

---

## 7. Stage 1 — Data Collection Guide

> These recommendations reflect engineering best practices.
> They do not guarantee any specific model performance on your data.

---

### 7.1 Camera Setup

#### Recommended Hardware
A smartphone or dashcam both work. A dedicated dashcam is more stable.

| Parameter | Recommended | Minimum acceptable |
|---|---|---|
| Resolution | 1920×1080 (1080p) | 1280×720 (720p) |
| Frame rate | 30 FPS | 24 FPS |
| Mount position | Center of windshield, behind rear-view mirror | Any fixed, central position |
| Tilt angle | Horizon visible in upper 35–45% of frame | Camera must see at least 5 m of road ahead |
| Stabilization | Optical or electronic preferred | Suction-cup mount acceptable |

#### What the frame should look like
```
┌────────────────────────────────────────┐  ← top of frame
│         sky / surroundings            │  ← ~35%
├────────────────────────────────────────┤  ← horizon
│                                        │
│   road, lane markings, vehicles        │  ← ~65%
│                                        │
└────────────────────────────────────────┘  ← bottom of frame (bonnet/hood visible is fine)
```

Avoid cameras mounted too high (too much sky) or too low (too close to hood).

---

### 7.2 What Road Conditions to Record

Collect **diversity**. A model trained only on clear-highway footage will fail
on urban roads, curves, or shadows. Aim for all conditions listed below.

| Priority | Condition | Notes |
|---|---|---|
| **Must have** | Clear daylight, dry road, highway | Most lane markings visible, easiest to annotate |
| **Must have** | Clear daylight, dry road, urban | Varied lane widths, more intersections |
| **Must have** | Curves (left and right) | Lane markings curve; model must handle polynomials |
| **Should have** | Shadows across lane markings | Common failure case for classical CV |
| **Should have** | Faded/worn lane markings | Real roads have imperfect markings |
| **Should have** | Light traffic | Vehicles partially blocking lane |
| **Nice to have** | Overcast / slightly cloudy | Reduces sharp shadows |
| **Nice to have** | Wet road (after rain, no active rain) | Reflections present |
| **Later** | Night with headlights | Requires separate model or augmentation |
| **Later** | Active rain | Very difficult; requires significant data |

**Do not mix night and day footage in your initial dataset.** Train on one
condition first; validate it works; then expand.

---

### 7.3 How Many Videos / Sessions

| Stage | Target | Notes |
|---|---|---|
| Proof of concept | 3–5 sessions, 5–10 min each | ~1,000–3,000 raw frames |
| Portfolio-quality dataset | 10+ sessions, varied conditions | ~5,000–15,000 raw frames |
| Annotated subset (minimum) | 300 image-mask pairs | Enough for initial training |
| Annotated subset (good) | 1,000–2,000 pairs | Better generalization |

You do **not** need to annotate all extracted frames. Extract generously,
then curate a diverse subset for annotation.

---

### 7.4 How to Avoid Thousands of Near-Identical Frames

Driving at highway speed generates very different frames every second.
Urban stop-and-go generates many nearly identical frames at low speed or
at red lights. Strategies to avoid redundant frames:

1. **Use the extraction FPS setting** — 3–5 FPS is almost always enough.
   There is rarely useful new information between frames at 30 FPS.
   ```bash
   # 5 FPS from a 30 FPS video → 1 in 6 frames kept
   python src/data_collection/video_to_frames.py --video my_drive.mp4 --fps 5
   ```

2. **Use the duplicate threshold** — frames that change less than this
   pixel-difference value are automatically skipped.
   ```bash
   # More aggressive dedup (skips more similar frames)
   python src/data_collection/video_to_frames.py --video my_drive.mp4 --fps 5 --dup_threshold 6.0
   ```

3. **Trim your videos before extraction** — cut out:
   - Parked / stationary segments (red lights > 30 s)
   - Car wash, garage, private driveway
   - Any non-road scene

4. **After extraction, inspect and manually delete** obviously bad frames:
   ```bash
   python src/data_collection/inspect_dataset.py --show 20
   ```

---

### 7.5 Session Naming Convention

Use a consistent, descriptive naming scheme. This makes it easy to trace
any frame back to its source session.

```
data/raw_videos/
├── s01_highway_day_clear.mp4
├── s02_urban_day_clear.mp4
├── s03_highway_day_shadows.mp4
├── s04_urban_curves.mp4
└── s05_highway_overcast.mp4
```

**Rules:**
- Start with session number `s01`, `s02` ... for easy sorting
- Include road type: `highway`, `urban`, `rural`
- Include time of day: `day`, `dusk`, `night`
- Include weather: `clear`, `cloudy`, `wet`, `rain`

Extracted frame filenames automatically follow the video stem:
```
s01_highway_day_clear_frame000150.jpg
s01_highway_day_clear_frame000156.jpg
```

This lets you instantly know which session a frame came from.

---

### 7.6 Extraction Commands

```bash
# Single video at default 5 FPS
python src/data_collection/video_to_frames.py \
    --video data/raw_videos/s01_highway_day_clear.mp4

# Single video at custom FPS
python src/data_collection/video_to_frames.py \
    --video data/raw_videos/s01_highway_day_clear.mp4 \
    --fps 3

# All videos in the folder at once
python src/data_collection/video_to_frames.py \
    --video_dir data/raw_videos \
    --fps 5

# More aggressive duplicate filtering
python src/data_collection/video_to_frames.py \
    --video data/raw_videos/s01_highway_day_clear.mp4 \
    --fps 5 --dup_threshold 6.0

# Save as PNG instead of JPEG (larger files, lossless)
python src/data_collection/video_to_frames.py \
    --video data/raw_videos/s01_highway_day_clear.mp4 \
    --fps 5 --format png
```

Metadata for every extracted frame is saved automatically to:
```
data/raw_frames/metadata.csv
```

---

### 7.7 Inspect the Extracted Frames

After extraction, always inspect before moving to annotation:

```bash
# Full report + show 9 random frames in a window
python src/data_collection/inspect_dataset.py

# More frames displayed
python src/data_collection/inspect_dataset.py --show 20

# Statistics only (no window)
python src/data_collection/inspect_dataset.py --stats-only

# Skip corruption check (faster for large datasets)
python src/data_collection/inspect_dataset.py --no-corrupt-check
```

The inspector will show:
- Total frame count
- Frames per source video/session
- Resolution statistics (flags mixed resolutions)
- Corrupted / unreadable file detection
- Random frame thumbnails

---

### 7.8 What to Do With the Frames Next

Once you have a satisfactory set of frames:

1. **Curate**: copy selected frames to `data/annotated/images/`
   (choose diverse, quality frames — not all extracted frames)
2. **Annotate**: use CVAT or LabelMe to draw lane mask polygons
3. **Split**: run `python run.py split`
4. **Train**: run `python run.py train`

See Section 8 (Annotation Guide) for exact labeling instructions.

---

## 8. Stage 2 — Annotation Guide

### Tool Recommendation

**Use [CVAT](https://cvat.ai) (free, runs locally or online):**

1. Go to [app.cvat.ai](https://app.cvat.ai) or self-host
2. Create a new project: `LKA Lane Detection`
3. Add labels:
   - `left_lane` — color: `#FF0000`
   - `right_lane` — color: `#00FF00`
4. Upload images from `data/annotated/images/`
5. Annotate using **polygon** tool
6. Export as **Segmentation Mask** format → place in `data/annotated/masks/`

**Alternative**: [LabelMe](https://github.com/wkentaro/labelme) (simpler, fully local)

### What to Annotate — Precise Rules

#### General Rules
- Annotate **only what is clearly visible** to a human driver
- Do **not** annotate inferred/guessed lane positions
- Lane markings include: solid lines, dashed lines, double lines, road edge markings

#### Class 1 — Left Lane Marking
The lane boundary immediately to the **left** of the vehicle's travel lane.
Includes the full visible width of the marking.

#### Class 2 — Right Lane Marking
The lane boundary immediately to the **right** of the vehicle's travel lane.

#### Difficult Cases

| Scenario | Annotation Rule |
|---|---|
| Straight road | Annotate full visible length of both lines |
| Curved road | Follow the curve precisely using polygon points |
| Dashed line | Annotate each visible dash individually (or connect with thin polygon) |
| Faded marking | Annotate only what you can actually see. If invisible → **leave unannotated** |
| Shadow crossing lane | Annotate the lane marking despite the shadow |
| Vehicle blocking lane | Annotate only visible portion. Do NOT guess what's behind the vehicle |
| Intersection | Annotate only stop-line/markings visible; lane boundaries typically end |
| Missing lane marking | If lane boundary is road edge/curb only → still annotate as appropriate class |
| Multiple lanes | Annotate ONLY the two lanes immediately bounding the ego vehicle |
| Night conditions | Annotate only what is illuminated by headlights |
| Rain/glare | Annotate what is visible through the glare. If completely obscured → skip |
| Merge/fork | Annotate the lane you are currently in |

#### Mask Format
- Output: **single-channel PNG** (grayscale)
- Pixel values: `0` = background, `1` = left lane, `2` = right lane
- Resolution: Same as input image (do NOT resize masks separately)
- Filename: **same stem as input image**
  - Image: `session01_frame000150.jpg`
  - Mask:  `session01_frame000150.png`

### Annotation Quality Check

```bash
# After annotating, validate coverage
python src/data_collection/inspect_dataset.py --split annotated --stats-only
```

---

## 9. Stage 3 — Training

> **Wait**: Complete annotation and splitting before training.

```bash
# Split annotated data
python run.py split

# Start training (uses configs/project_config.yaml)
python run.py train

# Monitor with TensorBoard
tensorboard --logdir logs/training
```

Training will:
- Save checkpoints every N epochs to `models/checkpoints/`
- Save best model (by validation IoU) to `models/exported/best_model.pth`
- Log training curves to `logs/training/`
- Print per-epoch metrics table

### Training Metrics Explained

| Metric | What it measures | Good value |
|---|---|---|
| **Mean IoU** | Intersection over Union across all classes. Primary metric. | > 0.70 |
| **Dice Coefficient** | 2× overlap / total. More sensitive than IoU. | > 0.75 |
| **Pixel Accuracy** | % pixels correctly classified. Can be misleading (imbalanced classes). | > 0.90 |
| **Precision** | Of pixels predicted as lane, how many actually are lane? | > 0.70 |
| **Recall** | Of actual lane pixels, how many did we detect? | > 0.70 |

---

## 10. Stage 4 — Evaluation

```bash
python run.py evaluate
```

Produces:
- Per-class IoU table
- Confusion matrix
- Example prediction images in `results/plots/`
- Full metrics JSON in `results/metrics/`

---

## 11. Stage 5 — Live Inference

```bash
# Webcam
python run.py infer --source webcam

# Video file
python run.py infer --source video --file data/raw_videos/test.mp4

# Single image
python run.py infer --source image --file data/raw_frames/sample.jpg

# Dashboard
python run.py dashboard
```

---

## 12. Lane Geometry & Steering Logic

### Coordinate Convention

```
Image coordinate system:
  Origin: top-left corner
  X: increases rightward
  Y: increases downward

Vehicle position:
  vehicle_center_x = image_width / 2  (camera assumed centered on vehicle)

Lateral error (signed):
  lateral_error = vehicle_center_x - lane_center_x

Sign convention:
  lateral_error > 0  →  vehicle is RIGHT of lane center  →  steer LEFT
  lateral_error < 0  →  vehicle is LEFT  of lane center  →  steer RIGHT
  lateral_error ≈ 0  →  vehicle is centered              →  KEEP CENTER
```

### Steering Controller

```
steering_command = Kp × lateral_error + Kd × d(lateral_error)/dt

Where:
  Kp = proportional gain (default: 0.5)
  Kd = derivative gain  (default: 0.1)
  lateral_error is normalized to [-1, 1]

Output mapping:
  steering_command > +threshold  →  STEER LEFT
  steering_command < -threshold  →  STEER RIGHT
  |steering_command| < threshold →  KEEP CENTER
  confidence < iou_threshold     →  LOW CONFIDENCE / NO RECOMMENDATION
```

---

## 13. Experiment Tracking

Every training run saves a row to `logs/training/experiment_log.csv`:

| Column | Description |
|---|---|
| `run_id` | Unique identifier |
| `timestamp` | Start time |
| `model_arch` | Architecture name |
| `backbone` | Backbone used |
| `num_images` | Training set size |
| `epochs` | Number of epochs run |
| `learning_rate` | LR used |
| `batch_size` | Batch size |
| `augmentation` | Augmentation config hash |
| `val_mean_iou` | Best validation IoU |
| `test_mean_iou` | Test IoU (run after training) |
| `model_path` | Path to saved best model |

---

## 14. Limitations

- Dataset size: Results depend entirely on how many diverse images are annotated.
- CPU inference: Expect 5–15 FPS without a GPU.
- Single road type: Trained on your own roads; may not generalize to other geographies.
- No temporal modeling: Frame-by-frame inference only; no Kalman smoothing yet.
- Fixed camera mount: Camera must be mounted consistently between training and inference.
- Night/rain: Requires explicit training examples for each condition.

---

## 15. Future Improvements

- [ ] Temporal smoothing with Kalman filter on lane estimates
- [ ] Bird's-eye-view (BEV) perspective transform for curvature estimation
- [ ] Instance segmentation to handle multiple lanes
- [ ] ONNX/TensorRT export for faster inference
- [ ] GPS + IMU fusion for better offset estimation
- [ ] Synthetic data augmentation using GAN
- [ ] CULane / TuSimple pre-training → fine-tune on own data

---

## Results

*(Populated after training)*

| Metric | Value |
|---|---|
| Dataset size | TBD |
| Best val IoU | TBD |
| Test IoU | TBD |
| Inference FPS (CPU) | TBD |

---

*Built as a genuine ML/ADAS portfolio project. All training data, metrics, and results are from actual execution.*
