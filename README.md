# ⚡ APEX LKA — Autonomous Lane Perception & Driver Assistance Workstation

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-5c3ee8.svg)](https://opencv.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-ff4b4b.svg)](https://streamlit.io/)
[![Test Suite](https://img.shields.io/badge/tests-72%20passed-22c55e.svg)](tests/)
[![Architecture](https://img.shields.io/badge/APEX--RLP-4--Class%20Schema-38bdf8.svg)](configs/project_config.yaml)

> **Production-grade Computer Vision & ADAS Engineering Platform:**  
> Seamlessly bridging raw sensor data ingestion, classical computer vision auto-annotation, multi-check quality assurance, deep learning semantic segmentation, and real-time lateral vehicle guidance telemetry.

---

## 📋 Table of Contents

1. [System Overview & Safety Notice](#-system-overview--safety-notice)
2. [Key Capabilities & Innovations](#-key-capabilities--innovations)
3. [System Architecture](#-system-architecture)
4. [4-Class Semantic Segmentation Standard](#-4-class-semantic-segmentation-standard)
5. [Project Structure](#-project-structure)
6. [Quick Start & Setup](#-quick-start--setup)
7. [CLI Operations Guide (`run.py`)](#-cli-operations-guide-runpy)
8. [Dataset Command Center & Upload Experience](#-dataset-command-center--upload-experience)
9. [Classical CV Auto-Annotation Engine](#-classical-cv-auto-annotation-engine)
10. [9-Check Automated Quality Assurance System](#-9-check-automated-quality-assurance-system)
11. [Lane Geometry & Lateral Control Model](#-lane-geometry--lateral-control-model)
12. [Verification & Test Suite](#-verification--test-suite)
13. [License & Acknowledgments](#-license--acknowledgments)

---

## 🛡️ System Overview & Safety Notice

APEX LKA is an end-to-end vision-based Lane Keeping Assist (LKA) and Lane Departure Warning (LDW) engineering workstation. It demonstrates production-level machine learning software design:

- **Perception Pipeline:** Decoupled multi-threaded camera ingestion, robust classical computer vision (HLS color space + Canny + Hough + 2nd-degree polynomial fitting), and ML semantic segmentation.
- **Data Engineering Infrastructure (Phase 2):** Drag-and-drop glassmorphic upload interface, automated zero-touch ingestion pipeline, thread-safe asynchronous batch processing, classical pseudo-mask generation, 9-check quality scoring, and a 7-tab command center console.
- **Guidance & Telemetry:** Signed lateral offset computation, heading error calculation, temporal state stabilization, LDW departure warnings, and steering recommendations.

> ⚠️ **SAFETY CRITICAL NOTICE:**  
> This system outputs **software-level steering recommendations and visual alerts only**. It is **strictly decoupled from vehicle CAN bus, steer-by-wire, motors, and physical actuators**. Designed for simulation, edge testing, and research prototyping.

---

## 🚀 Key Capabilities & Innovations

- **Heavenly Upload Experience:** Glassmorphic drag-and-drop dropzone supporting image batches, raw videos (`.mp4`, `.avi`, `.mov`, `.mkv`), and ZIP archives with Zip-Slip path traversal defense.
- **Instant Client-Side Preview Grid:** Progressive non-blocking thumbnail rendering (200×120px) with cached disk storage and virtual pagination.
- **Classical CV Auto-Annotation Engine:** Automatically produces 4-class semantic pseudo-masks from raw road frames, scoring each with multi-factor confidence (`HIGH`, `MEDIUM`, `LOW`) and 1-click candidate promotion.
- **9-Check QA Audit System:** Automatically scans image-mask pairs for dimension mismatch, illegal class IDs, road coverage, lane spatial consistency (crossover detection), lane width plausibility, continuity, void holes, and sky region leakage.
- **7-Tab Dataset Command Center:** Complete nerve center for upload, analytics, Kanban workflow tracking, class distribution loss weighting, interactive side-by-side annotation viewing, QA auditing, and train/val/test splitting.
- **Dual Perception Backends:** Hot-swappable between Classical CV and deep ML segmentation.
- **Automotive Workstation HUD:** Uncluttered presentation layer featuring real-time lateral centerline offset gauge, system health matrix, and live event stream.

---

## 🏗️ System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        RAW SENSOR / INTAKE SOURCES                     │
│   Live Camera (DirectShow/V4L2) │ Videos (.mp4/.avi) │ ZIP Archives    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   PHASE 2 DATA INGESTION & PIPELINE                    │
│   • Video Frame Extraction (FPS downsampling)                          │
│   • SHA-256 Exact & Perceptual dHash Deduplication                     │
│   • Multi-Format Validation & Corrupted File Quarantine                │
│   • Classical CV 4-Class Auto-Annotation & Confidence Scoring          │
│   • 9-Check Automated Quality Assurance Audit                          │
│   • Sequence-Preserving Train / Val / Test Partitioning                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                     DUAL-ENGINE PERCEPTION PIPELINE                    │
│                                                                        │
│   ┌──────────────────────────────┐    ┌────────────────────────────┐   │
│   │      CLASSICAL CV ENGINE     │    │    ML SEGMENTATION MODEL   │   │
│   │ • HLS Yellow/White Filtering │    │ • DeepLabV3+ / MobileNetV3 │   │
│   │ • Canny Edge & Hough Lines   │    │ • 4-Class Probability Map  │   │
│   │ • Road Surface Flood Fill    │    │ • Cross-Entropy Loss       │   │
│   └──────────────┬───────────────┘    └──────────────┬─────────────┘   │
└──────────────────┼───────────────────────────────────┼─────────────────┘
                   │                                   │
                   └─────────────────┬─────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      LANE GEOMETRY & LKA STATE ENGINE                  │
│   • 2nd-Order Polynomial Curve Fitting: x = a*y² + b*y + c             │
│   • Centerline Intercept & Signed Lateral Offset (cm / px)             │
│   • Heading Angle Error (θ) & Curvature Estimation (κ)                 │
│   • Temporal State Machine (NORMAL → DRIFTING → DEPARTING → CRITICAL)  │
│   • Steering Angle Recommendation Clamping (±15°)                      │
└────────────────────────────────────┬───────────────────────────────────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────┐
│                    AUTOMOTIVE PRESENTATION WORKSTATION                 │
│   • Real-Time Decoupled Video Stream Viewport & HUD Overlay            │
│   • Lateral Centerline Offset Barometer & Deviation Gauge              │
│   • 7-Tab Dataset Command Center & Ingestion Console                   │
│   • Session Recorder & Synchronized Replayer                           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🎨 4-Class Semantic Segmentation Standard

APEX LKA adheres to a single-channel `uint8` semantic class encoding where every pixel belongs strictly to one mutually exclusive class:

| Class ID | Semantic Label | Standard Color (RGB) | Visual Representation | Target Frequency |
|:---:|:---|:---:|:---:|:---:|
| **0** | **Background** | `(15, 23, 42)` | Dark Slate / Sky / Scenery | 60% – 70% |
| **1** | **Road Surface** | `(168, 85, 247)` | Violet / Purple Asphalt Region | 25% – 35% |
| **2** | **Left Lane Boundary** | `(34, 197, 94)` | Vibrant Green Marking | 2% – 5% |
| **3** | **Right Lane Boundary** | `(56, 189, 248)` | Sky Blue / Cyan Marking | 2% – 5% |

---

## 📂 Project Structure

```
APEX-LKA/
├── configs/
│   └── project_config.yaml         # Master configuration (paths, thresholds, hyperparameters)
├── dashboard/
│   ├── app.py                      # Primary Automotive Perception Workstation
│   ├── camera_stream.py            # Decoupled thread-safe camera frame grabber
│   ├── components/
│   │   ├── annotation_viewer.py    # Side-by-side & alpha-blended mask inspector
│   │   ├── dataset_stats.py        # KPI metrics and class frequency loss weights
│   │   ├── event_log.py            # Real-time event streaming and alert audit
│   │   ├── lka_hud.py              # Augmented Reality windshield HUD renderer
│   │   ├── offset_chart.py         # Temporal lateral offset telemetry graph
│   │   ├── progress_tracker.py     # Live background pipeline progress monitor
│   │   ├── recorder.py             # Session recording and telemetry sync
│   │   ├── steering_indicator.py   # Steering wheel angle recommendation gauge
│   │   ├── telemetry.py            # Diagnostic indicators and subsystem health
│   │   ├── thumbnail_grid.py       # Responsive preview grid with virtual pagination
│   │   └── upload_zone.py          # Glassmorphic drag-and-drop upload zone
│   └── pages/
│       ├── 01_Diagnostics.py       # Sensor health and pipeline latency diagnostics
│       ├── 02_Perception.py        # Intermediate filter stages deep-dive
│       ├── 03_Recordings.py        # Synchronized session recording player
│       ├── 04_Dataset.py           # 7-Tab Dataset Command Center
│       └── 07_Settings.py          # Dynamic parameter tuner
├── src/
│   ├── config.py                   # Dataclass schemas and path resolvers
│   ├── data_collection/
│   │   ├── dataset_splitter.py     # Sequence-aware train/val/test partitioning
│   │   ├── inspect_dataset.py      # Statistical distribution inspector
│   │   └── video_to_frames.py      # FPS-throttled frame extraction
│   ├── dataset/
│   │   ├── annotation_prep.py      # Workspace initializer & manifest manager
│   │   ├── auto_annotator.py       # Classical CV pseudo-mask generator
│   │   ├── batch_processor.py      # Thread-safe async batch execution queue
│   │   ├── ingestion.py            # ZIP/Folder ingestor with structure detection
│   │   ├── pipeline_orchestrator.py# End-to-end autonomous data pipeline
│   │   ├── quality_scorer.py       # 9-check annotation defect scoring engine
│   │   ├── report.py               # Markdown and JSON QC report generator
│   │   ├── thumbnail_generator.py  # High-throughput thumbnail downsampler
│   │   └── validator.py            # Image/mask integrity and dHash deduplication
│   ├── inference/
│   │   ├── classical_cv.py         # HLS + Canny + Hough lane detection engine
│   │   ├── pipeline.py             # Unified inference orchestration entry point
│   │   ├── postprocessing.py       # Polynomial fitting and geometry projection
│   │   ├── predictor.py            # Model factory and status coordinator
│   │   └── preprocessing.py        # Normalization and model ROI transform
│   ├── lane_geometry/
│   │   ├── lane_estimator.py       # 2nd-degree polynomial regression
│   │   └── offset_calculator.py    # Signed centerline offset and drift calculator
│   └── lka_engine/
│       └── lka_state.py            # Temporal stabilization and LDW state machine
├── tests/                          # 72 comprehensive automated unit tests
├── requirements.txt                # Pinned production dependencies
├── run.py                          # Master CLI dispatcher
└── README.md
```

---

## ⚡ Quick Start & Setup

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/adityaBuilds96/APEX-LKA.git
cd APEX-LKA

# Create and activate Python virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify System Readiness

```bash
python run.py env
```

### 3. Launch the Workstation

```bash
# Launch Streamlit dashboard
python run.py dashboard
```
*Navigate to `http://localhost:8501` to access the console.*

---

## 💻 CLI Operations Guide (`run.py`)

The unified CLI provides instant command execution for headless servers and automation pipelines:

| Command | Description | Example Usage |
|:---|:---|:---|
| `env` | Inspects environment, GPU availability, and dataset paths | `python run.py env` |
| `auto-annotate` | Generates 4-class pseudo-masks using Classical CV | `python run.py auto-annotate --confidence-threshold 0.70` |
| `quality-check` | Audits image-mask pairs with 9-check QA engine | `python run.py quality-check --export --strict` |
| `pipeline` | Full autonomous pipeline (Intake → Validate → Annotate → Split) | `python run.py pipeline --source data/raw_videos/drive.mp4 --auto-split` |
| `ingest` | Ingests and audits dataset archive or folder | `python run.py ingest --zip data/road_dataset.zip` |
| `split` | Partitions annotated data into Train/Val/Test by session | `python run.py split` |
| `infer` | Runs live inference on webcam, video file, or test frame | `python run.py infer --source webcam` |
| `dashboard` | Launches the Streamlit Perception Workstation | `python run.py dashboard` |

---

## 🎛️ Dataset Command Center & Upload Experience

Located at **`dashboard/pages/04_Dataset.py`**, the Dataset Command Center provides 7 specialized tabs:

1. **📤 Upload & Ingest:** Drag-and-drop dropzone, live background processing queue, and throughput telemetry.
2. **📊 Dataset Overview:** Total frame counts, resolution breakdown, partition distribution bar, and storage footprint metrics (MB).
3. **🎨 Annotation Status:** 3-column Kanban workflow board (`🔴 PENDING`, `🟡 AUTO-ANNOTATED`, `🟢 VERIFIED`), velocity tracker (~45 frames/hr), and bulk approval tools.
4. **🔍 Class Distribution:** Pixel-level frequency distribution for all 4 classes, imbalance audit, and calculated loss function weights ($w_c = \frac{1.0}{f_c \cdot C}$).
5. **🖼️ Annotation Viewer:** Interactive side-by-side inspection, alpha-blended overlay (0%–100% opacity slider), mask-only view, and boundary contour edge modes.
6. **✅ Quality Assurance:** 1-click 9-check automated QA scan with defect breakdown and flagged frame navigation.
7. **⚙️ Data Operations:** Custom ratio train/val/test splitting, sequence leakage verification, staging purge, and curated dataset ZIP export.

---

## 🤖 Classical CV Auto-Annotation Engine

The auto-annotation engine (`src/dataset/auto_annotator.py`) turns unlabelled road imagery into supervised segmentation training data:

1. **ROI Constraint:** Extracts lower 65% of the frame to isolate the active roadway.
2. **Road Surface Flood-Fill (Class 1):** Dual-threshold HLS filtering (`L: 20-180, S: 0-80`) with morphological closing ($15\times15$) and connected flood fill seeded from frame bottom-center.
3. **Lane Boundary Rasterization (Classes 2 & 3):** Combined yellow/white color masks $\to$ Canny edge detection $\to$ Hough Transform $\to$ left/right slope segregation ($m < -0.3$, $m > 0.3$) $\to$ 2nd-order polynomial fit $\to$ 14px anti-aliased polyline rasterization.
4. **Multi-Factor Confidence Scoring:**
   $$\text{Confidence} = 0.30 \cdot S_{\text{road}} + 0.20 \cdot L_{\text{det}} + 0.20 \cdot R_{\text{det}} + 0.15 \cdot W_{\text{plausible}} + 0.15 \cdot R^2$$
   - `HIGH` ($\ge 0.70$): Auto-approved candidate for training.
   - `MEDIUM` ($0.40 \le \text{conf} < 0.70$): Recommended for human visual review.
   - `LOW` ($< 0.40$): Flagged for manual correction.

---

## 🛡️ 9-Check Automated Quality Assurance System

The quality scoring engine (`src/dataset/quality_scorer.py`) applies 9 automated checks to every image-mask pair:

```
[CHECK 1] Dimension Match (CRITICAL)       ──> mask.shape == image.shape
[CHECK 2] Valid Class IDs (CRITICAL)       ──> pixels in {0, 1, 2, 3}
[CHECK 3] Road Coverage Ratio (WARNING)    ──> 0.10 <= road_roi <= 0.85
[CHECK 4] Lane Marking Coverage (WARNING)  ──> 0.005 <= lane_roi <= 0.08
[CHECK 5] Spatial Consistency (CRITICAL)   ──> x_left < x_center < x_right (No crossover)
[CHECK 6] Lane Width Plausibility (WARNING)──> 100px <= separation <= 400px at bottom
[CHECK 7] Lane Continuity (INFO)           ──> <= 3 disconnected segments per side
[CHECK 8] Background Contamination (WARN)  ──> Zero void holes inside road surface
[CHECK 9] Sky / Horizon Leak (WARNING)     ──> Upper 30% sky zone has < 2% road pixels
```

**Quality Score Calculation:**
$$\text{Score} = \frac{\sum (w_i \cdot \text{pass}_i)}{\sum w_i} \quad \text{where } w_{\text{critical}}=3.0, \; w_{\text{warning}}=1.5, \; w_{\text{info}}=0.5$$

- **EXCELLENT** ($\ge 0.85$): Auto-approved for training partitions.
- **ACCEPTABLE** ($0.65 - 0.85$): Included with minor advisory warnings.
- **NEEDS REVIEW** ($0.40 - 0.65$): Held for review in Annotation Studio.
- **REJECTED** ($< 0.40$): Quarantined; excluded from dataset splits.

---

## 📐 Lane Geometry & Lateral Control Model

### 1. Polynomial Curve Fitting
For detected left and right lane boundaries, coordinates are modeled via 2nd-degree polynomials:
$$x = f(y) = a \cdot y^2 + b \cdot y + c$$
- $a$: Lane curvature factor
- $b$: Heading angle tangent at frame origin
- $c$: Lateral bottom intercept position

### 2. Centerline & Signed Lateral Offset
At bottom scanline $y_{\text{eval}} = H - 1$:
$$x_{\text{center}} = \frac{x_{\text{left}}(y_{\text{eval}}) + x_{\text{right}}(y_{\text{eval}})}{2}$$
$$\text{Offset}_{\text{px}} = x_{\text{camera}} - x_{\text{center}} \quad (x_{\text{camera}} = W / 2)$$
$$\text{Offset}_{\text{meters}} = \text{Offset}_{\text{px}} \times \text{Scale Factor}$$

*Convention:*
- **Negative ($-$)**: Vehicle is shifted to the **Left** of lane center $\to$ Steer **Right**.
- **Positive ($+$)**: Vehicle is shifted to the **Right** of lane center $\to$ Steer **Left**.

### 3. Lane Departure Warning (LDW) State Machine
- **NORMAL** ($|\text{offset}| < 0.25\text{m}$): Stable in lane; green HUD status.
- **DRIFTING** ($0.25\text{m} \le |\text{offset}| < 0.50\text{m}$): Approaching boundary; amber warning.
- **DEPARTING** ($|\text{offset}| \ge 0.50\text{m}$): Lane departure event; red visual alert and audio pulse.

---

## 🧪 Verification & Test Suite

APEX LKA features comprehensive test coverage with zero dependencies on external network services:

```bash
# Run the complete test suite
pytest tests/ -v
```

```
============================= 72 passed in 4.68s ==============================
✓ tests/test_auto_annotator.py (9 tests passed)
✓ tests/test_batch_processor.py (4 tests passed)
✓ tests/test_camera_stream.py (4 tests passed)
✓ tests/test_config.py (6 tests passed)
✓ tests/test_dataset_ingestion.py (9 tests passed)
✓ tests/test_inference_pipeline.py (13 tests passed)
✓ tests/test_lka_state.py (5 tests passed)
✓ tests/test_quality_scorer.py (10 tests passed)
✓ tests/test_upload_pipeline.py (5 tests passed)
✓ tests/test_video_to_frames.py (7 tests passed)
```

---

## 📄 License & Acknowledgments

- **License:** MIT License. Free for academic, portfolio, and research evaluation.
- **Built With:** PyTorch, OpenCV, Streamlit, Albumentations, NumPy, and Rich.
- **Author:** Aditya Chavhan ([@adityaBuilds96](https://github.com/adityaBuilds96))
