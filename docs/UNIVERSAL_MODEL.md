# APEX-LKA Universal Road Perception & Training Guide

## 1. The Challenge of Universal Road Perception
Traditional Advanced Driver Assistance Systems (ADAS) rely on high-contrast white and yellow painted lane markings on standardized asphalt highways. In real-world autonomous deployments—such as SAE India BAJA off-road competitions, university campuses, and rural tracks—painted lines rarely exist.

APEX-LKA solves this through a **Unified 4-Class Drivable Path & Boundary Standard**:
- **Class 0 (Background):** Sky, trees, off-road vegetation, guardrails, obstacles.
- **Class 1 (Drivable Road Surface):** Asphalt, concrete, gravel, packed dirt, or paved paths.
- **Class 2 (Left Boundary):** Painted left lane marking, asphalt curb edge, or dirt-to-grass transition.
- **Class 3 (Right Boundary):** Painted right lane marking, shoulder edge, or trail boundary.

---

## 2. Multi-Mode Path Detection Engine

The system supports 3 operational modes selected dynamically via `mode: "auto"` in `project_config.yaml`:

### Mode A: Painted Highway Mode (`painted`)
- **Use Case:** Standard multilane freeways and marked municipal avenues.
- **Mechanism:** HLS color isolation (Yellow Hue $[15, 35]$, White Lightness $> 190$), Canny edge detection, and RANSAC 2nd-degree polynomial curve fitting.
- **Guardrail Rejection:** Tight adaptive trapezoidal ROI anchored exclusively to the lower drivable road surface.

### Mode B: Campus & Urban Edge Mode (`edge`)
- **Use Case:** Unpainted university campus roads, parking areas, and curbside alleys.
- **Mechanism:** Sobel vertical gradient filtering combined with structural asphalt-to-curb contrast transitions.
- **Fallback:** Uses road boundary contours to establish virtual vehicle centerlines.

### Mode C: Off-Road & Dirt Trail Mode (`drivable`)
- **Use Case:** SAE BAJA dirt circuits, gravel roads, and unpaved courses.
- **Mechanism:** Seeded morphological flood-fill from bottom-center $(W/2, H-1)$, color variance thresholding, and drivable surface segmentation.
- **Trajectory Estimation:** Centers the vehicle between the left and right extremities of the contiguous drivable polygon.

---

## 3. Recommended Dataset Composition

To achieve universal generalization without overfitting to a single environment, compose your training dataset according to the following ratio:

| Environment Tier | Target % | Road Types | Challenging Features |
| :--- | :--- | :--- | :--- |
| **Tier 1: Marked Highways** | $35\%$ | Freeways, divided avenues | Dash lines, solid white/yellow, merges |
| **Tier 2: Unmarked Campus Roads** | $35\%$ | Paved campus paths, alleys | Curbs, parked vehicles, pedestrian crossings |
| **Tier 3: Off-Road & Dirt Tracks** | $30\%$ | Dirt trails, gravel, grass borders | Rutting, dynamic shadows, dust, uneven terrain |

### Lighting & Weather Diversity
- $50\%$ Clear daylight / midday sun
- $25\%$ Low-angle golden hour (severe glare and elongated tree shadows)
- $15\%$ Overcast / cloudy conditions (low contrast)
- $10\%$ Dusk / low-light conditions

---

## 4. Transfer Learning & Fine-Tuning Recipe

To adapt a pre-trained APEX-LKA checkpoint (`best_model.pth`) to your specific vehicle or competition course:

### Step 1: Collect Custom Frames
Record 10–15 minutes of driving on your target track:
```bash
python run.py record --duration 180 --tag "baja_track_run1"
```

### Step 2: Auto-Annotate & Validate
Generate pseudo-masks and audit annotation quality:
```bash
python run.py auto-annotate --confidence-threshold 0.70
python run.py quality-check --export
python run.py split
```

### Step 3: Progressive Fine-Tuning
Fine-tune LaneSegNet using a reduced initial learning rate:
```bash
python run.py train --resume models/exported/best_model.pth --lr 0.0001 --epochs 20
```
- **Epochs 1–5:** Freezes MobileNetV3 encoder, adapting only the FPN decoder and attention heads to the new road texture.
- **Epochs 6+:** Unfreezes full network for end-to-end alignment.

### Step 4: Validate on Target Track
Evaluate the fine-tuned model and export to ONNX:
```bash
python run.py evaluate --split test --export-report
python run.py export --fp16 --package
```
