"""
dashboard/pages/03_Evaluation.py
==================================
APEX LKA — Evaluation & Metrics Page

Reads real evaluation results from results/metrics/.
If no evaluation has been run, displays that clearly.
Never fabricates IoU, Dice, or any other metric.
"""

import sys
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

st.set_page_config(
    page_title="APEX LKA — Evaluation",
    page_icon="📈",
    layout="wide",
)

st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#070d15; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0a1220; border-right:1px solid #1a2d42; }
.section-label {
    font-family:monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:0.18em; color:#3a5070; text-transform:uppercase;
    border-bottom:1px solid #1a2d42; padding-bottom:6px; margin-bottom:10px; margin-top:16px;
}
.not-available {
    background:#0d1829; border:1px solid #1a2d42; border-radius:6px;
    padding:40px; text-align:center; font-family:monospace; color:#3a5070;
    font-size:0.85rem; letter-spacing:0.12em;
}
.metric-block {
    background:#0c1524; border:1px solid #1a2d42; border-radius:6px;
    padding:16px; margin-bottom:8px; text-align:center;
}
.metric-val  { font-family:monospace; font-size:1.6rem; font-weight:700; color:#00b4cc; }
.metric-name { font-family:monospace; font-size:0.60rem; color:#3a5070;
               text-transform:uppercase; letter-spacing:0.15em; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📈 Evaluation Results")
st.caption("Reads actual metrics from `results/metrics/`. Only displays values that exist.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Check for evaluation script and results
# ═══════════════════════════════════════════════════════════════════════════

eval_script  = PROJECT_ROOT / "src" / "training" / "evaluate.py"
metrics_dir  = PATHS.results / "metrics"
metrics_files = sorted(metrics_dir.glob("*.json")) if metrics_dir.exists() else []
plots_dir    = PATHS.results / "plots"
plot_files   = sorted(plots_dir.glob("*.png")) if plots_dir.exists() else []

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Evaluate Script", "FOUND" if eval_script.exists() else "NOT IMPLEMENTED")
with col2:
    st.metric("Metrics Files", str(len(metrics_files)))
with col3:
    st.metric("Result Plots", str(len(plot_files)))

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Model status
# ═══════════════════════════════════════════════════════════════════════════

from src.inference.predictor import MLSegmentationPredictor, ModelStatus

ml = MLSegmentationPredictor()
ms = ml.model_status

if ms == ModelStatus.NOT_TRAINED:
    st.markdown(
        '<div class="not-available">'
        'NO TRAINED MODEL — EVALUATION NOT AVAILABLE<br><br>'
        '<span style="color:#2a3d52; font-size:0.70rem;">'
        'Train the model first:<br>'
        'Annotate → Split → python run.py train → python run.py evaluate'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

elif not eval_script.exists():
    st.markdown(
        '<div class="not-available">'
        'EVALUATION MODULE NOT CONNECTED<br><br>'
        '<span style="color:#2a3d52; font-size:0.70rem;">'
        'src/training/evaluate.py has not been implemented yet.<br>'
        'Implement it and run: python run.py evaluate'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════
# Load and display real metrics
# ═══════════════════════════════════════════════════════════════════════════

all_metrics = {}
for mf in metrics_files:
    try:
        with open(mf) as f:
            data = json.load(f)
        all_metrics[mf.stem] = data
    except Exception as e:
        st.warning(f"Could not read {mf.name}: {e}")

if not all_metrics:
    st.info("No metrics files found in `results/metrics/`. Run `python run.py evaluate` first.")
    st.stop()

# ── Select metrics file ────────────────────────────────────────────────────
selected_key = st.selectbox("Results file", list(all_metrics.keys()))
metrics = all_metrics[selected_key]

st.markdown('<div class="section-label">Segmentation Metrics</div>',
            unsafe_allow_html=True)

# Supported metric keys and their display names
METRIC_DISPLAY = {
    "iou":              ("IoU",              "Intersection over Union"),
    "mean_iou":         ("mIoU",             "Mean IoU across classes"),
    "dice":             ("Dice",             "Dice / F1 coefficient"),
    "precision":        ("Precision",        "Pixel-level precision"),
    "recall":           ("Recall",           "Pixel-level recall"),
    "f1":               ("F1",               "F1 score"),
    "accuracy":         ("Accuracy",         "Pixel accuracy"),
    "left_iou":         ("Left IoU",         "Left lane IoU"),
    "right_iou":        ("Right IoU",        "Right lane IoU"),
    "background_iou":   ("BG IoU",           "Background IoU"),
    "map":              ("mAP",              "Mean Average Precision"),
    "val_loss":         ("Val Loss",         "Validation loss"),
    "test_loss":        ("Test Loss",        "Test set loss"),
}

found_keys = [k for k in METRIC_DISPLAY if k in metrics]

if not found_keys:
    st.warning("Metrics file found but contains no recognised metric keys.")
    st.json(metrics)
else:
    # Display as metric blocks
    cols = st.columns(min(4, len(found_keys)))
    for i, key in enumerate(found_keys):
        val = metrics[key]
        name, desc = METRIC_DISPLAY[key]
        val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
        with cols[i % 4]:
            st.markdown(
                f'<div class="metric-block">'
                f'<div class="metric-val">{val_str}</div>'
                f'<div class="metric-name">{name}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(desc)

# ── Per-class breakdown ────────────────────────────────────────────────────
if "per_class" in metrics:
    st.markdown('<div class="section-label">Per-Class Breakdown</div>',
                unsafe_allow_html=True)
    import pandas as pd
    pc = metrics["per_class"]
    class_names = {0: "Background", 1: "Left Lane", 2: "Right Lane"}
    rows = []
    for cls_idx, class_metrics in pc.items():
        rows.append({
            "Class":     class_names.get(int(cls_idx), f"Class {cls_idx}"),
            **class_metrics,
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ── Result plots ───────────────────────────────────────────────────────────
if plot_files:
    st.markdown('<div class="section-label">Result Plots</div>',
                unsafe_allow_html=True)
    import cv2
    pcols = st.columns(min(3, len(plot_files)))
    for i, pf in enumerate(plot_files[:6]):
        with pcols[i % 3]:
            img = cv2.imread(str(pf))
            if img is not None:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                         caption=pf.stem, use_container_width=True)

# ── Full JSON dump ─────────────────────────────────────────────────────────
with st.expander("Raw metrics JSON"):
    st.json(metrics)
