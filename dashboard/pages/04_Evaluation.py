"""
dashboard/pages/04_Evaluation.py
==================================
APEX-LKA Phase 3 — Comprehensive Evaluation & Model Deployment Hub.

Tabs:
1. 🗃️ MODEL SELECTION: Checkpoint browser, metadata inspector, multi-model comparison.
2. 📊 METRICS OVERVIEW: Global KPI cards, per-class table, confusion matrix, performance radar.
3. 🖼️ PREDICTION GALLERY: 4-Column panels (Best / Worst / Median), interactive filters.
4. ⚠️ FAILURE ANALYSIS: Scenario bottleneck diagnostics, worst-frame thumbnails, recommendations.
5. 🚀 EXPORT & DEPLOY: One-click ONNX export (<20MB), Jetson TensorRT package, artifact downloader.
"""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg
from src.training.metrics import CLASS_NAMES
from src.training.model import LaneSegNet

log = logging.getLogger(__name__)

st.set_page_config(
    page_title="APEX-LKA — Model Evaluation",
    page_icon="📈",
    layout="wide",
)

# ═══════════════════════════════════════════════════════════════════════════════
# Design System Styling (APEX Dark / Cybernetic Cockpit)
# ═══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #070b12;
    color: #f1f5f9;
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: rgba(10, 18, 32, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}
.kpi-card {
    background: rgba(15, 23, 42, 0.85);
    border: 1px solid rgba(56, 189, 248, 0.20);
    border-radius: 8px;
    padding: 18px;
    text-align: center;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.4);
    margin-bottom: 12px;
}
.kpi-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.1rem;
    font-weight: 800;
    color: #38bdf8;
}
.kpi-lbl {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-top: 6px;
}
.badge-pass {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.35);
    padding: 3px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.75rem;
}
.badge-fail {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.35);
    padding: 3px 8px;
    border-radius: 4px;
    font-family: monospace;
    font-size: 0.75rem;
}
.rec-card {
    background: rgba(15, 23, 42, 0.7);
    border-left: 3px solid #38bdf8;
    border-radius: 4px;
    padding: 12px 16px;
    margin-bottom: 10px;
    color: #cbd5e1;
    font-size: 0.90rem;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers & State Management
# ═══════════════════════════════════════════════════════════════════════════════

def get_available_checkpoints() -> List[Dict[str, Any]]:
    """Scan models/exported and models/checkpoints for trained weights."""
    models: List[Dict[str, Any]] = []
    dirs_to_scan = [PATHS.exported, PATHS.checkpoints]

    for d in dirs_to_scan:
        if not d.exists():
            continue
        for p in sorted(list(d.glob("*.pth")) + list(d.glob("*.pt"))):
            size_mb = p.stat().st_size / (1024**2)
            mtime = time.strftime("%Y-%m-%d %H:%M", time.localtime(p.stat().st_mtime))

            # Try reading lightweight header
            meta = {}
            try:
                state = torch.load(p, map_location="cpu", weights_only=False)
                if isinstance(state, dict):
                    meta = state.get("metadata", {})
                    meta["epoch"] = state.get("epoch", "-")
                    meta["best_metric"] = state.get("best_metric", "-")
            except Exception:
                pass

            models.append({
                "path": p,
                "name": p.name,
                "folder": p.parent.name,
                "size_mb": round(size_mb, 2),
                "modified": mtime,
                "epoch": meta.get("epoch", "-"),
                "best_metric": meta.get("best_metric", "-"),
                "version": meta.get("version", "3.0.0"),
            })
    return models


def load_evaluation_report() -> Optional[Dict[str, Any]]:
    """Load latest results/metrics/evaluation_report.json."""
    rep_path = PATHS.results / "metrics" / "evaluation_report.json"
    if rep_path.exists():
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning("Could not read evaluation report: %s", e)
    return None


if "selected_checkpoint" not in st.session_state:
    ckpts = get_available_checkpoints()
    st.session_state.selected_checkpoint = str(ckpts[0]["path"]) if ckpts else None

# Header
st.markdown("## 📈 APEX-LKA Model Evaluation & Deployment Center")
st.caption("Comprehensive multi-class perception auditing, visual diagnostics, and edge deployment.")
st.markdown("---")

tab_models, tab_metrics, tab_gallery, tab_failures, tab_deploy = st.tabs([
    "🗃️ Model Selection",
    "📊 Metrics Overview",
    "🖼️ Prediction Gallery",
    "⚠️ Failure Analysis",
    "🚀 Export & Deploy",
])

eval_data = load_evaluation_report()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1: MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════════
with tab_models:
    st.markdown("### 🗃️ Available Checkpoints & Exported Weights")
    checkpoints = get_available_checkpoints()

    if not checkpoints:
        st.warning("No checkpoint weights found in `models/exported/` or `models/checkpoints/`.")
        st.info("Train a model first using `python run.py train` or the Training Page.")
    else:
        df_ckpts = pd.DataFrame(checkpoints)
        df_display = df_ckpts[["name", "folder", "size_mb", "epoch", "best_metric", "modified"]]
        df_display.columns = ["Checkpoint File", "Location", "Size (MB)", "Epoch", "Validation mIoU", "Last Modified"]

        st.dataframe(df_display, use_container_width=True)

        col_sel, col_btn = st.columns([3, 1])
        with col_sel:
            ckpt_options = [str(c["path"]) for c in checkpoints]
            sel_idx = 0
            if st.session_state.selected_checkpoint in ckpt_options:
                sel_idx = ckpt_options.index(st.session_state.selected_checkpoint)
            chosen = st.selectbox("Select Active Model for Inspection & Evaluation", ckpt_options, index=sel_idx)

        with col_btn:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            if st.button("Set Active Model", use_container_width=True):
                st.session_state.selected_checkpoint = chosen
                st.success(f"Active model set to: {Path(chosen).name}")

        # Multi-model comparison mode
        st.markdown("---")
        st.markdown("#### ⚖️ Multi-Model Comparison Mode")
        multi_sel = st.multiselect(
            "Select 2 or more checkpoints to compare metrics",
            ckpt_options,
            default=ckpt_options[:min(2, len(ckpt_options))],
        )
        if len(multi_sel) >= 2:
            comp_rows = []
            for m_path in multi_sel:
                c_info = next((c for c in checkpoints if str(c["path"]) == m_path), {})
                comp_rows.append({
                    "Model": Path(m_path).name,
                    "Location": c_info.get("folder", "-"),
                    "Size (MB)": c_info.get("size_mb", 0.0),
                    "Epoch": c_info.get("epoch", "-"),
                    "mIoU": c_info.get("best_metric", "-"),
                })
            st.table(pd.DataFrame(comp_rows))


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2: METRICS OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    st.markdown("### 📊 Test Set Perception Benchmarks")

    if not eval_data:
        st.info("No evaluation report found at `results/metrics/evaluation_report.json`.")
        st.write("Run evaluation now via CLI: `python run.py evaluate` or trigger below.")
        if st.button("Run Test Set Evaluation Now", type="primary"):
            with st.spinner("Executing ModelEvaluator on test split..."):
                from src.training.evaluate import ModelEvaluator
                evaluator = ModelEvaluator(checkpoint_path=st.session_state.selected_checkpoint, split="test")
                evaluator.evaluate(export_reports=True, generate_galleries=True)
                st.rerun()
    else:
        m = eval_data.get("metrics", {})
        lat = eval_data.get("latency", {})

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m.get("mean_iou", 0.0):.3f}</div><div class="kpi-lbl">Mean IoU (mIoU)</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m.get("lane_mean_iou", 0.0):.3f}</div><div class="kpi-lbl">Lane Class IoU</div></div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m.get("mean_dice", 0.0):.3f}</div><div class="kpi-lbl">Mean Dice (F1)</div></div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{m.get("overall_pixel_accuracy", 0.0):.1%}</div><div class="kpi-lbl">Pixel Accuracy</div></div>', unsafe_allow_html=True)
        with col5:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{lat.get("fps", 0.0):.1f}</div><div class="kpi-lbl">Throughput (FPS)</div></div>', unsafe_allow_html=True)

        st.markdown("---")
        col_tbl, col_plot = st.columns([1, 1])

        with col_tbl:
            st.markdown("#### 📋 Per-Class Detailed Breakdown")
            per_class = m.get("per_class", {})
            rows = []
            for c_name, vals in per_class.items():
                rows.append({
                    "Class": c_name,
                    "IoU": vals.get("iou", 0.0),
                    "Dice (F1)": vals.get("dice", 0.0),
                    "Precision": vals.get("precision", 0.0),
                    "Recall": vals.get("recall", 0.0),
                    "Accuracy": vals.get("pixel_accuracy", 0.0),
                    "Share": f"{vals.get('pixel_fraction', 0.0):.2%}",
                })
            df_metrics = pd.DataFrame(rows)
            st.dataframe(df_metrics, use_container_width=True)

            cm_path = PATHS.results / "plots" / "evaluation" / "confusion_matrix.png"
            if cm_path.exists():
                st.markdown("#### 🧩 Confusion Matrix Heatmap")
                st.image(str(cm_path), use_container_width=True)

        with col_plot:
            iou_bar_path = PATHS.results / "plots" / "evaluation" / "per_class_iou.png"
            if iou_bar_path.exists():
                st.markdown("#### 📊 Perception Performance Chart")
                st.image(str(iou_bar_path), use_container_width=True)

            st.markdown("#### ⏱️ Latency & Efficiency Profile")
            st.json({
                "Mean Latency": f"{lat.get('mean_ms', 0.0)} ms",
                "p50 (Median)": f"{lat.get('p50_ms', 0.0)} ms",
                "p95": f"{lat.get('p95_ms', 0.0)} ms",
                "p99": f"{lat.get('p99_ms', 0.0)} ms",
                "Target FPS": f"{lat.get('fps', 0.0)} FPS",
            })


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3: PREDICTION GALLERY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_gallery:
    st.markdown("### 🖼️ Visual Prediction Gallery (4-Column Panels)")
    st.caption("Inspect side-by-side: `[ Original Image | Ground Truth | Prediction | Error Map ]`")

    pred_dir = PATHS.results / "plots" / "evaluation" / "predictions"
    if not pred_dir.exists():
        st.info("Prediction galleries have not been generated yet. Run evaluation with galleries enabled.")
    else:
        col_grp, col_count = st.columns([2, 1])
        with col_grp:
            group_sel = st.radio("Select Prediction Cohort", ["Best (Top 20)", "Median (Typical 20)", "Worst (Failures 20)"], horizontal=True)
        with col_count:
            max_disp = st.slider("Display Limit", min_value=1, max_value=20, value=6)

        grp_sub = "best" if "Best" in group_sel else "worst" if "Worst" in group_sel else "median"
        target_dir = pred_dir / grp_sub
        images = sorted(target_dir.glob("*.jpg")) if target_dir.exists() else []

        if not images:
            st.warning(f"No images found in `{grp_sub}` cohort directory.")
        else:
            for p in images[:max_disp]:
                st.image(str(p), caption=p.name, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4: FAILURE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
with tab_failures:
    st.markdown("### ⚠️ Perception Failure Diagnostics & Scenario Analysis")

    if not eval_data or "failure_analysis" not in eval_data:
        st.info("No failure analysis data found. Run `python run.py evaluate` to generate diagnostics.")
    else:
        fa = eval_data["failure_analysis"]
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.metric("Severe Failures (mIoU < 0.30)", f"{fa.get('severe_failures_count', 0)} frames")
        with col_c2:
            st.metric("Lane Drops (Lane IoU < 0.20)", f"{fa.get('lane_failures_count', 0)} frames")

        fail_chart = PATHS.results / "plots" / "evaluation" / "failure_distribution.png"
        if fail_chart.exists():
            st.image(str(fail_chart), caption="Failure Distribution Across Environmental Scenarios", use_container_width=True)

        st.markdown("#### 💡 Actionable Engineering Recommendations")
        recs = fa.get("recommendations", [])
        if not recs:
            st.success("All perception parameters satisfy autonomous competition requirements.")
        else:
            for r in recs:
                st.markdown(f'<div class="rec-card">💡 {r}</div>', unsafe_allow_html=True)

        # Worst frame list
        worst_frames = fa.get("severe_failures", [])
        if worst_frames:
            st.markdown("#### 🔍 Critical Failure Frames")
            st.dataframe(pd.DataFrame(worst_frames), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPORT & DEPLOY
# ═══════════════════════════════════════════════════════════════════════════════
with tab_deploy:
    st.markdown("### 🚀 Production Model Export & Deployment")
    st.caption("Deploy LaneSegNet to RTX 4060 workstations and Jetson Orin Nano edge hardware.")

    col_onnx, col_trt = st.columns(2)

    with col_onnx:
        st.markdown("#### 📦 ONNX Engine Export (< 20 MB)")
        st.write("Converts PyTorch LaneSegNet weights to optimized Open Neural Network Exchange format.")

        if st.button("Export to ONNX (RTX 4060 / Jetson)", type="primary"):
            try:
                ckpt_to_use = st.session_state.selected_checkpoint
                model = LaneSegNet(num_classes=4, pretrained=False)
                if ckpt_to_use and Path(ckpt_to_use).exists():
                    sd = torch.load(ckpt_to_use, map_location="cpu", weights_only=False)
                    weights = sd.get("ema_model", sd.get("model_state_dict", sd))
                    model.load_state_dict(weights)
                model.eval()

                dummy_input = torch.randn(1, 3, 360, 640)
                onnx_path = PATHS.exported / "lanesegnet_rtx4060.onnx"
                onnx_path.parent.mkdir(parents=True, exist_ok=True)

                torch.onnx.export(
                    model,
                    dummy_input,
                    str(onnx_path),
                    input_names=["input"],
                    output_names=["output"],
                    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
                    opset_version=14,
                )
                file_mb = onnx_path.stat().st_size / (1024**2)
                st.success(f"Successfully exported ONNX model to `{onnx_path}` ({file_mb:.2f} MB)")
                if file_mb < 20.0:
                    st.markdown('<span class="badge-pass">✅ MEETS JETSON <20MB SPECIFICATION</span>', unsafe_allow_html=True)
                else:
                    st.markdown('<span class="badge-fail">⚠️ EXCEEDS 20MB SPECIFICATION</span>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"ONNX export failed: {e}")

    with col_trt:
        st.markdown("#### ⚡ Jetson Orin Nano TensorRT Guide")
        st.write("Commands to compile ONNX to INT8/FP16 TensorRT engine on the Jetson Orin Nano:")
        st.code("""
# Run directly on Jetson Orin Nano
trtexec --onnx=lanesegnet_rtx4060.onnx \\
        --saveEngine=lanesegnet_fp16.engine \\
        --fp16 \\
        --workspace=2048
        """, language="bash")

    st.markdown("---")
    st.markdown("#### 📥 Download Deployment Assets")

    best_pth = PATHS.exported / "best_model.pth"
    if best_pth.exists():
        with open(best_pth, "rb") as f:
            st.download_button(
                label="⬇️ Download best_model.pth",
                data=f,
                file_name="best_model.pth",
                mime="application/octet-stream",
            )
    else:
        st.info("best_model.pth not yet exported to models/exported/.")

    # Generate full deployment zip package
    if st.button("Generate Standalone Deployment ZIP Package"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            # Include model, predictor, config
            if best_pth.exists():
                zip_file.write(best_pth, arcname="weights/best_model.pth")
            pred_file = PROJECT_ROOT / "src" / "inference" / "predictor.py"
            if pred_file.exists():
                zip_file.write(pred_file, arcname="src/predictor.py")
            model_file = PROJECT_ROOT / "src" / "training" / "model.py"
            if model_file.exists():
                zip_file.write(model_file, arcname="src/model.py")
            cfg_file = PROJECT_ROOT / "project_config.yaml"
            if cfg_file.exists():
                zip_file.write(cfg_file, arcname="config.yaml")

        st.download_button(
            label="⬇️ Download Standalone APEX-LKA Deployment ZIP",
            data=zip_buffer.getvalue(),
            file_name="apex_lka_deployment_package.zip",
            mime="application/zip",
        )
