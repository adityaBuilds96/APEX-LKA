"""
dashboard/pages/03_Training.py
===============================
APEX LKA — Training Center & Neural Engine Monitor.

Displays real-time model training metrics, validation mIoU curves,
checkpoint archives, exported EMA weights, and allows launching training runs.
"""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg

st.set_page_config(
    page_title="APEX LKA — Training Center",
    page_icon="🔬",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

[data-testid="stAppViewContainer"] { background:#070d15; color:#e2e8f0; font-family:'Inter', sans-serif; }
[data-testid="stSidebar"] { background:#0a1220; border-right:1px solid #1a2d42; }

.section-label {
    font-family:'JetBrains Mono', monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:0.18em; color:#38bdf8; text-transform:uppercase;
    border-bottom:1px solid #1a2d42; padding-bottom:6px; margin-bottom:12px; margin-top:20px;
}
.stat-card {
    background:#0d1829; border:1px solid #1e293b; border-radius:8px;
    padding:14px 18px; margin-bottom:10px;
}
.stat-val {
    font-family:'JetBrains Mono', monospace; font-size:1.4rem; font-weight:700;
    color:#38bdf8; margin-top:4px;
}
.stat-lbl {
    font-size:0.65rem; color:#64748b; letter-spacing:0.08em; text-transform:uppercase;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Neural Training Center (LaneSegNet)")
st.caption("PyTorch AMP, Cosine Annealing with Warmup, Model EMA, and Automated Export.")
st.markdown("---")

train_script = PROJECT_ROOT / "src" / "training" / "train.py"
model_path = PATHS.exported / "best_model.pth"
checkpoints = list(PATHS.checkpoints.glob("*.pt")) if PATHS.checkpoints.exists() else []
csv_log = PATHS.logs / "training" / "training_metrics.csv"

# ── Top KPIs ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-lbl">Engine Architecture</div>
          <div class="stat-val" style="font-size:1.1rem;">LaneSegNet</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c2:
    status_str = "READY" if model_path.exists() else "NOT TRAINED"
    color = "#10b981" if model_path.exists() else "#ef4444"
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-lbl">Exported Model</div>
          <div class="stat-val" style="color:{color};">{status_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-lbl">Saved Checkpoints</div>
          <div class="stat-val">{len(checkpoints)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with c4:
    best_miou = "N/A"
    if csv_log.exists():
        try:
            df_log = pd.read_csv(csv_log)
            if "val_mean_iou" in df_log.columns and not df_log.empty:
                best_miou = f"{df_log['val_mean_iou'].max():.4f}"
        except Exception:
            pass
    st.markdown(
        f"""
        <div class="stat-card">
          <div class="stat-lbl">Peak Validation mIoU</div>
          <div class="stat-val" style="color:#10b981;">{best_miou}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Training Metrics & Curves ──────────────────────────────────────────────
st.markdown('<div class="section-label">Training Telemetry & Convergence Curves</div>', unsafe_allow_html=True)

if csv_log.exists():
    try:
        df_metrics = pd.read_csv(csv_log)
        if not df_metrics.empty:
            st.dataframe(df_metrics, use_container_width=True, height=200)

            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                st.caption("LOSS CONVERGENCE (TRAIN vs VAL)")
                loss_cols = [c for c in ["train_loss", "val_loss"] if c in df_metrics.columns]
                if loss_cols:
                    st.line_chart(df_metrics[loss_cols], height=240)

            with ch_col2:
                st.caption("VALIDATION mIoU")
                iou_cols = [c for c in ["val_mean_iou", "val_lane_mean_iou"] if c in df_metrics.columns]
                if iou_cols:
                    st.line_chart(df_metrics[iou_cols], height=240)
        else:
            st.info("Log file is empty. Run training to generate telemetry curves.")
    except Exception as e:
        st.error(f"Error loading training metrics: {e}")
else:
    st.info("No training log found at `logs/training/training_metrics.csv`. Launch training below or run `python run.py train`.")

# ── Checkpoint & Model Explorer ───────────────────────────────────────────
st.markdown('<div class="section-label">Checkpoint Management</div>', unsafe_allow_html=True)
c_ck1, c_ck2 = st.columns(2)

with c_ck1:
    st.markdown("##### Checkpoints Archive (`models/checkpoints/`)")
    if checkpoints:
        for ck in sorted(checkpoints, reverse=True)[:6]:
            sz_mb = ck.stat().st_size / (1024**2)
            st.caption(f"💾 **`{ck.name}`** &bull; {sz_mb:.1f} MB")
    else:
        st.caption("No checkpoints saved yet.")

with c_ck2:
    st.markdown("##### Exported Production Model (`models/exported/best_model.pth`)")
    if model_path.exists():
        sz_mb = model_path.stat().st_size / (1024**2)
        st.success(f"✅ **`best_model.pth`** ({sz_mb:.1f} MB) — Inference Ready")
        st.caption("Contains pure Exponential Moving Average (EMA) weights.")
    else:
        st.warning("No exported model found. Train model to export best weights automatically.")

# ── Interactive Training Launcher ─────────────────────────────────────────
st.markdown('<div class="section-label">Pipeline Control: Launch Training</div>', unsafe_allow_html=True)

with st.expander("⚙️ Launch APEX Training Process"):
    lp1, lp2, lp3 = st.columns(3)
    with lp1:
        cfg_epochs = st.number_input("Epochs", min_value=1, max_value=200, value=cfg.get("training", {}).get("epochs", 50))
    with lp2:
        cfg_bs = st.number_input("Batch Size", min_value=1, max_value=64, value=cfg.get("training", {}).get("batch_size", 8))
    with lp3:
        cfg_lr = st.number_input("Learning Rate", min_value=1e-5, max_value=1e-1, value=cfg.get("training", {}).get("learning_rate", 0.001), format="%.5f")

    resume_opt = st.checkbox("Resume from latest crash checkpoint if available", value=True)

    if st.button("🚀 INITIATE TRAINING (BACKGROUND PROCESS)", type="primary"):
        cmd = [sys.executable, str(PROJECT_ROOT / "run.py"), "train", "--epochs", str(cfg_epochs), "--batch-size", str(cfg_bs), "--lr", str(cfg_lr)]
        if resume_opt:
            crash_ckpt = PATHS.checkpoints / "crash_checkpoint.pt"
            if crash_ckpt.exists():
                cmd += ["--resume", str(crash_ckpt)]

        try:
            subprocess.Popen(cmd)
            st.success(f"Training job launched in background! Command: `{' '.join(cmd)}`")
            st.caption("Check terminal output or refresh this page to inspect live metrics.")
        except Exception as e:
            st.error(f"Failed to launch training: {e}")
