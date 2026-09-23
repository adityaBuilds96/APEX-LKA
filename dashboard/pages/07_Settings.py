"""
dashboard/pages/04_Settings.py
================================
APEX LKA — Settings Page

Exposes configuration values that the backend can actually use.
Reads from configs/project_config.yaml.
Any changes made here are informational — they do not hot-reload
the config (yaml reloading on disk would require a full Streamlit restart).
"""

import sys
from pathlib import Path

import streamlit as st
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS, cfg

CONFIG_PATH = PROJECT_ROOT / "configs" / "project_config.yaml"

st.set_page_config(
    page_title="APEX LKA — Settings",
    page_icon="⚙️",
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
</style>
""", unsafe_allow_html=True)

st.markdown("## ⚙️ Settings")
st.caption(
    "Configuration values loaded from `configs/project_config.yaml`.  \n"
    "To change settings, edit the YAML file and restart the dashboard."
)
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Current effective configuration (read from loaded cfg)
# ═══════════════════════════════════════════════════════════════════════════

col_inf, col_pp = st.columns(2)

with col_inf:
    st.markdown('<div class="section-label">Inference Settings</div>',
                unsafe_allow_html=True)
    inf_cfg = cfg.get("inference", {})
    st.markdown(f"""
| Setting | Current Value |
|---|---|
| Default backend | `{inf_cfg.get('default_backend', 'N/A')}` |
| Confidence threshold | `{inf_cfg.get('confidence_threshold', 'N/A')}` |
| Max frames (video) | `{inf_cfg.get('max_video_frames', 'N/A')}` |
| Show original image | `{inf_cfg.get('show_original_image', 'N/A')}` |
""")

with col_pp:
    st.markdown('<div class="section-label">Preprocessing</div>',
                unsafe_allow_html=True)
    pp_cfg = cfg.get("preprocessing", {})
    st.markdown(f"""
| Setting | Current Value |
|---|---|
| Model input width | `{pp_cfg.get('image_width', 'N/A')} px` |
| Model input height | `{pp_cfg.get('image_height', 'N/A')} px` |
| Normalize | `{pp_cfg.get('normalize', 'N/A')}` |
| Mean | `{pp_cfg.get('mean', 'N/A')}` |
| Std | `{pp_cfg.get('std', 'N/A')}` |
""")

st.markdown("---")

col_sc, col_lg = st.columns(2)

with col_sc:
    st.markdown('<div class="section-label">Steering Controller (PD)</div>',
                unsafe_allow_html=True)
    sc_cfg = cfg.get("steering", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Kp (proportional gain) | `{sc_cfg.get('Kp', 'N/A')}` |
| Kd (derivative gain) | `{sc_cfg.get('Kd', 'N/A')}` |
| Keep-center threshold | `{sc_cfg.get('keep_center_threshold', 'N/A')}` |
| Low-confidence IoU | `{sc_cfg.get('low_confidence_iou', 'N/A')}` |
| Max steering command | `{sc_cfg.get('max_steering_command', 'N/A')}` |
""")
    st.caption("⚠ Steering command is **software only**. No physical actuation.")

with col_lg:
    st.markdown('<div class="section-label">Lane Geometry</div>',
                unsafe_allow_html=True)
    geo_cfg = cfg.get("lane_geometry", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Polynomial degree | `{geo_cfg.get('poly_degree', 'N/A')}` |
| Min lane pixels | `{geo_cfg.get('min_lane_pixels', 'N/A')}` |
| ROI top fraction | `{geo_cfg.get('roi_top_fraction', 'N/A')}` |
""")

st.markdown("---")

col_m, col_t = st.columns(2)

with col_m:
    st.markdown('<div class="section-label">Model Architecture</div>',
                unsafe_allow_html=True)
    m_cfg = cfg.get("model", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Architecture | `{m_cfg.get('architecture', 'N/A')}` |
| Backbone | `{m_cfg.get('backbone', 'N/A')}` |
| Num classes | `{m_cfg.get('num_classes', 'N/A')}` |
| Pretrained backbone | `{m_cfg.get('pretrained_backbone', 'N/A')}` |
| Decoder channels | `{m_cfg.get('decoder_channels', 'N/A')}` |
""")

with col_t:
    st.markdown('<div class="section-label">Training</div>',
                unsafe_allow_html=True)
    t_cfg = cfg.get("training", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Epochs | `{t_cfg.get('epochs', 'N/A')}` |
| Batch size | `{t_cfg.get('batch_size', 'N/A')}` |
| Learning rate | `{t_cfg.get('learning_rate', 'N/A')}` |
| Optimizer | `{t_cfg.get('optimizer', 'N/A')}` |
| Loss | `{t_cfg.get('loss', 'N/A')}` |
| Early stopping | `{t_cfg.get('early_stopping_patience', 'N/A')} epochs` |
| Scheduler | `{t_cfg.get('lr_scheduler', 'N/A')}` |
""")

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Directory paths
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="section-label">Project Paths</div>',
            unsafe_allow_html=True)

path_rows = {
    "Raw Videos":       PATHS.raw_videos,
    "Raw Frames":       PATHS.raw_frames,
    "Annotated Images": PATHS.annotated / "images",
    "Annotated Masks":  PATHS.annotated / "masks",
    "Train":            PATHS.train / "images",
    "Val":              PATHS.val   / "images",
    "Test":             PATHS.test  / "images",
    "Model Checkpoints": PATHS.checkpoints,
    "Exported Model":   PROJECT_ROOT / "models" / "exported",
    "Training Logs":    PATHS.logs / "training",
    "Results Metrics":  PATHS.results / "metrics",
}

rows_md = "| Directory | Path | Exists |\n|---|---|---|\n"
for name, p in path_rows.items():
    exists = "✅" if p.exists() else "❌"
    rows_md += f"| {name} | `{p.relative_to(PROJECT_ROOT)}` | {exists} |\n"

st.markdown(rows_md)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Full config dump
# ═══════════════════════════════════════════════════════════════════════════

with st.expander("Full project_config.yaml"):
    if CONFIG_PATH.exists():
        st.code(CONFIG_PATH.read_text(), language="yaml")
    else:
        st.error(f"Config file not found: {CONFIG_PATH}")
