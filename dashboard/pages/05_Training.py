"""
dashboard/pages/02_Training.py
================================
APEX LKA — Training Status Page

Checks for the training module and reads real training logs if available.
If training has not yet been implemented, shows TRAINING MODULE NOT CONNECTED.
Never fabricates training curves or accuracy numbers.
"""

import sys
import json
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS, cfg

st.set_page_config(
    page_title="APEX LKA — Training",
    page_icon="🔬",
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
.not-connected {
    background:#0d1829; border:1px solid #1a2d42; border-radius:6px;
    padding:40px; text-align:center; font-family:monospace; color:#3a5070;
    font-size:0.85rem; letter-spacing:0.12em;
}
</style>
""", unsafe_allow_html=True)

st.markdown("## 🔬 Training Status")
st.caption("Reads actual training logs from `logs/training/` and `results/metrics/`. No fabrication.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Check if training module exists
# ═══════════════════════════════════════════════════════════════════════════

train_script  = PROJECT_ROOT / "src" / "training" / "train.py"
model_path    = PROJECT_ROOT / "models" / "exported" / "best_model.pth"
checkpoints   = list((PATHS.checkpoints).glob("*.pth")) if PATHS.checkpoints.exists() else []
training_logs = list((PATHS.logs / "training").glob("*.json")) if (PATHS.logs / "training").exists() else []
metrics_files = list((PATHS.results / "metrics").glob("*.json")) if (PATHS.results / "metrics").exists() else []

# ── Status indicators ──────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)
with col1:
    exists = train_script.exists()
    st.metric("Training Script", "FOUND" if exists else "NOT IMPLEMENTED",
              delta=None)
with col2:
    exists_m = model_path.exists()
    st.metric("Trained Model", "FOUND" if exists_m else "NOT FOUND",
              delta=None)
with col3:
    st.metric("Checkpoints", str(len(checkpoints)))

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Training module check
# ═══════════════════════════════════════════════════════════════════════════

if not train_script.exists():
    st.markdown(
        '<div class="not-connected">'
        'TRAINING MODULE NOT CONNECTED<br><br>'
        '<span style="color:#2a3d52; font-size:0.70rem;">'
        'src/training/train.py has not been implemented yet.<br>'
        'Complete dataset annotation → splitting → then implement training.'
        '</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-label">Next Steps to Enable Training</div>',
                unsafe_allow_html=True)

    from src.config import PATHS
    IMG_EXT = {".jpg", ".jpeg", ".png"}
    n_ann   = sum(1 for f in (PATHS.annotated / "images").iterdir()
                  if f.suffix.lower() in IMG_EXT) if (PATHS.annotated / "images").exists() else 0
    n_train = sum(1 for f in (PATHS.train / "images").iterdir()
                  if f.suffix.lower() in IMG_EXT) if (PATHS.train / "images").exists() else 0

    st.markdown(f"""
| Step | Command | Status |
|---|---|---|
| 1. Extract frames | `python run.py collect --fps 5` | {'✅' if n_ann > 0 else '⏳ pending'} |
| 2. Annotate frames | CVAT / LabelMe → `data/annotated/` | {'✅' if n_ann > 0 else '⏳ pending'} |
| 3. Split dataset | `python run.py split` | {'✅' if n_train > 0 else '⏳ pending'} |
| 4. Implement training | `src/training/train.py` | ⏳ not yet built |
| 5. Run training | `python run.py train` | ⏳ pending |
""")

else:
    # ── Training script exists ─────────────────────────────────────────────
    st.success("Training script found at `src/training/train.py`")

    # ── Model configuration ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model Configuration</div>',
                unsafe_allow_html=True)
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    col_m, col_t = st.columns(2)
    with col_m:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Architecture | {model_cfg.get('architecture','N/A')} |
| Backbone | {model_cfg.get('backbone','N/A')} |
| Num Classes | {model_cfg.get('num_classes','N/A')} |
| Pretrained backbone | {model_cfg.get('pretrained_backbone','N/A')} |
""")
    with col_t:
        st.markdown(f"""
| Parameter | Value |
|---|---|
| Epochs | {train_cfg.get('epochs','N/A')} |
| Batch size | {train_cfg.get('batch_size','N/A')} |
| Learning rate | {train_cfg.get('learning_rate','N/A')} |
| Optimizer | {train_cfg.get('optimizer','N/A')} |
| Loss | {train_cfg.get('loss','N/A')} |
| Early stopping patience | {train_cfg.get('early_stopping_patience','N/A')} |
""")

    # ── Training logs ──────────────────────────────────────────────────────
    if training_logs:
        st.markdown('<div class="section-label">Training Logs</div>',
                    unsafe_allow_html=True)
        log_data = []
        for lf in sorted(training_logs)[-1:]:   # latest log
            try:
                with open(lf) as f:
                    log_data = json.load(f)
            except Exception:
                pass

        if log_data and isinstance(log_data, list):
            import pandas as pd
            df = pd.DataFrame(log_data)
            st.dataframe(df, use_container_width=True)

            # Chart loss/iou over epochs
            if "train_loss" in df.columns and "val_loss" in df.columns:
                try:
                    import altair as alt
                    melt = df.melt(id_vars=["epoch"], value_vars=["train_loss", "val_loss"],
                                   var_name="split", value_name="loss")
                    chart = (
                        alt.Chart(melt)
                        .mark_line()
                        .encode(
                            x="epoch:Q",
                            y="loss:Q",
                            color=alt.Color("split:N"),
                        )
                        .properties(height=200, background="#0c1524")
                        .configure_view(strokeWidth=0)
                    )
                    st.altair_chart(chart, use_container_width=True)
                except Exception:
                    pass
    else:
        st.info("No training logs found in `logs/training/`. Run `python run.py train` to generate them.")

    # ── Checkpoints ───────────────────────────────────────────────────────
    if checkpoints:
        st.markdown('<div class="section-label">Checkpoints</div>',
                    unsafe_allow_html=True)
        for ckpt in sorted(checkpoints):
            size_mb = ckpt.stat().st_size / 1_000_000
            st.markdown(f"`{ckpt.name}` — {size_mb:.1f} MB")
    else:
        st.markdown('<div class="section-label">Checkpoints</div>',
                    unsafe_allow_html=True)
        st.info("No checkpoints found. Train the model first.")

    # ── Trained model ─────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Exported Model</div>',
                unsafe_allow_html=True)
    if model_path.exists():
        size_mb = model_path.stat().st_size / 1_000_000
        st.success(f"`{model_path.name}` — {size_mb:.1f} MB  ✅ Ready for inference")
    else:
        st.info(f"No exported model at `{model_path}`. Training will save the best model here automatically.")
