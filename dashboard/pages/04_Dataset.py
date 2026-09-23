"""
dashboard/pages/01_Dataset.py
==============================
APEX LKA — Dataset Status Page

Reads the actual dataset state from existing data/ directories
and the existing data_collection modules. Does NOT modify any
pipeline or dataset logic.
"""

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg

st.set_page_config(
    page_title="APEX LKA — Dataset",
    page_icon="📊",
    layout="wide",
)

# Minimal CSS reuse
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#070d15; color:#e2e8f0; }
[data-testid="stSidebar"] { background:#0a1220; border-right:1px solid #1a2d42; }
.section-label {
    font-family:monospace; font-size:0.65rem; font-weight:700;
    letter-spacing:0.18em; color:#3a5070; text-transform:uppercase;
    border-bottom:1px solid #1a2d42; padding-bottom:6px; margin-bottom:10px; margin-top:16px;
}
.ds-row {
    display:flex; justify-content:space-between;
    padding:6px 0; border-bottom:1px solid #0d1829;
    font-family:monospace; font-size:0.72rem;
}
.ds-label { color:#3a5070; }
.ds-val   { color:#e2e8f0; font-weight:600; }
.ds-val.ok { color:#10b981; }
.ds-val.empty { color:#3a5070; }
</style>
""", unsafe_allow_html=True)

st.markdown("## 📊 Dataset Status")
st.caption("Reads the current state of data/ directories. No modification to any pipeline.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Directory counts
# ═══════════════════════════════════════════════════════════════════════════

IMG_EXT  = {".jpg", ".jpeg", ".png", ".bmp"}
VID_EXT  = {".mp4", ".avi", ".mov", ".mkv"}
MASK_EXT = {".png"}


def count_files(directory: Path, exts: set) -> int:
    if not directory.exists():
        return 0
    return sum(1 for f in directory.iterdir() if f.suffix.lower() in exts)


col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="section-label">Raw Data</div>', unsafe_allow_html=True)

    n_videos = count_files(PATHS.raw_videos, VID_EXT)
    n_frames = count_files(PATHS.raw_frames, IMG_EXT)

    def _row(label, val, empty_threshold=0):
        cls = "ok" if val > empty_threshold else "empty"
        return f'<div class="ds-row"><span class="ds-label">{label}</span><span class="ds-val {cls}">{val}</span></div>'

    html = _row("Raw Videos", n_videos) + _row("Raw Frames", n_frames)

    # metadata.csv
    meta = PATHS.raw_frames / "metadata.csv"
    meta_str = f"{sum(1 for _ in open(meta)) - 1} rows" if meta.exists() else "not found"
    meta_cls = "ok" if meta.exists() else "empty"
    html += f'<div class="ds-row"><span class="ds-label">Metadata CSV</span><span class="ds-val {meta_cls}">{meta_str}</span></div>'

    st.markdown(html, unsafe_allow_html=True)


with col2:
    st.markdown('<div class="section-label">Annotated Data</div>', unsafe_allow_html=True)

    ann_img  = count_files(PATHS.annotated / "images", IMG_EXT)
    ann_mask = count_files(PATHS.annotated / "masks",  MASK_EXT)
    coverage = f"{ann_mask}/{ann_img}" if ann_img > 0 else "0/0"
    pct      = f"{ann_mask/ann_img*100:.1f}%" if ann_img > 0 else "N/A"

    html = (
        _row("Annotated Images", ann_img) +
        _row("Annotation Masks", ann_mask) +
        f'<div class="ds-row"><span class="ds-label">Coverage</span>'
        f'<span class="ds-val {"ok" if ann_mask>0 else "empty"}">{coverage} ({pct})</span></div>'
    )
    st.markdown(html, unsafe_allow_html=True)

with col3:
    st.markdown('<div class="section-label">Dataset Splits</div>', unsafe_allow_html=True)

    n_train = count_files(PATHS.train / "images", IMG_EXT)
    n_val   = count_files(PATHS.val   / "images", IMG_EXT)
    n_test  = count_files(PATHS.test  / "images", IMG_EXT)
    n_total = n_train + n_val + n_test

    html = (
        _row("Train",  n_train) +
        _row("Val",    n_val)   +
        _row("Test",   n_test)  +
        _row("Total",  n_total)
    )
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Config summary
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-label">Dataset Configuration (from project_config.yaml)</div>',
            unsafe_allow_html=True)

col_a, col_b = st.columns(2)
with col_a:
    dc = cfg.get("dataset", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Train ratio | {dc.get('train_ratio', 'N/A')} |
| Val ratio   | {dc.get('val_ratio',   'N/A')} |
| Test ratio  | {dc.get('test_ratio',  'N/A')} |
| Random seed | {dc.get('random_seed', 'N/A')} |
""")

with col_b:
    pp = cfg.get("preprocessing", {})
    dc2 = cfg.get("data_collection", {})
    st.markdown(f"""
| Parameter | Value |
|---|---|
| Model input size | {pp.get('image_width','?')} × {pp.get('image_height','?')} px |
| Target FPS (extract) | {dc2.get('target_fps', 'N/A')} |
| Dup threshold | {dc2.get('duplicate_threshold', 'N/A')} |
| Output format | {dc2.get('output_format', 'N/A')} |
""")


# ═══════════════════════════════════════════════════════════════════════════
# Test images
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
st.markdown('<div class="section-label">Test Images (data/test_images/)</div>',
            unsafe_allow_html=True)

test_dir  = PROJECT_ROOT / "data" / "test_images"
test_imgs = sorted(f for f in test_dir.iterdir()
                   if f.suffix.lower() in IMG_EXT) if test_dir.exists() else []

if test_imgs:
    st.caption(f"{len(test_imgs)} image(s) available for inference testing.")
    cols = st.columns(min(4, len(test_imgs)))
    import cv2
    for i, img_path in enumerate(test_imgs[:8]):
        with cols[i % 4]:
            img = cv2.imread(str(img_path))
            if img is not None:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                st.image(img_rgb, caption=img_path.name, use_container_width=True)
else:
    st.info("No test images found. Place `.jpg` or `.png` files in `data/test_images/`")


# ═══════════════════════════════════════════════════════════════════════════
# Next steps
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("---")
if n_frames == 0:
    st.warning("**No frames extracted yet.** Record road video → `python run.py collect --fps 5`")
elif ann_img == 0:
    st.warning("**No annotations yet.** Annotate frames in CVAT/LabelMe → see `README.md`")
elif n_train == 0:
    st.warning("**Dataset not split.** Run `python run.py split`")
else:
    st.success(f"Dataset ready: {n_train} train / {n_val} val / {n_test} test images.")
