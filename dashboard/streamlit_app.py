"""
dashboard/streamlit_app.py
===========================
LKA Inference Dashboard — ADAS Prototype UI

Launch:
    streamlit run dashboard/streamlit_app.py

Features
--------
* Image upload → full lane detection pipeline
* Classical CV baseline (always available) or ML model (when trained)
* Side-by-side original + annotated image display
* Full metrics panel: detection status, confidence, lane positions,
  lateral offset, drift direction, steering recommendation
* Performance timing breakdown
* Test image directory browser
* Model status indicator
* Video upload support (frame-by-frame)
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ── Project root on path ───────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import run_pipeline, InferenceResult
from src.inference.predictor import (
    ModelStatus, DetectionStatus,
    get_predictor, MLSegmentationPredictor,
)
from src.lane_geometry.offset_calculator import (
    SteeringRecommendation, DriftDirection,
)


# ═══════════════════════════════════════════════════════════════════════════
# Page config — must be FIRST Streamlit call
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title    = "LKA Inference Dashboard",
    page_icon     = "🛣️",
    layout        = "wide",
    initial_sidebar_state = "expanded",
)

# ═══════════════════════════════════════════════════════════════════════════
# CSS — engineering dark theme
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #161b22;
    border-right: 1px solid #30363d;
}

/* ── Metric cards ── */
.metric-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 8px;
}
.metric-label  { font-size:0.72rem; color:#8b949e; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:4px; }
.metric-value  { font-size:1.3rem;  color:#e6edf3; font-weight:600; }
.metric-value.ok     { color:#3fb950; }
.metric-value.warn   { color:#d29922; }
.metric-value.error  { color:#f85149; }
.metric-value.info   { color:#58a6ff; }

/* ── Status badge ── */
.badge { display:inline-block; padding:3px 10px; border-radius:12px;
         font-size:0.75rem; font-weight:600; letter-spacing:0.05em; }
.badge-green  { background:#1a4d2e; color:#3fb950; border:1px solid #3fb950; }
.badge-yellow { background:#4a3600; color:#d29922; border:1px solid #d29922; }
.badge-red    { background:#4d1a1a; color:#f85149; border:1px solid #f85149; }
.badge-blue   { background:#1a2d4d; color:#58a6ff; border:1px solid #58a6ff; }
.badge-gray   { background:#2d333b; color:#8b949e; border:1px solid #8b949e; }

/* ── Section headers ── */
.section-header {
    font-size:0.8rem; font-weight:700; text-transform:uppercase;
    letter-spacing:0.12em; color:#58a6ff; margin:18px 0 10px;
    border-bottom: 1px solid #30363d; padding-bottom:4px;
}

/* ── Classical CV label ── */
.cv-baseline-label {
    background:#2d1a00; border:1px solid #b58900;
    border-radius:6px; padding:8px 12px; color:#d29922;
    font-size:0.78rem; font-weight:600; margin-bottom:12px;
}

/* ── Timing row ── */
.timing-row { display:flex; gap:8px; flex-wrap:wrap; }
.timing-pill {
    background:#21262d; border:1px solid #30363d; border-radius:20px;
    padding:4px 12px; font-size:0.72rem; color:#8b949e;
}
.timing-pill span { color:#e6edf3; font-weight:600; }

/* ── Image caption ── */
.img-caption {
    text-align:center; font-size:0.72rem; color:#8b949e;
    margin-top:4px; text-transform:uppercase; letter-spacing:0.08em;
}

/* ── Recommendation box ── */
.rec-box {
    border-radius:8px; padding:12px 18px; text-align:center;
    font-size:1.1rem; font-weight:700; letter-spacing:0.1em;
    margin:10px 0;
}
.rec-center { background:#1a4d2e; color:#3fb950; border:1px solid #3fb950; }
.rec-steer  { background:#4a3600; color:#d29922; border:1px solid #d29922; }
.rec-none   { background:#2d333b; color:#8b949e; border:1px solid #8b949e; }
.rec-low    { background:#4d1a1a; color:#f85149; border:1px solid #f85149; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _badge(text: str, kind: str) -> str:
    return f'<span class="badge badge-{kind}">{text}</span>'


def _metric(label: str, value: str, kind: str = "") -> str:
    vc = f' {kind}' if kind else ""
    return (
        f'<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value{vc}">{value}</div>'
        f'</div>'
    )


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _rec_class(rec: SteeringRecommendation) -> str:
    if rec == SteeringRecommendation.KEEP_CENTER:
        return "rec-center"
    if rec in (SteeringRecommendation.STEER_LEFT, SteeringRecommendation.STEER_RIGHT):
        return "rec-steer"
    if rec == SteeringRecommendation.LOW_CONFIDENCE:
        return "rec-low"
    return "rec-none"


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def render_sidebar() -> tuple[str, bool]:
    """Render sidebar and return (backend, show_original)."""
    with st.sidebar:
        st.markdown("## 🛣️ LKA Dashboard")
        st.markdown("---")

        # ── Model status ────────────────────────────────────────────────
        st.markdown('<div class="section-header">Model Status</div>', unsafe_allow_html=True)

        ml_predictor = MLSegmentationPredictor()
        ms = ml_predictor.model_status

        if ms == ModelStatus.NOT_TRAINED:
            st.markdown(_badge("MODEL NOT TRAINED", "gray"), unsafe_allow_html=True)
            st.caption("No trained model found at `models/exported/best_model.pth`.")
            st.caption("Complete annotation → training before using ML backend.")
        elif ms == ModelStatus.READY:
            st.markdown(_badge("MODEL READY", "green"), unsafe_allow_html=True)
        elif ms == ModelStatus.LOAD_ERROR:
            st.markdown(_badge("LOAD ERROR", "red"), unsafe_allow_html=True)
        else:
            st.markdown(_badge(ms.value, "gray"), unsafe_allow_html=True)

        st.markdown("---")

        # ── Backend selection ────────────────────────────────────────────
        st.markdown('<div class="section-header">Detection Backend</div>', unsafe_allow_html=True)
        ml_disabled = ms != ModelStatus.READY

        backend_choice = st.radio(
            "Backend",
            options  = ["Classical CV (OpenCV)", "ML Segmentation (LaneSegNet)"],
            index    = 0,
            help     = (
                "Classical CV: always available, uses colour thresholding + Hough lines.\n"
                "ML Segmentation: requires trained model file."
            ),
            disabled = False,
        )

        if backend_choice == "ML Segmentation (LaneSegNet)" and ml_disabled:
            st.warning("ML backend not available — model not trained. Using Classical CV.")
            backend = "classical_cv"
        elif backend_choice == "Classical CV (OpenCV)":
            backend = "classical_cv"
        else:
            backend = "ml_segmentation"

        if backend == "classical_cv":
            st.markdown(
                '<div class="cv-baseline-label">'
                '⚠️ CLASSICAL CV BASELINE<br>'
                'NOT the final ML model.<br>'
                'For pipeline verification only.'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Display options ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Display</div>', unsafe_allow_html=True)
        show_original = st.checkbox("Show original image side-by-side", value=True)

        st.markdown("---")

        # ── Test images ──────────────────────────────────────────────────
        st.markdown('<div class="section-header">Test Images</div>', unsafe_allow_html=True)
        test_dir = PROJECT_ROOT / "data" / "test_images"
        test_imgs = sorted(
            f for f in test_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if test_dir.exists() else []

        if test_imgs:
            st.caption(f"{len(test_imgs)} image(s) in data/test_images/")
        else:
            st.caption("No test images found.\nPlace images in `data/test_images/`")

        st.markdown("---")
        st.caption(f"Project: `{PROJECT_ROOT.name}`")

    return backend, show_original


# ═══════════════════════════════════════════════════════════════════════════
# Results renderer
# ═══════════════════════════════════════════════════════════════════════════

def render_results(result: InferenceResult, show_original: bool) -> None:
    """Render the full results panel."""

    # ── Images ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Visual Output</div>',
                unsafe_allow_html=True)

    if show_original and result.preprocessed and result.preprocessed.valid:
        col_orig, col_proc = st.columns(2)
        with col_orig:
            st.image(result.preprocessed.original_rgb,
                     use_container_width=True)
            st.markdown('<div class="img-caption">Original Image</div>',
                        unsafe_allow_html=True)
        with col_proc:
            if result.annotated_bgr is not None:
                st.image(bgr_to_rgb(result.annotated_bgr),
                         use_container_width=True)
                st.markdown('<div class="img-caption">Lane Detection Output</div>',
                            unsafe_allow_html=True)
    else:
        if result.annotated_bgr is not None:
            st.image(bgr_to_rgb(result.annotated_bgr), use_container_width=True)

    # ── Steering recommendation (prominent) ───────────────────────────────
    if result.offset:
        rec = result.offset.recommendation
        rc  = _rec_class(rec)
        arrow = {
            SteeringRecommendation.STEER_LEFT:  "← ",
            SteeringRecommendation.STEER_RIGHT: "→ ",
            SteeringRecommendation.KEEP_CENTER: "↑ ",
        }.get(rec, "")
        st.markdown(
            f'<div class="rec-box {rc}">{arrow}{rec.value}</div>',
            unsafe_allow_html=True,
        )

    # ── Metrics grid ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Detection Metrics</div>',
                unsafe_allow_html=True)

    pred = result.prediction
    geo  = result.geometry
    off  = result.offset

    col1, col2, col3 = st.columns(3)

    with col1:
        # Detection status
        if pred:
            ds = pred.status.value
            dk = (
                "ok"    if "LANE DETECTED" == ds else
                "warn"  if "PARTIAL"       in ds else
                "error"
            )
            st.markdown(_metric("Lane Status", ds, dk), unsafe_allow_html=True)
        # Left lane
        if pred:
            lv = "DETECTED" if pred.left_detected else "NOT DETECTED"
            lk = "ok" if pred.left_detected else "error"
            lc_str = f"{pred.left_confidence:.2f}" if pred.left_detected else "--"
            st.markdown(_metric("Left Lane", f"{lv}  [{lc_str}]", lk), unsafe_allow_html=True)
        # Right lane
        if pred:
            rv = "DETECTED" if pred.right_detected else "NOT DETECTED"
            rk = "ok" if pred.right_detected else "error"
            rc_str = f"{pred.right_confidence:.2f}" if pred.right_detected else "--"
            st.markdown(_metric("Right Lane", f"{rv}  [{rc_str}]", rk), unsafe_allow_html=True)

    with col2:
        if off:
            # Positions
            vc = f"{off.vehicle_center_x:.0f} px"
            lc = f"{off.lane_center_x:.0f} px" if off.lane_center_x else "N/A"
            st.markdown(_metric("Vehicle Center", vc),     unsafe_allow_html=True)
            st.markdown(_metric("Lane Center",    lc),     unsafe_allow_html=True)
            # Lateral offset
            if off.lateral_error_px is not None:
                err_px   = f"{off.lateral_error_px:+.1f} px"
                err_norm = f"({off.lateral_error_norm:+.3f})"
                err_k    = "ok" if abs(off.lateral_error_norm) < 0.05 else "warn"
                st.markdown(_metric("Lateral Offset", f"{err_px} {err_norm}", err_k),
                            unsafe_allow_html=True)
            else:
                st.markdown(_metric("Lateral Offset", "N/A", "error"), unsafe_allow_html=True)

    with col3:
        if off:
            # Drift
            dd = off.drift_direction.value
            dk = "ok" if off.drift_direction == DriftDirection.CENTERED else "warn"
            st.markdown(_metric("Drift Direction", dd, dk), unsafe_allow_html=True)
            # Confidence
            conf_str = f"{off.confidence:.2f}"
            ck = "ok" if off.confidence >= 0.7 else ("warn" if off.confidence >= 0.3 else "error")
            st.markdown(_metric("Confidence", conf_str, ck), unsafe_allow_html=True)
            # One-lane estimate warning
            if off.one_lane_estimated:
                st.markdown(_metric("Note", "One-lane estimate", "warn"), unsafe_allow_html=True)
        # Curvature
        if geo:
            cl = f"{geo.curvature_left_m:.0f} m" if geo.curvature_left_m else "Straight"
            cr = f"{geo.curvature_right_m:.0f} m" if geo.curvature_right_m else "Straight"
            st.markdown(_metric("Curvature L/R", f"{cl} / {cr}"), unsafe_allow_html=True)

    # ── Timing breakdown ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Performance</div>',
                unsafe_allow_html=True)
    t = result.timings
    fps = t.get("fps_equiv", 0)
    timing_html = '<div class="timing-row">'
    for label, key in [
        ("Preprocess",  "preprocess_ms"),
        ("Inference",   "inference_ms"),
        ("Postprocess", "postprocess_ms"),
        ("Geometry",    "geometry_ms"),
        ("Offset Calc", "offset_ms"),
        ("Viz",         "viz_ms"),
        ("TOTAL",       "total_ms"),
    ]:
        val = t.get(key, "--")
        val_str = f"{val} ms" if isinstance(val, (int, float)) else val
        timing_html += (
            f'<div class="timing-pill">{label}: <span>{val_str}</span></div>'
        )
    timing_html += (
        f'<div class="timing-pill">FPS equiv: <span>{fps}</span></div>'
        '</div>'
    )
    st.markdown(timing_html, unsafe_allow_html=True)

    # ── Error display ─────────────────────────────────────────────────────
    if result.error:
        st.markdown("---")
        st.error(f"**Pipeline message:** {result.error}")


# ═══════════════════════════════════════════════════════════════════════════
# Main app
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    backend, show_original = render_sidebar()

    # ── Page header ────────────────────────────────────────────────────────
    st.markdown("# 🛣️ Lane Keep Assist — Inference Dashboard")
    st.markdown(
        "_ADAS Prototype — Perception & Software Recommendation Only_  \n"
        "**No physical vehicle control. No hardware actuation.**"
    )
    st.markdown("---")

    # ── Input section ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Input</div>', unsafe_allow_html=True)

    tab_upload, tab_test, tab_video = st.tabs(
        ["📁 Upload Image", "🖼️ Test Images", "🎬 Video"]
    )

    result: InferenceResult | None = None

    # ── Tab 1: Upload ──────────────────────────────────────────────────────
    with tab_upload:
        uploaded = st.file_uploader(
            "Upload a road image",
            type   = ["jpg", "jpeg", "png", "bmp"],
            help   = "Upload a forward-facing road image to run lane detection.",
        )
        if uploaded:
            img_bytes = uploaded.read()
            col_btn, _ = st.columns([1, 3])
            with col_btn:
                run_btn = st.button("▶ Run Lane Detection", type="primary",
                                    use_container_width=True)
            if run_btn:
                with st.spinner("Running inference pipeline..."):
                    result = run_pipeline(img_bytes, backend=backend,
                                          reset_pd_state=True)

    # ── Tab 2: Test images ─────────────────────────────────────────────────
    with tab_test:
        test_dir = PROJECT_ROOT / "data" / "test_images"
        test_imgs = sorted(
            f for f in test_dir.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ) if test_dir.exists() else []

        if not test_imgs:
            st.info(
                "No test images found.  \n"
                f"Place `.jpg` or `.png` files in:  \n"
                f"`{test_dir}`"
            )
        else:
            img_names = [f.name for f in test_imgs]
            selected  = st.selectbox("Select test image", img_names)
            sel_path  = test_dir / selected

            col_preview, col_run = st.columns([2, 1])
            with col_preview:
                preview = cv2.imread(str(sel_path))
                if preview is not None:
                    st.image(bgr_to_rgb(preview), use_container_width=True,
                             caption=selected)
            with col_run:
                if st.button("▶ Run Detection", type="primary",
                             use_container_width=True):
                    with st.spinner("Running inference pipeline..."):
                        result = run_pipeline(sel_path, backend=backend,
                                              reset_pd_state=True)

    # ── Tab 3: Video ───────────────────────────────────────────────────────
    with tab_video:
        st.info(
            "**Video inference** processes the video frame-by-frame.  \n"
            "For live webcam inference, use:  \n"
            "`python src/inference/pipeline.py --webcam`  \n\n"
            "Upload a short video clip below to test."
        )
        vid_file = st.file_uploader(
            "Upload video",
            type = ["mp4", "avi", "mov", "mkv"],
        )
        max_frames = st.slider("Max frames to process", 5, 100, 20)

        if vid_file and st.button("▶ Process Video", type="primary"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(vid_file.name).suffix
            ) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            frame_results = []
            progress_bar  = st.progress(0)
            status_text   = st.empty()
            n_processed   = 0

            while cap.isOpened() and n_processed < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                r = run_pipeline(
                    frame, backend=backend,
                    reset_pd_state=(n_processed == 0),
                )
                frame_results.append(r)
                n_processed += 1
                progress_bar.progress(n_processed / max_frames)
                status_text.text(f"Processed frame {n_processed}/{max_frames}")

            cap.release()
            os.unlink(tmp_path)
            status_text.empty()

            if frame_results:
                # Show last frame result
                result = frame_results[-1]
                # Average timings summary
                avg_total = sum(r.timings.get("total_ms", 0) for r in frame_results) / len(frame_results)
                st.success(
                    f"Processed {len(frame_results)} frames.  "
                    f"Average latency: {avg_total:.1f} ms  "
                    f"({1000/avg_total:.1f} FPS equivalent)"
                )

    # ── Results ────────────────────────────────────────────────────────────
    if result is not None:
        st.markdown("---")
        render_results(result, show_original)


if __name__ == "__main__":
    main()
