"""
dashboard/app.py
=================
APEX LKA — Real-Time Lane Perception & Driver Assistance System.

Automotive-grade ADAS interface, decoupled camera capture, background inference,
software LKA analysis engine, HUD viewport, recording, and replay.

Launch
------
    streamlit run dashboard/app.py
    python run.py dashboard

Architecture
------------
This file is an INTERFACE LAYER only.
All inference is delegated to the existing pipeline:
    src/inference/pipeline.run_pipeline()
Decoupled camera & inference threading:
    dashboard/camera_stream.StreamManager
LKA temporal stability & LDW:
    src/lka_engine/lka_state.LKAStateTracker
Session recording & replay:
    dashboard/components/recorder

Safety
------
This is SOFTWARE PERCEPTION + VISUALIZATION ONLY.
No motor control. No PWM. No CAN. No steering actuators.
"""

import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import streamlit as st

# ── Project root ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Existing pipeline (single import — no duplication) ────────────────────
from src.inference.pipeline import run_pipeline, InferenceResult
from src.inference.predictor import (
    ModelStatus, DetectionStatus,
    MLSegmentationPredictor,
)
from src.lane_geometry.offset_calculator import SteeringRecommendation, DriftDirection
from src.lka_engine.lka_state import (
    LKAStateTracker,
    LKATelemetry,
    LaneStabilityState,
    LDWState,
    FailureCondition,
)

# ── Dashboard components ───────────────────────────────────────────────────
from dashboard.camera_stream import get_stream_manager, PerformanceStats
from dashboard.components.event_log          import add_event, render_event_log, EventType, clear_events
from dashboard.components.offset_chart       import add_offset_sample, render_offset_chart, clear_offset_history
from dashboard.components.steering_indicator import render_steering_indicator
from dashboard.components.pipeline_status    import render_pipeline_status
from dashboard.components.telemetry          import render_telemetry_panel
from dashboard.components.lka_hud            import draw_hud_overlay
from dashboard.components.recorder           import (
    get_recorder,
    list_recorded_sessions,
    SessionReplayer,
)


# ═══════════════════════════════════════════════════════════════════════════
# Page configuration — MUST be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title    = "APEX LKA | ADAS Perception Console",
    page_icon     = "🎯",
    layout        = "wide",
    initial_sidebar_state = "expanded",
    menu_items    = {
        "Get Help":    None,
        "Report a bug": None,
        "About":       "APEX LKA — Automotive Lane Perception & Assistance Prototype. Software perception only.",
    },
)


# ═══════════════════════════════════════════════════════════════════════════
# CSS — APEX LKA dark engineering theme
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

/* ── Base ── */
[data-testid="stAppViewContainer"] {
    background-color: #070d15;
    color: #e2e8f0;
    font-family: 'Inter', system-ui, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0a1220;
    border-right: 1px solid #1a2d42;
}
[data-testid="stSidebar"] * { color: #c5d0dc !important; }
[data-testid="stHeader"] { background: transparent; }
.stButton button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    border-radius: 4px;
}
.stButton button[kind="primary"] {
    background: #00b4cc;
    color: #060c14;
    border: none;
}
.stButton button[kind="primary"]:hover { background: #00d4ee; }
.stButton button:not([kind="primary"]) {
    background: #0d1829;
    color: #c5d0dc;
    border: 1px solid #1a2d42;
}
hr { border-color: #1a2d42; }

/* ── Top bar ── */
.apex-topbar {
    background: #0c1524;
    border-bottom: 1px solid #1a2d42;
    padding: 12px 22px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 16px;
}
.apex-brand-block { display:flex; flex-direction:column; }
.apex-brand {
    font-family: 'Inter', sans-serif;
    font-size: 1.45rem;
    font-weight: 800;
    color: #e2e8f0;
    letter-spacing: 0.20em;
    line-height: 1;
}
.apex-brand .apex-accent { color: #00b4cc; }
.apex-subtitle {
    font-size: 0.56rem;
    color: #4a6178;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-top: 3px;
}
.apex-status-bar { display:flex; align-items:center; gap:16px; flex-wrap:wrap; }
.status-chip {
    display:flex; align-items:center; gap:6px;
    font-family:'JetBrains Mono', monospace;
    font-size:0.65rem; font-weight:600;
    color:#6b8299; letter-spacing:0.08em; text-transform:uppercase;
}
.status-chip .chip-label { color:#4a6178; font-size:0.56rem; }
.status-chip .chip-val   { color:#e2e8f0; }
.status-chip .chip-val.ok    { color:#10b981; }
.status-chip .chip-val.warn  { color:#f59e0b; }
.status-chip .chip-val.err   { color:#ef4444; }
.status-chip .chip-val.info  { color:#00b4cc; }
.dot {
    width:7px; height:7px; border-radius:50%;
    display:inline-block; flex-shrink:0;
}
.dot-green  { background:#10b981; box-shadow:0 0 5px #10b981; }
.dot-yellow { background:#f59e0b; }
.dot-red    { background:#ef4444; box-shadow:0 0 5px #ef4444; }
.dot-gray   { background:#3a5070; }

/* ── Section labels ── */
.section-label {
    font-family:'JetBrains Mono', monospace;
    font-size:0.65rem; font-weight:600; letter-spacing:0.18em;
    color:#4a6178; text-transform:uppercase;
    border-bottom: 1px solid #1a2d42;
    padding-bottom: 6px; margin-bottom: 10px; margin-top: 14px;
}

/* ── Perception viewport ── */
.viewport-label {
    font-family:'JetBrains Mono', monospace;
    font-size:0.6rem; color:#4a6178; letter-spacing:0.15em;
    text-align:center; text-transform:uppercase;
    margin-top:4px;
}
.viewport-border {
    border: 1px solid #1a2d42;
    border-radius: 6px;
    overflow: hidden;
    padding: 2px;
    background: #0c1524;
}

/* ── Telemetry panel ── */
.telemetry-panel { display:flex; flex-direction:column; gap:0; }
.tsection { margin-bottom:12px; }
.tsection-title {
    font-family:'JetBrains Mono', monospace;
    font-size:0.58rem; font-weight:700; letter-spacing:0.20em;
    color:#00b4cc; text-transform:uppercase;
    border-bottom:1px solid #1a2d42; padding-bottom:4px; margin-bottom:4px;
}
.trow {
    display:flex; justify-content:space-between; align-items:center;
    padding:4px 0; border-bottom:1px solid #0d1829;
}
.trow-label {
    font-family:'JetBrains Mono', monospace;
    font-size:0.60rem; color:#4a6178;
    text-transform:uppercase; letter-spacing:0.08em; flex-shrink:0;
}
.trow-value {
    font-family:'JetBrains Mono', monospace;
    font-size:0.70rem; font-weight:600; color:#e2e8f0;
    text-align:right;
}
.trow-detail {
    font-size:0.55rem; color:#4a6178; margin-left:6px;
}
.health-dot {
    display:inline-block; width:6px; height:6px; border-radius:50%;
    margin-right:5px; flex-shrink:0;
}

/* ── Event log ── */
.event-log {
    background:#0c1524; border:1px solid #1a2d42;
    border-radius:4px; padding:6px 10px;
    max-height:160px; overflow-y:auto;
    font-family:'JetBrains Mono', monospace; font-size:0.62rem;
}
.event-row {
    display:flex; align-items:center; gap:8px;
    padding:2px 0; border-bottom:1px solid #0d1829;
}
.event-ts    { color:#2a3d52; font-size:0.58rem; flex-shrink:0; }
.event-tag   { font-weight:700; font-size:0.60rem; flex-shrink:0; min-width:26px; }
.event-msg   { color:#8a9ab0; font-size:0.62rem; }
.event-ok    { color:#10b981; }
.event-warn  { color:#f59e0b; }
.event-err   { color:#ef4444; }
.event-info  { color:#00b4cc; }
.event-sys   { color:#6b8299; }
.event-log-empty {
    font-family:'JetBrains Mono', monospace;
    font-size:0.62rem; color:#2a3d52;
    text-align:center; padding:12px;
}

/* ── Mode tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background:#0c1524; border-radius:4px; gap:2px;
}
.stTabs [data-baseweb="tab"] {
    font-family:'JetBrains Mono', monospace !important;
    font-size:0.70rem; font-weight:600; letter-spacing:0.10em;
    color:#4a6178 !important; padding:8px 18px;
}
.stTabs [aria-selected="true"] { color:#00b4cc !important; }

/* ── Timing pills ── */
.timing-bar { display:flex; gap:6px; flex-wrap:wrap; margin-top:6px; }
.timing-pill {
    background:#0c1524; border:1px solid #1a2d42; border-radius:20px;
    padding:3px 10px; font-family:'JetBrains Mono', monospace;
    font-size:0.60rem; color:#4a6178;
}
.timing-pill span { color:#00b4cc; font-weight:600; }

.cv-label {
    background:#1a1000; border:1px solid #604000;
    border-radius:4px; padding:6px 10px;
    font-family:'JetBrains Mono', monospace;
    font-size:0.62rem; color:#f59e0b;
    margin-bottom:8px;
}

.sidebar-section {
    font-family:'JetBrains Mono', monospace;
    font-size:0.58rem; font-weight:700; letter-spacing:0.20em;
    color:#00b4cc; text-transform:uppercase;
    border-bottom:1px solid #1a2d42;
    padding-bottom:4px; margin:14px 0 8px;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _chip(label: str, value: str, kind: str = "") -> str:
    cls = f" chip-val {kind}" if kind else " chip-val"
    return (
        f'<div class="status-chip">'
        f'<span class="chip-label">{label}</span>'
        f'<span class="{cls}">{value}</span>'
        f'</div>'
    )


def _dot(kind: str) -> str:
    return f'<span class="dot dot-{kind}"></span>'


# ═══════════════════════════════════════════════════════════════════════════
# Session-state initialization
# ═══════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "cam_active":     False,
        "cam_frames":     0,
        "last_result":    None,
        "last_telemetry": None,
        "last_perf":      None,
        "backend":        "classical_cv",
        "webcam_id":      0,
        "conf_thresh":    0.5,
        "reset_pd":       True,
        "is_recording":   False,
        "replay_playing": False,
        "replay_frame":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _handle_result(result: InferenceResult, telemetry: Optional[LKATelemetry] = None, source: str = "image") -> None:
    """Update session state history and event log."""
    st.session_state.last_result = result
    st.session_state.last_telemetry = telemetry
    add_offset_sample(result.offset)

    if not result.prediction:
        add_event(EventType.ERROR, f"[{source}] No prediction returned")
        return

    pred = result.prediction
    status = pred.status.value

    if result.error:
        add_event(EventType.ERROR, f"[{source}] {result.error[:70]}")
    elif "LANE DETECTED" == status:
        lc = f"{pred.left_confidence:.2f}" if pred.left_detected else "--"
        rc = f"{pred.right_confidence:.2f}" if pred.right_detected else "--"
        add_event(EventType.LANE_OK, f"[{source}] Both lanes L:{lc} R:{rc}")
    elif "PARTIAL" in status:
        side = "left" if pred.left_detected else "right"
        add_event(EventType.LANE_PARTIAL, f"[{source}] Partial — {side} lane only")
    else:
        add_event(EventType.LANE_LOST, f"[{source}] Lane not detected")

    if telemetry and telemetry.ldw_state == LDWState.WARNING:
        add_event(EventType.LOW_CONF, f"[{source}] LDW Warning — Vehicle departing lane!")


# ═══════════════════════════════════════════════════════════════════════════
# Top bar
# ═══════════════════════════════════════════════════════════════════════════

def _render_topbar(
    result: Optional[InferenceResult],
    cam_active: bool,
    telemetry: Optional[LKATelemetry] = None,
    perf: Optional[PerformanceStats] = None,
) -> None:
    ml = MLSegmentationPredictor()
    ms = ml.model_status

    sys_dot, sys_val = "green", "ONLINE"

    # Model status
    if ms == ModelStatus.NOT_TRAINED:
        mdl_dot, mdl_val, mdl_kind = "gray",   "NOT TRAINED",  "warn"
    elif ms == ModelStatus.READY:
        mdl_dot, mdl_val, mdl_kind = "green",  "READY",        "ok"
    elif ms == ModelStatus.LOAD_ERROR:
        mdl_dot, mdl_val, mdl_kind = "red",    "LOAD ERROR",   "err"
    else:
        mdl_dot, mdl_val, mdl_kind = "gray",   ms.value,       ""

    # Camera status
    cam_dot = "green" if cam_active else ("yellow" if result else "gray")
    cam_val = "LIVE"  if cam_active else ("IMAGE"  if result else "IDLE")

    # Inference status
    inf_dot = "green" if cam_active or result else "gray"
    inf_val = "RUNNING" if cam_active else ("DONE" if result else "IDLE")

    # FPS & Latency from real measurements
    fps_str = "--"
    lat_str = "--"
    if perf and perf.camera_fps > 0:
        fps_str = f"{perf.camera_fps:.1f}"
        lat_str = f"{perf.pipeline_latency_ms:.0f}ms"
    elif result and result.timings:
        fps_v = result.timings.get("fps_equiv")
        lat_v = result.timings.get("total_ms")
        fps_str = f"{fps_v}" if fps_v else "--"
        lat_str = f"{lat_v:.0f}ms" if lat_v else "--"

    # LDW Alert chip
    ldw_str = telemetry.ldw_state.value if telemetry else "NORMAL"
    ldw_kind = "ok" if ldw_str == "NORMAL" else ("warn" if "DRIFT" in ldw_str else "err")

    # Backend
    bk = st.session_state.get("backend", "classical_cv")
    bk_label = "CV BASELINE" if bk == "classical_cv" else "ML MODEL"
    bk_kind  = "warn"        if bk == "classical_cv" else "ok"

    html = f"""
    <div class="apex-topbar">
      <div class="apex-brand-block">
        <div class="apex-brand">
          <span class="apex-accent">APEX</span> LKA
        </div>
        <div class="apex-subtitle">Real-Time ADAS Perception &amp; Driver Assistance System</div>
      </div>
      <div class="apex-status-bar">
        {_chip("SYSTEM", sys_val, "ok")}
        {_chip("MODEL", mdl_val, mdl_kind)}
        {_chip("CAMERA", cam_val, "info" if cam_active else "")}
        {_chip("INFERENCE", inf_val, "info" if inf_val == "RUNNING" else "")}
        {_chip("FPS", fps_str, "info")}
        {_chip("LATENCY", lat_str, "info")}
        {_chip("LDW", ldw_str, ldw_kind)}
        {_chip("BACKEND", bk_label, bk_kind)}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("## 🎯 APEX LKA")
        st.markdown("---")

        # ── Backend ────────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Detection Backend</div>', unsafe_allow_html=True)
        ml = MLSegmentationPredictor()
        ml_ready = ml.model_status == ModelStatus.READY

        bk_choice = st.radio(
            "Backend",
            options=["Classical CV (OpenCV)", "ML Segmentation"],
            index=0,
            help="Classical CV: always available. ML: requires trained model.",
        )
        if bk_choice == "Classical CV (OpenCV)":
            st.session_state.backend = "classical_cv"
            st.markdown(
                '<div class="cv-label">⚠ CLASSICAL CV BASELINE<br>Pipeline verification baseline.</div>',
                unsafe_allow_html=True,
            )
        else:
            if not ml_ready:
                st.warning("No trained model. Using Classical CV.")
                st.session_state.backend = "classical_cv"
            else:
                st.session_state.backend = "ml_segmentation"

        st.markdown("---")

        # ── Camera Settings ────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Camera Device</div>', unsafe_allow_html=True)
        st.session_state.webcam_id = st.selectbox(
            "Camera ID", [0, 1, 2], index=0,
            help="DirectShow device index on Windows / V4L2 on Jetson/Linux."
        )

        st.markdown("---")

        # ── Diagnostics Quick Link ─────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Diagnostics &amp; Health</div>', unsafe_allow_html=True)
        st.caption("Inspect latency breakdown, drop rates, and hardware load:")
        st.page_link("pages/05_Diagnostics.py", label="Open Diagnostics Console", icon="🔬")

        st.markdown("---")

        # ── Clear ─────────────────────────────────────────────────────────
        if st.button("🗑 Clear History", use_container_width=True):
            clear_events()
            clear_offset_history()
            st.session_state.last_result = None
            st.session_state.last_telemetry = None
            st.session_state.last_perf = None
            st.rerun()

        st.caption(
            "**APEX LKA** — Software perception only.  \n"
            "Strictly no hardware steering actuation."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mode: IMAGE
# ═══════════════════════════════════════════════════════════════════════════

def _mode_image(left_col, right_col) -> None:
    backend = st.session_state.backend

    with left_col:
        st.markdown('<div class="section-label">Image Input</div>', unsafe_allow_html=True)
        col_up, col_test = st.columns([1, 1])
        result: Optional[InferenceResult] = None
        telemetry: Optional[LKATelemetry] = None

        with col_up:
            uploaded = st.file_uploader(
                "Upload road image",
                type=["jpg", "jpeg", "png", "bmp"],
                key="img_upload",
            )
            if uploaded:
                img_bytes = uploaded.read()
                if st.button("▶ RUN DETECTION", type="primary", use_container_width=True, key="btn_img"):
                    with st.spinner("Executing LKA pipeline..."):
                        result = run_pipeline(img_bytes, backend=backend, reset_pd_state=True)
                        tracker = LKAStateTracker()
                        telemetry = tracker.update(result)
                    _handle_result(result, telemetry, source="image")

        with col_test:
            test_dir = PROJECT_ROOT / "data" / "test_images"
            test_imgs = sorted(
                f for f in test_dir.iterdir()
                if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ) if test_dir.exists() else []
            if test_imgs:
                sel = st.selectbox("Test image", [f.name for f in test_imgs], key="sel_test")
                if st.button("▶ RUN TEST IMAGE", type="primary", use_container_width=True, key="btn_test"):
                    with st.spinner("Executing LKA pipeline..."):
                        result = run_pipeline(test_dir / sel, backend=backend, reset_pd_state=True)
                        tracker = LKAStateTracker()
                        telemetry = tracker.update(result)
                    _handle_result(result, telemetry, source="test")
            else:
                st.info("Place images in `data/test_images/`")

        # ── Viewport Display ───────────────────────────────────────────────
        r = result or st.session_state.last_result
        telem = telemetry or st.session_state.last_telemetry

        if r and r.annotated_bgr is not None:
            st.markdown('<div class="section-label">Perception Viewport &amp; HUD</div>', unsafe_allow_html=True)
            hud_frame = draw_hud_overlay(r.annotated_bgr, telemetry=telem, perf=None, mode_label="IMAGE")
            st.image(bgr_to_rgb(hud_frame), use_container_width=True)

        st.markdown('<div class="section-label">Lateral Offset — History</div>', unsafe_allow_html=True)
        render_offset_chart()

        st.markdown('<div class="section-label">System Events</div>', unsafe_allow_html=True)
        render_event_log()

    with right_col:
        r = result or st.session_state.last_result
        telem = telemetry or st.session_state.last_telemetry
        _render_right_panel(r, cam_active=False, telemetry=telem)


# ═══════════════════════════════════════════════════════════════════════════
# Mode: VIDEO
# ═══════════════════════════════════════════════════════════════════════════

def _mode_video(left_col, right_col) -> None:
    backend = st.session_state.backend

    with left_col:
        st.markdown('<div class="section-label">Video Input</div>', unsafe_allow_html=True)
        vid_file = st.file_uploader("Upload road video", type=["mp4", "avi", "mov", "mkv"], key="vid_upload")

        col_ctrl1, col_ctrl2 = st.columns(2)
        with col_ctrl1:
            max_frames = st.slider("Max frames", 10, 300, 60, key="vid_maxframes")
        with col_ctrl2:
            skip_n = st.slider("Process every N frames", 1, 5, 1, key="vid_skip")

        if vid_file and st.button("▶ PROCESS VIDEO", type="primary", use_container_width=True, key="btn_video"):
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix) as tmp:
                tmp.write(vid_file.read())
                tmp_path = tmp.name

            add_event(EventType.CAMERA, f"Video stream loaded: {vid_file.name}")
            cap = cv2.VideoCapture(tmp_path)
            frame_ph = st.empty()
            prog = st.progress(0)
            status_ph = st.empty()

            tracker = LKAStateTracker()
            n_total = 0
            n_proc = 0
            last_result = None
            last_telem = None

            while cap.isOpened() and n_proc < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                n_total += 1
                if (n_total - 1) % skip_n != 0:
                    continue

                r = run_pipeline(frame, backend=backend, reset_pd_state=(n_proc == 0))
                telem = tracker.update(r)
                _handle_result(r, telem, source=f"video:{n_proc}")
                last_result = r
                last_telem = telem
                n_proc += 1

                if r.annotated_bgr is not None:
                    hud_frame = draw_hud_overlay(r.annotated_bgr, telemetry=telem, perf=None, mode_label="VIDEO")
                    frame_ph.image(bgr_to_rgb(hud_frame), use_container_width=True)

                prog.progress(min(n_proc / max_frames, 1.0))
                status_ph.markdown(f"`Frame {n_proc}/{max_frames}` | `Lat: {r.timings.get('total_ms', 0):.0f}ms`")

            cap.release()
            os.unlink(tmp_path)
            prog.empty()
            status_ph.empty()
            st.session_state.last_result = last_result
            st.session_state.last_telemetry = last_telem
            add_event(EventType.INFERENCE, f"Processed {n_proc} video frames.")

        st.markdown('<div class="section-label">Lateral Offset — Video History</div>', unsafe_allow_html=True)
        render_offset_chart()

        st.markdown('<div class="section-label">System Events</div>', unsafe_allow_html=True)
        render_event_log()

    with right_col:
        _render_right_panel(st.session_state.last_result, cam_active=False, telemetry=st.session_state.last_telemetry)


# ═══════════════════════════════════════════════════════════════════════════
# Mode: LIVE CAMERA (Decoupled Real-Time Architecture)
# ═══════════════════════════════════════════════════════════════════════════

def _mode_live(left_col, right_col) -> None:
    backend = st.session_state.backend
    cam_id = st.session_state.webcam_id
    cam_active = st.session_state.cam_active
    stream_mgr = get_stream_manager()
    recorder = get_recorder()

    with left_col:
        st.markdown('<div class="section-label">Live Camera &amp; ADAS Perception</div>', unsafe_allow_html=True)

        # ── Controls: Camera & Recording ───────────────────────────────────
        c_cam_btn, c_rec_btn, c_stat = st.columns([1.2, 1.2, 1.6])

        with c_cam_btn:
            if not cam_active:
                if st.button("▶ START CAMERA", type="primary", use_container_width=True, key="btn_start_cam"):
                    ok = stream_mgr.start(source=cam_id, backend=backend)
                    if ok:
                        st.session_state.cam_active = True
                        add_event(EventType.CAMERA, f"Real-time pipeline started on camera {cam_id}")
                        st.rerun()
                    else:
                        st.error(f"Cannot access camera {cam_id}. Check connection or camera ID.")
            else:
                if st.button("⏹ STOP CAMERA", use_container_width=True, key="btn_stop_cam"):
                    stream_mgr.stop()
                    if recorder.is_recording:
                        recorder.stop()
                    st.session_state.cam_active = False
                    add_event(EventType.CAMERA, "Camera stream stopped")
                    st.rerun()

        with c_rec_btn:
            if not recorder.is_recording:
                rec_disabled = not cam_active
                if st.button("🔴 START REC", disabled=rec_disabled, use_container_width=True, key="btn_start_rec"):
                    sess_path = recorder.start()
                    add_event(EventType.SYSTEM, f"Recording session started: {sess_path.name}")
                    st.rerun()
            else:
                if st.button("⏹ STOP REC", type="primary", use_container_width=True, key="btn_stop_rec"):
                    recorder.stop()
                    add_event(EventType.SYSTEM, f"Recording saved: {recorder.frames_recorded} frames ({recorder.disk_usage_mb} MB)")
                    st.rerun()

        with c_stat:
            if recorder.is_recording:
                st.markdown(
                    f'<div style="background:#2b0d0d; border:1px solid #ef4444; border-radius:4px; padding:6px 10px; '
                    f'font-family:monospace; font-size:0.65rem; color:#ef4444; text-align:center;">'
                    f'&#9679; REC: {recorder.frames_recorded} frames | {recorder.disk_usage_mb} MB ({recorder.elapsed_seconds}s)'
                    f'</div>',
                    unsafe_allow_html=True
                )
            elif cam_active:
                st.markdown(
                    f'<div style="background:#0d2b1e; border:1px solid #10b981; border-radius:4px; padding:6px 10px; '
                    f'font-family:monospace; font-size:0.65rem; color:#10b981; text-align:center;">'
                    f'&#9679; LIVE DECOUPLED STREAM'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Frame Viewport Placeholder ─────────────────────────────────────
        frame_ph = st.empty()
        status_ph = st.empty()

        if cam_active:
            res, telem, perf = stream_mgr.get_latest()

            if res is not None and res.annotated_bgr is not None:
                # Save frame if recording
                if recorder.is_recording and res.preprocessed and res.preprocessed.original_bgr is not None:
                    recorder.record_frame(res.preprocessed.original_bgr, telem, perf)

                # Draw high-tech HUD overlay
                hud_frame = draw_hud_overlay(res.annotated_bgr, telemetry=telem, perf=perf, mode_label="LIVE")
                frame_ph.image(bgr_to_rgb(hud_frame), use_container_width=True)

                st.session_state.last_result = res
                st.session_state.last_telemetry = telem
                st.session_state.last_perf = perf
                st.session_state.cam_frames += 1

                # Feed lateral offset sample
                if res.offset and st.session_state.cam_frames % 2 == 0:
                    add_offset_sample(res.offset)

                fps_show = perf.camera_fps if perf.camera_fps > 0 else perf.ui_fps
                status_ph.markdown(
                    f'`Frame #{st.session_state.cam_frames}` | '
                    f'`Camera: {fps_show:.1f} FPS` | '
                    f'`Inference: {perf.inference_latency_ms:.0f}ms` | '
                    f'`Dropped: {perf.dropped_frames}`'
                )

            # Minimal sleep to yield execution and maintain a smooth 30 FPS display loop
            time.sleep(0.01)
            st.rerun()

        else:
            frame_ph.markdown(
                '<div style="border:1px solid #1a2d42; border-radius:6px; '
                'background:#0c1524; padding:65px 20px; text-align:center;">'
                '<div style="font-family:monospace; font-size:0.80rem; color:#4a6178; '
                'letter-spacing:0.15em;">CAMERA IDLE &mdash; CLICK START CAMERA</div>'
                '<div style="font-family:monospace; font-size:0.62rem; color:#2a3d52; margin-top:6px;">'
                'Zero-Lag Real-Time Decoupled Pipeline Ready</div>'
                '</div>',
                unsafe_allow_html=True,
            )

        # ── Offset chart ───────────────────────────────────────────────────
        st.markdown('<div class="section-label">Lateral Offset — Live History</div>', unsafe_allow_html=True)
        render_offset_chart()

        # ── Event log ─────────────────────────────────────────────────────
        st.markdown('<div class="section-label">System Events</div>', unsafe_allow_html=True)
        render_event_log()

    with right_col:
        _render_right_panel(
            st.session_state.last_result,
            cam_active=cam_active,
            telemetry=st.session_state.last_telemetry,
            perf=st.session_state.last_perf,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mode: REPLAY (Playback Recorded Sessions)
# ═══════════════════════════════════════════════════════════════════════════

def _mode_replay(left_col, right_col) -> None:
    backend = st.session_state.backend

    with left_col:
        st.markdown('<div class="section-label">Session Replay</div>', unsafe_allow_html=True)

        sessions = list_recorded_sessions()
        if not sessions:
            st.info("No recorded sessions found in `data/recordings/`. Record a live session to replay.")
            return

        sess_options = {f"{s['id']} ({s['frames']} frames, {s['size_mb']} MB)": s for s in sessions}
        selected_label = st.selectbox("Select Recording Session", list(sess_options.keys()))
        selected_session = sess_options[selected_label]

        replayer = SessionReplayer(selected_session["path"])
        if replayer.total_frames == 0:
            st.warning("Selected session contains no frames.")
            return

        frame_idx = st.slider("Frame", 0, replayer.total_frames - 1, 0, key="replay_slider")
        frame_bgr = replayer.get_frame(frame_idx)

        if frame_bgr is not None:
            r = run_pipeline(frame_bgr, backend=backend, reset_pd_state=(frame_idx == 0))
            tracker = LKAStateTracker()
            telem = tracker.update(r)
            _handle_result(r, telem, source=f"replay:{frame_idx}")

            if r.annotated_bgr is not None:
                hud_frame = draw_hud_overlay(r.annotated_bgr, telemetry=telem, perf=None, mode_label="REPLAY")
                st.image(bgr_to_rgb(hud_frame), use_container_width=True)

        st.markdown('<div class="section-label">Lateral Offset — Replay History</div>', unsafe_allow_html=True)
        render_offset_chart()

        st.markdown('<div class="section-label">System Events</div>', unsafe_allow_html=True)
        render_event_log()

    with right_col:
        _render_right_panel(
            st.session_state.last_result,
            cam_active=False,
            telemetry=st.session_state.last_telemetry,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Right panel (shared across all modes)
# ═══════════════════════════════════════════════════════════════════════════

def _render_right_panel(
    result: Optional[InferenceResult],
    cam_active: bool,
    telemetry: Optional[LKATelemetry] = None,
    perf: Optional[PerformanceStats] = None,
) -> None:
    st.markdown('<div class="section-label">LKA Status &amp; Telemetry</div>', unsafe_allow_html=True)
    render_telemetry_panel(result, camera_active=cam_active, telemetry=telemetry, perf=perf)

    st.markdown("---")

    st.markdown('<div class="section-label">Steering Recommendation</div>', unsafe_allow_html=True)
    render_steering_indicator(result.offset if result else None, telemetry=telemetry)

    st.markdown("---")

    st.markdown('<div class="section-label">Pipeline Stage Status</div>', unsafe_allow_html=True)
    render_pipeline_status(result, camera_active=cam_active)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entrypoint
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    _render_sidebar()

    # Top bar
    _render_topbar(
        result     = st.session_state.last_result,
        cam_active = st.session_state.cam_active,
        telemetry  = st.session_state.last_telemetry,
        perf       = st.session_state.last_perf,
    )

    # Mode tabs
    tab_img, tab_vid, tab_live, tab_replay = st.tabs([
        "📁  IMAGE",
        "🎬  VIDEO",
        "📡  LIVE CAMERA",
        "📼  REPLAY",
    ])

    left_w, right_w = [3, 1]

    with tab_img:
        c_left, c_right = st.columns([left_w, right_w])
        _mode_image(c_left, c_right)

    with tab_vid:
        c_left, c_right = st.columns([left_w, right_w])
        _mode_video(c_left, c_right)

    with tab_live:
        c_left, c_right = st.columns([left_w, right_w])
        _mode_live(c_left, c_right)

    with tab_replay:
        c_left, c_right = st.columns([left_w, right_w])
        _mode_replay(c_left, c_right)


if __name__ == "__main__":
    main()
