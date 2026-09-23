"""
dashboard/app.py
=================
APEX LKA — Real-Time Lane Perception & Driver Assistance Console.

Automotive Workstation Presentation Layer.
Decoupled camera capture, background inference, software LKA analysis engine,
clean perception viewport, recording, and replay.

SAFETY NOTICE
-------------
Software perception and visualization ONLY.
Zero physical actuator control: No motor control, no PWM, no CAN, no steering actuation.
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

# ── Backend Interfaces (Strictly Preserved — Unchanged) ────────────────────
from src.inference.pipeline import run_pipeline, InferenceResult
from src.inference.predictor import ModelStatus, MLSegmentationPredictor
from src.lka_engine.lka_state import (
    LKAStateTracker,
    LKATelemetry,
    LaneStabilityState,
    LDWState,
)

# ── Decoupled Streaming, Recording, & HUD ──────────────────────────────────
from dashboard.camera_stream import get_stream_manager, PerformanceStats
from dashboard.components.lka_hud import draw_hud_overlay
from dashboard.components.event_log import (
    add_event,
    render_event_log,
    EventType,
    clear_events,
    get_last_event,
)
from dashboard.components.offset_chart import (
    add_offset_sample,
    clear_offset_history,
)
from dashboard.components.recorder import (
    get_recorder,
    list_recorded_sessions,
    SessionReplayer,
)
from dashboard.components.telemetry import (
    render_lka_status_card,
    render_lateral_offset_gauge,
    render_perception_cluster,
    render_performance_cluster,
    render_system_health_strip,
)


# ═══════════════════════════════════════════════════════════════════════════
# Page configuration — MUST be first Streamlit call
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="APEX LKA | Perception Workstation",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "APEX LKA — Automotive Lane Perception & Assistance Workstation. Software perception only.",
    },
)


# ═══════════════════════════════════════════════════════════════════════════
# Design System CSS — Clean Automotive Workstation Aesthetic
# ═══════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Base Theme ── */
[data-testid="stAppViewContainer"] {
    background-color: #070b12;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0a101b;
    border-right: 1px solid #142032;
}
[data-testid="stSidebar"] * { color: #b8c7d9 !important; }
[data-testid="stHeader"] { background: transparent; }

/* ── Typography & Clean Controls ── */
h1, h2, h3, h4 { font-family: 'Inter', sans-serif; font-weight: 700; }
.stButton button {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    border-radius: 4px;
    padding: 6px 16px;
    transition: all 0.15s ease-in-out;
}
.stButton button[kind="primary"] {
    background: #00b4cc;
    color: #060c14;
    border: none;
}
.stButton button[kind="primary"]:hover {
    background: #00d4ee;
    box-shadow: 0 0 10px #00b4cc66;
}
.stButton button:not([kind="primary"]) {
    background: #0c1421;
    color: #b8c7d9;
    border: 1px solid #1a283c;
}
.stButton button:not([kind="primary"]):hover {
    border-color: #00b4cc;
    color: #e2e8f0;
}

/* ── Topbar Header ── */
.workstation-header {
    background: #090e17;
    border-bottom: 1px solid #142032;
    padding: 12px 20px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 12px;
    margin-bottom: 14px;
    border-radius: 6px;
}
.brand-title {
    font-family: 'Inter', sans-serif;
    font-size: 1.35rem;
    font-weight: 800;
    letter-spacing: 0.18em;
    color: #f1f5f9;
    line-height: 1;
}
.brand-accent { color: #00d4ee; }
.brand-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    color: #4a6178;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin-top: 4px;
}
.header-chips {
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
}
.header-chip {
    display: flex;
    align-items: center;
    gap: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

/* ── Hero Viewport Container ── */
.viewport-hero {
    background: #090e17;
    border: 1px solid #162438;
    border-radius: 8px;
    padding: 4px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
    margin-bottom: 14px;
}
.standby-viewport {
    background: radial-gradient(circle at center, #0d1522 0%, #060a10 100%);
    border: 1px solid #162438;
    border-radius: 8px;
    padding: 60px 20px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 12px;
}

/* ── Mode Toolbar ── */
.mode-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: #0c1421;
    border: 1px solid #162438;
    border-radius: 6px;
    padding: 6px 12px;
    margin-bottom: 12px;
}

/* ── Event Summary Line ── */
.event-summary-bar {
    background: #080d16;
    border: 1px solid #142032;
    border-radius: 4px;
    padding: 8px 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    margin-top: 10px;
}

/* ── Streamlit Tabs Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0a101b;
    border-radius: 4px;
    gap: 4px;
    border-bottom: 1px solid #142032;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #4a6178 !important;
    padding: 6px 16px;
}
.stTabs [aria-selected="true"] {
    color: #00d4ee !important;
    border-bottom: 2px solid #00d4ee !important;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════

def _init_state() -> None:
    defaults = {
        "cam_active": False,
        "cam_frames": 0,
        "last_result": None,
        "last_telemetry": None,
        "last_perf": None,
        "backend": "classical_cv",
        "webcam_id": 0,
        "conf_thresh": 0.5,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _handle_result(result: InferenceResult, telemetry: Optional[LKATelemetry] = None, source: str = "image") -> None:
    """Store result, feed lateral offset history, and update event log."""
    st.session_state.last_result = result
    st.session_state.last_telemetry = telemetry
    add_offset_sample(result.offset)

    if not result.prediction:
        add_event(EventType.ERROR, f"[{source}] No prediction output")
        return

    pred = result.prediction
    status = pred.status.value

    if result.error:
        add_event(EventType.ERROR, f"[{source}] {result.error[:60]}")
    elif "LANE DETECTED" in status:
        add_event(EventType.LANE_OK, f"[{source}] Dual lanes locked | conf: {telemetry.combined_confidence:.2f}" if telemetry else f"[{source}] Both lanes detected")
    elif "PARTIAL" in status:
        side = "left" if pred.left_detected else "right"
        add_event(EventType.LANE_PARTIAL, f"[{source}] Single lane locked ({side})")
    else:
        add_event(EventType.LANE_LOST, f"[{source}] Lane boundaries lost")

    if telemetry and telemetry.ldw_state == LDWState.WARNING:
        add_event(EventType.LOW_CONF, f"[{source}] LDW ALERT: Vehicle departing lane boundary!")


# ═══════════════════════════════════════════════════════════════════════════
# Topbar Header
# ═══════════════════════════════════════════════════════════════════════════

def _render_header(
    cam_active: bool,
    result: Optional[InferenceResult],
    perf: Optional[PerformanceStats],
    telemetry: Optional[LKATelemetry],
) -> None:
    # System status
    if cam_active:
        sys_dot = "#10b981"
        sys_text = "SYSTEM ONLINE"
    elif result is not None:
        sys_dot = "#00d4ee"
        sys_text = "SYSTEM ACTIVE"
    else:
        sys_dot = "#4a6178"
        sys_text = "SYSTEM READY"

    # FPS readout
    if perf and (perf.camera_fps > 0 or perf.ui_fps > 0):
        fps_val = perf.camera_fps if perf.camera_fps > 0 else perf.ui_fps
        fps_str = f"{fps_val:.1f} FPS"
    elif result and result.timings:
        fps_v = result.timings.get("fps_equiv", "--")
        fps_str = f"{fps_v:.0f} FPS" if isinstance(fps_v, (int, float)) else f"{fps_v} FPS"
    else:
        fps_str = "-- FPS"

    # Backend badge
    backend_key = st.session_state.get("backend", "classical_cv")
    bk_name = "CLASSICAL CV" if backend_key == "classical_cv" else "ML MODEL"

    html = f"""
    <div class="workstation-header">
      <div>
        <div class="brand-title"><span class="brand-accent">APEX</span> LKA</div>
        <div class="brand-sub">Real-Time ADAS Lane Perception Workstation</div>
      </div>
      <div class="header-chips">
        <div class="header-chip">
          <span style="width:7px; height:7px; border-radius:50%; background:{sys_dot}; box-shadow:0 0 8px {sys_dot};"></span>
          <span style="color:#e2e8f0;">{sys_text}</span>
        </div>
        <div class="header-chip" style="color:#00d4ee;">
          <span>{fps_str}</span>
        </div>
        <div class="header-chip" style="color:#7a8d9e;">
          <span>BACKEND: {bk_name}</span>
        </div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Sidebar Navigation & Settings
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🎯 APEX LKA")
        st.caption("Automotive Perception Workstation")
        st.markdown("---")

        # ── Backend selection ──────────────────────────────────────────────
        st.markdown('<div style="font-family:monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">PERCEPTION BACKEND</div>', unsafe_allow_html=True)
        ml = MLSegmentationPredictor()
        ml_ready = (ml.model_status == ModelStatus.READY)

        bk_choice = st.radio(
            "Backend",
            options=["Classical CV (OpenCV)", "ML Segmentation"],
            index=0 if st.session_state.backend == "classical_cv" else 1,
            label_visibility="collapsed",
        )
        if bk_choice == "Classical CV (OpenCV)":
            st.session_state.backend = "classical_cv"
        else:
            if not ml_ready:
                st.warning("ML model checkpoint not found. Fallback: Classical CV.")
                st.session_state.backend = "classical_cv"
            else:
                st.session_state.backend = "ml_segmentation"

        st.markdown("---")

        # ── Camera Selection ──────────────────────────────────────────────
        st.markdown('<div style="font-family:monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">CAMERA DEVICE</div>', unsafe_allow_html=True)
        st.session_state.webcam_id = st.selectbox(
            "Device ID", [0, 1, 2], index=st.session_state.webcam_id,
            help="Windows DirectShow device index / V4L2 device",
            label_visibility="collapsed",
        )

        st.markdown("---")

        # ── Navigation Links to Secondary Pages ───────────────────────────
        st.markdown('<div style="font-family:monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">SECONDARY CONSOLES</div>', unsafe_allow_html=True)
        st.page_link("pages/01_Diagnostics.py", label="Diagnostics Console", icon="🔬")
        st.page_link("pages/02_Perception.py", label="Perception Deep-Dive", icon="👁")
        st.page_link("pages/03_Recordings.py", label="Session Recordings", icon="📼")
        st.page_link("pages/04_Dataset.py", label="Dataset Status", icon="📊")
        st.page_link("pages/07_Settings.py", label="System Configuration", icon="⚙")

        st.markdown("---")

        # ── Clear Session ─────────────────────────────────────────────────
        if st.button("🗑 Reset Session Data", use_container_width=True):
            clear_events()
            clear_offset_history()
            st.session_state.last_result = None
            st.session_state.last_telemetry = None
            st.session_state.last_perf = None
            st.rerun()

        st.caption(
            "APEX LKA — Software perception prototype.  \n"
            "Strictly zero hardware actuation."
        )


# ═══════════════════════════════════════════════════════════════════════════
# Mode Handlers
# ═══════════════════════════════════════════════════════════════════════════

def _run_live_camera(frame_container) -> None:
    stream_mgr = get_stream_manager()
    recorder = get_recorder()
    cam_active = st.session_state.cam_active

    if cam_active:
        res, telem, perf = stream_mgr.get_latest()

        if res is not None and res.annotated_bgr is not None:
            # Sync recording if enabled
            if recorder.is_recording and res.preprocessed and res.preprocessed.original_bgr is not None:
                recorder.record_frame(res.preprocessed.original_bgr, telem, perf)

            # Draw clean, uncluttered HUD
            hud_frame = draw_hud_overlay(res.annotated_bgr, telemetry=telem, perf=perf, mode_label="LIVE")
            rgb_frame = cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB)
            frame_container.image(rgb_frame, use_container_width=True)

            st.session_state.last_result = res
            st.session_state.last_telemetry = telem
            st.session_state.last_perf = perf
            st.session_state.cam_frames += 1

            if res.offset and st.session_state.cam_frames % 2 == 0:
                add_offset_sample(res.offset)

        time.sleep(0.01)
        st.rerun()

    else:
        # Sleek standby screen
        frame_container.markdown(
            """
            <div class="standby-viewport">
              <div style="font-family:'JetBrains Mono', monospace; font-size:1.1rem; font-weight:700; color:#cbd5e1; letter-spacing:0.12em;">
                CAMERA STANDBY
              </div>
              <div style="font-family:'JetBrains Mono', monospace; font-size:0.65rem; color:#4a6178; letter-spacing:0.10em;">
                DECOUPLED REAL-TIME PERCEPTION PIPELINE READY
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _run_image_mode(frame_container) -> None:
    backend = st.session_state.backend
    c_up, c_test = st.columns([1, 1])

    with c_up:
        uploaded = st.file_uploader("Upload road image", type=["jpg", "jpeg", "png", "bmp"], key="up_img")
        if uploaded and st.button("▶ RUN PERCEPTION ON UPLOAD", type="primary", use_container_width=True):
            img_bytes = uploaded.read()
            with st.spinner("Processing..."):
                res = run_pipeline(img_bytes, backend=backend, reset_pd_state=True)
                tracker = LKAStateTracker()
                telem = tracker.update(res)
            _handle_result(res, telem, source="image")

    with c_test:
        test_dir = PROJECT_ROOT / "data" / "test_images"
        test_files = sorted(f.name for f in test_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}) if test_dir.exists() else []
        if test_files:
            chosen = st.selectbox("Select Test Benchmark Image", test_files, key="sel_test_img")
            if st.button("▶ RUN BENCHMARK IMAGE", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    res = run_pipeline(test_dir / chosen, backend=backend, reset_pd_state=True)
                    tracker = LKAStateTracker()
                    telem = tracker.update(res)
                _handle_result(res, telem, source="test")
        else:
            st.info("Place images in data/test_images/")

    # Display image in hero viewport
    r = st.session_state.last_result
    t = st.session_state.last_telemetry
    if r and r.annotated_bgr is not None:
        hud_frame = draw_hud_overlay(r.annotated_bgr, telemetry=t, perf=None, mode_label="IMAGE")
        frame_container.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)


def _run_video_mode(frame_container) -> None:
    backend = st.session_state.backend
    vid_file = st.file_uploader("Upload road video file", type=["mp4", "avi", "mov", "mkv"], key="up_vid")

    c1, c2 = st.columns(2)
    with c1:
        max_f = st.slider("Max Frames to Process", 10, 300, 60, key="vid_max_f")
    with c2:
        skip_f = st.slider("Step (Process 1 in N frames)", 1, 5, 1, key="vid_skip_f")

    if vid_file and st.button("▶ PROCESS VIDEO", type="primary", use_container_width=True):
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(vid_file.name).suffix) as tmp:
            tmp.write(vid_file.read())
            tmp_path = tmp.name

        add_event(EventType.CAMERA, f"Video stream loaded: {vid_file.name}")
        cap = cv2.VideoCapture(tmp_path)
        prog = st.progress(0)
        tracker = LKAStateTracker()
        n_total = 0
        n_proc = 0
        last_r = None
        last_t = None

        while cap.isOpened() and n_proc < max_f:
            ret, frame = cap.read()
            if not ret:
                break
            n_total += 1
            if (n_total - 1) % skip_f != 0:
                continue

            r = run_pipeline(frame, backend=backend, reset_pd_state=(n_proc == 0))
            telem = tracker.update(r)
            _handle_result(r, telem, source=f"video:{n_proc}")
            last_r = r
            last_t = telem
            n_proc += 1

            if r.annotated_bgr is not None:
                hud = draw_hud_overlay(r.annotated_bgr, telemetry=telem, perf=None, mode_label="VIDEO")
                frame_container.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), use_container_width=True)

            prog.progress(min(n_proc / max_f, 1.0))

        cap.release()
        os.unlink(tmp_path)
        prog.empty()
        st.session_state.last_result = last_r
        st.session_state.last_telemetry = last_t
        add_event(EventType.INFERENCE, f"Completed processing {n_proc} video frames")


def _run_replay_mode(frame_container) -> None:
    backend = st.session_state.backend
    sessions = list_recorded_sessions()

    if not sessions:
        st.info("No recorded sessions found in data/recordings/. Record a live session to replay.")
        return

    sess_map = {f"{s['id']}  ({s['frames']} frames, {s['size_mb']} MB)": s for s in sessions}
    chosen_label = st.selectbox("Select Session to Replay", list(sess_map.keys()), key="sel_replay")
    chosen = sess_map[chosen_label]

    replayer = SessionReplayer(chosen["path"])
    if replayer.total_frames == 0:
        st.warning("Session has no frames.")
        return

    idx = st.slider("Frame Scrubber", 0, replayer.total_frames - 1, 0, key="scrub_frame")
    frame_bgr = replayer.get_frame(idx)

    if frame_bgr is not None:
        r = run_pipeline(frame_bgr, backend=backend, reset_pd_state=(idx == 0))
        tracker = LKAStateTracker()
        telem = tracker.update(r)
        _handle_result(r, telem, source=f"replay:{idx}")

        if r.annotated_bgr is not None:
            hud = draw_hud_overlay(r.annotated_bgr, telemetry=telem, perf=None, mode_label="REPLAY")
            frame_container.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    _init_state()
    _render_sidebar()

    stream_mgr = get_stream_manager()
    recorder = get_recorder()
    cam_active = st.session_state.cam_active

    result = st.session_state.last_result
    telemetry = st.session_state.last_telemetry
    perf = st.session_state.last_perf

    # ── Header ────────────────────────────────────────────────────────────
    _render_header(cam_active, result, perf, telemetry)

    # ── Input Mode Selector Toolbar ──
    tab_live, tab_img, tab_vid, tab_replay = st.tabs([
        "📡  LIVE CAMERA",
        "📁  IMAGE",
        "🎬  VIDEO",
        "📼  REPLAY",
    ])

    # ── Tab 1: Live Camera ────────────────────────────────────────────────
    with tab_live:
        c_cam_ctrl, c_rec_ctrl, c_cam_stat = st.columns([1.2, 1.2, 1.8])

        with c_cam_ctrl:
            if not cam_active:
                if st.button("▶ START CAMERA", type="primary", use_container_width=True, key="live_btn_start"):
                    ok = stream_mgr.start(source=st.session_state.webcam_id, backend=st.session_state.backend)
                    if ok:
                        st.session_state.cam_active = True
                        add_event(EventType.CAMERA, f"Live camera {st.session_state.webcam_id} stream started")
                        st.rerun()
                    else:
                        st.error(f"Cannot access camera {st.session_state.webcam_id}.")
            else:
                if st.button("⏹ STOP CAMERA", use_container_width=True, key="live_btn_stop"):
                    stream_mgr.stop()
                    if recorder.is_recording:
                        recorder.stop()
                    st.session_state.cam_active = False
                    add_event(EventType.CAMERA, "Live camera stream stopped")
                    st.rerun()

        with c_rec_ctrl:
            if not recorder.is_recording:
                if st.button("🔴 REC SESSION", disabled=(not cam_active), use_container_width=True, key="live_btn_rec"):
                    sess_path = recorder.start()
                    add_event(EventType.SYSTEM, f"Session recording started: {sess_path.name}")
                    st.rerun()
            else:
                if st.button("⏹ STOP REC", type="primary", use_container_width=True, key="live_btn_stoprec"):
                    recorder.stop()
                    add_event(EventType.SYSTEM, f"Recording saved: {recorder.frames_recorded} frames ({recorder.disk_usage_mb} MB)")
                    st.rerun()

        with c_cam_stat:
            if recorder.is_recording:
                st.markdown(
                    f'<div style="background:#220909; border:1px solid #ef4444; border-radius:4px; padding:6px 12px; '
                    f'font-family:\'JetBrains Mono\', monospace; font-size:0.65rem; color:#ef4444; text-align:center;">'
                    f'● REC ACTIVE: {recorder.frames_recorded} frames | {recorder.disk_usage_mb} MB'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            elif cam_active:
                st.markdown(
                    f'<div style="background:#091a13; border:1px solid #10b981; border-radius:4px; padding:6px 12px; '
                    f'font-family:\'JetBrains Mono\', monospace; font-size:0.65rem; color:#10b981; text-align:center;">'
                    f'● DECOUPLED STREAM LIVE'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # Hero Viewport for Live Camera
        live_frame_container = st.empty()
        _run_live_camera(live_frame_container)

    # ── Tab 2: Image Benchmark ────────────────────────────────────────────
    with tab_img:
        img_frame_container = st.empty()
        _run_image_mode(img_frame_container)

    # ── Tab 3: Video Stream ───────────────────────────────────────────────
    with tab_vid:
        vid_frame_container = st.empty()
        _run_video_mode(vid_frame_container)

    # ── Tab 4: Session Replay ─────────────────────────────────────────────
    with tab_replay:
        replay_frame_container = st.empty()
        _run_replay_mode(replay_frame_container)

    # ═══════════════════════════════════════════════════════════════════════
    # Primary Operational Telemetry (Below Hero Viewport)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # 1. LKA State & Departure Alert Card
        render_lka_status_card(telemetry=telemetry, result=result)

        # 2. Precision Lateral Offset Centerline Gauge
        render_lateral_offset_gauge(telemetry=telemetry, result=result)

    with col_right:
        # 3. Perception Cluster
        render_perception_cluster(telemetry=telemetry, result=result)

        # 4. Performance Cluster
        render_performance_cluster(perf=perf, result=result)

    # ── Subsystem Health Strip ─────────────────────────────────────────────
    render_system_health_strip(camera_active=cam_active, result=result, telemetry=telemetry)

    # ── Last Event & Expandable Event Stream ───────────────────────────────
    last_ev = get_last_event()
    if last_ev:
        last_ev_text = f"[{last_ev['type'].value}] {last_ev['message']} ({last_ev['ts']})"
    else:
        last_ev_text = "System initialized and waiting for perception data"

    html_event_bar = f"""
    <div class="event-summary-bar">
      <div>
        <span style="color:#4a6178; font-weight:700; letter-spacing:0.12em;">LAST EVENT:</span>
        <span style="color:#94a3b8; margin-left:8px;">{last_ev_text}</span>
      </div>
      <div style="color:#4a6178; font-size:0.58rem;">
        Click below to inspect full event stream
      </div>
    </div>
    """
    st.markdown(html_event_bar, unsafe_allow_html=True)

    with st.expander("▼ Detailed System Event Stream"):
        render_event_log(max_rows=15)


if __name__ == "__main__":
    main()
