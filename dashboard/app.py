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
# Error Boundary & Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════

def render_error_boundary(title: str, func, *args, **kwargs):
    """
    Execute UI component safely within an exception shield.
    If the component raises an error, renders a graceful warning banner
    instead of crashing the Streamlit app.
    """
    try:
        return func(*args, **kwargs)
    except Exception as exc:
        st.warning(f"⚠️ **{title}**: Component temporarily unavailable ({exc})")
        return None


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
# Streamlined Sidebar (Objective 3)
# ═══════════════════════════════════════════════════════════════════════════

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### 🏎️ APEX LKA")
        st.caption("SAE BAJA & Autonomous Lane Perception")
        st.markdown("---")

        # ── Backend indicator & selector ──────────────────────────────────
        ml = MLSegmentationPredictor()
        ml_ready = (ml.model_status == ModelStatus.READY)

        st.markdown(
            '<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">'
            'PERCEPTION BACKEND'
            '</div>',
            unsafe_allow_html=True,
        )

        if ml_ready:
            bk_choice = st.radio(
                "Perception Backend",
                options=["Classical CV (OpenCV)", "Deep Learning (LaneSegNet)"],
                index=0 if st.session_state.backend == "classical_cv" else 1,
                label_visibility="collapsed",
            )
            st.session_state.backend = "classical_cv" if bk_choice == "Classical CV (OpenCV)" else "ml_segmentation"
        else:
            st.session_state.backend = "classical_cv"
            st.markdown(
                '<div style="background:#0c1524; border:1px solid #1e293b; border-radius:6px; padding:8px 12px; font-family:\'JetBrains Mono\', monospace; font-size:0.72rem; color:#38bdf8;">'
                '● CLASSICAL CV (Active)<br>'
                '<span style="color:#64748b; font-size:0.62rem;">Deep Learning: Train model in Training tab to unlock</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Model Status Badge ────────────────────────────────────────────
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">'
            'NEURAL MODEL STATUS'
            '</div>',
            unsafe_allow_html=True,
        )
        if ml_ready:
            st.markdown(
                '<div style="background:rgba(16,185,129,0.15); border:1px solid #10b981; border-radius:6px; padding:6px 12px; font-family:\'JetBrains Mono\', monospace; font-size:0.75rem; color:#10b981; font-weight:700;">'
                '● READY (best_model.pth)'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:rgba(239,68,68,0.15); border:1px solid rgba(239,68,68,0.5); border-radius:6px; padding:6px 12px; font-family:\'JetBrains Mono\', monospace; font-size:0.75rem; color:#f87171;">'
                '○ NOT TRAINED (Baseline CV)'
                '</div>',
                unsafe_allow_html=True,
            )

        st.markdown("---")

        # ── Quick Stats ───────────────────────────────────────────────────
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">'
            'DATASET & TRAINING STATS'
            '</div>',
            unsafe_allow_html=True,
        )

        # Count total frames & annotated frames
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        n_raw = len([f for f in PATHS.raw_frames.glob("*.*") if f.suffix.lower() in img_exts]) if PATHS.raw_frames.exists() else 0
        n_ann_img = len([f for f in (PATHS.annotated / "images").glob("*.*") if f.suffix.lower() in img_exts]) if (PATHS.annotated / "images").exists() else 0
        n_train_img = len([f for f in (PATHS.train / "images").glob("*.*") if f.suffix.lower() in img_exts]) if (PATHS.train / "images").exists() else 0
        total_frames = n_raw + n_ann_img + n_train_img

        n_masks = len(list((PATHS.annotated / "masks").glob("*.png"))) if (PATHS.annotated / "masks").exists() else 0
        n_train_masks = len(list((PATHS.train / "masks").glob("*.png"))) if (PATHS.train / "masks").exists() else 0
        annotated_frames = n_masks + n_train_masks

        # Check last training mIoU
        last_miou_str = "N/A"
        csv_metric_file = PATHS.logs / "training" / "training_metrics.csv"
        if csv_metric_file.exists():
            try:
                import pandas as pd
                df_m = pd.read_csv(csv_metric_file)
                if "val_mean_iou" in df_m.columns and not df_m.empty:
                    last_miou_str = f"{df_m['val_mean_iou'].max():.4f}"
            except Exception:
                pass

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.metric("Total Frames", f"{total_frames:,}")
        with col_s2:
            st.metric("Annotated", f"{annotated_frames:,}")

        st.metric("Last Training mIoU", last_miou_str)

        st.markdown("---")

        # ── Global Settings Collapsible ───────────────────────────────────
        with st.expander("⚙️ Global Workstation Settings", expanded=False):
            st.session_state.webcam_id = st.selectbox(
                "Camera Device ID",
                [0, 1, 2, 3],
                index=st.session_state.webcam_id if st.session_state.webcam_id < 4 else 0,
                help="Camera device index",
            )
            st.selectbox(
                "Viewport Resolution",
                ["640x360 (Standard)", "1280x720 (HD)", "1920x1080 (FHD)"],
                index=0,
            )
            st.selectbox(
                "Compute Hardware",
                ["GPU (CUDA Accelerated)", "CPU (Portable)"],
                index=0,
            )

        st.markdown("---")

        # ── Secondary Workflow Pages ──────────────────────────────────────
        st.markdown(
            '<div style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase; margin-bottom:8px;">'
            'WORKSTATION PAGES'
            '</div>',
            unsafe_allow_html=True,
        )
        st.page_link("pages/01_Recording.py", label="01. Recording & Ingestion", icon="📼")
        st.page_link("pages/02_Dataset.py", label="02. Dataset Management", icon="⚡")
        st.page_link("pages/03_Training.py", label="03. Training Center", icon="🔬")
        st.page_link("pages/04_Evaluation.py", label="04. Evaluation & Export", icon="📈")

        st.markdown("---")

        # ── Clear & Self-Healing Reset ─────────────────────────────────────
        c_rst1, c_rst2 = st.columns(2)
        with c_rst1:
            if st.button("🗑 Reset", use_container_width=True, help="Clear event history and lateral offset cache"):
                clear_events()
                clear_offset_history()
                st.session_state.last_result = None
                st.session_state.last_telemetry = None
                st.session_state.last_perf = None
                st.rerun()
        with c_rst2:
            if st.button("🔄 Self-Heal", use_container_width=True, help="Restart background workers and recover state"):
                try:
                    sm = get_stream_manager()
                    if sm.is_running:
                        sm.stop()
                except Exception:
                    pass
                clear_events()
                clear_offset_history()
                st.session_state.cam_active = False
                st.session_state.last_result = None
                st.session_state.last_telemetry = None
                st.session_state.last_perf = None
                st.success("Workers reset.")
                st.rerun()

        st.caption("APEX LKA &mdash; SAE BAJA Autonomous System")


# ═══════════════════════════════════════════════════════════════════════════
# Mode Handlers
# ═══════════════════════════════════════════════════════════════════════════

def _run_live_camera(frame_container) -> None:
    try:
        stream_mgr = get_stream_manager()
        recorder = get_recorder()
        cam_active = st.session_state.cam_active

        if cam_active:
            # Auto-reconnect if background worker died
            if stream_mgr.is_running and (
                (stream_mgr.camera_thread and not stream_mgr.camera_thread.is_alive()) or
                (stream_mgr.inference_worker and not stream_mgr.inference_worker.is_alive())
            ):
                add_event(EventType.ERROR, "Worker thread died unexpectedly; attempting auto-recovery...")
                stream_mgr.start(source=st.session_state.webcam_id, backend=st.session_state.backend)

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
    except Exception as exc:
        frame_container.warning(f"Live camera stream error: {exc}")


def _run_image_mode(frame_container) -> None:
    backend = st.session_state.backend

    # ── Premium glassmorphic drag-and-drop zone ───────────────────────────
    st.markdown(
        """
        <style>
        .img-upload-zone {
            background: rgba(15,23,42,0.85);
            border: 1.5px dashed rgba(56,189,248,0.45);
            border-radius: 12px;
            padding: 16px 20px 14px;
            margin-bottom: 14px;
            transition: all 0.25s ease-in-out;
        }
        .img-upload-zone:hover {
            border-color: #38bdf8;
            background: rgba(20,30,55,0.95);
        }
        .zone-label {
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.70rem;
            letter-spacing: 0.12em;
            color: #38bdf8;
            text-transform: uppercase;
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .kpi-card {
            background: #0d1829;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 12px 14px;
            text-align: center;
        }
        .kpi-val {
            font-family: 'JetBrains Mono', monospace;
            font-size: 1.25rem;
            font-weight: 700;
            margin-top: 2px;
        }
        .kpi-lbl {
            font-size: 0.62rem;
            color: #64748b;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        </style>
        <div class="img-upload-zone">
          <div class="zone-label">⬆ DROP IMAGE OR ZIP ARCHIVE TO RUN INSTANT PERCEPTION</div>
          <div style="font-size:0.72rem; color:#94a3b8; margin-bottom:8px;">
            Supports single road frames (.jpg, .jpeg, .png, .webp) or batch datasets (.zip). Zero button clicks required.
          </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload image or zip",
        type=["jpg", "jpeg", "png", "webp", "zip"],
        key="up_img_or_zip",
        label_visibility="collapsed",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Handle Upload Instantly ───────────────────────────────────────────
    if uploaded:
        is_zip = uploaded.name.lower().endswith(".zip")

        if is_zip:
            # ── ZIP Batch Processing ──────────────────────────────────────
            if st.session_state.get("_last_zip_name") != uploaded.name:
                st.session_state["_last_zip_name"] = uploaded.name

                import io
                import zipfile

                staging_dir = PROJECT_ROOT / "data" / ".staging" / f"zip_{Path(uploaded.name).stem}_{int(time.time())}"
                staging_dir.mkdir(parents=True, exist_ok=True)

                with st.spinner(f"Extracting and verifying {uploaded.name}..."):
                    try:
                        with zipfile.ZipFile(io.BytesIO(uploaded.read())) as zf:
                            # Zip-slip safe path extraction
                            for member in zf.infolist():
                                target = (staging_dir / member.filename).resolve()
                                if not str(target).startswith(str(staging_dir.resolve())):
                                    raise ValueError(f"Zip slip security violation in {member.filename}")
                            zf.extractall(staging_dir)

                        img_exts = {".jpg", ".jpeg", ".png", ".webp"}
                        extracted_imgs = sorted([
                            p for p in staging_dir.rglob("*.*") if p.suffix.lower() in img_exts
                        ])

                        st.session_state["_zip_images"] = [str(p) for p in extracted_imgs]
                        st.session_state["_zip_staging_dir"] = str(staging_dir)

                    except Exception as e:
                        st.error(f"Failed to extract ZIP archive: {e}")
                        st.session_state["_zip_images"] = []

            # Display ZIP Batch Results
            zip_imgs = st.session_state.get("_zip_images", [])
            if zip_imgs:
                st.success(f"📦 Successfully unpacked **{len(zip_imgs)}** road frames from `{uploaded.name}`")

                # Action: Push All to Dataset
                if st.button("🚀 Push All Valid Frames to Training Dataset (data/raw_frames/)", type="primary"):
                    raw_dir = PATHS.raw_frames
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    copied = 0
                    for imp in zip_imgs:
                        shutil.copy2(imp, raw_dir / Path(imp).name)
                        copied += 1
                    st.success(f"✅ Ingested {copied} frames into `{raw_dir.name}/` for annotation & training!")

                # Batch Perception Grid Preview
                st.markdown("##### Batch Perception Previews")
                sample_imgs = zip_imgs[:8]
                cols = st.columns(min(4, max(1, len(sample_imgs))))

                for i, img_path_str in enumerate(sample_imgs):
                    p = Path(img_path_str)
                    res = run_pipeline(p, backend=backend, reset_pd_state=True)
                    acc_pct = getattr(res.prediction, "lane_accuracy_percent", None)
                    if acc_pct is None:
                        acc_pct = int(round(max(res.prediction.left_confidence, res.prediction.right_confidence) * 100))

                    with cols[i % len(cols)]:
                        if res.annotated_bgr is not None:
                            st.image(cv2.cvtColor(res.annotated_bgr, cv2.COLOR_BGR2RGB), use_container_width=True)
                        badge_col = "#10b981" if acc_pct >= 65 else ("#f59e0b" if acc_pct >= 40 else "#ef4444")
                        st.markdown(
                            f"<div style='font-family:monospace; font-size:0.70rem; color:{badge_col}; font-weight:700; text-align:center;'>"
                            f"Accuracy: {acc_pct:.0f}%"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

        else:
            # ── Single Image Instant Perception ───────────────────────────
            if st.session_state.get("_last_img_name") != uploaded.name:
                st.session_state["_last_img_name"] = uploaded.name
                img_bytes = uploaded.read()
                st.session_state["_last_uploaded_bytes"] = img_bytes

                with st.spinner("Running high-precision lane perception..."):
                    res = run_pipeline(img_bytes, backend=backend, reset_pd_state=True)
                    tracker = LKAStateTracker()
                    telem = tracker.update(res)
                _handle_result(res, telem, source="image")

    # ── Hero Viewport & KPI Panel for Single Image ─────────────────────────
    r = st.session_state.last_result
    t = st.session_state.last_telemetry

    if r and r.annotated_bgr is not None:
        # Display annotated viewport
        hud_frame = draw_hud_overlay(r.annotated_bgr, telemetry=t, perf=None, mode_label="IMAGE")
        frame_container.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        # ── KPI Panel (Objective 4) ───────────────────────────────────────
        acc_pct = getattr(r.prediction, "lane_accuracy_percent", None)
        if acc_pct is None:
            acc_pct = int(round(max(r.prediction.left_confidence, r.prediction.right_confidence) * 100))

        if acc_pct >= 70:
            acc_badge = f"{acc_pct:.0f}% (High Confidence)"
            acc_color = "#10b981"
            status_text = "READY FOR ANNOTATION / TRAINING"
            status_color = "#10b981"
        elif acc_pct >= 40:
            acc_badge = f"{acc_pct:.0f}% (Moderate)"
            acc_color = "#f59e0b"
            status_text = "NEEDS REVIEW (MODERATE CONFIDENCE)"
            status_color = "#f59e0b"
        else:
            acc_badge = f"{acc_pct:.0f}% (Low Confidence)"
            acc_color = "#ef4444"
            status_text = "LOW CONFIDENCE (CHECK MARKINGS)"
            status_color = "#ef4444"

        # Metric offset
        if t and t.lateral_offset_m is not None:
            sign = "+" if t.lateral_offset_m >= 0 else ""
            desc = "Centered" if abs(t.lateral_offset_m) < 0.20 else ("Drift Left" if t.lateral_offset_m < 0 else "Drift Right")
            off_str = f"{sign}{t.lateral_offset_m:.2f} m ({desc})"
        else:
            off_str = "-- m"

        k1, k2, k3 = st.columns(3)
        with k1:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-lbl">LANE ACCURACY</div>
                  <div class="kpi-val" style="color:{acc_color};">{acc_badge}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-lbl">METRIC LATERAL OFFSET</div>
                  <div class="kpi-val" style="color:#38bdf8;">{off_str}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k3:
            st.markdown(
                f"""
                <div class="kpi-card">
                  <div class="kpi-lbl">PERCEPTION STATUS</div>
                  <div class="kpi-val" style="color:{status_color}; font-size:0.95rem;">{status_text}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # One-Click Send to Annotated Dataset
        col_act1, col_act2 = st.columns([1.5, 2.5])
        with col_act1:
            if st.button("✅ Send to Annotated Dataset (data/annotated/images/)", type="primary", use_container_width=True):
                ann_img_dir = PATHS.annotated / "images"
                ann_img_dir.mkdir(parents=True, exist_ok=True)
                stem = f"frame_{int(time.time())}"
                dest_file = ann_img_dir / f"{stem}.jpg"

                if "_last_uploaded_bytes" in st.session_state:
                    dest_file.write_bytes(st.session_state["_last_uploaded_bytes"])
                elif r.preprocessed and r.preprocessed.original_bgr is not None:
                    cv2.imwrite(str(dest_file), r.preprocessed.original_bgr)

                st.success(f"Successfully saved clean frame -> `{dest_file.name}`")


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
        render_error_boundary("Image Mode", _run_image_mode, img_frame_container)

    # ── Tab 3: Video Stream ───────────────────────────────────────────────
    with tab_vid:
        vid_frame_container = st.empty()
        render_error_boundary("Video Stream", _run_video_mode, vid_frame_container)

    # ── Tab 4: Session Replay ─────────────────────────────────────────────
    with tab_replay:
        replay_frame_container = st.empty()
        render_error_boundary("Session Replay", _run_replay_mode, replay_frame_container)

    # ═══════════════════════════════════════════════════════════════════════
    # Primary Operational Telemetry (Below Hero Viewport)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        # 1. LKA State & Departure Alert Card
        render_error_boundary("LKA Status", render_lka_status_card, telemetry=telemetry, result=result)

        # 2. Precision Lateral Offset Centerline Gauge
        render_error_boundary("Lateral Offset Gauge", render_lateral_offset_gauge, telemetry=telemetry, result=result)

    with col_right:
        # 3. Perception Cluster
        render_error_boundary("Perception Cluster", render_perception_cluster, telemetry=telemetry, result=result)

        # 4. Performance Cluster
        render_error_boundary("Performance Cluster", render_performance_cluster, perf=perf, result=result)

    # ── Subsystem Health Strip ─────────────────────────────────────────────
    render_error_boundary("Health Strip", render_system_health_strip, camera_active=cam_active, result=result, telemetry=telemetry)

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
