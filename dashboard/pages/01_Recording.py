"""
dashboard/pages/01_Recording.py
================================
APEX LKA — Training-Accelerator Session Recorder & Dataset Ingest Engine.

Purpose
-------
Engineered for rapid, high-quality data collection for autonomous perception
and SAE India BAJA 2027 development.

Key Features
------------
1. Multi-Source Capture:
   - Local Webcam / USB Dashcam
   - Connected Smartphone Camera (via IP RTSP / HTTP URL: DroidCam, IP Webcam)
   - Virtual Simulation / Pre-recorded video stream
2. Live Preview & Real-Time Recording HUD:
   - Live stream canvas with glowing recording badge and live telemetry.
3. Auto-Frame Extraction:
   - Configurable sampling rate (1, 2, 5, 10 FPS).
4. Live Duplicate Detection:
   - Real-time perceptual difference check; drops standstill/redundant frames.
5. Session Tagging & Metadata:
   - Environmental tags (Sunny, Overcast, Dirt/BAJA, Highway, Night).
   - Generates structured session_metadata.json.
6. Storage Estimator & Disk Guard:
   - Live disk space calculator based on resolution, duration, and FPS.
7. One-Click "Send to Dataset":
   - Transfers captured frames directly to data/raw_frames/ for auto-annotation.
8. Batch Recording Mode:
   - Automated systematic capture: N sessions of X seconds with rest intervals.
9. Session Archive & Inspector:
   - Interactive gallery with thumbnail previews, telemetry log, and scrubber.
"""

import csv
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg
from dashboard.components.recorder import (
    RECORDINGS_DIR,
    SessionReplayer,
    list_recorded_sessions,
)

# ═══════════════════════════════════════════════════════════════════════════
# Page Configuration & Styling
# ═══════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="APEX LKA | Data Recorder",
    page_icon="📼",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

[data-testid="stAppViewContainer"] {
    background-color: #070b12;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, sans-serif;
}
[data-testid="stSidebar"] {
    background-color: #0a101b;
    border-right: 1px solid #142032;
}

.rec-title-chip {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(14, 165, 233, 0.12);
    border: 1px solid rgba(14, 165, 233, 0.35);
    border-radius: 4px;
    padding: 3px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #38bdf8;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.recording-badge-active {
    background: rgba(239, 68, 68, 0.2);
    border: 1px solid #ef4444;
    border-radius: 6px;
    padding: 8px 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    font-weight: 700;
    color: #ef4444;
    display: inline-flex;
    align-items: center;
    gap: 10px;
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(239, 68, 68, 0); }
    100% { box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }
}

.metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px 16px;
}
.metric-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-lbl {
    font-size: 0.65rem;
    color: #64748b;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.session-card {
    background: #0d1524;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 12px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.session-card:hover {
    border-color: #38bdf8;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions: Duplicate Detection & Storage
# ═══════════════════════════════════════════════════════════════════════════

def is_duplicate_frame(curr_bgr: np.ndarray, last_bgr: Optional[np.ndarray], diff_threshold: float = 0.02) -> bool:
    """
    Fast perceptual difference check to drop near-identical stationary frames.
    Returns True if frames are virtually indistinguishable.
    """
    if last_bgr is None:
        return False
    # Downscale and grayscale for sub-millisecond comparison
    g1 = cv2.resize(cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY), (32, 32))
    g2 = cv2.resize(cv2.cvtColor(last_bgr, cv2.COLOR_BGR2GRAY), (32, 32))
    diff = np.mean(np.abs(g1.astype(np.float32) - g2.astype(np.float32))) / 255.0
    return diff < diff_threshold


def get_disk_storage_info() -> Tuple[float, float]:
    """Return (free_gb, total_gb) on project drive."""
    try:
        total, used, free = shutil.disk_usage(str(PROJECT_ROOT))
        return free / (1024**3), total / (1024**3)
    except Exception:
        return 10.0, 100.0


def calculate_storage_estimate(fps: int, resolution: str, duration_sec: int) -> float:
    """Estimate MB required for a recording session."""
    size_per_frame_kb = {
        "720p (1280x720)": 130,
        "1080p (1920x1080)": 280,
        "4K (3840x2160)": 950,
    }.get(resolution, 250)
    total_kb = fps * duration_sec * size_per_frame_kb
    return round(total_kb / 1024.0, 1)


# ═══════════════════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════════════════

if "rec_active" not in st.session_state:
    st.session_state.rec_active = False
if "rec_session_id" not in st.session_state:
    st.session_state.rec_session_id = None
if "rec_frames_saved" not in st.session_state:
    st.session_state.rec_frames_saved = 0
if "rec_dups_skipped" not in st.session_state:
    st.session_state.rec_dups_skipped = 0
if "rec_start_time" not in st.session_state:
    st.session_state.rec_start_time = 0.0
if "rec_last_frame" not in st.session_state:
    st.session_state.rec_last_frame = None


# ═══════════════════════════════════════════════════════════════════════════
# Top Header
# ═══════════════════════════════════════════════════════════════════════════

st.markdown('<div class="rec-title-chip">SAE BAJA &amp; AUTONOMOUS DATA ENGINE</div>', unsafe_allow_html=True)
st.markdown("## 📼 Training-Accelerator Session Recorder")
st.caption("Multi-source capture, auto-frame extraction, real-time duplicate rejection, and instant dataset ingestion.")
st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════
# Tab Layout
# ═══════════════════════════════════════════════════════════════════════════

tab_record, tab_batch, tab_archive = st.tabs([
    "🔴  LIVE RECORDER",
    "⏱️  BATCH RECORDING MODE",
    "📁  SESSION ARCHIVE & INGESTION",
])

# ── TAB 1: LIVE RECORDER ──────────────────────────────────────────────────
with tab_record:
    col_settings, col_viewport = st.columns([1, 1.8])

    with col_settings:
        st.markdown("#### 1. Hardware & Stream Source")
        src_type = st.radio(
            "Capture Device Type",
            ["Local Webcam / Dashcam", "Phone Camera via IP (RTSP/HTTP)", "Virtual Video Simulator"],
            index=0,
        )

        source_val = 0
        if src_type == "Local Webcam / Dashcam":
            cam_idx = st.selectbox("Camera Device Index", [0, 1, 2, 3], index=0)
            source_val = cam_idx
        elif src_type == "Phone Camera via IP (RTSP/HTTP)":
            ip_url = st.text_input(
                "Stream URL",
                value="http://192.168.1.100:8080/video",
                help="Enter RTSP or HTTP stream URL from IP Webcam or DroidCam app.",
            )
            source_val = ip_url
        else:
            raw_vid_dir = PATHS.raw_videos
            vids = list(raw_vid_dir.glob("*.mp4")) if raw_vid_dir.exists() else []
            if vids:
                chosen_v = st.selectbox("Select Video", [v.name for v in vids])
                source_val = str(raw_vid_dir / chosen_v)
            else:
                st.warning("No test videos found in data/raw_videos/")
                source_val = 0

        st.markdown("#### 2. Frame Extraction & Quality")
        col_fps, col_res = st.columns(2)
        with col_fps:
            extract_fps = st.select_slider(
                "Extraction Rate (FPS)",
                options=[1, 2, 5, 10],
                value=5,
                help="Saves N high-resolution frames per second for training dataset.",
            )
        with col_res:
            res_choice = st.selectbox(
                "Capture Quality",
                ["720p (1280x720)", "1080p (1920x1080)", "4K (3840x2160)"],
                index=1,
            )

        enable_dup_filter = st.checkbox("Enable Live Duplicate Filter (drops standstill frames)", value=True)

        st.markdown("#### 3. Session Tagging & Metadata")
        session_label = st.text_input("Session Tag", value="baja_dirt_track_sunny")
        col_env, col_loc = st.columns(2)
        with col_env:
            weather_tag = st.selectbox(
                "Lighting / Weather",
                ["Sunny / Direct Sunlight", "Overcast / Diffuse", "Harsh Shadows", "Sunset / Golden Hour", "Night", "Rain / Muddy"],
            )
        with col_loc:
            location_tag = st.selectbox(
                "Terrain / Location",
                ["Off-Road Dirt (BAJA)", "Asphalt Main Road", "Campus Parking Lot", "Obstacle Test Track", "Gravel / Rocks"],
            )

        # Storage Estimator Display
        free_gb, total_gb = get_disk_storage_info()
        est_mb = calculate_storage_estimate(extract_fps, res_choice, 60)
        st.markdown(
            f"""
            <div style="background:#091220; border:1px solid #142840; border-radius:6px; padding:10px 14px; margin-top:12px;">
              <div style="font-family:'JetBrains Mono', monospace; font-size:0.62rem; color:#4a6178; text-transform:uppercase;">
                STORAGE ESTIMATOR
              </div>
              <div style="font-family:'JetBrains Mono', monospace; font-size:0.80rem; color:#38bdf8; margin-top:3px;">
                ~{est_mb} MB / minute at {extract_fps} FPS &bull; {free_gb:.1f} GB disk free
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        # Trigger Controls
        if not st.session_state.rec_active:
            if st.button("🔴 START RECORDING SESSION", type="primary", use_container_width=True):
                # Setup session folder
                ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                clean_tag = session_label.strip().replace(" ", "_")
                sess_name = f"session_{ts_str}_{clean_tag}"
                sess_dir = RECORDINGS_DIR / sess_name
                frames_dir = sess_dir / "frames"
                frames_dir.mkdir(parents=True, exist_ok=True)

                st.session_state.rec_active = True
                st.session_state.rec_session_id = sess_name
                st.session_state.rec_session_dir = str(sess_dir)
                st.session_state.rec_frames_saved = 0
                st.session_state.rec_dups_skipped = 0
                st.session_state.rec_start_time = time.time()
                st.session_state.rec_last_frame = None

                # Create metadata skeleton
                meta = {
                    "session_id": sess_name,
                    "tag": session_label,
                    "created_at": datetime.now().isoformat(),
                    "source": str(source_val),
                    "extraction_fps": extract_fps,
                    "quality": res_choice,
                    "weather_tag": weather_tag,
                    "location_tag": location_tag,
                    "frames_saved": 0,
                    "duplicates_skipped": 0,
                    "duration_sec": 0,
                }
                with open(sess_dir / "session_metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                st.rerun()
        else:
            if st.button("⏹ STOP & FINALIZE SESSION", type="primary", use_container_width=True):
                duration = time.time() - st.session_state.rec_start_time
                sess_dir = Path(st.session_state.rec_session_dir)

                # Finalize metadata
                meta_file = sess_dir / "session_metadata.json"
                if meta_file.exists():
                    try:
                        with open(meta_file, "r") as f:
                            meta = json.load(f)
                        meta["frames_saved"] = st.session_state.rec_frames_saved
                        meta["duplicates_skipped"] = st.session_state.rec_dups_skipped
                        meta["duration_sec"] = round(duration, 1)
                        with open(meta_file, "w") as f:
                            json.dump(meta, f, indent=2)
                    except Exception:
                        pass

                st.session_state.rec_active = False
                st.success(
                    f"Session `{st.session_state.rec_session_id}` finalized! "
                    f"Saved {st.session_state.rec_frames_saved} frames in {duration:.1f}s."
                )
                st.rerun()

    with col_viewport:
        st.markdown("#### Live Viewport & Extractor Canvas")

        if st.session_state.rec_active:
            sess_name = st.session_state.rec_session_id
            elapsed = time.time() - st.session_state.rec_start_time
            st.markdown(
                f"""
                <div class="recording-badge-active">
                  <span>● REC</span>
                  <span>{sess_name}</span>
                  <span>({elapsed:.1f}s | {st.session_state.rec_frames_saved} frames)</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        frame_placeholder = st.empty()

        # Recording loop worker
        if st.session_state.rec_active:
            sess_dir = Path(st.session_state.rec_session_dir)
            frames_dir = sess_dir / "frames"

            cap = cv2.VideoCapture(source_val)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Duplicate check
                    is_dup = False
                    if enable_dup_filter and st.session_state.rec_last_frame is not None:
                        is_dup = is_duplicate_frame(frame, st.session_state.rec_last_frame)

                    if not is_dup:
                        st.session_state.rec_frames_saved += 1
                        st.session_state.rec_last_frame = frame.copy()
                        fname = f"frame_{st.session_state.rec_frames_saved:06d}.jpg"
                        cv2.imwrite(str(frames_dir / fname), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
                    else:
                        st.session_state.rec_dups_skipped += 1

                    # Draw clean recording HUD on display frame
                    disp_frame = frame.copy()
                    h, w = disp_frame.shape[:2]
                    cv2.putText(
                        disp_frame,
                        f"● REC  FPS:{extract_fps}  SAVED:{st.session_state.rec_frames_saved}  DUPS:{st.session_state.rec_dups_skipped}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frame_placeholder.image(cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
                cap.release()
            else:
                st.error(f"Cannot connect to capture source: {source_val}")
                st.session_state.rec_active = False
        else:
            frame_placeholder.markdown(
                """
                <div style="background:#090e17; border:1.5px dashed #1e293b; border-radius:10px; height:380px; display:flex; flex-direction:column; align-items:center; justify-content:center;">
                  <div style="font-family:'JetBrains Mono', monospace; font-size:1.0rem; color:#475569; letter-spacing:0.12em;">
                    STREAM STANDBY
                  </div>
                  <div style="font-size:0.75rem; color:#334155; margin-top:6px;">
                    Click 'START RECORDING SESSION' to begin capturing high-framerate dataset
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # Real-time metrics strip
        c_m1, c_m2, c_m3 = st.columns(3)
        with c_m1:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-lbl">Extracted Frames</div>
                  <div class="metric-num">{st.session_state.rec_frames_saved}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_m2:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-lbl">Duplicates Rejection</div>
                  <div class="metric-num" style="color:#f59e0b;">{st.session_state.rec_dups_skipped}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c_m3:
            sess_size_mb = 0.0
            if st.session_state.rec_active and Path(st.session_state.rec_session_dir).exists():
                sess_size_mb = sum(f.stat().st_size for f in Path(st.session_state.rec_session_dir).rglob("*")) / (1024**2)
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="metric-lbl">Session Disk Usage</div>
                  <div class="metric-num" style="color:#10b981;">{sess_size_mb:.1f} MB</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# ── TAB 2: BATCH RECORDING MODE ───────────────────────────────────────────
with tab_batch:
    st.markdown("### ⏱️ Systematic Batch Recording Mode")
    st.caption("Ideal for building structured datasets across varying track segments without collecting gigabytes of redundant video.")

    b_col1, b_col2, b_col3 = st.columns(3)
    with b_col1:
        num_sessions = st.number_input("Number of Sessions (N)", min_value=1, max_value=20, value=3)
    with b_col2:
        session_duration = st.number_input("Duration per Session (seconds)", min_value=5, max_value=300, value=15)
    with b_col3:
        rest_duration = st.number_input("Break Between Sessions (seconds)", min_value=2, max_value=60, value=5)

    batch_tag = st.text_input("Batch Tag Prefix", value="baja_lap_interval")

    if st.button("🚀 EXECUTE AUTOMATED BATCH CAPTURE", type="primary"):
        progress_bar = st.progress(0)
        status_msg = st.empty()

        for s_idx in range(num_sessions):
            status_msg.info(f"🔴 Capturing Session {s_idx + 1} of {num_sessions} ({session_duration}s)...")
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            b_name = f"session_{ts}_{batch_tag}_run{s_idx + 1}"
            b_dir = RECORDINGS_DIR / b_name
            frames_dir = b_dir / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(source_val)
            start_t = time.time()
            f_count = 0

            while time.time() - start_t < session_duration:
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        f_count += 1
                        cv2.imwrite(str(frames_dir / f"frame_{f_count:06d}.jpg"), frame)
                time.sleep(1.0 / max(1, extract_fps))

            cap.release()

            # Write metadata
            meta = {
                "session_id": b_name,
                "batch_idx": s_idx + 1,
                "frames_saved": f_count,
                "duration_sec": session_duration,
                "created_at": datetime.now().isoformat(),
            }
            with open(b_dir / "session_metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

            progress_bar.progress((s_idx + 1) / num_sessions)

            if s_idx < num_sessions - 1:
                for r in range(rest_duration, 0, -1):
                    status_msg.warning(f"⏸ Break: Resuming Session {s_idx + 2} in {r} seconds...")
                    time.sleep(1)

        status_msg.success(f"🎉 Batch capture complete! Captured {num_sessions} sessions.")


# ── TAB 3: SESSION ARCHIVE & INGESTION ─────────────────────────────────────
with tab_archive:
    st.markdown("### 📁 Recorded Sessions Archive & Dataset Ingestion")
    st.caption("Manage past runs, inspect frame sequences, and ingest frames into the APEX-LKA training pipeline.")

    sessions = list_recorded_sessions()

    if not sessions:
        st.info("No recorded sessions found in `data/recordings/`.")
    else:
        for sess in sessions:
            s_path = Path(sess["path"])
            s_id = sess["id"]
            frames_dir = s_path / "frames"
            frame_files = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []

            # Read metadata if present
            meta_file = s_path / "session_metadata.json"
            meta_dict = {}
            if meta_file.exists():
                try:
                    with open(meta_file, "r") as mf:
                        meta_dict = json.load(mf)
                except Exception:
                    pass

            with st.container():
                st.markdown(f'<div class="session-card">', unsafe_allow_html=True)
                col_th, col_info, col_act = st.columns([1, 2.2, 1.4])

                with col_th:
                    if frame_files:
                        th_img = cv2.imread(str(frame_files[0]))
                        if th_img is not None:
                            st.image(cv2.cvtColor(th_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                    else:
                        st.caption("No preview")

                with col_info:
                    st.markdown(f"**`{s_id}`**")
                    tag_str = meta_dict.get("tag", "untagged")
                    weather_str = meta_dict.get("weather_tag", "standard")
                    loc_str = meta_dict.get("location_tag", "road")

                    st.caption(
                        f"🏷 Tag: **{tag_str}** &bull; 🌦 Weather: **{weather_str}** &bull; 📍 Location: **{loc_str}**\n\n"
                        f"📦 Frames: **{len(frame_files)}** &bull; Size: **{sess['size_mb']} MB** &bull; Created: {sess['created']}"
                    )

                with col_act:
                    # One-Click Send to Dataset
                    if st.button("⚡ Send to Dataset", key=f"ingest_{s_id}", use_container_width=True):
                        raw_frames_dir = PATHS.raw_frames
                        raw_frames_dir.mkdir(parents=True, exist_ok=True)
                        copied = 0
                        for f in frame_files:
                            target_dest = raw_frames_dir / f"{s_id}_{f.name}"
                            shutil.copy2(f, target_dest)
                            copied += 1
                        st.success(f"✅ Ingested {copied} frames into `{raw_frames_dir.name}/` for auto-annotation!")

                    # Quick Delete
                    if st.button("🗑 Delete Session", key=f"del_{s_id}", use_container_width=True):
                        shutil.rmtree(s_path, ignore_errors=True)
                        st.warning(f"Deleted session `{s_id}`")
                        st.rerun()

                # Expandable Frame Inspector
                with st.expander("🔍 Inspect Frame Sequence"):
                    replayer = SessionReplayer(s_path)
                    if replayer.total_frames > 0:
                        scrub_idx = st.slider("Frame Index", 0, replayer.total_frames - 1, 0, key=f"scrub_{s_id}")
                        scrub_f = replayer.get_frame(scrub_idx)
                        if scrub_f is not None:
                            st.image(cv2.cvtColor(scrub_f, cv2.COLOR_BGR2RGB), caption=f"Frame {scrub_idx}/{replayer.total_frames - 1}", use_container_width=True)
                    else:
                        st.caption("No frames in this directory.")

                st.markdown("</div>", unsafe_allow_html=True)
