"""
dashboard/pages/03_Recordings.py
=================================
APEX LKA — Session Recording Manager & Playback Inspector.

Inspects recorded ADAS perception sessions, synchronized telemetry logs,
and allows frame-by-frame verification.
Uses existing recorder and SessionReplayer implementation strictly.
"""

import sys
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.recorder import (
    get_recorder,
    list_recorded_sessions,
    SessionReplayer,
)

st.set_page_config(
    page_title="APEX LKA | Recordings",
    page_icon="📼",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
[data-testid="stAppViewContainer"] {
    background-color: #070d15;
    color: #e2e8f0;
    font-family: 'Inter', sans-serif;
}
.page-title {
    font-size: 1.3rem; font-weight: 800; letter-spacing: 0.15em;
    color: #e2e8f0; text-transform: uppercase; margin-bottom: 2px;
}
.page-sub {
    font-size: 0.65rem; color: #4a6178; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 18px;
}
.metric-card {
    background: #0c1524; border: 1px solid #1a2d42; border-radius: 6px;
    padding: 12px 16px; margin-bottom: 8px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem;
    color: #4a6178; letter-spacing: 0.12em; text-transform: uppercase;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace; font-size: 1.1rem;
    font-weight: 700; color: #00b4cc; margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-title">APEX LKA &mdash; <span style="color:#00b4cc;">SESSION RECORDINGS</span></div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Synchronized Camera Frame &amp; ADAS Telemetry Archive</div>', unsafe_allow_html=True)

recorder = get_recorder()

# ── Active Recording Status ────────────────────────────────────────────────
if recorder.is_recording:
    st.error(f"🔴 RECORDING ACTIVE: {recorder.frames_recorded} frames captured | {recorder.disk_usage_mb} MB ({recorder.elapsed_seconds}s)")
    if st.button("⏹ STOP RECORDING", type="primary"):
        recorder.stop()
        st.rerun()
    st.markdown("---")

sessions = list_recorded_sessions()

if not sessions:
    st.info("No recorded sessions found in `data/recordings/`. Start a recording session from the main dashboard.")
else:
    # ── Sessions Overview Table ───────────────────────────────────────────
    st.subheader("📁 Saved Sessions Archive")

    df_sessions = pd.DataFrame([
        {
            "Session ID": s["id"],
            "Frames": s["frames"],
            "Size (MB)": s["size_mb"],
            "Created": s["created"],
            "Telemetry Log": "✅ Available" if s["has_csv"] else "❌ Missing",
        }
        for s in sessions
    ])
    st.dataframe(df_sessions, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Session Inspector & Replay ─────────────────────────────────────────
    st.subheader("🔍 Session Inspector & Telemetry Log")
    sess_map = {f"{s['id']}  ({s['frames']} frames, {s['size_mb']} MB)": s for s in sessions}
    chosen_label = st.selectbox("Select Session to Inspect", list(sess_map.keys()))
    chosen_sess = sess_map[chosen_label]
    sess_path = chosen_sess["path"]

    csv_path = sess_path / "telemetry.csv"
    col_meta1, col_meta2 = st.columns([1, 1])

    with col_meta1:
        st.caption("FRAME VIEWER")
        replayer = SessionReplayer(sess_path)
        if replayer.total_frames > 0:
            frame_idx = st.slider("Frame Index", 0, replayer.total_frames - 1, 0)
            frame_bgr = replayer.get_frame(frame_idx)
            if frame_bgr is not None:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, use_container_width=True)
        else:
            st.warning("No frames recorded in this session folder.")

    with col_meta2:
        st.caption("SYNCHRONOUS TELEMETRY LOG")
        if csv_path.exists():
            try:
                df_telem = pd.read_csv(csv_path)
                st.dataframe(df_telem.head(100), use_container_width=True, height=360)
                st.caption(f"Showing first {min(len(df_telem), 100)} of {len(df_telem)} rows from `{csv_path.name}`")
            except Exception as e:
                st.error(f"Cannot load CSV: {e}")
        else:
            st.info("No telemetry.csv file present in this session.")
