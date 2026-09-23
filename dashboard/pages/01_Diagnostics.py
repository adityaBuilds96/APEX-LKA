"""
dashboard/pages/05_Diagnostics.py
=================================
APEX LKA — ADAS Engineering Diagnostics Console.

Real-time telemetry analytics, latency profiling, frame drop tracking,
and safety state event counters. All data is calculated from active runtime sessions.
"""

import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import psutil
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.camera_stream import get_stream_manager
from dashboard.components.event_log import render_event_log
from dashboard.components.pipeline_status import render_pipeline_status
from dashboard.components.offset_chart import render_offset_chart
from dashboard.components.steering_indicator import render_steering_indicator



st.set_page_config(
    page_title="APEX LKA | Diagnostics",
    page_icon="🔬",
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
.diag-title {
    font-size: 1.3rem; font-weight: 800; letter-spacing: 0.15em;
    color: #e2e8f0; text-transform: uppercase; margin-bottom: 2px;
}
.diag-sub {
    font-size: 0.65rem; color: #4a6178; letter-spacing: 0.15em;
    text-transform: uppercase; margin-bottom: 18px;
}
.metric-card {
    background: #0c1524; border: 1px solid #1a2d42; border-radius: 6px;
    padding: 14px 18px; display: flex; flex-direction: column; gap: 4px;
}
.metric-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.60rem;
    color: #4a6178; letter-spacing: 0.12em; text-transform: uppercase;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace; font-size: 1.4rem;
    font-weight: 700; color: #00b4cc;
}
.metric-sub {
    font-family: 'JetBrains Mono', monospace; font-size: 0.58rem; color: #8a9ab0;
}
</style>
""", unsafe_allow_html=True)


st.markdown('<div class="diag-title">APEX LKA &mdash; <span style="color:#00b4cc;">ENGINEERING DIAGNOSTICS</span></div>', unsafe_allow_html=True)
st.markdown('<div class="diag-sub">Real-Time Perception Performance &amp; Hardware Telemetry</div>', unsafe_allow_html=True)

stream_mgr = get_stream_manager()
res, telem, perf = stream_mgr.get_latest()

# ── Summary KPI Cards ──────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    fps_val = perf.camera_fps if perf.camera_fps > 0 else perf.ui_fps
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">STREAM FRAMERATE</div>
      <div class="metric-val">{fps_val:.1f} <span style="font-size:0.7rem; color:#4a6178;">FPS</span></div>
      <div class="metric-sub">UI Loop: {perf.ui_fps:.1f} FPS</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">TOTAL LATENCY</div>
      <div class="metric-val">{perf.pipeline_latency_ms:.0f} <span style="font-size:0.7rem; color:#4a6178;">ms</span></div>
      <div class="metric-sub">Inference: {perf.inference_latency_ms:.0f} ms</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    drop_pct = (perf.dropped_frames / max(perf.total_captured_frames, 1)) * 100
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">DROPPED FRAMES</div>
      <div class="metric-val" style="color:{'#10b981' if perf.dropped_frames == 0 else '#f59e0b'};">
        {perf.dropped_frames}
      </div>
      <div class="metric-sub">Drop Rate: {drop_pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    tot_frames = perf.total_inferred_frames
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">FRAMES PROCESSED</div>
      <div class="metric-val">{tot_frames}</div>
      <div class="metric-sub">Captured: {perf.total_captured_frames}</div>
    </div>
    """, unsafe_allow_html=True)

with c5:
    cpu_pct = perf.cpu_usage_pct
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-label">HARDWARE LOAD</div>
      <div class="metric-val" style="color:{'#10b981' if cpu_pct < 75 else '#ef4444'};">
        {cpu_pct:.0f}%
      </div>
      <div class="metric-sub">RAM: {perf.ram_usage_pct:.0f}% | GPU: CPU MODE</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Latency Breakdown & Detection Health ────────────────────────────────────
col_left, col_right = st.columns([1, 1])

with col_left:
    st.subheader("⚡ Pipeline Latency Breakdown")
    if res and res.timings:
        t = res.timings
        timing_data = [
            {"Stage": "1. Preprocess",  "Latency (ms)": t.get("preprocess_ms", 0.0)},
            {"Stage": "2. ML Inference", "Latency (ms)": t.get("inference_ms", 0.0)},
            {"Stage": "3. Postprocess", "Latency (ms)": t.get("postprocess_ms", 0.0)},
            {"Stage": "4. Lane Geometry", "Latency (ms)": t.get("geometry_ms", 0.0)},
            {"Stage": "5. Offset & LKA", "Latency (ms)": t.get("offset_ms", 0.0)},
            {"Stage": "6. Overlay Viz", "Latency (ms)": t.get("viz_ms", 0.0)},
        ]
        df_time = pd.DataFrame(timing_data)
        chart = (
            alt.Chart(df_time)
            .mark_bar(color="#00b4cc", cornerRadiusEnd=3)
            .encode(
                x=alt.X("Latency (ms):Q", title="Latency (ms)"),
                y=alt.Y("Stage:N", sort=None, title=None),
                tooltip=["Stage", "Latency (ms)"]
            )
            .properties(height=240, background="#0c1524")
            .configure_axis(labelColor="#8a9ab0", titleColor="#4a6178", gridColor="#1a2d42")
        )
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No active pipeline execution. Start camera or process an image.")

with col_right:
    st.subheader("🛡 ADAS Safety & Stability State")
    stab_val = telem.stability_state.value if telem else "LANE LOST"
    ldw_val = telem.ldw_state.value if telem else "LANE LOST"
    fail_val = telem.failure_condition.value if telem else "NONE"
    curv_val = f"{telem.curvature_radius_m:.0f} m" if (telem and telem.curvature_radius_m) else "STRAIGHT / UNKNOWN"
    width_val = f"{telem.lane_width_m:.2f} m" if (telem and telem.lane_width_m) else "N/A"

    df_health = pd.DataFrame([
        {"Metric": "Perception Stability", "Status": stab_val},
        {"Metric": "LDW Departure State", "Status": ldw_val},
        {"Metric": "Edge-Case Condition", "Status": fail_val},
        {"Metric": "Tracked Lane Width", "Status": width_val},
        {"Metric": "Curve Radius", "Status": curv_val},
        {"Metric": "Consecutive Detected", "Status": f"{telem.consecutive_detected_frames if telem else 0} frames"},
        {"Metric": "Consecutive Lost", "Status": f"{telem.consecutive_lost_frames if telem else 0} frames"},
    ])
    st.dataframe(df_health, use_container_width=True, hide_index=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Pipeline Architecture & Offset History ────────────────────────────────
c_diag_left, c_diag_right = st.columns([1, 1])

with c_diag_left:
    st.subheader("🔄 Pipeline Stage Flow")
    render_pipeline_status(res, camera_active=stream_mgr.is_running)

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🧭 Software Steering Recommendation")
    render_steering_indicator(res.offset if res else None, telemetry=telem)

with c_diag_right:
    st.subheader("📈 Lateral Offset History")
    render_offset_chart()

st.markdown("<br>", unsafe_allow_html=True)

# ── Event Log ──────────────────────────────────────────────────────────────
st.subheader("📜 System Event Stream")
render_event_log(max_rows=25)

