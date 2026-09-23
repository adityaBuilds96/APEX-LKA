"""
dashboard/pages/02_Perception.py
=================================
APEX LKA — Vision Perception & Lane Geometry Deep-Dive Workbench.

Inspects intermediate perception stages directly from InferenceResult:
- Raw Camera Input vs. Annotated Perception Frame
- Fitted Lane Polynomial Coefficients & Curvature
- Lane Coordinate Geometry & Normalized Deviation Analysis
- Intermediate Prediction Boundaries
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.camera_stream import get_stream_manager
from src.inference.pipeline import InferenceResult
from src.lka_engine.lka_state import LKATelemetry

st.set_page_config(
    page_title="APEX LKA | Perception Deep-Dive",
    page_icon="👁",
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

st.markdown('<div class="page-title">APEX LKA &mdash; <span style="color:#00b4cc;">PERCEPTION DEEP-DIVE</span></div>', unsafe_allow_html=True)
st.markdown('<div class="page-sub">Computer Vision Pipeline, Polynomial Curve Fitting &amp; Geometric Analysis</div>', unsafe_allow_html=True)

stream_mgr = get_stream_manager()
res, telem, perf = stream_mgr.get_latest()

# Fallback to session state if stream manager not active
if res is None and "last_result" in st.session_state:
    res = st.session_state.last_result
if telem is None and "last_telemetry" in st.session_state:
    telem = st.session_state.last_telemetry

if res is None:
    st.info("No active perception data available. Start the camera or run an image detection on the main dashboard.")
else:
    # ── Viewport Side-by-Side: Raw Frame vs Annotated ─────────────────────
    st.subheader("🖼 Perception Viewports")
    c1, c2 = st.columns(2)

    with c1:
        st.caption("RAW CAMERA INPUT (SOURCE)")
        if res.preprocessed and res.preprocessed.original_bgr is not None:
            raw_rgb = cv2.cvtColor(res.preprocessed.original_bgr, cv2.COLOR_BGR2RGB)
            st.image(raw_rgb, use_container_width=True)
        else:
            st.markdown('<div class="metric-card">Raw frame buffer not available</div>', unsafe_allow_html=True)

    with c2:
        st.caption("PERCEPTION OVERLAY & LANE CORRIDOR")
        if res.annotated_bgr is not None:
            annot_rgb = cv2.cvtColor(res.annotated_bgr, cv2.COLOR_BGR2RGB)
            st.image(annot_rgb, use_container_width=True)
        else:
            st.markdown('<div class="metric-card">Annotated frame not available</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Lane Geometry & Polynomial Curves ─────────────────────────────────
    st.subheader("📐 Lane Polynomial Geometry")
    col_geom1, col_geom2 = st.columns([1, 1])

    with col_geom1:
        st.caption("FITTED 2ND-ORDER POLYNOMIALS (x = a·y² + b·y + c)")
        geom = res.geometry
        if geom:
            l_poly = geom.left_poly
            r_poly = geom.right_poly

            l_str = f"{l_poly[0]:+.2e} y² {l_poly[1]:+.3f} y {l_poly[2]:+.1f}" if l_poly is not None else "NOT DETECTED / FITTED"
            r_str = f"{r_poly[0]:+.2e} y² {r_poly[1]:+.3f} y {r_poly[2]:+.1f}" if r_poly is not None else "NOT DETECTED / FITTED"

            poly_df = pd.DataFrame([
                {"Boundary": "Left Lane", "Polynomial Equation": l_str, "Detected": geom.left_detected},
                {"Boundary": "Right Lane", "Polynomial Equation": r_str, "Detected": geom.right_detected},
            ])
            st.dataframe(poly_df, use_container_width=True, hide_index=True)
        else:
            st.info("Lane geometry fitting result not available for this frame.")

    with col_geom2:
        st.caption("METRIC SPACE GEOMETRY & CURVATURE")
        if telem:
            m1, m2, m3 = st.columns(3)
            with m1:
                curv_txt = f"{telem.curvature_radius_m:.0f} m" if telem.curvature_radius_m else "Straight"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Curvature Radius</div><div class="metric-val">{curv_txt}</div></div>', unsafe_allow_html=True)
            with m2:
                width_txt = f"{telem.lane_width_m:.2f} m" if telem.lane_width_m else "--"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Lane Width (m)</div><div class="metric-val">{width_txt}</div></div>', unsafe_allow_html=True)
            with m3:
                head_txt = f"{telem.heading_error_deg:+.1f}°" if telem.heading_error_deg is not None else "--"
                st.markdown(f'<div class="metric-card"><div class="metric-label">Heading Error</div><div class="metric-val">{head_txt}</div></div>', unsafe_allow_html=True)

    # ── Pixel Coordinate Geometry Table ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("🎯 Centerline & Offset Telemetry")
    if telem:
        df_coords = pd.DataFrame([
            {"Parameter": "Vehicle Center X (Model Space)", "Value": f"{telem.vehicle_center_x:.1f} px"},
            {"Parameter": "Lane Center X (Estimated)", "Value": f"{telem.lane_center_x:.1f} px" if telem.lane_center_x else "--"},
            {"Parameter": "Lateral Error (Pixels)", "Value": f"{telem.lateral_offset_px:+.1f} px" if telem.lateral_offset_px else "--"},
            {"Parameter": "Lateral Error (Meters)", "Value": f"{telem.lateral_offset_m:+.3f} m" if telem.lateral_offset_m else "--"},
            {"Parameter": "Normalized Deviation [-1, +1]", "Value": f"{telem.lateral_offset_norm:+.4f}" if telem.lateral_offset_norm else "--"},
            {"Parameter": "Smoothed Normalized Deviation", "Value": f"{telem.smoothed_offset_norm:+.4f}" if telem.smoothed_offset_norm else "--"},
            {"Parameter": "Combined Confidence", "Value": f"{telem.combined_confidence * 100:.1f}%"},
        ])
        st.dataframe(df_coords, use_container_width=True, hide_index=True)
