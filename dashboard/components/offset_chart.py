"""
dashboard/components/offset_chart.py
=======================================
Lateral offset vs. time chart for APEX LKA.

Data source: actual OffsetResult.lateral_error_norm from the inference pipeline.
Never generates fake data — displays "WAITING FOR DATA" if history is empty.

Usage
-----
    from dashboard.components.offset_chart import add_offset_sample, render_offset_chart

    add_offset_sample(result.offset)   # call after each inference
    render_offset_chart()
"""

from pathlib import Path
from typing import Optional

import streamlit as st

MAX_SAMPLES = 200   # rolling window of frames


# ═══════════════════════════════════════════════════════════════════════════
# Session-state history
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_history() -> None:
    if "apex_offset_history" not in st.session_state:
        st.session_state.apex_offset_history = []
    if "apex_offset_frame_idx" not in st.session_state:
        st.session_state.apex_offset_frame_idx = 0


def add_offset_sample(offset_result) -> None:
    """
    Add one real lateral_error_norm sample.
    Pass the OffsetResult object from run_pipeline().
    Silently skips if offset_result is None or no error available.
    """
    _ensure_history()
    if offset_result is None:
        return
    val = offset_result.lateral_error_norm
    if val is None:
        return

    idx = st.session_state.apex_offset_frame_idx
    st.session_state.apex_offset_history.append({"frame": idx, "offset": round(float(val), 4)})
    st.session_state.apex_offset_frame_idx = idx + 1

    if len(st.session_state.apex_offset_history) > MAX_SAMPLES:
        st.session_state.apex_offset_history = (
            st.session_state.apex_offset_history[-MAX_SAMPLES:]
        )


def clear_offset_history() -> None:
    st.session_state.apex_offset_history = []
    st.session_state.apex_offset_frame_idx = 0


def render_offset_chart() -> None:
    """
    Render the lateral offset vs. time chart using Altair.
    Only uses real data from session_state.
    """
    _ensure_history()
    history = st.session_state.apex_offset_history

    if not history:
        st.markdown(
            '<div class="chart-empty">LATERAL OFFSET GRAPH — WAITING FOR INFERENCE DATA</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        import altair as alt
        import pandas as pd

        df = pd.DataFrame(history)

        # Zero line
        zero_line = (
            alt.Chart(pd.DataFrame({"y": [0]}))
            .mark_rule(color="#2a3d52", strokeWidth=1, strokeDash=[4, 4])
            .encode(y="y:Q")
        )

        # Offset line
        offset_line = (
            alt.Chart(df)
            .mark_line(color="#00b4cc", strokeWidth=2, interpolate="monotone")
            .encode(
                x=alt.X("frame:Q", title="Frame", axis=alt.Axis(labelColor="#4a6178", titleColor="#4a6178", gridColor="#1a2d42")),
                y=alt.Y("offset:Q", title="Lateral Offset (norm)", scale=alt.Scale(domain=[-1.1, 1.1]),
                        axis=alt.Axis(labelColor="#4a6178", titleColor="#4a6178", gridColor="#1a2d42")),
            )
        )

        # Current value dot
        latest_df = df.iloc[[-1]]
        dot = (
            alt.Chart(latest_df)
            .mark_point(color="#00b4cc", size=60, filled=True)
            .encode(x="frame:Q", y="offset:Q")
        )

        chart = (
            (zero_line + offset_line + dot)
            .properties(height=130, background="#0c1524")
            .configure_view(strokeWidth=0)
        )

        st.altair_chart(chart, use_container_width=True)

    except ImportError:
        # Fallback: simple text display of last values
        vals = [h["offset"] for h in history[-10:]]
        st.caption("Altair not available. Last 10 offsets: " + ", ".join(f"{v:+.3f}" for v in vals))
