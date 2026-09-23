"""
dashboard/components/telemetry.py
===================================
Automotive Telemetry & State Presentation Components for APEX LKA.

Provides focused, high-density, automotive-grade telemetry components:
- render_lka_status_card(): LKA State & Lane Departure Warning (LDW) Status
- render_lateral_offset_gauge(): High-precision horizontal offset track & numerical value
- render_perception_cluster(): Compact technical perception metrics (Confidence, Heading, Curvature, Width)
- render_performance_cluster(): Framerate, inference latency, end-to-end pipeline latency
- render_system_health_strip(): Inline subsystem health indicator strip

All values originate strictly from backend InferenceResult, LKATelemetry, and PerformanceStats.
Zero fabricated or mocked values.
"""

from typing import Optional
import streamlit as st

from src.lka_engine.lka_state import LaneStabilityState, LDWState, LKATelemetry
from src.inference.pipeline import InferenceResult
from dashboard.camera_stream import PerformanceStats


# ═══════════════════════════════════════════════════════════════════════════
# Styling & Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _metric_row(label: str, value: str, color: str = "#e2e8f0") -> str:
    return (
        f'<div style="display:flex; justify-content:space-between; align-items:center; '
        f'padding:4px 0; border-bottom:1px solid #121c2b;">'
        f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:0.62rem; '
        f'color:#4a6178; letter-spacing:0.08em; text-transform:uppercase;">{label}</span>'
        f'<span style="font-family:\'JetBrains Mono\', monospace; font-size:0.80rem; '
        f'font-weight:600; color:{color}; text-align:right;">{value}</span>'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# 1. LKA State & LDW Card
# ═══════════════════════════════════════════════════════════════════════════

def render_lka_status_card(
    telemetry: Optional[LKATelemetry] = None,
    result: Optional[InferenceResult] = None,
) -> None:
    """
    Renders high-visibility LKA State and LDW departure alert.
    """
    # LKA State derivation
    if telemetry is not None:
        stab = telemetry.stability_state
        if stab == LaneStabilityState.STABLE:
            state_text = "LKA ACTIVE"
            state_color = "#10b981"
            dot_color = "#10b981"
        elif stab == LaneStabilityState.UNCERTAIN:
            state_text = "UNCERTAIN"
            state_color = "#f59e0b"
            dot_color = "#f59e0b"
        elif stab == LaneStabilityState.TEMPORARILY_LOST:
            state_text = "TEMPORARILY LOST"
            state_color = "#f59e0b"
            dot_color = "#f59e0b"
        else:
            state_text = "LANE LOST"
            state_color = "#ef4444"
            dot_color = "#ef4444"
    elif result and result.prediction:
        status_val = result.prediction.status.value
        if "LANE DETECTED" in status_val:
            state_text = "LKA ACTIVE"
            state_color = "#10b981"
            dot_color = "#10b981"
        elif "PARTIAL" in status_val:
            state_text = "PARTIAL DETECTION"
            state_color = "#f59e0b"
            dot_color = "#f59e0b"
        else:
            state_text = "LANE LOST"
            state_color = "#ef4444"
            dot_color = "#ef4444"
    else:
        state_text = "STANDBY"
        state_color = "#62758d"
        dot_color = "#3a5070"

    # LDW State
    if telemetry is not None:
        ldw = telemetry.ldw_state
        if ldw == LDWState.NORMAL:
            ldw_text = "NORMAL"
            ldw_color = "#10b981"
        elif ldw in (LDWState.DRIFT_LEFT, LDWState.DRIFT_RIGHT):
            ldw_text = ldw.value
            ldw_color = "#f59e0b"
        else:
            ldw_text = ldw.value
            ldw_color = "#ef4444"
    elif result and result.offset:
        rec = result.offset.recommendation.value
        ldw_text = rec
        ldw_color = "#10b981" if "CENTER" in rec else ("#f59e0b" if "STEER" in rec else "#ef4444")
    else:
        ldw_text = "--"
        ldw_color = "#62758d"

    html = f"""
    <div style="background:#0c1421; border:1px solid #1a283c; border-radius:6px; padding:14px 18px; margin-bottom:12px;">
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #142032; padding-bottom:6px; margin-bottom:10px;">
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.60rem; font-weight:700; letter-spacing:0.18em; color:#4a6178; text-transform:uppercase;">
          LKA STATE
        </span>
        <span style="display:inline-flex; align-items:center; gap:5px; font-family:'JetBrains Mono', monospace; font-size:0.60rem; color:{ldw_color};">
          <span style="width:6px; height:6px; border-radius:50%; background:{ldw_color};"></span>
          {ldw_text}
        </span>
      </div>
      <div style="display:flex; align-items:baseline; gap:10px;">
        <span style="width:10px; height:10px; border-radius:50%; background:{dot_color}; box-shadow:0 0 8px {dot_color}88; flex-shrink:0;"></span>
        <span style="font-family:'Inter', sans-serif; font-size:1.45rem; font-weight:800; color:{state_color}; letter-spacing:0.06em;">
          {state_text}
        </span>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Lateral Offset with Precision Center Indicator
# ═══════════════════════════════════════════════════════════════════════════

def render_lateral_offset_gauge(
    telemetry: Optional[LKATelemetry] = None,
    result: Optional[InferenceResult] = None,
) -> None:
    """
    Renders a clean automotive horizontal lateral offset gauge:
    LEFT                CENTER                RIGHT
    ──────────────────────●────────────────────────
                         +0.12 m
    """
    offset_m = None
    offset_norm = None

    if telemetry is not None:
        offset_m = telemetry.lateral_offset_m
        offset_norm = telemetry.smoothed_offset_norm if telemetry.smoothed_offset_norm is not None else telemetry.lateral_offset_norm
    elif result and result.offset:
        offset_norm = result.offset.lateral_error_norm

    # Position on gauge: [-1, +1] maps to [5%, 95%]
    if offset_norm is not None:
        # Clamp to [-1.0, 1.0]
        clamped_norm = max(-1.0, min(1.0, float(offset_norm)))
        pct = 50.0 + (clamped_norm * 45.0)

        # Semantic color
        abs_norm = abs(clamped_norm)
        if abs_norm < 0.08:
            color = "#10b981"  # Centered
        elif abs_norm < 0.25:
            color = "#f59e0b"  # Drift
        else:
            color = "#ef4444"  # Departure warning

        if offset_m is not None:
            sign = "+" if offset_m >= 0 else ""
            reading_text = f"{sign}{offset_m:.2f} m"
        else:
            sign = "+" if clamped_norm >= 0 else ""
            reading_text = f"{sign}{clamped_norm:.2f} norm"
    else:
        pct = 50.0
        color = "#4a6178"
        reading_text = "-- m"

    html = f"""
    <div style="background:#0c1421; border:1px solid #1a283c; border-radius:6px; padding:14px 18px; margin-bottom:12px;">
      <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid #142032; padding-bottom:6px; margin-bottom:8px;">
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.60rem; font-weight:700; letter-spacing:0.18em; color:#4a6178; text-transform:uppercase;">
          LATERAL POSITION
        </span>
        <span style="font-family:'JetBrains Mono', monospace; font-size:0.65rem; font-weight:700; color:{color};">
          {reading_text}
        </span>
      </div>

      <div style="display:flex; justify-content:space-between; font-family:'JetBrains Mono', monospace; font-size:0.55rem; color:#4a6178; letter-spacing:0.12em; margin-bottom:4px;">
        <span>LEFT</span>
        <span style="color:#7a8d9e;">CENTER</span>
        <span>RIGHT</span>
      </div>

      <div style="position:relative; width:100%; height:6px; background:#142032; border-radius:3px; margin:8px 0 12px;">
        <!-- Center reference mark -->
        <div style="position:absolute; left:50%; top:-4px; width:2px; height:14px; background:#3a5070; transform:translateX(-50%);"></div>
        <!-- Lateral vehicle position pointer -->
        <div style="position:absolute; left:{pct:.1f}%; top:50%; width:12px; height:12px; border-radius:50%; background:{color}; box-shadow:0 0 8px {color}; transform:translate(-50%, -50%); transition:left 0.12s ease-out;"></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Perception Telemetry Cluster
# ═══════════════════════════════════════════════════════════════════════════

def render_perception_cluster(
    telemetry: Optional[LKATelemetry] = None,
    result: Optional[InferenceResult] = None,
) -> None:
    """
    Renders compact perception metrics: Confidence, Offset, Heading, Curvature.
    """
    # Confidence
    if telemetry is not None:
        conf_val = telemetry.combined_confidence
        conf_str = f"{conf_val * 100:.0f}%"
        conf_col = "#10b981" if conf_val >= 0.6 else ("#f59e0b" if conf_val >= 0.3 else "#ef4444")
    elif result and result.prediction:
        p = result.prediction
        conf_val = (p.left_confidence + p.right_confidence) / 2.0
        conf_str = f"{conf_val * 100:.0f}%"
        conf_col = "#10b981" if conf_val >= 0.5 else "#f59e0b"
    else:
        conf_str = "--"
        conf_col = "#62758d"

    # Lateral Offset
    if telemetry and telemetry.lateral_offset_m is not None:
        sign = "+" if telemetry.lateral_offset_m >= 0 else ""
        off_str = f"{sign}{telemetry.lateral_offset_m:.2f} m"
        off_col = "#10b981" if abs(telemetry.lateral_offset_norm or 0.0) < 0.08 else "#f59e0b"
    else:
        off_str = "--"
        off_col = "#62758d"

    # Heading Error
    if telemetry and telemetry.heading_error_deg is not None:
        h_sign = "+" if telemetry.heading_error_deg >= 0 else ""
        head_str = f"{h_sign}{telemetry.heading_error_deg:.1f}\u00b0"
        head_col = "#10b981" if abs(telemetry.heading_error_deg) < 2.0 else "#f59e0b"
    else:
        head_str = "--"
        head_col = "#62758d"

    # Curvature
    if telemetry and telemetry.curvature_radius_m is not None:
        curv_str = f"{telemetry.curvature_radius_m:.0f} m"
        curv_col = "#00d4ee"
    elif telemetry and telemetry.stability_state in (LaneStabilityState.STABLE, LaneStabilityState.UNCERTAIN):
        curv_str = "Straight"
        curv_col = "#10b981"
    else:
        curv_str = "--"
        curv_col = "#62758d"

    rows = (
        _metric_row("Confidence", conf_str, conf_col) +
        _metric_row("Offset", off_str, off_col) +
        _metric_row("Heading", head_str, head_col) +
        _metric_row("Curvature", curv_str, curv_col)
    )

    html = f"""
    <div style="background:#0c1421; border:1px solid #1a283c; border-radius:6px; padding:14px 18px; margin-bottom:12px;">
      <div style="font-family:'JetBrains Mono', monospace; font-size:0.60rem; font-weight:700; letter-spacing:0.18em; color:#00d4ee; text-transform:uppercase; border-bottom:1px solid #142032; padding-bottom:6px; margin-bottom:6px;">
        PERCEPTION
      </div>
      {rows}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4. Performance Telemetry Cluster
# ═══════════════════════════════════════════════════════════════════════════

def render_performance_cluster(
    perf: Optional[PerformanceStats] = None,
    result: Optional[InferenceResult] = None,
) -> None:
    """
    Renders compact performance metrics: FPS, Inference latency, E2E latency.
    """
    if perf is not None and (perf.camera_fps > 0 or perf.ui_fps > 0):
        fps_val = perf.camera_fps if perf.camera_fps > 0 else perf.ui_fps
        fps_str = f"{fps_val:.1f} FPS"
        fps_col = "#10b981" if fps_val >= 24 else ("#f59e0b" if fps_val >= 15 else "#ef4444")
        inf_str = f"{perf.inference_latency_ms:.0f} ms" if perf.inference_latency_ms > 0 else "-- ms"
        e2e_str = f"{perf.pipeline_latency_ms:.0f} ms" if perf.pipeline_latency_ms > 0 else "-- ms"
    elif result and result.timings:
        t = result.timings
        fps_v = t.get("fps_equiv")
        fps_str = f"{fps_v:.1f} FPS" if isinstance(fps_v, (int, float)) else f"{fps_v} FPS"
        fps_col = "#00d4ee"
        inf_v = t.get("inference_ms", 0.0)
        tot_v = t.get("total_ms", 0.0)
        inf_str = f"{inf_v:.0f} ms"
        e2e_str = f"{tot_v:.0f} ms"
    else:
        fps_str = "-- FPS"
        fps_col = "#62758d"
        inf_str = "-- ms"
        e2e_str = "-- ms"

    rows = (
        _metric_row("Framerate", fps_str, fps_col) +
        _metric_row("Inference", inf_str, "#e2e8f0") +
        _metric_row("End-to-End", e2e_str, "#00d4ee")
    )

    html = f"""
    <div style="background:#0c1421; border:1px solid #1a283c; border-radius:6px; padding:14px 18px; margin-bottom:12px;">
      <div style="font-family:'JetBrains Mono', monospace; font-size:0.60rem; font-weight:700; letter-spacing:0.18em; color:#00d4ee; text-transform:uppercase; border-bottom:1px solid #142032; padding-bottom:6px; margin-bottom:6px;">
        PERFORMANCE
      </div>
      {rows}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# 5. System Health Strip
# ═══════════════════════════════════════════════════════════════════════════

def render_system_health_strip(
    camera_active: bool = False,
    result: Optional[InferenceResult] = None,
    telemetry: Optional[LKATelemetry] = None,
) -> None:
    """
    Renders compact inline subsystem status indicators using actual states:
    ● CAMERA OK  ● INFERENCE OK  ● LANE ENGINE OK  ● TELEMETRY OK  ● STREAM OK
    """
    cam_ok = camera_active or (result is not None and result.preprocessed is not None and result.preprocessed.valid)
    infer_ok = (result is not None and result.prediction is not None and result.prediction.error_message is None)
    lane_ok = (telemetry is not None and telemetry.stability_state != LaneStabilityState.LANE_LOST) or (result is not None and result.success)
    telem_ok = (telemetry is not None) or (result is not None and result.offset is not None)
    stream_ok = camera_active

    def _chip(label: str, ok: bool, active_text: str = "OK", idle_text: str = "IDLE") -> str:
        color = "#10b981" if ok else "#62758d"
        status_txt = active_text if ok else idle_text
        return (
            f'<div style="display:flex; align-items:center; gap:6px; font-family:\'JetBrains Mono\', monospace; font-size:0.60rem; letter-spacing:0.08em; text-transform:uppercase;">'
            f'<span style="width:6px; height:6px; border-radius:50%; background:{color}; flex-shrink:0;"></span>'
            f'<span style="color:#7a8d9e;">{label}</span>'
            f'<span style="color:{color}; font-weight:600;">{status_txt}</span>'
            f'</div>'
        )

    html = f"""
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:12px; background:#080e18; border:1px solid #162438; border-radius:5px; padding:8px 16px; margin-top:8px;">
      <div style="font-family:'JetBrains Mono', monospace; font-size:0.58rem; font-weight:700; color:#4a6178; letter-spacing:0.16em;">
        SYSTEM HEALTH
      </div>
      <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        {_chip("CAMERA", cam_ok, "ONLINE", "IDLE")}
        {_chip("INFERENCE", infer_ok, "ACTIVE", "IDLE")}
        {_chip("LANE ENGINE", lane_ok, "OK", "DEGRADED")}
        {_chip("TELEMETRY", telem_ok, "STREAMING", "NO DATA")}
        {_chip("PIPELINE", stream_ok or cam_ok, "LOCKED", "STANDBY")}
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Backward Compatibility Wrapper
# ═══════════════════════════════════════════════════════════════════════════

def render_telemetry_panel(
    result: Optional[InferenceResult] = None,
    camera_active: bool = False,
    telemetry: Optional[LKATelemetry] = None,
    perf: Optional[PerformanceStats] = None,
) -> None:
    """
    Renders the consolidated telemetry components.
    """
    render_lka_status_card(telemetry=telemetry, result=result)
    render_lateral_offset_gauge(telemetry=telemetry, result=result)
    render_perception_cluster(telemetry=telemetry, result=result)
    render_performance_cluster(perf=perf, result=result)
    render_system_health_strip(camera_active=camera_active, result=result, telemetry=telemetry)
