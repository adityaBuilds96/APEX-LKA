"""
dashboard/components/telemetry.py
===================================
Right-side telemetry panel for APEX LKA.

Renders three sections:
  PERCEPTION   — lane status, confidence, left/right detection
  VEHICLE POS  — vehicle center, lane center, lateral offset, drift
  SYSTEM HEALTH — camera, model, inference, pipeline

All values come directly from InferenceResult.
N/A is displayed for any unavailable value — never fabricated.

Usage
-----
    from dashboard.components.telemetry import render_telemetry_panel
    render_telemetry_panel(result, camera_active=True)
"""

from typing import Optional
import streamlit as st


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _trow(label: str, value: str, color: str = "#e2e8f0") -> str:
    """One telemetry row: label + value."""
    return (
        f'<div class="trow">'
        f'<span class="trow-label">{label}</span>'
        f'<span class="trow-value" style="color:{color};">{value}</span>'
        f'</div>'
    )


def _tsection(title: str, rows_html: str) -> str:
    return (
        f'<div class="tsection">'
        f'<div class="tsection-title">{title}</div>'
        f'{rows_html}'
        f'</div>'
    )


def _health_dot(ok: bool, label: str, detail: str = "") -> str:
    color  = "#10b981" if ok else "#ef4444"
    state  = "ONLINE" if ok else "OFFLINE"
    detail_html = f'<span class="trow-detail">{detail}</span>' if detail else ""
    return (
        f'<div class="trow">'
        f'<span class="trow-label">{label}</span>'
        f'<span class="trow-value" style="color:{color};">'
        f'<span class="health-dot" style="background:{color};"></span>'
        f'{state}'
        f'</span>'
        f'{detail_html}'
        f'</div>'
    )


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def render_telemetry_panel(
    result=None,
    camera_active: bool = False,
    telemetry=None,
    perf=None,
) -> None:
    """
    Render the full right-side telemetry panel with LKA Engine & Performance data.
    """
    html = '<div class="telemetry-panel">'

    # ── PERCEPTION & STABILITY ────────────────────────────────────────────
    rows = ""
    if telemetry is not None:
        stab = telemetry.stability_state
        s_color = "#10b981" if stab.value == "STABLE" else ("#f59e0b" if "UNCERTAIN" in stab.value or "TEMP" in stab.value else "#ef4444")
        rows += _trow("LANE STABILITY", stab.value, s_color)

        ldw = telemetry.ldw_state
        ldw_color = "#10b981" if ldw.value == "NORMAL" else ("#f59e0b" if "DRIFT" in ldw.value else "#ef4444")
        rows += _trow("LDW ALERT", ldw.value, ldw_color)

        rows += _trow("LEFT LANE",
                      "DETECTED" if telemetry.left_detected else "NOT DETECTED",
                      "#10b981" if telemetry.left_detected else "#ef4444")
        rows += _trow("RIGHT LANE",
                      "DETECTED" if telemetry.right_detected else "NOT DETECTED",
                      "#10b981" if telemetry.right_detected else "#ef4444")

        conf = telemetry.combined_confidence
        c_color = "#10b981" if conf >= 0.6 else ("#f59e0b" if conf >= 0.3 else "#ef4444")
        rows += _trow("CONFIDENCE", f"{conf * 100:.0f}%", c_color)

        if telemetry.failure_condition.value != "NONE":
            rows += _trow("DIAGNOSTIC", telemetry.failure_condition.value, "#f59e0b")

    elif result and result.prediction:
        pred = result.prediction
        ds = pred.status.value
        rows += _trow("LANE STATUS", ds, _det_color(ds))
        rows += _trow("LEFT LANE", "DETECTED" if pred.left_detected else "NOT DETECTED", "#10b981" if pred.left_detected else "#ef4444")
        rows += _trow("RIGHT LANE", "DETECTED" if pred.right_detected else "NOT DETECTED", "#10b981" if pred.right_detected else "#ef4444")
        conf = (pred.left_confidence + pred.right_confidence) / 2.0
        rows += _trow("CONFIDENCE", f"{conf * 100:.0f}%", "#10b981" if conf >= 0.5 else "#f59e0b")
    else:
        rows = _trow("LANE STATUS", "IDLE") + _trow("LEFT LANE", "--") + _trow("RIGHT LANE", "--") + _trow("CONFIDENCE", "--")

    html += _tsection("PERCEPTION & STABILITY", rows)

    # ── GEOMETRY & VEHICLE POSITION ───────────────────────────────────────
    rows = ""
    if telemetry is not None:
        vc = f"{telemetry.vehicle_center_x:.0f} px"
        lc = f"{telemetry.lane_center_x:.0f} px" if telemetry.lane_center_x is not None else "--"
        rows += _trow("VEHICLE CTR", vc)
        rows += _trow("LANE CTR", lc)

        if telemetry.lateral_offset_m is not None:
            sign = "+" if telemetry.lateral_offset_m >= 0 else ""
            off_m_str = f"{sign}{telemetry.lateral_offset_m:.2f} m"
            off_px_str = f"{sign}{telemetry.lateral_offset_px:.0f} px"
            ok_err = abs(telemetry.lateral_offset_norm or 0.0) < 0.08
            e_color = "#10b981" if ok_err else ("#f59e0b" if abs(telemetry.lateral_offset_norm or 0.0) < 0.25 else "#ef4444")
            rows += _trow("LAT OFFSET", f"{off_m_str} ({off_px_str})", e_color)
        else:
            rows += _trow("LAT OFFSET", "--")

        if telemetry.heading_error_deg is not None:
            h_sign = "+" if telemetry.heading_error_deg >= 0 else ""
            h_col = "#10b981" if abs(telemetry.heading_error_deg) < 2.0 else "#f59e0b"
            rows += _trow("HEADING ERROR", f"{h_sign}{telemetry.heading_error_deg:.1f}\u00b0", h_col)
        else:
            rows += _trow("HEADING ERROR", "--")

        if telemetry.lane_width_m is not None:
            rows += _trow("LANE WIDTH", f"{telemetry.lane_width_m:.2f} m ({telemetry.lane_width_px:.0f} px)")
        else:
            rows += _trow("LANE WIDTH", "--")

        if telemetry.curvature_radius_m is not None:
            rows += _trow("CURVATURE R", f"{telemetry.curvature_radius_m:.0f} m")

        rows += _trow("DRIFT", telemetry.drift_direction.value)

    elif result and result.offset:
        off = result.offset
        vc = f"{off.vehicle_center_x:.0f} px"
        lc = f"{off.lane_center_x:.0f} px" if off.lane_center_x is not None else "--"
        rows += _trow("VEHICLE CTR", vc)
        rows += _trow("LANE CTR", lc)
        rows += _trow("LAT OFFSET", f"{off.lateral_error_px:+.1f} px" if off.lateral_error_px is not None else "--")
        rows += _trow("DRIFT", off.drift_direction.value)
    else:
        for lbl in ["VEHICLE CTR", "LANE CTR", "LAT OFFSET", "HEADING ERROR", "LANE WIDTH", "DRIFT"]:
            rows += _trow(lbl, "--")

    html += _tsection("LANE GEOMETRY", rows)

    # ── SOFTWARE STEERING RECOMMENDATION ──────────────────────────────────
    rows = ""
    if telemetry is not None and telemetry.steering_recommendation is not None:
        rec_str = telemetry.steering_recommendation.value
        r_color = _rec_color(rec_str)
        angle_sign = "+" if telemetry.recommended_angle_deg >= 0 else ""
        angle_str = f"{angle_sign}{telemetry.recommended_angle_deg:.1f}\u00b0"
        rows += _trow("RECOMMENDATION", rec_str, r_color)
        rows += _trow("CORRECTION ANGLE", angle_str, "#00b4cc")
        if telemetry.steering_command is not None:
            rows += _trow("PD COMMAND", f"{telemetry.steering_command:+.3f}", "#8a9ab0")
    elif result and result.offset:
        rec_str = result.offset.recommendation.value
        rows += _trow("RECOMMENDATION", rec_str, _rec_color(rec_str))
        rows += _trow("CORRECTION ANGLE", "--")
    else:
        rows = _trow("RECOMMENDATION", "NO DATA") + _trow("CORRECTION ANGLE", "--")

    html += _tsection("STEERING (SOFTWARE ONLY)", rows)

    # ── PERFORMANCE MONITOR ───────────────────────────────────────────────
    rows = ""
    if perf is not None:
        fps_c = "#10b981" if perf.camera_fps >= 24 else ("#f59e0b" if perf.camera_fps >= 15 else "#4a6178")
        rows += _trow("CAMERA FPS", f"{perf.camera_fps:.1f}" if perf.camera_fps > 0 else "--", fps_c)
        rows += _trow("INFERENCE FPS", f"{perf.inference_fps:.1f}" if perf.inference_fps > 0 else "--", "#00b4cc")
        rows += _trow("RENDER FPS", f"{perf.ui_fps:.1f}" if perf.ui_fps > 0 else "--")
        rows += _trow("INF LATENCY", f"{perf.inference_latency_ms:.0f} ms" if perf.inference_latency_ms > 0 else "--")
        rows += _trow("TOTAL LATENCY", f"{perf.pipeline_latency_ms:.0f} ms" if perf.pipeline_latency_ms > 0 else "--")
        drop_color = "#10b981" if perf.dropped_frames == 0 else "#f59e0b"
        rows += _trow("DROPPED FRAMES", f"{perf.dropped_frames}", drop_color)
        rows += _trow("CPU USAGE", f"{perf.cpu_usage_pct:.0f}%", "#10b981" if perf.cpu_usage_pct < 70 else "#f59e0b")
        rows += _trow("MEMORY USAGE", f"{perf.ram_usage_pct:.0f}%")
    elif result and result.timings:
        t = result.timings
        rows += _trow("INF LATENCY", f"{t.get('inference_ms', '--')} ms")
        rows += _trow("TOTAL LATENCY", f"{t.get('total_ms', '--')} ms")
        rows += _trow("FPS EQUIV", f"{t.get('fps_equiv', '--')}")
    else:
        for lbl in ["CAMERA FPS", "INFERENCE FPS", "INF LATENCY", "DROPPED FRAMES"]:
            rows += _trow(lbl, "--")

    html += _tsection("PERFORMANCE MONITOR", rows)

    # ── SYSTEM HEALTH ─────────────────────────────────────────────────────
    cam_ok = camera_active or (result is not None and result.preprocessed is not None and result.preprocessed.valid)
    mdl_ok = result is not None and result.prediction is not None and result.prediction.error_message is None
    infer_ok = (telemetry is not None and telemetry.stability_state.value != "LANE LOST") or (result is not None and result.prediction is not None)
    pipe_ok = (telemetry is not None and telemetry.stability_state.value == "STABLE") or (result is not None and result.success)

    model_detail = ""
    if result and result.prediction:
        model_detail = result.prediction.model_status.value[:14]

    rows = (
        _health_dot(cam_ok,   "CAMERA",    "LIVE" if camera_active else "IDLE") +
        _health_dot(mdl_ok,   "MODEL",     model_detail) +
        _health_dot(infer_ok, "INFERENCE", "RUNNING" if infer_ok else "IDLE") +
        _health_dot(pipe_ok,  "PERCEPTION", "STABLE" if pipe_ok else "DEGRADED")
    )
    html += _tsection("SYSTEM HEALTH", rows)

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# Color helpers
# ═══════════════════════════════════════════════════════════════════════════

def _det_color(status: str) -> str:
    if "LANE DETECTED" == status:
        return "#10b981"
    if "PARTIAL" in status:
        return "#f59e0b"
    return "#ef4444"


def _rec_color(rec_str: str) -> str:
    if "CENTER" in rec_str:
        return "#10b981"
    if "LEFT" in rec_str or "RIGHT" in rec_str:
        return "#f59e0b"
    if "LOW" in rec_str:
        return "#ef4444"
    return "#4a6178"
