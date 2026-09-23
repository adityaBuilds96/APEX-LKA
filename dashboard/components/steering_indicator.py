"""
dashboard/components/steering_indicator.py
============================================
SVG-based steering wheel visualization for APEX LKA.

SAFETY NOTICE
-------------
This is a VISUALIZATION ONLY component.
It does NOT send any signal to a motor, PWM output, CAN bus,
steering actuator, or any physical vehicle component.
It renders the software-level steering recommendation computed
by the existing offset_calculator.py PD controller.

Usage
-----
    from dashboard.components.steering_indicator import render_steering_indicator
    render_steering_indicator(offset_result)
"""

import math
from typing import Optional

import streamlit as st


def render_steering_indicator(offset_result=None, telemetry=None) -> None:
    """
    Render an SVG steering wheel and software recommendation indicator.

    Parameters
    ----------
    offset_result : OffsetResult | None
    telemetry : LKATelemetry | None
    """
    if telemetry is not None and telemetry.steering_recommendation is not None:
        rec_val = telemetry.steering_recommendation.value
        angle_deg = -telemetry.recommended_angle_deg  # CCW for positive (left)
        cmd_val = telemetry.steering_command if telemetry.steering_command is not None else 0.0
        angle_str = f"{telemetry.recommended_angle_deg:+.1f}\u00b0"
        no_data = (rec_val == "NO RECOMMENDATION")
        _render_svg(angle_deg=angle_deg, command=cmd_val, label=rec_val, angle_str=angle_str, no_data=no_data)
        return

    if offset_result is None or offset_result.steering_command is None:
        _render_svg(angle_deg=0.0, command=0.0, label="N/A", angle_str="0.0\u00b0", no_data=True)
        return

    # Map steering_command [-1, 1] → angle [-50, +50] degrees
    cmd = float(offset_result.steering_command)
    angle = -cmd * 50.0   # positive cmd → wheel turns left
    angle = max(-55.0, min(55.0, angle))
    rec = offset_result.recommendation
    label = rec.value if rec is not None else "N/A"
    angle_str = f"{-angle:+.1f}\u00b0"

    _render_svg(angle_deg=angle, command=cmd, label=label, angle_str=angle_str, no_data=False)


def _render_svg(
    angle_deg: float,
    command:   float,
    label:     str,
    angle_str: str,
    no_data:   bool,
) -> None:
    """Render the SVG + HTML block."""

    rad         = math.radians(angle_deg)
    needle_x2   = round(38 * math.sin(rad), 2)
    needle_y2   = round(-38 * math.cos(rad), 2)

    # Needle tip for arrowhead
    tip_angle1  = angle_deg + 150
    tip_angle2  = angle_deg - 150
    r_tip       = 8
    tip1_x = round(38 * math.sin(rad) + r_tip * math.sin(math.radians(tip_angle1)), 2)
    tip1_y = round(-38 * math.cos(rad) - r_tip * math.cos(math.radians(tip_angle1)), 2)
    tip2_x = round(38 * math.sin(rad) + r_tip * math.sin(math.radians(tip_angle2)), 2)
    tip2_y = round(-38 * math.cos(rad) - r_tip * math.cos(math.radians(tip_angle2)), 2)

    stroke_color = "#4a6178" if no_data else "#00b4cc"
    rec_color    = _rec_color(label)

    # Arc indicator (shows deviation from center)
    arc_deg   = angle_deg
    arc_large = 1 if abs(arc_deg) > 180 else 0
    arc_sweep = 1 if arc_deg > 0 else 0
    arc_end_x = round(30 * math.sin(rad), 2)
    arc_end_y = round(-30 * math.cos(rad), 2)

    cmd_str = f"{command:+.3f}" if not no_data else "---"

    svg = f"""
    <div style="display:flex; flex-direction:column; align-items:center; gap:4px; padding:6px 0;">
      <div style="font-family:'JetBrains Mono', monospace; font-size:0.58rem; color:#4a6178; letter-spacing:0.15em; text-transform:uppercase;">
        STEERING RECOMMENDATION
      </div>
      <svg viewBox="-60 -65 120 120" width="110" height="110"
           style="overflow:visible;">
        <!-- Background ring -->
        <circle cx="0" cy="0" r="50"
                fill="none" stroke="#1a2d42" stroke-width="3"/>

        <!-- Tick marks at 12 positions -->
        {''.join(_tick(i * 30) for i in range(12))}

        <!-- LEFT / RIGHT labels -->
        <text x="-54" y="4" font-family="monospace" font-size="7"
              fill="#2a3d52" text-anchor="middle">L</text>
        <text x="54"  y="4" font-family="monospace" font-size="7"
              fill="#2a3d52" text-anchor="middle">R</text>

        <!-- Deviation arc from 12 o'clock to needle tip -->
        {"" if no_data or abs(arc_deg) < 1 else
         f'<path d="M 0 -30 A 30 30 0 {arc_large} {arc_sweep} {arc_end_x} {arc_end_y}" '
         f'fill="none" stroke="#00b4cc" stroke-width="2" stroke-opacity="0.4"/>'}

        <!-- Wheel spokes (static) -->
        <line x1="0" y1="-50" x2="0"  y2="-15" stroke="#1a2d42" stroke-width="2.5"/>
        <line x1="0" y1="50"  x2="0"  y2="15"  stroke="#1a2d42" stroke-width="2.5"/>
        <line x1="-50" y1="0" x2="-15" y2="0"  stroke="#1a2d42" stroke-width="2.5"/>
        <line x1="50"  y1="0" x2="15"  y2="0"  stroke="#1a2d42" stroke-width="2.5"/>

        <!-- Needle -->
        <line x1="0" y1="0" x2="{needle_x2}" y2="{needle_y2}"
              stroke="{stroke_color}" stroke-width="2.5" stroke-linecap="round"/>

        <!-- Arrowhead -->
        <polygon points="{needle_x2},{needle_y2} {tip1_x},{tip1_y} {tip2_x},{tip2_y}"
                 fill="{stroke_color}"/>

        <!-- Center hub -->
        <circle cx="0" cy="0" r="5"
                fill="#0c1524" stroke="{stroke_color}" stroke-width="1.5"/>

        <!-- 12 o'clock marker (center reference) -->
        <line x1="0" y1="-44" x2="0" y2="-56"
              stroke="#2a3d52" stroke-width="2"/>
      </svg>

      <!-- Correction angle value -->
      <div style="font-family:monospace; font-size:1.0rem; color:{stroke_color};
                  font-weight:700; letter-spacing:0.08em;">
        CORRECTION: {angle_str if not no_data else '--'}
      </div>

      <!-- Recommendation label -->
      <div style="font-family:monospace; font-size:0.7rem; font-weight:700;
                  color:{rec_color}; letter-spacing:0.12em; text-transform:uppercase;
                  padding: 3px 10px; border: 1px solid {rec_color};
                  border-radius: 4px; margin-top:2px;">
        {label}
      </div>

      <!-- Horizontal guidance scale -->
      <div style="font-family:monospace; font-size:0.60rem; color:#4a6178; letter-spacing:0.10em; margin-top:4px;">
        LEFT &larr;──────&bull;──────&rarr; RIGHT
      </div>

      <div style="font-size:0.55rem; color:#ef4444; text-align:center;
                  font-weight:600; text-transform:uppercase; letter-spacing:0.08em; margin-top:4px;">
        SOFTWARE RECOMMENDATION ONLY<br>NO PHYSICAL MOTOR / STEERING ACTUATION
      </div>
    </div>
    """

    st.markdown(svg, unsafe_allow_html=True)


def _tick(angle_deg: float) -> str:
    """Generate one tick mark on the wheel rim."""
    r_outer = 50
    r_inner = 45 if angle_deg % 90 != 0 else 40
    rad = math.radians(angle_deg)
    x1 = round(r_inner * math.sin(rad), 2)
    y1 = round(-r_inner * math.cos(rad), 2)
    x2 = round(r_outer * math.sin(rad), 2)
    y2 = round(-r_outer * math.cos(rad), 2)
    w  = "2" if angle_deg % 90 == 0 else "1"
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
        f'stroke="#1e2d40" stroke-width="{w}"/>'
    )


def _rec_color(label: str) -> str:
    if "CENTER" in label:
        return "#10b981"
    if "LEFT" in label or "RIGHT" in label:
        return "#f59e0b"
    if "LOW" in label or "ERROR" in label:
        return "#ef4444"
    return "#4a6178"
