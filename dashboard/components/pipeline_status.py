"""
dashboard/components/pipeline_status.py
=========================================
Live pipeline stage visualization for APEX LKA.

Shows the status of each pipeline stage based on the actual InferenceResult.

Stages
------
CAMERA → PREPROCESSING → MODEL → LANE GEOMETRY → RECOMMENDATION

Each stage has one of:
  ACTIVE      — stage completed successfully this frame
  WAITING     — pipeline has not run yet
  ERROR       — stage produced an error
  NOT AVAILABLE — stage skipped due to upstream failure

Usage
-----
    from dashboard.components.pipeline_status import render_pipeline_status
    render_pipeline_status(result, camera_active=True)
"""

from typing import Optional
import streamlit as st


# Stage definitions
_STAGES = [
    ("CAMERA",          "camera"),
    ("PREPROCESS",      "preprocess"),
    ("MODEL",           "model"),
    ("LANE GEOMETRY",   "geometry"),
    ("RECOMMENDATION",  "recommendation"),
]

_STATE_COLOR = {
    "active":        ("#10b981", "#0d2b1e"),   # green
    "waiting":       ("#4a6178", "#0c1524"),   # gray
    "error":         ("#ef4444", "#2b0d0d"),   # red
    "not_available": ("#f59e0b", "#2b1a00"),   # amber
}


def render_pipeline_status(result=None, camera_active: bool = False) -> None:
    """
    Render the pipeline stage status panel.

    Parameters
    ----------
    result : InferenceResult | None
    camera_active : bool  — whether the camera is actively streaming
    """
    stages = _compute_stages(result, camera_active)

    html = '<div class="pipeline-panel">'
    for i, (name, state) in enumerate(stages):
        color, bg = _STATE_COLOR.get(state, _STATE_COLOR["waiting"])
        dot_style = (
            f"width:8px; height:8px; border-radius:50%; "
            f"background:{color}; flex-shrink:0;"
        )
        label_style = (
            f"font-family:monospace; font-size:0.68rem; font-weight:600; "
            f"color:{color}; letter-spacing:0.08em; text-transform:uppercase;"
        )
        state_style = (
            f"font-family:monospace; font-size:0.55rem; "
            f"color:{color}; opacity:0.75;"
        )
        row_style = (
            f"display:flex; align-items:center; gap:10px; "
            f"padding:7px 12px; background:{bg}; "
            f"border:1px solid {color}22; border-radius:4px; "
            f"margin-bottom:{'0' if i == len(stages)-1 else '2px'};"
        )

        # Arrow connector (except last)
        connector = ""
        if i < len(stages) - 1:
            connector = (
                f'<div style="text-align:center; color:#1a2d42; '
                f'font-size:0.7rem; line-height:0.8; margin:0; padding:0;">↓</div>'
            )

        html += (
            f'<div style="{row_style}">'
            f'<div style="{dot_style}"></div>'
            f'<div style="flex:1;">'
            f'<div style="{label_style}">{name}</div>'
            f'<div style="{state_style}">{state.upper().replace("_", " ")}</div>'
            f'</div>'
            f'</div>'
            f'{connector}'
        )

    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)


def _compute_stages(result, camera_active: bool) -> list[tuple[str, str]]:
    """
    Determine the state of each pipeline stage from InferenceResult.
    Returns list of (stage_name, state) tuples.
    """
    if result is None:
        state = "active" if camera_active else "waiting"
        return [
            ("CAMERA",         state if camera_active else "waiting"),
            ("PREPROCESS",     "waiting"),
            ("MODEL",          "waiting"),
            ("LANE GEOMETRY",  "waiting"),
            ("RECOMMENDATION", "waiting"),
        ]

    # Camera
    cam_state = "active"

    # Preprocessing
    pre = result.preprocessed
    if pre is None:
        pre_state = "error"
    elif not pre.valid:
        pre_state = "error"
    else:
        pre_state = "active"

    # Model / inference
    pred = result.prediction
    if pred is None:
        mdl_state = "not_available"
    else:
        from src.inference.predictor import DetectionStatus, ModelStatus
        if pred.model_status == ModelStatus.NOT_TRAINED:
            mdl_state = "not_available"
        elif pred.status == DetectionStatus.MODEL_UNAVAILABLE:
            mdl_state = "not_available"
        elif pred.status == DetectionStatus.INFERENCE_ERROR:
            mdl_state = "error"
        else:
            mdl_state = "active"

    # Lane geometry
    geo = result.geometry
    if geo is None:
        geo_state = "not_available"
    elif not geo.geometry_valid:
        geo_state = "waiting"   # ran but found nothing
    else:
        geo_state = "active"

    # Recommendation
    off = result.offset
    if off is None:
        rec_state = "not_available"
    elif off.recommendation is not None:
        rec_state = "active"
    else:
        rec_state = "waiting"

    return [
        ("CAMERA",         cam_state),
        ("PREPROCESS",     pre_state),
        ("MODEL",          mdl_state),
        ("LANE GEOMETRY",  geo_state),
        ("RECOMMENDATION", rec_state),
    ]
