"""
src/visualization/lane_overlay.py
===================================
Draw lane detection results onto an image.

All drawing operates in MODEL-space coordinates.
The final image is then scaled back to ORIGINAL resolution for display.

Overlay elements
----------------
1. Left lane marking           — solid green polyline
2. Right lane marking          — solid blue polyline
3. Lane fill (between lanes)   — semi-transparent green polygon
4. Lane center line            — dashed yellow vertical line
5. Vehicle center line         — dashed white vertical line
6. Lateral error arrow         — red horizontal arrow (vehicle → lane center)
7. HUD text panel              — status, offset, confidence, recommendation
8. Backend/mode label          — bottom-left watermark (CLASSICAL CV BASELINE or ML)

Color coding (BGR)
------------------
  Left lane    : (0, 255,   0)  green
  Right lane   : (255, 100,  0)  blue-orange
  Lane fill    : (0, 200,   0)  green, 30% alpha
  Lane center  : (0, 255, 255)  yellow
  Vehicle center: (255,255,255) white
  Error arrow  : (0,   0, 255)  red
  HUD bg       : (20,  20,  20) dark gray
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference.predictor import LanePrediction, DetectionStatus, ModelStatus
from src.lane_geometry.lane_estimator import LaneGeometry, eval_poly_y_range
from src.lane_geometry.offset_calculator import OffsetResult, SteeringRecommendation, DriftDirection


# ── Colour palette (BGR) ───────────────────────────────────────────────────
C_LEFT_LANE    = (  0, 220,   0)
C_RIGHT_LANE   = (220, 100,   0)
C_LANE_FILL    = (  0, 180,   0)
C_LANE_CENTER  = (  0, 230, 230)
C_VEH_CENTER   = (255, 255, 255)
C_ERROR_ARROW  = (  0,   0, 230)
C_HUD_BG       = ( 18,  18,  18)
C_TEXT_PRIMARY = (240, 240, 240)
C_TEXT_WARN    = (  0, 200, 255)
C_TEXT_OK      = (100, 230, 100)
C_TEXT_ERR     = ( 80,  80, 230)

# ── Drawing settings ──────────────────────────────────────────────────────
LANE_THICKNESS = 4
POLY_ALPHA     = 0.25   # lane fill transparency


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def draw_overlay(
    original_bgr:  np.ndarray,
    prediction:    LanePrediction,
    geometry:      LaneGeometry,
    offset:        OffsetResult,
    scale_x:       float = 1.0,
    scale_y:       float = 1.0,
) -> np.ndarray:
    """
    Draw all lane detection overlays onto the original-resolution image.

    Parameters
    ----------
    original_bgr : np.ndarray
        Original image (uint8 BGR) at full resolution.
    prediction : LanePrediction
        Raw prediction result (for model status label).
    geometry : LaneGeometry
        Fitted lane curves (model-space).
    offset : OffsetResult
        Lateral offset and recommendation.
    scale_x, scale_y : float
        Multipliers to map model-space coords to original image coords.

    Returns
    -------
    np.ndarray
        Annotated image (uint8 BGR, same size as original_bgr).
    """
    canvas = original_bgr.copy()
    h, w   = canvas.shape[:2]
    mh     = geometry.model_h
    mw     = geometry.model_w

    # ── Lane fill ──────────────────────────────────────────────────────────
    if geometry.left_poly is not None and geometry.right_poly is not None:
        canvas = _draw_lane_fill(canvas, geometry, scale_x, scale_y, h, w)

    # ── Lane polylines ─────────────────────────────────────────────────────
    y_top = int(0.35 * mh)
    if geometry.left_poly is not None:
        pts = eval_poly_y_range(geometry.left_poly, y_top, mh, num_points=50)
        pts = _scale_pts(pts, scale_x, scale_y, w, h)
        cv2.polylines(canvas, [pts], False, C_LEFT_LANE, LANE_THICKNESS, cv2.LINE_AA)

    if geometry.right_poly is not None:
        pts = eval_poly_y_range(geometry.right_poly, y_top, mh, num_points=50)
        pts = _scale_pts(pts, scale_x, scale_y, w, h)
        cv2.polylines(canvas, [pts], False, C_RIGHT_LANE, LANE_THICKNESS, cv2.LINE_AA)

    # ── Lane center vertical dashed line ───────────────────────────────────
    if offset.lane_center_x is not None:
        lc_x = int(offset.lane_center_x * scale_x)
        _draw_dashed_vline(canvas, lc_x, int(0.35 * h), h, C_LANE_CENTER, thickness=2)

    # ── Vehicle center vertical line ───────────────────────────────────────
    vc_x = int(offset.vehicle_center_x * scale_x)
    _draw_dashed_vline(canvas, vc_x, int(0.5 * h), h, C_VEH_CENTER, thickness=2)

    # ── Lateral error arrow (at bottom 15% of image) ───────────────────────
    if offset.lane_center_x is not None:
        arrow_y = int(0.88 * h)
        cv2.arrowedLine(
            canvas,
            (vc_x, arrow_y),
            (int(offset.lane_center_x * scale_x), arrow_y),
            C_ERROR_ARROW, 2, cv2.LINE_AA, tipLength=0.25,
        )

    # ── HUD panel ──────────────────────────────────────────────────────────
    canvas = _draw_hud(canvas, prediction, geometry, offset)

    # ── Backend watermark ──────────────────────────────────────────────────
    _draw_watermark(canvas, prediction.backend_name, prediction.model_status)

    return canvas


def draw_no_detection(
    original_bgr: np.ndarray,
    message: str,
    prediction:   LanePrediction,
) -> np.ndarray:
    """Return the original image with an overlay message when detection fails."""
    canvas = original_bgr.copy()
    h, w = canvas.shape[:2]

    # Dark semi-transparent overlay band
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h // 3), (w, 2 * h // 3), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, canvas, 0.4, 0, canvas)

    # Message
    for i, line in enumerate(message.split("\n")):
        y = h // 2 - 30 + i * 35
        cv2.putText(canvas, line, (30, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 180, 255), 2, cv2.LINE_AA)

    _draw_watermark(canvas, prediction.backend_name, prediction.model_status)
    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# Internal drawing helpers
# ═══════════════════════════════════════════════════════════════════════════

def _scale_pts(
    pts: np.ndarray,
    sx: float, sy: float,
    max_w: int, max_h: int,
) -> np.ndarray:
    """Scale (x, y) points from model-space to original-image-space."""
    scaled = pts.copy().astype(np.float32)
    scaled[:, 0] = np.clip(scaled[:, 0] * sx, 0, max_w - 1)
    scaled[:, 1] = np.clip(scaled[:, 1] * sy, 0, max_h - 1)
    return scaled.reshape(-1, 1, 2).astype(np.int32)


def _draw_lane_fill(
    canvas: np.ndarray,
    geometry: LaneGeometry,
    sx: float, sy: float,
    h: int, w: int,
) -> np.ndarray:
    """Fill the region between left and right lane curves with semi-transparent green."""
    y_top = int(0.35 * geometry.model_h)
    n     = 50

    left_pts  = eval_poly_y_range(geometry.left_poly,  y_top, geometry.model_h, n)
    right_pts = eval_poly_y_range(geometry.right_poly, y_top, geometry.model_h, n)

    # Build polygon: left top-to-bottom, right bottom-to-top
    left_sc  = _scale_pts(left_pts,        sx, sy, w, h).reshape(-1, 2)
    right_sc = _scale_pts(right_pts[::-1], sx, sy, w, h).reshape(-1, 2)
    poly_pts = np.vstack([left_sc, right_sc]).astype(np.int32)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [poly_pts], C_LANE_FILL)
    cv2.addWeighted(overlay, POLY_ALPHA, canvas, 1 - POLY_ALPHA, 0, canvas)
    return canvas


def _draw_dashed_vline(
    img: np.ndarray,
    x: int,
    y_start: int,
    y_end: int,
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 18,
    gap_len:  int = 10,
) -> None:
    """Draw a dashed vertical line."""
    y = y_start
    draw = True
    while y < y_end:
        y2 = min(y + (dash_len if draw else gap_len), y_end)
        if draw:
            cv2.line(img, (x, y), (x, y2), color, thickness, cv2.LINE_AA)
        y = y2
        draw = not draw


def _draw_hud(
    canvas: np.ndarray,
    prediction: LanePrediction,
    geometry: LaneGeometry,
    offset: OffsetResult,
) -> np.ndarray:
    """Draw a HUD information panel in the top-right corner."""
    h, w = canvas.shape[:2]

    # Panel dimensions
    panel_w  = min(320, w)
    panel_h  = 260
    pad      = 12
    x0       = w - panel_w - 8
    y0       = 8

    # Semi-transparent background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

    # Border
    cv2.rectangle(canvas, (x0, y0), (x0 + panel_w, y0 + panel_h),
                  (80, 80, 80), 1)

    def put(text: str, row: int, color=C_TEXT_PRIMARY, scale: float = 0.5):
        y = y0 + pad + row * 24
        cv2.putText(canvas, text, (x0 + pad, y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    # Detection status
    det_status = prediction.status.value
    status_color = (
        C_TEXT_OK  if "DETECTED" == det_status else
        C_TEXT_WARN if "PARTIAL" in det_status else
        C_TEXT_ERR
    )
    put(det_status, 0, status_color, scale=0.55)

    # Lane presence
    left_str  = "L: YES" if prediction.left_detected  else "L: NO"
    right_str = "R: YES" if prediction.right_detected else "R: NO"
    put(f"{left_str}  {right_str}", 1)

    # Confidence
    lc = f"{prediction.left_confidence:.2f}"  if prediction.left_detected  else "--"
    rc = f"{prediction.right_confidence:.2f}" if prediction.right_detected else "--"
    put(f"Conf  L:{lc}  R:{rc}", 2)

    # Positions
    vc = f"{offset.vehicle_center_x:.0f}"
    lnc = f"{offset.lane_center_x:.0f}" if offset.lane_center_x else "--"
    put(f"Veh:{vc}px  Lane:{lnc}px", 3)

    # Lateral error
    err_px   = f"{offset.lateral_error_px:+.1f}px"  if offset.lateral_error_px   is not None else "--"
    err_norm = f"{offset.lateral_error_norm:+.3f}"   if offset.lateral_error_norm is not None else "--"
    put(f"Offset: {err_px}  ({err_norm})", 4)

    # Drift
    drift_col = C_TEXT_OK if offset.drift_direction == DriftDirection.CENTERED else C_TEXT_WARN
    put(f"Drift: {offset.drift_direction.value}", 5, drift_col)

    # One-lane estimate warning
    if offset.one_lane_estimated:
        put("! One-lane estimate", 6, (0, 180, 255))

    # Steering recommendation
    rec = offset.recommendation
    rec_color = (
        C_TEXT_OK   if rec == SteeringRecommendation.KEEP_CENTER      else
        C_TEXT_WARN if rec in (SteeringRecommendation.STEER_LEFT,
                               SteeringRecommendation.STEER_RIGHT)     else
        C_TEXT_ERR
    )
    put(f">> {rec.value}", 8, rec_color, scale=0.60)

    # Steering command value
    if offset.steering_command is not None:
        put(f"   cmd: {offset.steering_command:+.4f}", 9, C_TEXT_PRIMARY)

    return canvas


def _draw_watermark(
    canvas: np.ndarray,
    backend_name: str,
    model_status: ModelStatus,
) -> None:
    """Draw backend label in bottom-left corner."""
    h, w = canvas.shape[:2]
    label = f"[{model_status.value}]  {backend_name}"
    color = (80, 80, 230) if "CLASSICAL" in model_status.value else (100, 230, 100)
    cv2.putText(canvas, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(canvas, label, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, color,   1, cv2.LINE_AA)
