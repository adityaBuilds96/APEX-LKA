"""
src/visualization/lane_overlay.py
===================================
Draw high-contrast, automotive-grade lane detection overlays onto road imagery.

Styling & Elements
------------------
1. Left lane line          — Bright Cyan/Green solid polyline
2. Right lane line         — Bright Safety Orange solid polyline
3. Drivable lane center    — Yellow dashed line
4. Drivable corridor fill  — Semi-transparent green polygon (alpha blend)
5. Vehicle centerline      — Subtle dashed white line
6. Metric lateral offset   — High-contrast horizontal error arrow
7. Clean structured HUD    — Resolution-scaled (H / 720) top-right / top-left HUD box
"""

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import DetectionStatus, LanePrediction, ModelStatus
from src.lane_geometry.lane_estimator import LaneGeometry, eval_poly_y_range
from src.lane_geometry.offset_calculator import (
    DriftDirection,
    OffsetResult,
    SteeringRecommendation,
)

# ── Color Palette (BGR) ────────────────────────────────────────────────────
C_LEFT_LANE    = (  0, 255, 128)  # Bright Cyan/Green
C_RIGHT_LANE   = (  0, 140, 255)  # Bright Safety Orange
C_LANE_CENTER  = (  0, 255, 255)  # Yellow Dashed Center Path
C_LANE_FILL    = (  0, 200, 100)  # Road Corridor Fill
C_VEH_CENTER   = (240, 240, 240)  # White Vehicle Center
C_ERROR_ARROW  = (  0,  60, 255)  # Amber-Red Error Vector
C_HUD_BG       = ( 10,  16,  26)  # Dark Translucent Backdrop
C_TEXT_MAIN    = (245, 245, 245)
C_TEXT_MUTED   = (148, 163, 184)
C_TEXT_ACCENT  = (248, 189,  56)  # APEX Cyan (BGR)

LANE_THICKNESS = 4
POLY_ALPHA     = 0.22


# ═══════════════════════════════════════════════════════════════════════════
# Public Overlay API
# ═══════════════════════════════════════════════════════════════════════════

def draw_overlay(
    original_bgr: np.ndarray,
    prediction: LanePrediction,
    geometry: LaneGeometry,
    offset: OffsetResult,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> np.ndarray:
    """
    Render lane lines, drivable corridor, centerline, and structured HUD.
    """
    canvas = original_bgr.copy()
    h, w = canvas.shape[:2]
    mh = geometry.model_h
    mw = geometry.model_w

    # ── 1. Corridor Alpha Fill ────────────────────────────────────────────
    if geometry.left_poly is not None and geometry.right_poly is not None:
        canvas = _draw_lane_fill(canvas, geometry, scale_x, scale_y, h, w)

    # ── 2. Left & Right Lane Polylines ────────────────────────────────────
    y_top = int(0.35 * mh)

    # Left Lane (Bright Cyan/Green)
    if geometry.left_poly is not None:
        pts_left = eval_poly_y_range(geometry.left_poly, y_top, mh, num_points=50)
        pts_left = _scale_pts(pts_left, scale_x, scale_y, w, h)
        cv2.polylines(canvas, [pts_left], False, C_LEFT_LANE, LANE_THICKNESS, cv2.LINE_AA)

    # Right Lane (Bright Safety Orange)
    if geometry.right_poly is not None:
        pts_right = eval_poly_y_range(geometry.right_poly, y_top, mh, num_points=50)
        pts_right = _scale_pts(pts_right, scale_x, scale_y, w, h)
        cv2.polylines(canvas, [pts_right], False, C_RIGHT_LANE, LANE_THICKNESS, cv2.LINE_AA)

    # ── 3. Drivable Lane Center Path (Yellow Dashed) ──────────────────────
    if offset.lane_center_x is not None:
        lc_x = int(offset.lane_center_x * scale_x)
        _draw_dashed_vline(canvas, lc_x, int(0.35 * h), h, C_LANE_CENTER, thickness=2, dash_len=16, gap_len=10)

    # ── 4. Vehicle Center Reference ───────────────────────────────────────
    vc_x = int(offset.vehicle_center_x * scale_x)
    _draw_dashed_vline(canvas, vc_x, int(0.55 * h), h, C_VEH_CENTER, thickness=1, dash_len=8, gap_len=6)

    # ── 5. Lateral Offset Error Vector Arrow ──────────────────────────────
    if offset.lane_center_x is not None:
        arrow_y = int(0.88 * h)
        target_x = int(offset.lane_center_x * scale_x)
        if abs(target_x - vc_x) > 3:
            cv2.arrowedLine(
                canvas,
                (vc_x, arrow_y),
                (target_x, arrow_y),
                C_ERROR_ARROW, 2, cv2.LINE_AA, tipLength=0.20,
            )

    # ── 6. Clean, Non-Overlapping Structured HUD Box ──────────────────────
    _draw_structured_hud(canvas, prediction, offset)

    return canvas


def draw_no_detection(
    original_bgr: np.ndarray,
    message: str,
    prediction: LanePrediction,
) -> np.ndarray:
    """Return clean image with clear, non-intrusive status banner when lanes are absent."""
    canvas = original_bgr.copy()
    h, w = canvas.shape[:2]
    scale = max(0.40, min(0.80, (h / 720.0) * 0.65))

    # Subtle banner across lower-middle
    bw = int(w * 0.70)
    bh = 50
    bx = (w - bw) // 2
    by = int(h * 0.45)

    overlay = canvas.copy()
    cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (60, 80, 110), 1)

    txt = "LANE MARKINGS NOT DETECTED"
    cv2.putText(canvas, txt, (bx + 20, by + 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 230, 230), 1, cv2.LINE_AA)

    return canvas


# ═══════════════════════════════════════════════════════════════════════════
# Internal Drawing Routines
# ═══════════════════════════════════════════════════════════════════════════

def _scale_pts(pts: np.ndarray, sx: float, sy: float, max_w: int, max_h: int) -> np.ndarray:
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
    y_top = int(0.35 * geometry.model_h)
    n = 45

    left_pts = eval_poly_y_range(geometry.left_poly, y_top, geometry.model_h, n)
    right_pts = eval_poly_y_range(geometry.right_poly, y_top, geometry.model_h, n)

    left_sc = _scale_pts(left_pts, sx, sy, w, h).reshape(-1, 2)
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
    dash_len: int = 16,
    gap_len: int = 10,
) -> None:
    y = y_start
    draw = True
    while y < y_end:
        y2 = min(y + (dash_len if draw else gap_len), y_end)
        if draw:
            cv2.line(img, (x, y), (x, y2), color, thickness, cv2.LINE_AA)
        y = y2
        draw = not draw


def _draw_structured_hud(
    canvas: np.ndarray,
    prediction: LanePrediction,
    offset: OffsetResult,
) -> None:
    """
    Renders clean, resolution-scaled HUD box in the top-right corner.
    Never duplicates text or overflows frame bounds.
    """
    h, w = canvas.shape[:2]
    # Resolution-adaptive font scale based on H / 720
    font_scale = max(0.32, min(0.65, (h / 720.0) * 0.44))
    line_h = int(24 * (h / 720.0))
    line_h = max(16, min(line_h, 32))

    box_w = int(max(180, min(260, w * 0.32)))
    box_h = line_h * 4 + 14
    box_x = w - box_w - 12
    box_y = 12

    # Translucent card background
    overlay = canvas.copy()
    cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), C_HUD_BG, -1)
    cv2.addWeighted(overlay, 0.78, canvas, 0.22, 0, canvas)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (40, 60, 90), 1)

    # Line 1: Backend & Status
    b_tag = "CV (OPENCV)" if "CLASSICAL" in prediction.model_status.value else "ML (LANESEGNET)"
    status_txt = "LOCKED" if prediction.status == DetectionStatus.LANE_DETECTED else "PARTIAL"
    s_col = (100, 230, 100) if status_txt == "LOCKED" else (0, 180, 255)
    cv2.putText(canvas, f"{b_tag} | {status_txt}", (box_x + 10, box_y + line_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, s_col, 1, cv2.LINE_AA)

    # Line 2: Lateral offset
    err_m = getattr(offset, "lateral_error_m", None)
    if err_m is None and getattr(offset, "lateral_error_px", None) is not None:
        err_m = round(offset.lateral_error_px * 0.0185, 3)

    if err_m is not None:
        sign = "+" if err_m >= 0 else ""
        off_txt = f"OFFSET: {sign}{err_m:.2f} m"
    elif getattr(offset, "lateral_error_px", None) is not None:
        sign = "+" if offset.lateral_error_px >= 0 else ""
        off_txt = f"OFFSET: {sign}{offset.lateral_error_px:.0f} px"
    else:
        off_txt = "OFFSET: --"
    cv2.putText(canvas, off_txt, (box_x + 10, box_y + line_h * 2),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_TEXT_MAIN, 1, cv2.LINE_AA)

    # Line 3: Heading error & Accuracy
    acc_pct = getattr(prediction, "lane_accuracy_percent", None)
    if acc_pct is None:
        acc_pct = int(round(max(prediction.left_confidence, prediction.right_confidence) * 100))
    acc_txt = f"ACCURACY: {acc_pct:.0f}%"
    cv2.putText(canvas, acc_txt, (box_x + 10, box_y + line_h * 3),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 0), 1, cv2.LINE_AA)

    # Line 4: Steering recommendation
    rec_txt = f"STEER: {offset.recommendation.value}"
    cv2.putText(canvas, rec_txt, (box_x + 10, box_y + line_h * 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, C_TEXT_MUTED, 1, cv2.LINE_AA)
