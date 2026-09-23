"""
dashboard/components/lka_hud.py
===============================
Automotive-Grade Perception Viewport HUD Overlay for APEX LKA.

Draws high-contrast, clean HUD elements directly onto the viewport frame:
- Top-Left: System state, live indicator, FPS, Latency.
- Top-Right: LKA state, lane stability, lateral offset (m/px), heading error, confidence.
- Center / Road: Look-ahead point on trajectory curve.
- Alerts: Lane Departure Warning (LDW) banner and diagnostic condition tags.
"""

from typing import Optional, Tuple
import cv2
import numpy as np

from src.lka_engine.lka_state import (
    FailureCondition,
    LaneStabilityState,
    LDWState,
    LKATelemetry,
)
from dashboard.camera_stream import PerformanceStats


# ── Color Palette (BGR) ────────────────────────────────────────────────────
COLOR_BG_DARK    = (10, 16, 26)       # Semi-transparent HUD dark backdrop
COLOR_BORDER     = (40, 60, 85)       # Border slate
COLOR_TEXT_MAIN  = (235, 240, 245)    # Clean white
COLOR_TEXT_MUTED = (120, 145, 170)    # Technical blue-gray
COLOR_CYAN       = (204, 180, 0)      # APEX cyan #00b4cc (BGR)
COLOR_GREEN      = (60, 200, 70)      # Normal / OK
COLOR_AMBER      = (30, 160, 245)     # Warning / Amber
COLOR_RED        = (40, 40, 235)      # Error / Alert


def draw_hud_overlay(
    image_bgr: np.ndarray,
    telemetry: Optional[LKATelemetry] = None,
    perf: Optional[PerformanceStats] = None,
    mode_label: str = "LIVE",
) -> np.ndarray:
    """
    Renders a minimal, clean automotive HUD onto the camera viewport image.
    Maintains road view visual dominance without cluttering boxes.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # ── 1. Minimal Corner Performance Badge ───────────────────────────────
    fps_val = perf.camera_fps if (perf and perf.camera_fps > 0) else (perf.ui_fps if perf else 0.0)
    lat_val = perf.pipeline_latency_ms if (perf and perf.pipeline_latency_ms > 0) else (telemetry.total_ms if telemetry else 0.0)

    _draw_minimal_corner_badge(canvas, mode_label, fps_val, lat_val, x=14, y=14)

    # ── 2. Road Reticle & Trajectory Look-Ahead ───────────────────────────
    if telemetry is not None:
        if (
            telemetry.stability_state in (LaneStabilityState.STABLE, LaneStabilityState.UNCERTAIN)
            and telemetry.lane_center_x is not None
        ):
            _draw_lookahead_reticle(canvas, telemetry, h, w)

        # ── 3. Critical LDW Warning Banner (Active departure only) ─────────
        if telemetry.ldw_state in (LDWState.WARNING, LDWState.DRIFT_LEFT, LDWState.DRIFT_RIGHT):
            _draw_ldw_banner(canvas, telemetry.ldw_state, w, h)

    return canvas


def _draw_minimal_corner_badge(
    canvas: np.ndarray,
    mode: str,
    fps: float,
    latency_ms: float,
    x: int = 14,
    y: int = 14,
) -> None:
    """Minimal, non-intrusive corner performance badge."""
    bw, bh = 142, 54
    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.82)

    # Dot + Mode label
    dot_color = COLOR_GREEN if mode in ("LIVE", "REPLAY") else COLOR_CYAN
    cv2.circle(canvas, (x + 12, y + 15), 4, dot_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, mode, (x + 22, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # FPS
    fps_str = f"FPS  {fps:.1f}" if fps > 0 else "FPS  --"
    cv2.putText(canvas, fps_str, (x + 12, y + 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Latency
    lat_str = f"LAT  {latency_ms:.0f} ms" if latency_ms > 0 else "LAT  --"
    cv2.putText(canvas, lat_str, (x + 12, y + 47),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_CYAN, 1, cv2.LINE_AA)



# ═══════════════════════════════════════════════════════════════════════════
# HUD Component Primitives
# ═══════════════════════════════════════════════════════════════════════════

def _draw_perf_badge(
    canvas: np.ndarray,
    mode: str,
    fps: float,
    latency_ms: float,
    inf_ms: float,
    x: int,
    y: int,
) -> None:
    """Top-left technical performance badge."""
    bw, bh = 175, 68
    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.75)

    # Dot + Mode label
    dot_color = COLOR_GREEN if mode in ("LIVE", "REPLAY") else COLOR_CYAN
    cv2.circle(canvas, (x + 14, y + 16), 5, dot_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, mode, (x + 26, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # FPS
    fps_str = f"FPS: {fps:.1f}" if fps > 0 else "FPS: --"
    cv2.putText(canvas, fps_str, (x + 12, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    if fps > 0:
        cv2.putText(canvas, f"{fps:.1f}", (x + 48, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_CYAN, 1, cv2.LINE_AA)

    # Latency
    lat_str = f"LAT: {latency_ms:.0f}ms (INF {inf_ms:.0f}ms)" if latency_ms > 0 else "LAT: --"
    cv2.putText(canvas, lat_str, (x + 12, y + 58),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)


def _draw_lka_badge(canvas: np.ndarray, telem: LKATelemetry, x: int, y: int) -> None:
    """Top-right LKA perception status badge."""
    bw, bh = 216, 116
    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.80)

    # LKA Title + Stability Status
    stab = telem.stability_state
    if stab == LaneStabilityState.STABLE:
        s_color = COLOR_GREEN
    elif stab == LaneStabilityState.UNCERTAIN:
        s_color = COLOR_AMBER
    elif stab == LaneStabilityState.TEMPORARILY_LOST:
        s_color = COLOR_AMBER
    else:
        s_color = COLOR_RED

    cv2.putText(canvas, "LKA", (x + 12, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(canvas, stab.value, (x + 55, y + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, s_color, 1, cv2.LINE_AA)

    # Offset in meters & px
    if telem.lateral_offset_m is not None:
        off_sign = "+" if telem.lateral_offset_m >= 0 else ""
        off_str = f"OFFSET: {off_sign}{telem.lateral_offset_m:.2f}m ({off_sign}{telem.lateral_offset_px:.0f}px)"
    else:
        off_str = "OFFSET: --"
    cv2.putText(canvas, off_str, (x + 12, y + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # Heading error
    if telem.heading_error_deg is not None:
        h_sign = "+" if telem.heading_error_deg >= 0 else ""
        head_str = f"HEADING: {h_sign}{telem.heading_error_deg:.1f}\u00b0"
    else:
        head_str = "HEADING: --"
    cv2.putText(canvas, head_str, (x + 12, y + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # Lane Width
    if telem.lane_width_m is not None:
        width_str = f"WIDTH: {telem.lane_width_m:.2f}m ({telem.lane_width_px:.0f}px)"
    else:
        width_str = "WIDTH: --"
    cv2.putText(canvas, width_str, (x + 12, y + 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Confidence
    conf_pct = int(round(telem.combined_confidence * 100))
    c_color = COLOR_GREEN if conf_pct >= 60 else (COLOR_AMBER if conf_pct >= 30 else COLOR_RED)
    cv2.putText(canvas, f"CONF: {conf_pct}%", (x + 12, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, c_color, 1, cv2.LINE_AA)

    # Steering recommendation shorthand
    rec_txt = telem.steering_recommendation.value
    if "LEFT" in rec_txt:
        rec_col = COLOR_AMBER
    elif "RIGHT" in rec_txt:
        rec_col = COLOR_AMBER
    elif "CENTER" in rec_txt:
        rec_col = COLOR_GREEN
    else:
        rec_col = COLOR_TEXT_MUTED

    cv2.putText(canvas, f">> {rec_txt}", (x + 95, y + 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, rec_col, 1, cv2.LINE_AA)


def _draw_lookahead_reticle(canvas: np.ndarray, telem: LKATelemetry, h: int, w: int) -> None:
    """Draw look-ahead target point on the lane center trajectory."""
    # Scale from 640x360 model space if needed
    sx = w / 640.0
    sy = h / 360.0

    target_y = int(0.62 * h)
    target_x = int((telem.lane_center_x or 320.0) * sx)

    # Clamp
    target_x = max(20, min(w - 20, target_x))

    # Outer dashed reticle
    cv2.circle(canvas, (target_x, target_y), 9, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.circle(canvas, (target_x, target_y), 3, COLOR_CYAN, -1, cv2.LINE_AA)

    # Crosshair ticks
    cv2.line(canvas, (target_x - 14, target_y), (target_x - 10, target_y), COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(canvas, (target_x + 10, target_y), (target_x + 14, target_y), COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(canvas, (target_x, target_y - 14), (target_x, target_y - 10), COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.line(canvas, (target_x, target_y + 10), (target_x, target_y + 14), COLOR_CYAN, 1, cv2.LINE_AA)


def _draw_ldw_banner(canvas: np.ndarray, ldw: LDWState, w: int, h: int) -> None:
    """Warning banner displayed when departure occurs."""
    is_warn = (ldw == LDWState.WARNING)
    banner_text = "LANE DEPARTURE WARNING" if is_warn else f"LANE DRIFT: {ldw.value}"
    b_color = COLOR_RED if is_warn else COLOR_AMBER

    bw, bh = 340, 32
    bx = (w - bw) // 2
    by = int(0.12 * h)

    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.85, border_color=b_color)
    cv2.putText(canvas, f"! {banner_text}", (bx + 16, by + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, b_color, 2, cv2.LINE_AA)


def _draw_diag_banner(canvas: np.ndarray, diag: FailureCondition, w: int, h: int) -> None:
    """Subtle diagnostic indicator badge."""
    bw, bh = 280, 24
    bx = (w - bw) // 2
    by = int(0.08 * h)
    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.70, border_color=COLOR_BORDER)
    cv2.putText(canvas, f"DIAG: {diag.value}", (bx + 12, by + 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_AMBER, 1, cv2.LINE_AA)


def _draw_semi_trans_box(
    canvas: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    alpha: float = 0.75,
    border_color: Tuple[int, int, int] = COLOR_BORDER,
) -> None:
    """Draws a semi-transparent dark rectangle with a clean border."""
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(canvas.shape[1], x + w), min(canvas.shape[0], y + h)

    if x2 <= x1 or y2 <= y1:
        return

    sub_img = canvas[y1:y2, x1:x2]
    bg_rect = np.full(sub_img.shape, COLOR_BG_DARK, dtype=np.uint8)
    cv2.addWeighted(bg_rect, alpha, sub_img, 1.0 - alpha, 0, sub_img)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
