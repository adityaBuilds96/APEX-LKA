"""
dashboard/components/lka_hud.py
===============================
Automotive-Grade Perception Viewport HUD Overlay for APEX LKA.

Draws high-contrast, clean HUD elements directly onto the viewport frame:
- Top-Left: System state, live indicator, FPS, Latency, Environmental status badge.
- Road Surface: Free-space drivable corridor polygon (semi-transparent cyan alpha=0.25).
- Trajectory: Look-ahead metric distance ticks (2m, 5m, 10m) and steering target reticle.
- Top-Right: LKA state, lane stability, lateral offset (m/px), heading error, confidence.
- Alerts: Lane Departure Warning (LDW) banner and diagnostic condition tags.
"""

import math
from typing import Any, Dict, List, Optional, Tuple, Union
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
    scene_perception: Optional[Any] = None,
    risk_report: Optional[Any] = None,
    detection_result: Optional[Any] = None,
    objects_3d: Optional[Any] = None,
    occupancy_grid: Optional[Any] = None,
    depth_map: Optional[np.ndarray] = None,
    show_depth_heatmap: bool = False,
    tracked_objects: Optional[Any] = None,
    trajectory_report: Optional[Any] = None,
    vio_trail: Optional[List[Tuple[float, float]]] = None,
    vehicle_pose: Optional[Any] = None,
    is_lane_retained: bool = False,
    planned_trajectory: Optional[Any] = None,
    behavior_context: Optional[Any] = None,
    rss_result: Optional[Any] = None,
    fleet_status: Optional[Any] = None,
    ota_hotswap_alert: bool = False,
) -> np.ndarray:
    """
    Renders an automotive HUD onto the camera viewport image.
    Incorporates multi-task scene perception (free-space polygon, distance ticks,
    lane boundary polylines, drivable trajectory, environmental state badge,
    multi-object bounding boxes, 3D metric labels, Track IDs, motion vectors,
    trajectory waypoints, top-down 3D occupancy grid widget, VIO trajectory trail,
    ESKF kinematic state panel, kinematic lane retention badge, and collision risk badges)
    with zero text overlap.
    """
    if image_bgr is None or image_bgr.size == 0:
        return image_bgr

    canvas = image_bgr.copy()
    h, w = canvas.shape[:2]

    # Resolve scene perception telemetry if passed directly or via telemetry object
    sp = scene_perception
    if sp is None and telemetry is not None:
        sp = getattr(telemetry, "scene_perception", None)

    # Resolve risk report and detection result
    rr = risk_report
    if rr is None and sp is not None:
        rr = getattr(sp, "risk_report", None)
    if rr is None and telemetry is not None:
        rr = getattr(telemetry, "risk_report", None)

    det_res = detection_result
    if det_res is None and sp is not None:
        det_res = getattr(sp, "detection_result", None)

    # Resolve 3D objects and occupancy grid
    objs_3d = objects_3d
    if objs_3d is None and sp is not None:
        objs_3d = getattr(sp, "objects_3d", None)

    occ_grid = occupancy_grid
    if occ_grid is None and sp is not None:
        occ_grid = getattr(sp, "occupancy_grid", None)

    d_map = depth_map
    if d_map is None and sp is not None:
        depth_res = getattr(sp, "depth_result", None)
        if depth_res is not None:
            d_map = getattr(depth_res, "depth_map", None)

    # Resolve tracked objects and trajectory prediction
    t_objs = tracked_objects
    if t_objs is None and sp is not None:
        t_objs = getattr(sp, "tracked_objects", None)
    if t_objs is None and telemetry is not None:
        t_objs = getattr(telemetry, "tracked_objects", None)

    t_rep = trajectory_report
    if t_rep is None and sp is not None:
        t_rep = getattr(sp, "trajectory_report", None)
    if t_rep is None and telemetry is not None:
        t_rep = getattr(telemetry, "trajectory_report", None)

    # Resolve sensor fusion elements (VIO trail, Kinematic Vehicle Pose, Lane Retention)
    v_trail = vio_trail
    if v_trail is None and sp is not None:
        v_trail = getattr(sp, "vio_trail", None)
    if v_trail is None and telemetry is not None:
        v_trail = getattr(telemetry, "vio_trail", None)

    v_pose = vehicle_pose
    if v_pose is None and sp is not None:
        v_pose = getattr(sp, "vehicle_pose", None)
    if v_pose is None and telemetry is not None:
        v_pose = getattr(telemetry, "vehicle_pose", None)

    lane_ret = is_lane_retained
    if not lane_ret and sp is not None:
        lane_ret = getattr(sp, "is_lane_retained", False)
    if not lane_ret and telemetry is not None:
        lane_ret = getattr(telemetry, "is_lane_retained", False)

    # Resolve Section 5 planning elements
    p_traj = planned_trajectory
    if p_traj is None and sp is not None:
        p_traj = getattr(sp, "planned_trajectory", None)
    if p_traj is None and telemetry is not None:
        p_traj = getattr(telemetry, "planned_trajectory", None)

    b_ctx = behavior_context
    if b_ctx is None and sp is not None:
        b_ctx = getattr(sp, "behavior_context", None)
    if b_ctx is None and telemetry is not None:
        b_ctx = getattr(telemetry, "behavior_context", None)

    rss_res = rss_result
    if rss_res is None and sp is not None:
        rss_res = getattr(sp, "rss_result", None)
    if rss_res is None and telemetry is not None:
        rss_res = getattr(telemetry, "rss_result", None)

    # ── 0. Optional Depth Heatmap Overlay (Magma False-Color α=0.35) ──────
    if show_depth_heatmap and d_map is not None and d_map.size > 0:
        try:
            d_norm = np.clip(d_map * 255.0, 0, 255).astype(np.uint8)
            heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)
            if heatmap.shape[:2] != (h, w):
                heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
            cv2.addWeighted(heatmap, 0.35, canvas, 0.65, 0, canvas)
        except Exception:
            pass

    # ── 0.5 VIO Trajectory Trail (Blue/Cyan Breadcrumb Trail) ─────────────
    if v_trail:
        _draw_vio_trajectory_trail(canvas, v_trail, w, h)

    # ── 0.6 Planned Path Trajectory Spline (Bright Neon Green Line Ahead) ─
    if p_traj is not None:
        _draw_planned_trajectory_spline(canvas, p_traj, w, h)

    # ── 1. Free-Space Drivable Corridor Overlay (Cyan Polygon α=0.25) ─────
    if sp is not None and getattr(sp, "free_space", None) is not None:
        _draw_free_space_corridor(canvas, sp.free_space, alpha=0.25)

    # ── 1.5 Crisp Lane Boundary Polylines (Cyan/Orange) & Drivable Center (Yellow)
    s_paths = getattr(sp, "smoothed_paths", None) if sp is not None else None
    if s_paths is not None:
        _draw_lane_boundaries(canvas, s_paths, h, w)

    # ── 2. Metric Lookahead Distance Ticks (2m, 5m, 10m) ───────────────────
    if sp is not None and getattr(sp, "distance_profile", None) is not None:
        c_poly = None
        if getattr(sp, "smoothed_paths", None) is not None:
            c_poly = sp.smoothed_paths.center_poly
        _draw_distance_ticks(canvas, sp.distance_profile, c_poly, h, w)

    # ── 3. Multi-Object Detections, Track IDs, Motion Vectors & Waypoints ──
    if rr is not None or det_res is not None or objs_3d is not None or t_objs is not None:
        _draw_object_detections(
            canvas,
            det_res=det_res,
            risk_report=rr,
            objects_3d=objs_3d,
            tracked_objects=t_objs,
            trajectory_report=t_rep,
        )

    # ── 3.5 Mobileye RSS Safety Distance Envelope ──────────────────────────
    if rss_res is not None:
        _draw_rss_safety_envelope(canvas, rss_res, t_objs, w, h)

    # ── 4. Mini Top-Down 3D BEV Occupancy Grid Widget (Bottom-Right) ──────
    if occ_grid is not None:
        _draw_occupancy_grid_widget(canvas, occ_grid, w, h)

    # ── 5. Road Reticle & Trajectory Look-Ahead ───────────────────────────
    if telemetry is not None:
        if (
            telemetry.stability_state in (LaneStabilityState.STABLE, LaneStabilityState.UNCERTAIN)
            and telemetry.lane_center_x is not None
        ):
            _draw_lookahead_reticle(canvas, telemetry, h, w)

        # ── Critical LDW Warning Banner (Active departure only) ─────────
        if telemetry.ldw_state in (LDWState.WARNING, LDWState.DRIFT_LEFT, LDWState.DRIFT_RIGHT):
            _draw_ldw_banner(canvas, telemetry.ldw_state, w, h)

    # ── 5.5 Lane Retention Active Status Badge (ESKF Dead Reckoning Hold) ──
    if lane_ret:
        _draw_lane_retention_badge(canvas, w, h)

    # ── 5.6 Autonomous FSM Behavior Mode Badge (Top-Center) ────────────────
    if b_ctx is not None or rss_res is not None:
        _draw_fsm_behavior_badge(canvas, b_ctx, rss_res, w, h)

    # ── 6. Obstacle Emergency Brake / Trajectory Alert Banner ─────────────
    if t_rep is not None:
        _draw_trajectory_alert_banner(canvas, t_rep, w, h)
    elif rr is not None and getattr(rr, "emergency_brake_flag", False):
        _draw_obstacle_brake_banner(canvas, w, h)

    # ── 7. Minimal Corner Performance Badge ───────────────────────────────
    fps_val = perf.camera_fps if (perf and perf.camera_fps > 0) else (perf.ui_fps if perf else 0.0)
    lat_val = perf.pipeline_latency_ms if (perf and perf.pipeline_latency_ms > 0) else (telemetry.total_ms if telemetry else 0.0)

    bw, bh = _draw_minimal_corner_badge(canvas, mode_label, fps_val, lat_val, x=14, y=14)

    # ── 8. Environmental Adaptation Status Badge ───────────────────────────
    curr_y = 14 + bh + 6
    if sp is not None and getattr(sp, "environmental", None) is not None:
        _draw_environmental_badge(canvas, sp.environmental, x=14, y=curr_y, bw=bw)
        curr_y += 36

    # ── 9. ESKF Kinematic Vehicle State Panel ─────────────────────────────
    if v_pose is not None:
        _draw_kinematic_state_panel(canvas, v_pose, x=14, y=curr_y, bw=bw)

    # ── 10. Fleet & OTA Status Panel & Hotswap Banner ─────────────────────
    if fleet_status is not None:
        _draw_fleet_ota_panel(canvas, fleet_status, x=14, y=h - 85, bw=bw)
    if ota_hotswap_alert:
        _draw_ota_hotswap_banner(canvas, w, h)

    return canvas


def _draw_free_space_corridor(
    canvas: np.ndarray,
    free_space: Any,
    alpha: float = 0.25,
) -> None:
    """Overlay free-space drivable corridor as a semi-transparent cyan polygon."""
    polygon = getattr(free_space, "polygon", None)
    if polygon is None or len(polygon) < 3:
        return

    h, w = canvas.shape[:2]
    # Scale from 640x360 model space if needed
    sx = w / 640.0
    sy = h / 360.0

    pts = np.asarray(polygon, dtype=np.float32)
    if pts.ndim == 3 and pts.shape[1] == 1:
        pts = pts[:, 0, :]

    pts[:, 0] *= sx
    pts[:, 1] *= sy
    pts_int = np.round(pts).astype(np.int32)

    overlay = canvas.copy()
    cv2.fillPoly(overlay, [pts_int], COLOR_CYAN)
    cv2.addWeighted(overlay, alpha, canvas, 1.0 - alpha, 0, canvas)
    cv2.polylines(canvas, [pts_int], True, COLOR_CYAN, 1, cv2.LINE_AA)


def _draw_distance_ticks(
    canvas: np.ndarray,
    dist_profile: Any,
    center_poly: Optional[np.ndarray],
    h: int,
    w: int,
) -> None:
    """Render lookahead distance ticks (2m, 5m, 10m) along center trajectory path."""
    y_map: Dict[float, int] = getattr(dist_profile, "lookahead_y_map", {})
    if not y_map:
        return

    sx = w / 640.0
    sy = h / 360.0

    # Draw ticks for 2m, 5m, 10m
    tick_half_widths = {2.0: 24, 5.0: 18, 10.0: 12}

    for d in (2.0, 5.0, 10.0):
        if d not in y_map:
            continue
        y_model = y_map[d]
        yc = int(y_model * sy)
        if yc <= 0 or yc >= h - 2:
            continue

        if center_poly is not None:
            try:
                xc_model = float(np.polyval(center_poly, y_model))
                xc = int(xc_model * sx)
            except Exception:
                xc = w // 2
        else:
            xc = w // 2

        xc = int(np.clip(xc, 30, w - 30))
        half_w = int(tick_half_widths.get(d, 16) * sx)

        # Draw tick cross-bar
        cv2.line(canvas, (xc - half_w, yc), (xc + half_w, yc), COLOR_CYAN, 2, cv2.LINE_AA)

        # Distance text tag
        tag_str = f"{int(d)}m"
        font_scale = max(0.32, min(0.45, 0.38 * (h / 360.0)))
        (tw, th), _ = cv2.getTextSize(tag_str, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        tx = xc + half_w + 5
        ty = yc + th // 2
        if tx + tw < w - 5:
            # Text backing for high legibility
            cv2.rectangle(
                canvas,
                (tx - 2, yc - th // 2 - 2),
                (tx + tw + 2, yc + th // 2 + 2),
                COLOR_BG_DARK,
                -1,
            )
            cv2.putText(
                canvas,
                tag_str,
                (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                COLOR_TEXT_MAIN,
                1,
                cv2.LINE_AA,
            )


def _draw_environmental_badge(
    canvas: np.ndarray,
    env_state: Any,
    x: int,
    y: int,
    bw: int,
) -> None:
    """Render environmental status badge directly under corner performance badge."""
    h, w = canvas.shape[:2]
    bh = int(max(24, min(32, h * 0.055)))
    scale = max(0.30, min(0.48, (h / 720.0) * 0.40))

    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.82)

    # Determine condition text and dot color
    badge_text = getattr(env_state, "badge_text", "NORMAL")
    if "NIGHT" in badge_text:
        dot_color = COLOR_CYAN
    elif "GLARE" in badge_text or "SHADOW" in badge_text:
        dot_color = COLOR_AMBER
    else:
        dot_color = COLOR_GREEN

    dot_r = max(3, int(4 * (h / 720.0)))
    cv2.circle(canvas, (x + 12, y + bh // 2), dot_r, dot_color, -1, cv2.LINE_AA)

    label = f"ENV  {badge_text}"
    cv2.putText(
        canvas,
        label,
        (x + 22, y + int(bh * 0.65)),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        COLOR_TEXT_MAIN,
        1,
        cv2.LINE_AA,
    )


def _draw_minimal_corner_badge(
    canvas: np.ndarray,
    mode: str,
    fps: float,
    latency_ms: float,
    x: int = 14,
    y: int = 14,
) -> Tuple[int, int]:
    """Minimal, non-intrusive corner performance badge, resolution-scaled."""
    h, w = canvas.shape[:2]
    scale = max(0.32, min(0.60, (h / 720.0) * 0.42))
    bw = int(max(130, min(180, w * 0.22)))
    bh = int(max(48, min(70, h * 0.12)))

    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.82)

    # Dot + Mode label
    dot_color = COLOR_GREEN if mode in ("LIVE", "REPLAY") else COLOR_CYAN
    dot_r = max(3, int(4 * (h / 720.0)))
    cv2.circle(canvas, (x + 12, y + int(bh * 0.30)), dot_r, dot_color, -1, cv2.LINE_AA)
    cv2.putText(canvas, mode, (x + 22, y + int(bh * 0.35)),
                cv2.FONT_HERSHEY_SIMPLEX, scale, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)

    # FPS
    fps_str = f"FPS  {fps:.1f}" if fps > 0 else "FPS  --"
    cv2.putText(canvas, fps_str, (x + 12, y + int(bh * 0.65)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)

    # Latency
    lat_str = f"LAT  {latency_ms:.0f} ms" if latency_ms > 0 else "LAT  --"
    cv2.putText(canvas, lat_str, (x + 12, y + int(bh * 0.92)),
                cv2.FONT_HERSHEY_SIMPLEX, scale * 0.9, COLOR_CYAN, 1, cv2.LINE_AA)

    return bw, bh


# ═══════════════════════════════════════════════════════════════════════════
# HUD Component Primitives
# ═══════════════════════════════════════════════════════════════════════════

def _draw_lookahead_reticle(canvas: np.ndarray, telem: LKATelemetry, h: int, w: int) -> None:
    """Draw look-ahead target point on the lane center trajectory."""
    sx = w / 640.0
    target_y = int(0.62 * h)
    target_x = int((telem.lane_center_x or 320.0) * sx)
    target_x = max(20, min(w - 20, target_x))

    cv2.circle(canvas, (target_x, target_y), 9, COLOR_CYAN, 1, cv2.LINE_AA)
    cv2.circle(canvas, (target_x, target_y), 3, COLOR_CYAN, -1, cv2.LINE_AA)

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


def _draw_lane_boundaries(
    canvas: np.ndarray,
    smoothed_paths: Any,
    h: int,
    w: int,
) -> None:
    """
    Render crisp, non-overlapping lane boundary polylines:
    - Left lane: Cyan (204, 180, 0)
    - Right lane: Safety Orange (0, 140, 255)
    - Center drivable trajectory: Yellow (0, 255, 255)
    """
    if smoothed_paths is None:
        return

    l_poly = getattr(smoothed_paths, "left_poly", None)
    r_poly = getattr(smoothed_paths, "right_poly", None)
    c_poly = getattr(smoothed_paths, "center_poly", None)

    if l_poly is None and r_poly is None and c_poly is None:
        return

    y_start = int(0.38 * h)
    y_vals = np.linspace(y_start, h - 1, num=40, dtype=np.float32)

    # Left lane boundary (Cyan)
    if l_poly is not None:
        try:
            x_left = np.polyval(l_poly, y_vals * (360.0 / h)) * (w / 640.0)
            pts_l = np.column_stack([x_left, y_vals]).astype(np.int32)
            valid_l = (pts_l[:, 0] >= 0) & (pts_l[:, 0] < w)
            if np.count_nonzero(valid_l) > 5:
                cv2.polylines(canvas, [pts_l[valid_l]], False, COLOR_CYAN, 3, cv2.LINE_AA)
        except Exception:
            pass

    # Right lane boundary (Safety Orange)
    if r_poly is not None:
        try:
            x_right = np.polyval(r_poly, y_vals * (360.0 / h)) * (w / 640.0)
            pts_r = np.column_stack([x_right, y_vals]).astype(np.int32)
            valid_r = (pts_r[:, 0] >= 0) & (pts_r[:, 0] < w)
            if np.count_nonzero(valid_r) > 5:
                cv2.polylines(canvas, [pts_r[valid_r]], False, (0, 140, 255), 3, cv2.LINE_AA)
        except Exception:
            pass

    # Center drivable trajectory (Yellow)
    if c_poly is not None:
        try:
            x_center = np.polyval(c_poly, y_vals * (360.0 / h)) * (w / 640.0)
            pts_c = np.column_stack([x_center, y_vals]).astype(np.int32)
            valid_c = (pts_c[:, 0] >= 0) & (pts_c[:, 0] < w)
            if np.count_nonzero(valid_c) > 5:
                cv2.polylines(canvas, [pts_c[valid_c]], False, (0, 255, 255), 2, cv2.LINE_AA)
        except Exception:
            pass


def _draw_object_detections(
    canvas: np.ndarray,
    det_res: Any = None,
    risk_report: Any = None,
    objects_3d: Optional[List[Any]] = None,
    tracked_objects: Optional[List[Any]] = None,
    trajectory_report: Optional[Any] = None,
) -> None:
    """
    Renders multi-object bounding boxes, Track IDs, 3D metric distances,
    velocity motion vectors, and predicted trajectory waypoints.
    """
    h_canvas, w_canvas = canvas.shape[:2]

    # Map trajectory reports by track_id
    traj_map = {}
    if trajectory_report is not None:
        p_trajs = getattr(trajectory_report, "predicted_trajectories", [])
        for traj in p_trajs:
            traj_map[traj.track_id] = traj

    # Priority 1: Persistent Tracked Objects with Trajectory Prediction
    if tracked_objects:
        fx = 500.0 * (w_canvas / 640.0)
        fy = 500.0 * (h_canvas / 360.0)
        cx_cam = w_canvas / 2.0
        cy_cam = h_canvas / 2.0
        cam_h = 1.25  # Camera height (meters)

        for tobj in tracked_objects:
            box = getattr(tobj, "box_2d", None)
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1 = max(0, min(w_canvas - 1, x1))
            y1 = max(0, min(h_canvas - 1, y1))
            x2 = max(0, min(w_canvas - 1, x2))
            y2 = max(0, min(h_canvas - 1, y2))

            t_id = tobj.track_id
            c_name = tobj.class_name
            traj = traj_map.get(t_id, None)

            # Determine risk and box color
            if traj is not None:
                risk_lvl = getattr(traj.collision_risk, "name", str(traj.collision_risk))
                if "IMMINENT" in risk_lvl:
                    box_color = COLOR_RED
                elif "WARNING" in risk_lvl:
                    box_color = COLOR_AMBER
                elif "MONITOR" in risk_lvl:
                    box_color = COLOR_CYAN
                else:
                    box_color = COLOR_GREEN
            else:
                box_color = COLOR_CYAN

            # Draw bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2), box_color, 2, cv2.LINE_AA)

            # Corner accents
            c_len = max(6, min(14, (x2 - x1) // 4))
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2, cv2.LINE_AA)

            # Metric distance Z
            z_m = 0.0
            if getattr(tobj, "bbox_3d", None) is not None and len(tobj.bbox_3d) >= 3:
                z_m = float(tobj.bbox_3d[2])

            # Track ID badge text: e.g. "ID: 04 | 2.4 m | 1.8 s TTI"
            if traj is not None and traj.tti_s < 30.0:
                badge_text = f"ID: {t_id:02d} | {z_m:.1f} m | {traj.tti_s:.1f} s TTI"
            elif z_m > 0.0:
                badge_text = f"ID: {t_id:02d} | {z_m:.1f} m | {c_name.upper()}"
            else:
                badge_text = f"ID: {t_id:02d} | {c_name.upper()}"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            bx1 = x1
            by1 = max(16, y1 - th - 6)
            _draw_semi_trans_box(canvas, bx1, by1, tw + 8, th + 6, alpha=0.85, border_color=box_color)
            cv2.putText(canvas, badge_text, (bx1 + 4, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

            # Motion vector arrow showing object velocity direction (vx, vz)
            vx, vy, vz = getattr(tobj, "velocity_3d", (0.0, 0.0, 0.0))
            spd_2d = float(np.hypot(vx, vz))
            cx_box = (x1 + x2) // 2
            cy_box = y2

            if spd_2d > 0.25:
                # Lateral displacement in pixels (+right, -left)
                # Longitudinal displacement in pixels (+forward away: -dy, -closing in: +dy)
                arrow_scale = 8.0 * (h_canvas / 360.0)
                dx = int(np.clip(vx * arrow_scale, -70, 70))
                dy = int(np.clip(-vz * arrow_scale, -70, 70))
                arrow_end = (cx_box + dx, cy_box + dy)
                arr_col = (0, 255, 255) if box_color != COLOR_RED else COLOR_RED
                cv2.arrowedLine(canvas, (cx_box, cy_box), arrow_end, arr_col, 2, cv2.LINE_AA, tipLength=0.25)
                cv2.putText(
                    canvas,
                    f"{spd_2d:.1f}m/s",
                    (arrow_end[0] + 3, arrow_end[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Projected future position dots along predicted object pathways
            if traj is not None and getattr(traj, "waypoints", None):
                prev_wp_pt = (cx_box, cy_box)
                for wp in traj.waypoints:
                    if wp.z > 0.5:
                        u_wp = int(cx_cam + (fx * wp.x) / wp.z)
                        v_wp = int(cy_cam + (fy * cam_h) / wp.z)
                        if 0 <= u_wp < w_canvas and 0 <= v_wp < h_canvas:
                            cv2.circle(canvas, (u_wp, v_wp), 3, box_color, -1, cv2.LINE_AA)
                            cv2.circle(canvas, (u_wp, v_wp), 5, (255, 255, 255), 1, cv2.LINE_AA)
                            cv2.line(canvas, prev_wp_pt, (u_wp, v_wp), box_color, 1, cv2.LINE_AA)
                            prev_wp_pt = (u_wp, v_wp)
        return

    # Priority 2: Assessed objects with risk report
    assessed_objects = getattr(risk_report, "assessed_objects", []) if risk_report is not None else []
    objs_3d_list = objects_3d or []

    if assessed_objects:
        for i, obj in enumerate(assessed_objects):
            det = obj.detection
            x1, y1, x2, y2 = [int(v) for v in det.box]
            x1 = max(0, min(w_canvas - 1, x1))
            y1 = max(0, min(h_canvas - 1, y1))
            x2 = max(0, min(w_canvas - 1, x2))
            y2 = max(0, min(h_canvas - 1, y2))

            risk = obj.risk_score
            if risk < 30.0:
                color = COLOR_GREEN
            elif risk <= 70.0:
                color = COLOR_AMBER
            else:
                color = COLOR_RED

            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            c_len = max(6, min(14, (x2 - x1) // 4))
            cv2.line(canvas, (x1, y1), (x1 + c_len, y1), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x1, y1), (x1, y1 + c_len), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2 - c_len, y2), (255, 255, 255), 2, cv2.LINE_AA)
            cv2.line(canvas, (x2, y2), (x2, y2 - c_len), (255, 255, 255), 2, cv2.LINE_AA)

            z_metric = None
            if i < len(objs_3d_list):
                z_metric = getattr(objs_3d_list[i], "z", None)

            if z_metric is not None and z_metric > 0.0:
                dist_str = f"Z={z_metric:.1f}m"
            elif obj.distance_mm > 0:
                dist_str = f"{obj.distance_mm / 1000.0:.1f}m"
            else:
                dist_str = ""

            dist_suffix = f" ({dist_str})" if dist_str else ""
            badge_text = f"{det.class_name.upper()} {risk:.0f}%{dist_suffix}"

            (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            bx1 = x1
            by1 = max(16, y1 - th - 6)
            bx2 = bx1 + tw + 8
            by2 = by1 + th + 6

            _draw_semi_trans_box(canvas, bx1, by1, bx2 - bx1, by2 - by1, alpha=0.85, border_color=color)
            cv2.putText(canvas, badge_text, (bx1 + 4, by2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    elif objs_3d_list:
        for obj3d in objs_3d_list:
            box = getattr(obj3d, "box_2d", None)
            if box is None or len(box) < 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1 = max(0, min(w_canvas - 1, x1))
            y1 = max(0, min(h_canvas - 1, y1))
            x2 = max(0, min(w_canvas - 1, x2))
            y2 = max(0, min(h_canvas - 1, y2))

            color = COLOR_CYAN
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{obj3d.class_name.upper()} (Z={obj3d.z:.1f}m)"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            by1 = max(16, y1 - th - 6)
            _draw_semi_trans_box(canvas, x1, by1, tw + 8, th + 6, alpha=0.85, border_color=color)
            cv2.putText(canvas, label, (x1 + 4, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)

    elif det_res is not None and hasattr(det_res, "get_detections"):
        for det in det_res.get_detections():
            x1, y1, x2, y2 = [int(v) for v in det.box]
            x1 = max(0, min(w_canvas - 1, x1))
            y1 = max(0, min(h_canvas - 1, y1))
            x2 = max(0, min(w_canvas - 1, x2))
            y2 = max(0, min(h_canvas - 1, y2))

            color = COLOR_CYAN
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{det.class_name.upper()} {det.score*100:.0f}%"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            by1 = max(16, y1 - th - 6)
            _draw_semi_trans_box(canvas, x1, by1, tw + 8, th + 6, alpha=0.85, border_color=color)
            cv2.putText(canvas, label, (x1 + 4, by1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_trajectory_alert_banner(canvas: np.ndarray, traj_rep: Any, w: int, h: int) -> None:
    """Banner displayed when an imminent trajectory collision is predicted."""
    if traj_rep is None:
        return
    max_risk = getattr(traj_rep, "max_risk_level", None)
    risk_val = getattr(max_risk, "name", str(max_risk))
    if "IMMINENT" not in risk_val and "WARNING" not in risk_val:
        return

    is_imminent = "IMMINENT" in risk_val
    b_color = COLOR_RED if is_imminent else COLOR_AMBER
    action = getattr(traj_rep, "recommended_action", None)
    act_name = getattr(action, "name", str(action))
    track_id = getattr(traj_rep, "critical_track_id", 0)
    tti_s = getattr(traj_rep, "min_tti_s", 0.0)

    bw, bh = 480, 38
    bx = (w - bw) // 2
    by = int(0.18 * h)

    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.90, border_color=b_color)
    msg = f"COLLISION ALERT: TRK#{track_id:02d} | TTI {tti_s:.1f}s | {act_name}"
    cv2.putText(canvas, f"! {msg}", (bx + 14, by + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, b_color, 2, cv2.LINE_AA)


def _draw_occupancy_grid_widget(
    canvas: np.ndarray,
    occ_grid: Any,
    w: int,
    h: int,
) -> None:
    """
    Renders a mini top-down 3D Occupancy Grid widget in the bottom-right corner of the HUD.
    """
    if occ_grid is None or not hasattr(occ_grid, "to_rgb"):
        return

    grid_rgb = occ_grid.to_rgb()  # shape (150, 100, 3) RGB uint8
    if grid_rgb is None or grid_rgb.size == 0:
        return

    # Flip vertically so forward lookahead (Z=15m) points upward
    grid_bgr = cv2.cvtColor(grid_rgb, cv2.COLOR_RGB2BGR)
    grid_bev = np.flipud(grid_bgr)

    # Widget sizing: 90 x 135 px
    gw, gh = 90, 135
    grid_resized = cv2.resize(grid_bev, (gw, gh), interpolation=cv2.INTER_NEAREST)

    # Widget location: bottom-right corner
    pad = 8
    margin_right = 14
    margin_bottom = 14
    bw = gw + pad * 2
    bh = gh + pad * 2 + 16  # extra space for title

    bx = w - bw - margin_right
    by = h - bh - margin_bottom

    if bx < 0 or by < 0:
        return

    # Semi-transparent backing
    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.85, border_color=COLOR_BORDER)

    # Title header
    cv2.putText(
        canvas,
        "3D OCCUPANCY",
        (bx + pad, by + 14),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        COLOR_TEXT_MUTED,
        1,
        cv2.LINE_AA,
    )

    # Paste BEV grid
    gx = bx + pad
    gy = by + 20
    canvas[gy:gy + gh, gx:gx + gw] = grid_resized
    cv2.rectangle(canvas, (gx, gy), (gx + gw, gy + gh), COLOR_BORDER, 1, cv2.LINE_AA)

    # Ego vehicle marker at bottom-center (triangle pointing up)
    ego_x = gx + gw // 2
    ego_y = gy + gh - 3
    pts_ego = np.array([
        [ego_x, ego_y - 6],
        [ego_x - 4, ego_y],
        [ego_x + 4, ego_y],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [pts_ego], COLOR_CYAN)

    # Range labels: 15m at top-right, 0m at bottom-right
    cv2.putText(canvas, "15m", (gx + gw - 24, gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, "0m", (gx + gw - 18, gy + gh - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.28, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)


def _draw_obstacle_brake_banner(canvas: np.ndarray, w: int, h: int) -> None:
    """Flashing high-priority emergency brake banner for imminent obstacle collision."""
    bw, bh = 420, 38
    bx = (w - bw) // 2
    by = int(0.20 * h)

    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.90, border_color=COLOR_RED)
    cv2.putText(canvas, "!! OBSTACLE BRAKE WARNING !!", (bx + 18, by + 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, COLOR_RED, 2, cv2.LINE_AA)


def _draw_vio_trajectory_trail(
    canvas: np.ndarray,
    trail_pts: List[Tuple[float, float]],
    w: int,
    h: int,
) -> None:
    """Render VIO ego-motion relative displacement breadcrumb trail."""
    if not trail_pts or len(trail_pts) < 2:
        return

    pts_px = []
    base_x = w // 2
    base_y = h - 25

    for pt in trail_pts:
        dx, dy = pt[0], pt[1]
        px = int(base_x - dx * 35.0)
        py = int(base_y - dy * 12.0)
        if 0 <= px < w and 0 <= py < h:
            pts_px.append((px, py))

    if len(pts_px) >= 2:
        pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(canvas, [pts_arr], False, (220, 160, 40), 2, cv2.LINE_AA)
        for p in pts_px[::3]:
            cv2.circle(canvas, p, 3, (255, 200, 50), -1, cv2.LINE_AA)


def _draw_lane_retention_badge(canvas: np.ndarray, w: int, h: int) -> None:
    """Render glowing retention status badge when ESKF dead reckoning hold is active."""
    bw, bh = 340, 32
    bx = (w - bw) // 2
    by = 52

    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.88, border_color=COLOR_AMBER)
    cv2.circle(canvas, (bx + 18, by + bh // 2), 6, COLOR_AMBER, -1, cv2.LINE_AA)
    cv2.putText(
        canvas,
        "LANE RETENTION ACTIVE (ESKF)",
        (bx + 34, by + 21),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        COLOR_TEXT_MAIN,
        2,
        cv2.LINE_AA,
    )


def _draw_kinematic_state_panel(
    canvas: np.ndarray,
    pose: Any,
    x: int = 14,
    y: int = 100,
    bw: int = 190,
) -> None:
    """Render 100 Hz ESKF vehicle state telemetry panel."""
    bh = 74
    _draw_semi_trans_box(canvas, x, y, bw, bh, alpha=0.85, border_color=COLOR_BORDER)

    vx = getattr(pose, "vx", 0.0)
    wz = getattr(pose, "yaw_rate", 0.0)
    slip = getattr(pose, "slip_angle_deg", 0.0)
    stat = getattr(pose, "localization_status", 0)

    stat_map = {
        0: ("INIT", COLOR_TEXT_MUTED),
        1: ("DEAD RECKONING", COLOR_AMBER),
        2: ("VIO ACTIVE", COLOR_CYAN),
        3: ("GNSS FIX", COLOR_CYAN),
        4: ("RTK LOCKED", COLOR_GREEN),
    }
    s_label, s_color = stat_map.get(stat, ("ACTIVE", COLOR_CYAN))

    cv2.putText(canvas, "ESKF KINEMATICS", (x + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, s_label, (x + bw - 8 - len(s_label) * 6, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, s_color, 1, cv2.LINE_AA)

    t1 = f"Vx: {vx * 3.6:.1f} km/h"
    t2 = f"Yaw: {math.degrees(wz):+.1f} deg/s"
    t3 = f"Slip: {slip:+.1f} deg"

    cv2.putText(canvas, t1, (x + 8, y + 33), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(canvas, t2, (x + 8, y + 49), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)
    cv2.putText(canvas, t3, (x + 8, y + 65), cv2.FONT_HERSHEY_SIMPLEX, 0.36, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)


def _draw_planned_trajectory_spline(
    canvas: np.ndarray,
    trajectory: Any,
    w: int,
    h: int,
) -> None:
    """
    Renders the 3.0-second planned path spline as a bright neon green trajectory line ahead.
    """
    waypoints = getattr(trajectory, "waypoints", None)
    if waypoints is None and isinstance(trajectory, list):
        waypoints = trajectory
    if not waypoints:
        return

    pts_px: List[Tuple[int, int]] = []
    for wp in waypoints:
        x_m = float(getattr(wp, "x", 0.0))
        y_m = float(getattr(wp, "y", 0.0))
        if x_m < 0.2:
            continue
        # Inverse perspective camera projection
        norm_dist = np.clip((x_m - 0.5) / 18.0, 0.0, 1.0)
        py = int(h - 12 - (h * 0.46) * (norm_dist ** 0.82))
        lateral_px = (y_m * 200.0) / max(0.8, x_m * 0.20)
        px = int(w / 2.0 + lateral_px)
        if 0 <= px < w and 0 <= py < h:
            pts_px.append((px, py))

    if len(pts_px) >= 2:
        pts_arr = np.array(pts_px, dtype=np.int32).reshape((-1, 1, 2))
        # Draw glowing path outline
        cv2.polylines(canvas, [pts_arr], False, (20, 160, 40), 5, cv2.LINE_AA)
        # Main neon green planned path
        cv2.polylines(canvas, [pts_arr], False, (60, 255, 90), 2, cv2.LINE_AA)

        # Draw lookahead target point circle
        end_pt = pts_px[-1]
        cv2.circle(canvas, end_pt, 4, (60, 255, 90), -1, cv2.LINE_AA)
        cv2.circle(canvas, end_pt, 6, (20, 160, 40), 1, cv2.LINE_AA)


def _draw_rss_safety_envelope(
    canvas: np.ndarray,
    rss_result: Any,
    tracked_objects: Optional[List[Any]],
    w: int,
    h: int,
) -> None:
    """
    Renders the Mobileye RSS safety envelope boundary box around lead vehicle threats.
    """
    override = bool(getattr(rss_result, "override_active", False))
    status = str(getattr(rss_result, "safety_status", "SAFE"))
    if not override and status not in ("CAUTION", "OVERRIDE_BRAKE"):
        return

    lead_id = getattr(rss_result, "lead_track_id", None)
    target_obj = None

    if tracked_objects and lead_id is not None:
        for obj in tracked_objects:
            if getattr(obj, "track_id", None) == lead_id:
                target_obj = obj
                break

    if target_obj is None and tracked_objects:
        # Find closest object ahead in corridor
        ahead_objs = [o for o in tracked_objects if getattr(o, "z_m", 0.0) > 0.5]
        if ahead_objs:
            target_obj = min(ahead_objs, key=lambda o: getattr(o, "z_m", 999.0))

    if target_obj is not None:
        box_2d = getattr(target_obj, "bbox_2d", None)
        if box_2d is None:
            box_2d = getattr(target_obj, "box_2d", None)

        if box_2d is not None and len(box_2d) >= 4:
            x1, y1, x2, y2 = [int(v) for v in box_2d[:4]]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            # Envelope color based on severity
            env_color = COLOR_RED if override else COLOR_AMBER
            # Draw outer safety boundary bounding box
            pad = 6
            bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
            bx2, by2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)

            cv2.rectangle(canvas, (bx1, by1), (bx2, by2), env_color, 2, cv2.LINE_AA)

            # RSS Label
            d_act = getattr(rss_result, "d_actual_long", 0.0)
            d_min = getattr(rss_result, "d_min_long", 0.0)
            lbl = f"RSS BRAKE: {d_act:.1f}m < {d_min:.1f}m" if override else f"RSS CAUTION: {d_act:.1f}m"

            (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
            cv2.rectangle(canvas, (bx1, max(0, by1 - 18)), (bx1 + tw + 6, by1), COLOR_BG_DARK, -1)
            cv2.putText(canvas, lbl, (bx1 + 3, max(12, by1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, env_color, 1, cv2.LINE_AA)


def _draw_fsm_behavior_badge(
    canvas: np.ndarray,
    behavior_ctx: Any,
    rss_res: Any,
    w: int,
    h: int,
) -> None:
    """
    Renders top-center Autonomous Behavior Mode and RSS Health Banner pill.
    Format: MODE: ACC | SET: 15 km/h | RSS: CLEAR
    """
    state_str = "LANE_FOLLOWING"
    v_set = 4.17
    rss_override = False

    if behavior_ctx is not None:
        state_str = str(getattr(behavior_ctx, "state_name", "LANE_FOLLOWING")).replace("_", " ")
        v_set = float(getattr(behavior_ctx, "target_speed_mps", 4.17))

    if rss_res is not None:
        rss_override = bool(getattr(rss_res, "override_active", False))

    rss_str = "OVERRIDE" if rss_override else "CLEAR"
    rss_color = COLOR_RED if rss_override else COLOR_GREEN

    badge_text = f"MODE: {state_str}  |  SET: {v_set * 3.6:.0f} km/h  |  RSS: {rss_str}"

    font_scale = 0.38
    (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    bx = (w - tw - 24) // 2
    by = 12
    bw = tw + 24
    bh = th + 14

    border_col = COLOR_RED if rss_override else COLOR_CYAN
    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.88, border_color=border_col)

    # Render main text
    cv2.putText(canvas, badge_text, (bx + 12, by + bh - 6),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT_MAIN, 1, cv2.LINE_AA)


def _draw_fleet_ota_panel(
    canvas: np.ndarray,
    fleet_status: Any,
    x: int = 14,
    y: int = 275,
    bw: int = 220,
) -> None:
    """
    Renders the Fleet & Cryptographic OTA Telemetry Panel.
    Shows Active Model Hash, Sync State, and Harvest Queue Count.
    """
    if isinstance(fleet_status, dict):
        m_hash = fleet_status.get("model_version_hash", 0)
        sync_val = fleet_status.get("ota_sync_state", fleet_status.get("sync_state_enum", 0))
        disc_val = fleet_status.get("shadow_discrepancy_score", fleet_status.get("shadow_discrepancy", 0))
        queue_cnt = fleet_status.get("harvest_queue_count", 0)
        backend_val = fleet_status.get("active_backend", fleet_status.get("backend_enum", 0))
        sync_str = fleet_status.get("sync_state_str", None)
        backend_str = fleet_status.get("backend_str", None)
    else:
        m_hash = getattr(fleet_status, "model_version_hash", 0)
        sync_val = getattr(fleet_status, "ota_sync_state", getattr(fleet_status, "sync_state_enum", 0))
        disc_val = getattr(fleet_status, "shadow_discrepancy_score", getattr(fleet_status, "shadow_discrepancy", 0))
        queue_cnt = getattr(fleet_status, "harvest_queue_count", 0)
        backend_val = getattr(fleet_status, "active_backend", getattr(fleet_status, "backend_enum", 0))
        sync_str = getattr(fleet_status, "sync_state_str", None)
        backend_str = getattr(fleet_status, "backend_str", None)

    sync_map = {0: "IDLE", 1: "CHECKING", 2: "DOWNLOADING", 3: "VERIFYING", 4: "HOTSWAPPING", 5: "SYNC_OK", 6: "ERROR"}
    backend_map = {0: "CLASSICAL", 1: "PYTORCH", 2: "ONNX", 3: "TENSORRT"}

    try:
        sync_int = int(sync_val)
    except Exception:
        sync_int = 0

    if not sync_str:
        sync_str = sync_map.get(sync_int, "SYNC_OK")
    if not backend_str:
        try:
            backend_str = backend_map.get(int(backend_val), "ONNX")
        except Exception:
            backend_str = "ONNX"

    panel_h = 56
    _draw_semi_trans_box(canvas, x, y, bw, panel_h, alpha=0.85, border_color=COLOR_BORDER)

    # Header
    cv2.putText(canvas, f"FLEET OTA  |  {backend_str}", (x + 8, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, COLOR_CYAN, 1, cv2.LINE_AA)

    # Line 1: Hash & Sync State
    sync_col = COLOR_RED if sync_val == 6 else (COLOR_AMBER if sync_val in (2, 3, 4) else COLOR_GREEN)
    l1 = f"HASH: 0x{m_hash:04X}  |  {sync_str}"
    cv2.putText(canvas, l1, (x + 8, y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.33, sync_col, 1, cv2.LINE_AA)

    # Line 2: Shadow Discrepancy & Harvest Queue
    l2 = f"SHADOW: {disc_val}%  |  QUEUE: {queue_cnt} pkgs"
    cv2.putText(canvas, l2, (x + 8, y + 46),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, COLOR_TEXT_MUTED, 1, cv2.LINE_AA)


def _draw_ota_hotswap_banner(canvas: np.ndarray, w: int, h: int) -> None:
    """Banner displayed when an atomic zero-downtime model hot-swap completes."""
    bw, bh = 460, 32
    bx = (w - bw) // 2
    by = int(0.24 * h)

    _draw_semi_trans_box(canvas, bx, by, bw, bh, alpha=0.92, border_color=COLOR_GREEN)
    msg = "OTA HOTSWAP COMPLETE — ZERO-DOWNTIME MODEL ACTIVE"
    cv2.putText(canvas, f"[OK] {msg}", (bx + 14, by + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_GREEN, 1, cv2.LINE_AA)




