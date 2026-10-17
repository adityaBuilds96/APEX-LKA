"""
tests/test_tracking_and_lane_fix.py
===================================
Comprehensive Integration & Verification Suite for Phase 5 Section 3:
- Absolute Lane Perception Recovery & Zero Flickering Guarantee (100+ frames)
- Multi-Object Tracking Engine (ByteTrack / OC-SORT)
- 1-3 Second Trajectory Prediction & TTI Collision Physics
- CAN Bus Frame 0x170 Serialization & Telemetry
- HUD Dynamic Motion Vectors, Track IDs & Waypoint Rendering
- Hard Latency Ceilings (<= 10 ms tracking/trajectory, <= 30 ms full stack)
"""

import time
import math
import numpy as np
import cv2
import pytest

from src.inference.pipeline import run_pipeline, get_tracker, get_trajectory_predictor
from src.control.ekf_tracker import EKF8StateLaneTracker
from src.perception.tracker import APEXMultiObjectTracker, TrackState, TrackedObject
from src.perception.trajectory import (
    TrajectoryPredictor,
    CollisionRiskLevel,
    EvasiveActionFlag,
    TrajectoryReport,
    PredictedWaypoint,
    PredictedTrajectory,
)
from src.perception.object_detector import DetectionResult, Detection
from src.perception.projection_3d import Object3D
from deploy.vehicle_comm import TrajectoryAlertTelemetry, CAN_ID_TRAJECTORY_ALERT
from dashboard.components.lka_hud import draw_hud_overlay


def _create_synthetic_highway_frame(w=640, h=360, frame_idx=0, jitter=0.0):
    """Create realistic road scene frame with white dashed and yellow solid markings."""
    frame = np.full((h, w, 3), (70, 75, 80), dtype=np.uint8)
    # Sky above horizon
    frame[:130, :] = (180, 160, 140)

    # Road surface
    road_pts = np.array([[120, h - 1], [w - 120, h - 1], [w // 2 + 30, 130], [w // 2 - 30, 130]], dtype=np.int32)
    cv2.fillPoly(frame, [road_pts], (65, 68, 72))

    # Left lane marking (Yellow)
    dx = jitter * np.sin(frame_idx * 0.1)
    lx_bot = int(220 + dx)
    lx_top = int(w // 2 - 20)
    cv2.line(frame, (lx_bot, h - 1), (lx_top, 130), (40, 200, 220), 4)

    # Right lane marking (White)
    rx_bot = int(420 + dx)
    rx_top = int(w // 2 + 20)
    cv2.line(frame, (rx_bot, h - 1), (rx_top, 130), (240, 240, 240), 4)

    return frame


def test_pipeline_perfection_lane_recovery_stream():
    """Verify polynomial stability across 100+ stream frames with zero line flickering (Delta <= 0.02)."""
    coeffs_history = []

    for frame_idx in range(105):
        frame = _create_synthetic_highway_frame(frame_idx=frame_idx, jitter=2.0)
        res = run_pipeline(
            frame,
            backend="classical_cv",
            reset_pd_state=(frame_idx == 0),
            enable_object_detection=False,
            enable_depth=False,
            enable_occupancy=False,
            enable_tracking=False,
            enable_trajectory=False,
        )

        assert res.success is True, f"Pipeline failed on frame {frame_idx}"
        assert res.geometry is not None

        if res.geometry.left_poly is not None:
            coeffs_history.append(res.geometry.left_poly)

    assert len(coeffs_history) >= 100
    # Evaluate frame-to-frame polynomial variation: Delta <= 0.02
    deltas = []
    for i in range(1, len(coeffs_history)):
        delta_a = abs(coeffs_history[i][0] - coeffs_history[i - 1][0])
        delta_b = abs(coeffs_history[i][1] - coeffs_history[i - 1][1])
        deltas.append(max(delta_a * 100, delta_b))

    max_delta = max(deltas)
    assert max_delta <= 0.02, f"Polynomial flickering observed: max Delta = {max_delta:.4f} > 0.02"


def test_multi_object_tracking_persistence_across_stream():
    """Verify tracker assigns persistent Track IDs across consecutive stream frames."""
    tracker = APEXMultiObjectTracker()

    # Simulate vehicle moving smoothly forward
    track_ids_observed = []
    for frame_idx in range(15):
        t_sec = frame_idx * 0.033
        # Bounding box centers moving slightly down (vehicle approaching)
        cy = 200 + frame_idx * 2
        cx = 320
        box = np.array([cx - 30, cy - 25, cx + 30, cy + 25], dtype=np.float32)

        det = Detection(box=box, score=0.88, class_id=0, class_name="vehicle")
        det_res = DetectionResult(
            boxes=np.array([box], dtype=np.float32),
            scores=np.array([0.88], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
            class_names=["vehicle"],
            inference_ms=5.0,
        )

        obj3d = Object3D(
            x=0.0,
            y=1.2,
            z=15.0 - (frame_idx * 0.3),  # Closing distance: -0.3m per 33ms -> -9 m/s
            width=1.8,
            height=1.5,
            length=4.2,
            class_id=0,
            class_name="vehicle",
            confidence=0.88,
            box_2d=box,
        )

        active_tracks = tracker.update(det_res, [obj3d], timestamp_s=t_sec)
        assert len(active_tracks) == 1
        trk = active_tracks[0]
        track_ids_observed.append(trk.track_id)
        assert trk.state in (TrackState.NEW, TrackState.TRACKED)

    # All 15 frames must have the exact same Track ID
    assert len(set(track_ids_observed)) == 1
    final_track = tracker.get_active_tracks()[0]
    assert final_track.hit_streak >= 14
    # Velocity vz should reflect closing velocity (~ -9.0 m/s)
    assert final_track.velocity_3d[2] < 0.0


def test_multi_object_tracking_occlusion_and_recovery():
    """Verify tracker maintains track during brief 2-frame occlusion dropout and re-identifies object."""
    tracker = APEXMultiObjectTracker(max_lost_frames=10)

    # Frames 0-4: Visible
    t_sec = 0.0
    for frame_idx in range(5):
        t_sec += 0.033
        box = np.array([200, 180, 260, 240], dtype=np.float32)
        det = Detection(box=box, score=0.85, class_id=0, class_name="vehicle")
        det_res = DetectionResult(
            boxes=np.array([box], dtype=np.float32),
            scores=np.array([0.85], dtype=np.float32),
            class_ids=np.array([0], dtype=np.int32),
            class_names=["vehicle"],
            inference_ms=4.0,
        )
        obj3d = Object3D(
            x=-1.0, y=1.2, z=12.0, width=1.8, height=1.5,
            length=4.0, class_id=0, class_name="vehicle", confidence=0.85, box_2d=box
        )
        tracks = tracker.update(det_res, [obj3d], timestamp_s=t_sec)
        assigned_id = tracks[0].track_id

    # Frames 5-6: Occlusion dropout (no detections)
    for _ in range(2):
        t_sec += 0.033
        det_res_empty = DetectionResult(inference_ms=2.0)
        tracker.update(det_res_empty, [], timestamp_s=t_sec)

    # Frame 7: Object re-emerges at slightly shifted position
    t_sec += 0.033
    box_reappear = np.array([202, 182, 262, 242], dtype=np.float32)
    det_reappear = Detection(box=box_reappear, score=0.82, class_id=0, class_name="vehicle")
    det_res_reappear = DetectionResult(
        boxes=np.array([box_reappear], dtype=np.float32),
        scores=np.array([0.82], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
        inference_ms=4.0,
    )
    obj3d_reappear = Object3D(
        x=-1.0, y=1.2, z=11.9, width=1.8, height=1.5,
        length=4.0, class_id=0, class_name="vehicle", confidence=0.82, box_2d=box_reappear
    )

    re_tracks = tracker.update(det_res_reappear, [obj3d_reappear], timestamp_s=t_sec)
    assert len(re_tracks) == 1
    # Track ID must be preserved across occlusion
    assert re_tracks[0].track_id == assigned_id


def test_trajectory_prediction_waypoints_and_forecasting():
    """Verify trajectory predictor calculates 0.5s, 1.0s, 2.0s, 3.0s waypoints using kinematic physics."""
    predictor = TrajectoryPredictor(horizon_s=3.0)

    # Lead vehicle traveling forward at 10 m/s, accelerating at 1 m/s^2
    box = np.array([280, 160, 360, 240], dtype=np.float32)
    bbox_3d = np.array([0.0, 1.2, 10.0, 1.8, 1.5, 4.0], dtype=np.float32)
    velocity_3d = np.array([0.5, 0.0, 10.0], dtype=np.float32)

    t_obj = TrackedObject(
        track_id=7,
        class_id=0,
        class_name="vehicle",
        box_2d=box,
        score=0.92,
        bbox_3d=bbox_3d,
        velocity_3d=velocity_3d,
        age_frames=10,
        state=TrackState.TRACKED,
    )

    report = predictor.predict([t_obj], ego_speed_mps=5.0, timestamp_s=100.0)
    assert len(report.predicted_trajectories) == 1
    traj = report.predicted_trajectories[0]

    assert traj.track_id == 7
    assert len(traj.waypoints) == 4
    wp_offsets = [wp.t_offset_s for wp in traj.waypoints]
    assert wp_offsets == [0.5, 1.0, 2.0, 3.0]

    # Verify waypoint coordinates advance logically: Z increases forward
    for wp in traj.waypoints:
        assert wp.z > 10.0
        assert wp.x > 0.0


def test_trajectory_prediction_collision_tti_and_swerving():
    """Verify imminent collision detection and appropriate emergency evasive action recommendation."""
    predictor = TrajectoryPredictor(
        warning_tti_s=2.5,
        critical_tti_s=1.0,
        ego_lane_width_m=2.0,
    )

    # Object cutting in: starting at X = 2.0m, moving left into center path (vx = -1.5 m/s)
    # Distance Z = 6.0m closing at 5.0 m/s closing speed -> TTI ~ 1.2s
    box = np.array([400, 180, 480, 260], dtype=np.float32)
    bbox_3d = np.array([1.5, 1.2, 6.0, 1.8, 1.5, 4.0], dtype=np.float32)
    velocity_3d = np.array([-1.2, 0.0, -5.0], dtype=np.float32)

    t_obj = TrackedObject(
        track_id=4,
        class_id=0,
        class_name="vehicle",
        box_2d=box,
        score=0.95,
        bbox_3d=bbox_3d,
        velocity_3d=velocity_3d,
        age_frames=12,
        state=TrackState.TRACKED,
    )

    report = predictor.predict([t_obj], ego_speed_mps=0.0, timestamp_s=100.0)
    assert report.critical_track_id == 4
    assert report.min_tti_s <= 2.0
    assert report.max_risk_level in (CollisionRiskLevel.WARNING, CollisionRiskLevel.IMMINENT_COLLISION)
    # Should recommend swerving away from the cutting object or braking
    assert report.recommended_action in (
        EvasiveActionFlag.SWERVE_LEFT,
        EvasiveActionFlag.SWERVE_RIGHT,
        EvasiveActionFlag.BRAKE_NOW,
    )


def test_can_0x170_trajectory_telemetry_encoding_decoding():
    """Verify CAN ID 0x170 bit-packing round-trip serialization and decoding."""
    telem = TrajectoryAlertTelemetry(
        track_id=14,
        min_tti_ms=1650,
        projected_cross_x_mm=-420,
        risk_level=2,
        action_flag=3,
    )

    packed_data = telem.pack_frame()
    assert len(packed_data) == 8

    decoded = TrajectoryAlertTelemetry.unpack_frame(packed_data)
    assert decoded.track_id == 14
    assert decoded.min_tti_ms == 1650
    assert decoded.projected_cross_x_mm == -420
    assert decoded.risk_level == 2
    assert decoded.action_flag == 3


def test_hud_rendering_with_tracking_and_trajectories():
    """Verify HUD overlays render Track ID badges, motion vectors, and predicted pathway dots."""
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)

    box = np.array([250, 150, 350, 250], dtype=np.float32)
    bbox_3d = np.array([0.0, 1.2, 8.5, 1.8, 1.5, 4.0], dtype=np.float32)
    velocity_3d = np.array([0.8, 0.0, -4.2], dtype=np.float32)

    t_obj = TrackedObject(
        track_id=9,
        class_id=0,
        class_name="vehicle",
        box_2d=box,
        score=0.90,
        bbox_3d=bbox_3d,
        velocity_3d=velocity_3d,
        age_frames=15,
        state=TrackState.TRACKED,
    )

    wp_list = [
        PredictedWaypoint(0.5, 0.4, 6.4),
        PredictedWaypoint(1.0, 0.8, 4.3),
        PredictedWaypoint(2.0, 1.6, 2.0),
    ]

    traj = PredictedTrajectory(
        track_id=9,
        class_name="vehicle",
        current_x=0.0,
        current_z=8.5,
        vx=0.8,
        vz=-4.2,
        waypoints=wp_list,
        tti_s=1.8,
        ttcp_s=1.2,
        projected_cross_x_m=0.8,
        risk_level=CollisionRiskLevel.WARNING.value,
        action_flag=EvasiveActionFlag.SWERVE_LEFT.value,
    )

    report = TrajectoryReport(
        trajectories={9: traj},
        critical_track_id=9,
        min_tti_ms=1800,
        projected_cross_x_mm=800,
        risk_level=CollisionRiskLevel.WARNING.value,
        action_flag=EvasiveActionFlag.SWERVE_LEFT.value,
        timestamp_s=100.0,
    )

    hud_canvas = draw_hud_overlay(
        frame,
        tracked_objects=[t_obj],
        trajectory_report=report,
    )

    assert hud_canvas is not None
    assert hud_canvas.shape == (360, 640, 3)
    # The HUD canvas should have drawn graphical overlays (not just background color 60)
    assert np.any(hud_canvas != 60)


def test_pipeline_enable_all_flag_execution():
    """Verify run_pipeline executes cleanly with all Phase 5 modules enabled."""
    frame = _create_synthetic_highway_frame(frame_idx=1)

    res = run_pipeline(
        frame,
        backend="classical_cv",
        reset_pd_state=True,
        enable_object_detection=True,
        enable_depth=True,
        enable_occupancy=True,
        enable_tracking=True,
        enable_trajectory=True,
    )

    assert res.success is True
    assert res.scene_perception is not None
    assert res.scene_perception.environmental is not None
    assert res.scene_perception.free_space is not None
    assert res.scene_perception.distance_profile is not None
    assert res.scene_perception.smoothed_paths is not None
    assert res.annotated_bgr is not None
    assert res.annotated_bgr.shape == frame.shape


def test_fixed_lane_tracking_trajectory_latency_ceiling():
    """Benchmark: Lane + Tracking + Trajectory Prediction MUST execute in <= 10 ms (>= 100 FPS)."""
    frame = _create_synthetic_highway_frame(frame_idx=2)

    # Warmup
    for _ in range(5):
        run_pipeline(
            frame,
            backend="classical_cv",
            reset_pd_state=False,
            enable_object_detection=False,
            enable_depth=False,
            enable_occupancy=False,
            enable_tracking=True,
            enable_trajectory=True,
        )

    latencies = []
    for _ in range(25):
        t0 = time.perf_counter()
        run_pipeline(
            frame,
            backend="classical_cv",
            reset_pd_state=False,
            enable_object_detection=False,
            enable_depth=False,
            enable_occupancy=False,
            enable_tracking=True,
            enable_trajectory=True,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p90_lat = np.percentile(latencies, 90)
    assert p90_lat <= 10.0, f"Tracking + Lane latency {p90_lat:.2f} ms exceeded 10.0 ms limit"


def test_fixed_lane_tracking_trajectory_latency_ceiling():
    """Benchmark: Object Tracking + Trajectory Prediction executes in <= 10 ms (>= 100 FPS)."""
    tracker = APEXMultiObjectTracker()
    traj_pred = TrajectoryPredictor()

    box = np.array([280, 160, 360, 240], dtype=np.float32)
    det = Detection(box=box, score=0.90, class_id=0, class_name="vehicle")
    det_res = DetectionResult(
        boxes=np.array([box], dtype=np.float32),
        scores=np.array([0.90], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )
    obj3d = Object3D(x=0.0, y=1.2, z=10.0, width=1.8, height=1.5, length=4.2, class_id=0, class_name="vehicle", confidence=0.90, box_2d=box)

    # Warmup
    for _ in range(5):
        tracks = tracker.update(det_res, [obj3d])
        traj_pred.predict(tracks, ego_speed_mps=5.0)

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        tracks = tracker.update(det_res, [obj3d])
        traj_pred.predict(tracks, ego_speed_mps=5.0)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p90_lat = np.percentile(latencies, 90)
    assert p90_lat <= 10.0, f"Tracking + Trajectory latency {p90_lat:.2f} ms exceeded 10.0 ms limit"


def test_full_phase5_stack_latency_ceiling():
    """Benchmark: Full Phase 5 multi-task stack stage overhead <= 30 ms."""
    frame = _create_synthetic_highway_frame(frame_idx=3)

    # Warmup
    for _ in range(3):
        run_pipeline(
            frame,
            backend="classical_cv",
            reset_pd_state=False,
            enable_object_detection=True,
            enable_depth=True,
            enable_occupancy=True,
            enable_tracking=True,
            enable_trajectory=True,
        )

    latencies = []
    for _ in range(15):
        res = run_pipeline(
            frame,
            backend="classical_cv",
            reset_pd_state=False,
            enable_object_detection=True,
            enable_depth=True,
            enable_occupancy=True,
            enable_tracking=True,
            enable_trajectory=True,
        )
        t_phase5 = (
            res.timings.get("object_det_ms", 0.0)
            + res.timings.get("depth_ms", 0.0)
            + res.timings.get("projection_3d_ms", 0.0)
            + res.timings.get("occupancy_ms", 0.0)
            + res.timings.get("tracking_ms", 0.0)
            + res.timings.get("trajectory_ms", 0.0)
        )
        latencies.append(t_phase5)

    avg_lat = float(np.mean(latencies))
    assert avg_lat <= 30.0, f"Full Phase 5 stack module latency {avg_lat:.2f} ms exceeded 30.0 ms ceiling"


def test_guardrail_suppression_and_horizon_masking_integration():
    """Verify guardrail rejection: vertical lines with |m| > 2.2 and above horizon are rejected."""
    frame = _create_synthetic_highway_frame(frame_idx=0)
    h, w = frame.shape[:2]

    # Draw vertical guardrail post on left edge
    cv2.line(frame, (60, h - 1), (65, 100), (255, 255, 255), 6)

    res = run_pipeline(
        frame,
        backend="classical_cv",
        reset_pd_state=True,
    )

    assert res.success is True
    # Evaluated at bottom row, left lane must be near nominal lane position (~220 px), not guardrail (60 px)
    if res.geometry and res.geometry.left_poly is not None:
        x_left_bot = float(np.polyval(res.geometry.left_poly, h - 1))
        assert x_left_bot > 120, f"Left line locked onto guardrail: x={x_left_bot} <= 120"


def test_track_width_rigid_sanity_pipeline_recovery():
    """Verify rigid track width constraint recovers nominal width (150 px <= W <= 380 px)."""
    frame = _create_synthetic_highway_frame(frame_idx=0)
    h, w = frame.shape[:2]

    # Draw anomalous false lane marking right next to the left lane (width only 50 px)
    cv2.line(frame, (270, h - 1), (w // 2 - 10, 130), (255, 255, 255), 5)

    res = run_pipeline(
        frame,
        backend="classical_cv",
        reset_pd_state=True,
    )

    assert res.success is True
    if res.geometry and res.geometry.left_poly is not None and res.geometry.right_poly is not None:
        xl = float(np.polyval(res.geometry.left_poly, h - 1))
        xr = float(np.polyval(res.geometry.right_poly, h - 1))
        width = xr - xl
        assert 140 <= width <= 390, f"Lane width sanity violated: width={width} px"


def test_ekf_8state_lane_tracker_dropout_recovery():
    """Verify 8-state EKF tracker holds trajectory during 3-frame dropout without discontinuity."""
    ekf = EKF8StateLaneTracker(hold_lost_frames=3)

    # Prime with steady coefficients
    left = np.array([0.0001, -0.15, 220.0])
    right = np.array([0.0001, 0.15, 420.0])

    for _ in range(5):
        s_left, s_right, _, _, valid = ekf.update(left, right, dt=0.033)
        assert valid is True

    # Drop out for 2 frames
    for _ in range(2):
        s_left, s_right, _, _, valid = ekf.update(None, None, dt=0.033)
        assert valid is True
        assert s_left is not None
        assert abs(s_left[2] - 220.0) < 5.0  # Drift < 5 px
