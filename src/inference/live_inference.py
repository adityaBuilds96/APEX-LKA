"""
src/inference/live_inference.py
===============================
Standalone high-FPS inference and perception runner for APEX-LKA.
Processes webcam, video streams, or synthetic streams at >= 30 FPS.
Includes active environmental adaptation and temporal anti-flicker smoothing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS
from src.inference.pipeline import run_pipeline
from dashboard.components.lka_hud import draw_hud_overlay


def make_synthetic_road_frame(frame_idx: int, h: int = 360, w: int = 640) -> np.ndarray:
    """Generate a realistic test road frame with simulated camera motion."""
    frame = np.full((h, w, 3), (80, 80, 80), dtype=np.uint8)
    # Sky above horizon
    frame[:126, :] = (180, 160, 130)

    # Lane markings
    left_x = int(180 + 15 * np.sin(frame_idx * 0.05))
    right_x = int(460 + 15 * np.sin(frame_idx * 0.05))
    cv2.line(frame, (left_x, h - 1), (w // 2 - 25, 130), (255, 255, 255), 5)
    cv2.line(frame, (right_x, h - 1), (w // 2 + 25, 130), (255, 255, 255), 5)
    return frame


def run_live_inference(
    source: str = "webcam",
    file_path: Optional[str] = None,
    backend: str = "classical_cv",
    max_frames: int = 150,
    headless: bool = True,
    enable_object_detection: bool = False,
    enable_depth: bool = False,
    enable_occupancy: bool = False,
    show_depth_heatmap: bool = False,
    enable_tracking: bool = False,
    enable_trajectory: bool = False,
    enable_sensor_fusion: bool = False,
    enable_planner: bool = False,
    enable_shadow_mode: bool = False,
    enable_data_harvesting: bool = False,
    enable_all: bool = False,
    profile: str = "baja_2027",
) -> None:
    """Run stream inference loop and print benchmark FPS metrics."""
    if enable_all:
        enable_object_detection = True
        enable_depth = True
        enable_occupancy = True
        enable_tracking = True
        enable_trajectory = True
        enable_sensor_fusion = True
        enable_planner = True
        enable_shadow_mode = True
        enable_data_harvesting = True

    from src.config.vehicle_profile import VehicleProfileRegistry
    profile_reg = VehicleProfileRegistry()
    active_prof = profile_reg.switch_profile(profile)
    print(f"[APEX-LKA] Loaded Vehicle Profile: {active_prof.name} ({active_prof.vehicle_type}, wheelbase={active_prof.kinematics.wheelbase_m}m)")
    print(f"[APEX-LKA] Initializing Scene Perception Runner (source={source}, object_detection={enable_object_detection}, depth={enable_depth}, occupancy={enable_occupancy}, tracking={enable_tracking}, trajectory={enable_trajectory}, sensor_fusion={enable_sensor_fusion}, planner={enable_planner}, shadow_mode={enable_shadow_mode}, harvesting={enable_data_harvesting})...")

    cap = None
    use_synthetic = False

    # Check if source is directly a file path
    if source.endswith((".mp4", ".avi", ".mov", ".mkv")):
        file_path = source
        source = "video"

    if source == "video":
        target_file = file_path
        if not target_file:
            # Check for existing video in raw_videos
            v_dir = PATHS.raw_videos
            vids = list(v_dir.glob("*.mp4")) + list(v_dir.glob("*.avi")) if v_dir.exists() else []
            if vids:
                target_file = str(vids[0])
            else:
                use_synthetic = True

        if target_file and not use_synthetic:
            cap = cv2.VideoCapture(target_file)
            if not cap.isOpened():
                print(f"[APEX-LKA] Cannot open {target_file}, falling back to synthetic video stream.")
                use_synthetic = True

    elif source == "webcam":
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("[APEX-LKA] Webcam 0 unavailable, falling back to synthetic camera stream.")
            use_synthetic = True

    frame_idx = 0
    t_start = time.perf_counter()
    latencies = []

    print("[APEX-LKA] Running Multi-Task Perception Pipeline (Target: >= 30 FPS)...")

    while frame_idx < max_frames:
        t_frame_start = time.perf_counter()

        if use_synthetic or cap is None:
            frame_bgr = make_synthetic_road_frame(frame_idx)
        else:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frame_bgr = frame

        # Run pipeline with active environmental adaptation, temporal smoothing, object detection, depth, tracking, trajectory, sensor fusion, planner, shadow mode, harvesting
        res = run_pipeline(
            frame_bgr,
            backend=backend,
            reset_pd_state=(frame_idx == 0),
            enable_object_detection=enable_object_detection,
            enable_depth=enable_depth,
            enable_occupancy=enable_occupancy,
            show_depth_heatmap=show_depth_heatmap,
            enable_tracking=enable_tracking,
            enable_trajectory=enable_trajectory,
            enable_sensor_fusion=enable_sensor_fusion,
            enable_planner=enable_planner,
            enable_shadow_mode=enable_shadow_mode,
            enable_data_harvesting=enable_data_harvesting,
        )

        # Render HUD
        hud_frame = draw_hud_overlay(
            res.annotated_bgr,
            scene_perception=res.scene_perception,
            risk_report=res.risk_report,
            detection_result=res.detection_result,
            objects_3d=res.objects_3d,
            occupancy_grid=res.occupancy_grid,
            depth_map=res.depth_result.depth_map if res.depth_result is not None else None,
            show_depth_heatmap=show_depth_heatmap,
            tracked_objects=res.tracked_objects,
            trajectory_report=res.trajectory_report,
            vio_trail=res.vio_trail,
            vehicle_pose=res.vehicle_pose,
            is_lane_retained=res.is_lane_retained,
            planned_trajectory=res.planned_trajectory,
            behavior_context=res.behavior_context,
            rss_result=res.rss_result,
            fleet_status=res.fleet_status,
        )

        frame_lat_ms = (time.perf_counter() - t_frame_start) * 1000.0
        latencies.append(frame_lat_ms)
        frame_idx += 1

        if not headless:
            cv2.imshow("APEX-LKA Scene Perception HUD", hud_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap is not None:
        cap.release()
    if not headless:
        cv2.destroyAllWindows()

    total_time = time.perf_counter() - t_start
    avg_fps = frame_idx / max(total_time, 1e-4)
    avg_lat = float(np.mean(latencies)) if latencies else 0.0

    print(f"\n[APEX-LKA BENCHMARK COMPLETE]")
    print(f"  Frames processed: {frame_idx}")
    print(f"  Total time:       {total_time:.2f} s")
    print(f"  Average FPS:      {avg_fps:.1f} FPS (Target: >= 30 FPS)")
    print(f"  Average Latency:  {avg_lat:.2f} ms")
    if avg_fps >= 30.0:
        print("  Status:           PASSED [PERFORMANCE TARGET ACHIEVED]")
    else:
        print("  Status:           SUB-30 FPS (Check CPU load)")


def main():
    parser = argparse.ArgumentParser(description="Live Perception Runner")
    parser.add_argument("--source", type=str, default="video", help="Source type ('webcam', 'video', 'image') or direct video file path")
    parser.add_argument("--file", type=str, default=None)
    parser.add_argument("--backend", type=str, default="classical_cv")
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--gui", action="store_true", help="Display GUI window")
    parser.add_argument("--enable-object-detection", action="store_true", help="Enable multi-object detection and threat risk assessment")
    parser.add_argument("--enable-depth", action="store_true", help="Enable monocular metric depth estimation")
    parser.add_argument("--enable-occupancy", action="store_true", help="Enable 3D top-down occupancy grid rasterization")
    parser.add_argument("--show-depth-heatmap", action="store_true", help="Overlay false-color depth heatmap on viewport")
    parser.add_argument("--enable-tracking", action="store_true", help="Enable ByteTrack multi-object tracking")
    parser.add_argument("--enable-trajectory", action="store_true", help="Enable trajectory prediction and TTI calculation")
    parser.add_argument("--enable-sensor-fusion", action="store_true", help="Enable 15-DOF ESKF, VIO, and Kinematic Lane Retention")
    parser.add_argument("--enable-planner", action="store_true", help="Enable hierarchical behavior FSM, local path planner, ACC, and RSS safety shield")
    parser.add_argument("--enable-shadow-mode", action="store_true", help="Enable silent candidate shadow model execution")
    parser.add_argument("--enable-data-harvesting", action="store_true", help="Enable edge anomaly closed-loop data harvesting")
    parser.add_argument("--enable-all", action="store_true", help="Enable all Phase 5 modules (detection, depth, occupancy, tracking, trajectory, fusion, planner, shadow, harvester)")
    parser.add_argument("--profile", type=str, default="baja_2027", help="Vehicle profile to apply (baja_2027, campus_shuttle, agv_warehouse, passenger_car)")
    args = parser.parse_args()

    run_live_inference(
        source=args.source,
        file_path=args.file,
        backend=args.backend,
        max_frames=args.max_frames,
        headless=not args.gui,
        enable_object_detection=args.enable_object_detection,
        enable_depth=args.enable_depth,
        enable_occupancy=args.enable_occupancy,
        show_depth_heatmap=args.show_depth_heatmap,
        enable_tracking=args.enable_tracking,
        enable_trajectory=args.enable_trajectory,
        enable_sensor_fusion=args.enable_sensor_fusion,
        enable_planner=args.enable_planner,
        enable_shadow_mode=args.enable_shadow_mode,
        enable_data_harvesting=args.enable_data_harvesting,
        enable_all=args.enable_all,
        profile=args.profile,
    )



if __name__ == "__main__":
    main()
