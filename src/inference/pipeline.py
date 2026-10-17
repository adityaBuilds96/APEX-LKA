"""
src/inference/pipeline.py
=========================
Single entry-point for the complete lane detection & multi-task scene perception pipeline.

Call run_pipeline(image_source, backend) and get back InferenceResult.
All other modules are internal to this function — the Streamlit app and
the Jetson client need only import this one function.

Future Jetson compatibility
---------------------------
On a Jetson Orin Nano, replace the image_source argument with a frame
from cv2.VideoCapture(0) and call run_pipeline() in a loop.
The backend will switch to "ml_segmentation" once the model is trained.
No other code changes are needed.

Usage
-----
    from src.inference.pipeline import run_pipeline, InferenceResult

    result = run_pipeline("path/to/image.jpg", backend="classical_cv")
    result = run_pipeline(numpy_array,          backend="ml_segmentation")

    if result.success:
        # result.annotated_bgr   — image with overlays
        # result.offset          — OffsetResult
        # result.geometry        — LaneGeometry
        # result.prediction      — LanePrediction
        # result.scene_perception— ScenePerceptionResult (Phase 4 Multi-Task)
        # result.timings         — dict of ms values
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.preprocessing    import preprocess, PreprocessResult
from src.inference.predictor        import get_predictor, LanePrediction, DetectionStatus
from src.inference.postprocessing   import postprocess, PostprocessedLanes
from src.lane_geometry.lane_estimator  import estimate_geometry, LaneGeometry
from src.lane_geometry.offset_calculator import OffsetCalculator, OffsetResult
from src.visualization.lane_overlay    import draw_overlay, draw_no_detection

from src.inference.environmental_adaptor import (
    EnvironmentalAutoAdaptor,
    EnvironmentalCondition,
    EnvironmentalState,
)
from src.inference.free_space import (
    RoadMaterial,
    FreeSpaceResult,
    extract_drivable_corridor,
)
from src.inference.distance_estimator import (
    DistanceProfile,
    MonocularDistanceEstimator,
)
from src.inference.temporal_smoother import (
    PathTrackingState,
    SmoothedPathResult,
    TemporalPathSmoother,
)


# ═══════════════════════════════════════════════════════════════════════════
# Result containers
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ScenePerceptionResult:
    """Multi-task autonomous scene perception telemetry."""
    environmental: EnvironmentalState = field(default_factory=EnvironmentalState)
    free_space: Optional[FreeSpaceResult] = None
    distance_profile: Optional[DistanceProfile] = None
    smoothed_paths: Optional[SmoothedPathResult] = None
    detection_result: Optional[Any] = None
    risk_report: Optional[Any] = None
    depth_result: Optional[Any] = None
    objects_3d: Optional[List[Any]] = None
    occupancy_grid: Optional[Any] = None
    tracked_objects: Optional[List[Any]] = None
    trajectory_report: Optional[Any] = None
    total_perception_ms: float = 0.0


@dataclass
class InferenceResult:
    """
    Complete output from run_pipeline().

    success : bool
        True if at least partial detection occurred.
    annotated_bgr : np.ndarray | None
        Original image with all overlays drawn (BGR uint8).
        Always returned — even on failure it shows an error overlay.
    preprocessed : PreprocessResult
    prediction : LanePrediction
    postprocessed : PostprocessedLanes | None
    geometry : LaneGeometry | None
    offset : OffsetResult | None
    scene_perception : ScenePerceptionResult | None
    detection_result : Any | None
    risk_report : Any | None
    depth_result : Any | None
    objects_3d : List[Any] | None
    occupancy_grid : Any | None
    tracked_objects : List[Any] | None
    trajectory_report : Any | None
    timings : dict
        Keys: preprocess_ms, env_adapt_ms, free_space_ms, inference_ms,
              postprocess_ms, geometry_ms, offset_ms, temporal_ms,
              distance_ms, scene_perception_ms, object_det_ms, risk_assess_ms,
              depth_ms, projection_3d_ms, occupancy_ms, tracking_ms, trajectory_ms,
              viz_ms, total_ms, fps_equiv
    error : str | None
    """
    success:          bool                         = False
    annotated_bgr:    Optional[np.ndarray]         = None
    preprocessed:     Optional[PreprocessResult]   = None
    prediction:       Optional[LanePrediction]     = None
    postprocessed:    Optional[PostprocessedLanes] = None
    geometry:         Optional[LaneGeometry]       = None
    offset:           Optional[OffsetResult]       = None
    scene_perception: Optional[ScenePerceptionResult] = None
    detection_result: Optional[Any]                = None
    risk_report:      Optional[Any]                = None
    depth_result:     Optional[Any]                = None
    objects_3d:       Optional[List[Any]]          = None
    occupancy_grid:   Optional[Any]                = None
    tracked_objects:  Optional[List[Any]]          = None
    trajectory_report: Optional[Any]               = None
    timings:          dict                         = field(default_factory=dict)
    error:            Optional[str]                = None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline Engine State Singletons (Stateful across video streams)
# ═══════════════════════════════════════════════════════════════════════════

_calculator = OffsetCalculator()
_adaptor = EnvironmentalAutoAdaptor()
_distance_estimator = MonocularDistanceEstimator()
_path_smoother = TemporalPathSmoother()
_object_detector = None
_risk_assessor = None
_depth_engine = None
_projector_3d = None
_occupancy_rasterizer = None
_tracker = None
_trajectory_predictor = None


def get_object_detector():
    global _object_detector
    if _object_detector is None:
        from src.perception.object_detector import ObjectDetector
        _object_detector = ObjectDetector()
    return _object_detector


def get_risk_assessor():
    global _risk_assessor
    if _risk_assessor is None:
        from src.perception.risk_assessment import PriorityRiskAssessor
        _risk_assessor = PriorityRiskAssessor()
    return _risk_assessor


def get_depth_engine():
    global _depth_engine
    if _depth_engine is None:
        from src.perception.depth_engine import MonocularDepthEngine
        _depth_engine = MonocularDepthEngine()
    return _depth_engine


def get_projector_3d():
    global _projector_3d
    if _projector_3d is None:
        from src.perception.projection_3d import BoundingBox3DProjector
        _projector_3d = BoundingBox3DProjector()
    return _projector_3d


def get_occupancy_rasterizer():
    global _occupancy_rasterizer
    if _occupancy_rasterizer is None:
        from src.perception.occupancy_grid import OccupancyGridRasterizer
        _occupancy_rasterizer = OccupancyGridRasterizer()
    return _occupancy_rasterizer


def get_tracker():
    global _tracker
    if _tracker is None:
        from src.perception.tracker import APEXMultiObjectTracker
        _tracker = APEXMultiObjectTracker()
    return _tracker


def get_trajectory_predictor():
    global _trajectory_predictor
    if _trajectory_predictor is None:
        from src.perception.trajectory import TrajectoryPredictor
        _trajectory_predictor = TrajectoryPredictor()
    return _trajectory_predictor


def run_pipeline(
    image_source: Union[str, Path, np.ndarray, bytes],
    backend: str = "classical_cv",
    reset_pd_state: bool = False,
    enable_object_detection: bool = False,
    enable_depth: bool = False,
    enable_occupancy: bool = False,
    show_depth_heatmap: bool = False,
    enable_tracking: bool = False,
    enable_trajectory: bool = False,
) -> InferenceResult:
    """
    Run the full autonomous scene perception and lane detection pipeline on one frame.

    Parameters
    ----------
    image_source : path | np.ndarray | bytes
    backend : "classical_cv" | "ml_segmentation"
    reset_pd_state : bool
        Set True for the first frame of a new video stream to reset
        the PD derivative state and temporal anti-flicker filters.
    enable_object_detection : bool
        Run YOLO multi-object detection and dynamic risk assessment.
    enable_depth : bool
        Run monocular metric depth estimation and 3D bounding box projection.
    enable_occupancy : bool
        Rasterize 3D top-down occupancy grid.
    show_depth_heatmap : bool
        Overlay false-color depth heatmap onto viewport.
    enable_tracking : bool
        Assign persistent Track IDs using ByteTrack/OC-SORT and estimate 3D velocity.
    enable_trajectory : bool
        Forecast 1.0-3.0s future waypoints and compute Time-To-Impact (TTI).

    Returns
    -------
    InferenceResult — always populated, never raises.
    """
    t_total_start = time.perf_counter()
    timings: dict = {}

    if reset_pd_state:
        _calculator.reset()
        _path_smoother.reset()
        if _tracker is not None:
            _tracker.reset()

    # ── Step 1: Preprocessing ──────────────────────────────────────────────
    preprocessed = preprocess(image_source)
    timings["preprocess_ms"] = round(preprocessed.preprocess_ms, 2)

    if not preprocessed.valid:
        return InferenceResult(
            success       = False,
            preprocessed  = preprocessed,
            timings       = timings,
            error         = preprocessed.error,
        )

    t_perc_start = time.perf_counter()

    # ── Step 2: Environmental Auto-Adaptation ───────────────────────────────
    adapted_bgr, env_state = _adaptor.analyze_and_adapt(preprocessed.resized_bgr)
    if env_state.adaptation_applied:
        # Use adapted frame for downstream perception
        preprocessed.resized_bgr = adapted_bgr
    timings["env_adapt_ms"] = round(env_state.latency_ms, 2)

    # ── Step 3: Drivable Free-Space Corridor Segmentation ───────────────────
    free_space_res = extract_drivable_corridor(
        preprocessed.resized_bgr,
        horizon_y=_distance_estimator.horizon_y,
    )
    timings["free_space_ms"] = round(free_space_res.latency_ms, 2)

    # ── Step 4: Lane Predictor ─────────────────────────────────────────────
    predictor = get_predictor(backend)
    prediction = predictor.predict(preprocessed)
    timings["inference_ms"] = round(prediction.inference_ms, 2)

    # Model not ready → return with overlay message
    if prediction.status == DetectionStatus.MODEL_UNAVAILABLE:
        overlay = draw_no_detection(
            preprocessed.original_bgr,
            prediction.error_message or "Model unavailable.",
            prediction,
        )
        timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        return InferenceResult(
            success       = False,
            annotated_bgr = overlay,
            preprocessed  = preprocessed,
            prediction    = prediction,
            timings       = timings,
            error         = prediction.error_message,
        )

    if prediction.status == DetectionStatus.INFERENCE_ERROR:
        overlay = draw_no_detection(
            preprocessed.original_bgr,
            f"Inference error:\n{prediction.error_message}",
            prediction,
        )
        timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
        return InferenceResult(
            success       = False,
            annotated_bgr = overlay,
            preprocessed  = preprocessed,
            prediction    = prediction,
            timings       = timings,
            error         = prediction.error_message,
        )

    # ── Step 5: Postprocessing ─────────────────────────────────────────────
    postprocessed = postprocess(prediction)
    timings["postprocess_ms"] = round(postprocessed.postprocess_ms, 2)

    # ── Step 6: Lane Geometry & Lateral Offset ─────────────────────────────
    t_geo = time.perf_counter()
    geometry = estimate_geometry(
        postprocessed,
        preprocessed.model_h,
        preprocessed.model_w,
    )
    timings["geometry_ms"] = round((time.perf_counter() - t_geo) * 1000, 2)

    t_off = time.perf_counter()
    combined_conf = (prediction.left_confidence + prediction.right_confidence) / 2.0
    offset = _calculator.calculate(geometry, detection_confidence=combined_conf)
    timings["offset_ms"] = round((time.perf_counter() - t_off) * 1000, 2)

    # ── Step 7: Temporal Anti-Flicker Path Smoothing ───────────────────────
    t_smooth = time.perf_counter()
    l_poly = geometry.left_poly if geometry.left_poly is not None else free_space_res.left_poly
    r_poly = geometry.right_poly if geometry.right_poly is not None else free_space_res.right_poly
    c_poly = free_space_res.center_poly
    if c_poly is None and l_poly is not None and r_poly is not None:
        c_poly = (l_poly + r_poly) / 2.0

    smoothed_paths = _path_smoother.smooth(
        left_poly=l_poly,
        right_poly=r_poly,
        center_poly=c_poly,
        img_h=preprocessed.model_h,
        img_w=preprocessed.model_w,
    )
    timings["temporal_ms"] = round((time.perf_counter() - t_smooth) * 1000, 2)

    # Update geometry with anti-flicker stabilized polynomials if beneficial
    if smoothed_paths.left_poly is not None and geometry.left_poly is None:
        geometry.left_poly = smoothed_paths.left_poly
    if smoothed_paths.right_poly is not None and geometry.right_poly is None:
        geometry.right_poly = smoothed_paths.right_poly

    # ── Step 8: Monocular Real-World Distance Estimation ───────────────────
    t_dist = time.perf_counter()
    b_pts = (
        free_space_res.right_boundary
        if len(free_space_res.right_boundary) > 0
        else free_space_res.left_boundary
    )
    dist_profile = _distance_estimator.compute_distance_profile(
        left_poly=smoothed_paths.left_poly,
        right_poly=smoothed_paths.right_poly,
        boundary_pts=b_pts,
    )
    timings["distance_ms"] = round((time.perf_counter() - t_dist) * 1000, 2)

    # ── Step 9: Multi-Object Detection, 3D Spatial Depth & Occupancy Grid ─
    detection_res = None
    risk_rep = None
    depth_res = None
    objects_3d = None
    occ_grid = None
    tracked_objects = None
    trajectory_report = None

    if enable_object_detection or enable_depth or enable_occupancy or enable_tracking or enable_trajectory:
        if enable_object_detection or enable_tracking or enable_trajectory:
            det_engine = get_object_detector()
            risk_engine = get_risk_assessor()
            detection_res = det_engine.detect(preprocessed.resized_bgr)
            risk_rep = risk_engine.assess(
                detections=detection_res,
                drivable_corridor_poly=free_space_res.polygon,
                frame_shape=(preprocessed.model_h, preprocessed.model_w),
            )
            timings["object_det_ms"] = round(detection_res.inference_ms, 2)
            timings["risk_assess_ms"] = round(risk_rep.assessment_ms, 2)

        if enable_depth:
            t_depth = time.perf_counter()
            d_engine = get_depth_engine()
            depth_res = d_engine.estimate_depth(preprocessed.resized_bgr)
            timings["depth_ms"] = round((time.perf_counter() - t_depth) * 1000, 2)

            if detection_res is not None and not detection_res.is_empty:
                t_proj = time.perf_counter()
                proj_3d = get_projector_3d()
                proj_res = proj_3d.project_2d_to_3d(detection_res, depth_res.metric_depth_map)
                objects_3d = proj_res.objects_3d
                timings["projection_3d_ms"] = round((time.perf_counter() - t_proj) * 1000, 2)

        if enable_occupancy:
            t_occ = time.perf_counter()
            occ_engine = get_occupancy_rasterizer()
            occ_grid = occ_engine.rasterize(
                objects_3d=objects_3d,
                drivable_corridor_poly=free_space_res.polygon,
                frame_shape=(preprocessed.model_h, preprocessed.model_w),
            )
            timings["occupancy_ms"] = round((time.perf_counter() - t_occ) * 1000, 2)

        if enable_tracking or enable_trajectory:
            t_track = time.perf_counter()
            tracker = get_tracker()
            if detection_res is not None:
                tracked_objects = tracker.update(
                    detections=detection_res,
                    objects_3d=objects_3d,
                    timestamp_s=time.time(),
                )
            else:
                tracked_objects = []
            timings["tracking_ms"] = round((time.perf_counter() - t_track) * 1000, 2)

            if enable_trajectory and tracked_objects:
                t_traj = time.perf_counter()
                traj_engine = get_trajectory_predictor()
                trajectory_report = traj_engine.predict(
                    tracked_objects=tracked_objects,
                    ego_speed_mps=2.5,
                    timestamp_s=time.time(),
                )
                timings["trajectory_ms"] = round((time.perf_counter() - t_traj) * 1000, 2)

    # ── Step 10: Assemble Multi-Task Scene Perception Telemetry ────────────
    total_perc_ms = (time.perf_counter() - t_perc_start) * 1000.0
    scene_perception = ScenePerceptionResult(
        environmental=env_state,
        free_space=free_space_res,
        distance_profile=dist_profile,
        smoothed_paths=smoothed_paths,
        detection_result=detection_res,
        risk_report=risk_rep,
        depth_result=depth_res,
        objects_3d=objects_3d,
        occupancy_grid=occ_grid,
        tracked_objects=tracked_objects,
        trajectory_report=trajectory_report,
        total_perception_ms=round(total_perc_ms, 2),
    )
    timings["scene_perception_ms"] = round(total_perc_ms, 2)

    # ── Step 11: Base Overlay Visualization ────────────────────────────────
    t_viz = time.perf_counter()
    if geometry.geometry_valid:
        annotated = draw_overlay(
            preprocessed.original_bgr,
            prediction,
            geometry,
            offset,
            scale_x = preprocessed.scale_x,
            scale_y = preprocessed.scale_y,
        )
    else:
        annotated = draw_no_detection(
            preprocessed.original_bgr,
            "LANE NOT DETECTED\nNo lane markings found in this image.",
            prediction,
        )

    # 3D Depth, Object Detection, Tracking & Occupancy HUD overlays
    if (enable_object_detection or enable_depth or enable_occupancy or enable_tracking or enable_trajectory) and annotated is not None:
        try:
            from dashboard.components.lka_hud import (
                _draw_object_detections,
                _draw_occupancy_grid_widget,
                _draw_obstacle_brake_banner,
                _draw_trajectory_alert_banner,
            )
            if show_depth_heatmap and depth_res is not None:
                d_norm = np.clip(depth_res.depth_map * 255.0, 0, 255).astype(np.uint8)
                heatmap = cv2.applyColorMap(d_norm, cv2.COLORMAP_MAGMA)
                if heatmap.shape[:2] != annotated.shape[:2]:
                    heatmap = cv2.resize(heatmap, (annotated.shape[1], annotated.shape[0]), interpolation=cv2.INTER_LINEAR)
                cv2.addWeighted(heatmap, 0.35, annotated, 0.65, 0, annotated)

            if detection_res is not None or risk_rep is not None or objects_3d is not None or tracked_objects is not None:
                _draw_object_detections(
                    annotated,
                    det_res=detection_res,
                    risk_report=risk_rep,
                    objects_3d=objects_3d,
                    tracked_objects=tracked_objects,
                    trajectory_report=trajectory_report,
                )

            if occ_grid is not None:
                _draw_occupancy_grid_widget(annotated, occ_grid, annotated.shape[1], annotated.shape[0])

            if trajectory_report is not None:
                _draw_trajectory_alert_banner(annotated, trajectory_report, annotated.shape[1], annotated.shape[0])
            elif risk_rep and getattr(risk_rep, "emergency_brake_flag", False):
                _draw_obstacle_brake_banner(annotated, annotated.shape[1], annotated.shape[0])
        except Exception:
            pass

    timings["viz_ms"] = round((time.perf_counter() - t_viz) * 1000, 2)

    # ── Totals ─────────────────────────────────────────────────────────────
    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
    timings["fps_equiv"] = round(1000.0 / max(timings["total_ms"], 0.1), 1)

    success = prediction.status in (
        DetectionStatus.LANE_DETECTED,
        DetectionStatus.PARTIAL_LANE_DETECTED,
    )

    return InferenceResult(
        success          = success,
        annotated_bgr    = annotated,
        preprocessed     = preprocessed,
        prediction       = prediction,
        postprocessed    = postprocessed,
        geometry         = geometry,
        offset           = offset,
        scene_perception = scene_perception,
        detection_result = detection_res,
        risk_report      = risk_rep,
        depth_result     = depth_res,
        objects_3d       = objects_3d,
        occupancy_grid   = occ_grid,
        tracked_objects  = tracked_objects,
        trajectory_report= trajectory_report,
        timings          = timings,
    )

