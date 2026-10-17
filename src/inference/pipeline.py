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

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
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
    vehicle_pose: Optional[Any] = None
    vio_trail: Optional[List[Tuple[float, float]]] = None
    is_lane_retained: bool = False
    retained_lane_result: Optional[Any] = None
    planned_trajectory: Optional[Any] = None
    behavior_context: Optional[Any] = None
    acc_output: Optional[Any] = None
    rss_result: Optional[Any] = None
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
    vehicle_pose : Any | None
    vio_trail : List[Tuple[float, float]] | None
    is_lane_retained : bool
    retained_lane_result : Any | None
    timings : dict
        Keys: preprocess_ms, env_adapt_ms, free_space_ms, inference_ms,
              postprocess_ms, geometry_ms, offset_ms, temporal_ms,
              distance_ms, scene_perception_ms, object_det_ms, risk_assess_ms,
              depth_ms, projection_3d_ms, occupancy_ms, tracking_ms, trajectory_ms,
              sensor_fusion_ms, viz_ms, total_ms, fps_equiv
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
    vehicle_pose:     Optional[Any]                = None
    vio_trail:        Optional[List[Tuple[float, float]]] = None
    is_lane_retained: bool                         = False
    retained_lane_result: Optional[Any]            = None
    planned_trajectory: Optional[Any]              = None
    behavior_context: Optional[Any]                = None
    acc_output:       Optional[Any]                = None
    rss_result:       Optional[Any]                = None
    shadow_report:    Optional[Any]                = None
    harvest_event:    Optional[Any]                = None
    fleet_status:     Optional[Any]                = None
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


_eskf = None
_vio_engine = None
_lane_retainer = None


def get_eskf():
    global _eskf
    if _eskf is None:
        from src.fusion.eskf import ErrorStateKalmanFilter15DOF
        _eskf = ErrorStateKalmanFilter15DOF()
    return _eskf


def get_vio_engine():
    global _vio_engine
    if _vio_engine is None:
        from src.fusion.vio_engine import VisualInertialOdometryEngine
        _vio_engine = VisualInertialOdometryEngine()
    return _vio_engine


def get_lane_retainer():
    global _lane_retainer
    if _lane_retainer is None:
        from src.fusion.lane_retention import KinematicLaneRetainer
        _lane_retainer = KinematicLaneRetainer()
    return _lane_retainer


_behavior_fsm = None
_path_planner = None
_acc_controller = None
_rss_shield = None


def get_behavior_fsm():
    global _behavior_fsm
    if _behavior_fsm is None:
        from src.planning.behavior_fsm import BehaviorStateMachine
        _behavior_fsm = BehaviorStateMachine()
    return _behavior_fsm


def get_path_planner():
    global _path_planner
    if _path_planner is None:
        from src.planning.path_planner import LocalPathPlanner
        _path_planner = LocalPathPlanner()
    return _path_planner


def get_acc_controller():
    global _acc_controller
    if _acc_controller is None:
        from src.planning.acc_controller import AdaptiveCruiseController
        _acc_controller = AdaptiveCruiseController()
    return _acc_controller


def get_rss_shield():
    global _rss_shield
    if _rss_shield is None:
        from src.planning.rss_shield import MobileyeRSSShield
        _rss_shield = MobileyeRSSShield()
    return _rss_shield


_shadow_mode_engine = None
_data_harvester = None
_ota_manager = None


def get_shadow_mode_engine():
    global _shadow_mode_engine
    if _shadow_mode_engine is None:
        from src.fleet.shadow_mode import ShadowModeEngine
        from src.fleet.ota_updater import MockPredictor
        _shadow_mode_engine = ShadowModeEngine(shadow_predictor=MockPredictor(name="candidate_shadow", version="2.0.0"))
    return _shadow_mode_engine


def get_data_harvester():
    global _data_harvester
    if _data_harvester is None:
        from src.fleet.data_collector import ClosedLoopDataHarvester
        _data_harvester = ClosedLoopDataHarvester()
    return _data_harvester


def get_ota_manager():
    global _ota_manager
    if _ota_manager is None:
        from src.fleet.ota_updater import OTAModelManager
        _ota_manager = OTAModelManager()
    return _ota_manager


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
    enable_sensor_fusion: bool = False,
    enable_planner: bool = False,
    enable_shadow_mode: bool = False,
    enable_data_harvesting: bool = False,
    sim_mode: bool = False,
) -> InferenceResult:
    """
    Run the full autonomous scene perception and lane detection pipeline on one frame.
    Supports SIM_MODE virtual sensor routing.
    """
    t_total_start = time.perf_counter()
    timings: dict = {}

    # ── Simulation Mode Sensor Routing ─────────────────────────────────────
    if (sim_mode or os.environ.get("SIM_MODE", "").lower() in ("1", "true", "yes")) and image_source is None:
        try:
            from src.simulation.carla_bridge import CARLABridge
            bridge = CARLABridge(mock_mode=True, synchronous=True)
            if not bridge.is_connected():
                bridge.connect()
            image_source = bridge.get_camera_frame()
        except Exception:
            pass

    if reset_pd_state:
        _calculator.reset()
        _path_smoother.reset()
        if _tracker is not None:
            _tracker.reset()
        if _lane_retainer is not None:
            _lane_retainer.reset()
        if _vio_engine is not None:
            _vio_engine.reset()

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

    # ── Step 9.5: Sensor Fusion, VIO, State Estimation & Lane Retention ───
    vehicle_pose = None
    vio_trail = None
    is_lane_retained = False
    retained_lane_res = None

    if enable_sensor_fusion:
        t_fusion = time.perf_counter()
        vio = get_vio_engine()
        eskf = get_eskf()
        retainer = get_lane_retainer()

        ego_motion = vio.process_frame(preprocessed.resized_bgr)
        eskf.update_visual_odometry(ego_motion.dx_m, ego_motion.dy_m, ego_motion.dyaw_rad)
        vehicle_pose = eskf.get_vehicle_pose()
        vio_trail = vio.get_trajectory_trail()

        # Kinematic Lane Retention if visual perception is missing or low-confidence
        conf = offset.confidence if offset else 0.0
        if not geometry.geometry_valid or conf < retainer.confidence_threshold:
            retained_lane_res = retainer.update(
                current_prediction=prediction,
                current_geometry=geometry,
                current_pose=vehicle_pose,
            )
            if retained_lane_res.is_retained and retained_lane_res.geometry is not None:
                geometry = retained_lane_res.geometry
                if retained_lane_res.offset is not None:
                    offset = retained_lane_res.offset
                is_lane_retained = True
        else:
            retainer.update(
                current_prediction=prediction,
                current_geometry=geometry,
                current_pose=vehicle_pose,
            )

        timings["sensor_fusion_ms"] = round((time.perf_counter() - t_fusion) * 1000, 2)

    # ── Step 9.6: Hierarchical Motion Planning, ACC & RSS Shield ───────────
    planned_traj = None
    behavior_ctx = None
    acc_out = None
    rss_res = None

    if enable_planner:
        t_plan = time.perf_counter()
        rss = get_rss_shield()
        fsm = get_behavior_fsm()
        acc = get_acc_controller()
        planner = get_path_planner()

        ego_spd = float(getattr(vehicle_pose, "vx", 3.5)) if vehicle_pose is not None else 3.5

        # 1. RSS Safety Evaluation
        rss_res = rss.evaluate_objects(ego_speed_mps=ego_spd, tracked_objects=tracked_objects)

        # 2. Behavior State Machine
        lane_valid = bool((geometry is not None and getattr(geometry, "geometry_valid", False)) or is_lane_retained)
        behavior_ctx = fsm.update(
            ego_speed_mps=ego_spd,
            tracked_objects=tracked_objects,
            rss_override=rss_res.override_active,
            lane_detected=lane_valid,
            lane_retained=is_lane_retained,
            dt=0.033,
        )

        # 3. Adaptive Cruise Control
        lead_d = getattr(behavior_ctx, "lead_distance_m", None)
        lead_rel = getattr(behavior_ctx, "lead_relative_speed_mps", None)
        lead_spd = max(0.0, ego_spd + lead_rel) if lead_rel is not None else None
        curv = float(getattr(geometry, "curvature", 0.0)) if geometry else 0.0

        acc_out = acc.update(
            ego_speed_mps=ego_spd,
            v_set_mps=behavior_ctx.target_speed_mps,
            lead_distance_m=lead_d,
            lead_speed_mps=lead_spd,
            path_curvature=curv,
            dt=0.033,
        )

        # 4. TEB Metric Local Path Planner
        center_poly = None
        if geometry is not None:
            if hasattr(geometry, "center_poly") and geometry.center_poly is not None:
                center_poly = geometry.center_poly
            elif getattr(geometry, "left_poly", None) is not None and getattr(geometry, "right_poly", None) is not None:
                center_poly = (np.asarray(geometry.left_poly) + np.asarray(geometry.right_poly)) * 0.5
            elif getattr(geometry, "left_poly", None) is not None:
                center_poly = geometry.left_poly

        planned_traj = planner.plan(
            ego_speed_mps=ego_spd,
            target_speed_mps=acc_out.target_speed_mps,
            lane_center_poly=center_poly,
            tracked_objects=tracked_objects,
            occupancy_grid=occ_grid,
            behavior_ctx=behavior_ctx,
            dt=0.033,
        )

        timings["planning_ms"] = round((time.perf_counter() - t_plan) * 1000, 2)

    # ── Step 9.7: Fleet OTA, Silent Shadow-Mode & Closed-Loop Data Harvesting ───
    shadow_rep = None
    harvest_evt = None
    fleet_stat = None

    off_val = float(getattr(offset, "offset_m", 0.0) or getattr(offset, "lateral_error_m", 0.0) or 0.0) if offset else 0.0

    if enable_shadow_mode:
        t_shadow = time.perf_counter()
        shadow_eng = get_shadow_mode_engine()
        shadow_rep = shadow_eng.step(
            frame=preprocessed.original_bgr,
            primary_prediction=prediction,
            primary_offset_m=off_val,
            primary_confidence=combined_conf,
        )
        timings["shadow_mode_ms"] = round((time.perf_counter() - t_shadow) * 1000, 2)

    if enable_data_harvesting:
        t_harvest = time.perf_counter()
        harvester = get_data_harvester()
        can_dict = {
            "speed_mps": float(getattr(vehicle_pose, "vx", 3.5)) if vehicle_pose is not None else 3.5,
            "steer_angle_deg": float(getattr(offset, "steering_angle_deg", 0.0)) if offset else 0.0,
        }
        harvest_evt = harvester.observe(
            frame_bgr=preprocessed.original_bgr,
            confidence=combined_conf,
            offset_m=off_val,
            depth_map=depth_res.metric_depth_map if depth_res is not None else None,
            objects_3d=objects_3d,
            vehicle_pose=vehicle_pose,
            can_state=can_dict,
            rss_override_active=bool(rss_res.override_active) if rss_res is not None else False,
            driver_override_active=False,
            shadow_divergence_m=float(shadow_rep.offset_delta_m) if shadow_rep is not None else 0.0,
        )
        timings["data_harvest_ms"] = round((time.perf_counter() - t_harvest) * 1000, 2)

    if enable_shadow_mode or enable_data_harvesting:
        ota_mgr = get_ota_manager()
        h_queue = harvester.harvest_queue_count if (enable_data_harvesting and 'harvester' in locals()) else 0
        s_score = shadow_rep.discrepancy_score_u8 if shadow_rep is not None else 0
        fleet_stat = {
            "model_version_hash": ota_mgr.active_version_hash,
            "sync_state_str": ota_mgr.sync_state.name,
            "shadow_discrepancy": s_score,
            "harvest_queue_count": h_queue,
            "backend_str": ota_mgr.active_backend.name,
        }

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
        vehicle_pose=vehicle_pose,
        vio_trail=vio_trail,
        is_lane_retained=is_lane_retained,
        retained_lane_result=retained_lane_res,
        planned_trajectory=planned_traj,
        behavior_context=behavior_ctx,
        acc_output=acc_out,
        rss_result=rss_res,
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

    # 3D Depth, Object Detection, Tracking, Occupancy, Sensor Fusion, Planning & Fleet HUD overlays
    if (enable_object_detection or enable_depth or enable_occupancy or enable_tracking or enable_trajectory or enable_sensor_fusion or enable_planner or enable_shadow_mode or enable_data_harvesting) and annotated is not None:
        try:
            from dashboard.components.lka_hud import (
                _draw_object_detections,
                _draw_occupancy_grid_widget,
                _draw_obstacle_brake_banner,
                _draw_trajectory_alert_banner,
                _draw_vio_trajectory_trail,
                _draw_lane_retention_badge,
                _draw_kinematic_state_panel,
                _draw_planned_trajectory_spline,
                _draw_rss_safety_envelope,
                _draw_fsm_behavior_badge,
                _draw_fleet_ota_panel,
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

            if enable_sensor_fusion:
                if vio_trail:
                    _draw_vio_trajectory_trail(annotated, vio_trail, annotated.shape[1], annotated.shape[0])
                if is_lane_retained:
                    _draw_lane_retention_badge(annotated, annotated.shape[1], annotated.shape[0])
                if vehicle_pose is not None:
                    _draw_kinematic_state_panel(annotated, vehicle_pose, x=14, y=100)

            if enable_planner:
                if planned_traj is not None:
                    _draw_planned_trajectory_spline(annotated, planned_traj, annotated.shape[1], annotated.shape[0])
                if rss_res is not None:
                    _draw_rss_safety_envelope(annotated, rss_res, tracked_objects, annotated.shape[1], annotated.shape[0])
                if behavior_ctx is not None or rss_res is not None:
                    _draw_fsm_behavior_badge(annotated, behavior_ctx, rss_res, annotated.shape[1], annotated.shape[0])

            if fleet_stat is not None:
                _draw_fleet_ota_panel(annotated, fleet_stat, x=14, y=annotated.shape[0] - 85)
        except Exception:
            pass

    timings["viz_ms"] = round((time.perf_counter() - t_viz) * 1000, 2)

    # ── Totals ─────────────────────────────────────────────────────────────
    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
    timings["fps_equiv"] = round(1000.0 / max(timings["total_ms"], 0.1), 1)

    success = (
        prediction.status in (
            DetectionStatus.LANE_DETECTED,
            DetectionStatus.PARTIAL_LANE_DETECTED,
        )
        or is_lane_retained
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
        vehicle_pose     = vehicle_pose,
        vio_trail        = vio_trail,
        is_lane_retained = is_lane_retained,
        retained_lane_result = retained_lane_res,
        planned_trajectory = planned_traj,
        behavior_context = behavior_ctx,
        acc_output       = acc_out,
        rss_result       = rss_res,
        shadow_report    = shadow_rep,
        harvest_event    = harvest_evt,
        fleet_status     = fleet_stat,
        timings          = timings,
    )

