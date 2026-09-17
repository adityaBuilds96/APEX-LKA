"""
src/inference/pipeline.py
==========================
Single entry-point for the complete lane detection inference pipeline.

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
        # result.annotated_bgr  — image with overlays
        # result.offset         — OffsetResult
        # result.geometry       — LaneGeometry
        # result.prediction     — LanePrediction
        # result.timings        — dict of ms values
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

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


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════

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
    timings : dict
        Keys: preprocess_ms, inference_ms, postprocess_ms,
              geometry_ms, offset_ms, viz_ms, total_ms
    error : str | None
    """
    success:       bool                         = False
    annotated_bgr: Optional[np.ndarray]         = None
    preprocessed:  Optional[PreprocessResult]   = None
    prediction:    Optional[LanePrediction]     = None
    postprocessed: Optional[PostprocessedLanes] = None
    geometry:      Optional[LaneGeometry]       = None
    offset:        Optional[OffsetResult]       = None
    timings:       dict                         = field(default_factory=dict)
    error:         Optional[str]                = None


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════════════

# One shared OffsetCalculator per process (maintains state for video)
_calculator = OffsetCalculator()


def run_pipeline(
    image_source: Union[str, Path, np.ndarray, bytes],
    backend: str = "classical_cv",
    reset_pd_state: bool = False,
) -> InferenceResult:
    """
    Run the full lane detection pipeline on one image.

    Parameters
    ----------
    image_source : path | np.ndarray | bytes
    backend : "classical_cv" | "ml_segmentation"
    reset_pd_state : bool
        Set True for the first frame of a new video stream to reset
        the PD derivative state.

    Returns
    -------
    InferenceResult — always populated, never raises.
    """
    t_total_start = time.perf_counter()
    timings: dict = {}

    if reset_pd_state:
        _calculator.reset()

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

    # ── Step 2: Get predictor ──────────────────────────────────────────────
    predictor = get_predictor(backend)

    # ── Step 3: Inference ──────────────────────────────────────────────────
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

    # ── Step 4: Postprocessing ─────────────────────────────────────────────
    postprocessed = postprocess(prediction)
    timings["postprocess_ms"] = round(postprocessed.postprocess_ms, 2)

    # ── Step 5: Lane geometry ──────────────────────────────────────────────
    t_geo = time.perf_counter()
    geometry = estimate_geometry(
        postprocessed,
        preprocessed.model_h,
        preprocessed.model_w,
    )
    timings["geometry_ms"] = round((time.perf_counter() - t_geo) * 1000, 2)

    # ── Step 6: Lateral offset ─────────────────────────────────────────────
    t_off = time.perf_counter()
    combined_conf = (prediction.left_confidence + prediction.right_confidence) / 2.0
    offset = _calculator.calculate(geometry, detection_confidence=combined_conf)
    timings["offset_ms"] = round((time.perf_counter() - t_off) * 1000, 2)

    # ── Step 7: Visualization ──────────────────────────────────────────────
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
    timings["viz_ms"] = round((time.perf_counter() - t_viz) * 1000, 2)

    # ── Totals ─────────────────────────────────────────────────────────────
    timings["total_ms"] = round((time.perf_counter() - t_total_start) * 1000, 2)
    timings["fps_equiv"] = round(1000.0 / max(timings["total_ms"], 0.1), 1)

    success = prediction.status in (
        DetectionStatus.LANE_DETECTED,
        DetectionStatus.PARTIAL_LANE_DETECTED,
    )

    return InferenceResult(
        success       = success,
        annotated_bgr = annotated,
        preprocessed  = preprocessed,
        prediction    = prediction,
        postprocessed = postprocessed,
        geometry      = geometry,
        offset        = offset,
        timings       = timings,
    )
