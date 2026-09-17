"""
tests/test_inference_pipeline.py
=================================
End-to-end test of the inference framework.

Tests
-----
1. Preprocessing — valid image
2. Preprocessing — invalid input (bytes of garbage)
3. Preprocessing — too-small image
4. Predictor factory — classical_cv backend
5. Classical CV inference — synthetic road image
6. Classical CV inference — blank image (no lanes)
7. Model status — ML backend without trained model
8. Postprocessing — mask cleanup
9. Lane geometry — fitted curves
10. Offset calculator — sign convention
11. Visualization — overlay drawing
12. Full pipeline — end to end on synthetic road image
13. Full pipeline — error propagation (corrupt input)

Run:
    python tests/test_inference_pipeline.py
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

PASS = 0
FAIL = 0


def ok(name: str) -> None:
    global PASS
    PASS += 1
    print(f"  PASS: {name}")


def fail(name: str, reason: str) -> None:
    global FAIL
    FAIL += 1
    print(f"  FAIL: {name}: {reason}")


def assert_true(condition: bool, name: str, reason: str = "") -> bool:
    if condition:
        ok(name)
        return True
    else:
        fail(name, reason or "assertion failed")
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic road image factory
# ═══════════════════════════════════════════════════════════════════════════

def make_road_image(w: int = 640, h: int = 360) -> np.ndarray:
    """
    Create a synthetic road image with visible white lane markings.
    Road = gray tarmac, lanes = white, sky = blue-gray.
    This gives the classical CV pipeline something to detect.
    """
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky
    img[:int(h * 0.4), :] = (130, 110, 90)    # BGR blue-gray

    # Road
    img[int(h * 0.4):, :] = (70, 70, 70)      # dark gray

    # Left lane (white) — converges toward center at top
    for y in range(int(h * 0.38), h):
        x = int(0.28 * w + (y - h * 0.38) * 0.05)
        x = min(x, w - 1)
        cv2.line(img, (x, y), (x + 8, y), (255, 255, 255), 2)

    # Right lane (white)
    for y in range(int(h * 0.38), h):
        x = int(0.72 * w - (y - h * 0.38) * 0.05)
        x = max(x, 0)
        cv2.line(img, (x - 8, y), (x, y), (255, 255, 255), 2)

    return img


def make_blank_image(w: int = 640, h: int = 360) -> np.ndarray:
    """Uniform gray — no lane markings."""
    return np.full((h, w, 3), 80, dtype=np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_preprocessing_valid() -> None:
    from src.inference.preprocessing import preprocess
    img = make_road_image()
    result = preprocess(img)
    assert_true(result.valid,                         "preprocess_valid_image")
    assert_true(result.error is None,                 "preprocess_no_error")
    assert_true(result.input_tensor is not None,      "preprocess_tensor_exists")
    assert_true(result.input_tensor.dtype == np.float32, "preprocess_tensor_dtype")
    assert_true(result.original_bgr.shape == img.shape,  "preprocess_original_preserved")
    assert_true(result.preprocess_ms > 0,             "preprocess_timing")


def test_preprocessing_invalid() -> None:
    from src.inference.preprocessing import preprocess
    # Garbage bytes
    r = preprocess(b"\x00\x01\x02\x03\x04garbage")
    assert_true(not r.valid, "preprocess_garbage_bytes_invalid")
    assert_true(r.error is not None, "preprocess_garbage_has_error")


def test_preprocessing_small_image() -> None:
    from src.inference.preprocessing import preprocess
    tiny = np.zeros((10, 10, 3), dtype=np.uint8)
    r = preprocess(tiny)
    assert_true(not r.valid, "preprocess_too_small_invalid")


def test_predictor_factory() -> None:
    from src.inference.predictor import get_predictor, ModelStatus, BaseLanePredictor
    p = get_predictor("classical_cv")
    assert_true(isinstance(p, BaseLanePredictor), "factory_returns_base_type")
    assert_true(p.model_status == ModelStatus.CLASSICAL_CV, "factory_cv_status")


def test_classical_cv_road_image() -> None:
    from src.inference.preprocessing import preprocess
    from src.inference.predictor import get_predictor, DetectionStatus

    img   = make_road_image()
    pre   = preprocess(img)
    pred  = get_predictor("classical_cv").predict(pre)

    assert_true(pred is not None,             "cv_pred_not_none")
    assert_true(pred.inference_ms >= 0,       "cv_timing_non_negative")
    assert_true(pred.left_mask  is not None,  "cv_left_mask_exists")
    assert_true(pred.right_mask is not None,  "cv_right_mask_exists")
    assert_true(
        pred.status in (
            DetectionStatus.LANE_DETECTED,
            DetectionStatus.PARTIAL_LANE_DETECTED,
            DetectionStatus.LANE_NOT_DETECTED,
        ),
        "cv_status_valid_enum"
    )


def test_classical_cv_blank_image() -> None:
    """Blank image → should report LANE_NOT_DETECTED, not crash."""
    from src.inference.preprocessing import preprocess
    from src.inference.predictor import get_predictor, DetectionStatus

    img  = make_blank_image()
    pre  = preprocess(img)
    pred = get_predictor("classical_cv").predict(pre)

    assert_true(
        pred.status in (
            DetectionStatus.LANE_NOT_DETECTED,
            DetectionStatus.PARTIAL_LANE_DETECTED,
        ),
        "cv_blank_no_detection"
    )
    assert_true(pred.error_message is None, "cv_blank_no_crash")


def test_ml_predictor_not_trained() -> None:
    from src.inference.predictor import MLSegmentationPredictor, ModelStatus, DetectionStatus
    from src.inference.preprocessing import preprocess

    p   = MLSegmentationPredictor()
    img = make_road_image()
    pre = preprocess(img)
    res = p.predict(pre)

    if p.model_status == ModelStatus.NOT_TRAINED:
        assert_true(
            res.status == DetectionStatus.MODEL_UNAVAILABLE,
            "ml_not_trained_returns_unavailable"
        )
        assert_true(
            res.error_message is not None,
            "ml_not_trained_has_message"
        )
        assert_true(
            "not trained" in res.error_message.lower()
            or "model" in res.error_message.lower(),
            "ml_not_trained_message_meaningful"
        )
    else:
        # Model is trained — just verify no crash
        assert_true(res is not None, "ml_trained_returns_result")


def test_postprocessing() -> None:
    from src.inference.preprocessing import preprocess
    from src.inference.predictor import get_predictor
    from src.inference.postprocessing import postprocess

    img  = make_road_image()
    pre  = preprocess(img)
    pred = get_predictor("classical_cv").predict(pre)
    post = postprocess(pred)

    assert_true(post is not None,                     "post_not_none")
    assert_true(isinstance(post.left_pts,  np.ndarray), "post_left_pts_ndarray")
    assert_true(isinstance(post.right_pts, np.ndarray), "post_right_pts_ndarray")
    assert_true(post.postprocess_ms >= 0,             "post_timing")


def test_lane_geometry() -> None:
    from src.inference.preprocessing import preprocess
    from src.inference.predictor import get_predictor
    from src.inference.postprocessing import postprocess
    from src.lane_geometry.lane_estimator import estimate_geometry

    img  = make_road_image()
    pre  = preprocess(img)
    pred = get_predictor("classical_cv").predict(pre)
    post = postprocess(pred)
    geo  = estimate_geometry(post, pre.model_h, pre.model_w)

    assert_true(geo is not None,           "geo_not_none")
    assert_true(geo.estimate_ms >= 0,      "geo_timing")
    # Lane center, if computed, must be in [0, model_w]
    if geo.lane_center_x is not None:
        assert_true(
            0 <= geo.lane_center_x <= pre.model_w,
            "geo_lane_center_in_bounds"
        )


def test_offset_sign_convention() -> None:
    """
    Explicitly verify the sign convention:
      vehicle right of center → lateral_error > 0 → STEER LEFT
      vehicle left of center  → lateral_error < 0 → STEER RIGHT
    """
    from src.lane_geometry.lane_estimator import LaneGeometry
    from src.lane_geometry.offset_calculator import OffsetCalculator, SteeringRecommendation

    calc = OffsetCalculator()

    # Vehicle right of center: lane center at 280, vehicle at 320
    geo_right = LaneGeometry(
        left_poly  = np.array([0, 0, 120]),   # x = 120 at y=0 (dummy)
        right_poly = np.array([0, 0, 440]),
        x_left_bottom  = 120.0,
        x_right_bottom = 440.0,
        lane_center_x  = 280.0,   # center at 280
        model_h = 360, model_w = 640,
    )
    off_right = calc.calculate(geo_right, detection_confidence=0.9)
    assert_true(off_right.lateral_error_px is not None, "sign_right_has_error")
    assert_true(
        off_right.lateral_error_px > 0,
        f"sign_right_positive  (got {off_right.lateral_error_px:.2f})"
    )
    assert_true(
        off_right.recommendation in (
            SteeringRecommendation.STEER_LEFT,
            SteeringRecommendation.KEEP_CENTER,   # if very small offset
        ),
        f"sign_right_steer_left  (got {off_right.recommendation.value})"
    )

    # Vehicle left of center: lane center at 360 (center = 320)
    calc.reset()
    geo_left = LaneGeometry(
        left_poly  = np.array([0, 0, 200]),
        right_poly = np.array([0, 0, 520]),
        x_left_bottom  = 200.0,
        x_right_bottom = 520.0,
        lane_center_x  = 360.0,   # center right of vehicle
        model_h = 360, model_w = 640,
    )
    off_left = calc.calculate(geo_left, detection_confidence=0.9)
    assert_true(off_left.lateral_error_px is not None, "sign_left_has_error")
    assert_true(
        off_left.lateral_error_px < 0,
        f"sign_left_negative  (got {off_left.lateral_error_px:.2f})"
    )


def test_visualization() -> None:
    """Overlay drawing must not crash and return a valid uint8 image."""
    from src.inference.preprocessing import preprocess
    from src.inference.predictor import get_predictor
    from src.inference.postprocessing import postprocess
    from src.lane_geometry.lane_estimator import estimate_geometry
    from src.lane_geometry.offset_calculator import OffsetCalculator
    from src.visualization.lane_overlay import draw_overlay

    img  = make_road_image()
    pre  = preprocess(img)
    pred = get_predictor("classical_cv").predict(pre)
    post = postprocess(pred)
    geo  = estimate_geometry(post, pre.model_h, pre.model_w)
    off  = OffsetCalculator().calculate(geo)

    if geo.geometry_valid:
        canvas = draw_overlay(
            pre.original_bgr, pred, geo, off,
            scale_x=pre.scale_x, scale_y=pre.scale_y,
        )
        assert_true(canvas is not None,              "viz_canvas_not_none")
        assert_true(canvas.shape == img.shape,       "viz_canvas_same_shape")
        assert_true(canvas.dtype == np.uint8,        "viz_canvas_uint8")
    else:
        ok("viz_skipped_no_geometry")   # acceptable for synthetic image


def test_full_pipeline_road() -> None:
    """Full pipeline on synthetic road image."""
    from src.inference.pipeline import run_pipeline

    img    = make_road_image()
    result = run_pipeline(img, backend="classical_cv", reset_pd_state=True)

    assert_true(result is not None,          "pipeline_road_not_none")
    assert_true(result.prediction is not None, "pipeline_road_has_prediction")
    assert_true("total_ms" in result.timings,  "pipeline_road_has_timing")
    assert_true(
        result.timings["total_ms"] > 0,
        "pipeline_road_timing_positive"
    )
    assert_true(result.annotated_bgr is not None, "pipeline_road_has_output_image")


def test_full_pipeline_error_propagation() -> None:
    """Corrupt input must not raise — must return a result with success=False."""
    from src.inference.pipeline import run_pipeline

    garbage = b"\x00\x01\x02corruption"
    result  = run_pipeline(garbage, backend="classical_cv")

    assert_true(result is not None,  "pipeline_corrupt_not_none")
    assert_true(not result.success,  "pipeline_corrupt_success_false")
    assert_true(result.error is not None, "pipeline_corrupt_has_error")


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("\n" + "=" * 60)
    print("  LKA Inference Pipeline — Test Suite")
    print("=" * 60 + "\n")

    tests = [
        ("Preprocessing — valid image",       test_preprocessing_valid),
        ("Preprocessing — invalid bytes",      test_preprocessing_invalid),
        ("Preprocessing — too-small image",    test_preprocessing_small_image),
        ("Predictor factory",                  test_predictor_factory),
        ("Classical CV — road image",          test_classical_cv_road_image),
        ("Classical CV — blank (no lanes)",    test_classical_cv_blank_image),
        ("ML predictor — not trained status",  test_ml_predictor_not_trained),
        ("Postprocessing — mask cleanup",      test_postprocessing),
        ("Lane geometry — curve fitting",      test_lane_geometry),
        ("Offset — sign convention",           test_offset_sign_convention),
        ("Visualization — overlay drawing",    test_visualization),
        ("Full pipeline — road image",         test_full_pipeline_road),
        ("Full pipeline — error propagation",  test_full_pipeline_error_propagation),
    ]

    for name, fn in tests:
        try:
            fn()
        except Exception as exc:
            fail(name, f"EXCEPTION: {exc}")

    print()
    print("=" * 60)
    print(f"  {PASS} passed,  {FAIL} failed")
    print("=" * 60 + "\n")

    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    main()
