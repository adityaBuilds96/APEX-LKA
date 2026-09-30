"""
tests/test_universal_detection.py
==================================
Comprehensive test suite for Universal Lane & Path Detection and Temporal Smoothing.

Tests:
1. Road Surface Segmentation (asphalt/concrete extraction, off-road exclusion).
2. Guardrail & Barrier Suppression (vertical lines, metallic reflections rejected).
3. MODE B: Road Edge Detection (unpainted road with grass boundaries).
4. MODE C: Drivable Region Envelope (dirt trail/unmarked corridor).
5. Temporal Smoother: EMA stability on jittery input.
6. Temporal Smoother: Outlier rejection on sudden jumps (> 30%).
7. Temporal Smoother: Short gap interpolation & gradual confidence decay.
8. Universal Configuration loading.
"""

from pathlib import Path
import sys
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import cfg
from src.inference.classical_cv import (
    ClassicalCVPredictor,
    detect_horizon,
    segment_road_surface,
    compute_lane_accuracy,
)
from src.inference.predictor import DetectionStatus, LanePrediction
from src.inference.preprocessing import preprocess
from src.inference.temporal_smoother import TemporalLaneSmoother, rasterize_poly_mask


def make_unpainted_road(w: int = 640, h: int = 360) -> np.ndarray:
    """Create synthetic unpainted road flanked by grass/dirt borders."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky
    img[:int(0.38 * h), :] = (150, 130, 110)
    # Grass on left and right
    img[int(0.38 * h):, :] = (35, 115, 35)

    # Road surface (trapezoid of dark gray asphalt)
    pts = np.array([
        [int(0.40 * w), int(0.40 * h)],
        [int(0.60 * w), int(0.40 * h)],
        [int(0.88 * w), h],
        [int(0.12 * w), h],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (75, 75, 75))
    return img


def make_highway_with_guardrail(w: int = 640, h: int = 360) -> np.ndarray:
    """Create highway scene with white/yellow lanes and bright metal guardrails."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Sky
    img[:int(0.38 * h), :] = (160, 140, 120)
    # Road
    img[int(0.38 * h):, :] = (65, 65, 65)

    # Valid left lane (white)
    cv2.line(img, (int(0.42 * w), int(0.40 * h)), (int(0.15 * w), h), (255, 255, 255), 6)
    # Valid right lane (yellow)
    cv2.line(img, (int(0.58 * w), int(0.40 * h)), (int(0.85 * w), h), (0, 215, 255), 6)

    # Metal guardrail on outer left (vertical posts + horizontal railing on grass)
    cv2.line(img, (15, int(0.20 * h)), (15, h), (210, 210, 210), 5)
    cv2.line(img, (30, int(0.20 * h)), (30, h), (210, 210, 210), 5)
    cv2.line(img, (5, int(0.60 * h)), (45, int(0.60 * h)), (220, 220, 220), 4)

    # Metal guardrail on outer right
    cv2.line(img, (w - 15, int(0.20 * h)), (w - 15, h), (210, 210, 210), 5)
    cv2.line(img, (w - 30, int(0.20 * h)), (w - 30, h), (210, 210, 210), 5)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_road_surface_segmentation():
    """Verify that road surface segmentation isolates pavement and excludes off-road terrain."""
    img = make_unpainted_road()
    h, w = img.shape[:2]
    horizon_y = detect_horizon(img)
    drivable = segment_road_surface(img, horizon_y)

    assert drivable is not None
    assert drivable.shape == (h, w)
    assert drivable.dtype == np.uint8

    # Road center at bottom should be drivable
    assert drivable[h - 15, w // 2] == 255

    # Sky must be 0
    assert np.all(drivable[:horizon_y, :] == 0)

    # Far corners (grass) must be 0
    assert drivable[h - 10, 10] == 0
    assert drivable[h - 10, w - 10] == 0


def test_guardrail_suppression_robustness():
    """Verify that vertical and off-road guardrails are suppressed without corrupting lane detections."""
    img = make_highway_with_guardrail()
    h, w = img.shape[:2]
    pre = preprocess(img)

    predictor = ClassicalCVPredictor(mode="painted", enable_smoothing=False)
    pred = predictor.predict(pre)

    assert pred.status in (DetectionStatus.LANE_DETECTED, DetectionStatus.PARTIAL_LANE_DETECTED)
    assert pred.left_mask is not None and pred.left_mask.any()

    # Left mask must NOT touch outer guardrail (x < 35)
    left_ys, left_xs = np.where(pred.left_mask > 0)
    assert left_xs.min() > 35, f"Left mask locked onto guardrail: min x={left_xs.min()}"

    # Right mask must NOT touch outer guardrail (x > w - 35)
    if pred.right_mask is not None and pred.right_mask.any():
        right_ys, right_xs = np.where(pred.right_mask > 0)
        assert right_xs.max() < (w - 35), f"Right mask locked onto guardrail: max x={right_xs.max()}"


def test_mode_b_unpainted_road_edges():
    """Verify MODE B extracts road edges when lane markings are completely absent."""
    img = make_unpainted_road()
    pre = preprocess(img)

    predictor = ClassicalCVPredictor(mode="edge", enable_smoothing=False)
    pred = predictor.predict(pre)

    assert pred.detection_mode == "edge"
    assert pred.status in (DetectionStatus.LANE_DETECTED, DetectionStatus.PARTIAL_LANE_DETECTED)
    assert pred.left_mask is not None and pred.left_mask.any()
    assert pred.right_mask is not None and pred.right_mask.any()
    assert pred.accuracy_score > 0.0


def test_mode_c_drivable_envelope():
    """Verify MODE C provides a drivable corridor envelope for unmarked paths."""
    img = make_unpainted_road()
    pre = preprocess(img)

    predictor = ClassicalCVPredictor(mode="drivable", enable_smoothing=False)
    pred = predictor.predict(pre)

    assert pred.detection_mode == "drivable"
    assert pred.status in (DetectionStatus.LANE_DETECTED, DetectionStatus.PARTIAL_LANE_DETECTED)
    assert pred.left_mask is not None and pred.left_mask.any()
    assert pred.right_mask is not None and pred.right_mask.any()


def test_temporal_smoother_ema():
    """Verify that EMA smoothing stabilizes jittery polynomial coefficients across frames."""
    smoother = TemporalLaneSmoother(buffer_size=5, ema_alpha=0.6)
    h, w = 360, 640

    # Base coefficients
    base_left = np.array([0.001, -1.2, 100.0])
    base_right = np.array([-0.001, 1.2, 540.0])

    # Frame 1
    p1 = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=rasterize_poly_mask(base_left, h, w, 140),
        right_mask=rasterize_poly_mask(base_right, h, w, 140),
        left_poly_coeffs=base_left,
        right_poly_coeffs=base_right,
        left_confidence=0.8,
        right_confidence=0.8,
        model_h=h,
        model_w=w,
    )
    res1 = smoother.smooth(p1)
    np.testing.assert_allclose(res1.left_poly_coeffs, base_left)

    # Frame 2 with jitter in intercept: +10px
    jittered_left = np.array([0.001, -1.2, 110.0])
    p2 = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=rasterize_poly_mask(jittered_left, h, w, 140),
        right_mask=rasterize_poly_mask(base_right, h, w, 140),
        left_poly_coeffs=jittered_left,
        right_poly_coeffs=base_right,
        left_confidence=0.8,
        right_confidence=0.8,
        model_h=h,
        model_w=w,
    )
    res2 = smoother.smooth(p2)
    # Expected EMA intercept: 0.6 * 110 + 0.4 * 100 = 66 + 40 = 106
    expected_c = 0.6 * 110.0 + 0.4 * 100.0
    assert abs(res2.left_poly_coeffs[2] - expected_c) < 1e-3


def test_temporal_smoother_outlier_rejection():
    """Verify that sudden massive coordinate jumps (> 30% width) are rejected as outliers."""
    smoother = TemporalLaneSmoother(buffer_size=5, outlier_threshold=0.30)
    h, w = 360, 640

    base_left = np.array([0.0, -0.68, 345.0])
    base_right = np.array([0.0, 0.68, 295.0])

    p1 = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=rasterize_poly_mask(base_left, h, w, 140),
        right_mask=rasterize_poly_mask(base_right, h, w, 140),
        left_poly_coeffs=base_left,
        right_poly_coeffs=base_right,
        left_confidence=0.8,
        right_confidence=0.8,
        model_h=h,
        model_w=w,
    )
    smoother.smooth(p1)

    # Outlier jump: lane shifts by 250px (> 38% of width)
    outlier_left = np.array([0.0, -0.68, 595.0])
    outlier_right = np.array([0.0, 0.68, 545.0])
    p_outlier = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=rasterize_poly_mask(outlier_left, h, w, 140),
        right_mask=rasterize_poly_mask(outlier_right, h, w, 140),
        left_poly_coeffs=outlier_left,
        right_poly_coeffs=outlier_right,
        left_confidence=0.8,
        right_confidence=0.8,
        model_h=h,
        model_w=w,
    )

    res = smoother.smooth(p_outlier)
    # Outlier should be skipped, maintaining previous smoothed coeffs
    assert abs(res.left_poly_coeffs[2] - base_left[2]) < 5.0


def test_temporal_smoother_gap_interpolation():
    """Verify that short occlusions (up to 3 frames) are interpolated with decaying confidence."""
    smoother = TemporalLaneSmoother(buffer_size=5, max_gap_frames=3, confidence_decay=0.85)
    h, w = 360, 640

    base_left = np.array([0.0, -0.68, 345.0])
    base_right = np.array([0.0, 0.68, 295.0])

    p_valid = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        left_mask=rasterize_poly_mask(base_left, h, w, 140),
        right_mask=rasterize_poly_mask(base_right, h, w, 140),
        left_poly_coeffs=base_left,
        right_poly_coeffs=base_right,
        left_confidence=0.9,
        right_confidence=0.9,
        model_h=h,
        model_w=w,
    )
    smoother.smooth(p_valid)

    # Frame 2: Gap 1 (missing detection)
    res_gap1 = smoother.smooth(LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED, model_h=h, model_w=w))
    assert res_gap1.status in (DetectionStatus.LANE_DETECTED, DetectionStatus.PARTIAL_LANE_DETECTED)
    assert res_gap1.left_mask.any()
    assert res_gap1.left_confidence < 0.9  # Decayed

    # Frame 3: Gap 2
    res_gap2 = smoother.smooth(LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED, model_h=h, model_w=w))
    assert res_gap2.left_confidence < res_gap1.left_confidence

    # Frame 4: Gap 3
    res_gap3 = smoother.smooth(LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED, model_h=h, model_w=w))
    assert res_gap3.left_confidence < res_gap2.left_confidence

    # Frame 5: Gap 4 (exceeds max_gap_frames=3) -> must drop to LANE_NOT_DETECTED
    res_gap4 = smoother.smooth(LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED, model_h=h, model_w=w))
    assert res_gap4.status == DetectionStatus.LANE_NOT_DETECTED



def test_universal_config_loading():
    """Verify lane_detection_universal section exists in project config."""
    assert "lane_detection_universal" in cfg
    u_cfg = cfg["lane_detection_universal"]
    assert u_cfg["mode"] in ("auto", "painted", "edge", "drivable")
    assert u_cfg["road_surface_detection"] is True
    assert u_cfg["temporal_smoothing"]["enabled"] is True
    assert u_cfg["temporal_smoothing"]["max_gap_frames"] == 3
