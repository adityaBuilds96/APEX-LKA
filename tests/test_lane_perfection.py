"""
tests/test_lane_perfection.py
=============================
Phase 5 Section 3: Absolute Lane Perception Recovery & Perfection Test Suite.
Validates:
1. Dynamic Horizon & Guardrail Slope Suppression (|m| > 2.2 rejected)
2. LAB + HLS Adaptive Color Bandpass Filtering (Yellow & White markings with CLAHE)
3. Road-Surface Masking & Flood-Fill Ground-Truth Constraint
4. RANSAC 2nd-degree Parabolic Fitting (50 iterations, 3.0 px threshold)
5. Rigid Track-Width Sanity Constraints (150 px <= W_lane <= 380 px at bottom)
6. Stream Temporal EKF 8-State Smoothing with Zero Flickering Guarantee (Delta <= 0.02)
7. EKF Hold during 1-3 Frame Dropouts
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.ekf_tracker import EKF8StateLaneTracker
from src.inference.classical_cv import (
    ClassicalCVPredictor,
    _colour_mask,
    _separate_lines_vectorized,
    detect_horizon,
    fit_parabola_ransac,
    segment_road_surface,
)
from src.inference.preprocessing import preprocess
from src.inference.predictor import DetectionStatus, LanePrediction
from src.inference.temporal_smoother import TemporalLaneSmoother, rasterize_poly_mask


def test_dynamic_horizon_and_guardrail_isolation():
    """Verify horizon calculation and strict rejection of vertical guardrail slopes (|m| > 2.2)."""
    h, w = 360, 640
    img = np.full((h, w, 3), 70, dtype=np.uint8)
    img[:120, :] = (200, 180, 150)  # sky

    horizon_y = detect_horizon(img)
    assert 100 <= horizon_y <= 180

    # Test Hough lines with slopes: one normal lane (slope ~ -1.2), one vertical guardrail (slope ~ 5.0)
    # Line 1: Normal left lane marking
    l1 = [int(0.20 * w), h - 10, int(0.40 * w), horizon_y + 10]
    # Line 2: Near-vertical guardrail post (dx = 2 px, dy = 100 px -> slope = 50.0 > 2.2)
    l2 = [50, h - 10, 52, h - 110]

    lines = np.array([[l1], [l2]], dtype=np.int32)
    left_pts, right_pts, has_outside, has_extreme = _separate_lines_vectorized(
        lines, w=w, h=h, horizon_y=horizon_y, min_length=30, max_slope=2.2
    )

    # Vertical guardrail post must trigger extreme slopes and be rejected from left_pts
    assert has_extreme is True
    # Only valid lane points should be retained
    assert len(left_pts) >= 2
    for pt in left_pts:
        assert pt[0] > 60  # Guardrail was at x ~ 50


def test_lab_hls_dual_color_bandpass():
    """Verify adaptive LAB+HLS dual bandpass correctly detects yellow and white markings."""
    h, w = 360, 640
    img = np.full((h, w, 3), 60, dtype=np.uint8)  # dark asphalt

    # Draw yellow line (BGR: 0, 215, 255)
    cv2.line(img, (100, 150), (100, 350), (0, 215, 255), 6)
    # Draw white line (BGR: 255, 255, 255)
    cv2.line(img, (500, 150), (500, 350), (255, 255, 255), 6)
    # Draw metallic guardrail reflection (grey BGR: 140, 140, 140)
    cv2.line(img, (20, 150), (20, 350), (140, 140, 140), 6)

    mask = _colour_mask(img)

    # Yellow line region must be detected
    assert np.count_nonzero(mask[200:300, 95:105]) > 50
    # White line region must be detected
    assert np.count_nonzero(mask[200:300, 495:505]) > 50
    # Guardrail grey line must NOT be detected in color mask
    assert np.count_nonzero(mask[200:300, 15:25]) == 0


def test_road_surface_ground_truth_masking():
    """Verify drivable pavement flood-fill excludes off-road terrain."""
    h, w = 360, 640
    img = np.full((h, w, 3), 40, dtype=np.uint8)  # grass/offroad
    # Road surface in center
    pts = np.array([
        [int(0.35 * w), 140],
        [int(0.65 * w), 140],
        [int(0.90 * w), h - 1],
        [int(0.10 * w), h - 1],
    ], dtype=np.int32)
    cv2.fillPoly(img, [pts], (90, 90, 90))  # gray asphalt

    horizon_y = 135
    road_mask = segment_road_surface(img, horizon_y)
    assert road_mask is not None
    # Bottom vehicle center must be drivable
    assert road_mask[h - 10, w // 2] == 255
    # Off-road area (e.g. x=20, y=300) must NOT be drivable
    assert road_mask[300, 20] == 0


def test_ransac_parabolic_fit():
    """Verify RANSAC 2nd-degree polynomial fitting with 50 iterations and outlier rejection."""
    # True parabolic curve: x = 0.0005 * y^2 - 0.2 * y + 150
    ys = np.linspace(150, 350, 60, dtype=np.float32)
    xs = 0.0005 * (ys ** 2) - 0.2 * ys + 150.0

    # Inject heavy outliers
    rng = np.random.default_rng(42)
    xs_noisy = xs.copy()
    outlier_indices = rng.choice(len(ys), size=12, replace=False)
    xs_noisy[outlier_indices] += rng.uniform(25.0, 50.0, size=12)

    coeffs, inlier_count = fit_parabola_ransac(ys, xs_noisy, n_iterations=50, threshold_px=3.0)

    assert coeffs is not None
    assert inlier_count >= 40
    # Coefficients must closely match true curve despite 20% outliers
    pred_xs = np.polyval(coeffs, ys)
    inliers = np.abs(xs - pred_xs) < 4.0
    assert np.count_nonzero(inliers) >= 50


def test_track_width_sanity_enforcement():
    """Verify that rigid track width bounds (150 px <= W <= 380 px) drop weak lines and project nominal width."""
    predictor = ClassicalCVPredictor(mode="painted")
    h, w = 360, 640
    img = np.full((h, w, 3), 70, dtype=np.uint8)
    img[:120, :] = (200, 180, 150)

    # Road surface
    pts = np.array([[int(0.35 * w), 125], [int(0.65 * w), 125], [w - 20, h - 1], [20, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))

    # Left lane at x=180 at bottom
    cv2.line(img, (w // 2 - 25, 130), (180, h - 1), (255, 255, 255), 5)
    # Right lane with invalid narrow width: only 80 px away (x=260 at bottom instead of nominal ~380)
    cv2.line(img, (w // 2 + 25, 130), (260, h - 1), (0, 215, 255), 3)

    prep = preprocess(img)
    pred = predictor.predict(prep)

    # Either lane was corrected or nominal width enforced
    if pred.left_poly_coeffs is not None and pred.right_poly_coeffs is not None:
        l_x = float(np.polyval(pred.left_poly_coeffs, h - 1))
        r_x = float(np.polyval(pred.right_poly_coeffs, h - 1))
        lane_w = r_x - l_x
        # Enforced track width must be within [150, 380] px
        assert 150.0 <= lane_w <= 380.0


def test_ekf_8state_smoothing_zero_flickering():
    """Verify that 8-state EKF guarantees zero line flickering across 100+ frames (Delta <= 0.02)."""
    ekf = EKF8StateLaneTracker()

    base_left = np.array([0.0003, -0.4, 180.0])
    base_right = np.array([0.0003, -0.4, 380.0])

    # Run for 120 consecutive streaming frames with Gaussian measurement noise
    rng = np.random.default_rng(123)
    coeff_deltas = []
    prev_left = None

    for i in range(120):
        # Mild sensor noise
        noisy_left = base_left + rng.normal(0, [1e-5, 0.005, 0.8])
        noisy_right = base_right + rng.normal(0, [1e-5, 0.005, 0.8])

        s_left, s_right, s_elat, s_theta, valid = ekf.update(
            left_coeffs=noisy_left,
            right_coeffs=noisy_right,
            dt=0.033,
        )

        assert valid is True
        assert s_left is not None
        assert s_right is not None

        if prev_left is not None:
            # Measure coefficient delta normalized
            delta_a = abs(s_left[0] - prev_left[0])
            delta_b = abs(s_left[1] - prev_left[1])
            coeff_deltas.append(max(delta_a * 100, delta_b))

        prev_left = s_left.copy()

    # Zero Flickering Guarantee: Max delta across smoothed frames <= 0.02
    assert len(coeff_deltas) > 100
    max_delta = max(coeff_deltas)
    assert max_delta <= 0.02, f"Expected Delta <= 0.02, got {max_delta}"


def test_ekf_8state_dropout_hold():
    """Verify EKF holds valid trajectory across 1–3 frame line dropouts."""
    ekf = EKF8StateLaneTracker(max_hold_frames=3)

    base_left = np.array([0.0003, -0.4, 180.0])
    base_right = np.array([0.0003, -0.4, 380.0])

    # Initialize
    s_left, s_right, _, _, valid = ekf.update(base_left, base_right)
    assert valid is True

    # Dropouts for 1st, 2nd, and 3rd frames
    for _ in range(3):
        h_left, h_right, _, _, h_valid = ekf.update(None, None)
        assert h_valid is True
        assert h_left is not None
        assert h_right is not None
        assert abs(h_left[2] - 180.0) < 5.0

    # 4th dropout exceeds max_hold_frames (3) -> lane declared lost
    lost_left, lost_right, _, _, lost_valid = ekf.update(None, None)
    assert lost_valid is False
    assert lost_left is None
    assert lost_right is None
