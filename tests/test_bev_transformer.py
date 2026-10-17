"""
tests/test_bev_transformer.py
==============================
Unit tests for Task 5.2: Metric Scaling & BEV (Inverse Perspective Mapping):
1. Grid resolution and dimension calculation
2. Coordinate round-trip: Metric (X, Z) <-> BEV Pixel (u, v) <-> Image Pixel (x, y)
3. Homography image warp and unwarp dimensions and numerical safety
4. Track width metric recovery within 5% tolerance
5. RANSAC + Weighted Least Squares polynomial curve fitting in BEV space
6. Homography singularity rejection
7. CLI execution with --dry-run and --synthetic
"""

import math
from pathlib import Path
import subprocess
import sys
import pytest
import numpy as np
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.bev_transformer import BEVTransformer, update_bev_scaling
from deploy.calibrate_bev import simulate_synthetic_track_measurement, calibrate_bev_scaling


@pytest.fixture
def bev():
    """Fixture providing a standard BEVTransformer instance."""
    return BEVTransformer()


def test_bev_grid_dimensions(bev):
    """Verify BEV grid size matches configured metric range and resolution."""
    # X in [-5.0, 5.0] -> 10m / 0.025 = 400 px
    # Z in [1.0, 20.0] -> 19m / 0.025 = 760 px
    assert bev.bev_width_px == 400
    assert bev.bev_height_px == 760
    assert bev.meters_per_pixel == 0.025


def test_metric_to_pixel_roundtrip(bev):
    """Verify bidirectional conversion between metric coordinates and BEV pixels."""
    test_points = [
        (0.0, 5.0),
        (-2.5, 10.0),
        (1.75, 15.0),
        (0.0, 20.0),
        (-5.0, 1.0),
    ]

    for xm, zm in test_points:
        u, v = bev.metric_to_pixel(xm, zm)
        xm_rec, zm_rec = bev.pixel_to_metric(u, v)
        assert math.isclose(xm, xm_rec, abs_tol=1e-5), f"Failed X recovery for ({xm}, {zm})"
        assert math.isclose(zm, zm_rec, abs_tol=1e-5), f"Failed Z recovery for ({xm}, {zm})"


def test_image_to_metric_roundtrip(bev):
    """Verify forward and backward projection between perspective image and vehicle metric space."""
    # Sample ground points in metric frame
    pts_m = np.array([
        [-1.0, 3.0],
        [ 1.0, 3.0],
        [-1.5, 8.0],
        [ 1.5, 8.0],
        [ 0.0, 12.0],
    ], dtype=np.float32)

    # Metric -> Image -> Metric
    pts_img = bev.metric_to_image(pts_m)
    pts_m_rec = bev.image_to_metric(pts_img)

    np.testing.assert_allclose(pts_m, pts_m_rec, atol=1e-4)


def test_track_width_recovery(bev):
    """Verify measured track width in BEV matches ground truth within 5% tolerance."""
    target_width_m = 1.80
    measured_width_m = simulate_synthetic_track_measurement(bev, known_width_m=target_width_m, eval_distance_z_m=7.0)

    err_pct = abs(measured_width_m - target_width_m) / target_width_m * 100.0
    assert err_pct < 5.0, f"Measured width {measured_width_m:.4f}m error {err_pct:.2f}% exceeds 5% tolerance"


def test_image_warp_and_unwarp(bev):
    """Verify warping a synthetic frame generates expected BEV shape without NaNs or Infs."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2_rect = cv2.line(img, (200, 350), (300, 200), (255, 255, 255), 4)

    warped = bev.warp(img)
    assert warped.shape == (760, 400, 3)
    assert not np.isnan(warped).any()
    assert not np.isinf(warped).any()

    # Unwarp back to original resolution
    unwarped = bev.unwarp(warped, (640, 360))
    assert unwarped.shape == (360, 640, 3)


def test_metric_polynomial_fitting_with_ransac(bev):
    """Verify robust polynomial curve fitting in metric BEV coordinates with outliers."""
    # True curve: X_m = 0.02 * Z^2 - 0.1 * Z + 0.5 (slight curve)
    Z = np.linspace(2.0, 16.0, 50)
    X = 0.02 * Z**2 - 0.1 * Z + 0.5

    # Add 5 severe outliers
    X_noisy = X.copy()
    X_noisy[10] += 1.5
    X_noisy[20] -= 1.8
    X_noisy[35] += 2.2

    points_m = np.column_stack((X_noisy, Z))
    coeffs = bev.fit_metric_polynomial(points_m, degree=2, ransac=True, inlier_tol_m=0.10)

    assert coeffs is not None
    assert len(coeffs) == 3
    # Assert recovered coefficients are close to ground truth (ignoring outliers)
    assert math.isclose(coeffs[0], 0.02, abs_tol=0.015)
    assert math.isclose(coeffs[1], -0.10, abs_tol=0.08)
    assert math.isclose(coeffs[2], 0.50, abs_tol=0.20)


def test_singular_homography_rejection():
    """Verify near-singular homography input raises a descriptive ValueError."""
    # Collinear degenerate source points
    degenerate_config = {
        "x_range_m": [-5.0, 5.0],
        "z_range_m": [1.0, 20.0],
        "meters_per_pixel": 0.025,
        "src_points": [[100, 100], [200, 100], [300, 100], [400, 100]],
        "ground_points_m": [[-1, 2], [1, 2], [1, 10], [-1, 10]],
    }
    with pytest.raises(ValueError, match="Singular homography"):
        BEVTransformer(degenerate_config)


def test_cli_dry_run_synthetic():
    """Verify CLI execution: python deploy/calibrate_bev.py --dry-run --synthetic."""
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "deploy" / "calibrate_bev.py"),
        "--dry-run",
        "--synthetic",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + res.stderr
    assert res.returncode == 0, f"CLI command failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert "Starting BEV Metric Calibration" in out
    assert "BEV Metric Scaling successfully verified" in out
