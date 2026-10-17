"""
tests/test_vio_engine.py
========================
Unit tests for Visual-Inertial Odometry (VIO) engine.
Validates:
- Lucas-Kanade optical flow feature tracking on synthetic ground frames
- Execution latency <= 5.0 ms per frame budget
- Relative displacement [dx, dy] and rotation dpsi estimation
- IMU gyro complementary filter fusion
- Trajectory trail history accumulation and reset
"""

import math
import time
import pytest
import cv2
import numpy as np

from src.fusion.vio_engine import (
    VisualInertialOdometryEngine,
    VIOConfig,
    EgoMotionMeasurement,
)


def make_test_frame_with_features(shift_x: float = 0.0, shift_y: float = 0.0, angle_deg: float = 0.0) -> np.ndarray:
    """Creates a high-contrast road surface frame with randomized corner textures."""
    np.random.seed(42)
    h, w = 360, 640
    frame = np.full((h, w), 80, dtype=np.uint8)

    # Draw grid of textured road patches
    for y in range(140, h - 20, 25):
        for x in range(40, w - 40, 30):
            cv2.rectangle(frame, (x, y), (x + 12, y + 12), int(150 + 80 * np.sin(x * y)), -1)
            cv2.circle(frame, (x + 6, y + 6), 3, 255, -1)

    # Apply rigid transformation (shift + rotation)
    if shift_x != 0.0 or shift_y != 0.0 or angle_deg != 0.0:
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        M[0, 2] += shift_x
        M[1, 2] += shift_y
        frame = cv2.warpAffine(frame, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    return frame


@pytest.fixture
def vio():
    config = VIOConfig(max_features=200, min_feature_inliers=15)
    return VisualInertialOdometryEngine(config=config)


def test_vio_initialization(vio):
    """Test initial VIO engine state and empty trajectory."""
    assert vio.prev_gray is None
    assert vio.prev_pts is None
    trail = vio.get_trajectory_trail()
    assert len(trail) == 1
    assert trail[0] == (0.0, 0.0)


def test_vio_first_frame_boot(vio):
    """First frame should initialize corner features and return valid=False."""
    f0 = make_test_frame_with_features()
    meas0 = vio.process_frame(f0)

    assert not meas0.valid
    assert meas0.num_tracked > 15
    assert vio.prev_pts is not None


def test_vio_tracking_translation(vio):
    """Test optical flow tracking on translated ground frames."""
    f0 = make_test_frame_with_features(shift_x=0.0, shift_y=0.0)
    vio.process_frame(f0, dt=0.033)

    # Downward optical flow (forward motion: shift_y = +6 px)
    f1 = make_test_frame_with_features(shift_x=0.0, shift_y=6.0)
    meas1 = vio.process_frame(f1, dt=0.033, speed_mps=3.0)

    assert meas1.valid
    assert meas1.num_tracked >= 15
    assert meas1.inlier_ratio >= 0.60
    assert meas1.dx > 0.05  # Forward displacement detected
    assert abs(meas1.dyaw) < 0.05  # Negligible yaw rotation


def test_vio_tracking_rotation_and_imu_fusion(vio):
    """Test relative yaw rotation recovery and complementary IMU gyro fusion."""
    f0 = make_test_frame_with_features(shift_x=0.0, shift_y=0.0)
    vio.process_frame(f0, dt=0.033)

    # Rotated frame by -1.5 degrees (~ -0.026 rad)
    f1 = make_test_frame_with_features(shift_x=0.0, shift_y=0.0, angle_deg=-1.5)
    meas1 = vio.process_frame(f1, dt=0.033, imu_yaw_rate=-0.80)

    assert meas1.valid
    # Yaw rotation should reflect negative heading change
    assert meas1.dyaw < 0.0


def test_vio_latency_budget(vio):
    """Hard technical constraint: VIO optical flow processing MUST execute in <= 5.0 ms."""
    f0 = make_test_frame_with_features(0.0, 0.0)
    f1 = make_test_frame_with_features(0.0, 3.0)

    # Warmup
    vio.process_frame(f0)
    vio.process_frame(f1)

    t0 = time.perf_counter()
    iterations = 50
    for i in range(iterations):
        shift = float(i % 5)
        f = make_test_frame_with_features(0.0, shift)
        vio.process_frame(f, dt=0.033)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / iterations

    assert elapsed_ms <= 5.0, f"VIO processing latency {elapsed_ms:.2f} ms exceeds <= 5.0 ms budget"


def test_vio_trajectory_trail_accumulation(vio):
    """Verify trajectory history is accumulated and bounded to 60 points."""
    f0 = make_test_frame_with_features(0.0, 0.0)
    vio.process_frame(f0)

    for i in range(70):
        f = make_test_frame_with_features(0.0, float((i % 4) + 1))
        vio.process_frame(f, speed_mps=2.5, dt=0.033)

    trail = vio.get_trajectory_trail()
    assert 2 <= len(trail) <= 60
    # Trajectory should have advanced forward in cumulative distance
    last_pt = trail[-1]
    assert last_pt[0] > 0.0 or last_pt[1] > 0.0


def test_vio_reset(vio):
    """Test clean reset of state and trajectory."""
    f0 = make_test_frame_with_features(0.0, 0.0)
    vio.process_frame(f0)
    vio.reset()

    assert vio.prev_gray is None
    assert vio.prev_pts is None
    assert len(vio.get_trajectory_trail()) == 1
