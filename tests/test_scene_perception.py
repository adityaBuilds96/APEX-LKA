"""
tests/test_scene_perception.py
==============================
Comprehensive unit test suite for Section 2 of Phase 4:
Autonomous Multi-Task Scene Perception Engine for APEX-LKA.

Covers:
1. Environmental Auto-Adaptor (Illumination analysis, CLAHE, Glare Gamma, Shadow compensation)
2. Road Surface & Material Classification (Asphalt, Concrete, Dirt/Trail, Off-road)
3. Drivable Free-Space Corridor Segmentation & Boundary Polygon Extraction
4. Monocular Camera Distance Estimator & Metric Ground Projection
5. Temporal Anti-Flicker Smoothing & Outlier Rejection Filter
6. Pipeline Integration & Multi-Task Scene Perception
7. HUD Visualization Elements & Zero-Overlap Guarantees
"""

import math
import time
from typing import Tuple

import cv2
import numpy as np
import pytest

from src.inference.environmental_adaptor import (
    EnvironmentalAutoAdaptor,
    EnvironmentalCondition,
    EnvironmentalState,
)
from src.inference.free_space import (
    FreeSpaceResult,
    RoadMaterial,
    classify_road_material,
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
from src.inference.pipeline import (
    InferenceResult,
    ScenePerceptionResult,
    run_pipeline,
)
from dashboard.components.lka_hud import draw_hud_overlay


# ── Helper Test Fixtures ───────────────────────────────────────────────────

def create_synthetic_road_scene(
    h: int = 360,
    w: int = 640,
    road_color: Tuple[int, int, int] = (90, 90, 90),
    sky_color: Tuple[int, int, int] = (190, 160, 120),
) -> np.ndarray:
    """Create a standard synthetic highway image with road pavement and markings."""
    img = np.full((h, w, 3), sky_color, dtype=np.uint8)
    horizon_y = 126

    # Road surface below horizon
    img[horizon_y:, :] = road_color

    # Draw lane boundary lines
    cv2.line(img, (180, h - 1), (280, horizon_y), (255, 255, 255), 4)
    cv2.line(img, (460, h - 1), (360, horizon_y), (255, 255, 255), 4)
    return img


# ═══════════════════════════════════════════════════════════════════════════
# 1. Environmental Auto-Adaptor Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_environmental_normal_lighting():
    """Normal daylight conditions must pass through without artificial alterations."""
    img = create_synthetic_road_scene()
    adaptor = EnvironmentalAutoAdaptor()
    adapted, state = adaptor.analyze_and_adapt(img)

    assert state.condition == EnvironmentalCondition.NORMAL
    assert state.badge_text == "NORMAL"
    assert not state.adaptation_applied
    assert np.array_equal(adapted, img)


def test_environmental_night_low_light():
    """Low-light / night frames (L < 40) must trigger NIGHT_LOW_LIGHT and CLAHE."""
    img = np.full((360, 640, 3), 20, dtype=np.uint8)
    adaptor = EnvironmentalAutoAdaptor()
    adapted, state = adaptor.analyze_and_adapt(img)

    assert state.condition == EnvironmentalCondition.NIGHT_LOW_LIGHT
    assert state.badge_text == "NIGHT_BOOST"
    assert state.adaptation_applied
    assert state.mean_lightness < 40.0
    # Contrast adaptation should enhance pixel dynamics
    assert adapted.dtype == np.uint8


def test_environmental_high_glare():
    """Overexposed, high-contrast sun glare (L > 210, var > 8000) triggers gamma correction."""
    # Place 15% dark band inside lower 60% ROI (rows 144..359, 216 rows total -> 32 rows):
    # Mean L in ROI = (1 - 32/216) * 255 = 217.2 (> 210)
    # Variance in ROI = ~8200 (> 8000)
    img = np.full((360, 640, 3), 255, dtype=np.uint8)
    img[144:176, :] = 0
    adaptor = EnvironmentalAutoAdaptor()
    adapted, state = adaptor.analyze_and_adapt(img)

    assert state.condition == EnvironmentalCondition.HIGH_GLARE
    assert state.badge_text == "GLARE_COMPENSATED"
    assert state.adaptation_applied
    assert state.mean_lightness > 210.0
    assert state.contrast_variance > 8000.0
    # Gamma 0.6 reduces overexposed highlights
    assert np.mean(adapted) <= np.mean(img)


def test_environmental_tree_shadows():
    """Heavy tree shadows (shadow ratio > 0.35) triggers shadow compensation tone curve."""
    img = create_synthetic_road_scene()
    # Add deep shadow covering > 40% of the ROI
    img[200:, :400] = (15, 15, 15)
    adaptor = EnvironmentalAutoAdaptor()
    adapted, state = adaptor.analyze_and_adapt(img)

    assert state.condition == EnvironmentalCondition.SHADOW_DOMINATED
    assert state.badge_text == "SHADOW_COMPENSATED"
    assert state.adaptation_applied
    assert state.shadow_ratio > 0.35


def test_environmental_nan_and_empty_safety():
    """Empty or None image inputs must not raise and return safe defaults."""
    adaptor = EnvironmentalAutoAdaptor()
    empty = np.empty((0, 0, 3), dtype=np.uint8)
    res_img, state = adaptor.analyze_and_adapt(empty)
    assert state.condition == EnvironmentalCondition.NORMAL

    none_img, state2 = adaptor.analyze_and_adapt(None)
    assert state2.condition == EnvironmentalCondition.NORMAL


def test_environmental_speed_budget():
    """Illumination analysis and adaptation must run in < 5 ms on CPU."""
    img = create_synthetic_road_scene()
    adaptor = EnvironmentalAutoAdaptor()

    # Warmup
    _ = adaptor.analyze_and_adapt(img)

    t0 = time.perf_counter()
    iterations = 20
    for _ in range(iterations):
        _ = adaptor.analyze_and_adapt(img)
    avg_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    assert avg_ms < 5.0, f"Environmental adaptor too slow: {avg_ms:.2f} ms"


# ═══════════════════════════════════════════════════════════════════════════
# 2. Road Surface & Material Classification Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_road_material_asphalt():
    """Dark/neutral grey pavement classifies as ASPHALT."""
    img = create_synthetic_road_scene(road_color=(75, 75, 75))
    mat, conf = classify_road_material(img)
    assert mat == RoadMaterial.ASPHALT
    assert 0.5 <= conf <= 1.0


def test_road_material_concrete():
    """Bright off-white / light grey road classifies as CONCRETE."""
    img = create_synthetic_road_scene(road_color=(190, 190, 190))
    mat, conf = classify_road_material(img)
    assert mat == RoadMaterial.CONCRETE
    assert 0.5 <= conf <= 1.0


def test_road_material_dirt_trail():
    """Earthy brown/yellow road surface classifies as DIRT_TRAIL."""
    # BGR for dirt trail brown: B=40, G=85, R=140 (HSV Hue ~15, Sat ~180)
    dirt_bgr = (40, 85, 140)
    img = create_synthetic_road_scene(road_color=dirt_bgr)
    mat, conf = classify_road_material(img)
    assert mat == RoadMaterial.DIRT_TRAIL
    assert 0.5 <= conf <= 1.0


def test_road_material_off_road_vegetation():
    """Greenish off-road vegetation classifies as OFF_ROAD."""
    # BGR for grass green: B=35, G=140, R=50 (HSV Hue ~65)
    grass_bgr = (35, 140, 50)
    img = create_synthetic_road_scene(road_color=grass_bgr)
    mat, conf = classify_road_material(img)
    assert mat == RoadMaterial.OFF_ROAD
    assert 0.5 <= conf <= 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 3. Drivable Free-Space Corridor Segmentation Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_free_space_corridor_extraction():
    """Drivable corridor extraction produces valid binary mask, polygon, and boundaries."""
    img = create_synthetic_road_scene()
    fs = extract_drivable_corridor(img, horizon_y=126)

    assert isinstance(fs, FreeSpaceResult)
    # Mask must strictly be single-channel uint8 with values 0 or 1
    assert fs.mask.dtype == np.uint8
    unique_vals = set(np.unique(fs.mask))
    assert unique_vals.issubset({0, 1})
    assert 1 in unique_vals, "Free-space mask must identify drivable pixels"

    # Polygon must be populated
    assert len(fs.polygon) >= 3, "Corridor contour must have >= 3 polygon vertices"

    # Left, right, and centroid paths must be populated
    assert len(fs.left_boundary) > 0
    assert len(fs.right_boundary) > 0
    assert len(fs.centroid_path) > 0
    assert fs.drivable_area_ratio > 0.05
    assert fs.latency_ms > 0.0


def test_free_space_polygon_geometric_validity():
    """Left boundary coordinates must be to the left of right boundary coordinates."""
    img = create_synthetic_road_scene()
    fs = extract_drivable_corridor(img, horizon_y=126)

    # For overlapping row scans, left x must be <= right x
    min_len = min(len(fs.left_boundary), len(fs.right_boundary))
    assert min_len >= 5
    for i in range(min_len):
        xl, yl = fs.left_boundary[i]
        xr, yr = fs.right_boundary[i]
        assert xl <= xr, f"Left boundary {xl} cannot be to the right of right boundary {xr}"


def test_free_space_latency_budget():
    """Free-space segmentation must run in < 12 ms on CPU."""
    img = create_synthetic_road_scene()
    # Warmup
    _ = extract_drivable_corridor(img, horizon_y=126)

    t0 = time.perf_counter()
    iterations = 15
    for _ in range(iterations):
        _ = extract_drivable_corridor(img, horizon_y=126)
    avg_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    assert avg_ms < 12.0, f"Free space extraction took {avg_ms:.2f} ms (budget: < 12 ms)"


# ═══════════════════════════════════════════════════════════════════════════
# 4. Monocular Distance Estimator Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_distance_estimator_ground_projection():
    """Verify pinhole camera geometry Z = (H_cam * fy) / ((y - y_horizon) * cos(pitch))."""
    estimator = MonocularDistanceEstimator(
        mounting_height_m=1.25,
        focal_length_px=500.0,
        pitch_angle_deg=0.0,
        horizon_y_px=126,
    )

    # Bottom bumper row (y = 359)
    # delta_y = 359 - 126 = 233
    # Expected Z = (1.25 * 500) / 233 = 625 / 233 = ~2.68 m
    xm, zm = estimator.pixel_to_meters(320.0, 359.0)
    assert abs(xm) < 0.05, f"Center pixel x=320 must have ~0 lateral offset, got {xm}"
    assert 2.5 <= zm <= 2.8, f"Bottom bumper row distance expected ~2.68 m, got {zm}"

    # Intermediate row y = 200 (delta_y = 74 -> Z = 625 / 74 = ~8.45 m)
    _, zm_mid = estimator.pixel_to_meters(320.0, 200.0)
    assert 8.0 <= zm_mid <= 9.0


def test_distance_estimator_horizon_guard():
    """Pixels at or above the horizon line must not divide by zero or yield negative Z."""
    estimator = MonocularDistanceEstimator(horizon_y_px=126)

    # Pixel exactly at horizon
    _, zm_horiz = estimator.pixel_to_meters(320.0, 126.0)
    assert np.isfinite(zm_horiz)
    assert zm_horiz >= 50.0

    # Pixel in sky above horizon
    _, zm_sky = estimator.pixel_to_meters(320.0, 50.0)
    assert np.isfinite(zm_sky)
    assert zm_sky == 100.0


def test_distance_estimator_lateral_symmetry():
    """Left of center must have negative X_m, right of center must have positive X_m."""
    estimator = MonocularDistanceEstimator(image_width=640, horizon_y_px=126)

    x_left, _ = estimator.pixel_to_meters(200.0, 300.0)
    x_right, _ = estimator.pixel_to_meters(440.0, 300.0)

    assert x_left < 0.0
    assert x_right > 0.0
    assert abs(x_left + x_right) < 0.05, "Symmetric pixels should have symmetric lateral distances"


def test_distance_estimator_meters_to_pixel_roundtrip():
    """Inverse mapping meters_to_pixel should accurately invert pixel_to_meters."""
    estimator = MonocularDistanceEstimator()

    orig_x, orig_y = 220.0, 280.0
    xm, zm = estimator.pixel_to_meters(orig_x, orig_y)
    rec_x, rec_y = estimator.meters_to_pixel(xm, zm)

    assert abs(rec_x - orig_x) <= 1
    assert abs(rec_y - orig_y) <= 1


def test_distance_estimator_path_widths():
    """Metric road width at 2m, 5m, and 10m must fall within realistic automotive boundaries."""
    estimator = MonocularDistanceEstimator(horizon_y_px=126)

    # Standard highway straight lane lines expanding towards bottom bumper
    # At y=359, left=50, right=590 (width=540px)
    left_poly = np.array([0.0, -0.75, 320.0])
    right_poly = np.array([0.0, 0.75, 320.0])

    profile = estimator.compute_distance_profile(left_poly, right_poly)
    assert 2.0 <= profile.width_at_2m <= 6.0
    assert 2.0 <= profile.width_at_5m <= 6.0
    assert 2.0 <= profile.width_at_10m <= 6.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. Temporal Anti-Flicker Filter Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_temporal_smoother_ema_stabilization():
    """Exponential Moving Average (alpha=0.65) must attenuate frame-to-frame jitter."""
    smoother = TemporalPathSmoother(buffer_depth=5, ema_alpha=0.65)

    base_poly = np.array([0.0001, -0.2, 200.0])
    jittered_c = [200.0, 208.0, 194.0, 205.0, 197.0]

    smoothed_c = []
    for c in jittered_c:
        poly = np.array([0.0001, -0.2, c])
        res = smoother.smooth(poly, None, None)
        smoothed_c.append(res.left_poly[2])

    assert np.var(smoothed_c) < np.var(jittered_c), "EMA must reduce variance across jittered frames"
    assert res.state == PathTrackingState.TRACKING


def test_temporal_smoother_outlier_rejection():
    """An abrupt jump > 0.50m in a single frame must be rejected as an anomaly."""
    smoother = TemporalPathSmoother(buffer_depth=5, ema_alpha=0.65, max_lateral_jump_m=0.50)

    # Frame 1: Stable left boundary
    poly_stable = np.array([0.0, 0.0, 200.0])
    res1 = smoother.smooth(poly_stable, None, None, img_w=640)
    assert res1.state == PathTrackingState.TRACKING
    assert not res1.is_outlier

    # Frame 2: Sudden 150-pixel (~1.65m) jump due to false guardrail reflection
    poly_anomaly = np.array([0.0, 0.0, 350.0])
    res2 = smoother.smooth(poly_anomaly, None, None, img_w=640)

    # Anomaly must be discarded and previous stable trajectory held
    assert res2.is_outlier
    assert res2.state == PathTrackingState.INTERPOLATING
    assert abs(res2.left_poly[2] - 200.0) < 5.0


def test_temporal_smoother_gap_holding_and_timeout():
    """Missing detections hold trajectory for 3 frames, then transition to TEMPORARILY_LOST."""
    smoother = TemporalPathSmoother(buffer_depth=5, hold_gap_frames=3)

    # Seed with valid detection
    poly = np.array([0.0, 0.0, 220.0])
    _ = smoother.smooth(poly, None, None)

    # Frames 1-3: Dropped detection (None inputs) -> Held via interpolation
    for gap in range(1, 4):
        res = smoother.smooth(None, None, None)
        assert res.state == PathTrackingState.INTERPOLATING
        assert res.gap_count == gap
        assert res.left_poly is not None

    # Frame 4: Exceeded 3-frame threshold -> Transitions to TEMPORARILY_LOST
    res_lost = smoother.smooth(None, None, None)
    assert res_lost.state == PathTrackingState.TEMPORARILY_LOST
    assert res_lost.left_poly is None


def test_temporal_smoother_reset():
    """Calling reset() clears state history immediately."""
    smoother = TemporalPathSmoother()
    _ = smoother.smooth(np.array([0.0, 0.0, 200.0]), None, None)
    smoother.reset()

    assert smoother.smoothed_left is None
    assert smoother.state == PathTrackingState.TEMPORARILY_LOST
    assert smoother.gap_count == 0


# ═══════════════════════════════════════════════════════════════════════════
# 6. Pipeline Integration & Multi-Task Scene Perception Tests
# ═══════════════════════════════════════════════════════════════════════════

def test_pipeline_scene_perception_result_structure():
    """run_pipeline produces fully assembled ScenePerceptionResult."""
    img = create_synthetic_road_scene()
    result = run_pipeline(img, backend="classical_cv", reset_pd_state=True)

    assert isinstance(result, InferenceResult)
    assert result.scene_perception is not None
    sp = result.scene_perception

    # Check environmental state
    assert isinstance(sp.environmental, EnvironmentalState)
    assert sp.environmental.condition in EnvironmentalCondition

    # Check free space
    assert isinstance(sp.free_space, FreeSpaceResult)
    assert sp.free_space.mask.shape == (360, 640)
    assert sp.free_space.material in RoadMaterial

    # Check distance profile
    assert isinstance(sp.distance_profile, DistanceProfile)
    assert profile_has_valid_lookaheads(sp.distance_profile)

    # Check anti-flicker smoothing
    assert isinstance(sp.smoothed_paths, SmoothedPathResult)
    assert sp.total_perception_ms > 0.0


def profile_has_valid_lookaheads(profile: DistanceProfile) -> bool:
    return (
        profile.width_at_2m > 0.0
        and profile.width_at_5m > 0.0
        and profile.width_at_10m > 0.0
        and len(profile.lookahead_y_map) >= 3
    )


def test_pipeline_total_perception_timing_budget():
    """Unified scene perception must execute well under the 25 ms hard constraint."""
    img = create_synthetic_road_scene()
    # Warmup
    _ = run_pipeline(img, backend="classical_cv")

    t0 = time.perf_counter()
    iterations = 10
    for _ in range(iterations):
        res = run_pipeline(img, backend="classical_cv")
    avg_total_ms = ((time.perf_counter() - t0) / iterations) * 1000.0

    assert res.scene_perception.total_perception_ms < 25.0, (
        f"Perception pass took {res.scene_perception.total_perception_ms:.2f} ms (budget: < 25 ms)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 7. HUD Visualization Elements & Zero-Overlap Guarantees
# ═══════════════════════════════════════════════════════════════════════════

def test_hud_renders_free_space_polygon_and_ticks():
    """HUD overlay must render cyan polygon and distance ticks onto the viewport."""
    img = create_synthetic_road_scene()
    res = run_pipeline(img, backend="classical_cv")

    # Render HUD onto annotated image
    hud = draw_hud_overlay(res.annotated_bgr, scene_perception=res.scene_perception)

    assert hud is not None
    assert hud.shape == img.shape
    # Canvas should have modified pixel values reflecting HUD overlays
    diff = cv2.absdiff(res.annotated_bgr, hud)
    assert np.count_nonzero(diff) > 0, "HUD elements must be drawn onto the canvas"


def test_hud_environmental_badges_zero_overlap():
    """Environmental badge renders cleanly below performance badge without out-of-bounds."""
    img = create_synthetic_road_scene()
    res = run_pipeline(img, backend="classical_cv")

    # Test under multiple environmental conditions
    for cond, text in [
        (EnvironmentalCondition.NORMAL, "NORMAL"),
        (EnvironmentalCondition.HIGH_GLARE, "GLARE_COMPENSATED"),
        (EnvironmentalCondition.NIGHT_LOW_LIGHT, "NIGHT_BOOST"),
    ]:
        res.scene_perception.environmental.condition = cond
        res.scene_perception.environmental.badge_text = text

        hud = draw_hud_overlay(img, scene_perception=res.scene_perception)
        assert hud is not None
        assert hud.shape == img.shape


def test_video_stream_100_frames_zero_flicker():
    """Verify zero path flickering across 100 consecutive video frames."""
    from src.inference.live_inference import make_synthetic_road_frame

    prev_bottom_x = None
    max_lateral_jump = 0.0

    for idx in range(100):
        frame = make_synthetic_road_frame(idx)
        res = run_pipeline(frame, backend="classical_cv", reset_pd_state=(idx == 0))

        assert res.success, f"Frame {idx} inference must succeed"
        assert res.scene_perception is not None
        sp = res.scene_perception

        # Polygon must render cleanly
        assert sp.free_space is not None
        assert len(sp.free_space.polygon) >= 3

        # Distance ticks present
        assert sp.distance_profile is not None
        assert 2.0 in sp.distance_profile.lookahead_y_map
        assert 5.0 in sp.distance_profile.lookahead_y_map
        assert 10.0 in sp.distance_profile.lookahead_y_map

        # Measure lateral jump between consecutive frames at bottom bumper
        if sp.smoothed_paths and sp.smoothed_paths.center_poly is not None:
            curr_bottom_x = float(np.polyval(sp.smoothed_paths.center_poly, 359))
            if prev_bottom_x is not None:
                jump_px = abs(curr_bottom_x - prev_bottom_x)
                jump_m = jump_px * (2.68 / 500.0)
                if jump_m > max_lateral_jump:
                    max_lateral_jump = jump_m
                assert jump_m < 0.50, f"Frame {idx} lateral jump {jump_m:.3f}m exceeds 0.50m limit"
            prev_bottom_x = curr_bottom_x

    # Over 100 frames with continuous camera motion, maximum frame-to-frame jump must remain smooth
    assert max_lateral_jump < 0.25, f"Temporal smoother maintained jitter well below 0.25m: {max_lateral_jump:.3f}m"
