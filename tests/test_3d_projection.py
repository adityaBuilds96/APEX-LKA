"""
tests/test_3d_projection.py
===========================
Unit test suite for BoundingBox3DProjector:
- Validates 2D-to-3D pinhole camera back-projection formulas
- Verifies central 50% ROI median depth sampling
- Validates 3D physical dimension estimations and class priors
- Tests Time-to-Impact (TTI) computation
- Tests robustness against NaN/Inf and boundary edge cases
"""

import math
import numpy as np
import pytest

from src.perception.object_detector import Detection, DetectionResult
from src.perception.projection_3d import (
    BoundingBox3DProjector,
    Object3D,
    ProjectionResult,
    CLASS_3D_PRIORS,
)


def test_projector_initialization():
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0)
    assert projector.fx == 500.0
    assert projector.fy == 500.0
    assert projector.cx == 320.0
    assert projector.cy == 180.0


def test_back_projection_geometry():
    """
    Test pinhole back-projection geometry:
    Given a detection centered at (u_c=320, v_c=180) (exact optical center) at depth 10.0m:
      X_m should be 0.0m
      Y_m should be 0.0m
      Z_m should be 10.0m
    """
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0)

    # Box centered at (320, 180) with width 40, height 40 -> [300, 160, 340, 200]
    det = Detection(
        box=np.array([300, 160, 340, 200], dtype=np.float32),
        score=0.90,
        class_id=0,
        class_name="vehicle",
    )
    det_res = DetectionResult(
        boxes=np.array([[300, 160, 340, 200]], dtype=np.float32),
        scores=np.array([0.90], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )

    # Depth map filled with 10.0m everywhere
    depth_map = np.full((360, 640), 10.0, dtype=np.float32)

    res = projector.project_2d_to_3d(det_res, depth_map)
    assert len(res) == 1
    obj = res.objects_3d[0]

    assert np.isclose(obj.x, 0.0, atol=1e-3)
    assert np.isclose(obj.y, 0.0, atol=1e-3)
    assert np.isclose(obj.z, 10.0, atol=1e-3)


def test_lateral_offset_math():
    """
    Given a pedestrian detection at u_c = 420 (100 px to the right of cx=320) at depth 5.0m:
      X_m = (420 - 320) * 5.0 / 500.0 = 100 * 5.0 / 500 = 1.0 m (1 meter to the right)
    """
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0)

    # Box from x1=400 to x2=440 -> uc=420. y1=160 to y2=200 -> vc=180
    det = Detection(
        box=np.array([400, 160, 440, 200], dtype=np.float32),
        score=0.85,
        class_id=1,
        class_name="pedestrian",
    )
    depth_map = np.full((360, 640), 5.0, dtype=np.float32)

    res = projector.project_2d_to_3d([det], depth_map)
    assert len(res) == 1
    obj = res.objects_3d[0]

    expected_x = (420.0 - 320.0) * 5.0 / 500.0  # 1.0 meter
    assert np.isclose(obj.x, expected_x, atol=1e-2)
    assert np.isclose(obj.z, 5.0, atol=1e-2)


def test_central_roi_median_filtering():
    """
    Test that the projector extracts the central 50% ROI and takes the median,
    ignoring outlier pixels on the edges (such as background road).
    """
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0, roi_central_fraction=0.50)

    # Box: x in [100, 200], y in [100, 200]. Central 50% is x in [125, 175], y in [125, 175]
    det = Detection(
        box=np.array([100, 100, 200, 200], dtype=np.float32),
        score=0.95,
        class_id=0,
        class_name="vehicle",
    )

    # Background depth is 30.0m
    depth_map = np.full((360, 640), 30.0, dtype=np.float32)
    # Target vehicle in central 50% ROI has depth 8.0m
    depth_map[125:175, 125:175] = 8.0

    res = projector.project_2d_to_3d([det], depth_map)
    assert len(res) == 1
    obj = res.objects_3d[0]

    # Median depth should be 8.0m, not 30.0m
    assert np.isclose(obj.z, 8.0, atol=1e-2)


def test_physical_dimension_estimation():
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0)

    det = Detection(
        box=np.array([250, 120, 390, 240], dtype=np.float32),
        score=0.88,
        class_id=0,
        class_name="vehicle",
    )
    depth_map = np.full((360, 640), 12.0, dtype=np.float32)

    res = projector.project_2d_to_3d([det], depth_map)
    obj = res.objects_3d[0]

    assert obj.width > 0.5 and obj.width < 5.0
    assert obj.height > 0.5 and obj.height < 4.0
    assert obj.length == CLASS_3D_PRIORS["vehicle"][2]


def test_time_to_impact_calculation():
    obj = Object3D(
        x=0.0,
        y=0.0,
        z=15.0,  # 15 meters away
        width=1.8,
        height=1.5,
        length=4.2,
        class_id=0,
        class_name="vehicle",
        confidence=0.9,
        box_2d=np.array([0, 0, 10, 10], dtype=np.float32),
    )

    # At 10 m/s (36 km/h), TTI = 15m / 10m/s = 1.5s = 1500 ms
    tti = BoundingBox3DProjector.compute_tti_ms(obj, ego_speed_mps=10.0)
    assert tti == 1500

    # Stationary vehicle (0 m/s) -> 65535 (safe / no impact)
    tti_stopped = BoundingBox3DProjector.compute_tti_ms(obj, ego_speed_mps=0.0)
    assert tti_stopped == 65535


def test_empty_and_nan_resilience():
    projector = BoundingBox3DProjector()
    empty_res = projector.project_2d_to_3d([], np.zeros((360, 640), dtype=np.float32))
    assert empty_res.is_empty
    assert empty_res.nearest_object is None

    # Depth map with NaNs
    nan_depth = np.full((360, 640), np.nan, dtype=np.float32)
    det = Detection(
        box=np.array([10, 10, 50, 50], dtype=np.float32),
        score=0.7,
        class_id=3,
        class_name="traffic_cone",
    )
    res = projector.project_2d_to_3d([det], nan_depth)
    assert len(res) == 1
    # Fallback must yield finite valid positive depth
    assert not np.isnan(res.objects_3d[0].z)
    assert res.objects_3d[0].z > 0.0
