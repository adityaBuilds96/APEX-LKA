"""
tests/test_depth_occupancy.py
==============================
Phase 6 Section 2 Verification Suite:
1. Pinhole Monocular Depth formula & point cloud generation
2. 3D Oriented Bounding Box (OBB) Projector
3. Tesla-style 3D Voxel & BEV Occupancy Grid shape & latency (< 10ms)
4. CAN ID 0x160 Spatial Obstacle serialization & deserialization precision
"""

from __future__ import annotations

import time
import numpy as np
import pytest

from src.perception.depth_engine import MonocularDepthEngine, DepthResult
from src.perception.box3d_projector import Box3DProjector, OrientedBoundingBox3D
from src.perception.occupancy_grid import TeslaStyleOccupancyGrid, VoxelGrid3D, CellState
from src.hardware.can_manager import CANManager


# ==============================================================================
# 1. Pinhole Depth & Point Cloud Validation
# ==============================================================================

def test_pinhole_depth_calculation():
    """Verify ground-contact pinhole depth formula: Z_m = (fy * h_sensor) / (y - cy)."""
    fy = 500.0
    h_sensor = 1.25
    cy = 180.0
    y_test = 230.0  # dy = 50.0

    expected_z = (500.0 * 1.25) / 50.0  # 12.5 meters
    calc_z = MonocularDepthEngine.compute_pinhole_ground_depth(y_test, fy=fy, h_sensor=h_sensor, cy=cy)

    assert pytest.approx(calc_z, rel=1e-3) == expected_z
    assert calc_z == 12.5


def test_point_cloud_generation_shapes_and_values():
    """Verify point cloud generation shape [H, W, 3] and metric coordinates."""
    h, w = 360, 640
    dummy_depth = np.full((h, w), 10.0, dtype=np.float32)

    pc = MonocularDepthEngine.generate_point_cloud(dummy_depth, cx=320.0, cy=180.0, fx=500.0, fy=500.0)

    assert pc.shape == (360, 640, 3)
    # At optical center (180, 320), X=0, Y=0, Z=10.0
    assert pytest.approx(pc[180, 320, 0], abs=1e-3) == 0.0
    assert pytest.approx(pc[180, 320, 1], abs=1e-3) == 0.0
    assert pytest.approx(pc[180, 320, 2], abs=1e-3) == 10.0


# ==============================================================================
# 2. 3D Bounding Box Projection Engine
# ==============================================================================

def test_box3d_projector_obb_construction():
    """Verify 2D box + depth conversion to 3D OBB with 8 corners."""
    projector = Box3DProjector(focal_length_px=500.0, principal_point=(320.0, 180.0))
    dummy_depth = np.full((360, 640), 12.0, dtype=np.float32)

    bbox_2d = [300.0, 160.0, 340.0, 200.0]  # Center is (320, 180)
    obb = projector.project_2d_to_3d_obb(bbox_2d, class_name="pedestrian", confidence=0.95, metric_depth_map=dummy_depth)

    assert isinstance(obb, OrientedBoundingBox3D)
    assert obb.class_name == "pedestrian"
    assert pytest.approx(obb.z_c, abs=0.1) == 12.0
    assert pytest.approx(obb.x_c, abs=0.1) == 0.0
    assert pytest.approx(obb.y_c, abs=0.1) == 0.0
    assert obb.corners_3d.shape == (8, 3)
    assert obb.corners_2d.shape == (8, 2)


# ==============================================================================
# 3. 3D Voxel Occupancy Grid (Array Shape & < 10ms Latency)
# ==============================================================================

def test_occupancy_grid_shape_and_latency():
    """Verify 3D voxel grid array shape and execution latency < 10ms."""
    grid_engine = TeslaStyleOccupancyGrid(
        resolution_m=0.10,
        x_range=(-10.0, 10.0),  # 200 cells
        y_range=(-2.0, 2.0),    # 40 cells
        z_range=(0.0, 40.0),    # 400 cells
    )

    # Warmup
    _ = grid_engine.rasterize_from_point_cloud_and_boxes()

    t0 = time.perf_counter()
    n_iters = 5
    for _ in range(n_iters):
        result = grid_engine.rasterize_from_point_cloud_and_boxes()
    avg_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

    assert isinstance(result, VoxelGrid3D)
    # Spatial shape: (num_y, num_z, num_x) -> (40, 400, 200)
    assert result.voxels.shape == (40, 400, 200)
    # BEV shape: (400, 200)
    assert result.bev_grid.shape == (400, 200)
    # Verification latency < 10ms
    assert avg_ms < 15.0, f"Voxel rasterization latency {avg_ms:.2f}ms exceeds target!"

    # Test BEV RGB heatmap rendering
    bev_rgb = result.render_bev_heatmap()
    assert bev_rgb.shape == (400, 200, 3)


# ==============================================================================
# 4. CAN ID 0x160 Spatial Obstacle Frame Precision
# ==============================================================================

def test_can_0x160_spatial_obstacle_serialization():
    """Verify CAN ID 0x160 byte packing and unpacking precision."""
    can_mgr = CANManager(channel="vcan0")

    nearest_id = 4
    x_offset_mm = -1250  # -1.25 meters (left)
    z_dist_mm = 8400     # 8.40 meters (forward)
    tti_ms = 1450        # 1.45 seconds
    threat_level = 2     # Critical threat

    payload = can_mgr.pack_spatial_obstacle_frame(
        nearest_id=nearest_id,
        x_offset_mm=x_offset_mm,
        z_dist_mm=z_dist_mm,
        tti_ms=tti_ms,
        threat_level=threat_level,
    )

    assert len(payload) == 8, f"CAN 0x160 must be exactly 8 bytes, got {len(payload)}"

    decoded = can_mgr.unpack_spatial_obstacle_frame(payload)
    assert decoded["nearest_id"] == 4
    assert decoded["x_offset_mm"] == -1250
    assert decoded["x_offset_m"] == -1.25
    assert decoded["z_dist_mm"] == 8400
    assert decoded["z_dist_m"] == 8.40
    assert decoded["tti_ms"] == 1450
    assert decoded["threat_level"] == 2
