"""
tests/test_occupancy_grid.py
============================
Unit test suite for OccupancyGridRasterizer:
- Validates grid dimensions: 100 cols (lateral X) x 150 rows (forward Z)
- Tests world-to-grid coordinate transforms
- Validates corridor rasterization as FREE (0)
- Validates 3D object projection as OCCUPIED_DYNAMIC_OBJECT (2)
- Tests radial shadow-casting occlusion behind obstacles
- Tests RGB visualization generation
"""

import numpy as np
import pytest

from src.perception.occupancy_grid import (
    CellState,
    OccupancyGrid2D,
    OccupancyGridRasterizer,
)
from src.perception.projection_3d import Object3D


def test_grid_initialization_and_shape():
    rasterizer = OccupancyGridRasterizer(
        grid_resolution_m=0.10,
        x_min_m=-5.0,
        x_max_m=5.0,
        z_min_m=0.0,
        z_max_m=15.0,
    )
    assert rasterizer.num_cols_x == 100
    assert rasterizer.num_rows_z == 150

    grid_res = rasterizer.rasterize(objects_3d=[])
    assert isinstance(grid_res, OccupancyGrid2D)
    assert grid_res.shape == (150, 100)
    assert grid_res.resolution_m == 0.10


def test_coordinate_transforms():
    grid = OccupancyGrid2D(
        grid=np.zeros((150, 100), dtype=np.uint8),
        x_min_m=-5.0,
        x_max_m=5.0,
        z_min_m=0.0,
        z_max_m=15.0,
        resolution_m=0.10,
    )

    # Origin (0.0m, 0.0m) should be at col 50, row 0
    row, col = grid.world_to_grid(0.0, 0.0)
    assert col == 50
    assert row == 0

    # Test roundtrip
    x_rec, z_rec = grid.grid_to_world(row, col)
    assert np.isclose(x_rec, 0.05, atol=0.05)
    assert np.isclose(z_rec, 0.05, atol=0.05)


def test_corridor_rasterization_free_space():
    rasterizer = OccupancyGridRasterizer()
    grid_res = rasterizer.rasterize(objects_3d=[], drivable_corridor_poly=None)

    # By default, central lane |X| <= 1.5m should be FREE (0)
    assert grid_res.get_cell(0.0, 5.0) == CellState.FREE
    assert grid_res.get_cell(1.0, 5.0) == CellState.FREE
    assert grid_res.get_cell(-1.0, 5.0) == CellState.FREE

    # Extreme lateral edges |X| >= 4.0m should be UNKNOWN (3)
    assert grid_res.get_cell(4.5, 5.0) == CellState.UNKNOWN_UNOBSERVED
    assert grid_res.get_cell(-4.5, 5.0) == CellState.UNKNOWN_UNOBSERVED


def test_object_3d_rasterization():
    rasterizer = OccupancyGridRasterizer(raycast_shadows=False)

    # Object at X=0.0m, Z=6.0m with width=1.8m, length=4.0m
    obj = Object3D(
        x=0.0,
        y=0.0,
        z=6.0,
        width=1.8,
        height=1.5,
        length=4.0,
        class_id=0,
        class_name="vehicle",
        confidence=0.9,
        box_2d=np.array([200, 100, 440, 260], dtype=np.float32),
    )

    grid_res = rasterizer.rasterize(objects_3d=[obj])

    # Object center should be OCCUPIED_DYNAMIC_OBJECT (2)
    assert grid_res.get_cell(0.0, 6.0) == CellState.OCCUPIED_DYNAMIC_OBJECT
    assert grid_res.get_cell(0.5, 6.0) == CellState.OCCUPIED_DYNAMIC_OBJECT
    assert grid_res.get_cell(-0.5, 6.0) == CellState.OCCUPIED_DYNAMIC_OBJECT

    # Ahead of object (e.g. Z=1.0m) should be FREE (0)
    assert grid_res.get_cell(0.0, 1.0) == CellState.FREE


def test_raycast_shadow_occlusion():
    rasterizer = OccupancyGridRasterizer(raycast_shadows=True)

    # Obstacle at Z=5.0m, length=2.0m (occupies Z in [4.0, 6.0])
    obj = Object3D(
        x=0.0,
        y=0.0,
        z=5.0,
        width=1.0,
        height=1.5,
        length=2.0,
        class_id=0,
        class_name="vehicle",
        confidence=0.9,
        box_2d=np.array([250, 120, 390, 240], dtype=np.float32),
    )

    grid_res = rasterizer.rasterize(objects_3d=[obj])

    # 1. Front of obstacle is FREE
    assert grid_res.get_cell(0.0, 2.0) == CellState.FREE

    # 2. Obstacle itself is OCCUPIED_DYNAMIC_OBJECT
    assert grid_res.get_cell(0.0, 5.0) == CellState.OCCUPIED_DYNAMIC_OBJECT

    # 3. Behind obstacle (Z=10.0m) along the same column should be occluded (UNKNOWN_UNOBSERVED)
    assert grid_res.get_cell(0.0, 10.0) == CellState.UNKNOWN_UNOBSERVED


def test_rgb_visualization():
    rasterizer = OccupancyGridRasterizer()
    grid_res = rasterizer.rasterize()
    rgb = grid_res.to_rgb()

    assert rgb.shape == (150, 100, 3)
    assert rgb.dtype == np.uint8
    assert np.any(rgb > 0)
