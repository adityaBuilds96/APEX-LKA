"""
src/perception/occupancy_grid.py
================================
High-Speed 2D/3D Top-Down Voxel Occupancy Grid Rasterizer for APEX-LKA:
- Discretizes 3D space ahead of the vehicle into a 10 cm x 10 cm top-down BEV metric grid.
- Grid Bounds: Lateral X in [-5.0 m, +5.0 m] (100 cells), Forward Z in [0.0 m, +15.0 m] (150 cells).
- Cell State Enum:
    0 = FREE (drivable roadway corridor)
    1 = OCCUPIED_STATIC_ROAD (lane boundaries, curbs, road limits)
    2 = OCCUPIED_DYNAMIC_OBJECT (vehicles, pedestrians, cyclists, obstacles)
    3 = UNKNOWN_UNOBSERVED (shadow-cast occluded regions and unobserved space)
- Multi-Source Sensor Fusion:
  1. Rasterizes free-space drivable corridor as FREE.
  2. Projects all 3D bounding boxes as OCCUPIED_DYNAMIC_OBJECT.
  3. Applies radial ray-tracing shadow casting from vehicle origin (0, 0) to mark
     blind spots behind obstacles as UNKNOWN_UNOBSERVED.
"""

from __future__ import annotations

from enum import IntEnum
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.perception.projection_3d import Object3D

log = logging.getLogger("apex_occupancy_grid")


class CellState(IntEnum):
    """Occupancy grid cell states."""
    FREE                    = 0  # Confirmed drivable free space
    OCCUPIED_STATIC_ROAD    = 1  # Static curb, road boundary, or barrier
    OCCUPIED_DYNAMIC_OBJECT = 2  # Dynamic obstacle (vehicle, pedestrian, etc.)
    UNKNOWN_UNOBSERVED      = 3  # Occluded by obstacle or outside sensor FOV


@dataclass
class OccupancyGrid2D:
    """
    2D Top-Down Metric Occupancy Grid representation.
    Shape: (num_rows_z, num_cols_x) = (150, 100)
    """
    grid: np.ndarray = field(default_factory=lambda: np.full((150, 100), CellState.UNKNOWN_UNOBSERVED, dtype=np.uint8))
    x_min_m: float = -5.0
    x_max_m: float = 5.0
    z_min_m: float = 0.0
    z_max_m: float = 15.0
    resolution_m: float = 0.10
    rasterize_ms: float = 0.0

    @property
    def shape(self) -> Tuple[int, int]:
        return self.grid.shape

    @property
    def num_rows_z(self) -> int:
        return self.grid.shape[0]

    @property
    def num_cols_x(self) -> int:
        return self.grid.shape[1]

    def world_to_grid(self, x_m: float, z_m: float) -> Tuple[int, int]:
        """Convert real-world metric coordinates (meters) to grid (row_z, col_x)."""
        col = int(math.floor((x_m - self.x_min_m) / self.resolution_m))
        row = int(math.floor((z_m - self.z_min_m) / self.resolution_m))
        return row, col

    def grid_to_world(self, row_z: int, col_x: int) -> Tuple[float, float]:
        """Convert grid cell (row_z, col_x) to real-world metric coordinates (meters)."""
        x_m = self.x_min_m + (col_x + 0.5) * self.resolution_m
        z_m = self.z_min_m + (row_z + 0.5) * self.resolution_m
        return x_m, z_m

    def is_in_bounds(self, row_z: int, col_x: int) -> bool:
        """Check if grid index is within rasterizer dimensions."""
        return 0 <= row_z < self.num_rows_z and 0 <= col_x < self.num_cols_x

    def get_cell(self, x_m: float, z_m: float) -> int:
        """Return state of cell at real-world coordinates (x_m, z_m)."""
        row, col = self.world_to_grid(x_m, z_m)
        if self.is_in_bounds(row, col):
            return int(self.grid[row, col])
        return CellState.UNKNOWN_UNOBSERVED

    def to_rgb(self) -> np.ndarray:
        """
        Renders the occupancy grid as an RGB visualization image.
        Palette:
          FREE: Dark green (25, 60, 30)
          OCCUPIED_STATIC_ROAD: Slate gray (140, 150, 160)
          OCCUPIED_DYNAMIC_OBJECT: Bright red (235, 45, 45)
          UNKNOWN_UNOBSERVED: Dark background (14, 18, 24)
        """
        h, w = self.grid.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Color lookup table
        lut = {
            CellState.FREE: (25, 60, 30),
            CellState.OCCUPIED_STATIC_ROAD: (140, 150, 160),
            CellState.OCCUPIED_DYNAMIC_OBJECT: (235, 45, 45),
            CellState.UNKNOWN_UNOBSERVED: (14, 18, 24),
        }

        for state, color in lut.items():
            rgb[self.grid == state] = color

        return rgb


class OccupancyGridRasterizer:
    """
    Constructs top-down metric occupancy grids by fusing lane drivable corridors,
    3D object bounding boxes, and shadow-casting ray tracing.
    """

    def __init__(
        self,
        grid_resolution_m: float = 0.10,
        x_min_m: float = -5.0,
        x_max_m: float = 5.0,
        z_min_m: float = 0.0,
        z_max_m: float = 15.0,
        raycast_shadows: bool = True,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 180.0,
        camera_height_m: float = 1.20,
        pitch_rad: float = 0.08726646,
    ) -> None:
        self.resolution = float(grid_resolution_m)
        self.x_min_m = float(x_min_m)
        self.x_max_m = float(x_max_m)
        self.z_min_m = float(z_min_m)
        self.z_max_m = float(z_max_m)
        self.raycast_shadows = bool(raycast_shadows)

        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float(cx)
        self.cy = float(cy)
        self.camera_height_m = float(camera_height_m)
        self.pitch_rad = float(pitch_rad)

        self.num_cols_x = int(round((self.x_max_m - self.x_min_m) / self.resolution))
        self.num_rows_z = int(round((self.z_max_m - self.z_min_m) / self.resolution))

        # Ego vehicle position in grid coordinates (bottom-center)
        self.ego_col = int(round((0.0 - self.x_min_m) / self.resolution))
        self.ego_row = 0

    def rasterize(
        self,
        objects_3d: Optional[List[Object3D]] = None,
        drivable_corridor_poly: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int] = (360, 640),
    ) -> OccupancyGrid2D:
        """
        Fuses corridor and 3D detections into a top-down occupancy grid.

        Parameters:
            objects_3d: List of 3D objects with metric spatial coordinates.
            drivable_corridor_poly: 2D image-space polygon of the free-space road corridor.
            frame_shape: (height, width) of the camera frame.

        Returns:
            OccupancyGrid2D dataclass.
        """
        t0 = time.perf_counter()

        # Initialize grid with UNKNOWN_UNOBSERVED
        grid = np.full((self.num_rows_z, self.num_cols_x), CellState.UNKNOWN_UNOBSERVED, dtype=np.uint8)

        # ── 1. Rasterize Drivable Corridor as FREE ───────────────────────────
        self._rasterize_corridor(grid, drivable_corridor_poly, frame_shape)

        # ── 2. Project 3D Objects as OCCUPIED_DYNAMIC_OBJECT ─────────────────
        occupied_boxes: List[Tuple[int, int, int, int]] = []
        if objects_3d:
            for obj in objects_3d:
                box_bounds = self._rasterize_object_3d(grid, obj)
                if box_bounds is not None:
                    occupied_boxes.append(box_bounds)

        # ── 3. Apply Radial Ray-Tracing Shadow Casting ───────────────────────
        if self.raycast_shadows and occupied_boxes:
            self._apply_shadow_casting(grid, occupied_boxes)

        rasterize_ms = (time.perf_counter() - t0) * 1000.0
        return OccupancyGrid2D(
            grid=grid,
            x_min_m=self.x_min_m,
            x_max_m=self.x_max_m,
            z_min_m=self.z_min_m,
            z_max_m=self.z_max_m,
            resolution_m=self.resolution,
            rasterize_ms=round(rasterize_ms, 2),
        )

    def _rasterize_corridor(
        self,
        grid: np.ndarray,
        drivable_corridor_poly: Optional[np.ndarray],
        frame_shape: Tuple[int, int],
    ) -> None:
        """Projects image-space corridor polygon to BEV grid and marks as FREE."""
        h_img, w_img = frame_shape

        if drivable_corridor_poly is not None and len(drivable_corridor_poly) >= 3:
            poly = np.asarray(drivable_corridor_poly, dtype=np.float32)
            if poly.ndim == 3 and poly.shape[1] == 1:
                poly = poly[:, 0, :]

            # Project image points (u, v) to ground plane (X_m, Z_m)
            y_horizon = self.cy - self.fy * math.tan(self.pitch_rad)
            bev_pts = []

            for u, v in poly:
                dy = max(1.0, v - y_horizon)
                z_m = (self.fy * self.camera_height_m) / dy
                x_m = ((u - self.cx) * z_m) / self.fx

                col = int(math.floor((x_m - self.x_min_m) / self.resolution))
                row = int(math.floor((z_m - self.z_min_m) / self.resolution))
                bev_pts.append([col, row])

            if len(bev_pts) >= 3:
                bev_pts_arr = np.array([bev_pts], dtype=np.int32)
                cv2.fillPoly(grid, bev_pts_arr, int(CellState.FREE))
                return

        # Default fallback forward corridor: lane width ~3.4m (+/- 1.7m) up to 15m
        c_left = int(math.floor((-1.70 - self.x_min_m) / self.resolution))
        c_right = int(math.floor((1.70 - self.x_min_m) / self.resolution))
        c_left = max(0, min(self.num_cols_x - 1, c_left))
        c_right = max(c_left + 1, min(self.num_cols_x, c_right))
        grid[:, c_left:c_right] = CellState.FREE

    def _rasterize_object_3d(
        self,
        grid: np.ndarray,
        obj: Object3D,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Rasterize a single 3D object footprint onto the grid as OCCUPIED_DYNAMIC_OBJECT."""
        half_w = max(0.2, obj.width / 2.0)
        half_l = max(0.2, obj.length / 2.0)

        x1_m = obj.x - half_w
        x2_m = obj.x + half_w
        z1_m = obj.z - half_l
        z2_m = obj.z + half_l

        # Out of bounds check
        if x2_m < self.x_min_m or x1_m > self.x_max_m or z2_m < self.z_min_m or z1_m > self.z_max_m:
            return None

        c1 = int(math.floor((x1_m - self.x_min_m) / self.resolution))
        c2 = int(math.ceil((x2_m - self.x_min_m) / self.resolution))
        r1 = int(math.floor((z1_m - self.z_min_m) / self.resolution))
        r2 = int(math.ceil((z2_m - self.z_min_m) / self.resolution))

        c1 = max(0, min(self.num_cols_x - 1, c1))
        c2 = max(c1 + 1, min(self.num_cols_x, c2))
        r1 = max(0, min(self.num_rows_z - 1, r1))
        r2 = max(r1 + 1, min(self.num_rows_z, r2))

        grid[r1:r2, c1:c2] = CellState.OCCUPIED_DYNAMIC_OBJECT
        return (r1, r2, c1, c2)

    def _apply_shadow_casting(
        self,
        grid: np.ndarray,
        occupied_boxes: List[Tuple[int, int, int, int]],
    ) -> None:
        """
        Radial shadow casting ray tracing:
        Casts occlusion shadows behind each occupied obstacle along rays from the ego origin (0, 0).
        """
        for r1, r2, c1, c2 in occupied_boxes:
            # For every column subtended by the obstacle, cast a shadow for all rows r >= r2
            for c in range(c1, c2):
                if 0 <= c < self.num_cols_x:
                    for r in range(r2, self.num_rows_z):
                        # Only mark as UNKNOWN if not already occupied by another obstacle
                        if grid[r, c] != CellState.OCCUPIED_DYNAMIC_OBJECT:
                            grid[r, c] = CellState.UNKNOWN_UNOBSERVED
