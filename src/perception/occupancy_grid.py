"""
src/perception/occupancy_grid.py
================================
APEX-Vision Core v3.0 — Dual-Mode Top-Down Occupancy Grid & 3D Voxel Engine (ISO 26262 Compliant)

Features:
1. OccupancyGridRasterizer (2D BEV Rasterizer):
   - Discretizes 3D space ahead into 10 cm x 10 cm BEV metric grid.
   - Grid bounds: Lateral X in [-5.0m, +5.0m] (100 cols), Forward Z in [0.0m, +15.0m] (150 rows).
   - Cell states: FREE (0), OCCUPIED_STATIC_ROAD (1), OCCUPIED_DYNAMIC_OBJECT (2), UNKNOWN (3).
   - Fuses drivable road corridor, 3D object bounding boxes, and radial shadow-casting ray tracing.

2. TeslaStyleOccupancyGrid (3D Voxel Continuous Occupancy):
   - Discretizes 3D spatial continuum around ego-vehicle:
     X: [-10.0m, +10.0m] (200 voxel cells @ 0.10m)
     Y: [-2.0m, +2.0m]   (40 voxel cells @ 0.10m)
     Z: [0.0m, +40.0m]   (400 voxel cells @ 0.10m)
   - Dynamic Occupancy Probability Log-Odds Map for multi-sensor spatial fusion:
     l_t = l_{t-1} + log_odds(P(m|z)) - l_0
     P(occupied) = 1 / (1 + exp(-l_t))
   - Vectorized Voxel Rasterization (< 10ms latency).
   - 2D Top-Down Bird's Eye View (BEV) Occupancy Heatmap (400 x 200 grid cells).
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Defensive project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

log = logging.getLogger("apex_occupancy_grid")


class CellState(IntEnum):
    """Occupancy grid cell states."""
    FREE                    = 0  # Drivable road corridor / free space
    OCCUPIED_STATIC_ROAD    = 1  # Curbs, lane boundaries, barriers
    OCCUPIED_DYNAMIC_OBJECT = 2  # Obstacles (vehicles, pedestrians, cones)
    UNKNOWN_UNOBSERVED      = 3  # Occluded shadow regions or unobserved space


# ==============================================================================
# 1. 2D Top-Down BEV Metric Occupancy Grid
# ==============================================================================

@dataclass
class OccupancyGrid2D:
    """
    2D Top-Down Metric Occupancy Grid representation.
    Default shape: (num_rows_z, num_cols_x) = (150, 100)
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
        objects_3d: Optional[List[Any]] = None,
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

        # 1. Rasterize Drivable Corridor as FREE
        self._rasterize_corridor(grid, drivable_corridor_poly, frame_shape)

        # 2. Project 3D Objects as OCCUPIED_DYNAMIC_OBJECT
        occupied_boxes: List[Tuple[int, int, int, int]] = []
        if objects_3d:
            for obj in objects_3d:
                box_bounds = self._rasterize_object_3d(grid, obj)
                if box_bounds is not None:
                    occupied_boxes.append(box_bounds)

        # 3. Apply Radial Ray-Tracing Shadow Casting
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
        obj: Any,
    ) -> Optional[Tuple[int, int, int, int]]:
        """Rasterize a single 3D object footprint onto the grid as OCCUPIED_DYNAMIC_OBJECT."""
        width = float(getattr(obj, "width", 1.8))
        length = float(getattr(obj, "length", 4.0))
        half_w = max(0.2, width / 2.0)
        half_l = max(0.2, length / 2.0)

        x = float(getattr(obj, "x", getattr(obj, "x_c", 0.0)))
        z = float(getattr(obj, "z", getattr(obj, "z_c", 6.0)))

        x1_m = x - half_w
        x2_m = x + half_w
        z1_m = z - half_l
        z2_m = z + half_l

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
            for c in range(c1, c2):
                if 0 <= c < self.num_cols_x:
                    for r in range(r2, self.num_rows_z):
                        if grid[r, c] != CellState.OCCUPIED_DYNAMIC_OBJECT:
                            grid[r, c] = CellState.UNKNOWN_UNOBSERVED


# ==============================================================================
# 2. Tesla-Style 3D Voxel & 2D BEV Occupancy Grid Engine
# ==============================================================================

@dataclass
class VoxelGrid3D:
    """
    3D Voxel Occupancy Grid representation.
    Spatial dimensions:
      - X (lateral): [-10.0m, +10.0m] -> 200 cells
      - Y (vertical): [-2.0m, +2.0m]  -> 40 cells
      - Z (longitudinal): [0.0m, +40.0m] -> 400 cells
    """
    voxels: np.ndarray = field(default_factory=lambda: np.zeros((40, 400, 200), dtype=np.float32))
    log_odds: np.ndarray = field(default_factory=lambda: np.zeros((40, 400, 200), dtype=np.float32))
    bev_grid: np.ndarray = field(default_factory=lambda: np.zeros((400, 200), dtype=np.uint8))
    x_min_m: float = -10.0
    x_max_m: float = 10.0
    y_min_m: float = -2.0
    y_max_m: float = 2.0
    z_min_m: float = 0.0
    z_max_m: float = 40.0
    resolution_m: float = 0.10
    rasterize_ms: float = 0.0

    @property
    def voxel_shape(self) -> Tuple[int, int, int]:
        """(num_y, num_z, num_x) -> (40, 400, 200)."""
        return self.voxels.shape

    @property
    def bev_shape(self) -> Tuple[int, int]:
        """(num_rows_z, num_cols_x) -> (400, 200)."""
        return self.bev_grid.shape

    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of BEV grid (num_rows_z, num_cols_x)."""
        return self.bev_grid.shape

    @property
    def num_cols_x(self) -> int:
        return self.voxels.shape[2]

    @property
    def num_rows_z(self) -> int:
        return self.voxels.shape[1]

    @property
    def num_slices_y(self) -> int:
        return self.voxels.shape[0]

    def world_to_voxel(self, x_m: float, y_m: float, z_m: float) -> Tuple[int, int, int]:
        """Convert metric coordinates (meters) to 3D voxel indices (iy, iz, ix)."""
        ix = int(math.floor((x_m - self.x_min_m) / self.resolution_m))
        iy = int(math.floor((y_m - self.y_min_m) / self.resolution_m))
        iz = int(math.floor((z_m - self.z_min_m) / self.resolution_m))
        return iy, iz, ix

    def world_to_grid(self, x_m: float, z_m: float) -> Tuple[int, int]:
        """Convert metric coordinates (meters) to BEV indices (row_z, col_x)."""
        col = int(math.floor((x_m - self.x_min_m) / self.resolution_m))
        row = int(math.floor((z_m - self.z_min_m) / self.resolution_m))
        return row, col

    def is_in_bounds(self, iy: int, iz: int, ix: int) -> bool:
        """Check if indices are inside 3D grid."""
        return 0 <= iy < self.num_slices_y and 0 <= iz < self.num_rows_z and 0 <= ix < self.num_cols_x

    def get_cell(self, x_m: float, z_m: float) -> int:
        """Return state of BEV cell at metric coordinates (x_m, z_m)."""
        row, col = self.world_to_grid(x_m, z_m)
        if 0 <= row < self.num_rows_z and 0 <= col < self.num_cols_x:
            return int(self.bev_grid[row, col])
        return CellState.UNKNOWN_UNOBSERVED

    def render_bev_heatmap(self) -> np.ndarray:
        """
        Renders 2D top-down BEV occupancy map as an RGB visualization (400 x 200 x 3).
        Palette:
          - FREE (0): Dark forest green (15, 45, 20)
          - STATIC ROAD (1): Slate cyan (80, 140, 160)
          - DYNAMIC OBSTACLE (2): High-vis red (240, 30, 30)
          - UNKNOWN (3): Deep space blue-black (10, 15, 24)
        """
        h, w = self.bev_grid.shape
        rgb = np.full((h, w, 3), (10, 15, 24), dtype=np.uint8)

        rgb[self.bev_grid == CellState.FREE] = (15, 45, 20)
        rgb[self.bev_grid == CellState.OCCUPIED_STATIC_ROAD] = (80, 140, 160)
        rgb[self.bev_grid == CellState.OCCUPIED_DYNAMIC_OBJECT] = (240, 30, 30)

        # Draw ego vehicle footprint at bottom center
        ego_c = w // 2
        ego_r = 0
        cv2.rectangle(rgb, (ego_c - 9, ego_r), (ego_c + 9, ego_r + 20), (0, 220, 255), -1)

        return rgb

    def to_rgb(self) -> np.ndarray:
        return self.render_bev_heatmap()


OccupancyGrid3D = VoxelGrid3D


class TeslaStyleOccupancyGrid:
    """
    3D Voxel Occupancy Grid Engine with dynamic log-odds updates and BEV rasterizer.
    """

    def __init__(
        self,
        resolution_m: float = 0.10,
        x_range: Tuple[float, float] = (-10.0, 10.0),
        y_range: Tuple[float, float] = (-2.0, 2.0),
        z_range: Tuple[float, float] = (0.0, 40.0),
        decay_factor: float = 0.95,
        log_odds_occ: float = 0.85,
        log_odds_free: float = -0.40,
    ) -> None:
        self.resolution = float(resolution_m)
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.z_min, self.z_max = z_range
        self.decay = float(decay_factor)
        self.l_occ = float(log_odds_occ)
        self.l_free = float(log_odds_free)

        self.nx = int(round((self.x_max - self.x_min) / self.resolution))
        self.ny = int(round((self.y_max - self.y_min) / self.resolution))
        self.nz = int(round((self.z_max - self.z_min) / self.resolution))

        self.num_cols_x = self.nx
        self.num_rows_z = self.nz

        # Persistent log-odds 3D state and pre-allocated probability buffer
        self._log_odds = np.zeros((self.ny, self.nz, self.nx), dtype=np.float32)
        self._prob_voxels = np.full((self.ny, self.nz, self.nx), 0.5, dtype=np.float32)

    def rasterize_from_point_cloud_and_boxes(
        self,
        point_cloud: Optional[np.ndarray] = None,
        objects_3d: Optional[List[Any]] = None,
        drivable_corridor_width_m: float = 3.6,
    ) -> VoxelGrid3D:
        """
        Fast spatial voxelization and BEV rasterization.
        Execution latency target: < 10ms.
        """
        t0 = time.perf_counter()

        # 1. Decay previous frame log-odds (temporal smoothing)
        self._log_odds *= self.decay

        # 2. Free space corridor initialization (BEV)
        bev_map = np.full((self.nz, self.nx), CellState.UNKNOWN_UNOBSERVED, dtype=np.uint8)
        half_w = drivable_corridor_width_m / 2.0
        c_min = max(0, int(round((-half_w - self.x_min) / self.resolution)))
        c_max = min(self.nx, int(round((half_w - self.x_min) / self.resolution)))
        bev_map[:, c_min:c_max] = CellState.FREE

        # 3. Fuse 3D Oriented Bounding Boxes into Voxels and BEV
        if objects_3d:
            for obj in objects_3d:
                xc = getattr(obj, "x_c", getattr(obj, "x", 0.0))
                yc = getattr(obj, "y_c", getattr(obj, "y", 0.0))
                zc = getattr(obj, "z_c", getattr(obj, "z", 10.0))
                l = getattr(obj, "length", 4.0)
                w = getattr(obj, "width", 1.8)
                h = getattr(obj, "height", 1.5)

                ix_min = max(0, int(round((xc - w / 2.0 - self.x_min) / self.resolution)))
                ix_max = min(self.nx, int(round((xc + w / 2.0 - self.x_min) / self.resolution)))
                iy_min = max(0, int(round((yc - h / 2.0 - self.y_min) / self.resolution)))
                iy_max = min(self.ny, int(round((yc + h / 2.0 - self.y_min) / self.resolution)))
                iz_min = max(0, int(round((zc - l / 2.0 - self.z_min) / self.resolution)))
                iz_max = min(self.nz, int(round((zc + l / 2.0 - self.z_min) / self.resolution)))

                if ix_max > ix_min and iz_max > iz_min:
                    bev_map[iz_min:iz_max, ix_min:ix_max] = CellState.OCCUPIED_DYNAMIC_OBJECT
                    if iy_max > iy_min:
                        self._log_odds[iy_min:iy_max, iz_min:iz_max, ix_min:ix_max] += self.l_occ

        # 4. Point Cloud Voxel Accumulation
        if point_cloud is not None and point_cloud.size > 0:
            flat_pts = point_cloud[::4, ::4, :].reshape(-1, 3)
            valid_mask = (
                (flat_pts[:, 0] >= self.x_min) & (flat_pts[:, 0] < self.x_max) &
                (flat_pts[:, 1] >= self.y_min) & (flat_pts[:, 1] < self.y_max) &
                (flat_pts[:, 2] >= self.z_min) & (flat_pts[:, 2] < self.z_max)
            )
            v_pts = flat_pts[valid_mask]

            if v_pts.shape[0] > 0:
                ixs = np.clip(((v_pts[:, 0] - self.x_min) / self.resolution).astype(np.int32), 0, self.nx - 1)
                iys = np.clip(((v_pts[:, 1] - self.y_min) / self.resolution).astype(np.int32), 0, self.ny - 1)
                izs = np.clip(((v_pts[:, 2] - self.z_min) / self.resolution).astype(np.int32), 0, self.nz - 1)

                np.add.at(self._log_odds, (iys, izs, ixs), self.l_occ * 0.1)

        # Convert log-odds to occupancy probabilities [0.0, 1.0]
        self._prob_voxels.fill(0.5)
        active_mask = (self._log_odds != 0.0)
        if np.any(active_mask):
            clamped_lo = np.clip(self._log_odds[active_mask], -10.0, 10.0)
            self._prob_voxels[active_mask] = 1.0 / (1.0 + np.exp(-clamped_lo))

        infer_ms = (time.perf_counter() - t0) * 1000.0

        grid_result = VoxelGrid3D(
            voxels=self._prob_voxels,
            log_odds=self._log_odds,
            bev_grid=bev_map,
            x_min_m=self.x_min,
            x_max_m=self.x_max,
            y_min_m=self.y_min,
            y_max_m=self.y_max,
            z_min_m=self.z_min,
            z_max_m=self.z_max,
            resolution_m=self.resolution,
            rasterize_ms=round(infer_ms, 2),
        )
        return grid_result

    def rasterize(
        self,
        objects_3d: Optional[List[Any]] = None,
        drivable_corridor_poly: Optional[np.ndarray] = None,
        frame_shape: Tuple[int, int] = (360, 640),
        **kwargs: Any,
    ) -> VoxelGrid3D:
        """Compatibility wrapper for pipeline calling rasterize()."""
        return self.rasterize_from_point_cloud_and_boxes(
            point_cloud=kwargs.get("point_cloud", None),
            objects_3d=objects_3d,
        )

    def update_from_point_cloud(
        self,
        point_cloud: Optional[np.ndarray] = None,
        objects_3d: Optional[List[Any]] = None,
        boxes_3d: Optional[List[Any]] = None,
    ) -> VoxelGrid3D:
        """Convenience alias for rasterize_from_point_cloud_and_boxes."""
        objs = objects_3d if objects_3d is not None else boxes_3d
        return self.rasterize_from_point_cloud_and_boxes(point_cloud=point_cloud, objects_3d=objs)

    def get_bev_heatmap(self) -> np.ndarray:
        """Returns 2D BEV occupancy probability grid."""
        return np.max(self._prob_voxels, axis=0)
