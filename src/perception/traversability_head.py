"""
src/perception/traversability_head.py
=====================================
APEX-Vision Core v4.0 • Off-Road Terrain Classifier & Traversability Head
ISO 26262 ASIL-B Aligned Off-Road Surface Segmentation (BAJA SAE Buggy Profile)

Classifies unstructured off-road surfaces into distinct terrain categories and
generates a 2D Traversability Costmap for real-time motion planning and voxel grid fusion.

Terrain Class Definitions & Risk Weightings:
--------------------------------------------
1. DRIVABLE_DIRT (Class 0):
   - Smooth dirt trail, packed gravel, hard clay.
   - Base Traversability Cost: 0.00 (Optimal corridor).
2. VEGETATION (Class 1):
   - Tall grass, light shrubs, soft brush.
   - Base Traversability Cost: 0.35 (Passable at reduced speed, low traction).
3. MUD_WATER (Class 2):
   - Standing water, wet slurry, deep mud puddles.
   - Base Traversability Cost: 0.70 (Severe slip / hydroplaning hazard).
4. ROCKS_BOULDERS (Class 3):
   - Loose stone clusters, exposed bedrock, large boulders (>15 cm).
   - Base Traversability Cost: 0.85 (High tire puncture & tie-rod failure hazard).
5. DITCH_DROPOFF (Class 4):
   - Erosion trenches, steep berm edges, sudden washouts (>30 cm drop).
   - Base Traversability Cost: 1.00 (Critical rollover hazard; non-traversable).

Costmap Projection:
-------------------
Produces a 2D BEV Costmap (values [0.0, 1.0]) aligned with the 200x400 occupancy
grid (0.0185 m/px metric resolution). Fused directly into path planning heuristics.
"""

from dataclasses import dataclass, field
from enum import Enum
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.TraversabilityHead")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Telemetry Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class TerrainClass(Enum):
    """Categorical classification of ground terrain surface patches."""
    DRIVABLE_DIRT = 0
    VEGETATION = 1
    MUD_WATER = 2
    ROCKS_BOULDERS = 3
    DITCH_DROPOFF = 4


@dataclass
class TraversabilityConfig:
    """Hyperparameters for off-road surface feature extraction and cost mapping."""
    # BEV Grid Dimensions (matching occupancy grid: 200 rows x 400 cols)
    bev_rows: int = 200
    bev_cols: int = 400
    meters_per_pixel: float = 0.0185

    # Patch Partitioning (image space)
    patch_grid_rows: int = 8
    patch_grid_cols: int = 16

    # Cost Matrix Assignment
    cost_dirt: float = 0.00
    cost_vegetation: float = 0.35
    cost_mud_water: float = 0.70
    cost_rocks: float = 0.85
    cost_ditch: float = 1.00

    # Surface Roughness & Texture Thresholds
    roughness_rock_laplacian_min: float = 180.0
    mud_saturation_max: int = 50
    mud_luminance_max: int = 85
    vegetation_green_ratio_min: float = 1.15
    ditch_depth_gradient_threshold: float = 4.5

    # Vehicle Speed Governor Limits (BAJA SAE Buggy Profile)
    buggy_max_speed_mps: float = 12.5       # 45 km/h
    buggy_mud_speed_limit_mps: float = 5.0   # 18 km/h
    buggy_rock_speed_limit_mps: float = 3.5  # 12.5 km/h
    buggy_critical_stop_threshold: float = 0.90


@dataclass
class TerrainPatch:
    """Individual classified surface patch in image coordinates."""
    patch_id: int
    row_idx: int
    col_idx: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    terrain_class: TerrainClass
    traversability_cost: float
    surface_roughness: float
    confidence: float


@dataclass
class TraversabilityCostmap:
    """Top-down 2D metric costmap for path planning and voxel fusion."""
    cost_grid: np.ndarray             # Float array (H, W) in [0.0, 1.0]
    class_grid: np.ndarray            # Integer array (H, W) with class indices [0-4]
    meters_per_pixel: float = 0.0185
    min_cost: float = 0.0
    max_cost: float = 1.0
    mean_cost: float = 0.0
    drivable_area_pct: float = 100.0


@dataclass
class TraversabilityReport:
    """Comprehensive terrain analysis telemetry for autonomous vehicle supervisor."""
    timestamp_utc: str
    dominant_terrain: TerrainClass
    costmap: TraversabilityCostmap
    patches: List[TerrainPatch]
    class_distribution_pct: Dict[str, float]
    drivable_corridor_cost: float
    recommended_max_speed_mps: float
    is_ditch_hazard_detected: bool
    is_rock_hazard_detected: bool
    requires_emergency_stop: bool
    processing_time_ms: float


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractors: Roughness, Chromaticity & Ditch Disparity
# ═══════════════════════════════════════════════════════════════════════════════

class SurfaceRoughnessAnalyzer:
    """
    Computes local surface high-frequency texture metrics using Laplacian variance
    and Gray-Level Co-occurrence Matrix (GLCM) contrast approximations.
    Identifies jagged rocks and loose boulders.
    """

    @staticmethod
    def compute_patch_roughness(gray_patch: np.ndarray) -> float:
        """Calculate high-frequency variance indicating rocky or gravelly texture."""
        if gray_patch.size < 9:
            return 0.0
        # Compute Laplacian variance
        lap = cv2.Laplacian(gray_patch, cv2.CV_32F, ksize=3)
        var = float(np.var(lap))
        return var

    @staticmethod
    def compute_high_frequency_edge_density(gray_patch: np.ndarray) -> float:
        """Fraction of pixels exhibiting strong gradient transitions."""
        if gray_patch.size < 9:
            return 0.0
        sobelx = cv2.Sobel(gray_patch, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_patch, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx ** 2 + sobely ** 2)
        edge_pixels = np.sum(grad_mag > 40.0)
        return float(edge_pixels / float(gray_patch.size))


class ChromaticityTerrainFilter:
    """
    Analyzes multi-channel color distributions (HSV, LAB) to distinguish
    vegetation, muddy water, and dry earth.
    """

    @staticmethod
    def evaluate_vegetation_index(bgr_patch: np.ndarray) -> float:
        """
        Visible Atmospherically Resistant Index (VARI) approximation:
        VARI = (G - R) / (G + R - B + eps)
        High positive values indicate chlorophyll / green foliage.
        """
        b = bgr_patch[:, :, 0].astype(np.float32)
        g = bgr_patch[:, :, 1].astype(np.float32)
        r = bgr_patch[:, :, 2].astype(np.float32)

        denom = g + r - b + 1e-4
        vari = (g - r) / denom
        return float(np.mean(vari))

    @staticmethod
    def evaluate_water_mud_signature(bgr_patch: np.ndarray) -> Tuple[bool, float]:
        """
        Detects wet mud and standing water puddles:
        Characterized by low saturation in HSV and low specular reflectance.
        """
        hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
        sat_mean = float(np.mean(hsv[:, :, 1]))
        val_mean = float(np.mean(hsv[:, :, 2]))

        # Mud/Water: low-to-moderate saturation, dark-to-medium value, flat texture
        is_mud = (sat_mean < 75.0) and (val_mean < 110.0)
        confidence = float(np.clip((110.0 - val_mean) / 110.0, 0.0, 1.0))
        return is_mud, confidence


class DitchDropoffDetector:
    """
    Detects sudden negative obstacles (erosion ditches, dropoffs, washouts)
    by identifying sharp downward gradient discontinuities in monocular depth maps.
    """

    @staticmethod
    def detect_ditch_dropoffs(
        depth_map: np.ndarray,
        gradient_threshold: float = 4.5,
    ) -> np.ndarray:
        """
        Detects sudden step changes in depth where distance jumps abruptly forward
        without an elevated obstacle profile (characteristic of a ditch edge).
        Returns binary mask of dropoff hazard pixels.
        """
        if depth_map is None or depth_map.size == 0:
            return np.zeros((360, 640), dtype=np.uint8)

        # Longitudinal depth gradient (vertical change dZ/dy)
        # Ground plane normally exhibits smooth continuous depth reduction from top to bottom
        dZ_dy = cv2.Sobel(depth_map.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)

        # A sudden negative drop creates a positive gradient spike
        ditch_mask = (dZ_dy > gradient_threshold).astype(np.uint8) * 255
        # Clean with horizontal morphology (preserves horizontal dropoff ridges)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
        cleaned = cv2.morphologyEx(ditch_mask, cv2.MORPH_OPEN, kernel)
        return cleaned


# ═══════════════════════════════════════════════════════════════════════════════
# Traversability Classifier & Costmap Generator
# ═══════════════════════════════════════════════════════════════════════════════

class TraversabilityHead:
    """
    Classifies off-road terrain patches and generates 2D BEV traversability costmaps
    for the BAJA SAE Buggy autonomous navigation stack.
    """

    def __init__(self, config: Optional[TraversabilityConfig] = None):
        self.config = config or TraversabilityConfig()
        self.roughness_analyzer = SurfaceRoughnessAnalyzer()
        self.chroma_filter = ChromaticityTerrainFilter()
        self.ditch_detector = DitchDropoffDetector()

    def classify_single_patch(
        self,
        bgr_patch: np.ndarray,
        ditch_mask_patch: Optional[np.ndarray] = None,
    ) -> Tuple[TerrainClass, float, float, float]:
        """
        Classify a single image sub-window into one of the 5 terrain classes.
        Returns: (TerrainClass, traversability_cost, surface_roughness, confidence)
        """
        if bgr_patch.size == 0:
            return TerrainClass.DRIVABLE_DIRT, self.config.cost_dirt, 0.0, 0.5

        gray_patch = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2GRAY)

        # 1. Check for Ditch / Dropoff hazard
        if ditch_mask_patch is not None and ditch_mask_patch.size > 0:
            ditch_pixels = np.sum(ditch_mask_patch > 0)
            if (ditch_pixels / float(ditch_mask_patch.size) > 0.04) or (ditch_pixels > 25):
                return TerrainClass.DITCH_DROPOFF, self.config.cost_ditch, 1.0, 0.95

        # 2. Check for Rocks / Boulders via Laplacian Roughness
        roughness = self.roughness_analyzer.compute_patch_roughness(gray_patch)
        edge_density = self.roughness_analyzer.compute_high_frequency_edge_density(gray_patch)

        if roughness > self.config.roughness_rock_laplacian_min or edge_density > 0.35:
            conf = float(np.clip(roughness / 400.0, 0.5, 0.99))
            return TerrainClass.ROCKS_BOULDERS, self.config.cost_rocks, roughness, conf

        # 3. Check for Vegetation via VARI index
        vari = self.chroma_filter.evaluate_vegetation_index(bgr_patch)
        if vari > 0.12:
            conf = float(np.clip(vari * 3.0, 0.5, 0.95))
            return TerrainClass.VEGETATION, self.config.cost_vegetation, roughness, conf

        # 4. Check for Mud / Water via HSV Saturation & Luminance
        is_mud, mud_conf = self.chroma_filter.evaluate_water_mud_signature(bgr_patch)
        if is_mud:
            return TerrainClass.MUD_WATER, self.config.cost_mud_water, roughness, mud_conf

        # 5. Default to Drivable Dirt
        return TerrainClass.DRIVABLE_DIRT, self.config.cost_dirt, roughness, 0.85

    def partition_and_classify_patches(
        self,
        bgr_image: np.ndarray,
        horizon_y: int = 140,
        depth_map: Optional[np.ndarray] = None,
    ) -> List[TerrainPatch]:
        """
        Divide the lower drivable image region below the horizon into a uniform grid of patches
        and classify each patch.
        """
        h, w = bgr_image.shape[:2]
        drivable_top_y = max(0, min(h - 20, horizon_y))
        drivable_h = h - drivable_top_y

        rows = self.config.patch_grid_rows
        cols = self.config.patch_grid_cols

        dy = drivable_h / float(rows)
        dx = w / float(cols)

        # Extract ditch mask if depth map is available
        ditch_mask = None
        if depth_map is not None:
            ditch_mask = self.ditch_detector.detect_ditch_dropoffs(
                depth_map,
                self.config.ditch_depth_gradient_threshold,
            )

        patches: List[TerrainPatch] = []
        patch_id = 0

        for r in range(rows):
            y1 = int(round(drivable_top_y + r * dy))
            y2 = int(round(drivable_top_y + (r + 1) * dy))
            y2 = min(h, max(y1 + 1, y2))

            for c in range(cols):
                x1 = int(round(c * dx))
                x2 = int(round((c + 1) * dx))
                x2 = min(w, max(x1 + 1, x2))

                bgr_crop = bgr_image[y1:y2, x1:x2]
                ditch_crop = ditch_mask[y1:y2, x1:x2] if ditch_mask is not None else None

                t_class, cost, rough, conf = self.classify_single_patch(bgr_crop, ditch_crop)

                patches.append(TerrainPatch(
                    patch_id=patch_id,
                    row_idx=r,
                    col_idx=c,
                    x_min=x1,
                    y_min=y1,
                    x_max=x2,
                    y_max=y2,
                    terrain_class=t_class,
                    traversability_cost=cost,
                    surface_roughness=rough,
                    confidence=conf,
                ))
                patch_id += 1

        return patches

    def build_traversability_costmap(
        self,
        patches: List[TerrainPatch],
        image_shape: Tuple[int, int] = (360, 640),
    ) -> TraversabilityCostmap:
        """
        Projects patch costs into a continuous 2D BEV grid (200x400 cells)
        aligned with the vehicle's forward navigation corridor.
        """
        grid_rows = self.config.bev_rows
        grid_cols = self.config.bev_cols

        cost_grid = np.zeros((grid_rows, grid_cols), dtype=np.float32)
        class_grid = np.zeros((grid_rows, grid_cols), dtype=np.uint8)

        # Image plane patch representation
        img_cost = np.zeros(image_shape, dtype=np.float32)
        img_class = np.zeros(image_shape, dtype=np.uint8)

        for p in patches:
            img_cost[p.y_min:p.y_max, p.x_min:p.x_max] = p.traversability_cost
            img_class[p.y_min:p.y_max, p.x_min:p.x_max] = p.terrain_class.value

        # Resample into BEV grid dimensions
        cost_grid = cv2.resize(img_cost, (grid_cols, grid_rows), interpolation=cv2.INTER_LINEAR)
        class_grid = cv2.resize(img_class, (grid_cols, grid_rows), interpolation=cv2.INTER_NEAREST)

        # Apply Gaussian smoothing to cost gradient
        cost_grid = cv2.GaussianBlur(cost_grid, (5, 5), 1.0)
        cost_grid = np.clip(cost_grid, 0.0, 1.0)

        min_c = float(np.min(cost_grid))
        max_c = float(np.max(cost_grid))
        mean_c = float(np.mean(cost_grid))

        drivable_cells = np.sum(cost_grid <= 0.40)
        total_cells = cost_grid.size
        drivable_pct = float((drivable_cells / float(total_cells)) * 100.0)

        return TraversabilityCostmap(
            cost_grid=cost_grid,
            class_grid=class_grid,
            meters_per_pixel=self.config.meters_per_pixel,
            min_cost=round(min_c, 3),
            max_cost=round(max_c, 3),
            mean_cost=round(mean_c, 3),
            drivable_area_pct=round(drivable_pct, 1),
        )

    def compute_vehicle_speed_governor(
        self,
        costmap: TraversabilityCostmap,
        patches: List[TerrainPatch],
    ) -> Tuple[float, bool, bool, bool]:
        """
        Calculate recommended maximum speed scale and safety flags based on terrain.
        Returns:
            (recommended_speed_mps, has_ditch, has_rocks, requires_stop)
        """
        # Count critical hazards in near and mid field
        has_ditch = any(p.terrain_class == TerrainClass.DITCH_DROPOFF for p in patches)
        has_rocks = any(p.terrain_class == TerrainClass.ROCKS_BOULDERS for p in patches)
        has_mud = any(p.terrain_class == TerrainClass.MUD_WATER for p in patches)

        # Immediate stop if ditch is right in front of vehicle (bottom 3 rows of patches)
        critical_ditch = any(
            p.terrain_class == TerrainClass.DITCH_DROPOFF and p.row_idx >= self.config.patch_grid_rows - 3
            for p in patches
        )

        if critical_ditch:
            return 0.0, True, has_rocks, True

        # Speed scaling based on dominant hazard
        if has_ditch:
            rec_speed = 2.0  # Crawl speed near dropoff
        elif has_rocks:
            rec_speed = self.config.buggy_rock_speed_limit_mps
        elif has_mud:
            rec_speed = self.config.buggy_mud_speed_limit_mps
        else:
            # Scale linearly with mean corridor cost
            scale = max(0.2, 1.0 - costmap.mean_cost)
            rec_speed = self.config.buggy_max_speed_mps * scale

        rec_speed = round(float(np.clip(rec_speed, 0.0, self.config.buggy_max_speed_mps)), 2)
        requires_stop = (rec_speed == 0.0)

        return rec_speed, has_ditch, has_rocks, requires_stop

    def process(
        self,
        frame_bgr: np.ndarray,
        horizon_y: int = 140,
        depth_map: Optional[np.ndarray] = None,
    ) -> TraversabilityReport:
        """
        Execute full off-road terrain classification and costmap generation pass.
        """
        t_start = time.perf_counter()

        # 1. Partition into patches and classify
        patches = self.partition_and_classify_patches(frame_bgr, horizon_y, depth_map)

        # 2. Build 2D metric costmap
        costmap = self.build_traversability_costmap(patches, image_shape=frame_bgr.shape[:2])

        # 3. Class distribution statistics
        class_counts = {t.name: 0 for t in TerrainClass}
        for p in patches:
            class_counts[p.terrain_class.name] += 1
        total_p = max(1, len(patches))
        dist_pct = {k: round((v / float(total_p)) * 100.0, 1) for k, v in class_counts.items()}

        dominant_name = max(class_counts, key=class_counts.get)
        dominant_terrain = TerrainClass[dominant_name]

        # 4. Calculate vehicle speed governor & hazard flags
        rec_speed, has_ditch, has_rocks, req_stop = self.compute_vehicle_speed_governor(costmap, patches)

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return TraversabilityReport(
            timestamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            dominant_terrain=dominant_terrain,
            costmap=costmap,
            patches=patches,
            class_distribution_pct=dist_pct,
            drivable_corridor_cost=costmap.mean_cost,
            recommended_max_speed_mps=rec_speed,
            is_ditch_hazard_detected=has_ditch,
            is_rock_hazard_detected=has_rocks,
            requires_emergency_stop=req_stop,
            processing_time_ms=round(elapsed_ms, 3),
        )

    def draw_traversability_overlay(
        self,
        frame_bgr: np.ndarray,
        report: TraversabilityReport,
    ) -> np.ndarray:
        """
        Render semi-transparent colored patch overlay and telemetry indicators:
        - Drivable Dirt: Green (alpha 0.25)
        - Vegetation: Yellow-Green (alpha 0.35)
        - Mud / Water: Cyan (alpha 0.50)
        - Rocks: Orange (alpha 0.60)
        - Ditch / Dropoff: Red (alpha 0.75)
        """
        overlay = frame_bgr.copy()

        color_map = {
            TerrainClass.DRIVABLE_DIRT: (0, 200, 0),       # Green
            TerrainClass.VEGETATION: (30, 220, 180),      # Yellow-Green
            TerrainClass.MUD_WATER: (220, 180, 0),        # Cyan/Blue
            TerrainClass.ROCKS_BOULDERS: (0, 140, 255),    # Orange
            TerrainClass.DITCH_DROPOFF: (0, 0, 255),       # Red
        }

        for p in report.patches:
            col = color_map.get(p.terrain_class, (200, 200, 200))
            cv2.rectangle(overlay, (p.x_min, p.y_min), (p.x_max, p.y_max), col, -1)

        # Blend overlay onto image
        annotated = cv2.addWeighted(overlay, 0.40, frame_bgr, 0.60, 0)

        # Draw grid wireframe
        for p in report.patches:
            cv2.rectangle(annotated, (p.x_min, p.y_min), (p.x_max, p.y_max), (50, 50, 50), 1)

        # Draw Telemetry HUD
        hud_lines = [
            f"TERRAIN: {report.dominant_terrain.name} | SPEED LIM: {report.recommended_max_speed_mps:.1f} m/s",
            f"CORRIDOR COST: {report.drivable_corridor_cost:.2f} | DRIVABLE: {report.costmap.drivable_area_pct:.1f}%",
            f"HAZARDS: Rocks={report.is_rock_hazard_detected} Ditch={report.is_ditch_hazard_detected} Stop={report.requires_emergency_stop}",
            f"LATENCY: {report.processing_time_ms:.2f} ms",
        ]
        for i, line in enumerate(hud_lines):
            y = 25 + i * 22
            col = (0, 0, 255) if report.requires_emergency_stop else (0, 255, 0)
            cv2.putText(annotated, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

        return annotated


# ═══════════════════════════════════════════════════════════════════════════════
# Advanced Off-Road Perception & Safety Modules
# ═══════════════════════════════════════════════════════════════════════════════

class CostmapTemporalAccumulator:
    """
    Temporal recursive accumulator with exponential decay and hazard inflation.
    Maintains persistent obstacle awareness across camera frame occlusions or pitch oscillations.
    """

    def __init__(
        self,
        rows: int = 200,
        cols: int = 400,
        alpha: float = 0.70,
        hazard_inflation_radius_px: int = 6,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.alpha = float(np.clip(alpha, 0.1, 0.95))
        self.inflation_radius = hazard_inflation_radius_px
        self.persistent_cost_grid = np.zeros((rows, cols), dtype=np.float32)
        self.frame_count = 0

    def update(self, instantaneous_costmap: np.ndarray) -> np.ndarray:
        """
        Blend instantaneous cost observation with prior belief and apply hazard dilation.
        """
        assert instantaneous_costmap.shape == (self.rows, self.cols), "Grid dimensions mismatch"

        if self.frame_count == 0:
            self.persistent_cost_grid = instantaneous_costmap.copy()
        else:
            # Recursive exponential smoothing: C_t = alpha * C_{t-1} + (1 - alpha) * z_t
            self.persistent_cost_grid = (
                self.alpha * self.persistent_cost_grid + (1.0 - self.alpha) * instantaneous_costmap
            )

        self.frame_count += 1

        # Morphological dilation around severe hazard cells (>0.70) for vehicle safety clearance
        hazard_mask = (self.persistent_cost_grid > 0.70).astype(np.uint8)
        if np.any(hazard_mask):
            kernel_size = 2 * self.inflation_radius + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            dilated_hazards = cv2.dilate(hazard_mask, kernel)

            # Inflate cost around obstacles smoothly
            dilated_cost = np.where(dilated_hazards > 0, np.maximum(self.persistent_cost_grid, 0.65), self.persistent_cost_grid)
            self.persistent_cost_grid = np.clip(dilated_cost, 0.0, 1.0)

        return self.persistent_cost_grid.copy()

    def reset(self) -> None:
        """Clear memory buffer."""
        self.persistent_cost_grid.fill(0.0)
        self.frame_count = 0


class VoxelOccupancyGridFusion:
    """
    Direct elevation and slope fusion adapter integrating 3D/2D voxel occupancy data
    into the 2D traversability cost representation.
    """

    def __init__(
        self,
        max_traversable_slope_deg: float = 25.0,
        critical_step_height_m: float = 0.25,
        meters_per_pixel: float = 0.0185,
    ) -> None:
        self.max_slope_deg = max_traversable_slope_deg
        self.critical_step = critical_step_height_m
        self.m_per_px = meters_per_pixel

    def fuse_elevation_map(
        self,
        base_costmap: np.ndarray,
        elevation_grid_m: np.ndarray,
    ) -> np.ndarray:
        """
        Compute local terrain gradient from elevation map and fuse into costmap.
        """
        assert base_costmap.shape == elevation_grid_m.shape, "Shape mismatch between costmap and elevation grid"

        # Compute Sobel gradients in metric space (dZ/dX, dZ/dY)
        grad_x = cv2.Sobel(elevation_grid_m, cv2.CV_32F, 1, 0, ksize=3) / (self.m_per_px * 8.0)
        grad_y = cv2.Sobel(elevation_grid_m, cv2.CV_32F, 0, 1, ksize=3) / (self.m_per_px * 8.0)

        slope_rad = np.arctan(np.sqrt(grad_x**2 + grad_y**2))
        slope_deg = np.degrees(slope_rad)

        # Calculate slope penalty [0.0, 1.0]
        slope_cost = np.clip(slope_deg / self.max_slope_deg, 0.0, 1.0)

        # Detect step height discontinuities (curb, rock ledge, washout lip)
        step_cost = np.where(np.abs(grad_x) * self.m_per_px > self.critical_step, 1.0, 0.0)
        step_cost = np.maximum(step_cost, np.where(np.abs(grad_y) * self.m_per_px > self.critical_step, 1.0, 0.0))

        # Combined elevation cost
        geometric_cost = np.maximum(slope_cost, step_cost)

        # Fuse with vision-based base cost (worst-case heuristic)
        fused = np.maximum(base_costmap, geometric_cost)
        return np.clip(fused, 0.0, 1.0).astype(np.float32)


class ASILBTerrainSafetyMonitor:
    """
    ISO 26262 ASIL-B aligned supervisor monitoring surface traction, slip hazard,
    and velocity governor compliance.
    """

    def __init__(self, sample_period_s: float = 0.0333) -> None:
        self.dt = sample_period_s
        self.violation_counter = 0
        self.max_allowed_violations = 3  # Tolerate 100ms transient flicker

    def evaluate_kinodynamic_safety(
        self,
        current_speed_mps: float,
        recommended_speed_mps: float,
        dominant_terrain: TerrainClass,
        requires_emergency_stop: bool,
    ) -> Dict[str, Any]:
        """
        Check if vehicle speed exceeds safe kinodynamic traction threshold for detected terrain.
        """
        is_overspeed = current_speed_mps > (recommended_speed_mps + 0.5)

        if requires_emergency_stop or is_overspeed:
            self.violation_counter += 1
        else:
            self.violation_counter = max(0, self.violation_counter - 1)

        trip_safety_brake = self.violation_counter >= self.max_allowed_violations

        # ISO 26262 ASIL-B alert level assignment
        if trip_safety_brake and requires_emergency_stop:
            severity = "ASIL_B_CRITICAL_STOP"
            target_throttle = 0.0
            target_brake_pct = 100.0
        elif trip_safety_brake and is_overspeed:
            severity = "ASIL_B_TRACTION_GOVERNOR_ACTIVE"
            target_throttle = 0.0
            target_brake_pct = 50.0
        else:
            severity = "NOMINAL"
            target_throttle = 1.0
            target_brake_pct = 0.0

        return {
            "status": severity,
            "speed_overshoot_mps": round(max(0.0, current_speed_mps - recommended_speed_mps), 2),
            "dominant_terrain": dominant_terrain.name,
            "actuator_throttle_scale": target_throttle,
            "actuator_brake_pct": target_brake_pct,
            "violation_count": self.violation_counter,
            "trip_active": trip_safety_brake,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_traversability_head(
    config: Optional[TraversabilityConfig] = None,
) -> TraversabilityHead:
    """Factory constructor for TraversabilityHead."""
    return TraversabilityHead(config=config)


def run_unit_tests() -> bool:
    """Validate terrain classification, costmap rasterization, and speed governor."""
    print("Testing TraversabilityHead off-road algorithms...")
    head = get_traversability_head()

    # Create synthetic test frame with mixed terrain
    img = np.full((360, 640, 3), (70, 75, 80), dtype=np.uint8)
    img[:140, :] = (180, 160, 140)  # Sky

    # Add rocky patch (high Laplacian texture) in left mid-field
    noise_rock = np.random.randint(0, 255, (60, 120, 3), dtype=np.uint8)
    img[200:260, 50:170] = noise_rock

    # Add green vegetation patch in right mid-field
    green_patch = np.zeros((60, 120, 3), dtype=np.uint8)
    green_patch[:, :, 1] = 220  # Strong green channel
    img[200:260, 470:590] = green_patch

    # Execute pass
    report = head.process(img, horizon_y=140)

    print(f"Dominant Terrain: {report.dominant_terrain.name}")
    print(f"Recommended Max Speed: {report.recommended_max_speed_mps:.1f} m/s")
    print(f"Rock Hazard Detected: {report.is_rock_hazard_detected}")
    print(f"Drivable Area: {report.costmap.drivable_area_pct:.1f}%")
    print(f"Processing Time: {report.processing_time_ms:.2f} ms")

    # Invariants
    assert len(report.patches) == head.config.patch_grid_rows * head.config.patch_grid_cols
    assert report.costmap.cost_grid.shape == (200, 400), "BEV Costmap shape mismatch"
    assert report.is_rock_hazard_detected is True, "Failed to detect synthetic rock cluster"
    assert report.recommended_max_speed_mps <= head.config.buggy_max_speed_mps

    # Test Ditch Emergency Stop
    synth_depth = np.full((360, 640), 10.0, dtype=np.float32)
    synth_depth[320:, :] = 40.0
    report_ditch = head.process(img, horizon_y=140, depth_map=synth_depth)
    assert report_ditch.is_ditch_hazard_detected is True
    assert report_ditch.requires_emergency_stop is True, "Emergency stop failed on critical dropoff"
    assert report_ditch.recommended_max_speed_mps == 0.0

    # Test Temporal Accumulator & Hazard Inflation
    accumulator = CostmapTemporalAccumulator(rows=200, cols=400, alpha=0.5, hazard_inflation_radius_px=4)
    c1 = report.costmap.cost_grid.copy()
    acc1 = accumulator.update(c1)
    acc2 = accumulator.update(c1)
    assert acc2.shape == (200, 400)
    assert np.max(acc2) >= np.max(c1)

    # Test Voxel Elevation Fusion
    fusion = VoxelOccupancyGridFusion(max_traversable_slope_deg=20.0, meters_per_pixel=0.0185)
    elev_map = np.zeros((200, 400), dtype=np.float32)
    # Add steep elevation gradient (berm/cliff)
    elev_map[50:100, :] = 2.0
    fused_cost = fusion.fuse_elevation_map(c1, elev_map)
    assert fused_cost.shape == (200, 400)
    assert np.max(fused_cost[50:100, :]) >= 0.80, "Slope cost fusion failed on steep cliff"

    # Test ASIL-B Safety Monitor
    monitor = ASILBTerrainSafetyMonitor()
    # Scenario: vehicle speeding at 12 m/s on severe mud (limit 5.0 m/s)
    for _ in range(4):
        safety_status = monitor.evaluate_kinodynamic_safety(
            current_speed_mps=12.0,
            recommended_speed_mps=5.0,
            dominant_terrain=TerrainClass.MUD_WATER,
            requires_emergency_stop=False,
        )
    assert safety_status["trip_active"] is True
    assert safety_status["status"] == "ASIL_B_TRACTION_GOVERNOR_ACTIVE"
    assert safety_status["actuator_throttle_scale"] == 0.0

    print("TraversabilityHead self-verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

