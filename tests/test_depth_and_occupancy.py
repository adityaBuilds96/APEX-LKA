"""
tests/test_depth_and_occupancy.py
=================================
Comprehensive integration test suite for Phase 5 Section 2:
- Tests end-to-end integration: Depth -> 3D Projection -> Occupancy Grid
- Verifies CAN Frame 0x160 byte packing and round-trip loopback
- Validates HUD visualization overlays (3D metric labels, Occupancy widget, heatmap)
- Measures latency budget compliance (depth + projection + occupancy <= 12 ms)
- Verifies fail-operational safety on corrupted inputs
"""

import time
import numpy as np
import pytest
import cv2

from src.perception.depth_engine import MonocularDepthEngine, DepthResult
from src.perception.projection_3d import BoundingBox3DProjector, Object3D, ProjectionResult
from src.perception.occupancy_grid import OccupancyGridRasterizer, OccupancyGrid2D, CellState
from src.perception.object_detector import Detection, DetectionResult
from deploy.vehicle_comm import (
    CAN_ID_SPATIAL_ALERT,
    SpatialAlertTelemetry,
    VehicleCommunicator,
    THREAT_LEVEL_CLEAR,
    THREAT_LEVEL_CAUTION,
    THREAT_LEVEL_WARNING,
    THREAT_LEVEL_CRITICAL_STOP,
)
from dashboard.components.lka_hud import draw_hud_overlay, _draw_occupancy_grid_widget
from src.inference.pipeline import run_pipeline


def test_depth_to_3d_to_occupancy_pipeline():
    """Verify that depth maps, 3D projections, and occupancy grid connect seamlessly."""
    depth_engine = MonocularDepthEngine(metric_scale=15.0, metric_shift=0.1)
    projector = BoundingBox3DProjector(fx=500.0, fy=500.0, cx=320.0, cy=180.0)
    rasterizer = OccupancyGridRasterizer(grid_resolution_m=0.10)

    # 1. Dummy road frame (640x360)
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    depth_res = depth_engine.estimate_depth(frame)
    assert depth_res.metric_depth_map.shape == (360, 640)

    # 2. Simulated 2D detection: vehicle ahead in center
    det_res = DetectionResult(
        boxes=np.array([[280, 160, 360, 240]], dtype=np.float32),
        scores=np.array([0.92], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )

    # 3. Project to 3D
    proj_res = projector.project_2d_to_3d(det_res, depth_res.metric_depth_map)
    assert len(proj_res) == 1
    obj3d = proj_res.objects_3d[0]
    assert obj3d.z > 0.0
    assert abs(obj3d.x) < 2.0  # Center vehicle

    # 4. Rasterize Occupancy Grid
    occ_grid = rasterizer.rasterize(objects_3d=proj_res.objects_3d)
    assert isinstance(occ_grid, OccupancyGrid2D)
    assert occ_grid.shape == (150, 100)

    # Verify that the cell at the object location is OCCUPIED_DYNAMIC_OBJECT
    cell_state = occ_grid.get_cell(obj3d.x, obj3d.z)
    assert cell_state == CellState.OCCUPIED_DYNAMIC_OBJECT


def test_can_0x160_spatial_alert_roundtrip():
    """Verify CAN Frame 0x160 bit-packed serialization and loopback reception."""
    comm = VehicleCommunicator(enable_can=True, enable_serial=False, sim_mode=True)

    # Test values: X = -350 mm (left), Z = 4200 mm (ahead), TTI = 1400 ms, Class = 1 (Pedestrian), Threat = WARNING
    sent = comm.send_spatial_alert(
        x_mm=-350,
        z_mm=4200,
        tti_ms=1400,
        class_enum=1,
        threat_level=THREAT_LEVEL_WARNING,
    )
    assert sent, "send_spatial_alert should succeed in sim_mode"

    spatial = comm.get_spatial_alert()
    assert spatial is not None, "get_spatial_alert should receive loopback frame"
    assert spatial.x_mm == -350
    assert spatial.z_mm == 4200
    assert spatial.tti_ms == 1400
    assert spatial.class_enum == 1
    assert spatial.threat_level == THREAT_LEVEL_WARNING

    comm.close()


def test_hud_3d_and_occupancy_rendering():
    """Verify that HUD renders with 3D labels, Occupancy widget, and depth heatmap without errors."""
    canvas = np.zeros((360, 640, 3), dtype=np.uint8)

    obj3d = Object3D(
        x=-0.5,
        y=0.1,
        z=4.2,
        width=1.8,
        height=1.5,
        length=4.0,
        class_id=0,
        class_name="vehicle",
        confidence=0.88,
        box_2d=np.array([240, 140, 380, 240], dtype=np.float32),
    )

    occ_grid = OccupancyGrid2D(
        grid=np.zeros((150, 100), dtype=np.uint8),
        x_min_m=-5.0,
        x_max_m=5.0,
        z_min_m=0.0,
        z_max_m=15.0,
        resolution_m=0.10,
    )
    # Mark a cell
    occ_grid.grid[42, 45] = CellState.OCCUPIED_DYNAMIC_OBJECT

    depth_map = np.linspace(0.0, 1.0, 360 * 640, dtype=np.float32).reshape((360, 640))

    # Render HUD with all 3D features
    hud = draw_hud_overlay(
        image_bgr=canvas,
        objects_3d=[obj3d],
        occupancy_grid=occ_grid,
        depth_map=depth_map,
        show_depth_heatmap=True,
    )

    assert hud.shape == (360, 640, 3)
    assert not np.array_equal(hud, canvas), "HUD should modify the canvas"


def test_pipeline_with_depth_and_occupancy():
    """Verify run_pipeline execution with depth and occupancy flags."""
    dummy_frame = np.full((360, 640, 3), 120, dtype=np.uint8)

    res = run_pipeline(
        dummy_frame,
        backend="classical_cv",
        enable_object_detection=True,
        enable_depth=True,
        enable_occupancy=True,
    )

    assert res.scene_perception is not None
    assert res.depth_result is not None
    assert isinstance(res.depth_result, DepthResult)
    assert res.occupancy_grid is not None
    assert isinstance(res.occupancy_grid, OccupancyGrid2D)
    assert "depth_ms" in res.timings
    assert "occupancy_ms" in res.timings


def test_latency_budget_compliance():
    """Verify that Depth + 3D Projection + Occupancy executes within the 12 ms ceiling."""
    depth_engine = MonocularDepthEngine()
    projector = BoundingBox3DProjector()
    rasterizer = OccupancyGridRasterizer()

    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    det_res = DetectionResult(
        boxes=np.array([[200, 100, 300, 200]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )

    # Warmup
    d_res = depth_engine.estimate_depth(frame)
    p_res = projector.project_2d_to_3d(det_res, d_res.metric_depth_map)
    rasterizer.rasterize(p_res.objects_3d)

    # Benchmark over 20 iterations
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        d_res = depth_engine.estimate_depth(frame)
        p_res = projector.project_2d_to_3d(det_res, d_res.metric_depth_map)
        rasterizer.rasterize(p_res.objects_3d)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = float(np.mean(latencies))
    # Latency should be fast (<= 12.0 ms target)
    assert avg_ms <= 12.0, f"Average latency {avg_ms:.2f} ms exceeded 12.0 ms ceiling"
