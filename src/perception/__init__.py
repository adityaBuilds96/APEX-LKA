"""
APEX-LKA Perception Package
===========================
Exports core perception engines:
- ObjectDetector & DetectionResult
- PriorityRiskAssessor & RiskAssessmentReport
- MonocularDepthEngine & DepthResult
- BoundingBox3DProjector, Object3D, & ProjectionResult
- OccupancyGridRasterizer & OccupancyGrid2D
"""

import importlib

from src.perception.object_detector import (
    Detection,
    DetectionResult,
    ObjectDetector,
    OBJECT_CLASSES,
)
from src.perception.risk_assessment import (
    AssessedObject,
    PriorityRiskAssessor,
    RiskAssessmentReport,
    CLASS_SEVERITY_MAP,
)
from src.perception.depth_engine import (
    DepthResult,
    MonocularDepthEngine,
)

from src.perception.occupancy_grid import (
    CellState,
    OccupancyGrid2D,
    OccupancyGridRasterizer,
)

# Dynamically import 3d_projection (leading digit module)
_proj_mod = importlib.import_module("src.perception.3d_projection")
BoundingBox3DProjector = _proj_mod.BoundingBox3DProjector
Object3D = _proj_mod.Object3D
ProjectionResult = _proj_mod.ProjectionResult
CLASS_3D_PRIORS = _proj_mod.CLASS_3D_PRIORS

__all__ = [
    "Detection",
    "DetectionResult",
    "ObjectDetector",
    "OBJECT_CLASSES",
    "AssessedObject",
    "PriorityRiskAssessor",
    "RiskAssessmentReport",
    "CLASS_SEVERITY_MAP",
    "DepthResult",
    "MonocularDepthEngine",
    "BoundingBox3DProjector",
    "Object3D",
    "ProjectionResult",
    "CLASS_3D_PRIORS",
    "CellState",
    "OccupancyGrid2D",
    "OccupancyGridRasterizer",
]
