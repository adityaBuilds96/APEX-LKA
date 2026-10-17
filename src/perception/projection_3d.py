"""
src/perception/projection_3d.py
===============================
Standard-named alias module for 3d_projection.py (compliant with PEP 8 import identifiers).
"""

from __future__ import annotations
import importlib

_mod = importlib.import_module("src.perception.3d_projection")
BoundingBox3DProjector = _mod.BoundingBox3DProjector
Object3D = _mod.Object3D
ProjectionResult = _mod.ProjectionResult
CLASS_3D_PRIORS = _mod.CLASS_3D_PRIORS

__all__ = [
    "BoundingBox3DProjector",
    "Object3D",
    "ProjectionResult",
    "CLASS_3D_PRIORS",
]
