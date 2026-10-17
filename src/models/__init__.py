"""
APEX-Vision Core v3.0 — Models Package
=======================================
Multi-Task Neural Network architectures for real-time autonomous vehicle perception.
"""

from src.models.perception_core import (
    HydraNet,
    APEXPerceptionCore,
    PerceptionOutput,
    Detection2D,
    OBJECT_CLASSES,
)

__all__ = [
    "HydraNet",
    "APEXPerceptionCore",
    "PerceptionOutput",
    "Detection2D",
    "OBJECT_CLASSES",
]
