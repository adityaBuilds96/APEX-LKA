"""
src/fleet
=========
APEX-LKA Phase 5 Section 6: Fleet Infrastructure & Continuous Edge MLOps:
- Cryptographic Over-The-Air (OTA) Model Updater with Atomic Hot-Swapping
- Silent Shadow-Mode Execution Engine & Divergence Benchmarking
- Closed-Loop Edge Failure Harvester & Data Collection
"""

from src.fleet.ota_updater import (
    OTAModelManager,
    OTASyncState,
    OTAManifest,
    ModelBackendType,
)
from src.fleet.shadow_mode import (
    ShadowModeEngine,
    ShadowDivergenceReport,
    PromotionStatus,
)
from src.fleet.data_collector import (
    ClosedLoopDataHarvester,
    HarvestTriggerReason,
    HarvestEvent,
)

__all__ = [
    "OTAModelManager",
    "OTASyncState",
    "OTAManifest",
    "ModelBackendType",
    "ShadowModeEngine",
    "ShadowDivergenceReport",
    "PromotionStatus",
    "ClosedLoopDataHarvester",
    "HarvestTriggerReason",
    "HarvestEvent",
]
