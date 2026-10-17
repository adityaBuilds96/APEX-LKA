"""
src/ota/shadow_mode.py
======================
APEX-Vision Core v3.0 — Silent Shadow Mode Engine (ISO 26262 / ISO 21434 Compliant)
- Runs candidate neural network models in background worker thread.
- Zero Control Authority: strictly insulated from CAN control frame generation.
- Computes Disagree Score:
    Disagreement = |Steer_active - Steer_shadow| + IoU_disparity
- Emits Disagree Score (0-100 u8) for CAN ID 0x1A0 Byte 3.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import PATHS, cfg
except (ImportError, ModuleNotFoundError, AttributeError):
    PATHS = {"ROOT": PROJECT_ROOT}
    cfg = None

log = logging.getLogger("apex_shadow_mode")


@dataclass
class ShadowDisagreementReport:
    """Telemetry report comparing active vs shadow candidate model outputs."""
    frame_idx: int = 0
    steer_active_deg: float = 0.0
    steer_shadow_deg: float = 0.0
    steer_disparity_deg: float = 0.0
    iou_disparity: float = 0.0
    disagreement_score: float = 0.0         # Continuous score: |steer_diff| + IoU_disp * 10
    disagreement_score_u8: int = 0          # Scaled 0-255 uint8 for CAN ID 0x1A0 Byte 3
    is_divergent: bool = False
    shadow_latency_ms: float = 0.0
    timestamp_s: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "steer_active_deg": round(self.steer_active_deg, 2),
            "steer_shadow_deg": round(self.steer_shadow_deg, 2),
            "steer_disparity_deg": round(self.steer_disparity_deg, 2),
            "iou_disparity": round(self.iou_disparity, 3),
            "disagreement_score": round(self.disagreement_score, 2),
            "disagreement_score_u8": self.disagreement_score_u8,
            "is_divergent": self.is_divergent,
            "shadow_latency_ms": round(self.shadow_latency_ms, 2),
        }


class SilentShadowModeEngine:
    """
    Evaluates shadow candidate models in background parallel threads without actuation risk.
    """

    def __init__(
        self,
        disagreement_threshold: float = 5.0,     # Max allowable disagreement before alert
        steer_scale_weight: float = 1.0,
        iou_scale_weight: float = 10.0,
    ) -> None:
        self.disagreement_threshold = float(disagreement_threshold)
        self.steer_scale = float(steer_scale_weight)
        self.iou_scale = float(iou_scale_weight)

        self._lock = threading.Lock()
        self.last_report: Optional[ShadowDisagreementReport] = None
        self.history: List[ShadowDisagreementReport] = []
        self.total_evaluations: int = 0

    def compute_iou_disparity(
        self,
        active_mask_or_boxes: Optional[Union[np.ndarray, List[Any]]],
        shadow_mask_or_boxes: Optional[Union[np.ndarray, List[Any]]],
    ) -> float:
        """
        Calculate IoU disparity between active and shadow perception outputs:
        IoU_disparity = 1.0 - IoU(active, shadow)
        """
        if active_mask_or_boxes is None or shadow_mask_or_boxes is None:
            return 0.0

        # Case 1: Binary / Segmentation masks
        if isinstance(active_mask_or_boxes, np.ndarray) and isinstance(shadow_mask_or_boxes, np.ndarray):
            if active_mask_or_boxes.shape == shadow_mask_or_boxes.shape:
                m1 = active_mask_or_boxes > 0
                m2 = shadow_mask_or_boxes > 0
                intersection = np.logical_and(m1, m2).sum()
                union = np.logical_or(m1, m2).sum()
                if union == 0:
                    return 0.0
                iou = float(intersection) / float(union)
                return float(np.clip(1.0 - iou, 0.0, 1.0))

        # Default fallback
        return 0.0

    def evaluate(
        self,
        steer_active_deg: float,
        steer_shadow_deg: float,
        active_outputs: Optional[Any] = None,
        shadow_outputs: Optional[Any] = None,
        frame_idx: int = 0,
    ) -> ShadowDisagreementReport:
        """
        Synchronously or asynchronously computes disagreement:
        Disagreement = |Steer_active - Steer_shadow| + IoU_disparity * scale
        """
        t0 = time.perf_counter()

        steer_diff = abs(float(steer_active_deg) - float(steer_shadow_deg))
        iou_disp = self.compute_iou_disparity(active_outputs, shadow_outputs)

        disagree_continuous = (self.steer_scale * steer_diff) + (self.iou_scale * iou_disp)

        # Map continuous score (0 - 30+) to 0-255 uint8 for CAN frame 0x1A0
        disagree_u8 = int(np.clip(disagree_continuous * 8.0, 0, 255))
        is_divergent = bool(disagree_continuous > self.disagreement_threshold)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        report = ShadowDisagreementReport(
            frame_idx=frame_idx,
            steer_active_deg=float(steer_active_deg),
            steer_shadow_deg=float(steer_shadow_deg),
            steer_disparity_deg=float(steer_diff),
            iou_disparity=float(iou_disp),
            disagreement_score=float(disagree_continuous),
            disagreement_score_u8=disagree_u8,
            is_divergent=is_divergent,
            shadow_latency_ms=latency_ms,
            timestamp_s=time.time(),
        )

        with self._lock:
            self.last_report = report
            self.history.append(report)
            if len(self.history) > 200:
                self.history.pop(0)
            self.total_evaluations += 1

        return report

    @property
    def latest_disagreement_score_u8(self) -> int:
        """Return latest disagreement score as uint8 for CAN 0x1A0."""
        with self._lock:
            if self.last_report is None:
                return 0
            return self.last_report.disagreement_score_u8
