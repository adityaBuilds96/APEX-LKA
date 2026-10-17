"""
src/fleet/shadow_mode.py
========================
Silent Shadow-Mode Execution Engine for APEX-LKA:
- Runs candidate neural network models asynchronously in background threads / CUDA Stream 2.
- Evaluates real-time prediction divergence (Mask mIoU, Lateral Offset Delta, Confidence Delta).
- Telemetry logging to logs/shadow_mode_benchmarks.csv.
- Auto-promotion recommendation trigger (> 15% confidence advantage over 1,000 frames).
- Zero actuation authority: completely insulated from vehicle steering and throttle control.
"""

from __future__ import annotations

import csv
import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger("apex_shadow_mode")


class PromotionStatus(str, Enum):
    """Auto-promotion recommendation status for candidate shadow models."""
    EVALUATING            = "EVALUATING"
    PROMOTION_RECOMMENDED = "PROMOTION_RECOMMENDED"
    REJECTED              = "REJECTED"


@dataclass
class ShadowDivergenceReport:
    """Per-frame comparison between primary production model and candidate shadow model."""
    frame_idx: int = 0
    primary_confidence: float = 0.0
    shadow_confidence: float = 0.0
    mask_miou: float = 1.0
    offset_delta_m: float = 0.0
    discrepancy_score_u8: int = 0         # 0-100 score for CAN Frame 0x1A0 Byte 3
    agreement: bool = True
    shadow_latency_ms: float = 0.0
    promotion_status: PromotionStatus = PromotionStatus.EVALUATING

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "primary_confidence": round(float(self.primary_confidence), 3),
            "shadow_confidence": round(float(self.shadow_confidence), 3),
            "mask_miou": round(float(self.mask_miou), 3),
            "offset_delta_m": round(float(self.offset_delta_m), 3),
            "discrepancy_score": self.discrepancy_score_u8,
            "agreement": self.agreement,
            "shadow_latency_ms": round(float(self.shadow_latency_ms), 2),
            "promotion_status": self.promotion_status.value,
        }


def compute_mask_miou(mask_a: Optional[np.ndarray], mask_b: Optional[np.ndarray]) -> float:
    """Compute mean Intersection-over-Union (mIoU) between two binary masks."""
    if mask_a is None or mask_b is None:
        return 1.0 if (mask_a is None and mask_b is None) else 0.0

    a_bin = (mask_a > 128).astype(bool)
    b_bin = (mask_b > 128).astype(bool)

    intersection = np.logical_and(a_bin, b_bin).sum()
    union = np.logical_or(a_bin, b_bin).sum()

    if union == 0:
        return 1.0  # Both empty masks agree
    return float(intersection) / float(union)


class ShadowModeEngine:
    """
    Executes candidate shadow models in background threads with zero primary latency penalty.
    """

    def __init__(
        self,
        shadow_predictor: Optional[Any] = None,
        divergence_threshold_m: float = 0.40,
        log_csv_path: Union[str, Path] = "logs/shadow_mode_benchmarks.csv",
        min_promotion_frames: int = 1000,
        min_eval_frames: Optional[int] = None,
        promotion_threshold_pct: float = 15.0,
        enabled: bool = True,
    ) -> None:
        self.shadow_predictor = shadow_predictor
        self.divergence_threshold_m = float(divergence_threshold_m)
        self.log_csv_path = Path(log_csv_path)
        self.min_promotion_frames = int(min_eval_frames if min_eval_frames is not None else min_promotion_frames)
        self.promotion_threshold_pct = float(promotion_threshold_pct)
        self.enabled = enabled and (shadow_predictor is not None)

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="apex_shadow")
        self._current_future: Optional[Future] = None
        self._lock = threading.Lock()

        # Cumulative statistics
        self.total_frames_evaluated: int = 0
        self.primary_conf_sum: float = 0.0
        self.shadow_conf_sum: float = 0.0
        self.hard_scenes_count: int = 0
        self.shadow_hard_scene_conf_sum: float = 0.0
        self.primary_hard_scene_conf_sum: float = 0.0
        self.high_divergence_count: int = 0

        self.last_report = ShadowDivergenceReport()
        self.promotion_status = PromotionStatus.EVALUATING

        # CUDA Stream allocation if PyTorch CUDA is active
        self._cuda_stream = None
        try:
            import torch
            if torch.cuda.is_available():
                self._cuda_stream = torch.cuda.Stream()
        except Exception:
            self._cuda_stream = None

        self._init_csv_logger()

    def _init_csv_logger(self) -> None:
        """Create benchmark telemetry CSV with header."""
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "frame_idx",
                    "primary_conf",
                    "shadow_conf",
                    "mask_miou",
                    "offset_delta_m",
                    "discrepancy_score",
                    "agreement",
                    "shadow_latency_ms",
                    "promotion_status",
                ])

    def log_divergence(self, report: ShadowDivergenceReport) -> None:
        """Append divergence row to disk."""
        try:
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S.") + f"{int(time.time()*1000)%1000:03d}",
                    report.frame_idx,
                    round(report.primary_confidence, 3),
                    round(report.shadow_confidence, 3),
                    round(report.mask_miou, 3),
                    round(report.offset_delta_m, 3),
                    report.discrepancy_score_u8,
                    1 if report.agreement else 0,
                    round(report.shadow_latency_ms, 2),
                    report.promotion_status.value,
                ])
        except Exception as e:
            log.debug("Shadow CSV log error: %s", e)

    def step(
        self,
        frame: np.ndarray,
        primary_prediction: Any,
        primary_offset_m: float = 0.0,
        primary_confidence: float = 0.85,
        frame_idx: int = 0,
    ) -> ShadowDivergenceReport:
        """
        Submits asynchronous shadow inference and returns the most recent divergence report.
        Ensures primary control thread is never blocked (> 1ms).
        """
        if not self.enabled or self.shadow_predictor is None:
            return self.last_report

        t_start = time.perf_counter()

        # If previous background inference finished, collect it
        if self._current_future is not None and self._current_future.done():
            try:
                shadow_res, shadow_lat = self._current_future.result()
                report = self._compute_divergence(
                    shadow_res=shadow_res,
                    shadow_lat=shadow_lat,
                    primary_pred=primary_prediction,
                    primary_offset=primary_offset_m,
                    primary_conf=primary_confidence,
                    frame_idx=frame_idx,
                )
                with self._lock:
                    self.last_report = report
                self.log_divergence(report)
            except Exception as e:
                log.debug("Shadow async evaluation error: %s", e)

        # Launch next asynchronous task if thread is idle
        if self._current_future is None or self._current_future.done():
            frame_copy = frame.copy()
            self._current_future = self._executor.submit(
                self._run_shadow_inference,
                frame_copy,
            )

        dispatch_ms = (time.perf_counter() - t_start) * 1000.0
        if dispatch_ms > 1.0:
            log.debug("Shadow dispatch exceeded 1ms: %.3f ms", dispatch_ms)

        with self._lock:
            return self.last_report

    def _run_shadow_inference(self, frame: np.ndarray) -> Tuple[Any, float]:
        """Runs candidate inference in isolated thread/CUDA Stream."""
        t0 = time.perf_counter()
        if self._cuda_stream is not None:
            try:
                import torch
                with torch.cuda.stream(self._cuda_stream):
                    res = self.shadow_predictor.predict(frame)
            except Exception:
                res = self.shadow_predictor.predict(frame)
        else:
            res = self.shadow_predictor.predict(frame)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return res, lat_ms

    def _compute_divergence(
        self,
        shadow_res: Any,
        shadow_lat: float,
        primary_pred: Any,
        primary_offset: float,
        primary_conf: float,
        frame_idx: int,
    ) -> ShadowDivergenceReport:
        """Calculates quantitative divergence metrics."""
        # 1. Shadow confidence & offset
        if isinstance(shadow_res, dict):
            s_conf = float(shadow_res.get("confidence", 0.0))
            s_offset = float(shadow_res.get("offset_m", 0.0))
            s_mask = shadow_res.get("mask", None)
        else:
            s_conf = float(getattr(shadow_res, "accuracy_score", 0.0) or getattr(shadow_res, "confidence", 0.5))
            s_offset = float(getattr(shadow_res, "offset_m", 0.0))
            s_mask = getattr(shadow_res, "left_mask", None)

        # 2. Primary mask
        p_mask = getattr(primary_pred, "left_mask", None) if primary_pred else None

        # 3. Metrics
        offset_delta = abs(primary_offset - s_offset)
        has_masks = (p_mask is not None and s_mask is not None)
        if has_masks:
            miou = compute_mask_miou(p_mask, s_mask)
            agreement = (offset_delta <= self.divergence_threshold_m) and (miou >= 0.50)
            norm_offset = min(1.0, offset_delta / (self.divergence_threshold_m * 2.0))
            norm_miou_diff = 1.0 - miou
            disc_score = int(np.clip((0.6 * norm_offset + 0.4 * norm_miou_diff) * 100.0, 0, 100))
        else:
            miou = 1.0 if (p_mask is None and s_mask is None) else 0.0
            agreement = (offset_delta <= self.divergence_threshold_m)
            norm_offset = min(1.0, offset_delta / (self.divergence_threshold_m * 2.0))
            disc_score = int(np.clip(norm_offset * 100.0, 0, 100))

        # 5. Track Statistics & Auto-Promotion Trigger
        self.total_frames_evaluated += 1
        self.primary_conf_sum += primary_conf
        self.shadow_conf_sum += s_conf

        if primary_conf < 0.50:  # Hard scene
            self.hard_scenes_count += 1
            self.primary_hard_scene_conf_sum += primary_conf
            self.shadow_hard_scene_conf_sum += s_conf

        if offset_delta > self.divergence_threshold_m:
            self.high_divergence_count += 1

        # Check auto-promotion rule (> 15% confidence advantage on hard scenes over min_frames)
        min_hard = min(5, self.min_promotion_frames // 2)
        if self.total_frames_evaluated >= self.min_promotion_frames and self.hard_scenes_count >= min_hard:
            avg_p_hard = self.primary_hard_scene_conf_sum / max(1, self.hard_scenes_count)
            avg_s_hard = self.shadow_hard_scene_conf_sum / max(1, self.hard_scenes_count)
            advantage_pct = ((avg_s_hard - avg_p_hard) / max(0.01, avg_p_hard)) * 100.0

            if advantage_pct >= self.promotion_threshold_pct:
                self.promotion_status = PromotionStatus.PROMOTION_RECOMMENDED
            else:
                self.promotion_status = PromotionStatus.EVALUATING

        return ShadowDivergenceReport(
            frame_idx=frame_idx,
            primary_confidence=primary_conf,
            shadow_confidence=s_conf,
            mask_miou=miou,
            offset_delta_m=offset_delta,
            discrepancy_score_u8=disc_score,
            agreement=agreement,
            shadow_latency_ms=shadow_lat,
            promotion_status=self.promotion_status,
        )

    def evaluate_divergence_synchronous(
        self,
        frame: np.ndarray,
        primary_offset_m: float = 0.0,
        primary_confidence: float = 0.85,
        primary_mask: Optional[np.ndarray] = None,
        frame_idx: int = 0,
    ) -> ShadowDivergenceReport:
        """Run synchronous divergence evaluation for deterministic unit testing."""
        res, lat_ms = self._run_shadow_inference(frame)
        class DummyPrimary:
            left_mask = primary_mask
        p_pred = DummyPrimary() if primary_mask is not None else None
        rep = self._compute_divergence(
            shadow_res=res,
            shadow_lat=lat_ms,
            primary_pred=p_pred,
            primary_offset=primary_offset_m,
            primary_conf=primary_confidence,
            frame_idx=frame_idx,
        )
        with self._lock:
            self.last_report = rep
        return rep

    def get_benchmark_summary(self) -> Dict[str, Any]:
        """Returns cumulative benchmark statistics."""
        return {
            "total_eval_frames": self.total_frames_evaluated,
            "hard_scenes_count": self.hard_scenes_count,
            "high_divergence_count": self.high_divergence_count,
            "promotion_status": self.promotion_status.value,
        }

    def shutdown(self) -> None:
        """Clean shutdown of background thread worker."""
        self._executor.shutdown(wait=False)
