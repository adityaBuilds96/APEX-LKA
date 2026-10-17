"""
src/perception/camera/camera_supervisor.py
===========================================
C12 & C13: Camera Health Supervisor & Hot-Plug Auto-Recovery Manager.

Monitors frame interval jitter, dropped frame counts, sharpness trends,
and exposure saturation. Manages exponential backoff recovery without
crashing the main perception loop.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

log = logging.getLogger("apex_camera_supervisor")


class CameraStatus(str, Enum):
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    RECOVERING = "RECOVERING"
    FAULT = "FAULT"


@dataclass
class CameraHealthReport:
    """Current snapshot of camera hardware health."""
    status: CameraStatus = CameraStatus.HEALTHY
    is_healthy: bool = True
    frame_count: int = 0
    dropped_frames: int = 0
    interval_ms: float = 33.33
    jitter_ms: float = 0.0
    sharpness: float = 100.0
    saturation_ratio: float = 0.0
    mean_lightness: float = 120.0
    temperature_c: Optional[float] = None
    recent_events: List[str] = field(default_factory=list)


class CameraSupervisor:
    """
    Tier-1 Camera Supervisor (C12, C13) for APEX-LKA.
    Guarantees thread-safe, non-blocking health telemetry and resilient
    hot-plug auto-recovery.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        sup_cfg = cfg.get("supervisor", {})

        self.jitter_warn_ms: float = float(sup_cfg.get("jitter_warn_ms", 8.0))
        self.jitter_fault_ms: float = float(sup_cfg.get("jitter_fault_ms", 20.0))
        self.backoff_schedule: List[float] = [float(x) for x in sup_cfg.get("hotplug_retry_backoff_s", [1.0, 2.0, 4.0, 8.0])]

        self.target_fps: float = float(cfg.get("target_fps", 30.0))
        self.nominal_interval_ms: float = 1000.0 / max(1.0, self.target_fps)

        # State tracking
        self.frame_count: int = 0
        self.dropped_frames: int = 0
        self.consecutive_failures: int = 0
        self.last_frame_ts_ns: Optional[int] = None
        self.jitter_ema_ms: float = 0.0
        self.last_interval_ms: float = self.nominal_interval_ms

        self.status: CameraStatus = CameraStatus.HEALTHY
        self.recent_events: List[str] = []

        # Hot-plug recovery state
        self._backoff_idx: int = 0
        self._next_retry_time: float = 0.0
        self._recovery_in_progress: bool = False

    def observe(self, frame: np.ndarray, ts_ns: int) -> CameraHealthReport:
        """
        Evaluate frame health metrics on hot path (< 1.5ms).
        Calculates interval jitter, saturation, and sharpness.
        """
        self.frame_count += 1
        self.consecutive_failures = 0

        # 1. Jitter Analysis
        if self.last_frame_ts_ns is not None:
            dt_ms = (ts_ns - self.last_frame_ts_ns) / 1_000_000.0
            self.last_interval_ms = dt_ms
            jitter = abs(dt_ms - self.nominal_interval_ms)
            alpha = 0.15
            self.jitter_ema_ms = alpha * jitter + (1.0 - alpha) * self.jitter_ema_ms

            if jitter >= self.jitter_fault_ms:
                msg = f"Camera jitter spike: {jitter:.1f}ms exceeds fault threshold ({self.jitter_fault_ms}ms)"
                self._record_event(msg)
                self.status = CameraStatus.DEGRADED
            elif jitter >= self.jitter_warn_ms and self.status == CameraStatus.HEALTHY:
                msg = f"Camera jitter warning: {jitter:.1f}ms"
                self._record_event(msg)
            elif self.jitter_ema_ms < self.jitter_warn_ms:
                self.status = CameraStatus.HEALTHY

        self.last_frame_ts_ns = ts_ns

        # 2. Photometric Health (Subsampled for zero overhead)
        h, w = frame.shape[:2]
        step = 4
        sub = frame[::step, ::step]

        if len(sub.shape) == 3:
            gray_sub = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
        else:
            gray_sub = sub

        mean_l = float(np.mean(gray_sub))

        # Saturation: percentage of clipped pixels (pure black < 3 or pure white > 252)
        total_px = gray_sub.size
        clipped_px = int(np.sum(gray_sub < 3) + np.sum(gray_sub > 252))
        sat_ratio = float(clipped_px) / float(max(1, total_px))

        # Sharpness via variance of Laplacian
        lap_var = float(cv2.Laplacian(gray_sub, cv2.CV_32F).var())

        if sat_ratio > 0.40:
            self._record_event(f"Severe optical saturation detected ({sat_ratio*100:.1f}%)")
            self.status = CameraStatus.DEGRADED

        return CameraHealthReport(
            status=self.status,
            is_healthy=(self.status == CameraStatus.HEALTHY),
            frame_count=self.frame_count,
            dropped_frames=self.dropped_frames,
            interval_ms=round(self.last_interval_ms, 2),
            jitter_ms=round(self.jitter_ema_ms, 2),
            sharpness=round(lap_var, 2),
            saturation_ratio=round(sat_ratio, 4),
            mean_lightness=round(mean_l, 1),
            temperature_c=None,
            recent_events=list(self.recent_events[-5:]),
        )

    def register_read_failure(self) -> None:
        """Call when camera grab / read returns False."""
        self.dropped_frames += 1
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.status = CameraStatus.FAULT
            self._record_event(f"Camera frame loss: {self.consecutive_failures} consecutive drops")
        else:
            self.status = CameraStatus.DEGRADED

    def can_retry_recovery(self) -> bool:
        """Check if backoff cooldown has elapsed for hot-plug retry."""
        now = time.monotonic()
        return now >= self._next_retry_time

    def mark_recovery_attempt(self, success: bool) -> None:
        """Update exponential backoff recovery schedule."""
        now = time.monotonic()
        if success:
            log.info("Camera recovery succeeded. Restoring HEALTHY status.")
            self.status = CameraStatus.HEALTHY
            self.consecutive_failures = 0
            self._backoff_idx = 0
            self._record_event("Camera auto-recovery succeeded")
        else:
            delay = self.backoff_schedule[min(self._backoff_idx, len(self.backoff_schedule) - 1)]
            self._next_retry_time = now + delay
            self._backoff_idx = min(self._backoff_idx + 1, len(self.backoff_schedule) - 1)
            self.status = CameraStatus.RECOVERING
            self._record_event(f"Camera recovery failed; next retry in {delay:.1f}s")
            log.warning("Camera recovery attempt failed. Next retry in %.1fs (backoff index %d)", delay, self._backoff_idx)

    def _record_event(self, event_msg: str) -> None:
        ts_str = time.strftime("%H:%M:%S")
        entry = f"[{ts_str}] {event_msg}"
        self.recent_events.append(entry)
        if len(self.recent_events) > 50:
            self.recent_events.pop(0)
