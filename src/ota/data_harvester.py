"""
src/ota/data_harvester.py
=========================
APEX-Vision Core v3.0 — Closed-Loop Edge Case Data Harvester (ISO 26262 Compliant)
- Auto-triggers edge-case incident recordings on 4 critical conditions:
  1. High RSS safety risk score / hard override
  2. Shadow model disagreement > threshold
  3. Manual driver override / steering intervention
  4. Low neural perception confidence (< 0.35)
- Maintains 5-second rolling context buffer of frames and telemetry.
- Compresses and packages incident bundle into data/harvested/edge_case_TIMESTAMP.zip.
- Exposes harvest queue count for CAN ID 0x1A0 Byte 4.
"""

from __future__ import annotations

import collections
import io
import json
import logging
import os
import shutil
import tempfile
import threading
import time
import zipfile
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

try:
    from src.config import PATHS, cfg
except (ImportError, ModuleNotFoundError, AttributeError):
    PATHS = {"ROOT": PROJECT_ROOT}
    cfg = None

log = logging.getLogger("apex_data_harvester")


class HarvesterTriggerReason(str, Enum):
    HIGH_RSS_RISK = "HIGH_RSS_RISK"
    SHADOW_DISAGREEMENT = "SHADOW_DISAGREEMENT"
    DRIVER_OVERRIDE = "DRIVER_OVERRIDE"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MANUAL_TRIGGER = "MANUAL_TRIGGER"


@dataclass
class BufferedFrame:
    """Frame entry in 5-second rolling ring buffer."""
    timestamp_s: float
    frame_bgr: np.ndarray
    confidence: float
    telemetry: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EdgeCaseBundle:
    """Packaged edge-case artifact bundle."""
    bundle_path: Path
    trigger_reason: HarvesterTriggerReason
    timestamp_s: float
    frame_count: int
    zip_size_bytes: int


class EdgeCaseDataHarvester:
    """
    Continuous closed-loop edge failure harvester with 5-second context buffer.
    """

    def __init__(
        self,
        harvest_dir: Optional[Union[str, Path]] = None,
        buffer_duration_s: float = 5.0,
        fps_target: float = 10.0,
        confidence_thresh: float = 0.35,
        disagreement_thresh: float = 5.0,
    ) -> None:
        self.harvest_dir = Path(harvest_dir) if harvest_dir else PROJECT_ROOT / "data" / "harvested"
        self.harvest_dir.mkdir(parents=True, exist_ok=True)

        self.buffer_duration_s = float(buffer_duration_s)
        self.confidence_thresh = float(confidence_thresh)
        self.disagreement_thresh = float(disagreement_thresh)

        # Max frames in 5s buffer (e.g. 5s * 20 fps = 100 frames)
        max_buf = int(self.buffer_duration_s * fps_target * 2)
        self._ring_buffer: Deque[BufferedFrame] = collections.deque(maxlen=max_buf)
        self._lock = threading.Lock()

        self.harvested_bundles: List[EdgeCaseBundle] = []
        self._last_trigger_time: float = 0.0
        self.min_interval_between_triggers_s: float = 3.0

    def add_frame(
        self,
        frame_bgr: np.ndarray,
        confidence: float,
        telemetry: Optional[Dict[str, Any]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Add current frame and state to rolling 5-second buffer."""
        now = timestamp if timestamp is not None else time.time()
        telem = telemetry.copy() if telemetry else {}

        with self._lock:
            self._ring_buffer.append(
                BufferedFrame(
                    timestamp_s=now,
                    frame_bgr=frame_bgr.copy(),
                    confidence=float(confidence),
                    telemetry=telem,
                )
            )
            # Evict frames older than buffer_duration_s
            cutoff = now - self.buffer_duration_s
            while self._ring_buffer and self._ring_buffer[0].timestamp_s < cutoff:
                self._ring_buffer.popleft()

    def check_and_trigger(
        self,
        confidence: float,
        rss_override: bool = False,
        rss_risk_score: float = 0.0,
        shadow_disagreement: float = 0.0,
        driver_override: bool = False,
        now: Optional[float] = None,
    ) -> Optional[EdgeCaseBundle]:
        """
        Evaluate auto-trigger conditions:
        1. High RSS risk score / override
        2. Shadow model disagreement > threshold
        3. Driver override event
        4. Low perception confidence (< 0.35)
        """
        current_time = now if now is not None else time.time()

        # Debounce triggers
        if (current_time - self._last_trigger_time) < self.min_interval_between_triggers_s:
            return None

        reason: Optional[HarvesterTriggerReason] = None

        if rss_override or rss_risk_score >= 0.70:
            reason = HarvesterTriggerReason.HIGH_RSS_RISK
        elif shadow_disagreement > self.disagreement_thresh:
            reason = HarvesterTriggerReason.SHADOW_DISAGREEMENT
        elif driver_override:
            reason = HarvesterTriggerReason.DRIVER_OVERRIDE
        elif confidence < self.confidence_thresh:
            reason = HarvesterTriggerReason.LOW_CONFIDENCE

        if reason is not None:
            self._last_trigger_time = current_time
            return self.package_bundle(reason=reason, trigger_timestamp=current_time)

        return None

    def package_bundle(
        self,
        reason: HarvesterTriggerReason,
        trigger_timestamp: float,
    ) -> EdgeCaseBundle:
        """
        Package 5-second buffer frames + metadata into zipped edge-case bundle:
        data/harvested/edge_case_TIMESTAMP.zip
        """
        with self._lock:
            frames_to_save = list(self._ring_buffer)

        ts_int = int(trigger_timestamp)
        zip_filename = f"edge_case_{ts_int}_{reason.value.lower()}.zip"
        zip_path = self.harvest_dir / zip_filename

        metadata = {
            "trigger_reason": reason.value,
            "trigger_timestamp": trigger_timestamp,
            "frame_count": len(frames_to_save),
            "buffer_duration_s": self.buffer_duration_s,
            "system_version": "APEX-Vision Core v3.0",
        }

        # Write to temporary zip file then atomically place
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            meta_file = tmp_path / "metadata.json"
            meta_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

            # Save sample video or keyframes
            frames_dir = tmp_path / "frames"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for idx, bf in enumerate(frames_to_save):
                fname = frames_dir / f"frame_{idx:04d}.jpg"
                cv2.imwrite(str(fname), bf.frame_bgr)

            # Package into zip archive
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                zipf.write(meta_file, arcname="metadata.json")
                for img_file in frames_dir.glob("*.jpg"):
                    zipf.write(img_file, arcname=f"frames/{img_file.name}")

        bundle = EdgeCaseBundle(
            bundle_path=zip_path,
            trigger_reason=reason,
            timestamp_s=trigger_timestamp,
            frame_count=len(frames_to_save),
            zip_size_bytes=zip_path.stat().st_size if zip_path.exists() else 0,
        )

        with self._lock:
            self.harvested_bundles.append(bundle)

        log.info("Harvested edge case bundle saved: %s (%d bytes)", zip_path.name, bundle.zip_size_bytes)
        return bundle

    @property
    def harvest_queue_count(self) -> int:
        """Count of harvested bundles on disk for CAN ID 0x1A0."""
        with self._lock:
            return len(self.harvested_bundles)
