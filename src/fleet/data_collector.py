"""
src/fleet/data_collector.py
===========================
Closed-Loop Edge Failure Harvester & Data Collection Engine:
- Auto-triggers snapshot capture on anomalies:
  1. Low Perception Confidence (< 0.40 for >= 3 consecutive frames)
  2. Mobileye RSS Shield Interventions
  3. Sharp Manual Driver Override
  4. High Shadow Model Divergence (> 0.40m)
- Bundles 5-frame temporal window [T-2, T-1, T, T+1, T+2] containing raw images,
  depth maps, 3D object detections, ESKF pose, and CAN control telemetry.
- Enforces strict 2.0 GB disk quota with FIFO rolling deletion to prevent storage exhaustion.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

log = logging.getLogger("apex_data_harvester")


class HarvestTriggerReason(str, Enum):
    """Event triggers initiating snapshot capture."""
    LOW_CONFIDENCE      = "LOW_CONFIDENCE"
    RSS_INTERVENTION    = "RSS_INTERVENTION"
    DRIVER_OVERRIDE     = "DRIVER_OVERRIDE"
    SHADOW_DIVERGENCE   = "SHADOW_DIVERGENCE"
    MANUAL_TRIGGER      = "MANUAL_TRIGGER"


@dataclass
class FrameSnapshot:
    """Telemetry bundle for a single time step."""
    frame_idx: int
    timestamp_s: float
    frame_bgr: np.ndarray
    confidence: float
    offset_m: float
    depth_map: Optional[np.ndarray] = None
    objects_3d: Optional[List[Any]] = None
    vehicle_pose: Optional[Any] = None
    can_state: Optional[Dict[str, Any]] = None


@dataclass
class HarvestEvent:
    """Metadata manifest of a packaged edge incident."""
    event_id: str
    trigger_reason: HarvestTriggerReason
    timestamp_s: float
    trigger_frame_idx: int
    frame_count: int
    confidence_at_trigger: float
    event_dir: Path
    package_size_bytes: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "trigger_reason": self.trigger_reason.value,
            "timestamp_s": self.timestamp_s,
            "trigger_frame_idx": self.trigger_frame_idx,
            "frame_count": self.frame_count,
            "confidence_at_trigger": round(float(self.confidence_at_trigger), 3),
            "event_dir": str(self.event_dir),
            "package_size_bytes": self.package_size_bytes,
        }


class ClosedLoopDataHarvester:
    """
    Automated continuous learning harvester capturing edge cases for retrain pipelines.
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "data/harvested",
        storage_quota_mb: float = 2048.0,       # 2.0 GB hard limit
        trigger_confidence_low: float = 0.40,
        consecutive_low_conf: int = 3,
        trigger_divergence_m: float = 0.40,
        pre_event_frames: int = 2,
        post_event_frames: int = 2,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.storage_quota_bytes = int(storage_quota_mb * 1024 * 1024)

        self.trigger_confidence_low = float(trigger_confidence_low)
        self.consecutive_low_conf = int(consecutive_low_conf)
        self.trigger_divergence_m = float(trigger_divergence_m)
        self.pre_event_frames = int(pre_event_frames)
        self.post_event_frames = int(post_event_frames)
        self.enabled = enabled

        self._buffer_size = self.pre_event_frames + self.post_event_frames + 1
        self._history_buffer: Deque[FrameSnapshot] = collections.deque(maxlen=self._buffer_size)

        # Trigger detection state
        self._consecutive_low_conf_count: int = 0
        self._active_capture: Optional[Dict[str, Any]] = None
        self._lock = threading.Lock()

        # Telemetry metrics
        self.total_events_captured: int = 0
        self.harvest_queue_count: int = 0

        # Scan existing events on startup to set queue count and enforce quota
        self._refresh_harvest_queue()

    def _refresh_harvest_queue(self) -> None:
        """Counts existing harvested packages and enforces FIFO storage limits."""
        if not self.output_dir.exists():
            self.harvest_queue_count = 0
            return

        event_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("event_")]
        self.harvest_queue_count = min(255, len(event_dirs))
        self._enforce_storage_quota()

    def _get_dir_size(self, path: Path) -> int:
        """Returns total recursive size in bytes."""
        total = 0
        for entry in path.rglob("*"):
            if entry.is_file():
                try:
                    total += entry.stat().st_size
                except OSError:
                    pass
        return total

    def _enforce_storage_quota(self) -> None:
        """Purge oldest event folders (FIFO) if total storage exceeds quota."""
        try:
            total_bytes = self._get_dir_size(self.output_dir)
            if total_bytes <= self.storage_quota_bytes:
                return

            # Sort event dirs oldest first
            event_dirs = sorted(
                [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("event_")],
                key=lambda p: p.stat().st_mtime,
            )

            for d in event_dirs:
                if total_bytes <= self.storage_quota_bytes:
                    break
                d_size = self._get_dir_size(d)
                shutil.rmtree(d, ignore_errors=True)
                total_bytes -= d_size
                log.info("Storage quota exceeded: Purged oldest package %s (%d KB)", d.name, d_size // 1024)

            remaining_dirs = [d for d in self.output_dir.iterdir() if d.is_dir() and d.name.startswith("event_")]
            self.harvest_queue_count = min(255, len(remaining_dirs))
        except Exception as e:
            log.warning("Quota enforcement error: %s", e)

    def observe(
        self,
        frame_bgr: np.ndarray,
        confidence: float,
        offset_m: float,
        frame_idx: int = 0,
        depth_map: Optional[np.ndarray] = None,
        objects_3d: Optional[List[Any]] = None,
        vehicle_pose: Optional[Any] = None,
        can_state: Optional[Dict[str, Any]] = None,
        rss_override_active: bool = False,
        driver_override_active: bool = False,
        shadow_divergence_m: float = 0.0,
    ) -> Optional[HarvestEvent]:
        """
        Observes current frame telemetry, checks trigger conditions,
        and saves multi-frame failure bundles.
        """
        if not self.enabled:
            return None

        # 1. Store snapshot in circular history
        snapshot = FrameSnapshot(
            frame_idx=frame_idx,
            timestamp_s=time.time(),
            frame_bgr=frame_bgr.copy(),
            confidence=confidence,
            offset_m=offset_m,
            depth_map=depth_map.copy() if depth_map is not None else None,
            objects_3d=objects_3d,
            vehicle_pose=vehicle_pose,
            can_state=can_state,
        )
        self._history_buffer.append(snapshot)

        # 2. Check Trigger Conditions (if not already completing a capture)
        trigger_reason: Optional[HarvestTriggerReason] = None

        # Check consecutive low confidence
        if confidence < self.trigger_confidence_low:
            self._consecutive_low_conf_count += 1
            if self._consecutive_low_conf_count >= self.consecutive_low_conf:
                trigger_reason = HarvestTriggerReason.LOW_CONFIDENCE
        else:
            self._consecutive_low_conf_count = 0

        # Check RSS override
        if rss_override_active:
            trigger_reason = HarvestTriggerReason.RSS_INTERVENTION

        # Check driver override
        if driver_override_active:
            trigger_reason = HarvestTriggerReason.DRIVER_OVERRIDE

        # Check shadow divergence
        if shadow_divergence_m > self.trigger_divergence_m:
            trigger_reason = HarvestTriggerReason.SHADOW_DIVERGENCE

        # 3. Handle Active or New Trigger
        trigger_event: Optional[HarvestEvent] = None
        if trigger_reason is not None and self._active_capture is None:
            t_id = int(time.time() * 1000)
            reason_str = trigger_reason.value.lower()
            event_name = f"event_{t_id}_{reason_str}"
            event_dir = self.output_dir / event_name
            event_dir.mkdir(parents=True, exist_ok=True)

            trigger_event = HarvestEvent(
                event_id=event_name,
                trigger_reason=trigger_reason,
                timestamp_s=time.time(),
                trigger_frame_idx=frame_idx,
                frame_count=len(self._history_buffer),
                confidence_at_trigger=confidence,
                event_dir=event_dir,
            )

            self._active_capture = {
                "event": trigger_event,
                "trigger_reason": trigger_reason,
                "trigger_frame_idx": frame_idx,
                "trigger_confidence": confidence,
                "remaining_post_frames": self.post_event_frames,
                "event_dir": event_dir,
                "timestamp_s": time.time(),
            }
            # Persist bundle with available frames
            self._save_bundle(self._active_capture)
            return trigger_event

        # 4. Countdown post-event frames
        if self._active_capture is not None:
            if self._active_capture["remaining_post_frames"] > 0:
                self._active_capture["remaining_post_frames"] -= 1
                if self._active_capture["remaining_post_frames"] == 0:
                    # Window completed: Finalize full bundle on disk
                    event = self._save_bundle(self._active_capture)
                    self._active_capture = None
                    self._consecutive_low_conf_count = 0
                    return event
                return None

        return None

    def trigger_manual(self, reason: str = "MANUAL_TRIGGER") -> None:
        """Manually trigger package capture on operator command."""
        t_id = int(time.time() * 1000)
        event_name = f"event_{t_id}_manual"
        event_dir = self.output_dir / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        self._active_capture = {
            "trigger_reason": HarvestTriggerReason.MANUAL_TRIGGER,
            "trigger_frame_idx": self._history_buffer[-1].frame_idx if self._history_buffer else 0,
            "trigger_confidence": self._history_buffer[-1].confidence if self._history_buffer else 1.0,
            "remaining_post_frames": self.post_event_frames,
            "event_dir": event_dir,
            "timestamp_s": time.time(),
        }
        self._save_bundle(self._active_capture)

    def _save_bundle(self, capture_meta: Dict[str, Any]) -> HarvestEvent:
        """Writes 5-frame sequence, depth maps, and telemetry files to storage."""
        event_dir = capture_meta.get("event_dir")
        if event_dir is None:
            t_id = int(capture_meta["timestamp_s"] * 1000)
            reason_str = capture_meta["trigger_reason"].value.lower()
            event_name = f"event_{t_id}_{reason_str}"
            event_dir = self.output_dir / event_name
            event_dir.mkdir(parents=True, exist_ok=True)
            capture_meta["event_dir"] = event_dir
        else:
            event_name = event_dir.name

        frames_to_save = list(self._history_buffer)
        saved_bytes = 0

        for i, snap in enumerate(frames_to_save):
            # 1. Save frame image
            img_path = event_dir / f"frame_{i:02d}_{snap.frame_idx}.jpg"
            cv2.imwrite(str(img_path), snap.frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if img_path.exists():
                saved_bytes += img_path.stat().st_size

            # 2. Save depth map if available (both .npy and .png)
            if snap.depth_map is not None:
                depth_npy = event_dir / f"depth_{i:02d}_{snap.frame_idx}.npy"
                np.save(str(depth_npy), snap.depth_map)
                if depth_npy.exists():
                    saved_bytes += depth_npy.stat().st_size

                depth_norm = cv2.normalize(snap.depth_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                depth_path = event_dir / f"depth_{i:02d}_{snap.frame_idx}.png"
                cv2.imwrite(str(depth_path), depth_norm)
                if depth_path.exists():
                    saved_bytes += depth_path.stat().st_size

            # 3. Save telemetry state per frame
            telemetry_data = {
                "frame_idx": snap.frame_idx,
                "timestamp_s": snap.timestamp_s,
                "confidence": snap.confidence,
                "offset_m": snap.offset_m,
                "can_state": snap.can_state or {},
                "num_3d_objects": len(snap.objects_3d) if snap.objects_3d else 0,
            }
            json_path = event_dir / f"telemetry_{i:02d}_{snap.frame_idx}.json"
            json_path.write_text(json.dumps(telemetry_data, indent=2))
            if json_path.exists():
                saved_bytes += json_path.stat().st_size

        # Save event manifest (both package_manifest.json and manifest.json)
        event = HarvestEvent(
            event_id=event_name,
            trigger_reason=capture_meta["trigger_reason"],
            timestamp_s=capture_meta["timestamp_s"],
            trigger_frame_idx=capture_meta["trigger_frame_idx"],
            frame_count=len(frames_to_save),
            confidence_at_trigger=capture_meta["trigger_confidence"],
            event_dir=event_dir,
            package_size_bytes=saved_bytes,
        )

        manifest_data = event.to_dict()
        manifest_data["total_frames"] = len(frames_to_save)

        pkg_manifest = event_dir / "package_manifest.json"
        pkg_manifest.write_text(json.dumps(manifest_data, indent=2))
        saved_bytes += pkg_manifest.stat().st_size

        legacy_manifest = event_dir / "manifest.json"
        legacy_manifest.write_text(json.dumps(manifest_data, indent=2))
        saved_bytes += legacy_manifest.stat().st_size

        event.package_size_bytes = saved_bytes

        self.total_events_captured += 1
        self._refresh_harvest_queue()
        log.info("Harvested edge event saved: %s (%d frames, %d KB)", event_name, len(frames_to_save), saved_bytes // 1024)

        return event
