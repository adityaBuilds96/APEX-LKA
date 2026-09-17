"""
dashboard/components/recorder.py
================================
Session Recording and Replay module for APEX LKA.

Features:
- Records live camera frames and synchronous ADAS telemetry to disk.
- Generates datasets for model training, evaluation, and edge-case analysis.
- Disk space safety guard: hard cap at 500MB to avoid filling storage.
- SessionReplayer allows playing back recorded sessions through the same pipeline.
"""

import csv
import os
import shutil
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.lka_engine.lka_state import LKATelemetry
from dashboard.camera_stream import PerformanceStats

# Default recording directory
RECORDINGS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "recordings"
MAX_SESSION_BYTES = 500 * 1024 * 1024  # 500 MB hard limit


class SessionRecorder:
    """
    Records image frames and synchronized telemetry to a timestamped session folder.
    """

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        self.base_dir = base_dir or RECORDINGS_DIR
        self.is_recording = False
        self.session_dir: Optional[Path] = None
        self.frames_dir: Optional[Path] = None
        self.csv_file: Optional[Path] = None
        self._csv_writer = None
        self._csv_fp = None

        self.frames_recorded = 0
        self.bytes_written = 0
        self.start_time: float = 0.0
        self.auto_stopped_due_to_quota = False

    def start(self) -> Path:
        """Initialize and start a new recording session."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / f"session_{timestamp_str}"
        self.frames_dir = self.session_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.csv_file = self.session_dir / "telemetry.csv"
        self._csv_fp = open(self.csv_file, mode="w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_fp)

        # Header
        self._csv_writer.writerow([
            "frame_idx",
            "timestamp",
            "image_filename",
            "lateral_offset_px",
            "lateral_offset_m",
            "lateral_offset_norm",
            "heading_error_deg",
            "lane_width_m",
            "confidence",
            "stability_state",
            "ldw_state",
            "steering_recommendation",
            "recommended_angle_deg",
            "steering_command",
            "inference_ms",
            "total_ms",
            "camera_fps",
        ])
        self._csv_fp.flush()

        self.frames_recorded = 0
        self.bytes_written = 0
        self.start_time = time.time()
        self.auto_stopped_due_to_quota = False
        self.is_recording = True
        return self.session_dir

    def record_frame(
        self,
        frame_bgr: np.ndarray,
        telemetry: Optional[LKATelemetry] = None,
        perf: Optional[PerformanceStats] = None,
    ) -> bool:
        """
        Save a single frame and its corresponding telemetry row.
        Returns False if quota exceeded or stopped.
        """
        if not self.is_recording or self.frames_dir is None:
            return False

        # Quota check
        if self.bytes_written >= MAX_SESSION_BYTES:
            self.auto_stopped_due_to_quota = True
            self.stop()
            return False

        self.frames_recorded += 1
        img_filename = f"frame_{self.frames_recorded:06d}.jpg"
        img_path = self.frames_dir / img_filename

        # Write JPEG (quality 92 for good balance of size and fidelity)
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 92]
        cv2.imwrite(str(img_path), frame_bgr, encode_param)

        try:
            file_size = img_path.stat().st_size
            self.bytes_written += file_size
        except Exception:
            pass

        # Write CSV telemetry
        if self._csv_writer and self._csv_fp:
            t = telemetry
            now_iso = datetime.now().isoformat()
            self._csv_writer.writerow([
                self.frames_recorded,
                now_iso,
                img_filename,
                f"{t.lateral_offset_px:.1f}" if (t and t.lateral_offset_px is not None) else "",
                f"{t.lateral_offset_m:.3f}" if (t and t.lateral_offset_m is not None) else "",
                f"{t.lateral_offset_norm:.4f}" if (t and t.lateral_offset_norm is not None) else "",
                f"{t.heading_error_deg:.2f}" if (t and t.heading_error_deg is not None) else "",
                f"{t.lane_width_m:.2f}" if (t and t.lane_width_m is not None) else "",
                f"{t.combined_confidence:.2f}" if t else "",
                t.stability_state.value if t else "",
                t.ldw_state.value if t else "",
                t.steering_recommendation.value if t else "",
                f"{t.recommended_angle_deg:.1f}" if t else "",
                f"{t.steering_command:.4f}" if (t and t.steering_command is not None) else "",
                f"{t.inference_ms:.1f}" if t else "",
                f"{t.total_ms:.1f}" if t else "",
                f"{perf.camera_fps:.1f}" if perf else "",
            ])
            self._csv_fp.flush()

        return True

    def stop(self) -> None:
        """Stop recording and flush files."""
        self.is_recording = False
        if self._csv_fp:
            try:
                self._csv_fp.close()
            except Exception:
                pass
            self._csv_fp = None
            self._csv_writer = None

    @property
    def disk_usage_mb(self) -> float:
        return round(self.bytes_written / (1024.0 * 1024.0), 1)

    @property
    def elapsed_seconds(self) -> float:
        if not self.is_recording:
            return 0.0
        return round(time.time() - self.start_time, 1)


# Global singleton instance
_recorder_instance: Optional[SessionRecorder] = None


def get_recorder() -> SessionRecorder:
    global _recorder_instance
    if _recorder_instance is None:
        _recorder_instance = SessionRecorder()
    return _recorder_instance


# ═══════════════════════════════════════════════════════════════════════════
# Replay Utilities
# ═══════════════════════════════════════════════════════════════════════════

def list_recorded_sessions(base_dir: Optional[Path] = None) -> List[Dict[str, any]]:
    """Scan recordings folder and return summary of existing sessions."""
    directory = base_dir or RECORDINGS_DIR
    if not directory.exists():
        return []

    sessions = []
    for p in sorted(directory.iterdir(), reverse=True):
        if p.is_dir() and p.name.startswith("session_"):
            frames_dir = p / "frames"
            csv_file = p / "telemetry.csv"
            frame_count = len(list(frames_dir.glob("*.jpg"))) if frames_dir.exists() else 0
            size_mb = sum(f.stat().st_size for f in p.rglob("*")) / (1024.0 * 1024.0)

            sessions.append({
                "id": p.name,
                "path": str(p),
                "created": p.name.replace("session_", ""),
                "frames": frame_count,
                "size_mb": round(size_mb, 1),
                "has_telemetry": csv_file.exists(),
            })

    return sessions


class SessionReplayer:
    """
    Loads a recorded session and replays frames sequentially.
    """

    def __init__(self, session_path: Union[str, Path]) -> None:
        self.session_path = Path(session_path)
        self.frames_dir = self.session_path / "frames"
        self.csv_path = self.session_path / "telemetry.csv"
        self.frame_files = sorted(self.frames_dir.glob("*.jpg")) if self.frames_dir.exists() else []
        self.total_frames = len(self.frame_files)
        self.current_idx = 0

    def get_frame(self, idx: int) -> Optional[np.ndarray]:
        """Fetch frame by 0-indexed position."""
        if 0 <= idx < self.total_frames:
            return cv2.imread(str(self.frame_files[idx]))
        return None
