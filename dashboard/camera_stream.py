"""
dashboard/camera_stream.py
==========================
Real-Time Decoupled Camera & Inference Streaming Pipeline.

Architecture:
  CameraThread (daemon)
       │ (continuous hardware capture at native FPS, CAP_PROP_BUFFERSIZE=1)
       ▼
  LatestFrameBuffer (depth=1, drops unread frames, tracks dropped count)
       │
       ▼
  InferenceWorker (daemon)
       │ (runs run_pipeline() on newest frame only, calls LKAStateTracker)
       ▼
  LatestResultBuffer (thread-safe latest InferenceResult + LKATelemetry)
       │
       ▼
  Streamlit UI (non-blocking read, 30 FPS display)

Guarantees:
- Zero stale frame accumulation in camera buffers.
- Minimal latency: inference always consumes the newest available frame.
- High UI responsiveness: capture and inference never block the UI thread.
- Hardware decoupling: Jetson Orin Nano / Linux / Windows compatible.
"""

import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Optional, Tuple, Union

import cv2
import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.pipeline import InferenceResult, run_pipeline
from src.lka_engine.lka_state import LKATelemetry, LKAStateTracker


# ═══════════════════════════════════════════════════════════════════════════
# Performance Telemetry
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PerformanceStats:
    """Real-time performance measurements from hardware, inference, and buffers."""
    camera_fps: float = 0.0
    inference_fps: float = 0.0
    ui_fps: float = 0.0
    inference_latency_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    dropped_frames: int = 0
    total_captured_frames: int = 0
    total_inferred_frames: int = 0
    cpu_usage_pct: float = 0.0
    ram_usage_pct: float = 0.0
    gpu_usage_pct: Optional[float] = None
    is_live: bool = False


# ═══════════════════════════════════════════════════════════════════════════
# Single-Slot Frame Buffer (Depth = 1)
# ═══════════════════════════════════════════════════════════════════════════

class LatestFrameBuffer:
    """
    Thread-safe, single-slot frame buffer that guarantees zero queue delay.
    When a new frame arrives, if the previous frame was not consumed,
    it is overwritten and recorded as a dropped frame.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._frame: Optional[np.ndarray] = None
        self._frame_id: int = 0
        self._timestamp: float = 0.0
        self._dropped_count: int = 0
        self._has_unread: bool = False

    def put(self, frame: np.ndarray, frame_id: int, timestamp: float) -> None:
        """Store the newest frame, overwriting any unread frame."""
        with self._lock:
            if self._has_unread:
                self._dropped_count += 1
            self._frame = frame
            self._frame_id = frame_id
            self._timestamp = timestamp
            self._has_unread = True
            self._new_frame_event.set()

    def get(self, timeout: Optional[float] = 0.5) -> Optional[Tuple[np.ndarray, int, float]]:
        """
        Wait for and retrieve the newest available frame.
        Clears the unread flag once consumed.
        """
        if not self._new_frame_event.wait(timeout):
            return None

        with self._lock:
            self._new_frame_event.clear()
            if self._frame is None:
                return None
            self._has_unread = False
            return self._frame, self._frame_id, self._timestamp

    @property
    def dropped_frames(self) -> int:
        with self._lock:
            return self._dropped_count

    def clear(self) -> None:
        with self._lock:
            self._frame = None
            self._has_unread = False
            self._new_frame_event.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Camera Capture Thread
# ═══════════════════════════════════════════════════════════════════════════

class CameraThread(threading.Thread):
    """
    Dedicated daemon thread for continuous video capture.
    Reads frames as fast as the camera hardware delivers them,
    bypassing OS buffer delay via buffer size configuration.
    """

    def __init__(
        self,
        source: Union[int, str],
        frame_buffer: LatestFrameBuffer,
        target_fps: int = 30,
        req_width: int = 640,
        req_height: int = 480,
    ) -> None:
        super().__init__(daemon=True, name="CameraThread")
        self.source = source
        self.frame_buffer = frame_buffer
        self.target_fps = target_fps
        self.req_width = req_width
        self.req_height = req_height

        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self.is_connected = False
        self.error_message: Optional[str] = None

        self.captured_count = 0
        self._fps_history: Deque[float] = deque(maxlen=30)
        self.current_fps: float = 0.0

    def run(self) -> None:
        # Open camera with DirectShow on Windows if integer device index
        if isinstance(self.source, int) and sys.platform.startswith("win"):
            self._cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        else:
            self._cap = cv2.VideoCapture(self.source)

        if not self._cap or not self._cap.isOpened():
            self.error_message = f"Failed to open camera device {self.source}"
            self.is_connected = False
            return

        # Attempt to configure camera properties
        try:
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.req_width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.req_height)
        except Exception:
            pass

        self.is_connected = True
        last_t = time.perf_counter()

        while not self._stop_event.is_set():
            ret, frame = self._cap.read()
            if not ret or frame is None:
                # Brief pause before retrying
                time.sleep(0.01)
                continue

            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            if dt > 0:
                self._fps_history.append(1.0 / dt)
                if len(self._fps_history) >= 5:
                    self.current_fps = round(float(np.mean(self._fps_history)), 1)

            self.captured_count += 1
            self.frame_buffer.put(frame, self.captured_count, time.time())

        # Cleanup
        if self._cap:
            try:
                self._cap.release()
            except Exception:
                pass
            self._cap = None
        self.is_connected = False

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Inference & LKA Worker Thread
# ═══════════════════════════════════════════════════════════════════════════

class InferenceWorker(threading.Thread):
    """
    Dedicated daemon thread running lane perception and LKA telemetry analysis.
    Consumes the newest frame from LatestFrameBuffer, runs run_pipeline(),
    and publishes latest results to an atomic slot.
    """

    def __init__(
        self,
        frame_buffer: LatestFrameBuffer,
        backend: str = "classical_cv",
    ) -> None:
        super().__init__(daemon=True, name="InferenceWorker")
        self.frame_buffer = frame_buffer
        self.backend = backend

        self._stop_event = threading.Event()
        self.lka_tracker = LKAStateTracker()

        self._result_lock = threading.Lock()
        self._latest_result: Optional[InferenceResult] = None
        self._latest_telemetry: Optional[LKATelemetry] = None

        self.inferred_count = 0
        self._fps_history: Deque[float] = deque(maxlen=30)
        self.current_fps: float = 0.0
        self.avg_inf_latency: float = 0.0
        self.avg_total_latency: float = 0.0

    def run(self) -> None:
        last_t = time.perf_counter()

        while not self._stop_event.is_set():
            frame_data = self.frame_buffer.get(timeout=0.2)
            if frame_data is None:
                continue

            frame, frame_id, t_capture = frame_data

            # Run inference pipeline
            reset_pd = (self.inferred_count == 0)
            res = run_pipeline(
                frame,
                backend=self.backend,
                reset_pd_state=reset_pd,
            )

            # Run LKA state engine
            telem = self.lka_tracker.update(res)

            now = time.perf_counter()
            dt = now - last_t
            last_t = now

            if dt > 0:
                self._fps_history.append(1.0 / dt)
                if len(self._fps_history) >= 5:
                    self.current_fps = round(float(np.mean(self._fps_history)), 1)

            inf_ms = res.timings.get("inference_ms", 0.0)
            tot_ms = res.timings.get("total_ms", 0.0)
            self.avg_inf_latency = inf_ms
            self.avg_total_latency = tot_ms
            self.inferred_count += 1

            with self._result_lock:
                self._latest_result = res
                self._latest_telemetry = telem

    def get_latest(self) -> Tuple[Optional[InferenceResult], Optional[LKATelemetry]]:
        with self._result_lock:
            return self._latest_result, self._latest_telemetry

    def stop(self) -> None:
        self._stop_event.set()
        self.join(timeout=1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Stream Manager (High-Level Controller)
# ═══════════════════════════════════════════════════════════════════════════

class StreamManager:
    """
    Singleton-capable manager that orchestrates the camera and inference threads.
    Provides clean start/stop lifecycle and performance profiling.
    """

    def __init__(self) -> None:
        self.frame_buffer = LatestFrameBuffer()
        self.camera_thread: Optional[CameraThread] = None
        self.inference_worker: Optional[InferenceWorker] = None
        self.is_running = False

        # Profiling
        self._ui_last_time = time.perf_counter()
        self._ui_fps_history: Deque[float] = deque(maxlen=20)
        self._ui_fps: float = 0.0
        self._last_system_check: float = 0.0
        self._cached_cpu: float = 0.0
        self._cached_ram: float = 0.0

    def start(self, source: Union[int, str] = 0, backend: str = "classical_cv") -> bool:
        """Start both camera and inference threads."""
        if self.is_running:
            self.stop()

        self.frame_buffer.clear()
        self.camera_thread = CameraThread(source=source, frame_buffer=self.frame_buffer)
        self.inference_worker = InferenceWorker(frame_buffer=self.frame_buffer, backend=backend)

        self.camera_thread.start()
        # Brief wait for camera initialization
        time.sleep(0.15)
        if not self.camera_thread.is_connected and self.camera_thread.error_message:
            self.camera_thread.stop()
            self.camera_thread = None
            return False

        self.inference_worker.start()
        self.is_running = True
        return True

    def stop(self) -> None:
        """Cleanly stop background streaming."""
        self.is_running = False
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread = None
        if self.inference_worker:
            self.inference_worker.stop()
            self.inference_worker = None
        self.frame_buffer.clear()

    def get_latest(self) -> Tuple[Optional[InferenceResult], Optional[LKATelemetry], PerformanceStats]:
        """
        Non-blocking read of latest perception output, LKA telemetry, and performance.
        Calculates UI rendering FPS and hardware load.
        """
        # Measure UI loop rate
        now = time.perf_counter()
        dt = now - self._ui_last_time
        self._ui_last_time = now
        if dt > 0:
            self._ui_fps_history.append(1.0 / dt)
            if len(self._ui_fps_history) >= 5:
                self._ui_fps = round(float(np.mean(self._ui_fps_history)), 1)

        # Update system resource usage once per second to avoid overhead
        if now - self._last_system_check > 1.0:
            self._cached_cpu = psutil.cpu_percent()
            self._cached_ram = psutil.virtual_memory().percent
            self._last_system_check = now

        cam_fps = self.camera_thread.current_fps if self.camera_thread else 0.0
        inf_fps = self.inference_worker.current_fps if self.inference_worker else 0.0
        inf_lat = self.inference_worker.avg_inf_latency if self.inference_worker else 0.0
        pipe_lat = self.inference_worker.avg_total_latency if self.inference_worker else 0.0
        tot_cap = self.camera_thread.captured_count if self.camera_thread else 0
        tot_inf = self.inference_worker.inferred_count if self.inference_worker else 0
        dropped = self.frame_buffer.dropped_frames

        perf = PerformanceStats(
            camera_fps=cam_fps,
            inference_fps=inf_fps,
            ui_fps=self._ui_fps,
            inference_latency_ms=inf_lat,
            pipeline_latency_ms=pipe_lat,
            dropped_frames=dropped,
            total_captured_frames=tot_cap,
            total_inferred_frames=tot_inf,
            cpu_usage_pct=self._cached_cpu,
            ram_usage_pct=self._cached_ram,
            gpu_usage_pct=None,  # CPU inference
            is_live=self.is_running,
        )

        if self.inference_worker:
            res, telem = self.inference_worker.get_latest()
            return res, telem, perf

        return None, None, perf


# Global manager instance for reuse across Streamlit reruns
_stream_manager: Optional[StreamManager] = None


def get_stream_manager() -> StreamManager:
    """Get or initialize the shared stream manager."""
    global _stream_manager
    if _stream_manager is None:
        _stream_manager = StreamManager()
    return _stream_manager
