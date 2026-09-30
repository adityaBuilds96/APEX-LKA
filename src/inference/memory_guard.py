"""
src/inference/memory_guard.py
=============================
Continuous RAM & VRAM monitoring, monotonic leak detection, and automated cleanup.

Features:
- Periodic or per-frame memory tracking (Process RSS & CUDA VRAM).
- Monotonic growth detection over sliding evaluation windows.
- Automated multi-stage memory reclamation:
    1. gc.collect()
    2. torch.cuda.empty_cache() (if CUDA available)
    3. Pruning registered history ring buffers
    4. Model reload callback on persistent memory exhaustion
- Persistent diagnostic logging to logs/memory_monitor.csv
- Fixed-capacity thread-safe history buffers (depth <= 100)
"""

from __future__ import annotations

import csv
import gc
import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

import psutil

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

log = logging.getLogger("apex_memory_guard")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class BoundedHistoryBuffer:
    """
    Thread-safe circular ring-buffer strictly bounded to `max_size` (default 100).
    Guarantees no unbounded memory expansion over extended runtimes.
    """

    def __init__(self, max_size: int = 100) -> None:
        self.max_size = max(1, max_size)
        self._deque: Deque[Any] = deque(maxlen=self.max_size)
        self._lock = threading.Lock()

    def append(self, item: Any) -> None:
        with self._lock:
            self._deque.append(item)

    def clear(self) -> None:
        with self._lock:
            self._deque.clear()

    def to_list(self) -> List[Any]:
        with self._lock:
            return list(self._deque)

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)

    def __getitem__(self, index: int) -> Any:
        with self._lock:
            return self._deque[index]


class MemorySnapshot:
    """Immutable record of memory consumption at a specific timestamp."""

    def __init__(
        self,
        timestamp: float,
        ram_rss_mb: float,
        ram_vms_mb: float,
        gpu_alloc_mb: float = 0.0,
        gpu_res_mb: float = 0.0,
    ) -> None:
        self.timestamp = timestamp
        self.ram_rss_mb = ram_rss_mb
        self.ram_vms_mb = ram_vms_mb
        self.gpu_alloc_mb = gpu_alloc_mb
        self.gpu_res_mb = gpu_res_mb

    def to_dict(self) -> Dict[str, float]:
        return {
            "timestamp": self.timestamp,
            "ram_rss_mb": round(self.ram_rss_mb, 2),
            "ram_vms_mb": round(self.ram_vms_mb, 2),
            "gpu_alloc_mb": round(self.gpu_alloc_mb, 2),
            "gpu_res_mb": round(self.gpu_res_mb, 2),
        }


class MemoryGuard:
    """
    Active monitor that tracks memory consumption, detects leaks, and triggers cleanups.
    """

    def __init__(
        self,
        check_interval_s: float = 30.0,
        leak_window_size: int = 5,
        leak_threshold_mb: float = 50.0,
        persistent_leak_threshold: int = 3,
        log_csv_path: Optional[Union[str, Path]] = None,
        on_persistent_leak: Optional[Callable[[], None]] = None,
    ) -> None:
        self.check_interval_s = float(check_interval_s)
        self.leak_window_size = max(3, leak_window_size)
        self.leak_threshold_mb = float(leak_threshold_mb)
        self.persistent_leak_threshold = max(1, persistent_leak_threshold)
        self.on_persistent_leak = on_persistent_leak

        self._process = psutil.Process(os.getpid())
        self._history: Deque[MemorySnapshot] = deque(maxlen=50)
        self._registered_buffers: List[BoundedHistoryBuffer] = []
        self._lock = threading.Lock()

        self.consecutive_leak_detections: int = 0
        self.total_cleanups_triggered: int = 0
        self._last_check_time: float = 0.0
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Diagnostic CSV log file
        if log_csv_path is None:
            log_dir = PROJECT_ROOT / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            self.log_csv_path = log_dir / "memory_monitor.csv"
        else:
            self.log_csv_path = Path(log_csv_path)
            self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._init_csv()

    def _init_csv(self) -> None:
        if not self.log_csv_path.exists():
            try:
                with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "timestamp",
                        "ram_rss_mb",
                        "ram_vms_mb",
                        "gpu_alloc_mb",
                        "gpu_res_mb",
                        "cleanup_triggered",
                        "leak_count",
                    ])
            except Exception as e:
                log.warning("Could not initialize memory monitor CSV: %s", e)

    def register_buffer(self, buffer: BoundedHistoryBuffer) -> None:
        """Register a history buffer to be purged on memory reclamation events."""
        with self._lock:
            if buffer not in self._registered_buffers:
                self._registered_buffers.append(buffer)

    def sample_memory(self) -> MemorySnapshot:
        """Capture instantaneous RAM and VRAM metrics."""
        now = time.time()
        try:
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)
            vms_mb = mem_info.vms / (1024 * 1024)
        except Exception:
            rss_mb, vms_mb = 0.0, 0.0

        gpu_alloc_mb = 0.0
        gpu_res_mb = 0.0
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_alloc_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                gpu_res_mb = torch.cuda.memory_reserved() / (1024 * 1024)
            except Exception:
                pass

        snapshot = MemorySnapshot(now, rss_mb, vms_mb, gpu_alloc_mb, gpu_res_mb)
        with self._lock:
            self._history.append(snapshot)
            self._last_check_time = now
        return snapshot

    def detect_monotonic_growth(self) -> bool:
        """
        Evaluate whether RAM consumption has grown monotonically across the last
        `leak_window_size` samples by more than `leak_threshold_mb`.
        """
        with self._lock:
            if len(self._history) < self.leak_window_size:
                return False

            samples = list(self._history)[-self.leak_window_size:]

        # Verify strict monotonic or near-monotonic rise
        net_increase = samples[-1].ram_rss_mb - samples[0].ram_rss_mb
        if net_increase < self.leak_threshold_mb:
            return False

        increases = sum(
            1 for i in range(1, len(samples))
            if samples[i].ram_rss_mb >= samples[i - 1].ram_rss_mb
        )
        # If at least 80% of step transitions are non-decreasing
        is_monotonic = increases >= (len(samples) - 2)
        return is_monotonic

    def perform_cleanup(self, force: bool = False) -> bool:
        """
        Execute tiered memory reclamation:
        1. Explicit garbage collection
        2. PyTorch CUDA cache flush
        3. Clear registered history buffers
        4. Model reload callback on persistent leak
        """
        self.total_cleanups_triggered += 1
        log.warning("MemoryGuard: Triggering active memory reclamation (Pass #%d)", self.total_cleanups_triggered)

        # 1. Python GC
        collected = gc.collect()

        # 2. PyTorch CUDA flush
        if HAS_TORCH and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning("CUDA cache empty failed: %s", e)

        # 3. Clear buffers
        with self._lock:
            for buf in self._registered_buffers:
                buf.clear()

        # 4. Check if leak persists across cleanup passes
        if self.consecutive_leak_detections >= self.persistent_leak_threshold or force:
            log.critical(
                "MemoryGuard: Persistent memory leak detected (%d consecutive cycles). Triggering reload callback.",
                self.consecutive_leak_detections,
            )
            if self.on_persistent_leak:
                try:
                    self.on_persistent_leak()
                except Exception as e:
                    log.error("Failed executing persistent leak callback: %s", e)

        return True

    def check(self, force_sample: bool = False) -> Dict[str, Any]:
        """
        Evaluate system memory, log to CSV, and trigger cleanup if growth is detected.
        """
        now = time.time()
        if not force_sample and (now - self._last_check_time < self.check_interval_s):
            return {"checked": False}

        snapshot = self.sample_memory()
        is_leak = self.detect_monotonic_growth()
        cleanup_performed = False

        if is_leak:
            self.consecutive_leak_detections += 1
            cleanup_performed = self.perform_cleanup()
        else:
            if self.consecutive_leak_detections > 0:
                self.consecutive_leak_detections = max(0, self.consecutive_leak_detections - 1)

        # Append to CSV
        try:
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(snapshot.timestamp)),
                    f"{snapshot.ram_rss_mb:.2f}",
                    f"{snapshot.ram_vms_mb:.2f}",
                    f"{snapshot.gpu_alloc_mb:.2f}",
                    f"{snapshot.gpu_res_mb:.2f}",
                    1 if cleanup_performed else 0,
                    self.consecutive_leak_detections,
                ])
        except Exception:
            pass

        return {
            "checked": True,
            "snapshot": snapshot.to_dict(),
            "leak_detected": is_leak,
            "consecutive_leaks": self.consecutive_leak_detections,
            "cleanup_triggered": cleanup_performed,
        }

    def start_background_monitoring(self) -> None:
        """Start daemon monitoring thread."""
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        self._stop_event.clear()

        def _monitor_loop() -> None:
            while not self._stop_event.is_set():
                try:
                    self.check()
                except Exception as e:
                    log.error("Error in MemoryGuard background loop: %s", e)
                self._stop_event.wait(self.check_interval_s)

        self._worker_thread = threading.Thread(
            target=_monitor_loop,
            name="ApexMemoryGuardWorker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop_background_monitoring(self) -> None:
        """Stop daemon monitoring thread."""
        self._stop_event.set()
        if self._worker_thread is not None:
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None
