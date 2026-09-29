"""
src/dataset/batch_processor.py
==============================
Thread-safe asynchronous batch processing queue with progress telemetry.

Provides:
  - Non-blocking background worker daemon thread
  - Thread-safe FIFO task queue
  - Per-item state tracking: QUEUED -> PROCESSING -> DONE -> FAILED
  - Real-time throughput calculations (frames/sec) and ETA estimation
  - Dynamic cancellation and pause support
  - Safe error isolation (individual item exceptions do not halt batch)
"""

import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


class ItemStatus(str, Enum):
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    DONE = "DONE"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class BatchItem:
    item_id: str
    source_path: Path
    status: ItemStatus = ItemStatus.QUEUED
    result: Optional[Any] = None
    error_message: Optional[str] = None
    queued_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


@dataclass
class BatchProgress:
    total_items: int = 0
    queued: int = 0
    processing: int = 0
    completed: int = 0
    failed: int = 0
    percent: float = 0.0
    items_per_sec: float = 0.0
    eta_sec: float = 0.0
    is_running: bool = False
    is_cancelled: bool = False


class BatchProcessor:
    """
    Asynchronous daemon worker managing batch operations.
    """

    def __init__(
        self,
        worker_func: Callable[[Path], Any],
        max_workers: int = 1,
    ):
        """
        Initialize batch processor.

        Args:
            worker_func: Callable that receives an item Path and returns a result.
            max_workers: Number of background worker threads.
        """
        self.worker_func = worker_func
        self.max_workers = max_workers
        self._queue: queue.Queue = queue.Queue()
        self._items: Dict[str, BatchItem] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Unpaused by default
        self._threads: List[threading.Thread] = []
        self._start_time: Optional[float] = None

    def add_item(self, item_path: Path, item_id: Optional[str] = None) -> str:
        """Enqueue an item for processing."""
        p = Path(item_path)
        uid = item_id or f"{p.name}_{time.time()}_{len(self._items)}"
        item = BatchItem(item_id=uid, source_path=p)
        with self._lock:
            self._items[uid] = item
        self._queue.put(item)
        return uid

    def add_items(self, item_paths: List[Path]) -> List[str]:
        """Enqueue multiple items."""
        return [self.add_item(p) for p in item_paths]

    def start(self) -> None:
        """Start the background daemon worker threads."""
        self._stop_event.clear()
        self._start_time = time.time()

        for i in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, name=f"BatchWorker-{i}", daemon=True)
            self._threads.append(t)
            t.start()

    def _worker_loop(self) -> None:
        """Core worker loop reading items from the thread-safe queue."""
        while not self._stop_event.is_set():
            self._pause_event.wait()
            try:
                item: BatchItem = self._queue.get(timeout=0.2)
            except queue.Empty:
                if self._queue.empty():
                    break
                continue

            if self._stop_event.is_set():
                item.status = ItemStatus.SKIPPED
                self._queue.task_done()
                break

            with self._lock:
                item.status = ItemStatus.PROCESSING
                item.started_at = time.time()

            try:
                res = self.worker_func(item.source_path)
                with self._lock:
                    item.result = res
                    item.status = ItemStatus.DONE
                    item.completed_at = time.time()
            except Exception as exc:
                with self._lock:
                    item.error_message = str(exc)
                    item.status = ItemStatus.FAILED
                    item.completed_at = time.time()
            finally:
                self._queue.task_done()

    def pause(self) -> None:
        """Pause worker processing."""
        self._pause_event.clear()

    def resume(self) -> None:
        """Resume worker processing."""
        self._pause_event.set()

    def cancel(self) -> None:
        """Cancel remaining queued items and stop workers."""
        self._stop_event.set()
        self._pause_event.set()
        # Empty queue
        while not self._queue.empty():
            try:
                item = self._queue.get_nowait()
                with self._lock:
                    item.status = ItemStatus.SKIPPED
                self._queue.task_done()
            except queue.Empty:
                break

    def get_progress(self) -> BatchProgress:
        """Retrieve real-time processing telemetry snapshot."""
        with self._lock:
            items = list(self._items.values())

        total = len(items)
        if total == 0:
            return BatchProgress()

        queued = sum(1 for i in items if i.status == ItemStatus.QUEUED)
        processing = sum(1 for i in items if i.status == ItemStatus.PROCESSING)
        completed = sum(1 for i in items if i.status == ItemStatus.DONE)
        failed = sum(1 for i in items if i.status == ItemStatus.FAILED)
        finished = completed + failed

        pct = round((finished / total) * 100.0, 1)

        rate = 0.0
        eta = 0.0
        if self._start_time and finished > 0:
            elapsed = time.time() - self._start_time
            rate = round(finished / max(0.01, elapsed), 2)
            remaining = total - finished
            eta = round(remaining / max(0.01, rate), 1)

        is_running = any(t.is_alive() for t in self._threads) and not self._queue.empty()

        return BatchProgress(
            total_items=total,
            queued=queued,
            processing=processing,
            completed=completed,
            failed=failed,
            percent=pct,
            items_per_sec=rate,
            eta_sec=eta,
            is_running=is_running,
            is_cancelled=self._stop_event.is_set(),
        )

    def get_items(self) -> List[BatchItem]:
        """Return shallow copy of all tracked batch items."""
        with self._lock:
            return list(self._items.values())

    def wait_completion(self, timeout: Optional[float] = None) -> bool:
        """Block until all items are completed or timeout expires."""
        for t in self._threads:
            t.join(timeout=timeout)
        return self._queue.empty()
