"""
tests/test_batch_processor.py
=============================
Unit tests for BatchProcessor async daemon worker (Section 3.2 & Section 8).
"""

import time
from pathlib import Path

import pytest

from src.dataset.batch_processor import BatchProcessor, ItemStatus


def _dummy_worker(path: Path) -> str:
    """Mock worker simulating task execution."""
    if "error" in path.name:
        raise ValueError("Simulated task error")
    time.sleep(0.05)
    return f"processed_{path.name}"


def test_batch_processor_queue_operations(tmp_path):
    proc = BatchProcessor(worker_func=_dummy_worker, max_workers=2)
    p1 = tmp_path / "file1.jpg"
    p2 = tmp_path / "file2.jpg"
    p1.touch()
    p2.touch()

    proc.add_items([p1, p2])
    prog_init = proc.get_progress()
    assert prog_init.total_items == 2
    assert prog_init.queued == 2

    proc.start()
    completed = proc.wait_completion(timeout=5.0)
    assert completed is True

    prog_done = proc.get_progress()
    assert prog_done.completed == 2
    assert prog_done.failed == 0
    assert prog_done.percent == 100.0


def test_batch_processor_error_isolation(tmp_path):
    proc = BatchProcessor(worker_func=_dummy_worker, max_workers=1)
    p_good = tmp_path / "good.jpg"
    p_bad = tmp_path / "error_file.jpg"
    p_good.touch()
    p_bad.touch()

    proc.add_item(p_good)
    proc.add_item(p_bad)
    proc.start()
    proc.wait_completion(timeout=5.0)

    items = proc.get_items()
    assert len(items) == 2

    good_item = next(i for i in items if i.source_path == p_good)
    bad_item = next(i for i in items if i.source_path == p_bad)

    assert good_item.status == ItemStatus.DONE
    assert bad_item.status == ItemStatus.FAILED
    assert "Simulated task error" in str(bad_item.error_message)


def test_batch_processor_cancellation(tmp_path):
    # Slow worker
    def _slow_worker(p: Path):
        time.sleep(0.2)
        return "ok"

    proc = BatchProcessor(worker_func=_slow_worker, max_workers=1)
    for i in range(5):
        p = tmp_path / f"file_{i}.jpg"
        p.touch()
        proc.add_item(p)

    proc.start()
    time.sleep(0.05)
    proc.cancel()
    proc.wait_completion(timeout=3.0)

    prog = proc.get_progress()
    assert prog.is_cancelled is True
    # At least some items should be SKIPPED
    items = proc.get_items()
    assert any(i.status == ItemStatus.SKIPPED for i in items)


def test_batch_processor_pause_and_resume(tmp_path):
    proc = BatchProcessor(worker_func=_dummy_worker, max_workers=1)
    p = tmp_path / "item.jpg"
    p.touch()
    proc.add_item(p)

    proc.pause()
    proc.start()
    time.sleep(0.1)

    # Item should still be queued while paused
    prog = proc.get_progress()
    assert prog.completed == 0

    proc.resume()
    proc.wait_completion(timeout=3.0)
    prog_resumed = proc.get_progress()
    assert prog_resumed.completed == 1
