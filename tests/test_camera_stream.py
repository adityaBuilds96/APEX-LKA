"""
tests/test_camera_stream.py
===========================
Unit tests for the real-time decoupled camera buffer and stream mechanics.
"""

import time
import numpy as np
import pytest

from dashboard.camera_stream import LatestFrameBuffer, PerformanceStats


def test_latest_frame_buffer_basic():
    buf = LatestFrameBuffer()
    assert buf.dropped_frames == 0

    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    buf.put(frame1, frame_id=1, timestamp=time.time())

    ret = buf.get(timeout=0.2)
    assert ret is not None
    f, fid, ts = ret
    assert fid == 1
    assert f.shape == (100, 100, 3)
    assert buf.dropped_frames == 0


def test_latest_frame_buffer_drops_stale_frames():
    buf = LatestFrameBuffer()

    # Put 5 frames rapidly without reading
    for i in range(1, 6):
        f = np.full((10, 10, 3), i, dtype=np.uint8)
        buf.put(f, frame_id=i, timestamp=time.time())

    # 4 frames were overwritten/dropped
    assert buf.dropped_frames == 4

    # Read should return the latest frame (frame 5)
    ret = buf.get(timeout=0.1)
    assert ret is not None
    f, fid, ts = ret
    assert fid == 5
    assert f[0, 0, 0] == 5


def test_latest_frame_buffer_timeout():
    buf = LatestFrameBuffer()
    t0 = time.perf_counter()
    ret = buf.get(timeout=0.05)
    dt = time.perf_counter() - t0
    assert ret is None
    assert dt >= 0.04


def test_latest_frame_buffer_clear():
    buf = LatestFrameBuffer()
    f = np.zeros((10, 10, 3), dtype=np.uint8)
    buf.put(f, frame_id=1, timestamp=time.time())
    buf.clear()
    assert buf.get(timeout=0.05) is None
