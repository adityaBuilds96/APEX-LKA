"""
tests/test_tracker.py
=====================
Unit tests for Task 3.1: Multi-Object Tracking Engine (ByteTrack / OC-SORT):
1. Track ID assignment & persistence across frames
2. Two-stage ByteTrack association (high-confidence matching + low-confidence recovery)
3. 3D Kalman Filter velocity vector estimation [vx, vy, vz]
4. Lost track buffer hold and culling
5. Tracking latency benchmark (< 2.0 ms execution)
"""

import sys
import time
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.object_detector import Detection, DetectionResult
from src.perception.projection_3d import Object3D
from src.perception.tracker import APEXMultiObjectTracker, TrackState, TrackedObject


def test_tracker_initialization():
    """Verify tracker initializes correctly from config or default parameters."""
    tracker = APEXMultiObjectTracker()
    assert tracker.track_thresh == 0.45
    assert tracker.track_buffer_frames == 30
    assert len(tracker.tracks) == 0


def test_track_id_persistence():
    """Verify persistent Track ID across consecutive video frames as an object moves."""
    tracker = APEXMultiObjectTracker(track_thresh=0.40)

    # Frame 1: Vehicle appears at [100, 150, 180, 230]
    d1 = Detection(box=np.array([100.0, 150.0, 180.0, 230.0]), score=0.90, class_id=0, class_name="vehicle")
    tracks_f1 = tracker.update([d1])
    assert len(tracks_f1) == 1
    t1_id = tracks_f1[0].track_id

    # Frame 2: Vehicle moves slightly to [104, 150, 184, 230]
    d2 = Detection(box=np.array([104.0, 150.0, 184.0, 230.0]), score=0.88, class_id=0, class_name="vehicle")
    tracks_f2 = tracker.update([d2])
    assert len(tracks_f2) == 1
    assert tracks_f2[0].track_id == t1_id
    assert tracks_f2[0].age_frames == 2

    # Frame 3: Vehicle continues moving to [108, 150, 188, 230]
    d3 = Detection(box=np.array([108.0, 150.0, 188.0, 230.0]), score=0.85, class_id=0, class_name="vehicle")
    tracks_f3 = tracker.update([d3])
    assert len(tracks_f3) == 1
    assert tracks_f3[0].track_id == t1_id
    assert tracks_f3[0].age_frames == 3


def test_two_stage_bytetrack_low_confidence_recovery():
    """Verify that a drop in detection confidence to low-conf (<0.45) still maintains the track ID."""
    tracker = APEXMultiObjectTracker(track_thresh=0.45)

    # Frame 1: High confidence detection
    d1 = Detection(box=np.array([200.0, 100.0, 250.0, 180.0]), score=0.85, class_id=1, class_name="pedestrian")
    tracks_1 = tracker.update([d1])
    assert len(tracks_1) == 1
    ped_id = tracks_1[0].track_id

    # Frame 2: Momentary partial occlusion causes confidence drop to 0.35 (< track_thresh=0.45)
    d2 = Detection(box=np.array([202.0, 100.0, 252.0, 180.0]), score=0.35, class_id=1, class_name="pedestrian")
    tracks_2 = tracker.update([d2])
    assert len(tracks_2) == 1
    # Must recover the same pedestrian ID in stage 2 matching
    assert tracks_2[0].track_id == ped_id


def test_3d_spatial_velocity_estimation():
    """Verify that Kalman filter accurately estimates 3D metric velocity vector [vx, vy, vz]."""
    tracker = APEXMultiObjectTracker()

    # Simulate an approaching vehicle with forward distance Z decreasing by 0.5m each 0.05s (vz = -10 m/s)
    for frame_i in range(10):
        z_curr = 20.0 - (frame_i * 0.5)
        box = np.array([250.0, 150.0, 350.0, 250.0], dtype=np.float32)
        det = Detection(box=box, score=0.92, class_id=0, class_name="vehicle")
        obj3d = Object3D(
            x=0.0,
            y=0.0,
            z=z_curr,
            width=1.8,
            height=1.5,
            length=4.2,
            class_id=0,
            class_name="vehicle",
            confidence=0.92,
            box_2d=box,
        )

        tracks = tracker.update([det], objects_3d=[obj3d], dt=0.05)
        assert len(tracks) == 1

    final_track = tracks[0]
    # Longitudinal velocity vz should be negative (approaching vehicle)
    vz = float(final_track.velocity_3d[2])
    assert vz < -3.0, f"Expected negative approaching velocity, got {vz}"
    assert final_track.speed_mps > 3.0


def test_tracker_latency_benchmark():
    """Verify that tracker updates in < 2.0 ms for 10 active tracks."""
    tracker = APEXMultiObjectTracker()

    # Pre-populate 10 tracks
    dets = [
        Detection(
            box=np.array([i * 50.0, 100.0, i * 50.0 + 40.0, 160.0]),
            score=0.80,
            class_id=0,
            class_name="vehicle",
        )
        for i in range(10)
    ]
    tracker.update(dets)

    # Benchmark next 30 frames
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        tracker.update(dets)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_latency = sum(times) / len(times)
    assert avg_latency < 2.0, f"Tracker latency too high: {avg_latency:.2f} ms"
