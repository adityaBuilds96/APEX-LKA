"""
tests/test_zero_lag_detection.py
================================
Unit and benchmark tests for CUDA Zero-Lag Object Detection Pipeline:
- Zero-copy temporal keyframe decimation
- Sub-3ms Lucas-Kanade optical flow box propagation
- BatchedNMS acceleration and score ordering
- Fallback resilience and MultiObjectTracker integration
"""

import time
import pytest
import numpy as np
import cv2

from src.perception.object_detector import (
    ObjectDetector,
    Detection,
    DetectionResult,
    batched_nms,
    vectorized_nms,
)
from src.perception.tracker import MultiObjectTracker, TrackState


def test_batched_nms_correctness():
    """Verify batched_nms produces equivalent suppression results to vectorized_nms."""
    boxes = np.array([
        [50.0, 50.0, 150.0, 150.0],
        [52.0, 51.0, 149.0, 152.0],  # Overlapping box, same class -> suppress
        [200.0, 200.0, 300.0, 300.0], # Distinct box -> keep
        [51.0, 51.0, 150.0, 150.0],  # Overlapping box, DIFFERENT class -> keep
    ], dtype=np.float32)
    scores = np.array([0.90, 0.70, 0.85, 0.95], dtype=np.float32)
    classes = np.array([0, 0, 0, 1], dtype=np.int32)

    keep_vec = vectorized_nms(boxes, scores, classes, iou_threshold=0.45)
    keep_batch = batched_nms(boxes, scores, classes, iou_threshold=0.45)

    assert set(keep_vec) == set(keep_batch)
    assert len(keep_batch) == 3


def test_keyframe_decimation_rhythm():
    """Verify detector runs full keyframe on frame T, and optical flow decimation on T+1, T+2."""
    detector = ObjectDetector(model_path=None, keyframe_interval=3, enable_keyframe_decimation=True)

    # Synthetic frame
    frame = np.full((360, 640, 3), 120, dtype=np.uint8)

    # Frame 0 (boot keyframe, but mock mode returns empty)
    res0 = detector.detect(frame, frame_id=0)
    assert res0.is_keyframe is True

    # Manually inject a known detection to simulate neural detection output
    mock_boxes = np.array([[100.0, 100.0, 200.0, 200.0]], dtype=np.float32)
    mock_res = DetectionResult(
        boxes=mock_boxes,
        scores=np.array([0.95], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
        is_keyframe=True,
    )
    detector._last_detections = mock_res
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector._last_gray_frame = gray
    detector._tracked_points, detector._point_box_ids = detector._extract_box_feature_points(mock_boxes, gray)
    detector._frame_count = 0

    # Frame 1 (T+1): Decimated optical flow frame
    res1 = detector.detect(frame, frame_id=1)
    assert res1.is_keyframe is False
    assert res1.decimated is True
    assert res1.tracking_method == "optical_flow"
    assert len(res1) == 1

    # Frame 2 (T+2): Decimated optical flow frame
    res2 = detector.detect(frame, frame_id=2)
    assert res2.is_keyframe is False
    assert res2.decimated is True
    assert len(res2) == 1

    # Frame 3 (T+3): Keyframe trigger
    res3 = detector.detect(frame, frame_id=3)
    assert res3.is_keyframe is True
    assert res3.decimated is False


def test_optical_flow_box_propagation_motion():
    """Verify moving pattern in frame translates tracked box accurately across decimated frames."""
    detector = ObjectDetector(model_path=None, keyframe_interval=3, enable_keyframe_decimation=True)

    # Create Frame 0 with a bright square at (100, 100) -> (200, 200)
    frame0 = np.full((360, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame0, (100, 100), (200, 200), (220, 220, 220), -1)
    # Add high-contrast texture inside square for optical flow
    for i in range(120, 180, 15):
        cv2.circle(frame0, (i, i), 4, (0, 0, 0), -1)

    mock_boxes = np.array([[95.0, 95.0, 205.0, 205.0]], dtype=np.float32)
    detector._last_detections = DetectionResult(
        boxes=mock_boxes,
        scores=np.array([0.90], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
        is_keyframe=True,
    )
    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
    detector._last_gray_frame = gray0
    detector._tracked_points, detector._point_box_ids = detector._extract_box_feature_points(mock_boxes, gray0)
    detector._frame_count = 0

    # Frame 1: Shift the texture by (+10, +5) pixels
    frame1 = np.full((360, 640, 3), 50, dtype=np.uint8)
    cv2.rectangle(frame1, (110, 105), (210, 205), (220, 220, 220), -1)
    for i in range(120, 180, 15):
        cv2.circle(frame1, (i + 10, i + 5), 4, (0, 0, 0), -1)

    res1 = detector.detect(frame1, frame_id=1)
    assert not res1.is_keyframe
    assert len(res1) == 1

    # Box should have shifted towards (+10, +5)
    dx = res1.boxes[0][0] - mock_boxes[0][0]
    dy = res1.boxes[0][1] - mock_boxes[0][1]
    assert dx > 4.0  # Positive translation tracked
    assert dy > 2.0


def test_decimated_frame_latency_budget():
    """Verify optical flow intermediate frame executes in under 3.0 ms."""
    detector = ObjectDetector(model_path=None, keyframe_interval=3, enable_keyframe_decimation=True)
    frame = np.full((360, 640, 3), 80, dtype=np.uint8)
    cv2.rectangle(frame, (150, 100), (250, 200), (200, 200, 200), -1)

    mock_boxes = np.array([[150.0, 100.0, 250.0, 200.0]], dtype=np.float32)
    detector._last_detections = DetectionResult(
        boxes=mock_boxes,
        scores=np.array([0.92], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector._last_gray_frame = gray
    detector._tracked_points, detector._point_box_ids = detector._extract_box_feature_points(mock_boxes, gray)
    detector._frame_count = 0

    # Benchmark 20 intermediate frames
    latencies = []
    for _ in range(20):
        t0 = time.perf_counter()
        res = detector.detect(frame, frame_id=1)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    median_lat = float(np.median(latencies))
    assert median_lat < 3.0, f"Expected < 3.0 ms, got {median_lat:.2f} ms"


def test_force_keyframe_and_reset():
    """Verify force_keyframe overrides decimation schedule and reset() clears tracking cache."""
    detector = ObjectDetector(model_path=None, keyframe_interval=10)
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)

    # Seed detector
    detector._last_detections = DetectionResult(
        boxes=np.array([[50, 50, 100, 100]], dtype=np.float32),
        scores=np.array([0.9], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
    )
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector._last_gray_frame = gray
    detector._tracked_points, detector._point_box_ids = detector._extract_box_feature_points(
        detector._last_detections.boxes, gray
    )

    # Frame 1 with force_keyframe=True
    res_forced = detector.detect(frame, frame_id=1, force_keyframe=True)
    assert res_forced.is_keyframe is True

    # Reset
    detector.reset()
    assert detector._frame_count == 0
    assert detector._last_detections is None
    assert detector._last_gray_frame is None
    assert detector._tracked_points is None


def test_tracker_integration_with_decimated_detections():
    """Verify MultiObjectTracker maintains track IDs across decimated frames."""
    tracker = MultiObjectTracker()

    # Frame 0: Keyframe detection
    det0 = DetectionResult(
        boxes=np.array([[200.0, 150.0, 300.0, 250.0]], dtype=np.float32),
        scores=np.array([0.90], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
        is_keyframe=True,
    )
    tracks0 = tracker.update(det0, dt=0.033)
    assert len(tracks0) == 1
    track_id = tracks0[0].track_id

    # Frame 1: Decimated detection (slightly shifted)
    det1 = DetectionResult(
        boxes=np.array([[202.0, 151.0, 302.0, 251.0]], dtype=np.float32),
        scores=np.array([0.88], dtype=np.float32),
        class_ids=np.array([0], dtype=np.int32),
        class_names=["vehicle"],
        is_keyframe=False,
        decimated=True,
        tracking_method="optical_flow",
    )
    tracks1 = tracker.update(det1, dt=0.033)
    assert len(tracks1) == 1
    assert tracks1[0].track_id == track_id  # ID maintained
