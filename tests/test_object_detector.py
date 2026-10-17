"""
tests/test_object_detector.py
=============================
Unit tests for ObjectDetector engine, vectorized NMS, and detection decoding.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.perception.object_detector import (
    OBJECT_CLASSES,
    Detection,
    DetectionResult,
    ObjectDetector,
    vectorized_nms,
)


def test_detection_result_dataclass():
    """Verify DetectionResult methods and empty state."""
    empty_res = DetectionResult()
    assert empty_res.is_empty
    assert len(empty_res) == 0
    assert empty_res.get_detections() == []

    boxes = np.array([[10, 20, 100, 150]], dtype=np.float32)
    scores = np.array([0.88], dtype=np.float32)
    class_ids = np.array([1], dtype=np.int32)
    class_names = ["pedestrian"]

    res = DetectionResult(boxes=boxes, scores=scores, class_ids=class_ids, class_names=class_names)
    assert not res.is_empty
    assert len(res) == 1
    dets = res.get_detections()
    assert len(dets) == 1
    assert dets[0].class_name == "pedestrian"
    assert np.isclose(dets[0].score, 0.88, atol=1e-3)


def test_vectorized_class_aware_nms():
    """Verify vectorized NMS suppresses high-IoU boxes of same class, but preserves different classes."""
    # Two identical boxes for the SAME class (class 0: vehicle)
    boxes_same = np.array([
        [100.0, 100.0, 200.0, 200.0],
        [102.0, 102.0, 201.0, 201.0],
    ], dtype=np.float32)
    scores_same = np.array([0.90, 0.75], dtype=np.float32)
    classes_same = np.array([0, 0], dtype=np.int32)

    keep_same = vectorized_nms(boxes_same, scores_same, classes_same, iou_threshold=0.45)
    assert len(keep_same) == 1
    assert keep_same[0] == 0  # Highest score kept

    # Two identical boxes for DIFFERENT classes (class 0: vehicle, class 1: pedestrian)
    classes_diff = np.array([0, 1], dtype=np.int32)
    keep_diff = vectorized_nms(boxes_same, scores_same, classes_diff, iou_threshold=0.45)
    assert len(keep_diff) == 2  # Both preserved because classes differ


def test_decode_raw_detections():
    """Verify raw tensor decoding, coordinate denormalization, and score thresholding."""
    detector = ObjectDetector(model_path=None, conf_threshold=0.35, nms_threshold=0.45)

    # Synthetic raw candidate detections: shape (1, 3, 13)
    # Box 1: Pedestrian (conf 0.92, normalized [0.2, 0.3, 0.4, 0.7])
    # Box 2: Low-confidence box (conf 0.20, should be filtered)
    # Box 3: Vehicle (conf 0.85, normalized [0.5, 0.5, 0.8, 0.9])
    raw_det = np.zeros((1, 3, 13), dtype=np.float32)

    # Box 1
    raw_det[0, 0, :4] = [0.2, 0.3, 0.4, 0.7]
    raw_det[0, 0, 4 + 1] = 0.92  # class 1: pedestrian

    # Box 2 (below 0.35 conf threshold)
    raw_det[0, 1, :4] = [0.1, 0.1, 0.2, 0.2]
    raw_det[0, 1, 4 + 0] = 0.20

    # Box 3
    raw_det[0, 2, :4] = [0.5, 0.5, 0.8, 0.9]
    raw_det[0, 2, 4 + 0] = 0.85  # class 0: vehicle

    res = detector.decode_raw_detections(
        raw_det=raw_det,
        sx=1.0,
        sy=1.0,
        orig_w=640,
        orig_h=360,
    )

    assert len(res) == 2
    assert "pedestrian" in res.class_names
    assert "vehicle" in res.class_names

    # Check pixel denormalization
    # Box 1: x1 = 0.2 * 640 = 128, y1 = 0.3 * 360 = 108
    ped_idx = res.class_names.index("pedestrian")
    assert np.isclose(res.boxes[ped_idx][0], 128.0, atol=1.0)
    assert np.isclose(res.boxes[ped_idx][1], 108.0, atol=1.0)


def test_detector_fail_operational():
    """Verify detect() does not raise exceptions on None or corrupt frames."""
    detector = ObjectDetector(model_path=None)

    # None frame
    res_none = detector.detect(None)
    assert res_none.is_empty

    # Empty frame
    res_empty = detector.detect(np.empty((0, 0, 3), dtype=np.uint8))
    assert res_empty.is_empty

    # Valid blank frame in mock mode
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)
    res_mock = detector.detect(dummy_frame)
    assert isinstance(res_mock, DetectionResult)
