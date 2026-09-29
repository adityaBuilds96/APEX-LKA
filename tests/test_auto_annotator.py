"""
tests/test_auto_annotator.py
============================
Unit tests for ClassicalAutoAnnotator (Section 4 & Section 8).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.dataset.auto_annotator import ClassicalAutoAnnotator


def _create_road_image(w: int = 640, h: int = 360, curved: bool = False) -> np.ndarray:
    """Helper to synthesize test road image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Road gray
    pts = np.array([[int(w * 0.4), int(h * 0.55)], [int(w * 0.6), int(h * 0.55)],
                    [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (90, 90, 90))

    if curved:
        # Curve points
        pts_left = np.array([[int(w * 0.45), int(h * 0.55)], [int(w * 0.35), int(h * 0.75)], [int(w * 0.15), h - 1]])
        pts_right = np.array([[int(w * 0.60), int(h * 0.55)], [int(w * 0.75), int(h * 0.75)], [int(w * 0.85), h - 1]])
        cv2.polylines(img, [pts_left], False, (255, 255, 255), 4)
        cv2.polylines(img, [pts_right], False, (255, 255, 255), 4)
    else:
        # Straight lane markings
        cv2.line(img, (int(w * 0.42), int(h * 0.55)), (int(w * 0.18), h - 1), (255, 255, 255), 4)
        cv2.line(img, (int(w * 0.58), int(h * 0.55)), (int(w * 0.82), h - 1), (255, 255, 255), 4)

    return img


def test_auto_annotator_straight_road(tmp_path):
    img = _create_road_image(curved=False)
    p = tmp_path / "straight.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, meta = annotator.generate_pseudo_mask(p)

    assert mask.shape == (360, 640)
    assert 1 in meta.classes_present  # Road surface
    assert meta.road_coverage > 0.05


def test_auto_annotator_curved_road(tmp_path):
    img = _create_road_image(curved=True)
    p = tmp_path / "curved.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, meta = annotator.generate_pseudo_mask(p)

    assert mask.shape == (360, 640)
    assert set(meta.classes_present).issubset({0, 1, 2, 3})


def test_lane_marking_class_ids(tmp_path):
    img = _create_road_image()
    p = tmp_path / "lanes.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, meta = annotator.generate_pseudo_mask(p)

    unique_vals = set(np.unique(mask).tolist())
    assert unique_vals.issubset({0, 1, 2, 3})


def test_mask_mutual_exclusivity(tmp_path):
    img = _create_road_image()
    p = tmp_path / "exclusivity.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, _ = annotator.generate_pseudo_mask(p)

    # In a 2D uint8 mask, each pixel can only have exactly one scalar value
    assert mask.dtype == np.uint8
    assert len(mask.shape) == 2


def test_mask_dimensions_match_input(tmp_path):
    # Test non-standard resolution 800x450
    img = _create_road_image(w=800, h=450)
    p = tmp_path / "res_800.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, meta = annotator.generate_pseudo_mask(p)

    assert mask.shape == (450, 800)
    assert meta.mask_name == "res_800.png"


def test_confidence_scoring_bounds(tmp_path):
    img = _create_road_image()
    p = tmp_path / "conf.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    _, meta = annotator.generate_pseudo_mask(p)

    assert 0.0 <= meta.confidence <= 1.0
    assert meta.confidence_tier in {"HIGH", "MEDIUM", "LOW"}


def test_confidence_tier_thresholds():
    annotator = ClassicalAutoAnnotator()
    # Test formula directly with helper
    road_mask = np.ones((360, 640), dtype=np.uint8) * 255
    conf_high, tier_high, _, _ = annotator._compute_confidence(road_mask, True, True, 200.0, 360, 640)
    assert conf_high >= 0.70
    assert tier_high == "HIGH"

    conf_low, tier_low, _, _ = annotator._compute_confidence(np.zeros((360, 640), dtype=np.uint8), False, False, None, 360, 640)
    assert conf_low < 0.40
    assert tier_low == "LOW"


def test_road_only_graceful_handling(tmp_path):
    # Image with road but zero lane lines
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    pts = np.array([[200, 150], [440, 150], [639, 359], [0, 359]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    p = tmp_path / "no_lanes.jpg"
    cv2.imwrite(str(p), img)

    annotator = ClassicalAutoAnnotator()
    mask, meta = annotator.generate_pseudo_mask(p)

    assert mask.shape == (360, 640)
    assert 1 in meta.classes_present
    assert meta.confidence < 0.70  # Low or medium due to missing lane lines


def test_auto_annotator_batch_progress_callback(tmp_path):
    img = _create_road_image()
    p1 = tmp_path / "img1.jpg"
    p2 = tmp_path / "img2.jpg"
    cv2.imwrite(str(p1), img)
    cv2.imwrite(str(p2), img)

    events = []
    def _cb(step, total, msg):
        events.append((step, total, msg))

    annotator = ClassicalAutoAnnotator()
    metas = annotator.annotate_batch([p1, p2], output_dir=tmp_path / "masks", progress_callback=_cb)

    assert len(metas) == 2
    assert len(events) >= 2
