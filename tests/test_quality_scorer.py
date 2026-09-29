"""
tests/test_quality_scorer.py
============================
Unit tests for QualityScorer (Section 7 & Section 8).
"""

import cv2
import numpy as np
import pytest

from src.dataset.quality_scorer import CheckSeverity, QualityScorer, QualityTier


def _make_sample_pair(w: int = 640, h: int = 360) -> tuple[np.ndarray, np.ndarray]:
    """Create a passing synthetic image and mask pair."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)

    # Road polygon in lower 65%
    pts = np.array([[200, 150], [440, 150], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (80, 80, 80))
    cv2.fillPoly(mask, [pts], 1)

    # Left lane (2) and right lane (3)
    cv2.line(img, (220, 150), (120, h - 1), (255, 255, 255), 4)
    cv2.line(img, (420, 150), (520, h - 1), (255, 255, 255), 4)
    cv2.line(mask, (220, 150), (120, h - 1), 2, 8)
    cv2.line(mask, (420, 150), (520, h - 1), 3, 8)

    return img, mask


def test_quality_scorer_passing_pair():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    report = scorer.evaluate_pair(img, mask)

    assert report.score >= 0.85
    assert report.tier == QualityTier.EXCELLENT
    assert report.passed_all_critical is True


def test_dimension_mismatch_detection():
    scorer = QualityScorer()
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    mask = np.zeros((720, 1280), dtype=np.uint8)

    report = scorer.evaluate_pair(img, mask)
    c1 = next(c for c in report.checks if c.name == "Dimension Match")
    assert c1.passed is False
    assert c1.severity == CheckSeverity.CRITICAL


def test_illegal_class_id_detection():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    mask[100:110, 100:110] = 5  # Illegal class ID

    report = scorer.evaluate_pair(img, mask)
    c2 = next(c for c in report.checks if c.name == "Valid Class IDs")
    assert c2.passed is False
    assert c2.severity == CheckSeverity.CRITICAL


def test_road_coverage_ratio_flagging():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    # Fill 100% of image with road -> abnormal coverage
    mask[:] = 1

    report = scorer.evaluate_pair(img, mask)
    c3 = next(c for c in report.checks if c.name == "Road Coverage Ratio")
    assert c3.passed is False


def test_lane_spatial_consistency_crossed_lanes():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    # Invert bottom lanes: Left lane on right (x=500), Right lane on left (x=100)
    h, w = mask.shape[:2]
    mask[h - 1, :] = 0
    mask[h - 1, 500] = 2
    mask[h - 1, 100] = 3

    report = scorer.evaluate_pair(img, mask)
    c5 = next(c for c in report.checks if c.name == "Lane Spatial Consistency")
    assert c5.passed is False


def test_lane_width_plausibility():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    # Set lane width extremely narrow (10px apart at bottom)
    h, w = mask.shape[:2]
    mask[h - 1, :] = 0
    mask[h - 1, 310] = 2
    mask[h - 1, 320] = 3

    report = scorer.evaluate_pair(img, mask)
    c6 = next(c for c in report.checks if c.name == "Lane Width Plausibility")
    assert c6.passed is False


def test_lane_continuity_fragmentation():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    # Create 8 disconnected segments for left lane
    mask[mask == 2] = 0
    for y in range(160, 320, 20):
        mask[y:y+2, 200:202] = 2

    report = scorer.evaluate_pair(img, mask)
    c7 = next(c for c in report.checks if c.name == "Lane Continuity")
    assert c7.passed is False
    assert c7.severity == CheckSeverity.INFO


def test_sky_horizon_leak_detection():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    # Paint road/lane in top 20% (sky region)
    mask[10:50, 100:300] = 1

    report = scorer.evaluate_pair(img, mask)
    c9 = next(c for c in report.checks if c.name == "Sky / Horizon Leak")
    assert c9.passed is False


def test_quality_tier_classification():
    scorer = QualityScorer()
    # Completely empty/black mask
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    mask = np.zeros((360, 640), dtype=np.uint8)

    report = scorer.evaluate_pair(img, mask)
    assert report.score < 0.85
    assert report.tier in {QualityTier.NEEDS_REVIEW, QualityTier.REJECTED}


def test_scorer_to_dict():
    scorer = QualityScorer()
    img, mask = _make_sample_pair()
    report = scorer.evaluate_pair(img, mask)
    data = report.to_dict()
    assert "score" in data
    assert "tier" in data
    assert isinstance(data["checks"], list)
