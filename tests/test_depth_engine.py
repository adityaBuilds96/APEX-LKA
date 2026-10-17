"""
tests/test_depth_engine.py
==========================
Unit test suite for MonocularDepthEngine:
- Validates model loading and fallback resilience
- Checks output dimensions and data types (Float32 [0.0, 1.0], metric meters)
- Tests scale-and-shift metric calibration: Z = s / (d + eps) + t
- Ensures numerical safety (division-by-zero, NaN, Inf, out-of-range bounds)
- Tests depth colormap visualization output
"""

import numpy as np
import pytest
import cv2

from src.perception.depth_engine import MonocularDepthEngine, DepthResult


def test_depth_engine_initialization():
    engine = MonocularDepthEngine(metric_scale=15.0, metric_shift=0.1)
    assert engine.is_loaded
    assert engine.metric_scale == 15.0
    assert engine.metric_shift == 0.1
    assert engine.min_depth_m > 0.0
    assert engine.max_depth_m > engine.min_depth_m


def test_depth_engine_synthetic_fallback():
    engine = MonocularDepthEngine(
        model_path="models/exported/non_existent_depth_model.onnx",
        metric_scale=15.0,
        metric_shift=0.1,
    )
    # Feed dummy 640x360 BGR image
    dummy_frame = np.full((360, 640, 3), 128, dtype=np.uint8)
    res = engine.estimate_depth(dummy_frame)

    assert isinstance(res, DepthResult)
    assert res.depth_map.shape == (360, 640)
    assert res.metric_depth_map.shape == (360, 640)
    assert res.depth_map.dtype == np.float32
    assert res.metric_depth_map.dtype == np.float32

    # Check bounds
    assert np.all(res.depth_map >= 0.0)
    assert np.all(res.depth_map <= 1.0)
    assert np.all(res.metric_depth_map >= engine.min_depth_m)
    assert np.all(res.metric_depth_map <= engine.max_depth_m)
    assert not np.any(np.isnan(res.depth_map))
    assert not np.any(np.isnan(res.metric_depth_map))
    assert not np.any(np.isinf(res.depth_map))
    assert not np.any(np.isinf(res.metric_depth_map))


def test_metric_calibration_math():
    engine = MonocularDepthEngine(metric_scale=15.0, metric_shift=0.1, min_depth_m=0.2, max_depth_m=80.0)
    
    # Test specific values
    # At d = 1.0 (very close): Z = 15.0 / (1.0 + 1e-6) + 0.1 ≈ 15.1
    d_test = np.array([1.0, 0.5, 0.1, 0.0], dtype=np.float32)
    calibrated = engine.apply_metric_calibration(d_test)

    expected_1_0 = (15.0 / (1.0 + 1e-6)) + 0.1
    assert np.isclose(calibrated[0], expected_1_0, atol=1e-2)

    expected_0_5 = (15.0 / (0.5 + 1e-6)) + 0.1
    assert np.isclose(calibrated[1], expected_0_5, atol=1e-2)

    # d = 0.0 should not blow up to inf because of eps and max_depth_m clamp
    assert not np.isinf(calibrated[3])
    assert calibrated[3] <= engine.max_depth_m


def test_numerical_safety_guards():
    engine = MonocularDepthEngine(metric_scale=15.0, metric_shift=0.1)

    # Extreme edge cases: negative, zero, NaN, Inf
    problematic_depth = np.array([-10.0, 0.0, np.nan, np.inf, -np.inf, 999.0], dtype=np.float32)
    sanitized = engine.apply_metric_calibration(problematic_depth)

    assert not np.any(np.isnan(sanitized))
    assert not np.any(np.isinf(sanitized))
    assert np.all(sanitized >= engine.min_depth_m)
    assert np.all(sanitized <= engine.max_depth_m)


def test_empty_frame_handling():
    engine = MonocularDepthEngine()
    empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
    res = engine.estimate_depth(empty_frame)

    assert res.depth_map.shape == (360, 640)
    assert res.metric_depth_map.shape == (360, 640)
    assert not np.any(np.isnan(res.metric_depth_map))


def test_depth_colorization():
    engine = MonocularDepthEngine()
    dummy_depth = np.linspace(0.0, 1.0, 360 * 640, dtype=np.float32).reshape((360, 640))
    colored = engine.colorize_depth(dummy_depth, colormap=cv2.COLORMAP_MAGMA)

    assert colored.shape == (360, 640, 3)
    assert colored.dtype == np.uint8
    # Test with empty
    empty_colored = engine.colorize_depth(np.empty((0, 0), dtype=np.float32))
    assert empty_colored.shape == (360, 640, 3)
