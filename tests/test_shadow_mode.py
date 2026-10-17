"""
tests/test_shadow_mode.py
=========================
Unit tests for Silent Shadow-Mode Execution Engine:
- Parallel candidate inference on background thread
- Primary control latency overhead <= 1.0 ms benchmark
- Metric divergence: Mask mIoU, Lateral Offset Delta, Discrepancy score
- Divergence CSV logging
- Auto-promotion rule when candidate model outperforms primary on hard scenes
"""

import csv
import time
from pathlib import Path
import numpy as np
import pytest

from src.fleet.shadow_mode import (
    ShadowModeEngine,
    ShadowDivergenceReport,
    PromotionStatus,
    compute_mask_miou,
)
from src.fleet.ota_updater import MockPredictor


@pytest.fixture
def shadow_engine_fixture(tmp_path):
    """Fixture providing isolated shadow engine and mock predictors."""
    log_csv = tmp_path / "logs" / "test_shadow_benchmarks.csv"
    shadow_pred = MockPredictor(name="candidate_shadow", version="2.0.0")
    engine = ShadowModeEngine(
        shadow_predictor=shadow_pred,
        divergence_threshold_m=0.40,
        log_csv_path=log_csv,
        min_eval_frames=10,
    )
    return {
        "engine": engine,
        "log_csv": log_csv,
        "tmp_path": tmp_path,
        "shadow_pred": shadow_pred,
    }


def test_mask_miou_calculation():
    """Verify mean Intersection over Union calculation."""
    mask_a = np.zeros((100, 100), dtype=np.uint8)
    mask_b = np.zeros((100, 100), dtype=np.uint8)

    # Identical empty masks
    assert compute_mask_miou(mask_a, mask_b) == 1.0

    # Overlapping 50x50 region
    mask_a[:50, :50] = 255
    mask_b[:50, :50] = 255
    assert compute_mask_miou(mask_a, mask_b) == 1.0

    # Non-overlapping masks
    mask_c = np.zeros((100, 100), dtype=np.uint8)
    mask_c[50:, 50:] = 255
    assert compute_mask_miou(mask_a, mask_c) == 0.0

    # Partial overlap (2500 intersection, 5000 union -> 0.50 IoU)
    mask_d = np.zeros((100, 100), dtype=np.uint8)
    mask_d[:50, :100] = 255  # 5000 pixels
    # mask_a is 2500 pixels (:50, :50) -> intersection 2500, union 5000 -> 0.5
    assert abs(compute_mask_miou(mask_a, mask_d) - 0.5) < 1e-4


def test_shadow_step_latency_overhead_benchmark(shadow_engine_fixture):
    """
    CRITICAL HARD CONSTRAINT:
    Shadow model inference step MUST run asynchronously without increasing
    primary prediction latency beyond 1.0 ms.
    """
    engine = shadow_engine_fixture["engine"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Warm-up executor
    engine.step(dummy_frame, primary_prediction=None, primary_offset_m=0.0)
    time.sleep(0.01)

    # Measure dispatch latency over 20 iterations
    latencies_ms = []
    for i in range(20):
        t0 = time.perf_counter()
        rep = engine.step(
            dummy_frame,
            primary_prediction=None,
            primary_offset_m=0.10,
            primary_confidence=0.85,
            frame_idx=i,
        )
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000.0)

    avg_overhead_ms = float(np.mean(latencies_ms))
    max_overhead_ms = float(np.max(latencies_ms))

    assert avg_overhead_ms <= 1.0, f"Average dispatch overhead {avg_overhead_ms:.3f} ms exceeds 1.0 ms budget!"
    assert max_overhead_ms <= 3.0, f"Spike dispatch overhead {max_overhead_ms:.3f} ms is too high!"


def test_divergence_metrics_calculation(shadow_engine_fixture):
    """Verify divergence metrics and discrepancy score computation."""
    engine = shadow_engine_fixture["engine"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Primary offset: 0.10m, Shadow candidate offset: 0.15m -> delta: 0.05m
    report = engine.evaluate_divergence_synchronous(
        frame=dummy_frame,
        primary_offset_m=0.10,
        primary_confidence=0.90,
        primary_mask=np.ones((50, 50), dtype=np.uint8),
        frame_idx=1,
    )

    assert isinstance(report, ShadowDivergenceReport)
    assert abs(report.offset_delta_m - 0.05) < 1e-3
    assert report.discrepancy_score_u8 <= 50  # Low discrepancy
    assert report.agreement is True


def test_high_divergence_flagging(shadow_engine_fixture):
    """Verify high lateral discrepancy triggers divergence flag."""
    engine = shadow_engine_fixture["engine"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Large discrepancy: Primary offset: 0.80m, Shadow candidate offset: 0.15m -> delta: 0.65m > 0.40m
    report = engine.evaluate_divergence_synchronous(
        frame=dummy_frame,
        primary_offset_m=0.80,
        primary_confidence=0.90,
        frame_idx=2,
    )

    assert report.offset_delta_m > engine.divergence_threshold_m
    assert report.discrepancy_score_u8 >= 50
    assert report.agreement is False


def test_csv_telemetry_logging(shadow_engine_fixture):
    """Verify divergence report writes row to CSV file."""
    engine = shadow_engine_fixture["engine"]
    log_csv = shadow_engine_fixture["log_csv"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    report = engine.evaluate_divergence_synchronous(
        frame=dummy_frame,
        primary_offset_m=0.12,
        primary_confidence=0.88,
        frame_idx=10,
    )
    engine.log_divergence(report)

    assert log_csv.exists()
    with open(log_csv, "r", encoding="utf-8") as f:
        reader = list(csv.reader(f))
        assert len(reader) >= 2  # Header + 1 row
        assert reader[0][0] == "timestamp"
        assert reader[1][1] == "10"  # frame_idx


def test_candidate_auto_promotion_rule(shadow_engine_fixture):
    """
    Verify auto-promotion rule:
    If shadow model demonstrates > 15% higher average confidence on hard scenes,
    promotion_status becomes PROMOTION_RECOMMENDED.
    """
    engine = shadow_engine_fixture["engine"]
    dummy_frame = np.zeros((360, 640, 3), dtype=np.uint8)

    # Feed 12 frames where primary has low confidence (0.35, hard scene)
    # and shadow candidate has high confidence (0.92, +160% higher)
    for i in range(12):
        engine.evaluate_divergence_synchronous(
            frame=dummy_frame,
            primary_offset_m=0.15,
            primary_confidence=0.35,  # Hard scene
            frame_idx=i,
        )

    assert engine.promotion_status == PromotionStatus.PROMOTION_RECOMMENDED
    summary = engine.get_benchmark_summary()
    assert summary["promotion_status"] == "PROMOTION_RECOMMENDED"
    assert summary["total_eval_frames"] == 12
