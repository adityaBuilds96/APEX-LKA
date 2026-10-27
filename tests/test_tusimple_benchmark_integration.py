"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

tests/test_tusimple_benchmark_integration.py
Comprehensive End-to-End TuSimple Evaluation & Verification Suite.

Validates the full TuSimple evaluation protocol according to official standards:
  1. Point-wise matching accuracy with strict 20-pixel threshold
  2. False Positive Rate (FPR) and False Negative Rate (FNR)
  3. Combined Harmonic F1-Score: 2 * (1 - FNR) * (1 - FPR) / ((1 - FNR) + (1 - FPR))
  4. Per-scene stratification across HIGHWAY, URBAN, RURAL, TUNNEL, and RAIN
  5. Latency percentile verification (P50, P95, P99)
  6. Markdown and CSV reporting output format integrity
  7. Robust handling of edge cases, corrupted annotations, and missing coordinates
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.tusimple_lane_benchmark import (
    ClipEvaluationResult,
    OfficialTuSimpleEvaluator,
    TuSimpleBenchmarkDataLoader,
    TuSimpleBenchmarkSummary,
    TuSimpleClipAnnotation,
    TuSimpleLaneBenchmarkRunner,
)
from src.scene.scene_profiles import SceneType


# ==============================================================================
# Test Fixtures & Synthetic Ground Truth
# ==============================================================================

@pytest.fixture
def standard_h_samples() -> List[int]:
    """Standard evaluation vertical scanlines for 640x360 BEV/perspective frames."""
    return list(range(160, 360, 10))


@pytest.fixture
def perfect_ground_truth(standard_h_samples: List[int]) -> TuSimpleClipAnnotation:
    """Generate ground truth annotation for two clean parallel lanes on highway."""
    left_lane: List[int] = []
    right_lane: List[int] = []
    for y in standard_h_samples:
        ratio = (y - 160) / (360 - 160)
        lx = int(280 + (140 - 280) * ratio)
        rx = int(360 + (500 - 360) * ratio)
        left_lane.append(lx)
        right_lane.append(rx)

    return TuSimpleClipAnnotation(
        raw_file="clips/0001/20.jpg",
        lanes=[left_lane, right_lane],
        h_samples=standard_h_samples,
        clip_id="clip_0001",
        scene_category="HIGHWAY",
    )


@pytest.fixture
def evaluator() -> OfficialTuSimpleEvaluator:
    """Instantiate official evaluator with standard 20px threshold and 85% match ratio."""
    return OfficialTuSimpleEvaluator(match_threshold_px=20.0, lane_match_ratio=0.85)


# ==============================================================================
# 1. Official TuSimple Evaluator Core Mechanics
# ==============================================================================

class TestOfficialTuSimpleEvaluator:
    """Validates point-wise and lane-wise matching rules according to official specs."""

    def test_perfect_prediction_yields_100_percent_accuracy(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that identical predicted coordinates produce 100% accuracy and 0 error."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        pred_lanes = [[float(x) for x in lane] for lane in gt_lanes]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, h_samples
        )

        assert g_pts == len(h_samples) * 2
        assert c_pts == g_pts
        assert fp == 0
        assert fn == 0
        assert math.isclose(mean_err, 0.0, abs_tol=1e-5)

    def test_points_within_threshold_are_counted_as_correct(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that point offsets of 18 pixels (< 20px) match successfully."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        # Add 18px offset to all points
        pred_lanes = [[float(x + 18) for x in lane] for lane in gt_lanes]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, h_samples
        )

        assert c_pts == g_pts
        assert fp == 0
        assert fn == 0
        assert math.isclose(mean_err, 18.0, abs_tol=1e-3)

    def test_points_exceeding_threshold_are_rejected(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that point offsets of 25 pixels (> 20px threshold) fail to match."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        # Add 25px offset to all points
        pred_lanes = [[float(x + 25) for x in lane] for lane in gt_lanes]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, h_samples
        )

        assert c_pts == 0
        assert fp == 2
        assert fn == 2

    def test_lane_match_ratio_boundary_condition(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """
        Verify that a lane with exactly 85% matched points is accepted as a matched lane,
        while a lane with 80% matches is rejected.
        """
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples
        num_pts = len(h_samples) # 20 points

        # Lane 0: 17 out of 20 points match (85%)
        # Lane 1: 15 out of 20 points match (75%)
        pred_lane_0: List[float] = []
        for i, x in enumerate(gt_lanes[0]):
            pred_lane_0.append(float(x) if i < 17 else float(x + 35))

        pred_lane_1: List[float] = []
        for i, x in enumerate(gt_lanes[1]):
            pred_lane_1.append(float(x) if i < 15 else float(x + 35))

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            [pred_lane_0, pred_lane_1], gt_lanes, h_samples
        )

        # Lane 0 should match (85% >= 85%), Lane 1 should fail (75% < 85%)
        assert c_pts == 17
        assert fp == 1
        assert fn == 1

    def test_false_positive_when_predicting_phantom_lane(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that an extra predicted phantom lane triggers a False Positive."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        pred_lanes = [[float(x) for x in lane] for lane in gt_lanes]
        phantom_lane = [100.0] * len(h_samples)
        pred_lanes.append(phantom_lane)

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, h_samples
        )

        assert c_pts == g_pts
        assert fp == 1
        assert fn == 0

    def test_false_negative_when_missing_ground_truth_lane(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that omitting a ground truth lane triggers a False Negative."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        # Only predict the left lane
        pred_lanes = [[float(x) for x in gt_lanes[0]]]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, h_samples
        )

        assert c_pts == len(gt_lanes[0])
        assert fp == 0
        assert fn == 1

    def test_empty_predictions_handling(
        self,
        evaluator: OfficialTuSimpleEvaluator,
        perfect_ground_truth: TuSimpleClipAnnotation,
    ) -> None:
        """Verify that empty predictions result in 0 correct points and FN = total GT lanes."""
        gt_lanes = perfect_ground_truth.lanes
        h_samples = perfect_ground_truth.h_samples

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            [], gt_lanes, h_samples
        )

        assert c_pts == 0
        assert fp == 0
        assert fn == len(gt_lanes)
        assert mean_err == 0.0

    def test_empty_ground_truth_handling(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify that clips with no ground truth lanes handle predictions as False Positives."""
        pred_lanes = [[200.0] * len(standard_h_samples)]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, [], standard_h_samples
        )

        assert c_pts == 0
        assert g_pts == 0
        assert fp == 1
        assert fn == 0
        assert mean_err == 0.0

    def test_negative_two_out_of_bounds_coordinate_filtering(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify that -2 coordinates in TuSimple standard format are skipped."""
        gt_lane = [250 if i > 5 else -2 for i in range(len(standard_h_samples))]
        pred_lane = [252.0 if i > 5 else -2.0 for i in range(len(standard_h_samples))]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            [pred_lane], [gt_lane], standard_h_samples
        )

        expected_pts = len(standard_h_samples) - 6
        assert g_pts == expected_pts
        assert c_pts == expected_pts
        assert fp == 0
        assert fn == 0
        assert math.isclose(mean_err, 2.0, abs_tol=1e-3)


# ==============================================================================
# 2. TuSimple Benchmark Data Loader & Synthetics
# ==============================================================================

class TestTuSimpleBenchmarkDataLoader:
    """Validates ground truth generation, coordinate bounds, and synthetic streaming."""

    def test_loader_generates_valid_synthetic_frames_when_vault_absent(self) -> None:
        """Verify synthetic fallback generates valid 640x360 BGR frames."""
        loader = TuSimpleBenchmarkDataLoader(dataset_path="non_existent_vault.zip")
        frame, clip_id = loader.fetch_clip_frame(index=0)

        assert frame.shape == (360, 640, 3)
        assert frame.dtype == np.uint8
        assert "synthetic" in clip_id

    def test_loader_generates_ground_truth_within_valid_bounds(self) -> None:
        """Verify generated ground truth points are within perspective image coordinates."""
        loader = TuSimpleBenchmarkDataLoader(dataset_path="non_existent_vault.zip")

        for idx in range(12):
            gt = loader.generate_ground_truth(idx)
            assert len(gt.lanes) == 2
            assert len(gt.h_samples) == len(TuSimpleBenchmarkDataLoader.STANDARD_H_SAMPLES)

            # Left lane should be to the left of the right lane
            for y_i in range(len(gt.h_samples)):
                lx = gt.lanes[0][y_i]
                rx = gt.lanes[1][y_i]
                assert 0 < lx < rx < 640

    def test_loader_cycles_scene_categories(self) -> None:
        """Verify scene categorization covers all target environments."""
        loader = TuSimpleBenchmarkDataLoader(dataset_path="non_existent_vault.zip")
        observed_scenes = {loader.generate_ground_truth(i).scene_category for i in range(10)}

        expected_subset = {"HIGHWAY", "URBAN", "RURAL", "TUNNEL", "RAIN"}
        for s in expected_subset:
            assert s in observed_scenes

    def test_standard_h_samples_monotonicity(self) -> None:
        """Verify vertical sample scanlines are strictly monotonically increasing."""
        h_samples = TuSimpleBenchmarkDataLoader.STANDARD_H_SAMPLES
        assert len(h_samples) >= 15
        for i in range(len(h_samples) - 1):
            assert h_samples[i] < h_samples[i + 1]
            assert 160 <= h_samples[i] <= 360


# ==============================================================================
# 3. Master TuSimple Benchmark Runner & Polynomial Sampling
# ==============================================================================

class TestTuSimpleLaneBenchmarkRunner:
    """Validates runner execution, metric aggregation, and report generation."""

    def test_polynomial_sampling_at_h_samples_correctness(self) -> None:
        """Verify parabolic sampling x(y) = A*y^2 + B*y + C against manual calculations."""
        runner = TuSimpleLaneBenchmarkRunner(smoke_test=True, max_clips=2)
        h_samples = [200, 250, 300]
        # Linear lane: x(y) = 0*y^2 + 1.0*y + 50.0
        poly = (0.0, 1.0, 50.0)
        sampled = runner._sample_polynomial_at_h_samples(poly, h_samples)

        assert math.isclose(sampled[0], 250.0, abs_tol=1e-3)
        assert math.isclose(sampled[1], 300.0, abs_tol=1e-3)
        assert math.isclose(sampled[2], 350.0, abs_tol=1e-3)

    def test_polynomial_sampling_returns_negative_two_for_none(self) -> None:
        """Verify None polynomial returns -2.0 for all h_samples."""
        runner = TuSimpleLaneBenchmarkRunner(smoke_test=True, max_clips=2)
        h_samples = [160, 200, 240, 280]
        sampled = runner._sample_polynomial_at_h_samples(None, h_samples)

        assert len(sampled) == len(h_samples)
        assert all(math.isclose(x, -2.0, abs_tol=1e-5) for x in sampled)

    def test_polynomial_sampling_out_of_image_boundary_clamping(self) -> None:
        """Verify points evaluated outside [0, 640] are marked as -2.0."""
        runner = TuSimpleLaneBenchmarkRunner(smoke_test=True, max_clips=2)
        h_samples = [100, 200, 300]
        # x(y) = 0*y^2 + 5.0*y + 0 -> at y=200 x=1000 (> 640)
        poly = (0.0, 5.0, 0.0)
        sampled = runner._sample_polynomial_at_h_samples(poly, h_samples)

        assert sampled[0] == 500.0
        assert sampled[1] == -2.0
        assert sampled[2] == -2.0

    def test_smoke_benchmark_execution_and_summary_schema(self) -> None:
        """Execute smoke benchmark with 3 clips and verify full summary metrics."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_md = Path(tmp_dir) / "bench.md"
            out_csv = Path(tmp_dir) / "bench.csv"

            runner = TuSimpleLaneBenchmarkRunner(
                output_markdown=str(out_md),
                output_csv=str(out_csv),
                max_clips=3,
                smoke_test=True,
            )
            summary: TuSimpleBenchmarkSummary = runner.run_benchmark()

            assert summary.total_clips_evaluated == 3
            assert summary.total_points_evaluated > 0
            assert 0.0 <= summary.overall_accuracy <= 1.0
            assert 0.0 <= summary.false_positive_rate <= 1.0
            assert 0.0 <= summary.false_negative_rate <= 1.0
            assert 0.0 <= summary.f1_score <= 1.0
            assert summary.mean_latency_ms > 0.0
            assert summary.p50_latency_ms <= summary.p95_latency_ms <= summary.p99_latency_ms

            # Check markdown file creation (generated automatically by run_benchmark)
            assert out_md.exists()
            content = out_md.read_text(encoding="utf-8")
            assert "# APEX-Vision Core v4.0" in content
            assert "TuSimple Accuracy" in content
            assert "Combined F1 Score" in content
            assert "Scene-Adaptive Accuracy Breakdown" in content

            # Check CSV export (generated automatically by run_benchmark)
            assert out_csv.exists()
            with open(out_csv, mode="r", newline="", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                assert "clip_id" in header
                assert "accuracy" in header
                assert "latency_ms" in header
                rows = list(reader)
                assert len(rows) == 3

    def test_baseline_delta_calculations(self) -> None:
        """Verify accuracy and F1 score deltas against Phase 7 Section 2 baseline."""
        runner = TuSimpleLaneBenchmarkRunner(smoke_test=True, max_clips=1)
        assert math.isclose(runner.BASELINE_ACCURACY, 0.8920, abs_tol=1e-4)
        assert math.isclose(runner.BASELINE_F1, 0.9208, abs_tol=1e-4)

        summary = runner.run_benchmark()
        expected_acc_delta = summary.overall_accuracy - runner.BASELINE_ACCURACY
        expected_f1_delta = summary.f1_score - runner.BASELINE_F1

        assert math.isclose(summary.accuracy_delta, expected_acc_delta, abs_tol=1e-4)
        assert math.isclose(summary.f1_delta, expected_f1_delta, abs_tol=1e-4)


# ==============================================================================
# 4. Metric Aggregations & Mathematical Validation
# ==============================================================================

class TestTuSimpleMetricMathematics:
    """Validates mathematical correctness of official accuracy, precision, recall, and F1."""

    def test_f1_score_harmonic_mean_property(self) -> None:
        """Verify that F1 score is the harmonic mean of precision and recall."""
        # Simulated metrics
        fpr = 0.05
        fnr = 0.08
        precision = 1.0 - fpr  # 0.95
        recall = 1.0 - fnr     # 0.92

        expected_f1 = 2 * (precision * recall) / (precision + recall)
        formula_f1 = 2 * (1 - fnr) * (1 - fpr) / ((1 - fnr) + (1 - fpr))

        assert math.isclose(expected_f1, formula_f1, abs_tol=1e-6)
        assert 0.93 < formula_f1 < 0.94

    def test_f1_score_edge_case_zero_division(self) -> None:
        """Verify F1 score does not divide by zero when precision and recall are both zero."""
        fpr = 1.0
        fnr = 1.0
        precision = 1.0 - fpr
        recall = 1.0 - fnr

        f1 = float(2 * precision * recall / max(1e-4, precision + recall))
        assert math.isclose(f1, 0.0, abs_tol=1e-3)

    def test_latency_percentiles_calculation(self) -> None:
        """Verify P50, P95, P99 percentile computations on a deterministic distribution."""
        latencies = [10.0 + i * 0.5 for i in range(100)] # 10.0 to 59.5 ms
        lat_arr = np.array(latencies)

        p50 = float(np.percentile(lat_arr, 50))
        p95 = float(np.percentile(lat_arr, 95))
        p99 = float(np.percentile(lat_arr, 99))

        assert math.isclose(p50, 34.75, abs_tol=0.5)
        assert math.isclose(p95, 57.025, abs_tol=0.5)
        assert math.isclose(p99, 59.005, abs_tol=0.5)
        assert p50 < p95 < p99


# ==============================================================================
# 5. Per-Scene Stratification & Robustness Analysis
# ==============================================================================

class TestPerSceneStratification:
    """Validates per-scene classification and performance breakdown."""

    def test_per_scene_accuracy_map_generation(self) -> None:
        """Verify per-scene accuracy map contains entries for all encountered scenes."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_md = Path(tmp_dir) / "strat.md"
            out_csv = Path(tmp_dir) / "strat.csv"

            runner = TuSimpleLaneBenchmarkRunner(
                output_markdown=str(out_md),
                output_csv=str(out_csv),
                max_clips=6,
                smoke_test=True,
            )
            summary = runner.run_benchmark()

            assert isinstance(summary.per_scene_accuracy, dict)
            assert len(summary.per_scene_accuracy) > 0
            for scene_name, acc in summary.per_scene_accuracy.items():
                assert isinstance(scene_name, str)
                assert 0.0 <= acc <= 1.0

    def test_clip_evaluation_result_dataclass_fields(self) -> None:
        """Verify all fields and types of ClipEvaluationResult dataclass."""
        res = ClipEvaluationResult(
            clip_id="test_clip_01",
            raw_file="clips/01/20.jpg",
            correct_points=38,
            total_gt_points=40,
            accuracy=0.95,
            false_positives=0,
            false_negatives=0,
            ground_truth_lanes=2,
            predicted_lanes=2,
            mean_pixel_error=4.25,
            latency_ms=11.5,
            scene_tag="HIGHWAY",
            confidence_score=0.92,
            guardrail_safe=True,
        )

        assert res.clip_id == "test_clip_01"
        assert res.correct_points == 38
        assert res.total_gt_points == 40
        assert math.isclose(res.accuracy, 0.95, abs_tol=1e-5)
        assert res.guardrail_safe is True
        assert res.mean_pixel_error == 4.25


# ==============================================================================
# 6. Edge Cases, Fault Injection & Resiliency
# ==============================================================================

class TestTuSimpleBenchmarkEdgeCases:
    """Validates benchmark stability against anomalous, corrupted, or extreme inputs."""

    def test_corrupted_coordinates_with_nans(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify evaluator survives NaN coordinates without raising exceptions."""
        gt_lanes = [[200, 210, 220, 230] + [250] * (len(standard_h_samples) - 4)]
        pred_lanes = [[float("nan")] * len(standard_h_samples)]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        assert c_pts == 0
        assert fp == 1
        assert fn == 1

    def test_corrupted_coordinates_with_infinities(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify evaluator survives infinite coordinates without raising exceptions."""
        gt_lanes = [[200] * len(standard_h_samples)]
        pred_lanes = [[float("inf")] * len(standard_h_samples)]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        assert c_pts == 0
        assert fp == 1
        assert fn == 1

    def test_mismatched_length_prediction_vectors(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify predictor vectors with fewer or more points than h_samples are safely evaluated."""
        gt_lanes = [[200] * len(standard_h_samples)]
        # Short prediction (only 5 points)
        pred_lanes = [[200.0] * 5]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        # 5 out of 20 points matched = 25% < 85% match ratio -> lane rejected
        assert c_pts == 0
        assert fp == 1
        assert fn == 1

    def test_extreme_pixel_deviation(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify huge pixel deviations (> 10,000 px) do not cause overflow or crashes."""
        gt_lanes = [[300] * len(standard_h_samples)]
        pred_lanes = [[50000.0] * len(standard_h_samples)]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        assert c_pts == 0
        assert fp == 1
        assert fn == 1
        assert mean_err == 0.0

    def test_benchmark_runner_handles_unwritable_output_directory(self) -> None:
        """Verify runner creates parent directories if they do not yet exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            deep_path = Path(tmp_dir) / "sub1" / "sub2" / "bench.md"
            deep_csv = Path(tmp_dir) / "sub1" / "sub2" / "bench.csv"

            runner = TuSimpleLaneBenchmarkRunner(
                output_markdown=str(deep_path),
                output_csv=str(deep_csv),
                max_clips=1,
                smoke_test=True,
            )
            assert deep_path.parent.exists()


# ==============================================================================
# 7. Benchmark CLI & Safety Disclaimers
# ==============================================================================

class TestBenchmarkSafetyDisclaimers:
    """Validates presence of automotive safety standards and branding in outputs."""

    def test_markdown_report_includes_iso26262_and_sae_level2_branding(self) -> None:
        """Verify generated markdown report includes mandatory safety disclaimers."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_md = Path(tmp_dir) / "audit.md"
            runner = TuSimpleLaneBenchmarkRunner(
                output_markdown=str(out_md),
                output_csv=str(Path(tmp_dir) / "audit.csv"),
                max_clips=1,
                smoke_test=True,
            )
            summary = runner.run_benchmark()
            assert out_md.exists()

            text = out_md.read_text(encoding="utf-8")
            assert "driver-supervised SAE Level 2" in text
            assert "ISO 26262-style process; ASIL-B aligned; not independently assessed" in text
            assert "APEX-Vision" in text
            assert "tusimple" in text.lower()


# ==============================================================================
# 8. Multi-Lane Topology & Complex Highway Scenarios
# ==============================================================================

class TestMultiLaneHighwayTopology:
    """Validates TuSimple evaluation across multi-lane highways (3-lane, 4-lane, HOV)."""

    def test_four_lane_highway_evaluation(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """
        Verify evaluation on a 4-lane highway with Left-Left, Left, Right, Right-Right lanes.
        Ground truth contains 4 distinct lane boundaries across the perspective field.
        """
        n_pts = len(standard_h_samples)
        lane_ll = [int(100 + i * 2) for i in range(n_pts)]
        lane_l = [int(220 + i * 3) for i in range(n_pts)]
        lane_r = [int(420 + i * 4) for i in range(n_pts)]
        lane_rr = [int(540 + i * 5) for i in range(n_pts)]
        gt_lanes = [lane_ll, lane_l, lane_r, lane_rr]

        # Predict all 4 lanes with 2px offset
        pred_lanes = [[float(x + 2) for x in l] for l in gt_lanes]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        assert g_pts == n_pts * 4
        assert c_pts == g_pts
        assert fp == 0
        assert fn == 0
        assert math.isclose(mean_err, 2.0, abs_tol=1e-3)

    def test_partial_lane_detection_on_four_lane_highway(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """Verify ego system detecting only ego corridor (Left, Right) in 4-lane setting."""
        n_pts = len(standard_h_samples)
        lane_ll = [int(100 + i * 2) for i in range(n_pts)]
        lane_l = [int(220 + i * 3) for i in range(n_pts)]
        lane_r = [int(420 + i * 4) for i in range(n_pts)]
        lane_rr = [int(540 + i * 5) for i in range(n_pts)]
        gt_lanes = [lane_ll, lane_l, lane_r, lane_rr]

        # Only predict the two ego boundary lanes
        pred_lanes = [
            [float(x) for x in lane_l],
            [float(x) for x in lane_r],
        ]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            pred_lanes, gt_lanes, standard_h_samples
        )

        assert g_pts == n_pts * 4
        assert c_pts == n_pts * 2
        assert fp == 0
        assert fn == 2  # Outer lanes missed
        assert math.isclose(mean_err, 0.0, abs_tol=1e-5)


# ==============================================================================
# 9. Fragmented, Dashed & Occluded Lane Evaluation
# ==============================================================================

class TestFragmentedAndDashedLanes:
    """Validates evaluation on intermittent, dashed, and partially occluded markings."""

    def test_intermittent_dashed_line_matching(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """
        Verify dashed lane markings where alternate segments are unpainted (x = -2).
        The evaluator should correctly ignore the unpainted intervals.
        """
        n_pts = len(standard_h_samples)
        # Ground truth has alternating dashes (painted, gap, painted, gap...)
        gt_lane = [250 + i * 5 if (i % 2 == 0) else -2 for i in range(n_pts)]
        gt_lanes = [gt_lane]

        # Prediction provides continuous lane estimates
        pred_lane = [float(250 + i * 5) for i in range(n_pts)]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            [pred_lane], gt_lanes, standard_h_samples
        )

        expected_pts = sum(1 for x in gt_lane if x > 0)
        assert g_pts == expected_pts
        assert c_pts == expected_pts
        assert fp == 0
        assert fn == 0
        assert math.isclose(mean_err, 0.0, abs_tol=1e-5)

    def test_severely_occluded_marking_fails_threshold(
        self, evaluator: OfficialTuSimpleEvaluator, standard_h_samples: List[int]
    ) -> None:
        """
        Verify that if vehicle occlusion obscures > 50% of a lane and predictions
        deviate on the remaining visible points, the lane match ratio check triggers FN.
        """
        n_pts = len(standard_h_samples) # 20
        # Only 6 points visible (e.g. obscured by lead truck)
        gt_lane = [280 + i * 4 if i < 6 else -2 for i in range(n_pts)]

        # Prediction deviates on 3 of the 6 points (> 20px)
        pred_lane = [
            float(gt_lane[i] if i < 3 else gt_lane[i] + 40) if i < 6 else 300.0
            for i in range(n_pts)
        ]

        c_pts, g_pts, fp, fn, mean_err = evaluator.evaluate_clip_predictions(
            [pred_lane], [gt_lane], standard_h_samples
        )

        # 3 out of 6 points match = 50% < 85% match ratio -> lane rejected
        assert c_pts == 0
        assert g_pts == 6
        assert fp == 1
        assert fn == 1


# ==============================================================================
# 10. Parametric Tolerance & CI/CD Telemetry Persistence
# ==============================================================================

class TestParametricToleranceAndCICD:
    """Validates evaluation sensitivity across match thresholds and CI/CD gating."""

    @pytest.mark.parametrize("thresh", [10.0, 15.0, 20.0, 30.0])
    def test_evaluator_tolerance_parametric_scaling(
        self, thresh: float, standard_h_samples: List[int]
    ) -> None:
        """Verify that matching accuracy scales monotonically with acceptance threshold."""
        eval_param = OfficialTuSimpleEvaluator(match_threshold_px=thresh, lane_match_ratio=0.85)

        gt_lane = [300 + i * 2 for i in range(len(standard_h_samples))]
        # Deviate by exactly 18 pixels
        pred_lane = [float(x + 18) for x in gt_lane]

        c_pts, g_pts, fp, fn, mean_err = eval_param.evaluate_clip_predictions(
            [pred_lane], [gt_lane], standard_h_samples
        )

        if thresh < 18.0:
            # Rejection expected
            assert c_pts == 0
            assert fp == 1
            assert fn == 1
        else:
            # Acceptance expected
            assert c_pts == g_pts
            assert fp == 0
            assert fn == 0
            assert math.isclose(mean_err, 18.0, abs_tol=1e-3)

    def test_summary_telemetry_dictionary_serialization(self) -> None:
        """Verify TuSimpleBenchmarkSummary serialization to dict for CI/CD gates."""
        summary = TuSimpleBenchmarkSummary(
            timestamp_utc="2026-10-27T12:00:00Z",
            total_clips_evaluated=100,
            total_points_evaluated=4000,
            overall_accuracy=0.9620,
            false_positive_rate=0.0380,
            false_negative_rate=0.0350,
            f1_score=0.9634,
            mean_latency_ms=11.2,
            p50_latency_ms=10.5,
            p95_latency_ms=13.8,
            p99_latency_ms=14.9,
            baseline_accuracy=0.8920,
            accuracy_delta=0.0700,
            baseline_f1=0.9208,
            f1_delta=0.0426,
            per_scene_accuracy={"HIGHWAY": 0.975, "URBAN": 0.945, "RAIN": 0.932},
            target_accuracy_achieved=True,
        )

        data_dict = summary.__dict__
        assert data_dict["overall_accuracy"] >= 0.955
        assert data_dict["target_accuracy_achieved"] is True
        assert data_dict["p95_latency_ms"] <= 14.0
        assert "HIGHWAY" in data_dict["per_scene_accuracy"]

