"""
tests/test_evaluate.py
======================
Comprehensive test suite for APEX-LKA evaluation engine:
- Metrics computation (IoU, Dice, F1, Accuracy, Confusion Matrix)
- Edge cases (missing classes, empty predictions, zero denominators)
- Latency profiling & FPS computation
- Visual prediction gallery generation (4-column panels)
- Failure case analysis & scenario categorization
- Report generation (JSON, Markdown, HTML)
- ModelEvaluator integration and fallback handling
"""

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from src.training.metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_frame_metrics,
    compute_latency_statistics,
)
from src.training.reports import (
    analyze_failures,
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
)
from src.training.visualization import (
    build_4col_panel,
    create_error_map,
    generate_prediction_galleries,
    mask_to_rgb,
    plot_confusion_matrix,
    plot_failure_distribution,
    plot_per_class_iou,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Objective 1 & 7: Metric Computation Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_iou_computation_perfect_prediction():
    """Verify IoU equals 1.0 when prediction perfectly matches ground truth."""
    h, w = 64, 64
    target = np.zeros((h, w), dtype=np.int64)
    target[10:30, :] = 1  # Road
    target[35:40, 10:20] = 2  # Left lane
    target[35:40, 44:54] = 3  # Right lane

    pred = target.copy()

    conf_mat = compute_confusion_matrix(pred, target, num_classes=4)
    metrics = compute_classification_metrics(conf_mat)

    assert metrics["mean_iou"] == 1.0, f"Expected 1.0, got {metrics['mean_iou']}"
    assert metrics["lane_mean_iou"] == 1.0
    assert metrics["overall_pixel_accuracy"] == 1.0
    for c_name in CLASS_NAMES:
        assert metrics["per_class"][c_name]["iou"] == 1.0


def test_dice_computation_correctness():
    """Verify Dice coefficient matches expected 2*TP / (2*TP + FP + FN)."""
    h, w = 10, 10
    target = np.zeros((h, w), dtype=np.int64)
    pred = np.zeros((h, w), dtype=np.int64)

    # Class 1: Target has 20 pixels, Pred has 20 pixels, overlap is 10 pixels
    target[:2, :] = 1  # 20 pixels
    pred[1:3, :] = 1   # 20 pixels -> Overlap in row 1: 10 pixels
    # TP = 10, FP = 10, FN = 10 -> Dice = 2*10 / (20 + 20) = 0.50, IoU = 10 / 30 = 0.3333

    conf_mat = compute_confusion_matrix(pred, target, num_classes=4)
    metrics = compute_classification_metrics(conf_mat)

    road_metrics = metrics["per_class"]["Road"]
    assert pytest.approx(road_metrics["dice"], rel=1e-2) == 0.50
    assert pytest.approx(road_metrics["iou"], rel=1e-2) == 0.3333
    assert pytest.approx(road_metrics["precision"], rel=1e-2) == 0.50
    assert pytest.approx(road_metrics["recall"], rel=1e-2) == 0.50


def test_confusion_matrix_sum():
    """Verify confusion matrix sum equals total evaluated pixels."""
    h, w = 120, 160
    total_pixels = h * w
    target = np.random.randint(0, 4, size=(h, w), dtype=np.int64)
    pred = np.random.randint(0, 4, size=(h, w), dtype=np.int64)

    conf_mat = compute_confusion_matrix(pred, target, num_classes=4)
    assert conf_mat.shape == (4, 4)
    assert conf_mat.sum() == total_pixels
    assert np.all(conf_mat >= 0)

    # Tensor input support
    t_target = torch.from_numpy(target)
    t_pred = torch.from_numpy(pred)
    conf_mat_torch = compute_confusion_matrix(t_pred, t_target, num_classes=4)
    assert np.array_equal(conf_mat, conf_mat_torch)


def test_handles_empty_predictions_gracefully():
    """Verify system handles all-zero predictions without division by zero errors."""
    h, w = 50, 50
    target = np.ones((h, w), dtype=np.int64)  # All road (class 1)
    pred = np.zeros((h, w), dtype=np.int64)   # All background (class 0)

    conf_mat = compute_confusion_matrix(pred, target, num_classes=4)
    metrics = compute_classification_metrics(conf_mat)

    # Class 1 should have 0 TP, 0 Precision, 0 Recall, 0 IoU
    road = metrics["per_class"]["Road"]
    assert road["iou"] == 0.0
    assert road["precision"] == 0.0
    assert road["recall"] == 0.0
    assert road["dice"] == 0.0
    assert isinstance(metrics["mean_iou"], float)


def test_handles_missing_classes_in_ground_truth():
    """Verify classes absent from both ground truth and prediction are scored as 1.0 agreement."""
    h, w = 32, 32
    # Only background and road are present
    target = np.zeros((h, w), dtype=np.int64)
    target[16:, :] = 1
    pred = target.copy()

    conf_mat = compute_confusion_matrix(pred, target, num_classes=4)
    metrics = compute_classification_metrics(conf_mat)

    # Classes 2 and 3 have 0 GT and 0 Pred -> Perfect absence agreement (1.0)
    assert metrics["per_class"]["Left Lane"]["iou"] == 1.0
    assert metrics["per_class"]["Right Lane"]["iou"] == 1.0
    assert metrics["mean_iou"] == 1.0


def test_latency_statistics_computation():
    """Verify latency profiling calculates percentiles and FPS correctly."""
    # Empty case
    empty_stats = compute_latency_statistics([])
    assert empty_stats["mean_ms"] == 0.0
    assert empty_stats["fps"] == 0.0

    # Constant 20ms latency
    latencies = [20.0] * 50
    stats = compute_latency_statistics(latencies)
    assert stats["mean_ms"] == 20.0
    assert stats["p50_ms"] == 20.0
    assert stats["p95_ms"] == 20.0
    assert stats["p99_ms"] == 20.0
    assert stats["fps"] == 50.0  # 1000 / 20 = 50 FPS


# ═══════════════════════════════════════════════════════════════════════════════
# Objective 3 & 7: Failure Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_failure_analysis_categorization_works():
    """Verify severe failures and lane failures are categorized by scenario."""
    records = [
        # Normal frame
        {"name": "f1", "frame_miou": 0.85, "frame_lane_iou": 0.80, "metadata": {"time_of_day": "day", "weather": "clear", "road_type": "painted", "curvature": "straight"}},
        # Severe failure (mIoU < 0.30)
        {"name": "f2", "frame_miou": 0.22, "frame_lane_iou": 0.10, "metadata": {"time_of_day": "night", "weather": "shadowed", "road_type": "unpainted", "curvature": "curved"}},
        # Lane-specific failure (lane_iou < 0.20 but mIoU >= 0.30)
        {"name": "f3", "frame_miou": 0.45, "frame_lane_iou": 0.15, "metadata": {"time_of_day": "day", "weather": "shadowed", "road_type": "unpainted", "curvature": "curved"}},
    ]

    analysis = analyze_failures(records, miou_threshold=0.30, lane_iou_threshold=0.20)

    assert analysis["total_evaluated_frames"] == 3
    assert analysis["severe_failures_count"] == 1
    assert analysis["lane_failures_count"] == 2  # f2 and f3
    assert analysis["scenario_breakdown"]["road_type"]["unpainted"] == 2
    assert analysis["scenario_breakdown"]["weather"]["shadowed"] == 2
    assert len(analysis["recommendations"]) > 0
    # Check that recommendation addresses unpainted or shadows
    rec_text = " ".join(analysis["recommendations"])
    assert "unpainted" in rec_text.lower() or "shadow" in rec_text.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Objective 2 & 7: Visual Gallery & Plotting Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_prediction_gallery_generates_correct_images(tmp_path: Path):
    """Verify 4-column side-by-side galleries are saved for best, worst, and median cohorts."""
    gallery_dir = tmp_path / "predictions"

    # Create 5 dummy evaluation records
    records = []
    for i in range(5):
        img = np.full((64, 64, 3), 100 + i * 20, dtype=np.uint8)
        tgt = np.zeros((64, 64), dtype=np.int64)
        tgt[20:40, :] = 1
        prd = tgt.copy()
        if i == 0:
            prd[:, :] = 0  # Worst (mIoU 0.0)
            miou = 0.0
        elif i == 4:
            miou = 1.0  # Best
        else:
            miou = 0.5

        records.append({
            "name": f"frame_{i:04d}",
            "image": img,
            "target": tgt,
            "pred": prd,
            "frame_miou": miou,
            "frame_lane_iou": miou,
            "metadata": {},
        })

    galleries = generate_prediction_galleries(records, output_dir=gallery_dir, max_per_group=2)

    assert (gallery_dir / "best").exists()
    assert (gallery_dir / "worst").exists()
    assert (gallery_dir / "median").exists()

    assert len(galleries["best"]) == 2
    assert len(galleries["worst"]) == 2
    assert len(galleries["median"]) == 2

    for p in galleries["best"] + galleries["worst"] + galleries["median"]:
        assert p.exists()
        assert p.stat().st_size > 0


def test_build_4col_panel_dimensions_and_error_map():
    """Verify 4-column panel generates expected composite width and error map."""
    h, w = 50, 100
    img = np.zeros((h, w, 3), dtype=np.uint8)
    tgt = np.zeros((h, w), dtype=np.int64)
    prd = np.ones((h, w), dtype=np.int64)  # Mismatch everywhere

    panel = build_4col_panel(img, tgt, prd, frame_miou=0.0, frame_name="test_frame")
    header_h = 36
    divider_w = 3
    # 4 panels of width w + 3 dividers of width 3
    expected_w = (4 * w) + (3 * divider_w)
    expected_h = h + header_h

    assert panel.shape == (expected_h, expected_w, 3)

    # Test error map creation
    err = create_error_map(prd, tgt)
    assert err.shape == (h, w, 3)
    # Since mismatch everywhere, error pixels should be coral red (240, 50, 50)
    assert np.all(err == (240, 50, 50))


def test_plots_generation(tmp_path: Path):
    """Verify confusion matrix, per-class bar chart, and failure distribution plots save valid files."""
    conf_mat = np.array([
        [100, 5, 2, 1],
        [4, 80, 3, 2],
        [1, 2, 40, 0],
        [0, 1, 1, 45],
    ], dtype=np.int64)

    cm_path = tmp_path / "confusion_matrix.png"
    plot_confusion_matrix(conf_mat, output_file=cm_path)
    assert cm_path.exists()
    assert cm_path.stat().st_size > 500

    metrics = compute_classification_metrics(conf_mat)
    bar_path = tmp_path / "per_class_iou.png"
    plot_per_class_iou(metrics["per_class"], output_file=bar_path)
    assert bar_path.exists()
    assert bar_path.stat().st_size > 500

    scenarios = {
        "weather": {"clear": 10, "rain": 4, "shadowed": 8},
        "road_type": {"painted": 5, "unpainted": 17},
    }
    fail_path = tmp_path / "failure_distribution.png"
    plot_failure_distribution(scenarios, output_file=fail_path)
    assert fail_path.exists()
    assert fail_path.stat().st_size > 500


# ═══════════════════════════════════════════════════════════════════════════════
# Objective 4 & 7: Report Generation Tests
# ═══════════════════════════════════════════════════════════════════════════════

def test_report_generation_produces_valid_json(tmp_path: Path):
    """Verify generate_json_report outputs valid structured JSON."""
    metrics_data = {"mean_iou": 0.78, "lane_mean_iou": 0.72, "per_class": {}}
    failure_analysis = {"severe_failures_count": 0, "lane_failures_count": 1, "recommendations": ["Test recommendation"]}
    latency_stats = {"p50_ms": 14.5, "fps": 68.9}

    json_file = tmp_path / "evaluation_report.json"
    generate_json_report(metrics_data, failure_analysis, latency_stats, output_path=json_file)

    assert json_file.exists()
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert "metrics" in data
    assert "latency" in data
    assert "failure_analysis" in data
    assert data["metrics"]["mean_iou"] == 0.78
    assert data["latency"]["fps"] == 68.9


def test_markdown_and_html_reports(tmp_path: Path):
    """Verify Markdown and HTML reports are formatted and contain key section headers."""
    metrics_data = {
        "mean_iou": 0.785,
        "lane_mean_iou": 0.720,
        "mean_dice": 0.810,
        "overall_pixel_accuracy": 0.945,
        "per_class": {
            "Background": {"iou": 0.92, "dice": 0.95, "precision": 0.96, "recall": 0.95, "pixel_accuracy": 0.96, "pixel_fraction": 0.60},
            "Road": {"iou": 0.85, "dice": 0.91, "precision": 0.90, "recall": 0.92, "pixel_accuracy": 0.92, "pixel_fraction": 0.30},
        },
    }
    failure_analysis = {
        "total_evaluated_frames": 100,
        "severe_failures_count": 2,
        "lane_failures_count": 4,
        "scenario_breakdown": {"road_type": {"unpainted": 4}},
        "recommendations": ["Collect more unpainted road samples"],
    }
    latency_stats = {"mean_ms": 15.2, "p50_ms": 14.8, "p95_ms": 18.0, "p99_ms": 22.0, "fps": 65.8}

    md_file = tmp_path / "evaluation_report.md"
    generate_markdown_report(metrics_data, failure_analysis, latency_stats, output_path=md_file)
    assert md_file.exists()
    md_content = md_file.read_text(encoding="utf-8")
    assert "Executive Summary" in md_content
    assert "Per-Class Perception Breakdown" in md_content
    assert "0.7850" in md_content

    html_file = tmp_path / "evaluation_report.html"
    generate_html_report(metrics_data, failure_analysis, latency_stats, output_path=html_file)
    assert html_file.exists()
    html_content = html_file.read_text(encoding="utf-8")
    assert "<!DOCTYPE html>" in html_content
    assert "APEX-LKA Perception Evaluation Report" in html_content
    assert "0.7850" in html_content
