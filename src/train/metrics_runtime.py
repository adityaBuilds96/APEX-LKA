"""
src/train/metrics_runtime.py
============================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
High-Throughput Confusion Matrix Accumulator & Runtime IoU/F1 Metric Evaluator.
ISO 26262 ASIL-B Perception Quality & Verification Standard.

Tracked Metric Suite:
---------------------
1. Multi-Class Intersection-over-Union (IoU) per class:
   IoU_c = TP_c / (TP_c + FP_c + FN_c + eps)
2. Mean Intersection-over-Union (mIoU):
   - Macro mIoU (All 4 classes: Background, Corridor, Left, Right)
   - Foreground mIoU (Classes 1, 2, 3 only)
3. Precision, Recall & F1-Score:
   F1_c = 2 * Precision_c * Recall_c / (Precision_c + Recall_c + eps)
4. Overall Pixel Accuracy (PA) & Frequency-Weighted IoU (FWIoU).
"""

from __future__ import annotations

import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.MetricsRuntime")

EPSILON: float = 1e-7


# ──────────────────────────────────────────────────────────────────────────────
# 1. Metric Evaluation Result Dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class RuntimeEvaluationResult:
    """
    Forensic evaluation summary produced at epoch or evaluation batch boundaries.
    """
    mean_iou: float
    foreground_mean_iou: float
    pixel_accuracy: float
    mean_accuracy: float
    frequency_weighted_iou: float
    per_class_iou: Dict[str, float]
    per_class_f1: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    confusion_matrix: List[List[int]]
    total_pixels_evaluated: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def format_table(self) -> str:
        """Format metrics into a clean human-readable console table."""
        lines = [
            "-" * 72,
            f"{'Class Name':<22} | {'IoU (%)':<10} | {'F1 (%)':<10} | {'Prec (%)':<10} | {'Rec (%)':<10}",
            "-" * 72,
        ]
        for name in self.per_class_iou:
            iou = self.per_class_iou[name] * 100.0
            f1 = self.per_class_f1.get(name, 0.0) * 100.0
            prec = self.per_class_precision.get(name, 0.0) * 100.0
            rec = self.per_class_recall.get(name, 0.0) * 100.0
            lines.append(f"{name:<22} | {iou:10.2f} | {f1:10.2f} | {prec:10.2f} | {rec:10.2f}")
        lines.append("-" * 72)
        lines.append(f"Macro Mean IoU         : {self.mean_iou * 100.0:6.2f}%")
        lines.append(f"Foreground Mean IoU    : {self.foreground_mean_iou * 100.0:6.2f}%")
        lines.append(f"Global Pixel Accuracy  : {self.pixel_accuracy * 100.0:6.2f}%")
        lines.append("-" * 72)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 2. Confusion Matrix & Runtime Metric Accumulator
# ──────────────────────────────────────────────────────────────────────────────

class RuntimeMetricsEvaluator:
    """
    Stateful metric accumulator updating a 2D confusion matrix over multiple batches.
    Thread-safe and GPU-optimized using fast bincount tensor arithmetic.
    """

    def __init__(self, num_classes: int = 4) -> None:
        self.num_classes = num_classes
        self.schema = get_default_label_schema()
        self._confusion_matrix = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    def reset(self) -> None:
        """Clear accumulator matrix for a new evaluation epoch."""
        self._confusion_matrix.zero_()

    @torch.no_grad()
    def update(
        self,
        preds: Union[torch.Tensor, np.ndarray],
        targets: Union[torch.Tensor, np.ndarray],
    ) -> None:
        """
        Update confusion matrix with a new batch of predictions and ground truths.
        Args:
            preds: [B, H, W] argmax class indices or [B, C, H, W] logits
            targets: [B, H, W] ground-truth class indices
        """
        if isinstance(preds, np.ndarray):
            preds = torch.from_numpy(preds)
        if isinstance(targets, np.ndarray):
            targets = torch.from_numpy(targets)

        # If logits passed, take argmax across channel dimension
        if preds.dim() == 4:
            preds = torch.argmax(preds, dim=1)

        preds = preds.view(-1).long()
        targets = targets.view(-1).long()

        # Mask out-of-bounds targets
        valid_mask = (targets >= 0) & (targets < self.num_classes)
        p = preds[valid_mask]
        t = targets[valid_mask]

        # Flattened linear index: targets * num_classes + preds
        linear_indices = t * self.num_classes + p
        counts = torch.bincount(linear_indices, minlength=self.num_classes ** 2)

        batch_cm = counts.reshape(self.num_classes, self.num_classes).cpu()
        self._confusion_matrix += batch_cm

    def compute(self) -> RuntimeEvaluationResult:
        """
        Compute all multi-class IoU, F1, Precision, and Recall scores.
        """
        cm = self._confusion_matrix.float()
        total_pixels = int(cm.sum().item())

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        # IoU = TP / (TP + FP + FN + eps)
        denominator = tp + fp + fn
        iou = torch.where(denominator > 0, tp / (denominator + EPSILON), torch.zeros_like(tp))

        # Precision = TP / (TP + FP + eps)
        prec_denom = tp + fp
        precision = torch.where(prec_denom > 0, tp / (prec_denom + EPSILON), torch.zeros_like(tp))

        # Recall = TP / (TP + FN + eps)
        rec_denom = tp + fn
        recall = torch.where(rec_denom > 0, tp / (rec_denom + EPSILON), torch.zeros_like(tp))

        # F1 = 2 * P * R / (P + R + eps)
        f1_denom = precision + recall
        f1 = torch.where(f1_denom > 0, 2.0 * precision * recall / (f1_denom + EPSILON), torch.zeros_like(tp))

        # Accuracy
        pixel_accuracy = float(tp.sum().item()) / max(1.0, float(total_pixels))
        class_acc = torch.where(rec_denom > 0, tp / (rec_denom + EPSILON), torch.zeros_like(tp))
        mean_accuracy = float(class_acc.mean().item())

        # Frequency-Weighted IoU
        freq = cm.sum(dim=1) / max(1.0, float(total_pixels))
        fwiou = float((freq * iou).sum().item())

        # Dictionary mappings
        per_class_iou = {}
        per_class_f1 = {}
        per_class_prec = {}
        per_class_rec = {}

        for c_idx in range(self.num_classes):
            c_name = self.schema.get_class_name(c_idx)
            per_class_iou[c_name] = round(float(iou[c_idx].item()), 4)
            per_class_f1[c_name] = round(float(f1[c_idx].item()), 4)
            per_class_prec[c_name] = round(float(precision[c_idx].item()), 4)
            per_class_rec[c_name] = round(float(recall[c_idx].item()), 4)

        macro_miou = float(iou.mean().item())
        # Foreground mIoU (Classes 1, 2, 3)
        if self.num_classes > 1:
            fg_miou = float(iou[1:].mean().item())
        else:
            fg_miou = macro_miou

        return RuntimeEvaluationResult(
            mean_iou=round(macro_miou, 4),
            foreground_mean_iou=round(fg_miou, 4),
            pixel_accuracy=round(pixel_accuracy, 4),
            mean_accuracy=round(mean_accuracy, 4),
            frequency_weighted_iou=round(fwiou, 4),
            per_class_iou=per_class_iou,
            per_class_f1=per_class_f1,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            confusion_matrix=self._confusion_matrix.tolist(),
            total_pixels_evaluated=total_pixels,
        )

    # Backward compatibility alias
    compute_summary = compute


# ──────────────────────────────────────────────────────────────────────────────
# 3. Confusion Matrix ASCII Visualizer & Normalized Error Heatmap
# ──────────────────────────────────────────────────────────────────────────────

class ConfusionMatrixASCIIPlotter:
    """
    Renders normalized confusion matrix heatmaps to terminal or markdown logs
    for immediate ISO 26262 perception verification during training epochs.
    """

    @classmethod
    def format_normalized_matrix(
        cls,
        confusion_matrix: List[List[int]],
        class_names: Optional[List[str]] = None,
    ) -> str:
        cm = np.array(confusion_matrix, dtype=np.float64)
        num_classes = cm.shape[0]
        names = class_names or [f"Class {i}" for i in range(num_classes)]

        # Row-normalize: P(pred = j | true = i)
        row_sums = cm.sum(axis=1, keepdims=True)
        norm_cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0) * 100.0

        header_lbl = "True / Pred"
        lines: List[str] = [
            "Confusion Matrix (Row Normalized: Ground Truth -> Predictions %):",
            "-" * 65,
            f"{header_lbl:<16} | " + " | ".join(f"{name[:8]:>8}" for name in names),
            "-" * 65,
        ]

        for i, row in enumerate(norm_cm):
            row_str = " | ".join(f"{val:7.2f}%" for val in row)
            lines.append(f"{names[i]:<16} | {row_str}")

        lines.append("-" * 65)
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Boundary Distance-Tolerance F1-Score Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class BoundaryDistanceF1Evaluator:
    """
    Evaluates spatial lane line accuracy using pixel distance thresholds (e.g., 2px, 5px, 8px).
    Computes True Positives if predicted boundary pixels fall within radius R of ground-truth lines.
    """

    def __init__(self, tolerance_px: int = 5, num_classes: int = 4) -> None:
        self.tolerance_px = tolerance_px
        self.num_classes = num_classes
        self._tp_counts: Dict[int, int] = {2: 0, 3: 0}
        self._total_pred_counts: Dict[int, int] = {2: 0, 3: 0}
        self._total_gt_counts: Dict[int, int] = {2: 0, 3: 0}

    def reset(self) -> None:
        for c in (2, 3):
            self._tp_counts[c] = 0
            self._total_pred_counts[c] = 0
            self._total_gt_counts[c] = 0

    def update_batch(self, pred_masks: np.ndarray, gt_masks: np.ndarray) -> None:
        """
        Args:
            pred_masks: [B, H, W] int array of predictions
            gt_masks: [B, H, W] int array of ground truths
        """
        import cv2
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.tolerance_px * 2 + 1, self.tolerance_px * 2 + 1))

        b = pred_masks.shape[0]
        for i in range(b):
            p = pred_masks[i]
            g = gt_masks[i]

            for c in (2, 3):
                p_c = (p == c).astype(np.uint8)
                g_c = (g == c).astype(np.uint8)

                n_pred = int(p_c.sum())
                n_gt = int(g_c.sum())

                self._total_pred_counts[c] += n_pred
                self._total_gt_counts[c] += n_gt

                if n_pred > 0 and n_gt > 0:
                    # Dilate GT to match within tolerance R
                    g_c_dilated = cv2.dilate(g_c, kernel)
                    tp = int((p_c * g_c_dilated).sum())
                    self._tp_counts[c] += tp

    def compute(self) -> Dict[str, Dict[str, float]]:
        results: Dict[str, Dict[str, float]] = {}
        names = {2: "left_boundary", 3: "right_boundary"}

        for c, name in names.items():
            tp = self._tp_counts[c]
            pred = self._total_pred_counts[c]
            gt = self._total_gt_counts[c]

            precision = tp / max(1, pred)
            recall = tp / max(1, gt)
            f1 = (2.0 * precision * recall) / max(1e-7, precision + recall)

            results[name] = {
                "precision": round(float(precision), 4),
                "recall": round(float(recall), 4),
                "f1": round(float(f1), 4),
            }

        return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Perception Latency & Real-Time Throughput Profiler
# ──────────────────────────────────────────────────────────────────────────────

class PerceptionLatencyProfiler:
    """
    Profiles end-to-end perception latency and ensures compliance with
    real-time embedded compute budgets (NVIDIA Jetson Orin Nano <= 25ms @ 40 FPS).
    """

    def __init__(self, target_max_latency_ms: float = 25.0) -> None:
        self.target_max_latency_ms = target_max_latency_ms
        self.latencies_ms: List[float] = []

    def reset(self) -> None:
        self.latencies_ms.clear()

    def record_step(self, duration_sec: float) -> None:
        self.latencies_ms.append(duration_sec * 1000.0)

    def compute_summary(self) -> Dict[str, Any]:
        if not self.latencies_ms:
            return {"status": "NO_MEASUREMENTS"}

        arr = np.array(self.latencies_ms, dtype=np.float32)
        p50 = float(np.percentile(arr, 50))
        p95 = float(np.percentile(arr, 95))
        p99 = float(np.percentile(arr, 99))
        mean_lat = float(np.mean(arr))
        fps = 1000.0 / max(1e-3, mean_lat)

        return {
            "mean_latency_ms": round(mean_lat, 2),
            "p50_latency_ms": round(p50, 2),
            "p95_latency_ms": round(p95, 2),
            "p99_latency_ms": round(p99, 2),
            "effective_fps": round(fps, 1),
            "meets_realtime_budget": p95 <= self.target_max_latency_ms,
            "total_frames_profiled": len(arr),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 5. TuSimple Benchmark Standard Metric Evaluator
# ──────────────────────────────────────────────────────────────────────────────

class TuSimpleBenchmarkEvaluator:
    """
    Computes canonical TuSimple lane benchmark metrics:
    - Accuracy = sum(C_clip) / sum(S_clip)
    - False Positive Rate (FP)
    - False Negative Rate (FN)
    """

    def __init__(self, x_pixel_tolerance: float = 20.0) -> None:
        self.x_pixel_tolerance = x_pixel_tolerance
        self.total_points: int = 0
        self.correct_points: int = 0
        self.total_gt_lines: int = 0
        self.total_pred_lines: int = 0
        self.matched_lines: int = 0

    def reset(self) -> None:
        self.total_points = 0
        self.correct_points = 0
        self.total_gt_lines = 0
        self.total_pred_lines = 0
        self.matched_lines = 0

    def evaluate_polyline_sample(
        self,
        pred_x_samples: List[List[float]],
        gt_x_samples: List[List[float]],
    ) -> None:
        """
        Args:
            pred_x_samples: list of predicted X-coordinate arrays along fixed Y-heights
            gt_x_samples: list of ground-truth X-coordinate arrays (-2 = absent)
        """
        self.total_pred_lines += len(pred_x_samples)
        self.total_gt_lines += len(gt_x_samples)

        for gt_line in gt_x_samples:
            valid_gt_indices = [idx for idx, val in enumerate(gt_line) if val > 0]
            if not valid_gt_indices:
                continue

            self.total_points += len(valid_gt_indices)
            best_match_correct = 0

            for pred_line in pred_x_samples:
                correct_count = 0
                for idx in valid_gt_indices:
                    if idx < len(pred_line) and pred_line[idx] > 0:
                        if abs(pred_line[idx] - gt_line[idx]) <= self.x_pixel_tolerance:
                            correct_count += 1
                if correct_count > best_match_correct:
                    best_match_correct = correct_count

            self.correct_points += best_match_correct
            if best_match_correct / len(valid_gt_indices) >= 0.85:
                self.matched_lines += 1

    def compute(self) -> Dict[str, float]:
        accuracy = (self.correct_points / max(1, self.total_points)) if self.total_points > 0 else 0.0
        fn = (self.total_gt_lines - self.matched_lines) / max(1, self.total_gt_lines)
        fp = max(0, self.total_pred_lines - self.matched_lines) / max(1, self.total_pred_lines)

        return {
            "tusimple_accuracy": round(float(accuracy), 4),
            "false_negative_rate": round(float(fn), 4),
            "false_positive_rate": round(float(fp), 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing MetricsRuntime ISO 26262 ASIL-B self-check...")
    evaluator = RuntimeMetricsEvaluator(num_classes=4)

    # Perfect prediction test
    targets = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    preds = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    evaluator.update(preds, targets)
    res = evaluator.compute()

    assert res.mean_iou == 1.0, f"Expected 1.0 mIoU on identical inputs, got {res.mean_iou}"
    assert res.pixel_accuracy == 1.0

    # Test Latency Profiler
    profiler = PerceptionLatencyProfiler()
    profiler.record_step(0.012)
    profiler.record_step(0.015)
    summary = profiler.compute_summary()
    assert summary["meets_realtime_budget"]

    # Test TuSimple Evaluator
    tusimple_eval = TuSimpleBenchmarkEvaluator(x_pixel_tolerance=20.0)
    fake_gt = [[100.0, 110.0, 120.0, 130.0]]
    fake_pred = [[102.0, 109.0, 121.0, 128.0]]
    tusimple_eval.evaluate_polyline_sample(fake_pred, fake_gt)
    t_res = tusimple_eval.compute()
    assert t_res["tusimple_accuracy"] == 1.0

    print("Perfect accuracy check passed.")
    print(res.format_table())

