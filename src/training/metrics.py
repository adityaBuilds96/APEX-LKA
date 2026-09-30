"""
src/training/metrics.py
=======================
Comprehensive evaluation metrics engine for APEX-LKA LaneSegNet:
- Multi-class confusion matrix (4x4)
- Per-class and aggregate IoU, Dice, Precision, Recall, F1 score
- Overall and per-class pixel accuracy
- Per-frame metrics for best/worst ranking
- Inference latency profiling (mean, p50, p95, p99, FPS)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

log = logging.getLogger(__name__)

CLASS_NAMES = ["Background", "Road", "Left Lane", "Right Lane"]
NUM_CLASSES = 4


def compute_confusion_matrix(
    preds: Union[torch.Tensor, np.ndarray],
    targets: Union[torch.Tensor, np.ndarray],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """
    Compute confusion matrix of size (num_classes, num_classes).
    Rows: Ground Truth, Columns: Predictions.
    """
    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    preds = preds.astype(np.int64).flatten()
    targets = targets.astype(np.int64).flatten()

    mask = (targets >= 0) & (targets < num_classes)
    targets = targets[mask]
    preds = preds[mask]

    bins = targets * num_classes + preds
    conf_mat = np.bincount(bins, minlength=num_classes**2).reshape(num_classes, num_classes)
    return conf_mat.astype(np.int64)


def compute_classification_metrics(
    conf_mat: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Computes comprehensive semantic segmentation metrics from a 4x4 confusion matrix.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    num_classes = conf_mat.shape[0]
    total_pixels = float(conf_mat.sum())

    per_class: Dict[str, Dict[str, float]] = {}
    ious: List[float] = []
    dices: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    pixel_accs: List[float] = []

    total_tp = 0.0

    for c in range(num_classes):
        name = class_names[c] if c < len(class_names) else f"Class_{c}"
        tp = float(conf_mat[c, c])
        total_tp += tp

        fp = float(conf_mat[:, c].sum() - tp)
        fn = float(conf_mat[c, :].sum() - tp)
        tn = float(total_pixels - (tp + fp + fn))
        gt_pixels = float(conf_mat[c, :].sum())

        # IoU
        iou_denom = tp + fp + fn
        iou = (tp / iou_denom) if iou_denom > 0 else 1.0 if (gt_pixels == 0 and fp == 0) else 0.0
        ious.append(iou)

        # Precision & Recall
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        precisions.append(prec)
        recalls.append(rec)

        # Dice / F1
        dice_denom = (2.0 * tp) + fp + fn
        dice = (2.0 * tp / dice_denom) if dice_denom > 0 else 1.0 if (gt_pixels == 0 and fp == 0) else 0.0
        dices.append(dice)
        f1s.append(dice)

        # Per-class accuracy
        class_acc = ((tp + tn) / total_pixels) if total_pixels > 0 else 0.0
        pixel_accs.append(class_acc)

        per_class[name] = {
            "iou": round(iou, 4),
            "dice": round(dice, 4),
            "f1": round(dice, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "pixel_accuracy": round(class_acc, 4),
            "pixel_count": int(gt_pixels),
            "pixel_fraction": round(gt_pixels / max(1.0, total_pixels), 4),
        }

    overall_accuracy = (total_tp / total_pixels) if total_pixels > 0 else 0.0
    mean_iou = sum(ious) / max(1, len(ious))
    mean_dice = sum(dices) / max(1, len(dices))
    mean_precision = sum(precisions) / max(1, len(precisions))
    mean_recall = sum(recalls) / max(1, len(recalls))
    mean_f1 = sum(f1s) / max(1, len(f1s))

    # Lane-specific metrics (Classes 2 & 3: Left & Right Lane)
    lane_iou = (ious[2] + ious[3]) / 2.0 if num_classes >= 4 else 0.0
    lane_dice = (dices[2] + dices[3]) / 2.0 if num_classes >= 4 else 0.0
    lane_f1 = (f1s[2] + f1s[3]) / 2.0 if num_classes >= 4 else 0.0

    return {
        "mean_iou": round(mean_iou, 4),
        "mean_dice": round(mean_dice, 4),
        "mean_precision": round(mean_precision, 4),
        "mean_recall": round(mean_recall, 4),
        "mean_f1": round(mean_f1, 4),
        "overall_pixel_accuracy": round(overall_accuracy, 4),
        "lane_mean_iou": round(lane_iou, 4),
        "lane_mean_dice": round(lane_dice, 4),
        "lane_mean_f1": round(lane_f1, 4),
        "per_class": per_class,
        "confusion_matrix": conf_mat.tolist(),
        "total_pixels": int(total_pixels),
    }


def compute_frame_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> Dict[str, float]:
    """
    Compute frame-level mIoU and per-class IoU for ranking and failure classification.
    """
    ious: List[float] = []
    lane_ious: List[float] = []
    res: Dict[str, float] = {}

    for c in range(num_classes):
        pred_c = (pred == c)
        target_c = (target == c)

        inter = np.logical_and(pred_c, target_c).sum()
        union = np.logical_or(pred_c, target_c).sum()

        if union == 0:
            iou = 1.0  # Perfect agreement on absence of class
        else:
            iou = float(inter) / float(union)

        ious.append(iou)
        c_name = CLASS_NAMES[c].lower().replace(" ", "_")
        res[f"{c_name}_iou"] = round(iou, 4)

        if c in (2, 3):
            lane_ious.append(iou)

    res["frame_miou"] = round(float(np.mean(ious)), 4)
    res["frame_lane_iou"] = round(float(np.mean(lane_ious)), 4) if lane_ious else 0.0
    return res


def compute_latency_statistics(latencies_ms: Sequence[float]) -> Dict[str, float]:
    """
    Compute comprehensive latency percentiles and FPS.
    """
    if not latencies_ms:
        return {
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "fps": 0.0,
        }

    arr = np.array(latencies_ms, dtype=np.float64)
    mean_ms = float(np.mean(arr))
    p50_ms = float(np.percentile(arr, 50))
    p95_ms = float(np.percentile(arr, 95))
    p99_ms = float(np.percentile(arr, 99))
    min_ms = float(np.min(arr))
    max_ms = float(np.max(arr))
    fps = 1000.0 / max(0.1, mean_ms)

    return {
        "mean_ms": round(mean_ms, 2),
        "p50_ms": round(p50_ms, 2),
        "p95_ms": round(p95_ms, 2),
        "p99_ms": round(p99_ms, 2),
        "min_ms": round(min_ms, 2),
        "max_ms": round(max_ms, 2),
        "fps": round(fps, 1),
    }
