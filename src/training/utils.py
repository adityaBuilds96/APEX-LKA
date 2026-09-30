"""
src/training/utils.py
=====================
Utility functions for APEX-LKA Phase 3 Training Engine:
- Vectorized confusion matrix and semantic segmentation metrics (IoU, Dice, Precision, Recall).
- Checkpoint I/O with comprehensive metadata header.
- Dataset hashing (SHA-256) and Git commit tracking for experiment reproducibility.
- Live metrics publisher (JSON) for Streamlit dashboard synchronization.
- Visualization helpers for prediction mask overlays.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Metric Computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
) -> torch.Tensor:
    """
    Fast vectorized confusion matrix computation on device.
    Shape: (num_classes, num_classes), row=truth, col=pred.
    """
    mask = (targets >= 0) & (targets < num_classes)
    flat_targets = targets[mask].view(-1)
    flat_preds = preds[mask].view(-1)

    bins = flat_targets * num_classes + flat_preds
    hist = torch.bincount(bins, minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes)


def metrics_from_confusion_matrix(
    conf_mat: torch.Tensor,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Computes per-class IoU, mean IoU, per-class Dice, and lane mean IoU.
    """
    if class_names is None:
        class_names = ["background", "road", "left_lane", "right_lane"]

    num_classes = conf_mat.shape[0]
    ious: List[float] = []
    dices: List[float] = []
    res: Dict[str, float] = {}

    total_tp = 0.0
    total_pixels = conf_mat.sum().item()

    for c in range(num_classes):
        tp = conf_mat[c, c].item()
        fp = conf_mat[:, c].sum().item() - tp
        fn = conf_mat[c, :].sum().item() - tp
        total_tp += tp

        denom_iou = tp + fp + fn
        iou = (tp / denom_iou) if denom_iou > 0 else 0.0
        ious.append(iou)

        denom_dice = (2.0 * tp) + fp + fn
        dice = (2.0 * tp / denom_dice) if denom_dice > 0 else 0.0
        dices.append(dice)

        c_name = class_names[c] if c < len(class_names) else f"class_{c}"
        res[f"{c_name}_iou"] = round(iou, 4)
        res[f"{c_name}_dice"] = round(dice, 4)

    mean_iou = sum(ious) / max(1, len(ious))
    mean_dice = sum(dices) / max(1, len(dices))
    res["mean_iou"] = round(mean_iou, 4)
    res["mean_dice"] = round(mean_dice, 4)
    res["pixel_accuracy"] = round(total_tp / max(1.0, total_pixels), 4)

    # Specific focus metric: lanes only (classes 2 & 3)
    if num_classes >= 4:
        lane_miou = (ious[2] + ious[3]) / 2.0
        res["lane_mean_iou"] = round(lane_miou, 4)

    return res


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Reproducibility & Metadata Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def get_git_commit_hash() -> str:
    """Retrieve current Git commit SHA-1 hash, or 'unknown'."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
            check=True,
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def compute_dataset_hash(paths: Sequence[Union[str, Path]]) -> str:
    """
    Compute a deterministic SHA-256 fingerprint for a collection of dataset file paths.
    """
    hasher = hashlib.sha256()
    sorted_stems = sorted([str(Path(p).name) for p in paths])
    for s in sorted_stems:
        hasher.update(s.encode("utf-8"))
    return hasher.hexdigest()[:16]


def get_random_states() -> Dict[str, Any]:
    """Capture complete RNG state across Python, NumPy, PyTorch, and CUDA (weights_only safe)."""
    np_state = np.random.get_state()
    # Convert numpy array in state to python list for safe torch.load(weights_only=True)
    safe_np_state = (
        np_state[0],
        np_state[1].tolist() if hasattr(np_state[1], "tolist") else np_state[1],
        np_state[2],
        np_state[3],
        np_state[4],
    )
    states: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": safe_np_state,
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["cuda"] = torch.cuda.get_rng_state_all()
    return states


def set_random_states(states: Dict[str, Any]) -> None:
    """Restore complete RNG state across Python, NumPy, PyTorch, and CUDA."""
    if "python" in states:
        random.setstate(states["python"])
    if "numpy" in states:
        s = states["numpy"]
        if isinstance(s, (tuple, list)) and len(s) == 5:
            arr = np.array(s[1], dtype=np.uint32) if isinstance(s[1], list) else s[1]
            np.random.set_state((s[0], arr, s[2], s[3], s[4]))
        else:
            np.random.set_state(s)
    if "torch" in states:
        torch.set_rng_state(states["torch"])
    if "cuda" in states and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(states["cuda"])
        except Exception as e:
            log.warning("Could not restore CUDA RNG states: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Checkpoint Management & I/O
# ═══════════════════════════════════════════════════════════════════════════════

def find_latest_checkpoint(checkpoint_dir: Union[str, Path]) -> Optional[Path]:
    """
    Auto-detect the most recent valid checkpoint in checkpoint_dir.
    Priority:
    1. last_checkpoint.pt
    2. crash_checkpoint.pt
    3. Highest numbered epoch_*.pth or ckpt_epoch_*.pt
    """
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        return None

    last_pt = cdir / "last_checkpoint.pt"
    if last_pt.exists():
        return last_pt

    crash_pt = cdir / "crash_checkpoint.pt"
    if crash_pt.exists():
        return crash_pt

    emergency_pt = cdir / "emergency_checkpoint.pt"
    if emergency_pt.exists():
        return emergency_pt

    epoch_ckpts = list(cdir.glob("epoch_*.pth")) + list(cdir.glob("ckpt_epoch_*.pt"))
    if epoch_ckpts:
        # Sort by modification time or numeric name
        epoch_ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return epoch_ckpts[0]

    return None


def cleanup_old_checkpoints(checkpoint_dir: Union[str, Path], keep_last: int = 5) -> None:
    """
    Rolling cleanup: keep only the last `keep_last` epoch checkpoints.
    Never deletes best_model.pth, crash_checkpoint.pt, or last_checkpoint.pt.
    """
    cdir = Path(checkpoint_dir)
    if not cdir.exists():
        return

    epoch_ckpts = sorted(
        list(cdir.glob("epoch_*.pth")) + list(cdir.glob("ckpt_epoch_*.pt")),
        key=lambda p: p.stat().st_mtime,
    )
    if len(epoch_ckpts) > keep_last:
        to_delete = epoch_ckpts[:-keep_last]
        for p in to_delete:
            try:
                p.unlink()
                log.info("Rolling checkpoint cleanup: deleted %s", p.name)
            except OSError as e:
                log.warning("Failed to delete old checkpoint %s: %s", p.name, e)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Live Metrics Publisher (Dashboard Sync)
# ═══════════════════════════════════════════════════════════════════════════════

def update_live_metrics_json(
    filepath: Union[str, Path],
    epoch: int,
    total_epochs: int,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    lr: float,
    speed_fps: float,
    status: str = "TRAINING",
) -> None:
    """
    Atomically writes live training status to logs/training/live_metrics.json
    for real-time Streamlit dashboard rendering.
    """
    fp = Path(filepath)
    fp.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch + 1,
        "total_epochs": total_epochs,
        "progress_pct": round(((epoch + 1) / max(1, total_epochs)) * 100.0, 1),
        "learning_rate": lr,
        "speed_fps": round(speed_fps, 2),
        "status": status,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train": train_metrics,
        "val": val_metrics,
    }

    tmp = fp.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(fp)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Visualization Overlay Helper
# ═══════════════════════════════════════════════════════════════════════════════

def create_prediction_overlay(
    image: np.ndarray,
    pred_mask: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blends predicted lane classes onto input image:
    Class 1 (Road): Soft Green
    Class 2 (Left Lane): Bright Red
    Class 3 (Right Lane): Electric Cyan
    """
    overlay = image.copy()
    h, w = pred_mask.shape[:2]

    # Resize image to match mask if necessary
    if overlay.shape[:2] != (h, w):
        overlay = cv2.resize(overlay, (w, h))

    color_map = {
        1: (34, 197, 94),    # Road: Green
        2: (239, 68, 68),    # Left Lane: Red
        3: (56, 189, 248),   # Right Lane: Cyan
    }

    colored_mask = np.zeros_like(overlay)
    for class_id, color in color_map.items():
        colored_mask[pred_mask == class_id] = color

    has_class = pred_mask > 0
    overlay[has_class] = cv2.addWeighted(
        overlay[has_class], 1.0 - alpha,
        colored_mask[has_class], alpha,
        0
    )
    return overlay
