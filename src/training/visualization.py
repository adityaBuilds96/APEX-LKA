"""
src/training/visualization.py
=============================
Visual evaluation and prediction gallery generation for APEX-LKA:
- 4-Column side-by-side panels: [Original Image | Ground Truth | Prediction | Error Map]
- Auto-selection of best 20, worst 20, and median 20 predictions
- Confusion matrix heatmap generation
- Per-class IoU bar charts
- Scenario failure distribution charts
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import matplotlib
matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np

from src.training.metrics import CLASS_NAMES

log = logging.getLogger(__name__)

# Class color map (BGR for OpenCV, RGB for Matplotlib)
# 0: Background (Dark Charcoal)
# 1: Road Surface (Emerald Green)
# 2: Left Lane Line (Crimson Red)
# 3: Right Lane Line (Electric Cyan)
COLOR_PALETTE_BGR = {
    0: (35, 35, 35),
    1: (34, 197, 94),
    2: (239, 68, 68),
    3: (56, 189, 248),
}

COLOR_PALETTE_RGB = {
    0: (35, 35, 35),
    1: (94, 197, 34),
    2: (68, 68, 239),
    3: (248, 189, 56),
}


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """Convert an integer class mask to an RGB visualization image."""
    h, w = mask.shape[:2]
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in COLOR_PALETTE_RGB.items():
        colored[mask == class_id] = color
    return colored


def create_error_map(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Create a high-contrast error map:
    - Match: Muted Dark Gray / Soft Green
    - Mismatch / Error: High-visibility Neon Red
    """
    h, w = pred.shape[:2]
    error_img = np.zeros((h, w, 3), dtype=np.uint8)

    correct_mask = (pred == target)
    error_mask = (pred != target)

    # Correct pixels: Soft dark slate
    error_img[correct_mask] = (30, 45, 60)
    # Error pixels: Bright Coral Red
    error_img[error_mask] = (240, 50, 50)

    return error_img


def build_4col_panel(
    image: np.ndarray,
    target: np.ndarray,
    pred: np.ndarray,
    frame_miou: float,
    frame_name: str = "",
) -> np.ndarray:
    """
    Construct side-by-side 4-column panel:
    [Original Image | Ground Truth | Prediction | Error Map]
    """
    h, w = target.shape[:2]

    # Ensure image matches mask dimensions
    if image.shape[:2] != (h, w):
        img_disp = cv2.resize(image, (w, h))
    else:
        img_disp = image.copy()

    gt_disp = mask_to_rgb(target)
    pred_disp = mask_to_rgb(pred)
    error_disp = create_error_map(pred, target)

    # Convert RGB to BGR for OpenCV panel concatenation
    gt_bgr = cv2.cvtColor(gt_disp, cv2.COLOR_RGB2BGR)
    pred_bgr = cv2.cvtColor(pred_disp, cv2.COLOR_RGB2BGR)
    error_bgr = cv2.cvtColor(error_disp, cv2.COLOR_RGB2BGR)

    # Add header banner to each sub-panel
    panels = [
        ("ORIGINAL IMAGE", img_disp),
        ("GROUND TRUTH", gt_bgr),
        (f"PREDICTION (mIoU: {frame_miou:.3f})", pred_bgr),
        ("ERROR MAP (RED=DISAGREE)", error_bgr),
    ]

    decorated = []
    header_h = 36
    for title, panel in panels:
        canvas = np.zeros((h + header_h, w, 3), dtype=np.uint8)
        canvas[:header_h, :] = (15, 23, 42)  # Dark slate header
        cv2.putText(
            canvas,
            title,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (240, 245, 250),
            1,
            cv2.LINE_AA,
        )
        canvas[header_h:, :] = panel
        decorated.append(canvas)

    # Concat horizontally with divider lines
    divider = np.full((h + header_h, 3, 3), (40, 60, 90), dtype=np.uint8)
    combined = np.hstack([
        decorated[0], divider,
        decorated[1], divider,
        decorated[2], divider,
        decorated[3]
    ])

    return combined


def generate_prediction_galleries(
    evaluation_records: List[Dict[str, Any]],
    output_dir: Union[str, Path] = "results/plots/evaluation/predictions",
    max_per_group: int = 20,
) -> Dict[str, List[Path]]:
    """
    Selects best, worst, and median predictions and generates 4-column side-by-side images.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    (out_path / "best").mkdir(parents=True, exist_ok=True)
    (out_path / "worst").mkdir(parents=True, exist_ok=True)
    (out_path / "median").mkdir(parents=True, exist_ok=True)

    if not evaluation_records:
        return {"best": [], "worst": [], "median": []}

    # Sort records by frame_miou ascending (worst first)
    sorted_records = sorted(evaluation_records, key=lambda r: r.get("frame_miou", 0.0))
    n = len(sorted_records)

    k = min(max_per_group, n)
    worst_records = sorted_records[:k]
    best_records = sorted_records[-k:][::-1]

    # Median window
    mid_start = max(0, (n // 2) - (k // 2))
    median_records = sorted_records[mid_start : mid_start + k]

    groups = {
        "best": best_records,
        "worst": worst_records,
        "median": median_records,
    }

    generated_paths: Dict[str, List[Path]] = {"best": [], "worst": [], "median": []}

    for group_name, records in groups.items():
        grp_dir = out_path / group_name
        for i, rec in enumerate(records):
            img = rec["image"]
            target = rec["target"]
            pred = rec["pred"]
            miou = rec.get("frame_miou", 0.0)
            name = rec.get("name", f"frame_{i:04d}")

            panel = build_4col_panel(img, target, pred, miou, frame_name=name)
            save_file = grp_dir / f"{i+1:02d}_{name}_miou_{miou:.3f}.jpg"
            cv2.imwrite(str(save_file), panel)
            generated_paths[group_name].append(save_file)

    log.info(
        "Generated prediction galleries in %s: %d best, %d worst, %d median",
        out_path,
        len(generated_paths["best"]),
        len(generated_paths["worst"]),
        len(generated_paths["median"]),
    )
    return generated_paths


def plot_confusion_matrix(
    conf_mat: np.ndarray,
    output_file: Union[str, Path] = "results/plots/evaluation/confusion_matrix.png",
    class_names: Optional[List[str]] = None,
) -> Path:
    """
    Renders an interactive, publication-ready confusion matrix heatmap.
    """
    if class_names is None:
        class_names = CLASS_NAMES

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 7), facecolor="#0a1220")
    ax.set_facecolor("#0a1220")

    # Normalize by row (Ground truth)
    row_sums = conf_mat.sum(axis=1, keepdims=True).astype(np.float64)
    norm_mat = np.divide(conf_mat, row_sums, out=np.zeros_like(conf_mat, dtype=np.float64), where=row_sums > 0)

    im = ax.imshow(norm_mat, cmap="Blues", vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors="#cbd5e1")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, color="#cbd5e1", fontsize=11, fontweight="bold")
    ax.set_yticklabels(class_names, color="#cbd5e1", fontsize=11, fontweight="bold")

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = norm_mat[i, j]
            count = conf_mat[i, j]
            txt_color = "white" if val > 0.5 else "#94a3b8"
            ax.text(
                j, i,
                f"{val:.1%}\n({count:,})",
                ha="center", va="center",
                color=txt_color, fontsize=9, fontweight="bold",
            )

    ax.set_title("Semantic Segmentation Confusion Matrix", color="#f1f5f9", fontsize=14, pad=16, fontweight="bold")
    ax.set_xlabel("Predicted Class", color="#94a3b8", fontsize=12, labelpad=10)
    ax.set_ylabel("Ground Truth Class", color="#94a3b8", fontsize=12, labelpad=10)

    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_path


def plot_per_class_iou(
    per_class_metrics: Dict[str, Dict[str, float]],
    output_file: Union[str, Path] = "results/plots/evaluation/per_class_iou.png",
) -> Path:
    """
    Renders bar chart comparing per-class IoU and Dice coefficients.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    classes = list(per_class_metrics.keys())
    ious = [per_class_metrics[c].get("iou", 0.0) for c in classes]
    dices = [per_class_metrics[c].get("dice", 0.0) for c in classes]

    x = np.arange(len(classes))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="#0a1220")
    ax.set_facecolor("#0a1220")

    rects1 = ax.bar(x - width/2, ious, width, label="IoU", color="#38bdf8", edgecolor="#0284c7")
    rects2 = ax.bar(x + width/2, dices, width, label="Dice (F1)", color="#22c55e", edgecolor="#16a34a")

    ax.set_title("Per-Class Perception Performance", color="#f1f5f9", fontsize=14, pad=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(classes, color="#cbd5e1", fontsize=11, fontweight="bold")
    ax.set_ylim(0.0, 1.05)
    ax.tick_params(colors="#cbd5e1")
    ax.legend(facecolor="#0f172a", edgecolor="#1e293b", labelcolor="#f1f5f9")
    ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")

    # Add numeric tags
    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom",
            color="#cbd5e1", fontsize=9, fontweight="bold",
        )

    for spine in ax.spines.values():
        spine.set_color("#1e293b")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_path


def plot_failure_distribution(
    failure_counts_by_scenario: Dict[str, Dict[str, int]],
    output_file: Union[str, Path] = "results/plots/evaluation/failure_distribution.png",
) -> Path:
    """
    Renders grouped breakdown of failure counts across conditions.
    """
    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, max(1, len(failure_counts_by_scenario)), figsize=(12, 4.5), facecolor="#0a1220")
    if not isinstance(axes, np.ndarray):
        axes = [axes]

    colors = ["#ef4444", "#f97316", "#eab308", "#38bdf8", "#a855f7"]

    for idx, (cat_name, counts) in enumerate(failure_counts_by_scenario.items()):
        ax = axes[idx]
        ax.set_facecolor("#0a1220")
        labels = list(counts.keys())
        values = list(counts.values())

        ax.bar(labels, values, color=colors[idx % len(colors)], edgecolor="#1e293b")
        ax.set_title(f"Failures by {cat_name.replace('_', ' ').title()}", color="#f1f5f9", fontsize=11, fontweight="bold")
        ax.tick_params(colors="#cbd5e1")
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.15, color="#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#1e293b")

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    return out_path
