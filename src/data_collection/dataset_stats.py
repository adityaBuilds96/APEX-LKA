"""
dataset_stats.py
================
Print a full statistical report of the current dataset:
  - Frame count per split
  - Annotated vs unannotated
  - Pixel-class distribution in masks
  - Brightness/contrast distribution (photometric quality audit)
  - Export summary to results/metrics/dataset_stats.json

Usage
-----
  python src/data_collection/dataset_stats.py
  python src/data_collection/dataset_stats.py --full   # includes pixel stats (slower)
"""

import argparse
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg  # noqa: E402

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def _count_images(directory: Path) -> list[Path]:
    if not directory.exists():
        return []
    return [f for f in directory.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS]


def compute_class_distribution(mask_dir: Path, num_classes: int = 3,
                                max_sample: int = 200) -> dict:
    """
    Compute pixel class distribution across mask images.
    Classes: 0=background, 1=left_lane, 2=right_lane
    """
    masks = _count_images(mask_dir)
    if not masks:
        return {}

    sample = random.sample(masks, min(max_sample, len(masks)))
    total_pixels = Counter = {}
    for i in range(num_classes):
        Counter[i] = 0

    total = 0
    for mp in sample:
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        for cls in range(num_classes):
            Counter[cls] += int(np.sum(mask == cls))
        total += mask.size

    if total == 0:
        return {}

    return {cls: {"count": cnt, "fraction": cnt / total}
            for cls, cnt in Counter.items()}


def compute_brightness_stats(image_dir: Path, max_sample: int = 200) -> dict:
    """Sample images and compute mean brightness statistics."""
    images = _count_images(image_dir)
    if not images:
        return {}

    sample = random.sample(images, min(max_sample, len(images)))
    means = []
    for ip in sample:
        img = cv2.imread(str(ip), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            means.append(float(np.mean(img)))

    if not means:
        return {}

    return {
        "mean": round(float(np.mean(means)), 2),
        "std": round(float(np.std(means)), 2),
        "min": round(float(np.min(means)), 2),
        "max": round(float(np.max(means)), 2),
        "dark_fraction": round(sum(1 for m in means if m < 80) / len(means), 3),
    }


def main():
    parser = argparse.ArgumentParser(description="LKA Dataset Statistics")
    parser.add_argument("--full", action="store_true",
                        help="Include pixel-level class distribution (slower)")
    args = parser.parse_args()

    console.print(Panel("[bold cyan]LKA Dataset Statistics Report", border_style="cyan"))

    report = {}

    # ── Frame counts ───────────────────────────────────────────────────────
    splits = {
        "raw_frames": PATHS.raw_frames,
        "annotated_images": PATHS.annotated / "images",
        "annotated_masks":  PATHS.annotated / "masks",
        "train_images": PATHS.train / "images",
        "train_masks":  PATHS.train / "masks",
        "val_images":   PATHS.val / "images",
        "val_masks":    PATHS.val / "masks",
        "test_images":  PATHS.test / "images",
        "test_masks":   PATHS.test / "masks",
    }

    count_table = Table(title="Frame Counts", show_lines=True)
    count_table.add_column("Directory", style="cyan")
    count_table.add_column("Count", justify="right")

    for name, d in splits.items():
        files = _count_images(d)
        count_table.add_row(name, str(len(files)))
        report[name + "_count"] = len(files)

    console.print(count_table)

    # ── Annotation coverage ────────────────────────────────────────────────
    ann_imgs = set(f.stem for f in _count_images(PATHS.annotated / "images"))
    ann_masks = set(f.stem for f in _count_images(PATHS.annotated / "masks"))
    paired = ann_imgs & ann_masks
    coverage = len(paired) / max(len(ann_imgs), 1) * 100

    console.print(f"\n[bold]Annotation coverage:[/bold] {len(paired)}/{len(ann_imgs)} "
                  f"({coverage:.1f}%)")
    report["annotation_coverage_pct"] = round(coverage, 2)

    if args.full:
        # ── Class distribution ─────────────────────────────────────────────
        console.print("\n[bold]Computing pixel class distribution...[/bold]")
        cls_dist = compute_class_distribution(PATHS.annotated / "masks")
        if cls_dist:
            cls_table = Table(title="Pixel Class Distribution (annotated masks)", show_lines=True)
            cls_table.add_column("Class", style="cyan")
            cls_table.add_column("Label")
            cls_table.add_column("Pixels", justify="right")
            cls_table.add_column("Fraction", justify="right")
            labels = {0: "background", 1: "left_lane", 2: "right_lane"}
            for cls, d in cls_dist.items():
                cls_table.add_row(
                    str(cls), labels.get(cls, "unknown"),
                    str(d["count"]), f"{d['fraction']:.4f}"
                )
            console.print(cls_table)
            report["class_distribution"] = cls_dist

        # ── Brightness stats ───────────────────────────────────────────────
        console.print("\n[bold]Computing brightness statistics...[/bold]")
        brightness = compute_brightness_stats(PATHS.raw_frames)
        if brightness:
            console.print(f"  Mean brightness: {brightness['mean']}")
            console.print(f"  Std deviation:   {brightness['std']}")
            console.print(f"  Min / Max:       {brightness['min']} / {brightness['max']}")
            console.print(f"  Dark frames (<80): {brightness['dark_fraction']*100:.1f}%")
            report["brightness_stats"] = brightness

    # ── Save report ────────────────────────────────────────────────────────
    out_path = PATHS.results / "metrics" / "dataset_stats.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    console.print(f"\n[green]✓ Report saved → {out_path}[/green]")


if __name__ == "__main__":
    main()
