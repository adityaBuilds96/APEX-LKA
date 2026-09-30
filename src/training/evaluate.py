"""
src/training/evaluate.py
========================
Master Evaluation Engine for APEX-LKA (LaneSegNet).

Features:
1. Multi-class semantic segmentation evaluation on test or val split.
2. High-precision latency profiling and target FPS measurement.
3. 4-column visual prediction gallery (best, worst, and median frames).
4. Failure categorization across environmental and road scenarios.
5. Structured reporting: JSON, Markdown, and self-contained HTML.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg
from src.training.dataset import LaneSegDataset
from src.training.metrics import (
    CLASS_NAMES,
    NUM_CLASSES,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_frame_metrics,
    compute_latency_statistics,
)
from src.training.model import LaneSegNet
from src.training.reports import (
    analyze_failures,
    generate_html_report,
    generate_json_report,
    generate_markdown_report,
)
from src.training.visualization import (
    generate_prediction_galleries,
    plot_confusion_matrix,
    plot_failure_distribution,
    plot_per_class_iou,
)

log = logging.getLogger("apex_evaluator")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
console = Console()


def find_default_checkpoint() -> Optional[Path]:
    """Find the best available model checkpoint for evaluation."""
    best_exported = PATHS.exported / "best_model.pth"
    if best_exported.exists():
        return best_exported

    last_ckpt = PATHS.checkpoints / "last_checkpoint.pt"
    if last_ckpt.exists():
        return last_ckpt

    ckpts = sorted(PATHS.checkpoints.glob("*.pth")) + sorted(PATHS.checkpoints.glob("*.pt"))
    if ckpts:
        ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return ckpts[0]

    return None


class ModelEvaluator:
    """Evaluates LaneSegNet on test/validation dataset splits."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        split: str = "test",
        device: Optional[str] = None,
        batch_size: int = 8,
        num_classes: int = NUM_CLASSES,
    ) -> None:
        self.split = split
        self.batch_size = max(1, batch_size)
        self.num_classes = num_classes

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Resolve checkpoint
        if checkpoint_path is not None:
            self.checkpoint_path = Path(checkpoint_path)
        else:
            self.checkpoint_path = find_default_checkpoint()

        # 2. Build and load model
        self.model = LaneSegNet(num_classes=self.num_classes, pretrained=False)
        if self.checkpoint_path and self.checkpoint_path.exists():
            log.info("Loading model weights from: %s", self.checkpoint_path)
            state = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                self.model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict) and "ema_model" in state:
                self.model.load_state_dict(state["ema_model"])
            elif isinstance(state, dict):
                self.model.load_state_dict(state)
        else:
            log.warning("No checkpoint found; running evaluation with randomly initialized model.")

        self.model.to(self.device)
        self.model.eval()

        # 3. Initialize dataset
        self.dataset = self._resolve_dataset(split)
        self.loader = (
            DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
            if self.dataset and len(self.dataset) > 0
            else None
        )

    def _resolve_dataset(self, split: str) -> Optional[LaneSegDataset]:
        """Locates split directory, falling back from test to val if test is empty."""
        split_dir = PATHS.data / split
        if not (split_dir / "images").exists() or len(list((split_dir / "images").glob("*"))) == 0:
            if split == "test":
                log.info("Test split empty; falling back to validation split ('val').")
                split_dir = PATHS.val
                self.split = "val"

        if (split_dir / "images").exists() and (split_dir / "masks").exists():
            try:
                return LaneSegDataset(
                    split_dir,
                    split=self.split,
                    height=360,
                    width=640,
                    augment=False,
                    return_metadata=True,
                )
            except Exception as e:
                log.warning("Could not load LaneSegDataset for split %s: %s", self.split, e)
        return None

    @torch.no_grad()
    def evaluate(
        self,
        export_reports: bool = True,
        generate_galleries: bool = True,
        output_dir: Union[str, Path] = "results",
    ) -> Dict[str, Any]:
        """Execute full evaluation pass."""
        out_base = Path(output_dir)
        metrics_out = out_base / "metrics"
        plots_out = out_base / "plots" / "evaluation"
        metrics_out.mkdir(parents=True, exist_ok=True)
        plots_out.mkdir(parents=True, exist_ok=True)

        if self.loader is None or len(self.dataset) == 0:
            log.warning("Evaluation dataset is empty. Returning empty metrics.")
            return {"mean_iou": 0.0, "total_frames": 0}

        console.print(
            Panel(
                f"[bold cyan]APEX-LKA Model Evaluation Engine[/bold cyan]\n"
                f"Checkpoint: {self.checkpoint_path or 'Random Initialization'}\n"
                f"Evaluation Split: {self.split.upper()} ({len(self.dataset)} frames)\n"
                f"Device: {self.device}",
                border_style="cyan",
            )
        )

        conf_mat = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)
        latencies_ms: List[float] = []
        frame_records: List[Dict[str, Any]] = []

        mean_rgb = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std_rgb = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        for batch_data in self.loader:
            images = batch_data[0].to(self.device)
            targets = batch_data[1].cpu().numpy()
            metadata_list = batch_data[2] if len(batch_data) > 2 else [{}] * len(targets)

            # Benchmark inference latency
            t0 = time.perf_counter()
            logits = self.model(images)
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()

            batch_latency = (t1 - t0) * 1000.0 / images.shape[0]
            latencies_ms.extend([batch_latency] * images.shape[0])

            preds = torch.argmax(logits, dim=1).cpu().numpy()

            # Accumulate global confusion matrix
            conf_mat += compute_confusion_matrix(preds, targets, num_classes=self.num_classes)

            # Per-frame record collection
            for i in range(len(preds)):
                pred_i = preds[i]
                target_i = targets[i]
                meta_i = {k: v[i] if isinstance(v, (list, tuple)) else v for k, v in metadata_list.items()} if isinstance(metadata_list, dict) else metadata_list[i]

                # Denormalize image for visualization
                raw_img = images[i].permute(1, 2, 0).cpu().numpy()
                raw_bgr = np.clip((raw_img * std_rgb + mean_rgb) * 255.0, 0, 255).astype(np.uint8)
                raw_bgr = cv2.cvtColor(raw_bgr, cv2.COLOR_RGB2BGR)

                frame_m = compute_frame_metrics(pred_i, target_i, num_classes=self.num_classes)
                stem = meta_i.get("stem", f"frame_{len(frame_records):04d}")

                frame_records.append({
                    "name": stem,
                    "image": raw_bgr,
                    "target": target_i,
                    "pred": pred_i,
                    "frame_miou": frame_m["frame_miou"],
                    "frame_lane_iou": frame_m["frame_lane_iou"],
                    "metadata": meta_i,
                })

        # 4. Compute comprehensive classification metrics & latency stats
        metrics_data = compute_classification_metrics(conf_mat, class_names=CLASS_NAMES)
        latency_stats = compute_latency_statistics(latencies_ms)
        failure_analysis = analyze_failures(frame_records)

        # 5. Visual plots
        if generate_galleries:
            plot_confusion_matrix(conf_mat, output_file=plots_out / "confusion_matrix.png")
            plot_per_class_iou(metrics_data["per_class"], output_file=plots_out / "per_class_iou.png")
            if failure_analysis.get("scenario_breakdown"):
                plot_failure_distribution(failure_analysis["scenario_breakdown"], output_file=plots_out / "failure_distribution.png")
            generate_prediction_galleries(frame_records, output_dir=plots_out / "predictions", max_per_group=20)

        # 6. Structured reports
        if export_reports:
            generate_json_report(metrics_data, failure_analysis, latency_stats, output_path=metrics_out / "evaluation_report.json")
            generate_markdown_report(metrics_data, failure_analysis, latency_stats, output_path=metrics_out / "evaluation_report.md")
            generate_html_report(metrics_data, failure_analysis, latency_stats, output_path=metrics_out / "evaluation_report.html")

        # 7. Print Rich console summary
        self._print_console_summary(metrics_data, latency_stats, failure_analysis)

        return {
            "metrics": metrics_data,
            "latency": latency_stats,
            "failure_analysis": failure_analysis,
        }

    def _print_console_summary(
        self,
        metrics: Dict[str, Any],
        latency: Dict[str, Any],
        failures: Dict[str, Any],
    ) -> None:
        tbl = Table(title="APEX-LKA Test Set Performance Summary", show_lines=True)
        tbl.add_column("Class", style="cyan")
        tbl.add_column("IoU", justify="right")
        tbl.add_column("Dice (F1)", justify="right")
        tbl.add_column("Precision", justify="right")
        tbl.add_column("Recall", justify="right")
        tbl.add_column("Accuracy", justify="right")

        for c_name, vals in metrics.get("per_class", {}).items():
            tbl.add_row(
                c_name,
                f"{vals['iou']:.4f}",
                f"{vals['dice']:.4f}",
                f"{vals['precision']:.4f}",
                f"{vals['recall']:.4f}",
                f"{vals['pixel_accuracy']:.4f}",
            )

        tbl.add_row(
            "[bold green]MEAN / TOTAL[/bold green]",
            f"[bold cyan]{metrics['mean_iou']:.4f}[/bold cyan]",
            f"[bold green]{metrics['mean_dice']:.4f}[/bold green]",
            f"{metrics['mean_precision']:.4f}",
            f"{metrics['mean_recall']:.4f}",
            f"{metrics['overall_pixel_accuracy']:.4f}",
        )
        console.print(tbl)
        console.print(
            f"[bold]Inference Speed:[/bold] {latency['p50_ms']:.1f}ms / frame "
            f"([bold cyan]{latency['fps']:.1f} FPS[/bold cyan]) | "
            f"[bold]Severe Failures:[/bold] {failures['severe_failures_count']} frames"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-LKA Model Evaluation Engine")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint .pth/.pt")
    parser.add_argument("--split", type=str, choices=["test", "val", "train"], default="test", help="Dataset split to evaluate")
    parser.add_argument("--batch-size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Target execution device")
    parser.add_argument("--export-report", action="store_true", default=True, help="Generate JSON, MD, and HTML reports")
    args = parser.parse_args()

    evaluator = ModelEvaluator(
        checkpoint_path=args.checkpoint,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
    )
    evaluator.evaluate(export_reports=args.export_report)


if __name__ == "__main__":
    main()
