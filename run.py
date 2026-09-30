"""
run.py
======
Master entry point for the LKA (Lane Keep Assist) system.

Usage
-----
  python run.py --help
  python run.py collect --video data/raw_videos/drive01.mp4 --fps 5
  python run.py inspect
  python run.py split
  python run.py train
  python run.py evaluate
  python run.py infer --source webcam
  python run.py infer --source video --file data/raw_videos/test.mp4
  python run.py dashboard
  python run.py env           # Show environment info
"""

import argparse
import subprocess
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

console = Console()


def cmd_env(args):
    """Show environment and project status."""
    import platform
    import importlib

    console.print(Panel("[bold cyan]LKA Environment Check", border_style="cyan"))

    env_table = Table(show_lines=True)
    env_table.add_column("Component", style="cyan")
    env_table.add_column("Status")
    env_table.add_column("Version / Info")

    env_table.add_row("Python", "[green]OK", platform.python_version())
    env_table.add_row("Platform", "[green]OK", platform.platform())

    packages = {
        "torch":          "PyTorch",
        "cv2":            "OpenCV",
        "numpy":          "NumPy",
        "pandas":         "Pandas",
        "streamlit":      "Streamlit",
        "albumentations": "Albumentations",
        "matplotlib":     "Matplotlib",
        "sklearn":        "scikit-learn",
        "tqdm":           "tqdm",
        "rich":           "Rich",
        "yaml":           "PyYAML",
    }

    for module, name in packages.items():
        try:
            m = importlib.import_module(module)
            ver = getattr(m, "__version__", "installed")
            env_table.add_row(name, "[green]OK", ver)
        except ImportError:
            env_table.add_row(name, "[red]MISSING", "pip install " + module)

    # GPU check
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1)
            env_table.add_row("GPU", "[green]CUDA", f"{gpu} ({mem} GB)")
        else:
            env_table.add_row("GPU", "[yellow]CPU only", "Training will be slower")
    except Exception:
        env_table.add_row("GPU", "[yellow]Unknown", "")

    console.print(env_table)

    # Dataset status
    from src.config import PATHS
    ds_table = Table(title="Dataset Status", show_lines=True)
    ds_table.add_column("Directory", style="cyan")
    ds_table.add_column("Status")

    img_exts = {".jpg", ".jpeg", ".png"}
    dirs_to_check = {
        "raw_videos":  (PATHS.raw_videos, {".mp4", ".avi", ".mov", ".mkv"}),
        "raw_frames":  (PATHS.raw_frames, img_exts),
        "annotated":   (PATHS.annotated / "images", img_exts),
        "train":       (PATHS.train / "images", img_exts),
        "val":         (PATHS.val / "images", img_exts),
        "test":        (PATHS.test / "images", img_exts),
    }
    for name, (d, exts) in dirs_to_check.items():
        if d.exists():
            count = len([f for f in d.iterdir() if f.suffix.lower() in exts])
            color = "green" if count > 0 else "yellow"
            ds_table.add_row(name, f"[{color}]{count} files")
        else:
            ds_table.add_row(name, "[red]directory missing")

    console.print(ds_table)


def cmd_collect(args):
    """Run video frame extraction."""
    script = PROJECT_ROOT / "src" / "data_collection" / "video_to_frames.py"
    cmd = [sys.executable, str(script)]
    if args.video:
        cmd += ["--video", args.video]
    if args.video_dir:
        cmd += ["--video_dir", args.video_dir]
    if args.fps:
        cmd += ["--fps", str(args.fps)]
    subprocess.run(cmd, check=True)


def cmd_inspect(args):
    """Run dataset inspector."""
    script = PROJECT_ROOT / "src" / "data_collection" / "inspect_dataset.py"
    cmd = [sys.executable, str(script)]
    if args.stats_only:
        cmd += ["--stats-only"]
    if args.show:
        cmd += ["--show", str(args.show)]
    subprocess.run(cmd, check=True)


def cmd_split(args):
    """Split annotated data into train/val/test."""
    script = PROJECT_ROOT / "src" / "data_collection" / "dataset_splitter.py"
    cmd = [sys.executable, str(script)]
    if args.dry_run:
        cmd += ["--dry-run"]
    subprocess.run(cmd, check=True)


def cmd_train(args):
    """Run bulletproof training pipeline."""
    script = PROJECT_ROOT / "src" / "training" / "train.py"
    if not script.exists():
        console.print(
            "[yellow]Training pipeline not found at src/training/train.py.[/yellow]"
        )
        return
    cmd = [sys.executable, str(script)]
    if getattr(args, "epochs", None) is not None:
        cmd += ["--epochs", str(args.epochs)]
    if getattr(args, "batch_size", None) is not None:
        cmd += ["--batch-size", str(args.batch_size)]
    if getattr(args, "lr", None) is not None:
        cmd += ["--lr", str(args.lr)]
    if getattr(args, "resume", None) is not None:
        cmd += ["--resume", str(args.resume)]
    if getattr(args, "device", None) is not None:
        cmd += ["--device", str(args.device)]
    subprocess.run(cmd, check=True)


def cmd_evaluate(args):
    """Run model evaluation."""
    script = PROJECT_ROOT / "src" / "training" / "evaluate.py"
    if not script.exists():
        console.print("[yellow]Evaluation script not yet implemented.[/yellow]")
        return
    subprocess.run([sys.executable, str(script)], check=True)


def cmd_infer(args):
    """Run live inference."""
    script = PROJECT_ROOT / "src" / "inference" / "live_inference.py"
    if not script.exists():
        console.print("[yellow]Inference engine not yet implemented.[/yellow]")
        return
    cmd = [sys.executable, str(script), "--source", args.source]
    if args.file:
        cmd += ["--file", args.file]
    subprocess.run(cmd, check=True)


def cmd_dashboard(args):
    """Launch Streamlit dashboard."""
    dashboard = PROJECT_ROOT / "dashboard" / "app.py"
    if not dashboard.exists():
        console.print("[yellow]Dashboard not yet implemented.[/yellow]")
        return
    subprocess.run(
        ["streamlit", "run", str(dashboard)],
        check=True,
    )


def cmd_ingest(args):
    """Ingest and validate road dataset from ZIP or directory."""
    from src.dataset.ingestion import DatasetIngestor
    source = args.zip or args.dir or args.source
    if not source:
        console.print("[red]Please specify dataset source via --zip <path> or --dir <path>[/red]")
        return
    ingestor = DatasetIngestor(base_dir=PROJECT_ROOT)
    target = Path(args.target) if args.target else None
    report = ingestor.ingest(source, target_dir=target)

    console.print(
        Panel(
            f"[bold cyan]APEX-RLP Dataset Ingestion: {report.mode}[/bold cyan]\n"
            f"Status: {report.status_message}",
            border_style="cyan",
        )
    )

    table = Table(title="Dataset Ingestion Summary", show_lines=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_row("Total Images", str(report.total_images))
    table.add_row("Valid Images", f"[green]{report.valid_images}[/green]")
    table.add_row("Corrupted Images", f"[red]{report.corrupted_images}[/red]" if report.corrupted_images else "0")
    table.add_row("Exact Duplicates", str(report.exact_duplicates))
    table.add_row("Near Duplicates", str(report.near_duplicates))
    table.add_row("Masks Found", "YES" if report.masks_found else "[yellow]NO[/yellow]")
    table.add_row("Matched Pairs", f"[green]{report.matched_pairs}[/green]" if report.matched_pairs else "0")
    table.add_row("Workflow Mode", f"[bold green]{report.mode}[/bold green]" if report.mode == "MODE_A_LABELED" else f"[bold yellow]{report.mode}[/bold yellow]")
    table.add_row("Supervised Training Ready", "[green]YES[/green]" if report.can_train_supervised else "[red]NO (Annotation Required)[/red]")
    console.print(table)
def cmd_auto_annotate(args):
    """Generate pseudo-masks for all unannotated frames via Classical CV."""
    import cv2
    from src.config import PATHS
    from src.dataset.auto_annotator import ClassicalAutoAnnotator

    img_dir = PATHS.annotated / "images"
    mask_dir = PATHS.annotated / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in sorted(img_dir.iterdir()) if f.is_file() and f.suffix.lower() in img_exts] if img_dir.exists() else []

    if not images:
        console.print(f"[yellow]No images found in {img_dir}. Ingest or stage frames first.[/yellow]")
        return

    annotator = ClassicalAutoAnnotator()
    force = getattr(args, "force", False)
    conf_thresh = getattr(args, "confidence_threshold", 0.70)

    n_high = 0
    n_med = 0
    n_low = 0
    total_gen = 0

    with console.status("[bold cyan]Generating pseudo-masks via Classical CV...[/bold cyan]"):
        for img_path in images:
            stem = img_path.stem
            out_mask_path = mask_dir / f"{stem}.png"
            if out_mask_path.exists() and not force:
                continue

            bgr = cv2.imread(str(img_path))
            if bgr is None:
                continue

            mask, meta = annotator.generate_pseudo_mask(img_path)
            cv2.imwrite(str(out_mask_path), mask)
            total_gen += 1
            conf = meta.confidence

            if conf >= conf_thresh:
                n_high += 1
            elif conf >= 0.40:
                n_med += 1
            else:
                n_low += 1

    console.print(
        Panel(
            f"[bold green]Auto-Annotation Complete[/bold green]\n\n"
            f"Generated [bold cyan]{total_gen}[/bold cyan] masks: "
            f"[green]{n_high} HIGH[/green], [yellow]{n_med} MEDIUM[/yellow], [red]{n_low} LOW[/red]\n"
            f"Destination: [cyan]{mask_dir}[/cyan]",
            border_style="green",
        )
    )


def cmd_quality_check(args):
    """Run QA validation on all annotated image-mask pairs."""
    import json
    import cv2
    from src.config import PATHS
    from src.dataset.quality_scorer import QualityScorer, QualityTier, Severity

    img_dir = PATHS.annotated / "images"
    mask_dir = PATHS.annotated / "masks"

    if not img_dir.exists() or not mask_dir.exists():
        console.print("[red]Annotated images or masks directory missing.[/red]")
        return

    scorer = QualityScorer()
    img_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = [f for f in sorted(img_dir.iterdir()) if f.is_file() and f.suffix.lower() in img_exts]

    reports = []
    has_critical = False

    table = Table(title="Annotation Quality Assurance Summary", show_lines=True)
    table.add_column("Stem", style="cyan")
    table.add_column("Tier")
    table.add_column("Score", justify="right")
    table.add_column("Issues Flagged")

    with console.status("[bold cyan]Auditing annotations with 9-check QA system...[/bold cyan]"):
        for img_p in images:
            stem = img_p.stem
            mask_p = mask_dir / f"{stem}.png"
            if not mask_p.exists():
                continue

            img_bgr = cv2.imread(str(img_p))
            mask = cv2.imread(str(mask_p), cv2.IMREAD_GRAYSCALE)
            if img_bgr is None or mask is None:
                continue

            rep = scorer.evaluate(img_bgr, mask, stem=stem)
            reports.append(rep)

            if any(i.severity == Severity.CRITICAL for i in rep.issues):
                has_critical = True

            tier_col = (
                "[green]EXCELLENT[/green]" if rep.tier == QualityTier.EXCELLENT else
                "[cyan]ACCEPTABLE[/cyan]" if rep.tier == QualityTier.ACCEPTABLE else
                "[yellow]NEEDS_REVIEW[/yellow]" if rep.tier == QualityTier.NEEDS_REVIEW else
                "[red]REJECTED[/red]"
            )
            issues_str = ", ".join(f"[{i.check_name}] {i.message}" for i in rep.issues[:2])
            if len(rep.issues) > 2:
                issues_str += f" (+{len(rep.issues)-2} more)"
            if not issues_str:
                issues_str = "[green]All checks passed[/green]"

            table.add_row(stem, tier_col, f"{rep.overall_score:.2f}", issues_str)

    console.print(table)

    if reports:
        avg_score = sum(r.overall_score for r in reports) / len(reports)
        console.print(f"[bold]Overall Dataset QA Score:[/bold] [cyan]{avg_score:.2f} / 1.00[/cyan] ({len(reports)} pairs checked)")

    if getattr(args, "export", False):
        export_dir = PROJECT_ROOT / "results" / "metrics"
        export_dir.mkdir(parents=True, exist_ok=True)
        export_path = export_dir / "annotation_quality.json"

        export_data = {
            "total_pairs": len(reports),
            "average_score": round(sum(r.overall_score for r in reports) / max(1, len(reports)), 3),
            "critical_count": sum(1 for r in reports if any(i.severity == Severity.CRITICAL for i in r.issues)),
            "reports": [
                {
                    "stem": r.stem,
                    "score": r.overall_score,
                    "tier": r.tier.value,
                    "issues": [{"check": i.check_name, "severity": i.severity.value, "msg": i.message} for i in r.issues],
                }
                for r in reports
            ]
        }
        with open(export_path, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2)
        console.print(f"[green]Saved QA report -> {export_path}[/green]")

    if getattr(args, "strict", False) and has_critical:
        console.print("[bold red]STRICT MODE FAILED: Critical annotation defects detected.[/bold red]")
        sys.exit(1)


def cmd_pipeline(args):
    """Run full zero-touch automated data pipeline."""
    from src.dataset.pipeline_orchestrator import PipelineOrchestrator

    source = args.source or getattr(args, "zip", None) or getattr(args, "dir", None)
    if not source:
        console.print("[red]Please specify dataset source via --source <path>[/red]")
        return

    orchestrator = PipelineOrchestrator(base_dir=PROJECT_ROOT)
    console.print(Panel(f"[bold cyan]APEX-RLP Intelligent Pipeline Orchestrator[/bold cyan]\nSource: {source}", border_style="cyan"))

    auto_split = getattr(args, "auto_split", False)
    summary = orchestrator.run(source_path=source, auto_split=auto_split)

    table = Table(title="Pipeline Execution Summary", show_lines=True)
    table.add_column("Stage / Metric", style="cyan")
    table.add_column("Result", justify="right")
    table.add_row("Batch ID", summary.batch_id[:8])
    table.add_row("Input Files", str(summary.total_input_files))
    table.add_row("Validated Frames", f"[green]{summary.validated_count}[/green]")
    table.add_row("Corrupted Quarantined", f"[red]{summary.corrupted_count}[/red]" if summary.corrupted_count else "0")
    table.add_row("Duplicates Filtered", str(summary.duplicate_count))
    table.add_row("Auto-Annotated Masks", f"[bold green]{summary.annotated_count}[/bold green]")
    table.add_row("Average Quality Score", f"{summary.avg_quality_score:.2f} / 1.00")
    table.add_row("Splits (Train/Val/Test)", f"{summary.train_count} / {summary.val_count} / {summary.test_count}")
    table.add_row("Elapsed Time", f"{summary.elapsed_seconds:.2f}s")
    console.print(table)

    if summary.report_path:
        console.print(f"[green]Report saved -> {summary.report_path}[/green]")


def main():
    parser = argparse.ArgumentParser(
        description="LKA (Lane Keep Assist) — Master Entry Point",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(title="commands", dest="command")

    # env
    sub.add_parser("env", help="Show environment and dataset status")

    # collect
    p_collect = sub.add_parser("collect", help="Extract frames from video")
    p_collect.add_argument("--video", type=str, help="Path to a single video")
    p_collect.add_argument("--video_dir", type=str, help="Directory of videos")
    p_collect.add_argument("--fps", type=float, help="Target extraction FPS")

    # inspect
    p_inspect = sub.add_parser("inspect", help="Inspect dataset")
    p_inspect.add_argument("--stats-only", action="store_true")
    p_inspect.add_argument("--show", type=int, default=9)

    # split
    p_split = sub.add_parser("split", help="Split annotated data into train/val/test")
    p_split.add_argument("--dry-run", action="store_true")

    # train
    p_train = sub.add_parser("train", help="Train the lane detection model")
    p_train.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    p_train.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    p_train.add_argument("--lr", type=float, default=None, help="Initial learning rate")
    p_train.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    p_train.add_argument("--device", type=str, choices=["cuda", "cpu"], default=None, help="Target device")

    # evaluate
    sub.add_parser("evaluate", help="Evaluate model on test set")

    # infer
    p_infer = sub.add_parser("infer", help="Run live inference")
    p_infer.add_argument("--source", choices=["webcam", "video", "image"],
                         default="webcam")
    p_infer.add_argument("--file", type=str, help="Path for video/image source")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest and validate dataset from ZIP or directory")
    p_ingest.add_argument("--zip", type=str, help="Path to road_dataset.zip")
    p_ingest.add_argument("--dir", type=str, help="Path to extracted dataset directory")
    p_ingest.add_argument("--source", type=str, help="Alias for --zip or --dir")
    p_ingest.add_argument("--target", type=str, default=None, help="Target destination (default: data/annotated)")
    p_ingest.add_argument("--auto-annotate", action="store_true", help="Generate pseudo-masks via Classical CV")
    p_ingest.add_argument("--auto-split", action="store_true", help="Automatically partition into train/val/test")

    # auto-annotate (Section 9.1)
    p_auto = sub.add_parser("auto-annotate", help="Generate pseudo-masks for all unannotated frames")
    p_auto.add_argument("--force", action="store_true", help="Overwrite existing auto-generated masks")
    p_auto.add_argument("--confidence-threshold", type=float, default=0.70, help="Confidence threshold for auto-approval")

    # quality-check (Section 9.1)
    p_qa = sub.add_parser("quality-check", help="Run QA validation on all annotated image-mask pairs")
    p_qa.add_argument("--strict", action="store_true", help="Fail with exit code 1 if critical defects found")
    p_qa.add_argument("--export", action="store_true", help="Save report to results/metrics/annotation_quality.json")

    # pipeline (full zero-touch automation)
    p_pipe = sub.add_parser("pipeline", help="Run full automated pipeline on new data")
    p_pipe.add_argument("--source", type=str, help="Path to archive, video, image, or folder")
    p_pipe.add_argument("--zip", type=str, help="Path to ZIP archive")
    p_pipe.add_argument("--dir", type=str, help="Path to dataset directory")
    p_pipe.add_argument("--auto-split", action="store_true", help="Automatically split after annotation")

    args = parser.parse_args()

    if args.command is None:
        console.print(
            Panel(
                "[bold cyan]Lane Keep Assist (LKA) Project[/bold cyan]\n\n"
                "Run [bold]python run.py --help[/bold] for available commands.\n\n"
                "[bold]Quick start:[/bold]\n"
                "  1. [cyan]python run.py env[/cyan]              — Check environment\n"
                "  2. [cyan]python run.py ingest --zip road.zip[/cyan] — Ingest dataset\n"
                "  3. [cyan]python run.py pipeline --source data/raw_videos/drive.mp4[/cyan] — Full automated pipeline\n"
                "  4. [cyan]python run.py auto-annotate[/cyan]    — Generate pseudo-masks via Classical CV\n"
                "  5. [cyan]python run.py quality-check[/cyan]     — Run 9-check QA audit\n"
                "  6. [cyan]python run.py inspect[/cyan]          — Inspect frames\n"
                "  7. [cyan]python run.py split[/cyan]            — Split dataset\n"
                "  8. [cyan]python run.py train[/cyan]            — Train model\n"
                "  9. [cyan]python run.py evaluate[/cyan]         — Evaluate\n"
                " 10. [cyan]python run.py infer --source webcam[/cyan] — Live demo",
                title="[bold green]Welcome",
                border_style="green",
            )
        )
        return

    dispatch = {
        "env":           cmd_env,
        "collect":       cmd_collect,
        "inspect":       cmd_inspect,
        "split":         cmd_split,
        "train":         cmd_train,
        "evaluate":      cmd_evaluate,
        "infer":         cmd_infer,
        "dashboard":     cmd_dashboard,
        "ingest":        cmd_ingest,
        "auto-annotate": cmd_auto_annotate,
        "quality-check": cmd_quality_check,
        "pipeline":      cmd_pipeline,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

