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
    """Run training pipeline."""
    script = PROJECT_ROOT / "src" / "training" / "train.py"
    if not script.exists():
        console.print(
            "[yellow]Training pipeline not yet implemented. "
            "Complete annotation and splitting first.[/yellow]"
        )
        return
    subprocess.run([sys.executable, str(script)], check=True)


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
    sub.add_parser("train", help="Train the lane detection model")

    # evaluate
    sub.add_parser("evaluate", help="Evaluate model on test set")

    # infer
    p_infer = sub.add_parser("infer", help="Run live inference")
    p_infer.add_argument("--source", choices=["webcam", "video", "image"],
                         default="webcam")
    p_infer.add_argument("--file", type=str, help="Path for video/image source")

    # dashboard
    sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    args = parser.parse_args()

    if args.command is None:
        console.print(
            Panel(
                "[bold cyan]Lane Keep Assist (LKA) Project[/bold cyan]\n\n"
                "Run [bold]python run.py --help[/bold] for available commands.\n\n"
                "[bold]Quick start:[/bold]\n"
                "  1. [cyan]python run.py env[/cyan]              — Check environment\n"
                "  2. Copy video to data/raw_videos/\n"
                "  3. [cyan]python run.py collect --fps 5[/cyan]  — Extract frames\n"
                "  4. [cyan]python run.py inspect[/cyan]          — Inspect frames\n"
                "  5. Annotate frames (see README.md)\n"
                "  6. [cyan]python run.py split[/cyan]            — Split dataset\n"
                "  7. [cyan]python run.py train[/cyan]            — Train model\n"
                "  8. [cyan]python run.py evaluate[/cyan]         — Evaluate\n"
                "  9. [cyan]python run.py infer --source webcam[/cyan] — Live demo",
                title="[bold green]Welcome",
                border_style="green",
            )
        )
        return

    dispatch = {
        "env":       cmd_env,
        "collect":   cmd_collect,
        "inspect":   cmd_inspect,
        "split":     cmd_split,
        "train":     cmd_train,
        "evaluate":  cmd_evaluate,
        "infer":     cmd_infer,
        "dashboard": cmd_dashboard,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()

