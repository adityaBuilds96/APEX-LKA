"""
video_to_frames.py
==================
Extract frames from road-driving videos into data/raw_frames/.

Features
--------
* Configurable extraction FPS (default from config; override with --fps)
* Duplicate-frame detection via pixel-difference thumbnail comparison
* Overwite protection  — skips frames already on disk
* Rich progress bar with ETA
* Metadata CSV at data/raw_frames/metadata.csv
  (columns: filename, source_video, frame_number, timestamp_s, width, height)
* Supports MP4, AVI, MOV, MKV, M4V, WMV, FLV

Usage
-----
  # Single video
  python src/data_collection/video_to_frames.py --video data/raw_videos/drive01.mp4

  # Custom FPS
  python src/data_collection/video_to_frames.py --video data/raw_videos/drive01.mp4 --fps 3

  # All videos in a directory
  python src/data_collection/video_to_frames.py --video_dir data/raw_videos --fps 5

  # See all options
  python src/data_collection/video_to_frames.py --help

Example (copy your video first, then run):
  python src/data_collection/video_to_frames.py --video data/raw_videos/my_drive.mp4 --fps 5
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

# ── Resolve project root so this script works from any working directory ───
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg  # noqa: E402

console = Console()

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}

# Metadata CSV lives alongside the extracted frames
METADATA_CSV = PATHS.raw_frames / "metadata.csv"
METADATA_FIELDS = [
    "filename",
    "source_video",
    "frame_number",
    "timestamp_s",
    "width",
    "height",
]


# ═══════════════════════════════════════════════════════════════════════════
# Frame extractor
# ═══════════════════════════════════════════════════════════════════════════

class FrameExtractor:
    """
    Extracts frames from a single video file at a target FPS.

    Duplicate detection
    -------------------
    After interval-based subsampling we compare a 64×64 grayscale thumbnail
    of the current frame against the previous SAVED frame using mean absolute
    pixel difference.  If the difference is below `similarity_threshold`
    (on a 0–255 scale), the frame is treated as a near-duplicate and skipped.
    A threshold of 4.0 works well for typical road footage.  Raise it if you
    want to keep more frames; lower it to be more aggressive about skipping.

    Overwrite protection
    --------------------
    Before writing, we check whether the output filename already exists on
    disk.  Existing frames are NEVER overwritten; they are counted separately.
    This makes it safe to re-run the extractor after a partial run.
    """

    def __init__(
        self,
        video_path: Path,
        output_dir: Path,
        target_fps: float = 5.0,
        similarity_threshold: float = 4.0,
        output_format: str = "jpg",
        jpeg_quality: int = 95,
    ):
        self.video_path = video_path
        self.output_dir = output_dir
        self.target_fps = target_fps
        self.similarity_threshold = similarity_threshold
        self.output_format = output_format.lower().strip(".")
        self.jpeg_quality = jpeg_quality
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _thumb(self, frame: np.ndarray) -> np.ndarray:
        """64×64 grayscale thumbnail used for duplicate comparison."""
        return cv2.cvtColor(
            cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

    # ── Main extraction method ─────────────────────────────────────────────

    def extract(self) -> dict:
        """
        Run the extraction loop.

        Returns
        -------
        dict
            video_name, total_frames_in_video, frames_extracted,
            frames_skipped_duplicate, frames_skipped_interval,
            frames_skipped_exists, duration_seconds, video_fps,
            resolution, metadata_rows
        """
        cap = cv2.VideoCapture(str(self.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")

        video_fps: float = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width: int  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_s: float = total_frames / video_fps

        # Interval = how many source frames to skip between kept frames
        frame_interval = max(1, int(round(video_fps / self.target_fps)))

        console.print(
            Panel(
                f"[bold]Video:[/bold]          {self.video_path.name}\n"
                f"[bold]Resolution:[/bold]     {width}x{height}\n"
                f"[bold]Source FPS:[/bold]     {video_fps:.2f}\n"
                f"[bold]Target FPS:[/bold]     {self.target_fps}\n"
                f"[bold]Frame interval:[/bold] every {frame_interval} frames\n"
                f"[bold]Duration:[/bold]       {duration_s:.1f}s  "
                f"({total_frames} total frames)\n"
                f"[bold]Output dir:[/bold]     {self.output_dir}",
                title="[bold green]Frame Extractor",
                border_style="green",
            )
        )

        stem = self.video_path.stem
        metadata_rows: list[dict] = []
        frames_extracted = 0
        frames_skipped_dup = 0
        frames_skipped_interval = 0
        frames_skipped_exists = 0
        prev_thumb: np.ndarray | None = None

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Extracting {self.video_path.name}", total=total_frames
            )

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                progress.update(task, advance=1)

                # ── 1. Interval filter ─────────────────────────────────────
                if frame_idx % frame_interval != 0:
                    frames_skipped_interval += 1
                    frame_idx += 1
                    continue

                # ── 2. Overwrite protection ────────────────────────────────
                out_filename = (
                    f"{stem}_frame{frame_idx:06d}.{self.output_format}"
                )
                out_path = self.output_dir / out_filename
                if out_path.exists():
                    frames_skipped_exists += 1
                    frame_idx += 1
                    continue

                # ── 3. Duplicate check (only on interval-selected frames) ──
                thumb = self._thumb(frame)
                if prev_thumb is not None:
                    diff = np.mean(np.abs(thumb - prev_thumb))
                    if diff < self.similarity_threshold:
                        frames_skipped_dup += 1
                        frame_idx += 1
                        continue

                prev_thumb = thumb

                # ── 4. Write frame ─────────────────────────────────────────
                if self.output_format == "jpg":
                    cv2.imwrite(
                        str(out_path),
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                    )
                else:
                    cv2.imwrite(str(out_path), frame)

                timestamp_s = frame_idx / video_fps
                metadata_rows.append(
                    {
                        "filename":     out_filename,
                        "source_video": self.video_path.name,
                        "frame_number": frame_idx,
                        "timestamp_s":  round(timestamp_s, 4),
                        "width":        width,
                        "height":       height,
                    }
                )
                frames_extracted += 1
                frame_idx += 1

        cap.release()

        return {
            "video_name":             self.video_path.name,
            "total_frames_in_video":  total_frames,
            "frames_extracted":       frames_extracted,
            "frames_skipped_duplicate": frames_skipped_dup,
            "frames_skipped_interval":  frames_skipped_interval,
            "frames_skipped_exists":    frames_skipped_exists,
            "duration_seconds":       round(duration_s, 2),
            "video_fps":              round(video_fps, 2),
            "resolution":             f"{width}x{height}",
            "metadata_rows":          metadata_rows,
        }


# ═══════════════════════════════════════════════════════════════════════════
# Metadata writer — data/raw_frames/metadata.csv
# ═══════════════════════════════════════════════════════════════════════════

def save_metadata(rows: list[dict], csv_path: Path = METADATA_CSV) -> None:
    """
    Append new rows to the metadata CSV (creates headers if file is new).

    The CSV is stored at data/raw_frames/metadata.csv so it lives alongside
    the extracted frames — no separate annotations directory needed at this stage.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=METADATA_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)

    console.print(
        f"[green]Metadata saved[/green] -> {csv_path}  "
        f"({len(rows)} new rows)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Summary table
# ═══════════════════════════════════════════════════════════════════════════

def print_summary(results: list[dict]) -> None:
    table = Table(title="[bold]Extraction Summary", show_lines=True)
    table.add_column("Video",          style="cyan", no_wrap=True)
    table.add_column("Duration",       justify="right")
    table.add_column("Src FPS",        justify="right")
    table.add_column("Resolution",     justify="right")
    table.add_column("Extracted",      justify="right", style="green")
    table.add_column("Skip(interval)", justify="right")
    table.add_column("Skip(dup)",      justify="right", style="yellow")
    table.add_column("Skip(exists)",   justify="right", style="yellow")

    total_extracted = 0
    for r in results:
        table.add_row(
            r["video_name"],
            f"{r['duration_seconds']}s",
            str(r["video_fps"]),
            r["resolution"],
            str(r["frames_extracted"]),
            str(r["frames_skipped_interval"]),
            str(r["frames_skipped_duplicate"]),
            str(r["frames_skipped_exists"]),
        )
        total_extracted += r["frames_extracted"]

    console.print(table)
    console.print(
        f"\n[bold green]Total frames extracted this run: "
        f"{total_extracted}[/bold green]"
    )


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="video_to_frames.py",
        description=(
            "Extract frames from road-driving video(s) for the LKA dataset.\n"
            "Frames are saved to data/raw_frames/  with a metadata.csv file."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python src/data_collection/video_to_frames.py "
            "--video data/raw_videos/drive01.mp4\n"
            "  python src/data_collection/video_to_frames.py "
            "--video data/raw_videos/drive01.mp4 --fps 3\n"
            "  python src/data_collection/video_to_frames.py "
            "--video_dir data/raw_videos --fps 5\n"
        ),
    )
    parser.add_argument(
        "--video", type=str, default=None,
        metavar="PATH",
        help="Path to a single video file.",
    )
    parser.add_argument(
        "--video_dir", type=str, default=None,
        metavar="DIR",
        help="Directory containing multiple video files (all will be processed).",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        metavar="DIR",
        help=f"Output directory for frames. Default: {PATHS.raw_frames}",
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        metavar="N",
        help=(
            f"Frames to extract per second of video. "
            f"Default from config: {cfg['data_collection']['target_fps']}"
        ),
    )
    parser.add_argument(
        "--dup_threshold", type=float, default=4.0,
        metavar="DIFF",
        help=(
            "Mean pixel difference (0-255) below which a frame is "
            "considered a near-duplicate and skipped. Default: 4.0\n"
            "Higher = keep more frames; lower = more aggressive dedup."
        ),
    )
    parser.add_argument(
        "--format", type=str, default="jpg", choices=["jpg", "png"],
        help="Output image format. Default: jpg",
    )
    parser.add_argument(
        "--quality", type=int, default=95,
        metavar="1-100",
        help="JPEG quality (1-100). Only used when --format jpg. Default: 95",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    target_fps = args.fps or cfg["data_collection"]["target_fps"]
    output_dir = Path(args.out) if args.out else PATHS.raw_frames
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    # ── Collect video paths ────────────────────────────────────────────────
    video_paths: list[Path] = []

    if args.video:
        p = Path(args.video)
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        if not p.exists():
            console.print(f"[bold red]ERROR: Video not found: {p}[/bold red]")
            sys.exit(1)
        video_paths.append(p)

    elif args.video_dir:
        d = Path(args.video_dir)
        if not d.is_absolute():
            d = PROJECT_ROOT / d
        if not d.exists():
            console.print(f"[bold red]ERROR: Directory not found: {d}[/bold red]")
            sys.exit(1)
        video_paths = sorted(
            f for f in d.iterdir() if f.suffix.lower() in VIDEO_EXTENSIONS
        )
        if not video_paths:
            console.print(
                f"[yellow]No video files found in {d}\n"
                f"Supported: {', '.join(sorted(VIDEO_EXTENSIONS))}[/yellow]"
            )
            sys.exit(0)

    else:
        # Auto-scan default raw_videos directory
        video_paths = sorted(
            f for f in PATHS.raw_videos.iterdir()
            if f.suffix.lower() in VIDEO_EXTENSIONS
        ) if PATHS.raw_videos.exists() else []

        if not video_paths:
            console.print(
                Panel(
                    "[bold yellow]No video found.[/bold yellow]\n\n"
                    "To start extracting frames:\n\n"
                    "  [bold]1.[/bold] Record a road-driving video with your "
                    "phone or dashcam.\n"
                    "  [bold]2.[/bold] Copy the file into:\n"
                    f"          [cyan]{PATHS.raw_videos}[/cyan]\n"
                    "  [bold]3.[/bold] Re-run this script:\n"
                    "          [cyan]python src/data_collection/"
                    "video_to_frames.py[/cyan]\n\n"
                    "See README.md -> 'Data Collection Guide' for advice on\n"
                    "camera position, resolution, and driving conditions.",
                    title="[bold red]No Video Found",
                    border_style="red",
                )
            )
            sys.exit(0)

    # ── Run extraction ─────────────────────────────────────────────────────
    console.print(
        f"\n[bold]Processing {len(video_paths)} video(s) at "
        f"{target_fps} FPS...[/bold]\n"
    )

    all_results: list[dict] = []
    all_metadata: list[dict] = []
    t0 = time.time()

    for vp in video_paths:
        extractor = FrameExtractor(
            video_path=vp,
            output_dir=output_dir,
            target_fps=target_fps,
            similarity_threshold=args.dup_threshold,
            output_format=args.format,
            jpeg_quality=args.quality,
        )
        result = extractor.extract()
        all_results.append(result)
        all_metadata.extend(result["metadata_rows"])

        console.print(
            f"  [green]Done[/green] {vp.name}: "
            f"[bold]{result['frames_extracted']}[/bold] extracted, "
            f"[yellow]{result['frames_skipped_duplicate']}[/yellow] dup skipped, "
            f"[yellow]{result['frames_skipped_exists']}[/yellow] already existed\n"
        )

    # ── Save metadata ──────────────────────────────────────────────────────
    if all_metadata:
        save_metadata(all_metadata)
    else:
        console.print(
            "[yellow]No new frames extracted — nothing added to metadata.[/yellow]"
        )

    # ── Summary ────────────────────────────────────────────────────────────
    print_summary(all_results)
    elapsed = time.time() - t0
    console.print(f"\n[bold]Elapsed: {elapsed:.1f}s[/bold]")
    console.print(
        "\n[bold green]Next step:[/bold green]\n"
        "  Inspect your frames:\n"
        "  [cyan]python src/data_collection/inspect_dataset.py[/cyan]"
    )


if __name__ == "__main__":
    main()
