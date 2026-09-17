"""
inspect_dataset.py
==================
Comprehensive inspection tool for the LKA raw frames dataset.

Reports
-------
1. Total image count
2. Image resolution statistics (min, max, mode, mixed-resolution warning)
3. Corrupted / unreadable image detection
4. Number of source videos represented
5. Frames per source video
6. Random sample frame viewer (optional)

Usage
-----
  python src/data_collection/inspect_dataset.py
  python src/data_collection/inspect_dataset.py --show 12
  python src/data_collection/inspect_dataset.py --stats-only
  python src/data_collection/inspect_dataset.py --no-corrupt-check   # faster
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS  # noqa: E402

console = Console()

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
METADATA_CSV = PATHS.raw_frames / "metadata.csv"


# ═══════════════════════════════════════════════════════════════════════════
# 1. File discovery
# ═══════════════════════════════════════════════════════════════════════════

def find_images(directory: Path) -> list[Path]:
    """Return sorted list of image files in directory (non-recursive)."""
    if not directory.exists():
        return []
    return sorted(
        f for f in directory.iterdir()
        if f.suffix.lower() in IMAGE_EXTENSIONS
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. Corrupted image detection
# ═══════════════════════════════════════════════════════════════════════════

def find_corrupted(images: list[Path]) -> list[Path]:
    """
    Try to read each image with OpenCV.
    Returns list of files that could not be decoded.

    This is intentionally strict: an image that loads as None OR has
    0-dimension (e.g. truncated JPEG) is flagged.
    """
    bad: list[Path] = []
    for p in images:
        img = cv2.imread(str(p))
        if img is None or img.size == 0:
            bad.append(p)
    return bad


# ═══════════════════════════════════════════════════════════════════════════
# 3. Resolution statistics
# ═══════════════════════════════════════════════════════════════════════════

def resolution_stats(images: list[Path], max_sample: int = 500) -> dict:
    """
    Sample up to max_sample images to gather resolution statistics.

    Returns
    -------
    dict with keys:
        widths, heights, resolutions (Counter), mode_resolution,
        all_same (bool), sample_size
    """
    sample = random.sample(images, min(max_sample, len(images)))
    widths, heights = [], []
    res_counter: Counter = Counter()

    for p in sample:
        img = cv2.imread(str(p))
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)
            res_counter[f"{w}x{h}"] += 1

    if not widths:
        return {}

    mode_res, mode_count = res_counter.most_common(1)[0]
    return {
        "sample_size":      len(widths),
        "width_min":        min(widths),
        "width_max":        max(widths),
        "width_mode":       max(set(widths), key=widths.count),
        "height_min":       min(heights),
        "height_max":       max(heights),
        "height_mode":      max(set(heights), key=heights.count),
        "mode_resolution":  mode_res,
        "mode_count":       mode_count,
        "unique_resolutions": len(res_counter),
        "all_same":         len(res_counter) == 1,
        "resolution_counter": dict(res_counter.most_common(10)),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4. Frames per video (from metadata CSV or filename parsing)
# ═══════════════════════════════════════════════════════════════════════════

def frames_per_video(images: list[Path]) -> dict[str, int]:
    """
    Count extracted frames per source video.

    Strategy (in order):
    1. Read metadata.csv if it exists (accurate).
    2. Parse the filename stem: everything up to '_frame' is the video name.
       e.g.  drive01_session2_frame000150.jpg  ->  drive01_session2
    """
    # Strategy 1: metadata CSV
    if METADATA_CSV.exists():
        try:
            df = pd.read_csv(METADATA_CSV)
            if "source_video" in df.columns and "filename" in df.columns:
                # Only count files that are actually on disk
                on_disk = {p.name for p in images}
                df = df[df["filename"].isin(on_disk)]
                counts = df.groupby("source_video")["filename"].count()
                return counts.to_dict()
        except Exception:
            pass  # Fall through to strategy 2

    # Strategy 2: filename parsing
    counts: Counter = Counter()
    for p in images:
        stem = p.stem  # e.g. drive01_frame000150
        if "_frame" in stem:
            video_name = stem.rsplit("_frame", 1)[0]
        else:
            video_name = "unknown"
        counts[video_name] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════════
# 5. Report printers
# ═══════════════════════════════════════════════════════════════════════════

def print_overview(images: list[Path], videos: dict[str, int]) -> None:
    """Print the top-level overview panel."""
    total = len(images)
    n_videos = len(videos)

    if total == 0:
        console.print(
            Panel(
                "[bold yellow]No frames found in:[/bold yellow]\n"
                f"  [cyan]{PATHS.raw_frames}[/cyan]\n\n"
                "To collect frames:\n"
                "  [bold]1.[/bold] Copy a road video to:\n"
                f"       [cyan]{PATHS.raw_videos}[/cyan]\n"
                "  [bold]2.[/bold] Run:\n"
                "       [cyan]python src/data_collection/"
                "video_to_frames.py[/cyan]",
                title="[bold red]Dataset Empty",
                border_style="red",
            )
        )
        return

    console.print(
        Panel(
            f"[bold]Total frames:[/bold]   [green]{total}[/green]\n"
            f"[bold]Source videos:[/bold]  {n_videos}\n"
            f"[bold]Location:[/bold]       {PATHS.raw_frames}\n"
            f"[bold]Metadata CSV:[/bold]   "
            + ("[green]present[/green]" if METADATA_CSV.exists()
               else "[yellow]missing (run extractor to generate)[/yellow]"),
            title="[bold cyan]Dataset Overview",
            border_style="cyan",
        )
    )


def print_frames_per_video(videos: dict[str, int]) -> None:
    if not videos:
        return
    table = Table(title="[bold]Frames per Source Video", show_lines=True)
    table.add_column("Source Video / Session", style="cyan")
    table.add_column("Frames", justify="right", style="green")
    table.add_column("Share", justify="right")
    total = sum(videos.values())
    for vid, count in sorted(videos.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / total if total else 0
        table.add_row(vid, str(count), f"{pct:.1f}%")
    table.add_row("[bold]TOTAL", f"[bold]{total}", "100%")
    console.print(table)


def print_resolution_stats(stats: dict) -> None:
    if not stats:
        console.print("[yellow]No valid images sampled for resolution check.[/yellow]")
        return

    table = Table(
        title=f"[bold]Resolution Statistics (sample={stats['sample_size']})",
        show_lines=True,
    )
    table.add_column("Metric", style="cyan")
    table.add_column("Width", justify="right")
    table.add_column("Height", justify="right")
    table.add_row("Minimum",    str(stats["width_min"]),  str(stats["height_min"]))
    table.add_row("Maximum",    str(stats["width_max"]),  str(stats["height_max"]))
    table.add_row("Mode",       str(stats["width_mode"]), str(stats["height_mode"]))
    console.print(table)

    if stats["all_same"]:
        console.print(
            f"[green]All sampled frames have the same resolution: "
            f"{stats['mode_resolution']}[/green]"
        )
    else:
        console.print(
            f"[yellow]Mixed resolutions detected "
            f"({stats['unique_resolutions']} unique):[/yellow]"
        )
        res_table = Table(show_lines=True, title="Resolution Distribution (top 10)")
        res_table.add_column("Resolution", style="cyan")
        res_table.add_column("Count", justify="right")
        for res, cnt in stats["resolution_counter"].items():
            res_table.add_row(res, str(cnt))
        console.print(res_table)
        console.print(
            "[yellow]Run preprocessing to normalize all frames to a single "
            "resolution before training.[/yellow]"
        )


def print_corruption_report(bad: list[Path], total: int) -> None:
    if not bad:
        console.print(
            f"[green]Corruption check: all {total} images are readable.[/green]"
        )
        return

    console.print(
        f"[bold red]Corruption check: {len(bad)}/{total} images "
        f"are UNREADABLE:[/bold red]"
    )
    for p in bad[:20]:   # Show first 20 max
        console.print(f"  [red]{p.name}[/red]")
    if len(bad) > 20:
        console.print(f"  ... and {len(bad) - 20} more.")
    console.print(
        "[yellow]Delete corrupted files before annotation/training.[/yellow]"
    )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Random frame viewer
# ═══════════════════════════════════════════════════════════════════════════

def show_random_frames(images: list[Path], n: int = 9) -> None:
    """Display N random frames in a grid window. Press any key to close."""
    if not images:
        console.print("[yellow]No frames to display.[/yellow]")
        return

    sample = random.sample(images, min(n, len(images)))
    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols
    thumb_w, thumb_h = 320, 180

    grid_rows = []
    for row_i in range(rows):
        row_imgs = []
        for col_i in range(cols):
            idx = row_i * cols + col_i
            if idx >= len(sample):
                row_imgs.append(
                    np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                )
                continue
            img = cv2.imread(str(sample[idx]))
            if img is None:
                tile = np.zeros((thumb_h, thumb_w, 3), dtype=np.uint8)
                cv2.putText(
                    tile, "UNREADABLE",
                    (10, thumb_h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA,
                )
            else:
                tile = cv2.resize(img, (thumb_w, thumb_h))
                # Overlay filename (truncated)
                label = sample[idx].name[:36]
                cv2.putText(
                    tile, label,
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.35, (255, 255, 80), 1, cv2.LINE_AA,
                )
            row_imgs.append(tile)
        grid_rows.append(np.hstack(row_imgs))

    grid = np.vstack(grid_rows)
    window_title = f"LKA raw_frames — {len(sample)} random frames"
    cv2.imshow(window_title, grid)
    console.print(
        f"\n[cyan]Showing {len(sample)} random frames in a window. "
        "Press any key to close.[/cyan]"
    )
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ═══════════════════════════════════════════════════════════════════════════
# 7. Main
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="inspect_dataset.py",
        description=(
            "Inspect the LKA raw frames dataset.\n"
            "Reports frame counts, resolution stats, "
            "corruption, and source video breakdown."
        ),
    )
    parser.add_argument(
        "--show", type=int, default=9, metavar="N",
        help="Display N random frames in a window. Default: 9  (0 = disable)",
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Print statistics only; skip the image viewer.",
    )
    parser.add_argument(
        "--no-corrupt-check", action="store_true",
        help=(
            "Skip the corruption check (faster for large datasets). "
            "Not recommended before training."
        ),
    )
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]LKA Dataset Inspector[/bold cyan]\n"
            f"Inspecting: [cyan]{PATHS.raw_frames}[/cyan]",
            border_style="cyan",
        )
    )

    # ── Discover frames ────────────────────────────────────────────────────
    images = find_images(PATHS.raw_frames)

    # ── Frames per video ───────────────────────────────────────────────────
    videos = frames_per_video(images)

    # ── Overview ───────────────────────────────────────────────────────────
    print_overview(images, videos)

    if not images:
        return

    # ── Frames per video table ─────────────────────────────────────────────
    print_frames_per_video(videos)

    # ── Resolution statistics ──────────────────────────────────────────────
    console.print("\n[bold]Checking resolutions...[/bold]")
    res_stats = resolution_stats(images)
    print_resolution_stats(res_stats)

    # ── Corruption check ───────────────────────────────────────────────────
    if not args.no_corrupt_check:
        console.print(f"\n[bold]Checking {len(images)} images for corruption...[/bold]")
        bad = find_corrupted(images)
        print_corruption_report(bad, len(images))
    else:
        console.print("\n[yellow]Corruption check skipped (--no-corrupt-check).[/yellow]")

    # ── Frame viewer ───────────────────────────────────────────────────────
    if not args.stats_only and args.show > 0:
        show_random_frames(images, n=args.show)
    elif args.stats_only:
        console.print(
            "\n[dim]Frame viewer skipped (--stats-only). "
            "Re-run without that flag to view frames.[/dim]"
        )


if __name__ == "__main__":
    main()
