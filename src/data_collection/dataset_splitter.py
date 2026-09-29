"""
dataset_splitter.py
===================
Split annotated image+mask pairs into train / val / test sets.

Ensures:
  - Stratified split (random but reproducible)
  - Only pairs with BOTH image AND mask are included
  - Configurable ratios from project_config.yaml
  - Copies (not moves) files so originals remain in annotated/

Usage
-----
  python src/data_collection/dataset_splitter.py
  python src/data_collection/dataset_splitter.py --dry-run
"""

import argparse
import random
import shutil
import sys
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.panel import Panel

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg  # noqa: E402

console = Console()
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_pairs(image_dir: Path, mask_dir: Path) -> list[tuple[Path, Path]]:
    """Return (image_path, mask_path) pairs that exist in both directories."""
    if not image_dir.exists() or not mask_dir.exists():
        return []

    img_stems = {f.stem: f for f in image_dir.iterdir()
                 if f.suffix.lower() in IMAGE_EXTENSIONS}
    mask_stems = {f.stem: f for f in mask_dir.iterdir()
                  if f.suffix.lower() in IMAGE_EXTENSIONS}

    paired = []
    for stem in sorted(img_stems.keys()):
        if stem in mask_stems:
            paired.append((img_stems[stem], mask_stems[stem]))

    return paired


def copy_pairs(pairs: list[tuple[Path, Path]], img_out: Path, mask_out: Path,
               dry_run: bool = False) -> int:
    """Copy pairs to destination directories."""
    img_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)

    for img_path, mask_path in pairs:
        if not dry_run:
            shutil.copy2(img_path, img_out / img_path.name)
            shutil.copy2(mask_path, mask_out / mask_path.name)

    return len(pairs)


class DatasetSplitter:
    """
    Object-oriented dataset splitter for programmatic and pipeline execution.
    """

    def __init__(
        self,
        source_dir: Optional[Path] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        seed: int = 42,
    ):
        self.source_dir = Path(source_dir) if source_dir else PATHS.annotated
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.seed = seed

    def split(self, dry_run: bool = False) -> dict[str, list[tuple[Path, Path]]]:
        """
        Split annotated pairs into train, val, and test partitions.
        """
        img_dir = self.source_dir / "images"
        mask_dir = self.source_dir / "masks"
        pairs = find_pairs(img_dir, mask_dir)

        if not pairs:
            return {"train": [], "val": [], "test": []}

        rng = random.Random(self.seed)
        shuffled = list(pairs)
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)

        train_pairs = shuffled[:n_train]
        val_pairs = shuffled[n_train:n_train + n_val]
        test_pairs = shuffled[n_train + n_val:]

        if not dry_run:
            for split, split_pairs in [("train", train_pairs), ("val", val_pairs), ("test", test_pairs)]:
                img_out = getattr(PATHS, split) / "images"
                mask_out = getattr(PATHS, split) / "masks"
                copy_pairs(split_pairs, img_out, mask_out)

        return {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }


def main():
    parser = argparse.ArgumentParser(description="Split annotated dataset into train/val/test")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would happen without copying files")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (default: from config)")
    args = parser.parse_args()

    ds_cfg = cfg["dataset"]
    seed = args.seed if args.seed is not None else ds_cfg["random_seed"]
    train_ratio = ds_cfg["train_ratio"]
    val_ratio = ds_cfg["val_ratio"]
    test_ratio = ds_cfg["test_ratio"]

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "train + val + test ratios must sum to 1.0"

    # ── Find annotated pairs ───────────────────────────────────────────────
    pairs = find_pairs(PATHS.annotated / "images", PATHS.annotated / "masks")

    if not pairs:
        console.print(
            Panel(
                "[bold yellow]No annotated pairs found![/bold yellow]\n\n"
                "Images with corresponding masks must exist in:\n"
                f"  Images: [cyan]{PATHS.annotated / 'images'}[/cyan]\n"
                f"  Masks:  [cyan]{PATHS.annotated / 'masks'}[/cyan]\n\n"
                "Annotate data first, then re-run this script.",
                title="[bold red]Cannot Split",
                border_style="red",
            )
        )
        return

    console.print(f"[bold]Found {len(pairs)} annotated image-mask pairs.[/bold]")

    # ── Shuffle and split ──────────────────────────────────────────────────
    random.seed(seed)
    shuffled = list(pairs)
    random.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val  # remainder goes to test

    train_pairs = shuffled[:n_train]
    val_pairs = shuffled[n_train:n_train + n_val]
    test_pairs = shuffled[n_train + n_val:]

    console.print(
        f"Split (seed={seed}): "
        f"[green]train={len(train_pairs)}[/green] | "
        f"[yellow]val={len(val_pairs)}[/yellow] | "
        f"[cyan]test={len(test_pairs)}[/cyan]"
    )

    if args.dry_run:
        console.print("[yellow]DRY RUN — no files were copied.[/yellow]")
        return

    # ── Copy files ─────────────────────────────────────────────────────────
    for split, split_pairs in [
        ("train", train_pairs),
        ("val", val_pairs),
        ("test", test_pairs),
    ]:
        img_out = getattr(PATHS, split) / "images"
        mask_out = getattr(PATHS, split) / "masks"
        n_copied = copy_pairs(split_pairs, img_out, mask_out)
        console.print(f"[green]✓ {split}: {n_copied} pairs copied[/green]")

    console.print("\n[bold green]Dataset split complete! Ready for training.[/bold green]")
    console.print(
        "Next: [cyan]python src/training/train.py[/cyan]"
    )


if __name__ == "__main__":
    main()
