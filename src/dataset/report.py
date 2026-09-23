"""
src/dataset/report.py
=====================
Structured report container for dataset ingestion, validation, and quality control.

Guarantees:
  - Every metric is computed from actual data (no fake numbers).
  - Explicit distinction between MODE A (LABELED) and MODE B (UNLABELED).
  - Human-readable markdown summary and JSON export.
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class DatasetReport:
    """
    Comprehensive report on ingested dataset status, quality, and class distribution.
    """
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    source_name: str = "unknown"
    source_type: str = "directory"  # "zip" or "directory"
    target_dir: str = ""

    # Image statistics
    total_images: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    corrupted_files: list[str] = field(default_factory=list)

    # Resolution statistics
    resolutions: dict[str, int] = field(default_factory=dict)
    min_resolution: Optional[str] = None
    max_resolution: Optional[str] = None
    mode_resolution: Optional[str] = None

    # Redundancy / Leakage
    exact_duplicates: int = 0
    exact_duplicate_pairs: list[list[str]] = field(default_factory=list)
    near_duplicates: int = 0
    near_duplicate_pairs: list[list[str]] = field(default_factory=list)
    unique_images: int = 0
    sequence_groups: dict[str, int] = field(default_factory=dict)

    # Label / Mask statistics
    labels_found: bool = False
    masks_found: bool = False
    total_masks: int = 0
    matched_pairs: int = 0
    unmatched_images: int = 0
    unmatched_masks: int = 0
    dimension_mismatches: int = 0
    dimension_mismatch_files: list[str] = field(default_factory=list)
    empty_masks: int = 0
    invalid_class_masks: int = 0

    # Class distribution
    detected_classes: list[int] = field(default_factory=list)
    class_names: dict[int, str] = field(default_factory=dict)
    class_pixel_counts: dict[int, int] = field(default_factory=dict)
    class_pixel_fractions: dict[int, float] = field(default_factory=dict)
    extreme_class_imbalance: bool = False

    # Workflow mode determination
    mode: str = "MODE_B_UNLABELED"  # "MODE_A_LABELED" or "MODE_B_UNLABELED"
    can_train_supervised: bool = False
    status_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary."""
        return asdict(self)

    def save_json(self, output_path: Path) -> Path:
        """Serialize report to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)
        return output_path

    def to_markdown(self) -> str:
        """Format report into human-readable Markdown summary."""
        lines = [
            "# APEX-RLP Dataset Ingestion & Quality Control Report",
            f"**Generated:** {self.timestamp}  ",
            f"**Source:** `{self.source_name}` ({self.source_type})  ",
            f"**Target Directory:** `{self.target_dir}`  ",
            f"**Workflow Mode:** **{self.mode}**  ",
            f"**Supervised Training Ready:** {'✅ YES' if self.can_train_supervised else '❌ NO (Annotation Required)'}  ",
            "",
            "## 1. Image Overview",
            f"- **Total Images Discovered:** {self.total_images}",
            f"- **Valid & Readable:** {self.valid_images}",
            f"- **Corrupted / Unreadable:** {self.corrupted_images}",
            f"- **Exact Duplicates:** {self.exact_duplicates}",
            f"- **Near Duplicates:** {self.near_duplicates}",
            f"- **Unique Effective Images:** {self.unique_images}",
            f"- **Resolution Range:** {self.min_resolution or 'N/A'} to {self.max_resolution or 'N/A'} (Mode: {self.mode_resolution or 'N/A'})",
            f"- **Sequence Groups (Video/Session):** {len(self.sequence_groups)} distinct group(s)",
            "",
            "## 2. Mask & Label Status",
            f"- **Labels Found:** {'YES' if self.labels_found else 'NO'}",
            f"- **Masks Found:** {'YES' if self.masks_found else 'NO'}",
            f"- **Total Masks:** {self.total_masks}",
            f"- **Matched (Image + Mask) Pairs:** {self.matched_pairs}",
            f"- **Unmatched Images (Missing Mask):** {self.unmatched_images}",
            f"- **Unmatched Masks (Missing Image):** {self.unmatched_masks}",
            f"- **Dimension Mismatches:** {self.dimension_mismatches}",
            f"- **Empty Masks (All Background):** {self.empty_masks}",
            f"- **Invalid Class ID Masks:** {self.invalid_class_masks}",
            "",
        ]

        if self.masks_found and self.detected_classes:
            lines.extend([
                "## 3. Pixel Class Distribution",
                "| Class ID | Class Name | Pixel Count | Pixel Share (%) |",
                "|:---|:---|---:|---:|",
            ])
            for cid in sorted(self.detected_classes):
                cname = self.class_names.get(cid, f"class_{cid}")
                count = self.class_pixel_counts.get(cid, 0)
                frac = self.class_pixel_fractions.get(cid, 0.0) * 100.0
                lines.append(f"| {cid} | {cname} | {count:,} | {frac:.2f}% |")

            if self.extreme_class_imbalance:
                lines.append("\n> [!WARNING]\n> Extreme class imbalance detected. Weighted loss (Weighted Cross-Entropy + Dice) recommended.")
            lines.append("")

        lines.extend([
            "## 4. Operational Status & Next Steps",
            f"**Message:** {self.status_message}",
            "",
        ])

        if self.mode == "MODE_B_UNLABELED":
            lines.extend([
                "> [!IMPORTANT]",
                "> **Supervised segmentation training requires ground truth target masks.**",
                "> The uploaded dataset contains road images without valid segmentation masks.",
                "> To proceed:",
                "> 1. Open the annotation workspace at `annotations/`.",
                "> 2. Annotate or review candidate masks (Classes: 0=background, 1=road, 2=left_lane, 3=right_lane).",
                "> 3. Save approved masks to `data/annotated/masks/`.",
                "> 4. Re-run validation before launching supervised model training.",
            ])
        else:
            lines.extend([
                "> [!NOTE]",
                "> Dataset is validated and ready for sequence-aware train/val/test splitting and model training.",
            ])

        return "\n".join(lines)
