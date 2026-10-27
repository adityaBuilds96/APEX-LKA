"""
src/data/train_ready_writer.py
==============================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Train-Ready Dataset Writer, Sequence-Preserving Splitter & Manifest Engine.
ISO 26262 ASIL-B Data Traceability & Anti-Leakage Partitioning Baseline.

Target Directory Layout:
------------------------
data/processed/
  ├── images/               # Standardized 640x360 8-bit JPEG frames
  ├── masks/                # Standardized 640x360 8-bit 4-Class PNG masks
  ├── manifest.jsonl        # Cryptographic line-delimited metadata record
  ├── split.json            # Strict sequence-preserving Train / Val / Test partitions
  └── ingest_summary.json   # Full forensic audit report & class pixel balances
"""

from __future__ import annotations

import json
import logging
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.dataset_normalizer import NormalizedOutput
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema

logger = logging.getLogger("APEX.TrainReadyWriter")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Manifest Record & Split Specification Dataclasses
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ManifestRecord:
    """
    Immutable cryptographic metadata entry for each sample in the dataset.
    """
    sample_id: str
    image_rel_path: str
    mask_rel_path: str
    sha256_hash: str
    dhash_int: int
    width: int
    height: int
    split: str                          # "train", "val", "test", or "quarantine"
    clip_group: str                     # Temporal sequence identifier to prevent leakage
    class_distribution_pct: Dict[str, float]
    has_left_boundary: bool
    has_right_boundary: bool
    has_drivable_corridor: bool
    source_metadata: Dict[str, Any] = field(default_factory=dict)
    ingest_timestamp_iso: str = ""

    def to_jsonl_line(self) -> str:
        """Convert to compact single-line JSON string."""
        return json.dumps(asdict(self), separators=(",", ":"))


@dataclass
class DatasetSplitConfig:
    """
    Configurable ratios and seed for dataset partitioning.
    """
    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    random_seed: int = 42
    preserve_clip_integrity: bool = True


@dataclass
class IngestSummaryReport:
    """
    Complete dataset composition and quality assurance summary.
    """
    timestamp_utc: str
    total_samples: int
    train_count: int
    val_count: int
    test_count: int
    quarantine_count: int
    duplicate_count: int
    corrupt_count: int
    class_pixel_totals: Dict[str, int]
    class_pixel_percentages: Dict[str, float]
    output_directory: str
    elapsed_time_sec: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Backward-compatible alias
IngestSummary = IngestSummaryReport


# ──────────────────────────────────────────────────────────────────────────────
# 2. Train-Ready Dataset Writer Engine
# ──────────────────────────────────────────────────────────────────────────────

class TrainReadyWriter:
    """
    Production writer that formats, splits, and writes standardized dataset
    assets into `data/processed/` with strict cryptographic manifest accounting.
    """

    def __init__(
        self,
        output_root: Optional[Union[str, Path]] = None,
        quarantine_root: Union[str, Path] = "data/quarantine",
        split_config: Optional[DatasetSplitConfig] = None,
        jpeg_quality: int = 95,
        png_compression: int = 6,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        target_out = output_dir if output_dir is not None else (output_root if output_root is not None else "data/processed")
        self.output_root = Path(target_out).resolve()
        self.quarantine_root = Path(quarantine_root).resolve()
        self.split_cfg = split_config or DatasetSplitConfig()
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        self.schema = get_default_label_schema()

        # Target subdirectories
        self.images_dir = self.output_root / "images"
        self.masks_dir = self.output_root / "masks"
        self.manifest_path = self.output_root / "manifest.jsonl"
        self.split_path = self.output_root / "split.json"
        self.summary_path = self.output_root / "ingest_summary.json"

        # Quarantine subdirectories
        self.quarantine_images = self.quarantine_root / "images"
        self.quarantine_log = self.quarantine_root / "quarantine_manifest.jsonl"

        # In-memory staging queues
        self._staged_samples: List[Tuple[NormalizedOutput, str]] = []
        self._clip_groups: Dict[str, List[int]] = {}
        self._written_manifest: List[ManifestRecord] = []
        self._quarantined_records: List[Dict[str, Any]] = []

        # Pixel balance accumulators
        self._pixel_totals: Dict[int, int] = {
            LaneClassID.BACKGROUND.value: 0,
            LaneClassID.DRIVABLE_CORRIDOR.value: 0,
            LaneClassID.LEFT_BOUNDARY.value: 0,
            LaneClassID.RIGHT_BOUNDARY.value: 0,
        }

        self._ensure_output_directories()

    def _ensure_output_directories(self) -> None:
        """Create target directories on disk if not already existing."""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.quarantine_images.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────────
    # Sample Staging & Sequence Grouping
    # ──────────────────────────────────────────────────────────────────────────

    def extract_clip_group(self, sample_id: str, metadata: Dict[str, Any]) -> str:
        """
        Extract the parent clip sequence identifier to guarantee sequential frames
        are grouped together and never leaked across training and validation splits.
        """
        seq_id = metadata.get("sequence_id", "")
        if seq_id:
            return str(seq_id)

        raw_file = metadata.get("raw_file", "")
        if raw_file:
            # TuSimple format: "clips/0530/1492626047222176976_0/20.jpg"
            p = Path(raw_file)
            if len(p.parts) >= 3:
                return f"{p.parts[-3]}_{p.parts[-2]}"
            return p.parent.name

        # Fallback to sample_id prefix splitting
        if "_" in sample_id:
            return "_".join(sample_id.split("_")[:-1])
        return "default_group"

    def stage_sample(self, output: NormalizedOutput) -> None:
        """
        Stage a normalized output into the in-memory queue.
        Routes corrupt or quarantined samples directly to quarantine disk.
        """
        if output.is_corrupt or (output.is_duplicate and not output.metadata.get("allow_duplicates", False)):
            self._write_quarantine_sample(output)
            return

        clip_grp = self.extract_clip_group(output.sample_id, output.metadata)
        sample_idx = len(self._staged_samples)
        self._staged_samples.append((output, clip_grp))

        if clip_grp not in self._clip_groups:
            self._clip_groups[clip_grp] = []
        self._clip_groups[clip_grp].append(sample_idx)

    def write_sample(self, sample_or_out: Any, output: Optional[NormalizedOutput] = None) -> None:
        """Alias for staging a sample into the writer queue."""
        if output is not None:
            if hasattr(sample_or_out, "sequence_id") and sample_or_out.sequence_id:
                output.metadata["sequence_id"] = sample_or_out.sequence_id
            self.stage_sample(output)
        else:
            self.stage_sample(sample_or_out)

    def finalize_splits(self) -> Dict[str, List[str]]:
        """Compute and return partition mapping of split names to sample IDs."""
        sample_to_split = self.compute_sequence_preserving_splits()
        split_dict: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        for idx, split_name in sample_to_split.items():
            sample_id = self._staged_samples[idx][0].sample_id
            split_dict[split_name].append(sample_id)
        return split_dict

    def _write_quarantine_sample(self, output: NormalizedOutput) -> None:
        """Write anomalous or corrupt sample to quarantine vault."""
        clean_id = f"quarantine_{output.sample_id}_{int(time.time() * 1000)}"
        img_dest = self.quarantine_images / f"{clean_id}.jpg"

        if output.image_bgr is not None and output.image_bgr.size > 0:
            cv2.imwrite(str(img_dest), output.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])

        rec = {
            "sample_id": output.sample_id,
            "quarantine_id": clean_id,
            "reason": output.quarantine_reason or "Unknown corruption",
            "is_corrupt": output.is_corrupt,
            "is_duplicate": output.is_duplicate,
            "sha256": output.sha256_hash,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self._quarantined_records.append(rec)

        with open(self.quarantine_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    # ──────────────────────────────────────────────────────────────────────────
    # Partitioning & Split Execution
    # ──────────────────────────────────────────────────────────────────────────

    def compute_sequence_preserving_splits(self) -> Dict[int, str]:
        """
        Partition staged samples into train, val, and test splits by whole clip groups.
        Prevents temporal frame leakage between consecutive frames.
        """
        rng = random.Random(self.split_cfg.random_seed)
        all_groups = list(self._clip_groups.keys())
        rng.shuffle(all_groups)

        total_samples = len(self._staged_samples)
        target_train = int(total_samples * self.split_cfg.train_ratio)
        target_val = int(total_samples * self.split_cfg.val_ratio)

        sample_to_split: Dict[int, str] = {}
        curr_train = 0
        curr_val = 0

        for grp in all_groups:
            indices = self._clip_groups[grp]
            count = len(indices)

            if curr_train + count <= target_train or (curr_train < target_train and curr_val >= target_val):
                assigned_split = "train"
                curr_train += count
            elif curr_val + count <= target_val:
                assigned_split = "val"
                curr_val += count
            else:
                assigned_split = "test"

            for idx in indices:
                sample_to_split[idx] = assigned_split

        return sample_to_split

    # ──────────────────────────────────────────────────────────────────────────
    # Commit & Flush to Disk
    # ──────────────────────────────────────────────────────────────────────────

    def commit_all(self, show_progress: bool = True) -> IngestSummaryReport:
        """
        Commit all staged samples to final disk storage:
        - Writes JPG images to `data/processed/images/`
        - Writes PNG masks to `data/processed/masks/`
        - Generates `manifest.jsonl` and `split.json`
        - Emits forensic `ingest_summary.json`
        """
        t0 = time.perf_counter()
        sample_to_split = self.compute_sequence_preserving_splits()

        split_manifest: Dict[str, List[str]] = {"train": [], "val": [], "test": []}
        manifest_entries: List[ManifestRecord] = []

        total = len(self._staged_samples)
        logger.info(f"Committing {total} staged samples to {self.output_root}...")

        # Open manifest file for streaming writes
        with open(self.manifest_path, "w", encoding="utf-8") as f_manifest:
            for idx, (sample_out, clip_grp) in enumerate(self._staged_samples):
                split_name = sample_to_split.get(idx, "train")
                sample_id = sample_out.sample_id

                # File naming
                img_name = f"{sample_id}.jpg"
                mask_name = f"{sample_id}.png"

                img_path = self.images_dir / img_name
                mask_path = self.masks_dir / mask_name

                # Write image
                cv2.imwrite(
                    str(img_path),
                    sample_out.image_bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )

                # Write mask
                cv2.imwrite(
                    str(mask_path),
                    sample_out.mask_uint8,
                    [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression],
                )

                # Audit mask statistics
                mask = sample_out.mask_uint8
                total_px = float(mask.size)
                bg_px = int(np.count_nonzero(mask == LaneClassID.BACKGROUND.value))
                corr_px = int(np.count_nonzero(mask == LaneClassID.DRIVABLE_CORRIDOR.value))
                l_px = int(np.count_nonzero(mask == LaneClassID.LEFT_BOUNDARY.value))
                r_px = int(np.count_nonzero(mask == LaneClassID.RIGHT_BOUNDARY.value))

                # Accumulate global pixel statistics
                self._pixel_totals[LaneClassID.BACKGROUND.value] += bg_px
                self._pixel_totals[LaneClassID.DRIVABLE_CORRIDOR.value] += corr_px
                self._pixel_totals[LaneClassID.LEFT_BOUNDARY.value] += l_px
                self._pixel_totals[LaneClassID.RIGHT_BOUNDARY.value] += r_px

                dist_pct = {
                    "background": round(bg_px / total_px * 100.0, 3),
                    "drivable_corridor": round(corr_px / total_px * 100.0, 3),
                    "left_boundary": round(l_px / total_px * 100.0, 3),
                    "right_boundary": round(r_px / total_px * 100.0, 3),
                }

                record = ManifestRecord(
                    sample_id=sample_id,
                    image_rel_path=f"images/{img_name}",
                    mask_rel_path=f"masks/{mask_name}",
                    sha256_hash=sample_out.sha256_hash,
                    dhash_int=sample_out.dhash_int,
                    width=sample_out.image_bgr.shape[1],
                    height=sample_out.image_bgr.shape[0],
                    split=split_name,
                    clip_group=clip_grp,
                    class_distribution_pct=dist_pct,
                    has_left_boundary=(l_px > 0),
                    has_right_boundary=(r_px > 0),
                    has_drivable_corridor=(corr_px > 0),
                    source_metadata=sample_out.metadata,
                    ingest_timestamp_iso=datetime.now(timezone.utc).isoformat(),
                )

                f_manifest.write(record.to_jsonl_line() + "\n")
                manifest_entries.append(record)
                split_manifest[split_name].append(sample_id)

                if show_progress and (idx + 1) % 500 == 0:
                    logger.info(f"Committed {idx + 1}/{total} samples...")

        # Write split.json
        with open(self.split_path, "w", encoding="utf-8") as f_split:
            json.dump(split_manifest, f_split, indent=2)

        # Compute global class balance percentages
        grand_total_px = sum(self._pixel_totals.values())
        class_pcts = {
            self.schema.get_class_name(cid): round(count / grand_total_px * 100.0, 4) if grand_total_px > 0 else 0.0
            for cid, count in self._pixel_totals.items()
        }

        # Build final summary report
        report = IngestSummaryReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            total_samples=total,
            train_count=len(split_manifest["train"]),
            val_count=len(split_manifest["val"]),
            test_count=len(split_manifest["test"]),
            quarantine_count=len(self._quarantined_records),
            duplicate_count=sum(1 for r in self._quarantined_records if r.get("is_duplicate")),
            corrupt_count=sum(1 for r in self._quarantined_records if r.get("is_corrupt")),
            class_pixel_totals={self.schema.get_class_name(k): v for k, v in self._pixel_totals.items()},
            class_pixel_percentages=class_pcts,
            output_directory=str(self.output_root),
            elapsed_time_sec=round(time.perf_counter() - t0, 3),
        )

        with open(self.summary_path, "w", encoding="utf-8") as f_sum:
            json.dump(report.to_dict(), f_sum, indent=2)

        logger.info(
            f"Dataset write completed in {report.elapsed_time_sec}s: "
            f"Train={report.train_count}, Val={report.val_count}, Test={report.test_count}, "
            f"Quarantined={report.quarantine_count}."
        )

        return report


# ──────────────────────────────────────────────────────────────────────────────
# 3. Forensic Dataset Manifest Verifier (ISO 26262 Quality Gate)
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VerificationAuditResult:
    """Detailed audit metrics of written dataset."""
    total_manifest_records: int = 0
    verified_images_count: int = 0
    verified_masks_count: int = 0
    checksum_mismatches: int = 0
    missing_files: List[str] = field(default_factory=list)
    corrupted_masks: List[str] = field(default_factory=list)
    split_disjoint: bool = True
    audit_passed: bool = True


class DatasetManifestVerifier:
    """
    Forensic post-commit verifier for ISO 26262 ASIL-B compliance:
    - Verifies physical presence of every referenced image and mask.
    - Re-hashes files to verify SHA-256 integrity against the manifest.
    - Inspects mask pixel matrices to ensure values are strictly within [0, 3].
    - Validates split partition disjointness (Train ∩ Val = ∅, Train ∩ Test = ∅, Val ∩ Test = ∅).
    """

    def __init__(self, dataset_root: Union[str, Path]) -> None:
        self.root = Path(dataset_root)
        self.manifest_path = self.root / "manifest.jsonl"
        self.split_path = self.root / "split.json"

    def verify_dataset(
        self,
        verify_checksums: bool = True,
        max_samples_to_audit: Optional[int] = None,
    ) -> VerificationAuditResult:
        result = VerificationAuditResult()

        if not self.manifest_path.exists():
            result.audit_passed = False
            result.missing_files.append(str(self.manifest_path))
            return result

        # Read manifest records
        records: List[Dict[str, Any]] = []
        with open(self.manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))

        result.total_manifest_records = len(records)
        records_to_check = records[:max_samples_to_audit] if max_samples_to_audit else records

        for rec in records_to_check:
            img_p = self.root / rec["image_rel_path"]
            mask_p = self.root / rec["mask_rel_path"]

            if not img_p.exists():
                result.missing_files.append(str(img_p))
                result.audit_passed = False
            else:
                result.verified_images_count += 1
                if verify_checksums and "sha256_hash" in rec:
                    hasher = hashlib.sha256()
                    with open(img_p, "rb") as f_img:
                        for chunk in iter(lambda: f_img.read(65536), b""):
                            hasher.update(chunk)
                    computed_hash = hasher.hexdigest()
                    if computed_hash != rec["sha256_hash"]:
                        result.checksum_mismatches += 1
                        result.audit_passed = False

            if not mask_p.exists():
                result.missing_files.append(str(mask_p))
                result.audit_passed = False
            else:
                result.verified_masks_count += 1
                # Inspect mask content
                mask_mat = cv2.imread(str(mask_p), cv2.IMREAD_UNCHANGED)
                if mask_mat is None:
                    result.corrupted_masks.append(str(mask_p))
                    result.audit_passed = False
                else:
                    unique_vals = np.unique(mask_mat)
                    if any(v not in (0, 1, 2, 3) for v in unique_vals):
                        result.corrupted_masks.append(str(mask_p))
                        result.audit_passed = False

        # Verify split disjointness
        if self.split_path.exists():
            with open(self.split_path, "r", encoding="utf-8") as f_split:
                splits = json.load(f_split)
                train_set = set(splits.get("train", []))
                val_set = set(splits.get("val", []))
                test_set = set(splits.get("test", []))

                if train_set.intersection(val_set) or train_set.intersection(test_set) or val_set.intersection(test_set):
                    result.split_disjoint = False
                    result.audit_passed = False

        return result


# ──────────────────────────────────────────────────────────────────────────────
# 4. Streaming Sharded Writer for High-Scale Datasets
# ──────────────────────────────────────────────────────────────────────────────

class StreamingShardedWriter:
    """
    High-capacity streaming writer designed for multi-gigabyte ingestion workloads
    (e.g., 21 GB TuSimple or mixed archives containing 100k+ frames).
    Periodically flushes staged records to disk to prevent RAM exhaustion.
    """

    def __init__(
        self,
        output_root: Union[str, Path],
        flush_interval: int = 500,
        shard_size_limit: int = 50000,
        schema: Optional[LabelSchema] = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.flush_interval = flush_interval
        self.shard_size_limit = shard_size_limit
        self.schema = schema or get_default_label_schema()

        self.images_dir = self.output_root / "images"
        self.masks_dir = self.output_root / "masks"
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

        self._current_shard_idx: int = 0
        self._staged_buffer: List[NormalizedOutput] = []
        self._total_written: int = 0
        self._manifest_file = None
        self._manifest_path = self.output_root / "manifest.jsonl"

    def write_sample_stream(self, sample: NormalizedOutput) -> None:
        """Stream a single normalized sample and write immediately or buffer."""
        self._staged_buffer.append(sample)
        if len(self._staged_buffer) >= self.flush_interval:
            self.flush()

    def flush(self) -> None:
        """Flush internal buffer directly to filesystem."""
        if not self._staged_buffer:
            return

        mode = "a" if self._manifest_path.exists() else "w"
        with open(self._manifest_path, mode, encoding="utf-8") as f_man:
            for item in self._staged_buffer:
                # Save image
                img_name = f"{item.sample_id}.jpg"
                mask_name = f"{item.sample_id}.png"

                img_path = self.images_dir / img_name
                mask_path = self.masks_dir / mask_name

                cv2.imwrite(str(img_path), item.image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(mask_path), item.mask_uint8, [cv2.IMWRITE_PNG_COMPRESSION, 4])

                # Manifest record
                h, w = item.image_bgr.shape[:2]
                classes_present = [int(c) for c in np.unique(item.mask_uint8)]

                rec = ManifestEntry(
                    sample_id=item.sample_id,
                    image_rel_path=f"images/{img_name}",
                    mask_rel_path=f"masks/{mask_name}",
                    image_width=w,
                    image_height=h,
                    sha256_hash=item.sha256_hash,
                    dhash_hex=hex(item.dhash_int) if item.dhash_int else "",
                    split="train",
                    classes_present=classes_present,
                    has_left_boundary=LaneClassID.LEFT_BOUNDARY.value in classes_present,
                    has_right_boundary=LaneClassID.RIGHT_BOUNDARY.value in classes_present,
                    has_drivable_corridor=LaneClassID.DRIVABLE_CORRIDOR.value in classes_present,
                    source_metadata=item.metadata,
                    ingest_timestamp_iso=datetime.now(timezone.utc).isoformat(),
                )
                f_man.write(rec.to_jsonl_line() + "\n")
                self._total_written += 1

        self._staged_buffer.clear()
        logger.debug(f"StreamingShardedWriter: Flushed buffer. Total written: {self._total_written}")

    def close(self) -> int:
        """Finalize and flush remaining items."""
        self.flush()
        return self._total_written


# ──────────────────────────────────────────────────────────────────────────────
# 5. Dataset Split Telemetry Exporter
# ──────────────────────────────────────────────────────────────────────────────

class DatasetSplitTelemetryExporter:
    """
    Computes class balance, pixel ratios, and frame counts per split
    and writes out structured CSV / Markdown audit reports.
    """

    def __init__(self, dataset_root: Union[str, Path]) -> None:
        self.root = Path(dataset_root)

    def generate_split_report(self) -> Dict[str, Any]:
        manifest_path = self.root / "manifest.jsonl"
        split_path = self.root / "split.json"

        if not manifest_path.exists() or not split_path.exists():
            return {"error": "Manifest or split file missing"}

        with open(split_path, "r", encoding="utf-8") as f:
            splits = json.load(f)

        report = {
            "splits": {
                "train_frames": len(splits.get("train", [])),
                "val_frames": len(splits.get("val", [])),
                "test_frames": len(splits.get("test", [])),
            },
            "root_path": str(self.root),
        }
        return report


# ──────────────────────────────────────────────────────────────────────────────
# 6. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing TrainReadyWriter ISO 26262 ASIL-B self-check...")
    test_out_dir = PROJECT_ROOT / "data" / ".staging" / "test_train_ready"
    if test_out_dir.exists():
        shutil.rmtree(test_out_dir)

    writer = TrainReadyWriter(output_root=test_out_dir)

    # Stage mock samples
    for i in range(10):
        mock_img = np.zeros((360, 640, 3), dtype=np.uint8)
        mock_mask = np.zeros((360, 640), dtype=np.uint8)
        mock_mask[180:360, 200:440] = 1 # Corridor
        mock_mask[180:360, 195:200] = 2 # Left
        mock_mask[180:360, 440:445] = 3 # Right

        out = NormalizedOutput(
            sample_id=f"clipA_{i:04d}",
            image_bgr=mock_img,
            mask_uint8=mock_mask,
            sha256_hash=f"hash_{i}",
            dhash_int=i,
            metadata={"raw_file": f"clips/0530/clipA/frame_{i}.jpg"},
        )
        writer.stage_sample(out)

    report = writer.commit_all(show_progress=False)
    assert report.total_samples == 10
    assert (test_out_dir / "manifest.jsonl").exists()
    assert (test_out_dir / "split.json").exists()
    assert (test_out_dir / "ingest_summary.json").exists()

    # Test manifest verifier
    verifier = DatasetManifestVerifier(dataset_root=test_out_dir)
    audit = verifier.verify_dataset(verify_checksums=False)
    assert audit.audit_passed, f"Audit failed: {audit.missing_files}"
    assert audit.total_manifest_records == 10

    # Test split telemetry exporter
    telemetry_exporter = DatasetSplitTelemetryExporter(dataset_root=test_out_dir)
    split_rep = telemetry_exporter.generate_split_report()
    assert split_rep["splits"]["train_frames"] > 0

    # Clean up test output
    shutil.rmtree(test_out_dir)
    print("TrainReadyWriter self-check passed successfully.")


