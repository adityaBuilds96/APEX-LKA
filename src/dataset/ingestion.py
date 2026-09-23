"""
src/dataset/ingestion.py
========================
Automated dataset ingestion pipeline for user-provided road datasets (ZIP or directory).

Responsibilities:
  - Safe ZIP decompression (zip-slip protection)
  - Automatic structure detection (images/, masks/, train/val/test splits, or flat folder)
  - Comprehensive quality audits via DatasetValidator:
      * Corrupted / unreadable image detection
      * Resolution distribution
      * Exact and near-duplicate detection
      * Sequential frame leakage grouping
      * Mask dimension matching and class validation
      * Class pixel distribution analysis
  - Explicit mode determination:
      * MODE A: Labeled Dataset (valid image-mask pairs found)
      * MODE B: Unlabeled Dataset (images only, annotation required)
  - Automatic DatasetReport generation (JSON + Markdown)
"""

import os
import shutil
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np

from src.dataset.report import DatasetReport
from src.dataset.validator import DatasetValidator
from src.dataset.annotation_prep import AnnotationWorkspace


class DatasetIngestor:
    """
    Ingests and analyzes road datasets from ZIP files or directories.
    """

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        min_width: int = 160,
        min_height: int = 90,
        dhash_threshold: int = 2,
    ):
        self.base_dir = Path(base_dir) if base_dir else Path.cwd()
        self.validator = DatasetValidator(
            min_width=min_width,
            min_height=min_height,
            dhash_threshold=dhash_threshold,
        )

    def ingest(
        self,
        source: Union[str, Path],
        target_dir: Optional[Path] = None,
        copy_to_annotated: bool = True,
    ) -> DatasetReport:
        """
        Ingest a dataset source (ZIP file or directory) and produce a DatasetReport.
        """
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Dataset source path does not exist: {source_path}")

        # ── 1. Unpack if ZIP ───────────────────────────────────────────────
        temp_extracted: Optional[Path] = None
        source_type = "directory"

        if source_path.is_file() and source_path.suffix.lower() == ".zip":
            source_type = "zip"
            staging_root = self.base_dir / "data" / "staging"
            staging_root.mkdir(parents=True, exist_ok=True)
            temp_extracted = staging_root / f"ingest_{source_path.stem}_{int(datetime.now().timestamp())}"
            temp_extracted.mkdir(parents=True, exist_ok=True)
            self._safe_extract_zip(source_path, temp_extracted)
            working_dir = temp_extracted
        elif source_path.is_dir():
            working_dir = source_path
        else:
            raise ValueError(f"Source must be a .zip file or a directory, got: {source_path}")

        # ── 2. Discover file layout ────────────────────────────────────────
        discovery = self._discover_files(working_dir)
        image_files = discovery["images"]
        mask_files = discovery["masks"]

        # ── 3. Validate Images ─────────────────────────────────────────────
        valid_images: list[Path] = []
        corrupted_files: list[str] = []
        resolutions: Counter = Counter()
        widths, heights = [], []
        seq_groups: Counter = Counter()

        # Duplicate checking structures
        sha_hashes: dict[str, Path] = {}
        exact_dup_pairs: list[list[str]] = []
        dhashes: list[tuple[int, Path]] = []
        near_dup_pairs: list[list[str]] = []

        for img_p in image_files:
            is_valid, shape, err = self.validator.validate_image(img_p)
            if not is_valid or shape is None:
                corrupted_files.append(f"{img_p.name} ({err})")
                continue

            h, w, _ = shape
            valid_images.append(img_p)
            resolutions[f"{w}x{h}"] += 1
            widths.append(w)
            heights.append(h)

            # Sequence group
            grp = self.validator.extract_sequence_group(img_p)
            seq_groups[grp] += 1

            # Exact duplicate check via SHA-256
            sha = self.validator.compute_sha256(img_p)
            if sha in sha_hashes:
                exact_dup_pairs.append([sha_hashes[sha].name, img_p.name])
            else:
                sha_hashes[sha] = img_p

            # Near-duplicate check via dHash (sample if dataset large)
            if len(valid_images) <= 500:
                img_data = cv2.imread(str(img_p))
                if img_data is not None:
                    dh = self.validator.compute_dhash(img_data)
                    for existing_dh, existing_p in dhashes:
                        dist = self.validator.hamming_distance(dh, existing_dh)
                        if dist <= self.validator.dhash_threshold:
                            near_dup_pairs.append([existing_p.name, img_p.name])
                            break
                    dhashes.append((dh, img_p))

        # Resolution summary
        min_res = f"{min(widths)}x{min(heights)}" if widths else None
        max_res = f"{max(widths)}x{max(heights)}" if widths else None
        mode_res = resolutions.most_common(1)[0][0] if resolutions else None

        # ── 4. Validate Masks (if present) ─────────────────────────────────
        img_stems = {p.stem: p for p in valid_images}
        matched_pairs: list[tuple[Path, Path]] = []
        dim_mismatches: list[str] = []
        empty_mask_count = 0
        invalid_class_mask_count = 0
        class_pixel_totals: Counter = Counter()

        for mask_p in mask_files:
            stem = mask_p.stem
            # Clean common suffixes like "_mask"
            base_stem = stem.replace("_mask", "").replace("-mask", "")
            matching_img = img_stems.get(stem) or img_stems.get(base_stem)

            if matching_img:
                img_h, img_w = cv2.imread(str(matching_img)).shape[:2]
                is_valid, counts, is_empty, err = self.validator.validate_mask(
                    mask_p, expected_shape=(img_h, img_w)
                )

                if not is_valid:
                    if "does not match image" in (err or ""):
                        dim_mismatches.append(f"{mask_p.name} ({err})")
                    if "invalid class IDs" in (err or ""):
                        invalid_class_mask_count += 1
                else:
                    matched_pairs.append((matching_img, mask_p))
                    if is_empty:
                        empty_mask_count += 1
                    for cid, count in counts.items():
                        class_pixel_totals[cid] += count

        # Mode determination
        masks_found = len(mask_files) > 0
        has_sufficient_pairs = len(matched_pairs) > 0 and (len(matched_pairs) >= len(valid_images) * 0.1)

        if masks_found and has_sufficient_pairs:
            mode = "MODE_A_LABELED"
            can_train = True
            status_msg = f"Mode A (Labeled): {len(matched_pairs)} matched image-mask pairs ready."
        else:
            mode = "MODE_B_UNLABELED"
            can_train = False
            if masks_found and not has_sufficient_pairs:
                status_msg = "Mode B (Unlabeled): Masks were found but could not be matched with images."
            else:
                status_msg = "Mode B (Unlabeled): No segmentation masks found. Annotation required before training."

        # Compute class distribution fractions
        total_pixels = sum(class_pixel_totals.values())
        detected_cids = sorted(list(class_pixel_totals.keys()))
        class_fractions = {cid: (class_pixel_totals[cid] / total_pixels) for cid in detected_cids} if total_pixels > 0 else {}

        # Class names map (supports up to 4 classes dynamically)
        default_names = {0: "background", 1: "road", 2: "left_lane", 3: "right_lane"}
        class_names = {cid: default_names.get(cid, f"class_{cid}") for cid in detected_cids}

        # Imbalance check: lane markings usually < 5% of pixels
        extreme_imbalance = False
        if total_pixels > 0:
            for cid in detected_cids:
                if cid > 0 and class_fractions.get(cid, 0.0) < 0.005:
                    extreme_imbalance = True
                    break

        # ── 5. Optionally Copy / Populate Target Directories ───────────────
        target_destination = target_dir or (self.base_dir / "data" / "annotated")
        if copy_to_annotated and len(valid_images) > 0:
            target_images_dir = target_destination / "images"
            target_masks_dir = target_destination / "masks"
            target_images_dir.mkdir(parents=True, exist_ok=True)
            target_masks_dir.mkdir(parents=True, exist_ok=True)

            if mode == "MODE_A_LABELED":
                for img_p, mask_p in matched_pairs:
                    shutil.copy2(img_p, target_images_dir / img_p.name)
                    # Normalize mask name to match image stem
                    dest_mask_name = f"{img_p.stem}.png"
                    shutil.copy2(mask_p, target_masks_dir / dest_mask_name)
            else:
                # Mode B: Copy images only
                for img_p in valid_images:
                    shutil.copy2(img_p, target_images_dir / img_p.name)
                # Prepare annotation workspace
                ws = AnnotationWorkspace.create(self.base_dir)
                ws.update_manifest()

        # Clean up staging zip directory if created
        if temp_extracted and temp_extracted.exists():
            shutil.rmtree(temp_extracted, ignore_errors=True)

        # ── 6. Assemble Report ─────────────────────────────────────────────
        report = DatasetReport(
            source_name=source_path.name,
            source_type=source_type,
            target_dir=str(target_destination),
            total_images=len(image_files),
            valid_images=len(valid_images),
            corrupted_images=len(corrupted_files),
            corrupted_files=corrupted_files[:20],
            resolutions=dict(resolutions.most_common(10)),
            min_resolution=min_res,
            max_resolution=max_res,
            mode_resolution=mode_res,
            exact_duplicates=len(exact_dup_pairs),
            exact_duplicate_pairs=exact_dup_pairs[:10],
            near_duplicates=len(near_dup_pairs),
            near_duplicate_pairs=near_dup_pairs[:10],
            unique_images=len(valid_images) - len(exact_dup_pairs),
            sequence_groups=dict(seq_groups),
            labels_found=masks_found,
            masks_found=masks_found,
            total_masks=len(mask_files),
            matched_pairs=len(matched_pairs),
            unmatched_images=len(valid_images) - len(matched_pairs),
            unmatched_masks=len(mask_files) - len(matched_pairs),
            dimension_mismatches=len(dim_mismatches),
            dimension_mismatch_files=dim_mismatches[:10],
            empty_masks=empty_mask_count,
            invalid_class_masks=invalid_class_mask_count,
            detected_classes=detected_cids,
            class_names=class_names,
            class_pixel_counts=dict(class_pixel_totals),
            class_pixel_fractions=class_fractions,
            extreme_class_imbalance=extreme_imbalance,
            mode=mode,
            can_train_supervised=can_train,
            status_message=status_msg,
        )

        # Save report
        out_metrics = self.base_dir / "results" / "metrics"
        out_metrics.mkdir(parents=True, exist_ok=True)
        report.save_json(out_metrics / "dataset_report.json")

        md_report_path = out_metrics / "dataset_report.md"
        with open(md_report_path, "w", encoding="utf-8") as f:
            f.write(report.to_markdown())

        return report

    @staticmethod
    def _safe_extract_zip(zip_path: Path, extract_to: Path) -> None:
        """
        Extract ZIP archive while checking for directory traversal (Zip Slip).
        """
        extract_to_resolved = extract_to.resolve()
        with zipfile.ZipFile(zip_path, "r") as zf:
            for member in zf.infolist():
                # Normalize path and check for traversal
                member_path = Path(member.filename)
                target_path = (extract_to / member_path).resolve()
                if not str(target_path).startswith(str(extract_to_resolved)):
                    raise RuntimeError(f"Zip-slip security error: member {member.filename} resolves outside destination")

            zf.extractall(extract_to)

    def _discover_files(self, root_dir: Path) -> dict[str, list[Path]]:
        """
        Recursively find images and masks within directory.
        """
        img_exts = self.validator.SUPPORTED_IMG_EXTS
        mask_exts = self.validator.SUPPORTED_MASK_EXTS

        images: list[Path] = []
        masks: list[Path] = []

        for p in root_dir.rglob("*"):
            if not p.is_file():
                continue

            parent_names = [part.lower() for part in p.parts]
            is_in_mask_folder = any(m in parent_names for m in ["masks", "mask", "labels", "annotations"])
            has_mask_stem = "_mask" in p.stem.lower() or "-mask" in p.stem.lower()

            if (is_in_mask_folder or has_mask_stem) and p.suffix.lower() in mask_exts:
                masks.append(p)
            elif p.suffix.lower() in img_exts and not has_mask_stem and not is_in_mask_folder:
                images.append(p)

        return {"images": sorted(images), "masks": sorted(masks)}
