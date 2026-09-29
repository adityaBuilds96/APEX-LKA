"""
src/dataset/pipeline_automation.py
==================================
Zero-touch intelligent data pipeline automation for APEX LKA.

Orchestrates the entire data lifecycle:
  1. Ingestion (Directory, ZIP archive, or raw video extraction)
  2. Validation (Corrupt image detection, dimension profiling)
  3. Deduplication (SHA-256 exact-duplicate + dHash near-duplicate filtering)
  4. Auto-Annotation (Classical CV pseudo-mask generation with QA scoring)
  5. QA & Quality Audit (Class distribution, spatial checks, report export)
  6. Dataset Splitting (Sequence-aware train/val/test paired partition)

Designed for headless CLI dispatch and asynchronous Streamlit UI execution
with real-time progress callbacks.
"""

import shutil
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np

from src.config import PATHS, cfg
from src.data_collection.dataset_splitter import DatasetSplitter
from src.data_collection.video_to_frames import FrameExtractor
from src.dataset.annotation_prep import AnnotationWorkspace
from src.dataset.auto_annotator import ClassicalAutoAnnotator
from src.dataset.quality_assurance import AnnotationQA
from src.dataset.report import DatasetReport
from src.dataset.validator import DatasetValidator


@dataclass
class PipelineExecutionResult:
    success: bool
    total_input_files: int = 0
    valid_images: int = 0
    corrupted_images: int = 0
    exact_duplicates: int = 0
    near_duplicates: int = 0
    auto_annotated_count: int = 0
    train_count: int = 0
    val_count: int = 0
    test_count: int = 0
    mean_quality_score: float = 0.0
    elapsed_sec: float = 0.0
    report_json_path: Optional[Path] = None
    report_md_path: Optional[Path] = None
    logs: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs.append(f"[{timestamp}] {message}")


class AutomatedDataPipeline:
    """
    Master pipeline orchestrating end-to-end dataset transformation.
    """

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    SUPPORTED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        validator: Optional[DatasetValidator] = None,
        auto_annotator: Optional[ClassicalAutoAnnotator] = None,
        qa_evaluator: Optional[AnnotationQA] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else PATHS.data.parent
        self.validator = validator or DatasetValidator()
        self.qa = qa_evaluator or AnnotationQA()
        self.auto_annotator = auto_annotator or ClassicalAutoAnnotator(qa_evaluator=self.qa)
        self.workspace = AnnotationWorkspace.create(self.base_dir)

    def run(
        self,
        source: Union[str, Path, list[Path]],
        auto_annotate: bool = True,
        auto_split: bool = True,
        deduplicate: bool = True,
        target_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, int, int, str], None]] = None,
    ) -> PipelineExecutionResult:
        """
        Execute full pipeline lifecycle.
        
        Args:
            source: Path to zip file, directory of images/videos, video file, or list of files.
            auto_annotate: If True, generate classical CV pseudo-masks.
            auto_split: If True, partition into train/val/test splits.
            deduplicate: If True, filter exact and near duplicates.
            target_dir: Destination images directory (default: data/annotated/images).
            progress_callback: Callback(stage_name, step, total_steps, message).
        """
        t0 = time.time()
        res = PipelineExecutionResult(success=True)

        target_img_dir = target_dir or (self.base_dir / "data" / "annotated" / "images")
        target_mask_dir = self.base_dir / "data" / "annotated" / "masks"
        candidate_dir = self.base_dir / "annotations" / "auto_generated_candidates"

        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)
        candidate_dir.mkdir(parents=True, exist_ok=True)

        def _notify(stage: str, step: int, total: int, msg: str):
            res.log(f"[{stage}] {msg}")
            if progress_callback:
                progress_callback(stage, step, total, msg)

        # -------------------------------------------------------------
        # STAGE 1: INGESTION & EXTRACTION
        # -------------------------------------------------------------
        _notify("INGESTION", 1, 6, "Ingesting source files...")
        staging_dir = self.base_dir / "data" / "staging_temp"
        staging_dir.mkdir(parents=True, exist_ok=True)

        extracted_images: list[Path] = []
        try:
            if isinstance(source, (str, Path)):
                src_path = Path(source)
                if src_path.is_file():
                    ext = src_path.suffix.lower()
                    if ext == ".zip":
                        # Safe zip extraction
                        _notify("INGESTION", 1, 6, f"Extracting ZIP archive: {src_path.name}")
                        with zipfile.ZipFile(src_path, "r") as zf:
                            for member in zf.infolist():
                                target_path = (staging_dir / member.filename).resolve()
                                if not str(target_path).startswith(str(staging_dir.resolve())):
                                    continue
                                if not member.is_dir():
                                    target_path.parent.mkdir(parents=True, exist_ok=True)
                                    with zf.open(member) as src_f, open(target_path, "wb") as dst_f:
                                        shutil.copyfileobj(src_f, dst_f)
                                    if target_path.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                                        extracted_images.append(target_path)
                    elif ext in self.SUPPORTED_VID_EXTS:
                        # Extract frames from video
                        _notify("INGESTION", 1, 6, f"Extracting frames from video: {src_path.name}")
                        extractor = FrameExtractor(video_path=src_path, output_dir=staging_dir, target_fps=3.0)
                        extractor.extract()
                        for f in staging_dir.iterdir():
                            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                                extracted_images.append(f)
                    elif ext in self.SUPPORTED_IMG_EXTS:
                        dst = staging_dir / src_path.name
                        shutil.copy2(src_path, dst)
                        extracted_images.append(dst)

                elif src_path.is_dir():
                    _notify("INGESTION", 1, 6, f"Scanning directory: {src_path}")
                    for f in src_path.rglob("*"):
                        if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                            dst = staging_dir / f.name
                            shutil.copy2(f, dst)
                            extracted_images.append(dst)
                        elif f.is_file() and f.suffix.lower() in self.SUPPORTED_VID_EXTS:
                            extractor = FrameExtractor(video_path=f, output_dir=staging_dir, target_fps=3.0)
                            extractor.extract()
                            for frame_f in staging_dir.iterdir():
                                if frame_f.is_file() and frame_f.suffix.lower() in self.SUPPORTED_IMG_EXTS and frame_f not in extracted_images:
                                    extracted_images.append(frame_f)

            elif isinstance(source, list):
                _notify("INGESTION", 1, 6, f"Processing {len(source)} uploaded files...")
                for item in source:
                    p = Path(item)
                    if p.is_file():
                        if p.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                            dst = staging_dir / p.name
                            shutil.copy2(p, dst)
                            extracted_images.append(dst)
                        elif p.suffix.lower() in self.SUPPORTED_VID_EXTS:
                            extractor = VideoFrameExtractor()
                            frames = extractor.extract(p, output_dir=staging_dir, target_fps=3.0)
                            extracted_images.extend(frames)

            res.total_input_files = len(extracted_images)
            _notify("INGESTION", 1, 6, f"Staged {res.total_input_files} image files.")

            # ---------------------------------------------------------
            # STAGE 2: VALIDATION
            # ---------------------------------------------------------
            _notify("VALIDATION", 2, 6, "Validating image integrity & dimensions...")
            valid_images: list[Path] = []
            for img_p in extracted_images:
                is_valid, shape, err = self.validator.validate_image(img_p)
                if is_valid:
                    valid_images.append(img_p)
                else:
                    res.corrupted_images += 1
                    res.log(f"Corrupt/Invalid {img_p.name}: {err}")

            res.valid_images = len(valid_images)
            _notify("VALIDATION", 2, 6, f"Validation complete: {res.valid_images} valid, {res.corrupted_images} rejected.")

            # ---------------------------------------------------------
            # STAGE 3: DEDUPLICATION
            # ---------------------------------------------------------
            _notify("DEDUPLICATION", 3, 6, "Auditing duplicate images (SHA-256 & dHash)...")
            unique_images: list[Path] = []
            seen_shas: set[str] = set()
            seen_hashes: list[int] = []

            # Check existing images in target to prevent cross-run duplication
            for existing in target_img_dir.iterdir():
                if existing.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                    try:
                        seen_shas.add(self.validator.compute_sha256(existing))
                    except Exception:
                        pass

            if deduplicate:
                import cv2
                for img_p in valid_images:
                    sha = self.validator.compute_sha256(img_p)
                    if sha in seen_shas:
                        res.exact_duplicates += 1
                        continue
                    seen_shas.add(sha)

                    # Near-duplicate check via dHash
                    mat = cv2.imread(str(img_p))
                    if mat is not None:
                        h = self.validator.compute_dhash(mat)
                        is_near_dup = any(self.validator.hamming_distance(h, existing_h) <= 2 for existing_h in seen_hashes)
                        if is_near_dup:
                            res.near_duplicates += 1
                            continue
                        seen_hashes.append(h)

                    # Move to permanent target images directory
                    final_dest = target_img_dir / img_p.name
                    shutil.copy2(img_p, final_dest)
                    unique_images.append(final_dest)
            else:
                for img_p in valid_images:
                    final_dest = target_img_dir / img_p.name
                    shutil.copy2(img_p, final_dest)
                    unique_images.append(final_dest)

            _notify("DEDUPLICATION", 3, 6, f"Deduplication complete: {len(unique_images)} new unique images saved.")

            # ---------------------------------------------------------
            # STAGE 4: AUTO-ANNOTATION & QA
            # ---------------------------------------------------------
            qa_scores: list[float] = []
            if auto_annotate and unique_images:
                _notify("AUTO_ANNOTATION", 4, 6, f"Running Classical CV Auto-Annotator on {len(unique_images)} frames...")
                for i, img_p in enumerate(unique_images):
                    try:
                        res_ann = self.auto_annotator.annotate_single(
                            img_p,
                            output_dir=candidate_dir,
                            save_metadata=True,
                        )
                        res.auto_annotated_count += 1
                        qa_scores.append(res_ann.qa_report.quality_score)
                    except Exception as exc:
                        res.log(f"Auto-annotation failed for {img_p.name}: {exc}")

                res.mean_quality_score = float(np.mean(qa_scores)) if qa_scores else 0.0
                _notify(
                    "AUTO_ANNOTATION",
                    4,
                    6,
                    f"Generated {res.auto_annotated_count} candidate masks (Mean QA Score: {res.mean_quality_score:.1f}/100).",
                )
            else:
                _notify("AUTO_ANNOTATION", 4, 6, "Auto-annotation skipped.")

            # Refresh workspace manifest
            self.workspace.update_manifest()

            # ---------------------------------------------------------
            # STAGE 5: REPORT GENERATION
            # ---------------------------------------------------------
            _notify("REPORTING", 5, 6, "Generating QC audit reports...")
            report = DatasetReport(
                timestamp=datetime.now().isoformat(),
                source_name=str(source) if isinstance(source, (str, Path)) else "multi_upload",
                source_type="directory" if isinstance(source, Path) and source.is_dir() else "archive_or_files",
                mode="MODE_B_UNLABELED" if res.auto_annotated_count == 0 else "MODE_B_PRE_ANNOTATED",
                total_images=res.total_input_files,
                valid_images=res.valid_images,
                corrupted_images=res.corrupted_images,
                exact_duplicates=res.exact_duplicates,
                near_duplicates=res.near_duplicates,
                masks_found=res.auto_annotated_count > 0,
                matched_pairs=res.auto_annotated_count,
                resolutions={},
                class_pixel_counts={},
                sequence_groups={},
                status_message=f"Pipeline processed {len(unique_images)} images. Auto-annotated: {res.auto_annotated_count}.",
                can_train_supervised=False,
            )
            reports_dir = self.base_dir / "results" / "metrics"
            reports_dir.mkdir(parents=True, exist_ok=True)
            res.report_json_path = reports_dir / "dataset_report.json"
            res.report_md_path = reports_dir / "dataset_report.md"
            report.save_json(res.report_json_path)
            with open(res.report_md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())

            # ---------------------------------------------------------
            # STAGE 6: DATASET SPLIT (Optional)
            # ---------------------------------------------------------
            if auto_split:
                _notify("SPLIT", 6, 6, "Partitioning verified data into train/val/test splits...")
                splitter = DatasetSplitter(
                    source_dir=self.base_dir / "data" / "annotated",
                    train_ratio=cfg.get("dataset", {}).get("train_ratio", 0.70),
                    val_ratio=cfg.get("dataset", {}).get("val_ratio", 0.15),
                    test_ratio=cfg.get("dataset", {}).get("test_ratio", 0.15),
                )
                split_res = splitter.split()
                res.train_count = len(split_res.get("train", []))
                res.val_count = len(split_res.get("val", []))
                res.test_count = len(split_res.get("test", []))
                _notify("SPLIT", 6, 6, f"Split complete: {res.train_count} train, {res.val_count} val, {res.test_count} test.")
            else:
                _notify("SPLIT", 6, 6, "Splitting skipped.")

        finally:
            # Clean staging
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

        res.elapsed_sec = round(time.time() - t0, 2)
        res.log(f"Pipeline finished successfully in {res.elapsed_sec}s.")
        return res
