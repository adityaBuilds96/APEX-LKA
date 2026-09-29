"""
src/dataset/pipeline_orchestrator.py
====================================
Master end-to-end automated data pipeline orchestrator for APEX LKA.

Chains the 6 operational stages defined in Section 6:
  1. Intake (Images, Videos, ZIP archives) with safe staging
  2. Validation (Resolution profiling, SHA-256 exact & dHash near-duplicate filtering)
  3. Auto-Annotation (Classical CV 4-class pseudo-masks with confidence tiers)
  4. Quality Scoring (9-point ADAS QA check via QualityScorer)
  5. Reporting (JSON + Markdown audit metrics & auto-annotation report)
  6. Dataset Splitting (Sequence-aware partition into train/val/test)

Maintains persistent state in annotations/pipeline_state.json for real-time
dashboard progress polling and headless CLI operations.
"""

import json
import shutil
import time
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import PROJECT_ROOT, cfg
from src.data_collection.dataset_splitter import DatasetSplitter
from src.data_collection.video_to_frames import FrameExtractor
from src.dataset.annotation_prep import AnnotationWorkspace
from src.dataset.auto_annotator import ClassicalAutoAnnotator
from src.dataset.quality_scorer import QualityScorer, QualityTier
from src.dataset.report import DatasetReport
from src.dataset.validator import DatasetValidator


@dataclass
class PipelineState:
    batch_id: str
    stage: str  # "intake", "validation", "auto_annotation", "quality_scoring", "reporting", "split", "completed", "failed"
    progress: float  # [0.0, 1.0]
    started_at: str
    frames_total: int = 0
    frames_processed: int = 0
    frames_failed: int = 0
    frames_duplicate: int = 0
    mean_confidence: float = 0.0
    mean_quality_score: float = 0.0
    message: str = ""
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def save(self, filepath: Path) -> None:
        """Persist state to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = datetime.now().isoformat()
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)


@dataclass
class PipelineExecutionSummary:
    batch_id: str
    success: bool
    total_staged: int
    valid_frames: int
    duplicates_filtered: int
    corrupted_files: int
    auto_annotated_count: int
    high_confidence_count: int
    mean_quality_score: float
    train_count: int
    val_count: int
    test_count: int
    elapsed_seconds: float
    report_json: Optional[Path] = None
    report_md: Optional[Path] = None
    auto_annotation_report_md: Optional[Path] = None
    logs: List[str] = field(default_factory=list)


class PipelineOrchestrator:
    """
    Automated data lifecycle orchestrator.
    """

    SUPPORTED_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    SUPPORTED_VID_EXTS = {".mp4", ".avi", ".mov", ".mkv"}
    SUPPORTED_ARCHIVES = {".zip"}

    def __init__(
        self,
        base_dir: Optional[Path] = None,
        state_file: Optional[Path] = None,
    ):
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT
        pipe_cfg = cfg.get("pipeline", {})
        sf = pipe_cfg.get("state_file", "annotations/pipeline_state.json")
        self.state_file = Path(state_file) if state_file else self.base_dir / sf

        self.validator = DatasetValidator()
        self.annotator = ClassicalAutoAnnotator()
        self.scorer = QualityScorer()
        self.workspace = AnnotationWorkspace.create(self.base_dir)

    def run(
        self,
        source: Union[str, Path, List[Path]],
        auto_split: bool = False,
        progress_callback: Optional[Callable[[str, float, str], None]] = None,
    ) -> PipelineExecutionSummary:
        """
        Execute the end-to-end automated data pipeline.

        Args:
            source: Source folder, zip file, video, or list of file paths.
            auto_split: If True, automatically split verified data into train/val/test.
            progress_callback: Callback receiving (stage_name, progress_0_to_1, message).

        Returns:
            PipelineExecutionSummary dataclass.
        """
        t0 = time.time()
        batch_id = str(uuid.uuid4())[:8]
        logs: List[str] = []

        def _log(msg: str):
            t_str = datetime.now().strftime("%H:%M:%S")
            logs.append(f"[{t_str}] {msg}")

        state = PipelineState(
            batch_id=batch_id,
            stage="intake",
            progress=0.0,
            started_at=datetime.now().isoformat(),
            message="Initializing intake...",
        )
        state.save(self.state_file)

        staging_dir = self.base_dir / "data" / ".staging" / batch_id
        raw_staging = staging_dir / "raw"
        validated_staging = staging_dir / "validated"
        raw_staging.mkdir(parents=True, exist_ok=True)
        validated_staging.mkdir(parents=True, exist_ok=True)

        target_img_dir = self.base_dir / "data" / "annotated" / "images"
        target_mask_dir = self.base_dir / "data" / "annotated" / "masks"
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        def _update_stage(stage: str, prog: float, msg: str):
            state.stage = stage
            state.progress = prog
            state.message = msg
            state.save(self.state_file)
            _log(f"[{stage.upper()}] {msg}")
            if progress_callback:
                progress_callback(stage, prog, msg)

        staged_files: List[Path] = []
        valid_frames: List[Path] = []
        dup_count = 0
        corrupt_count = 0
        annotated_count = 0
        high_conf_count = 0
        quality_scores: List[float] = []

        try:
            # ---------------------------------------------------------
            # STAGE 1: INTAKE
            # ---------------------------------------------------------
            _update_stage("intake", 0.05, "Extracting and staging incoming files...")
            sources = [Path(source)] if isinstance(source, (str, Path)) else [Path(p) for p in source]

            for s in sources:
                if s.is_file():
                    ext = s.suffix.lower()
                    if ext in self.SUPPORTED_ARCHIVES:
                        _log(f"Unpacking archive: {s.name}")
                        with zipfile.ZipFile(s, "r") as zf:
                            for member in zf.infolist():
                                target = (raw_staging / member.filename).resolve()
                                if not str(target).startswith(str(raw_staging.resolve())):
                                    continue
                                if not member.is_dir():
                                    target.parent.mkdir(parents=True, exist_ok=True)
                                    with zf.open(member) as sf_in, open(target, "wb") as df_out:
                                        shutil.copyfileobj(sf_in, df_out)
                                    if target.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                                        staged_files.append(target)
                    elif ext in self.SUPPORTED_VID_EXTS:
                        _log(f"Extracting frames from video: {s.name}")
                        extractor = FrameExtractor(video_path=s, output_dir=raw_staging, target_fps=3.0)
                        extractor.extract()
                        for f in raw_staging.iterdir():
                            if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMG_EXTS and f not in staged_files:
                                staged_files.append(f)
                    elif ext in self.SUPPORTED_IMG_EXTS:
                        dst = raw_staging / s.name
                        shutil.copy2(s, dst)
                        staged_files.append(dst)

                elif s.is_dir():
                    _log(f"Scanning directory: {s}")
                    for f in s.rglob("*"):
                        if f.is_file() and f.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                            dst = raw_staging / f.name
                            shutil.copy2(f, dst)
                            staged_files.append(dst)
                        elif f.is_file() and f.suffix.lower() in self.SUPPORTED_VID_EXTS:
                            extractor = FrameExtractor(video_path=f, output_dir=raw_staging, target_fps=3.0)
                            extractor.extract()
                            for frame_f in raw_staging.iterdir():
                                if frame_f.is_file() and frame_f.suffix.lower() in self.SUPPORTED_IMG_EXTS and frame_f not in staged_files:
                                    staged_files.append(frame_f)

            state.frames_total = len(staged_files)
            _update_stage("intake", 0.20, f"Staged {state.frames_total} input frames.")

            # ---------------------------------------------------------
            # STAGE 2: VALIDATION & DEDUPLICATION
            # ---------------------------------------------------------
            _update_stage("validation", 0.25, "Running SHA-256 and dHash integrity validation...")
            seen_shas: set = set()
            seen_hashes: List[int] = []

            # Populate existing hashes to avoid cross-batch duplicates
            for existing in target_img_dir.iterdir():
                if existing.is_file() and existing.suffix.lower() in self.SUPPORTED_IMG_EXTS:
                    try:
                        seen_shas.add(self.validator.compute_sha256(existing))
                    except Exception:
                        pass

            for f in staged_files:
                is_valid, shape, err = self.validator.validate_image(f)
                if not is_valid:
                    corrupt_count += 1
                    continue

                sha = self.validator.compute_sha256(f)
                if sha in seen_shas:
                    dup_count += 1
                    continue
                seen_shas.add(sha)

                # dHash perceptual check
                mat = cv2.imread(str(f))
                if mat is not None:
                    h_val = self.validator.compute_dhash(mat)
                    if any(self.validator.hamming_distance(h_val, eh) <= 2 for eh in seen_hashes):
                        dup_count += 1
                        continue
                    seen_hashes.append(h_val)

                # Move to validated staging
                v_dst = validated_staging / f.name
                shutil.copy2(f, v_dst)
                valid_frames.append(v_dst)

            state.frames_duplicate = dup_count
            state.frames_failed = corrupt_count
            _update_stage("validation", 0.40, f"Validated: {len(valid_frames)} frames ({dup_count} dups, {corrupt_count} corrupt).")

            # ---------------------------------------------------------
            # STAGE 3 & 4: AUTO-ANNOTATION & QUALITY SCORING
            # ---------------------------------------------------------
            _update_stage("auto_annotation", 0.45, "Generating Classical CV pseudo-masks...")
            manifest_items = {}

            for idx, vf in enumerate(valid_frames):
                # Copy image to permanent annotated images folder
                perm_img = target_img_dir / vf.name
                shutil.copy2(vf, perm_img)

                # Auto-annotate
                mask_full, meta = self.annotator.generate_pseudo_mask(perm_img)
                perm_mask = target_mask_dir / f"{perm_img.stem}.png"
                cv2.imwrite(str(perm_mask), mask_full)

                annotated_count += 1
                if meta.confidence_tier == "HIGH":
                    high_conf_count += 1

                # Quality score
                q_rep = self.scorer.evaluate_pair(perm_img, perm_mask, stem=perm_img.stem)
                quality_scores.append(q_rep.score)

                manifest_items[perm_img.stem] = {
                    "stem": perm_img.stem,
                    "status": "auto_high" if meta.confidence_tier == "HIGH" else "auto_review",
                    "confidence": meta.confidence,
                    "quality_score": q_rep.score,
                    "tier": q_rep.tier.value,
                }

                prog_frac = 0.45 + (idx / max(1, len(valid_frames))) * 0.35
                state.frames_processed = idx + 1
                state.progress = round(prog_frac, 2)
                state.save(self.state_file)

            # Update workspace manifest
            self.workspace.update_manifest()

            mean_q = float(np.mean(quality_scores)) if quality_scores else 0.0
            state.mean_quality_score = round(mean_q, 2)

            # ---------------------------------------------------------
            # STAGE 5: REPORTING
            # ---------------------------------------------------------
            _update_stage("reporting", 0.85, "Compiling QC & auto-annotation audit reports...")
            reports_dir = self.base_dir / "results" / "metrics"
            reports_dir.mkdir(parents=True, exist_ok=True)

            rep_json_path = reports_dir / "dataset_report.json"
            rep_md_path = reports_dir / "dataset_report.md"
            auto_rep_md_path = reports_dir / "auto_annotation_report.md"

            report = DatasetReport(
                timestamp=datetime.now().isoformat(),
                source_name=batch_id,
                source_type="automated_pipeline",
                total_images=len(staged_files),
                valid_images=len(valid_frames),
                corrupted_images=corrupt_count,
                exact_duplicates=dup_count,
                near_duplicates=0,
                masks_found=(annotated_count > 0),
                matched_pairs=annotated_count,
                status_message=f"Pipeline batch {batch_id} processed {len(valid_frames)} frames successfully.",
            )
            report.save_json(rep_json_path)
            with open(rep_md_path, "w", encoding="utf-8") as f:
                f.write(report.to_markdown())

            # Write auto-annotation coverage report
            with open(auto_rep_md_path, "w", encoding="utf-8") as f:
                f.write(f"""# Auto-Annotation Coverage Report
**Batch ID:** `{batch_id}`  
**Generated At:** {datetime.now().isoformat()}  
**Total Frames Auto-Annotated:** {annotated_count}  
**High Confidence (Auto-Approved):** {high_conf_count} ({round(high_conf_count / max(1, annotated_count) * 100, 1)}%)  
**Mean Quality Score:** {mean_q:.2f} / 1.00  

## Class Legend
- **Class 0:** Background
- **Class 1:** Road Surface
- **Class 2:** Left Lane Boundary
- **Class 3:** Right Lane Boundary
""")

            # ---------------------------------------------------------
            # STAGE 6: SPLITTING (Optional)
            # ---------------------------------------------------------
            tr_cnt, val_cnt, te_cnt = 0, 0, 0
            if auto_split and annotated_count > 0:
                _update_stage("split", 0.95, "Partitioning verified data into train/val/test splits...")
                splitter = DatasetSplitter(source_dir=self.base_dir / "data" / "annotated")
                split_dict = splitter.split()
                tr_cnt = len(split_dict.get("train", []))
                val_cnt = len(split_dict.get("val", []))
                te_cnt = len(split_dict.get("test", []))

            _update_stage("completed", 1.0, f"Batch {batch_id} finished in {time.time() - t0:.1f}s.")

            return PipelineExecutionSummary(
                batch_id=batch_id,
                success=True,
                total_staged=len(staged_files),
                valid_frames=len(valid_frames),
                duplicates_filtered=dup_count,
                corrupted_files=corrupt_count,
                auto_annotated_count=annotated_count,
                high_confidence_count=high_conf_count,
                mean_quality_score=round(mean_q, 2),
                train_count=tr_cnt,
                val_count=val_cnt,
                test_count=te_cnt,
                elapsed_seconds=round(time.time() - t0, 2),
                report_json=rep_json_path,
                report_md=rep_md_path,
                auto_annotation_report_md=auto_rep_md_path,
                logs=logs,
            )

        except Exception as exc:
            _update_stage("failed", state.progress, f"Error: {exc}")
            return PipelineExecutionSummary(
                batch_id=batch_id,
                success=False,
                total_staged=len(staged_files),
                valid_frames=len(valid_frames),
                duplicates_filtered=dup_count,
                corrupted_files=corrupt_count,
                auto_annotated_count=annotated_count,
                high_confidence_count=high_conf_count,
                mean_quality_score=0.0,
                train_count=0,
                val_count=0,
                test_count=0,
                elapsed_seconds=round(time.time() - t0, 2),
                logs=logs,
            )

        finally:
            # Clean staging
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
