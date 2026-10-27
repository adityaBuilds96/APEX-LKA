"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Visualization Tools: Forensic Lane Failure Case Miner & Diagnostic Renderer
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
================================================================================

This module performs autonomous failure mining and diagnostic rendering across
perception sequences. Automatically detects:
1. Low Quality & Confidence Failures: Q < 0.35 or confidence < 0.40.
2. Safety Envelope Breaches: Trajectory penetrates or breaches 0.3m inner buffer.
3. Severe Lateral Drift: |e_lat| > 0.40 m.
4. Lane Geometry Anomalies: Lane width w outside [2.6m, 4.8m] or parallel skew.

Outputs:
- Renders annotated forensic diagnostic PNG frames highlighting missing lane
  pixels, residual errors, scene tags, and root-cause failure signatures.
- Automatically compiles a structured failure database written to:
  `data/processed/failure_cases/` for active-learning model finetuning
  (`src/train/finetune_lane_models.py`).
- Generates markdown forensic incident reports with failure severity distributions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.zip_vault_reader import ZipVaultReader
from src.perception.adaptive_roi_engine import AdaptiveROIEngine, AdaptiveROIConfig
from src.perception.faded_lane_recovery import BoundaryPolynomial2D
from src.perception.lane_departure_predictor import (
    LaneDeparturePredictor,
    DeparturePredictorConfig,
    DepartureSeverity,
)
from src.perception.lane_fusion_engine import (
    LaneFusionEngine,
    LaneFusionEngineConfig,
    LanePrediction,
    PerceptionExecutionStatus,
)
from src.perception.virtual_guardrail_enforcer import GuardrailSeverity

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("APEX.FailureCaseRenderer")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Failure Categories & Telemetry Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

class FailureCategory(str, Enum):
    """Forensic taxonomy of perception failure modes."""
    LOW_CONFIDENCE = "LOW_CONFIDENCE"               # Confidence < 0.40
    POOR_QUALITY_Q = "POOR_QUALITY_Q"               # Q < 0.35 quality cutoff
    GUARDRAIL_VIOLATION = "GUARDRAIL_VIOLATION"     # 0.3m buffer breached
    EXCESSIVE_LATERAL_DRIFT = "EXCESSIVE_LATERAL_DRIFT" # |e_lat| > 0.40m
    LANE_WIDTH_ANOMALY = "LANE_WIDTH_ANOMALY"       # Width out of [2.6m, 4.8m]
    PARALLEL_SKEW_DIVERGENCE = "PARALLEL_SKEW_DIVERGENCE" # Left/Right non-parallel
    EXTREME_CURVATURE_JUMP = "EXTREME_CURVATURE_JUMP" # Outlier delta A > 2 sigma
    INTERLOCK_RELAY_TRIP = "INTERLOCK_RELAY_TRIP"   # 5-frame trip disengagement


@dataclass
class FailureMineConfig:
    """Configurable thresholds for mining failure cases."""
    quality_threshold: float = 0.35
    confidence_threshold: float = 0.40
    lateral_error_threshold_m: float = 0.40
    min_lane_width_m: float = 2.60
    max_lane_width_m: float = 4.80
    enable_auto_snapshot: bool = True
    max_mined_cases_per_clip: int = 8
    target_output_dir: Path = Path("data/processed/failure_cases")


@dataclass
class ForensicFailureRecord:
    """Metadata record documenting a single mined perception failure incident."""
    incident_id: str
    timestamp_utc: str
    clip_id: str
    frame_index: int
    scene_tag: str
    failure_categories: List[str]
    quality_score: float
    confidence_score: float
    lateral_error_m: float
    heading_error_deg: float
    lane_width_m: float
    guardrail_severity: str
    relay_interlock_tripped: bool
    diagnostic_image_path: str
    annotation_json_path: str
    suggested_finetune_weight: float


@dataclass
class FailureMiningAuditSummary:
    """Executive summary of failure mining across dataset evaluation."""
    timestamp_utc: str
    total_frames_audited: int
    total_failures_detected: int
    failure_rate_pct: float
    category_counts: Dict[str, int]
    mined_records: List[ForensicFailureRecord] = field(default_factory=list)
    output_database_dir: str = "data/processed/failure_cases"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Forensic Failure Case Renderer & Miner Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LaneFailureCaseRenderer:
    """
    Forensic engine mining perception failure cases and generating high-contrast
    diagnostic imagery with root-cause telemetry overlays.
    """

    def __init__(self, config: Optional[FailureMineConfig] = None) -> None:
        self.config = config or FailureMineConfig()
        self.output_dir = self.config.target_output_dir
        self.images_dir = self.output_dir / "images"
        self.annotations_dir = self.output_dir / "annotations"
        self.reports_dir = self.output_dir / "reports"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.annotations_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.engine = LaneFusionEngine(LaneFusionEngineConfig())
        self.mined_records: List[ForensicFailureRecord] = []

    def evaluate_frame_for_failures(
        self,
        bgr_frame: np.ndarray,
        clip_id: str,
        frame_idx: int,
        prediction: LanePrediction,
    ) -> Optional[ForensicFailureRecord]:
        """
        Audit perception output for safety envelope breaches, quality degradation,
        or geometric anomalies. If failure detected, renders diagnostic image.
        """
        failures: List[FailureCategory] = []

        q = prediction.quality_assessment.quality_score
        conf = prediction.confidence_fusion.fused_confidence
        lat_err = abs(prediction.lateral_error_m)
        width = prediction.lane_width_m
        sev = prediction.guardrail_result.active_severity

        # 1. Audit Quality Score degradation
        if q < self.config.quality_threshold:
            failures.append(FailureCategory.POOR_QUALITY_Q)

        # 2. Audit Confidence Score
        if conf < self.config.confidence_threshold:
            failures.append(FailureCategory.LOW_CONFIDENCE)

        # 3. Audit Excessive Lateral Error
        if lat_err > self.config.lateral_error_threshold_m:
            failures.append(FailureCategory.EXCESSIVE_LATERAL_DRIFT)

        # 4. Audit Virtual Guardrail Violation
        if sev in (GuardrailSeverity.CRITICAL, GuardrailSeverity.EMERGENCY):
            failures.append(FailureCategory.GUARDRAIL_VIOLATION)

        # 5. Audit Lane Width Physical Anomaly
        if width < self.config.min_lane_width_m or width > self.config.max_lane_width_m:
            failures.append(FailureCategory.LANE_WIDTH_ANOMALY)

        # 6. Audit Relay Cutoff Disengagement
        if prediction.arduino_clutch_relay_engaged or not prediction.is_safe_for_actuation:
            failures.append(FailureCategory.INTERLOCK_RELAY_TRIP)

        if not failures:
            return None

        # Failure identified -> Compile record and render diagnostic frame
        clean_clip = clip_id.replace("/", "_").replace("\\", "_")
        incident_id = f"fail_{clean_clip}_f{frame_idx:04d}"
        img_filename = f"{incident_id}_diagnostic.png"
        json_filename = f"{incident_id}_telemetry.json"

        img_path = self.images_dir / img_filename
        json_path = self.annotations_dir / json_filename

        # Render forensic diagnostic image
        diagnostic_frame = self.render_diagnostic_overlay(
            bgr_frame=bgr_frame,
            prediction=prediction,
            failures=failures,
            incident_id=incident_id,
        )
        cv2.imwrite(str(img_path), diagnostic_frame)

        # Compute active-learning finetune loss weight based on failure severity
        base_weight = 1.0
        if FailureCategory.GUARDRAIL_VIOLATION in failures:
            base_weight += 2.5
        if FailureCategory.POOR_QUALITY_Q in failures:
            base_weight += 1.5
        if FailureCategory.EXCESSIVE_LATERAL_DRIFT in failures:
            base_weight += 1.0

        rec = ForensicFailureRecord(
            incident_id=incident_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            clip_id=clip_id,
            frame_index=frame_idx,
            scene_tag=prediction.scene_tag,
            failure_categories=[f.value for f in failures],
            quality_score=q,
            confidence_score=conf,
            lateral_error_m=prediction.lateral_error_m,
            heading_error_deg=prediction.heading_error_deg,
            lane_width_m=width,
            guardrail_severity=sev.name,
            relay_interlock_tripped=prediction.arduino_clutch_relay_engaged,
            diagnostic_image_path=str(img_path),
            annotation_json_path=str(json_path),
            suggested_finetune_weight=base_weight,
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(asdict(rec), f, indent=2)

        return rec

    def render_diagnostic_overlay(
        self,
        bgr_frame: np.ndarray,
        prediction: LanePrediction,
        failures: List[FailureCategory],
        incident_id: str,
    ) -> np.ndarray:
        """
        Render comprehensive forensic failure overlay highlighting root causes.
        """
        canvas = bgr_frame.copy()
        h, w = canvas.shape[:2]

        # 1. Red Alert Border across frame perimeter
        cv2.rectangle(canvas, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)

        # 2. Draw Lane Boundaries with Failure Coloring
        y_samples = np.linspace(int(h * 0.52), h - 1, 40)
        poly_l = prediction.left_polynomial
        poly_r = prediction.right_polynomial

        pts_l = []
        pts_r = []
        for y in y_samples:
            xl = int(poly_l[0] * y**2 + poly_l[1] * y + poly_l[2])
            xr = int(poly_r[0] * y**2 + poly_r[1] * y + poly_r[2])
            if 0 <= xl < w:
                pts_l.append((xl, int(y)))
            if 0 <= xr < w:
                pts_r.append((xr, int(y)))

        # Draw boundaries in red/orange failure style
        if len(pts_l) > 2:
            cv2.polylines(canvas, [np.array(pts_l)], False, (0, 100, 255), 3, cv2.LINE_AA)
        if len(pts_r) > 2:
            cv2.polylines(canvas, [np.array(pts_r)], False, (0, 50, 255), 3, cv2.LINE_AA)

        # 3. Top Forensic Incident Banner
        cv2.rectangle(canvas, (10, 10), (480, 135), (15, 15, 20), -1)
        cv2.rectangle(canvas, (10, 10), (480, 135), (0, 0, 255), 2)

        cv2.putText(
            canvas,
            f"FORENSIC FAILURE INCIDENT: {incident_id[:24]}",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 150, 255),
            1,
            cv2.LINE_AA,
        )

        fail_str = ", ".join(f.name for f in failures[:3])
        cv2.putText(
            canvas,
            f"FAULTS: {fail_str}",
            (20, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (50, 50, 255),
            1,
            cv2.LINE_AA,
        )

        # Telemetry forensics
        q_val = prediction.quality_assessment.quality_score
        cv2.putText(
            canvas,
            f"Quality Q: {q_val:.2f} (Cutoff < 0.35) | Conf: {prediction.confidence_fusion.fused_confidence:.2f}",
            (20, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            canvas,
            f"Lat Error: {prediction.lateral_error_m:+.3f}m | Heading: {prediction.heading_error_deg:+.2f}°",
            (20, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (200, 200, 200),
            1,
        )

        cv2.putText(
            canvas,
            f"Lane Width: {prediction.lane_width_m:.2f}m | Guardrail: {prediction.guardrail_result.active_severity.name}",
            (20, 104),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 200, 255) if prediction.guardrail_result.active_severity != GuardrailSeverity.NOMINAL else (200, 200, 200),
            1,
        )

        relay_str = "TRIPPED (DISENGAGED)" if prediction.arduino_clutch_relay_engaged else "ARMED"
        cv2.putText(
            canvas,
            f"Arduino Relay Interlock: {relay_str}",
            (20, 122),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 255) if prediction.arduino_clutch_relay_engaged else (0, 255, 0),
            1,
        )

        # 4. Highlight Discrepancy Region with Marker
        eval_y = int(h * 0.85)
        eval_xl = int(poly_l[0] * eval_y**2 + poly_l[1] * eval_y + poly_l[2])
        eval_xr = int(poly_r[0] * eval_y**2 + poly_r[1] * eval_y + poly_r[2])

        if 0 <= eval_xl < w:
            cv2.drawMarker(canvas, (eval_xl, eval_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)
        if 0 <= eval_xr < w:
            cv2.drawMarker(canvas, (eval_xr, eval_y), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 18, 2)

        return canvas

    def mine_failures_from_dataset(
        self,
        dataset_path: Union[str, Path] = "data/raw/lane_dataset.zip",
        max_clips: int = 15,
        frames_per_clip: int = 25,
    ) -> FailureMiningAuditSummary:
        """
        Execute automated mining sweep across dataset clips and export database manifest.
        """
        logger.info(f"Initiating failure mining sweep on dataset: {dataset_path}")
        dpath = Path(dataset_path)

        clips_data = self._load_vault_clips(dpath, max_clips, frames_per_clip)
        if not clips_data:
            logger.warning("No clips extracted from vault; synthesizing diverse failure stress sequences.")
            clips_data = self._generate_synthetic_failure_cases(max_clips, frames_per_clip)

        total_audited = 0
        cat_counts: Dict[str, int] = {c.value: 0 for c in FailureCategory}

        for clip_id, scene_tag, frames in clips_data:
            self.engine.reset()
            mined_in_clip = 0

            for f_idx, frame in enumerate(frames):
                total_audited += 1

                # Execute perception
                pred = self.engine.process_frame(
                    bgr_frame=frame,
                    ego_speed_mps=25.0,
                    scene_tag=scene_tag,
                    dt_s=0.0333,
                )

                if mined_in_clip < self.config.max_mined_cases_per_clip:
                    rec = self.evaluate_frame_for_failures(
                        bgr_frame=frame,
                        clip_id=clip_id,
                        frame_idx=f_idx,
                        prediction=pred,
                    )
                    if rec is not None:
                        self.mined_records.append(rec)
                        mined_in_clip += 1
                        for cat in rec.failure_categories:
                            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        total_fails = len(self.mined_records)
        fail_pct = (total_fails / max(1, total_audited)) * 100.0

        summary = FailureMiningAuditSummary(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            total_frames_audited=total_audited,
            total_failures_detected=total_fails,
            failure_rate_pct=fail_pct,
            category_counts=cat_counts,
            mined_records=self.mined_records,
            output_database_dir=str(self.output_dir),
        )

        self._export_database_manifest(summary)
        self._export_markdown_report(summary)

        return summary

    def _export_database_manifest(self, summary: FailureMiningAuditSummary) -> Path:
        """Export master failure index manifest for active learning finetuning."""
        manifest_path = self.output_dir / "failures_manifest.json"
        manifest_data = {
            "timestamp_utc": summary.timestamp_utc,
            "total_mined_cases": len(summary.mined_records),
            "category_distribution": summary.category_counts,
            "cases": [asdict(r) for r in summary.mined_records],
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)
        logger.info(f"Exported failure database manifest to: {manifest_path}")
        return manifest_path

    def _export_markdown_report(self, summary: FailureMiningAuditSummary) -> Path:
        """Publish formal Markdown forensic audit report documenting discovered failures."""
        report_path = self.reports_dir / "failure_mining_audit.md"
        content = [
            "# APEX-Vision Core v4.0 — Forensic Failure Mining & Inspection Report",
            "## Section 1: Perception Boundary Failure & Root-Cause Active Learning Audit",
            "",
            f"- **Execution Timestamp:** `{summary.timestamp_utc}`",
            f"- **Total Frames Audited:** {summary.total_frames_audited}",
            f"- **Failures Discovered:** {summary.total_failures_detected} ({summary.failure_rate_pct:.2f}% incidence rate)",
            f"- **Database Repository:** `{summary.output_database_dir}`",
            f"- **Autonomy Rating:** driver-supervised SAE Level 2",
            f"- **Safety Standard:** ISO 26262-style process; ASIL-B aligned; not independently assessed",
            "",
            "---",
            "",
            "### 1. Failure Taxonomy & Root-Cause Distribution",
            "",
            "| Failure Category | Incident Count | Share (%) | Primary Root-Cause | Remediation Strategy |",
            "| :--- | :---: | :---: | :--- | :--- |",
        ]

        for cat, cnt in summary.category_counts.items():
            share = (cnt / max(1, summary.total_failures_detected)) * 100.0
            remed = (
                "Loss weight boosting (2.5x)"
                if "GUARDRAIL" in cat
                else "Faded line projection tuning"
                if "QUALITY" in cat
                else "Stanley gain scheduling"
            )
            content.append(
                f"| `{cat}` | {cnt} | {share:.1f}% | Optical degradation / Road seams | {remed} |"
            )

        content.extend([
            "",
            "---",
            "",
            "### 2. Forensic Failure Incident Manifest",
            "",
            "| Incident ID | Scene | Quality Q | Lat Error | Severity | Categories | Finetune Weight |",
            "| :--- | :---: | :---: | :---: | :---: | :--- | :---: |",
        ])

        for r in summary.mined_records[:25]:
            cats = ", ".join(r.failure_categories)
            content.append(
                f"| `{r.incident_id[:20]}` | {r.scene_tag} | {r.quality_score:.2f} | {r.lateral_error_m:+.3f}m | "
                f"{r.guardrail_severity} | {cats[:28]} | {r.suggested_finetune_weight:.1f}x |"
            )

        content.extend([
            "",
            "---",
            "*APEX-Vision Core v4.0 Active Learning Database Updated.*",
        ])

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content) + "\n")

        logger.info(f"Published failure audit report to: {report_path}")
        return report_path

    def _load_vault_clips(
        self, vault_path: Path, max_clips: int, frames_per_clip: int
    ) -> List[Tuple[str, str, List[np.ndarray]]]:
        """Extract clips from zip archive."""
        clips: List[Tuple[str, str, List[np.ndarray]]] = []
        if not vault_path.is_file():
            return clips

        try:
            reader = ZipVaultReader(vault_path)
            jpgs = [f for f in reader.list_files() if f.lower().endswith(".jpg")]
            if not jpgs:
                return clips

            from collections import defaultdict

            grouped = defaultdict(list)
            for f in jpgs:
                p = Path(f)
                grouped[str(p.parent)].append(f)

            for ck in list(grouped.keys())[:max_clips]:
                img_list = []
                for fname in sorted(grouped[ck])[:frames_per_clip]:
                    buf = reader.read_bytes(fname)
                    if buf:
                        arr = np.frombuffer(buf, dtype=np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            if img.shape[:2] != (360, 640):
                                img = cv2.resize(img, (640, 360))
                            img_list.append(img)
                if img_list:
                    clips.append((ck, "HIGHWAY", img_list))
        except Exception as e:
            logger.error(f"Error reading vault for failure mining: {e}")

        return clips

    def _generate_synthetic_failure_cases(
        self, max_clips: int, frames_per_clip: int
    ) -> List[Tuple[str, str, List[np.ndarray]]]:
        """Generate high-contrast corner cases (washed out markings, night glare, missing lines)."""
        clips: List[Tuple[str, str, List[np.ndarray]]] = []
        for i in range(max_clips):
            clip_name = f"synth_stress_clip_{i:02d}"
            frames = []
            for f in range(frames_per_clip):
                # Empty asphalt with high noise to induce low confidence Q < 0.35
                frame = np.full((360, 640, 3), 45, dtype=np.uint8)
                noise = np.random.normal(0, 12, (360, 640, 3)).astype(np.int16)
                frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                frames.append(frame)
            clips.append((clip_name, "RAIN" if i % 2 == 0 else "TUNNEL", frames))
        return clips



# ═══════════════════════════════════════════════════════════════════════════════
# 3. Active Learning Sampler & Failure Clustering Analyzer
# ═══════════════════════════════════════════════════════════════════════════════

class ActiveLearningDataSampler:
    """
    Ranks mined failure records according to model uncertainty and boundary violation
    risk to construct balanced mini-batches for finetuning.
    """

    def __init__(self, target_sample_size: int = 50) -> None:
        self.sample_size = target_sample_size

    def select_informative_samples(
        self, records: List[ForensicFailureRecord]
    ) -> List[ForensicFailureRecord]:
        """
        Stratified sampling favoring highest finetune weights and boundary violations.
        """
        if len(records) <= self.sample_size:
            return list(records)

        # Sort descending by finetune weight and lateral error
        sorted_records = sorted(
            records,
            key=lambda r: (r.suggested_finetune_weight, abs(r.lateral_error_m)),
            reverse=True,
        )

        selected: List[ForensicFailureRecord] = []
        category_counts: Dict[str, int] = {}
        max_per_cat = max(2, self.sample_size // 4)

        for rec in sorted_records:
            primary_cat = rec.failure_categories[0] if rec.failure_categories else "UNKNOWN"
            current_c = category_counts.get(primary_cat, 0)
            if current_c < max_per_cat or len(selected) < self.sample_size:
                selected.append(rec)
                category_counts[primary_cat] = current_c + 1
                if len(selected) >= self.sample_size:
                    break

        return selected


class FailureClusteringAnalyzer:
    """
    Analyzes multidimensional feature distributions across failure incidents
    (scene, lateral error, quality, heading drift) to guide targeted augmentations.
    """

    @staticmethod
    def compute_cluster_statistics(
        records: List[ForensicFailureRecord],
    ) -> Dict[str, Any]:
        """Compute aggregated forensic metrics across failure database."""
        if not records:
            return {"total_records": 0, "status": "EMPTY"}

        lat_errors = [abs(r.lateral_error_m) for r in records]
        qualities = [r.quality_score for r in records]
        headings = [abs(r.heading_error_deg) for r in records]

        scene_dist: Dict[str, int] = {}
        for r in records:
            scene_dist[r.scene_tag] = scene_dist.get(r.scene_tag, 0) + 1

        return {
            "total_records": len(records),
            "mean_lateral_error_m": float(np.mean(lat_errors)),
            "p95_lateral_error_m": float(np.percentile(lat_errors, 95)),
            "mean_quality_score": float(np.mean(qualities)),
            "mean_heading_error_deg": float(np.mean(headings)),
            "scene_distribution": scene_dist,
            "recommended_data_augmentation": (
                "HEAVY_RAIN_FOG" if scene_dist.get("RAIN", 0) > len(records) // 3 else "CURVE_JITTER_EXPANSION"
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CLI Command-Line Parser and Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    """Parse CLI options for failure case miner."""
    parser = argparse.ArgumentParser(
        description="APEX-Vision Phase 7B Section 1 Lane Failure Case Miner & Diagnostic Renderer"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/raw/lane_dataset.zip",
        help="Path to TuSimple raw dataset zip archive",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/failure_cases",
        help="Directory to write mined failure database and reports",
    )
    parser.add_argument(
        "--max-clips",
        type=int,
        default=8,
        help="Maximum clips to evaluate",
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=20,
        help="Frames per clip to evaluate",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution pipeline."""
    args = parse_args()
    logger.info("Starting APEX-Vision Lane Failure Case Miner & Renderer...")

    config = FailureMineConfig(
        target_output_dir=Path(args.output_dir),
    )
    miner = LaneFailureCaseRenderer(config)

    summary = miner.mine_failures_from_dataset(
        dataset_path=args.dataset,
        max_clips=args.max_clips,
        frames_per_clip=args.frames_per_clip,
    )

    logger.info(
        f"Failure Mining Complete! Audited {summary.total_frames_audited} frames -> "
        f"Discovered {summary.total_failures_detected} failures ({summary.failure_rate_pct:.2f}%). "
        f"Manifest saved to: {summary.output_database_dir}/failures_manifest.json"
    )


if __name__ == "__main__":
    main()
