"""
src/dataset/quality_scorer.py
=============================
Per-frame semantic segmentation mask quality assessment system for APEX LKA.

Implements all 9 Quality Checks defined in Section 7:
  1. Mask-Image Dimension Match (CRITICAL)
  2. Valid Class IDs in {0, 1, 2, 3} (CRITICAL)
  3. Road Coverage Ratio in ROI (WARNING)
  4. Lane Marking Coverage Ratio (WARNING)
  5. Lane Spatial Consistency / Centroid Ordering (CRITICAL)
  6. Lane Width Plausibility at Frame Base (WARNING)
  7. Lane Component Continuity / Fragmentation (INFO)
  8. Background Contamination / Road Holes in ROI (WARNING)
  9. Sky / Horizon Leak Detection (WARNING)

Outputs weighted quality score in [0.0, 1.0] and classifications:
  - EXCELLENT (>= 0.85)
  - ACCEPTABLE (0.65 - 0.85)
  - NEEDS_REVIEW (0.40 - 0.65)
  - REJECTED (< 0.40)
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.config import cfg


class CheckSeverity(str, Enum):
    CRITICAL = "CRITICAL"
    WARNING = "WARNING"
    INFO = "INFO"


Severity = CheckSeverity


class QualityTier(str, Enum):
    EXCELLENT = "EXCELLENT"
    ACCEPTABLE = "ACCEPTABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    REJECTED = "REJECTED"


@dataclass
class CheckResult:
    check_id: int
    name: str
    severity: CheckSeverity
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def check_name(self) -> str:
        return self.name


@dataclass
class QualityScoreReport:
    stem: str
    image_shape: Tuple[int, int]
    mask_shape: Tuple[int, int]
    score: float  # [0.0, 1.0]
    tier: QualityTier
    passed_all_critical: bool
    checks: List[CheckResult]
    class_pixel_counts: Dict[int, int]
    class_pixel_fractions: Dict[int, float]
    summary_message: str = ""

    @property
    def overall_score(self) -> float:
        return self.score

    @property
    def issues(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tier"] = self.tier.value
        for c in data["checks"]:
            c["severity"] = c["severity"]
        return data


QualityReport = QualityScoreReport


class QualityScorer:
    """
    Evaluates semantic segmentation masks against production ADAS standards.
    """

    ALLOWED_CLASSES = {0, 1, 2, 3}

    def __init__(self, config_dict: Optional[dict] = None):
        qc = config_dict or cfg.get("quality_scoring", {})
        self.weights = {
            CheckSeverity.CRITICAL: float(qc.get("weights", {}).get("critical", 3.0)),
            CheckSeverity.WARNING:  float(qc.get("weights", {}).get("warning", 1.5)),
            CheckSeverity.INFO:     float(qc.get("weights", {}).get("info", 0.5)),
        }
        self.road_cov_min = float(qc.get("road_coverage_min", 0.10))
        self.road_cov_max = float(qc.get("road_coverage_max", 0.85))
        self.lane_cov_min = float(qc.get("lane_coverage_min", 0.005))
        self.lane_cov_max = float(qc.get("lane_coverage_max", 0.08))
        self.lane_w_min   = int(qc.get("lane_width_min_px", 100))
        self.lane_w_max   = int(qc.get("lane_width_max_px", 400))
        self.max_fragments = int(qc.get("max_lane_fragments", 3))
        self.sky_top_fraction = float(qc.get("sky_region_top_fraction", 0.30))

        th = qc.get("thresholds", {})
        self.thresh_excellent = float(th.get("excellent", 0.85))
        self.thresh_acceptable = float(th.get("acceptable", 0.65))
        self.thresh_review = float(th.get("needs_review", 0.40))

    def evaluate_pair(
        self,
        image_input: Union[str, Path, np.ndarray],
        mask_input: Union[str, Path, np.ndarray],
        stem: Optional[str] = None,
    ) -> QualityScoreReport:
        """
        Execute all 9 quality checks on an image-mask pair.

        Args:
            image_input: Path to image or RGB/BGR numpy array.
            mask_input: Path to mask or 2D uint8 numpy array.
            stem: File stem identifier.

        Returns:
            QualityScoreReport with individual check results and overall score.
        """
        # Load image
        img_name = stem or "sample"
        if isinstance(image_input, (str, Path)):
            p_img = Path(image_input)
            img_name = p_img.stem
            img = cv2.imread(str(p_img))
        else:
            img = image_input

        # Load mask
        if isinstance(mask_input, (str, Path)):
            p_mask = Path(mask_input)
            raw_mask = cv2.imread(str(p_mask), cv2.IMREAD_UNCHANGED)
        else:
            raw_mask = mask_input

        if img is None or raw_mask is None:
            return self._create_failure_report(img_name, "Unable to decode image or mask file.")

        # Ensure mask is 2D uint8
        if len(raw_mask.shape) == 3:
            if np.array_equal(raw_mask[:, :, 0], raw_mask[:, :, 1]) and np.array_equal(raw_mask[:, :, 1], raw_mask[:, :, 2]):
                mask_2d = raw_mask[:, :, 0].astype(np.uint8)
            else:
                mask_2d = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            mask_2d = raw_mask.astype(np.uint8)

        img_h, img_w = img.shape[:2]
        mask_h, mask_w = mask_2d.shape[:2]

        checks: List[CheckResult] = []

        # -------------------------------------------------------------
        # CHECK 1: Dimension Match (CRITICAL)
        # -------------------------------------------------------------
        c1_pass = (img_h == mask_h and img_w == mask_w)
        checks.append(CheckResult(
            check_id=1,
            name="Dimension Match",
            severity=CheckSeverity.CRITICAL,
            passed=c1_pass,
            message="Dimensions match" if c1_pass else f"Mismatch: Image ({img_w}x{img_h}) vs Mask ({mask_w}x{mask_h})",
            details={"image_shape": (img_h, img_w), "mask_shape": (mask_h, mask_w)},
        ))

        # If dimensions don't match, resize mask for remaining spatial tests
        if not c1_pass:
            mask_2d = cv2.resize(mask_2d, (img_w, img_h), interpolation=cv2.INTER_NEAREST)
            mask_h, mask_w = img_h, img_w

        total_pixels = mask_h * mask_w
        roi_top = int(mask_h * 0.35)
        roi_pixels = (mask_h - roi_top) * mask_w
        roi_mask = mask_2d[roi_top:mask_h, :]

        # -------------------------------------------------------------
        # CHECK 2: Valid Class IDs (CRITICAL)
        # -------------------------------------------------------------
        unique_classes = set(np.unique(mask_2d).tolist())
        invalid_classes = unique_classes - self.ALLOWED_CLASSES
        c2_pass = (len(invalid_classes) == 0)
        checks.append(CheckResult(
            check_id=2,
            name="Valid Class IDs",
            severity=CheckSeverity.CRITICAL,
            passed=c2_pass,
            message="All pixel classes in {0, 1, 2, 3}" if c2_pass else f"Illegal class IDs found: {sorted(list(invalid_classes))}",
            details={"detected_classes": sorted(list(unique_classes))},
        ))

        # Pixel counts
        class_counts = {cls_id: int(np.sum(mask_2d == cls_id)) for cls_id in self.ALLOWED_CLASSES}
        class_fractions = {cls_id: round(count / max(1, total_pixels), 4) for cls_id, count in class_counts.items()}

        # -------------------------------------------------------------
        # CHECK 3: Road Coverage Ratio in ROI (WARNING)
        # -------------------------------------------------------------
        road_roi_count = int(np.sum(roi_mask == 1))
        road_roi_ratio = road_roi_count / max(1, roi_pixels)
        c3_pass = (self.road_cov_min <= road_roi_ratio <= self.road_cov_max)
        checks.append(CheckResult(
            check_id=3,
            name="Road Coverage Ratio",
            severity=CheckSeverity.WARNING,
            passed=c3_pass,
            message=f"Road coverage in ROI ({road_roi_ratio*100:.1f}%) within [{self.road_cov_min*100:.0f}%, {self.road_cov_max*100:.0f}%]" if c3_pass else f"Road coverage abnormal ({road_roi_ratio*100:.1f}%)",
            details={"road_roi_fraction": round(road_roi_ratio, 3)},
        ))

        # -------------------------------------------------------------
        # CHECK 4: Lane Marking Coverage Ratio (WARNING)
        # -------------------------------------------------------------
        left_count = int(np.sum(roi_mask == 2))
        right_count = int(np.sum(roi_mask == 3))
        left_ratio = left_count / max(1, roi_pixels)
        right_ratio = right_ratio_val = right_count / max(1, roi_pixels)

        # Fail if both lanes are 0, or if individual lane exceeds max
        c4_pass = True
        c4_msg = "Lane coverage within limits"
        if left_count == 0 and right_count == 0:
            c4_pass = False
            c4_msg = "Zero lane markings detected in ROI"
        elif left_ratio > self.lane_cov_max or right_ratio_val > self.lane_cov_max:
            c4_pass = False
            c4_msg = f"Lane marking ratio exceeds {self.lane_cov_max*100:.1f}% limit"

        checks.append(CheckResult(
            check_id=4,
            name="Lane Marking Coverage",
            severity=CheckSeverity.WARNING,
            passed=c4_pass,
            message=c4_msg,
            details={"left_lane_fraction": round(left_ratio, 4), "right_lane_fraction": round(right_ratio_val, 4)},
        ))

        # -------------------------------------------------------------
        # CHECK 5: Lane Spatial Consistency (CRITICAL)
        # -------------------------------------------------------------
        # At bottom row (y = mask_h - 1)
        bottom_row = mask_2d[mask_h - 1, :]
        left_indices = np.where(bottom_row == 2)[0]
        right_indices = np.where(bottom_row == 3)[0]
        img_center_x = mask_w / 2.0

        left_x_bottom = float(np.mean(left_indices)) if len(left_indices) > 0 else None
        right_x_bottom = float(np.mean(right_indices)) if len(right_indices) > 0 else None

        c5_pass = True
        c5_msg = "Spatial ordering consistent"
        if left_x_bottom is None or right_x_bottom is None:
            c5_pass = False
            c5_msg = "Lane boundaries absent at bottom row"
        elif left_x_bottom >= right_x_bottom:
            c5_pass = False
            c5_msg = f"Lane crossing detected at bottom: Left ({left_x_bottom:.1f}) >= Right ({right_x_bottom:.1f})"

        checks.append(CheckResult(
            check_id=5,
            name="Lane Spatial Consistency",
            severity=CheckSeverity.CRITICAL,
            passed=c5_pass,
            message=c5_msg,
            details={"left_bottom_x": left_x_bottom, "right_bottom_x": right_x_bottom, "center_x": img_center_x},
        ))

        # -------------------------------------------------------------
        # CHECK 6: Lane Width Plausibility (WARNING)
        # -------------------------------------------------------------
        c6_pass = True
        c6_msg = "Lane width plausible"
        lane_w_bottom = None
        if left_x_bottom is None or right_x_bottom is None:
            c6_pass = False
            c6_msg = "Cannot compute bottom lane width (missing boundary)"
        else:
            # Scale to 640px model width reference
            scale_w = 640.0 / mask_w
            lane_w_bottom = (right_x_bottom - left_x_bottom) * scale_w
            if not (self.lane_w_min <= lane_w_bottom <= self.lane_w_max):
                c6_pass = False
                c6_msg = f"Bottom lane separation ({lane_w_bottom:.1f}px) outside [{self.lane_w_min}, {self.lane_w_max}]px"

        checks.append(CheckResult(
            check_id=6,
            name="Lane Width Plausibility",
            severity=CheckSeverity.WARNING,
            passed=c6_pass,
            message=c6_msg,
            details={"scaled_lane_width_px": round(lane_w_bottom, 1) if lane_w_bottom else None},
        ))

        # -------------------------------------------------------------
        # CHECK 7: Lane Continuity / Fragmentation (INFO)
        # -------------------------------------------------------------
        left_binary = (mask_2d == 2).astype(np.uint8)
        right_binary = (mask_2d == 3).astype(np.uint8)

        n_left_comps, _, _, _ = cv2.connectedComponentsWithStats(left_binary, connectivity=8)
        n_right_comps, _, _, _ = cv2.connectedComponentsWithStats(right_binary, connectivity=8)

        left_comps = max(0, n_left_comps - 1)
        right_comps = max(0, n_right_comps - 1)

        c7_pass = (left_comps <= self.max_fragments and right_comps <= self.max_fragments)
        checks.append(CheckResult(
            check_id=7,
            name="Lane Continuity",
            severity=CheckSeverity.INFO,
            passed=c7_pass,
            message="Lane continuity normal" if c7_pass else f"Fragmented lane markings (Left: {left_comps}, Right: {right_comps} components)",
            details={"left_components": left_comps, "right_components": right_comps},
        ))

        # -------------------------------------------------------------
        # CHECK 8: Background Contamination in ROI (WARNING)
        # -------------------------------------------------------------
        # Check for isolated background holes enclosed within road surface
        road_binary = (mask_2d == 1).astype(np.uint8)
        c8_pass = True
        c8_msg = "No road surface voids"
        if np.sum(road_binary) > 0:
            contours, hierarchy = cv2.findContours(road_binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            if hierarchy is not None and len(contours) > 1:
                # Check for internal holes
                holes = sum(1 for h_info in hierarchy[0] if h_info[3] != -1)
                if holes > 4:
                    c8_pass = False
                    c8_msg = f"Detected {holes} background holes inside road region"

        checks.append(CheckResult(
            check_id=8,
            name="Background Contamination in ROI",
            severity=CheckSeverity.WARNING,
            passed=c8_pass,
            message=c8_msg,
        ))

        # -------------------------------------------------------------
        # CHECK 9: Sky / Horizon Leak Detection (WARNING)
        # -------------------------------------------------------------
        sky_h = int(mask_h * self.sky_top_fraction)
        sky_region = mask_2d[0:sky_h, :]
        sky_leak_count = int(np.sum(sky_region > 0))
        c9_pass = (sky_leak_count < int(sky_h * mask_w * 0.02))  # Allow < 2% noise
        checks.append(CheckResult(
            check_id=9,
            name="Sky / Horizon Leak",
            severity=CheckSeverity.WARNING,
            passed=c9_pass,
            message="No sky leak detected" if c9_pass else f"Road/lane pixels found in upper {self.sky_top_fraction*100:.0f}% sky zone ({sky_leak_count}px)",
            details={"sky_leak_pixels": sky_leak_count},
        ))

        # -------------------------------------------------------------
        # Overall Weighted Quality Score
        # -------------------------------------------------------------
        total_weight = sum(self.weights[c.severity] for c in checks)
        earned_weight = sum(self.weights[c.severity] for c in checks if c.passed)
        score = round(earned_weight / max(1.0, total_weight), 3)

        passed_all_critical = all(c.passed for c in checks if c.severity == CheckSeverity.CRITICAL)

        if score >= self.thresh_excellent and passed_all_critical:
            tier = QualityTier.EXCELLENT
        elif score >= self.thresh_acceptable and passed_all_critical:
            tier = QualityTier.ACCEPTABLE
        elif score >= self.thresh_review:
            tier = QualityTier.NEEDS_REVIEW
        else:
            tier = QualityTier.REJECTED

        summary = f"{tier.value} (Score: {score:.2f}) - {'PASSED ALL CHECKS' if all(c.passed for c in checks) else 'SOME ISSUES FLAGGED'}"

        return QualityScoreReport(
            stem=img_name,
            image_shape=(img_h, img_w),
            mask_shape=(mask_h, mask_w),
            score=score,
            tier=tier,
            passed_all_critical=passed_all_critical,
            checks=checks,
            class_pixel_counts=class_counts,
            class_pixel_fractions=class_fractions,
            summary_message=summary,
        )

    evaluate = evaluate_pair

    def _create_failure_report(self, stem: str, reason: str) -> QualityScoreReport:
        """Create a default rejected report when files are unreadable."""
        return QualityScoreReport(
            stem=stem,
            image_shape=(0, 0),
            mask_shape=(0, 0),
            score=0.0,
            tier=QualityTier.REJECTED,
            passed_all_critical=False,
            checks=[CheckResult(0, "File Decodability", CheckSeverity.CRITICAL, False, reason)],
            class_pixel_counts={},
            class_pixel_fractions={},
            summary_message=f"REJECTED: {reason}",
        )
