"""
src/dataset/quality_assurance.py
================================
Automated Quality Assurance system for semantic segmentation lane masks.

Validates:
  - Correct class IDs (0=Background, 1=Road, 2=Left Lane, 3=Right Lane)
  - Plausible class coverage ratios (lanes, drivable road, background)
  - Spatial consistency (left lane strictly left of right lane, no crossovers)
  - Lateral lane separation / width reasonableness
  - Topological integrity (fragmentation / connected component count)
  - Excessive thickness / abnormal boundary width
  - Quantitative quality scoring (0 to 100 scale)
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass
class QualityIssue:
    check_name: str
    severity: Severity
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class MaskQualityReport:
    mask_path: Optional[Path]
    is_valid: bool
    quality_score: float
    issues: list[QualityIssue]
    class_counts: dict[int, int]
    class_percentages: dict[int, float]
    lane_width_mean_px: Optional[float] = None
    lane_width_std_px: Optional[float] = None
    spatial_crossover_count: int = 0
    left_components: int = 0
    right_components: int = 0
    summary: str = ""

    @property
    def has_errors(self) -> bool:
        return any(i.severity == Severity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == Severity.WARNING for i in self.issues)


class AnnotationQA:
    """
    Automated QA evaluator for road semantic segmentation masks.
    """

    ALLOWED_CLASSES = {0, 1, 2, 3}

    # Coverage percentage bounds (min, max) relative to total image area
    COVERAGE_BOUNDS = {
        0: (0.10, 0.99),    # Background: typically 10% - 99%
        1: (0.01, 0.85),    # Road surface: typically 1% - 85%
        2: (0.0002, 0.12),  # Left lane: 0.02% - 12%
        3: (0.0002, 0.12),  # Right lane: 0.02% - 12%
    }

    def __init__(
        self,
        min_lane_sep_px: int = 30,
        max_lane_sep_px: int = 600,
        max_fragments: int = 15,
        max_lane_thickness_px: int = 60,
    ):
        self.min_lane_sep_px = min_lane_sep_px
        self.max_lane_sep_px = max_lane_sep_px
        self.max_fragments = max_fragments
        self.max_lane_thickness_px = max_lane_thickness_px

    def evaluate_mask(
        self,
        mask_input: Union[Path, str, np.ndarray],
        expected_shape: Optional[tuple[int, int]] = None,
    ) -> MaskQualityReport:
        """
        Evaluate a single mask file or 2D numpy array.
        """
        mask_path = Path(mask_input) if isinstance(mask_input, (str, Path)) else None
        issues: list[QualityIssue] = []

        if isinstance(mask_input, (str, Path)):
            p = Path(mask_input)
            if not p.exists():
                return MaskQualityReport(
                    mask_path=p,
                    is_valid=False,
                    quality_score=0.0,
                    issues=[QualityIssue("file_existence", Severity.ERROR, f"Mask file not found: {p}")],
                    class_counts={},
                    class_percentages={},
                    summary="File does not exist",
                )
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                return MaskQualityReport(
                    mask_path=p,
                    is_valid=False,
                    quality_score=0.0,
                    issues=[QualityIssue("decodability", Severity.ERROR, "Unable to decode mask image")],
                    class_counts={},
                    class_percentages={},
                    summary="Corrupted mask file",
                )
        else:
            raw = mask_input

        # Convert to 2D uint8
        if len(raw.shape) == 3:
            if np.array_equal(raw[:, :, 0], raw[:, :, 1]) and np.array_equal(raw[:, :, 1], raw[:, :, 2]):
                mask_2d = raw[:, :, 0].astype(np.uint8)
            else:
                mask_2d = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        else:
            mask_2d = raw.astype(np.uint8)

        h, w = mask_2d.shape[:2]
        total_pixels = h * w

        # Shape consistency check
        if expected_shape is not None:
            exp_h, exp_w = expected_shape[:2]
            if (h, w) != (exp_h, exp_w):
                issues.append(
                    QualityIssue(
                        "dimension_match",
                        Severity.ERROR,
                        f"Mask dimensions {w}x{h} mismatch expected {exp_w}x{exp_h}",
                    )
                )

        # 1. Class ID validation
        unique_classes = sorted(list(set(np.unique(mask_2d).tolist())))
        invalid_classes = set(unique_classes) - self.ALLOWED_CLASSES
        if invalid_classes:
            issues.append(
                QualityIssue(
                    "class_ids",
                    Severity.ERROR,
                    f"Invalid class IDs found: {sorted(list(invalid_classes))}. Allowed: {sorted(list(self.ALLOWED_CLASSES))}",
                )
            )

        # 2. Pixel counts and percentages
        class_counts: dict[int, int] = {}
        class_pcts: dict[int, float] = {}
        for cls_id in self.ALLOWED_CLASSES:
            count = int(np.sum(mask_2d == cls_id))
            class_counts[cls_id] = count
            class_pcts[cls_id] = round(count / total_pixels * 100.0, 3)

        # Empty mask check
        lane_pixels = class_counts.get(2, 0) + class_counts.get(3, 0)
        road_pixels = class_counts.get(1, 0)
        if lane_pixels == 0 and road_pixels == 0:
            issues.append(
                QualityIssue(
                    "empty_annotation",
                    Severity.WARNING,
                    "Mask contains only background (0). No lanes or road annotated.",
                )
            )

        # 3. Coverage ratio checks
        for cls_id, (min_ratio, max_ratio) in self.COVERAGE_BOUNDS.items():
            count = class_counts.get(cls_id, 0)
            if count == 0 and cls_id in {2, 3}:
                # Missing individual lane is a warning, not necessarily an error
                issues.append(
                    QualityIssue(
                        f"coverage_class_{cls_id}",
                        Severity.INFO,
                        f"Class {cls_id} not present in mask.",
                    )
                )
                continue
            ratio = count / total_pixels
            if ratio < min_ratio and count > 0:
                issues.append(
                    QualityIssue(
                        f"coverage_low_class_{cls_id}",
                        Severity.WARNING,
                        f"Class {cls_id} coverage ({ratio * 100:.3f}%) below expected minimum ({min_ratio * 100:.2f}%)",
                    )
                )
            elif ratio > max_ratio:
                issues.append(
                    QualityIssue(
                        f"coverage_high_class_{cls_id}",
                        Severity.WARNING,
                        f"Class {cls_id} coverage ({ratio * 100:.2f}%) exceeds expected maximum ({max_ratio * 100:.2f}%)",
                    )
                )

        # 4. Spatial consistency & crossover checks
        crossovers, mean_sep, std_sep = self._check_spatial_consistency(mask_2d, issues)

        # 5. Topological & morphological checks
        left_comps, right_comps = self._check_morphology(mask_2d, issues)

        # 6. Overall Quality Score calculation (0 to 100)
        score = self._compute_quality_score(issues, lane_pixels, road_pixels)

        is_valid = not any(i.severity == Severity.ERROR for i in issues)
        summary = "PASS" if is_valid and score >= 75.0 else ("WARNING" if is_valid else "FAIL")

        return MaskQualityReport(
            mask_path=mask_path,
            is_valid=is_valid,
            quality_score=round(score, 1),
            issues=issues,
            class_counts=class_counts,
            class_percentages=class_pcts,
            lane_width_mean_px=mean_sep,
            lane_width_std_px=std_sep,
            spatial_crossover_count=crossovers,
            left_components=left_comps,
            right_components=right_comps,
            summary=summary,
        )

    def _check_spatial_consistency(
        self,
        mask: np.ndarray,
        issues: list[QualityIssue],
    ) -> tuple[int, Optional[float], Optional[float]]:
        """
        Check that left lane (2) is strictly left of right lane (3) at matching scanlines.
        Computes lane separation statistics.
        """
        h, w = mask.shape[:2]
        crossovers = 0
        separations = []

        left_pts = np.where(mask == 2)
        right_pts = np.where(mask == 3)

        if len(left_pts[0]) == 0 or len(right_pts[0]) == 0:
            return 0, None, None

        # Build scanline medians for left and right
        left_by_y: dict[int, list[int]] = {}
        for y, x in zip(left_pts[0], left_pts[1]):
            left_by_y.setdefault(y, []).append(x)

        right_by_y: dict[int, list[int]] = {}
        for y, x in zip(right_pts[0], right_pts[1]):
            right_by_y.setdefault(y, []).append(x)

        common_ys = sorted(list(set(left_by_y.keys()) & set(right_by_y.keys())))

        for y in common_ys:
            x_left = float(np.median(left_by_y[y]))
            x_right = float(np.median(right_by_y[y]))

            if x_left >= x_right:
                crossovers += 1
            else:
                sep = x_right - x_left
                separations.append(sep)

        if crossovers > 0:
            issues.append(
                QualityIssue(
                    "lane_crossover",
                    Severity.ERROR if crossovers > 5 else Severity.WARNING,
                    f"Detected {crossovers} scanlines where Left Lane is right of or overlapping Right Lane.",
                    {"crossover_scanlines": crossovers},
                )
            )

        mean_sep = None
        std_sep = None
        if separations:
            mean_sep = float(np.mean(separations))
            std_sep = float(np.std(separations))

            if mean_sep < self.min_lane_sep_px:
                issues.append(
                    QualityIssue(
                        "lane_too_narrow",
                        Severity.WARNING,
                        f"Mean lane width ({mean_sep:.1f}px) is unusually narrow (< {self.min_lane_sep_px}px)",
                    )
                )
            elif mean_sep > self.max_lane_sep_px:
                issues.append(
                    QualityIssue(
                        "lane_too_wide",
                        Severity.WARNING,
                        f"Mean lane width ({mean_sep:.1f}px) is unusually wide (> {self.max_lane_sep_px}px)",
                    )
                )

        return crossovers, (round(mean_sep, 1) if mean_sep is not None else None), (round(std_sep, 1) if std_sep is not None else None)

    def _check_morphology(
        self,
        mask: np.ndarray,
        issues: list[QualityIssue],
    ) -> tuple[int, int]:
        """
        Check connectivity and thickness of lane markings.
        """
        left_mask = (mask == 2).astype(np.uint8)
        right_mask = (mask == 3).astype(np.uint8)

        n_left, _, _, _ = cv2.connectedComponentsWithStats(left_mask, connectivity=8)
        n_right, _, _, _ = cv2.connectedComponentsWithStats(right_mask, connectivity=8)

        # Subtract background component
        left_comps = max(0, n_left - 1)
        right_comps = max(0, n_right - 1)

        if left_comps > self.max_fragments:
            issues.append(
                QualityIssue(
                    "left_lane_fragmentation",
                    Severity.WARNING,
                    f"Left lane has high fragmentation ({left_comps} components, threshold: {self.max_fragments})",
                )
            )

        if right_comps > self.max_fragments:
            issues.append(
                QualityIssue(
                    "right_lane_fragmentation",
                    Severity.WARNING,
                    f"Right lane has high fragmentation ({right_comps} components, threshold: {self.max_fragments})",
                )
            )

        # Check maximum thickness along horizontal scanlines
        for cls_id, binary_mask, name in [(2, left_mask, "Left"), (3, right_mask, "Right")]:
            row_sums = np.sum(binary_mask, axis=1)
            max_thickness = int(np.max(row_sums)) if len(row_sums) > 0 else 0
            if max_thickness > self.max_lane_thickness_px:
                issues.append(
                    QualityIssue(
                        f"{name.lower()}_lane_thickness",
                        Severity.WARNING,
                        f"{name} lane marking max horizontal thickness ({max_thickness}px) exceeds threshold ({self.max_lane_thickness_px}px)",
                    )
                )

        return left_comps, right_comps

    def _compute_quality_score(
        self,
        issues: list[QualityIssue],
        lane_pixels: int,
        road_pixels: int,
    ) -> float:
        """
        Calculates a 0-100 quality score penalizing errors and warnings.
        """
        score = 100.0

        for issue in issues:
            if issue.severity == Severity.ERROR:
                score -= 30.0
            elif issue.severity == Severity.WARNING:
                score -= 10.0
            elif issue.severity == Severity.INFO:
                score -= 2.0

        if lane_pixels == 0 and road_pixels == 0:
            score -= 25.0

        return max(0.0, min(100.0, score))

    def evaluate_batch(
        self,
        mask_inputs: list[Union[Path, str, np.ndarray]],
    ) -> list[MaskQualityReport]:
        """Evaluate a batch of masks."""
        return [self.evaluate_mask(m) for m in mask_inputs]
