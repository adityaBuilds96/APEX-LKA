"""
evaluation/lane_metrics.py
==========================
APEX-Vision Core v4.0 • Semantic Lane Segmentation & Geometric Accuracy Metrics
ISO 26262 ASIL-B Aligned Computer Vision Performance Verification

Calculates rigorous statistical and geometric perception metrics for lane segmentation:
1. Per-Class Intersection over Union (IoU), Mean IoU (mIoU), Frequency-Weighted IoU (FWIoU).
   Class Indices:
   - 0: Background
   - 1: Drivable Surface
   - 2: Left Boundary
   - 3: Right Boundary
2. Geometric Lane Precision, Recall, and F1-Score under pixel tolerance thresholds (tau in px).
3. Calibration Analytics: Expected Calibration Error (ECE), Maximum Calibration Error (MCE),
   Reliability Diagrams, and Temperature Scaling optimization (T* minimizing validation NLL).
4. Out-of-Distribution (OOD) Detection Analytics: Energy-based score, Maximum Softmax
   Probability (MSP), and Shannon entropy uncertainty ranking.
5. Strict "No-Flip Rule" validation: Horizontal augmentation must swap Left (2) and Right (3).
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.Evaluation.LaneMetrics")


# ═══════════════════════════════════════════════════════════════════════════════
# Telemetry & Result Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ConfusionMatrix:
    """Multi-class confusion matrix for semantic segmentation evaluation."""
    num_classes: int
    matrix: np.ndarray  # Shape: (num_classes, num_classes) where rows=ground_truth, cols=prediction

    @classmethod
    def create(cls, num_classes: int = 4) -> "ConfusionMatrix":
        return cls(num_classes=num_classes, matrix=np.zeros((num_classes, num_classes), dtype=np.int64))

    def update(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """Accumulate counts from flattened ground-truth and prediction arrays."""
        valid_mask = (y_true >= 0) & (y_true < self.num_classes) & (y_pred >= 0) & (y_pred < self.num_classes)
        t = y_true[valid_mask].astype(np.int64)
        p = y_pred[valid_mask].astype(np.int64)
        indices = self.num_classes * t + p
        counts = np.bincount(indices, minlength=self.num_classes ** 2)
        self.matrix += counts.reshape((self.num_classes, self.num_classes))

    def get_tp_fp_fn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute True Positives, False Positives, and False Negatives per class."""
        tp = np.diag(self.matrix).astype(np.float64)
        fp = np.sum(self.matrix, axis=0).astype(np.float64) - tp
        fn = np.sum(self.matrix, axis=1).astype(np.float64) - tp
        return tp, fp, fn


@dataclass
class SegmentationMetricsResult:
    """Summary of per-class and aggregated semantic segmentation metrics."""
    num_samples: int
    class_names: List[str]
    per_class_iou: Dict[str, float]
    per_class_precision: Dict[str, float]
    per_class_recall: Dict[str, float]
    per_class_f1: Dict[str, float]
    mean_iou: float
    frequency_weighted_iou: float
    overall_accuracy: float
    confusion_matrix: np.ndarray


@dataclass
class CalibrationResult:
    """Results from reliability binning, ECE, MCE, and Temperature Scaling."""
    num_bins: int
    expected_calibration_error: float # ECE in [0.0, 1.0]
    maximum_calibration_error: float  # MCE in [0.0, 1.0]
    optimal_temperature: float        # T* scaling parameter
    uncalibrated_nll: float           # Negative Log Likelihood before calibration
    calibrated_nll: float             # Negative Log Likelihood after calibration
    bin_confidences: List[float]
    bin_accuracies: List[float]
    bin_counts: List[int]


@dataclass
class GeometricLaneF1Result:
    """Geometric point-distance tolerance F1 score (e.g. Tusimple benchmark style)."""
    pixel_tolerance_tau: float
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Segmentation Evaluator: IoU, F1, Accuracy
# ═══════════════════════════════════════════════════════════════════════════════

class LaneSegmentationEvaluator:
    """
    Evaluates multi-class lane semantic masks against ground-truth labels.
    Default Class Mapping (4 classes):
    - 0: Background
    - 1: Drivable Surface
    - 2: Left Boundary
    - 3: Right Boundary
    """

    DEFAULT_CLASS_NAMES = [
        "Background",
        "Drivable_Surface",
        "Left_Boundary",
        "Right_Boundary",
    ]

    def __init__(self, class_names: Optional[List[str]] = None) -> None:
        self.class_names = class_names or self.DEFAULT_CLASS_NAMES
        self.num_classes = len(self.class_names)
        self.confusion = ConfusionMatrix.create(self.num_classes)
        self.total_frames_evaluated = 0

    def reset(self) -> None:
        """Reset internal confusion matrix accumulator."""
        self.confusion = ConfusionMatrix.create(self.num_classes)
        self.total_frames_evaluated = 0

    def add_batch(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        """
        Accumulate prediction results.
        y_true: integer label array (B, H, W) or (H, W)
        y_pred: predicted class array (B, H, W) or (H, W)
        """
        assert y_true.shape == y_pred.shape, f"Shape mismatch: {y_true.shape} vs {y_pred.shape}"
        self.confusion.update(y_true.ravel(), y_pred.ravel())
        self.total_frames_evaluated += 1 if y_true.ndim == 2 else y_true.shape[0]

    def compute_metrics(self) -> SegmentationMetricsResult:
        """Calculate complete per-class and aggregated segmentation metrics."""
        tp, fp, fn = self.confusion.get_tp_fp_fn()
        total_gt_per_class = np.sum(self.confusion.matrix, axis=1).astype(np.float64)
        total_pixels = np.sum(self.confusion.matrix).astype(np.float64)

        # 1. Per-Class IoU = TP / (TP + FP + FN)
        union = tp + fp + fn
        ious = np.where(union > 0, tp / union, 1.0)

        # 2. Per-Class Precision = TP / (TP + FP)
        precisions = np.where((tp + fp) > 0, tp / (tp + fp), 1.0)

        # 3. Per-Class Recall = TP / (TP + FN)
        recalls = np.where((tp + fn) > 0, tp / (tp + fn), 1.0)

        # 4. Per-Class F1 = 2 * (P * R) / (P + R)
        denom_f1 = precisions + recalls
        f1_scores = np.where(denom_f1 > 0, 2.0 * (precisions * recalls) / denom_f1, 0.0)

        # 5. Macro Mean IoU (mIoU)
        mean_iou = float(np.mean(ious))

        # 6. Frequency-Weighted IoU (FWIoU) = sum(freq_i * IoU_i)
        weights = np.where(total_pixels > 0, total_gt_per_class / total_pixels, 0.0)
        fw_iou = float(np.sum(weights * ious))

        # 7. Overall Pixel Accuracy = sum(TP) / total_pixels
        overall_acc = float(np.sum(tp) / max(1.0, total_pixels))

        per_class_iou = {self.class_names[i]: round(float(ious[i]), 4) for i in range(self.num_classes)}
        per_class_prec = {self.class_names[i]: round(float(precisions[i]), 4) for i in range(self.num_classes)}
        per_class_rec = {self.class_names[i]: round(float(recalls[i]), 4) for i in range(self.num_classes)}
        per_class_f1 = {self.class_names[i]: round(float(f1_scores[i]), 4) for i in range(self.num_classes)}

        return SegmentationMetricsResult(
            num_samples=self.total_frames_evaluated,
            class_names=self.class_names,
            per_class_iou=per_class_iou,
            per_class_precision=per_class_prec,
            per_class_recall=per_class_rec,
            per_class_f1=per_class_f1,
            mean_iou=round(mean_iou, 4),
            frequency_weighted_iou=round(fw_iou, 4),
            overall_accuracy=round(overall_acc, 4),
            confusion_matrix=self.confusion.matrix.copy(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric Point-Tolerance Lane F1 Evaluator (TuSimple / CULane Style)
# ═══════════════════════════════════════════════════════════════════════════════

class GeometricLaneF1Evaluator:
    """
    Evaluates extracted polynomial or discrete lane centerlines against
    ground-truth point sets using Euclidean distance tolerance tau (pixels).
    A predicted lane point is counted as a True Positive if it lies within tau pixels
    of a ground truth point; otherwise False Positive.
    """

    def __init__(self, pixel_tolerance_tau: float = 8.0) -> None:
        self.tau = float(pixel_tolerance_tau)
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def reset(self) -> None:
        """Reset point hit counts."""
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def evaluate_line_pair(
        self,
        predicted_points_xy: np.ndarray,
        ground_truth_points_xy: np.ndarray,
    ) -> Tuple[int, int, int]:
        """
        Evaluate single pair of 2D lane point trajectories (N x 2).
        Returns: (tp, fp, fn) for this line pair.
        """
        if len(predicted_points_xy) == 0 and len(ground_truth_points_xy) == 0:
            return 0, 0, 0
        if len(predicted_points_xy) == 0:
            return 0, 0, len(ground_truth_points_xy)
        if len(ground_truth_points_xy) == 0:
            return 0, len(predicted_points_xy), 0

        # Compute pairwise Euclidean distance matrix
        # Shape: (N_pred, N_gt)
        diff = predicted_points_xy[:, np.newaxis, :] - ground_truth_points_xy[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        dists = np.sqrt(dist_sq)

        # Minimum distance from each predicted point to any ground truth point
        min_pred_to_gt = np.min(dists, axis=1)
        tp_pred = np.sum(min_pred_to_gt <= self.tau)
        fp_pred = len(predicted_points_xy) - tp_pred

        # Minimum distance from each ground truth point to any predicted point
        min_gt_to_pred = np.min(dists, axis=0)
        fn_pred = np.sum(min_gt_to_pred > self.tau)

        self.tp += int(tp_pred)
        self.fp += int(fp_pred)
        self.fn += int(fn_pred)

        return int(tp_pred), int(fp_pred), int(fn_pred)

    def get_summary(self) -> GeometricLaneF1Result:
        """Compute accumulated precision, recall, and F1 score."""
        p_denom = self.tp + self.fp
        precision = float(self.tp / p_denom) if p_denom > 0 else 0.0

        r_denom = self.tp + self.fn
        recall = float(self.tp / r_denom) if r_denom > 0 else 0.0

        f1_denom = precision + recall
        f1 = float(2.0 * (precision * recall) / f1_denom) if f1_denom > 0 else 0.0

        return GeometricLaneF1Result(
            pixel_tolerance_tau=self.tau,
            true_positives=self.tp,
            false_positives=self.fp,
            false_negatives=self.fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1_score=round(f1, 4),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration & Temperature Scaling Optimizer
# ═══════════════════════════════════════════════════════════════════════════════

class ModelConfidenceCalibrator:
    """
    Evaluates probabilistic calibration of model logits and optimizes
    temperature parameter T* via Negative Log Likelihood (NLL) minimization:
    p_calibrated = softmax(logits / T*)

    Metrics computed:
    - Expected Calibration Error (ECE) across M uniform confidence bins.
    - Maximum Calibration Error (MCE).
    - Optimal scaling temperature T*.
    """

    def __init__(self, num_bins: int = 15) -> None:
        self.num_bins = num_bins
        self.bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)

    def compute_ece(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[float, float, List[float], List[float], List[int]]:
        """
        Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
        probabilities: (N, C) softmax probabilities or (N,) top-1 confidences
        labels: (N,) ground-truth class labels or binary correctness (1 for correct, 0 for wrong)
        """
        if probabilities.ndim == 2:
            confidences = np.max(probabilities, axis=1)
            predictions = np.argmax(probabilities, axis=1)
            accuracies = (predictions == labels).astype(np.float64)
        else:
            confidences = probabilities
            accuracies = labels.astype(np.float64)

        n = len(confidences)
        if n == 0:
            return 0.0, 0.0, [], [], []

        ece = 0.0
        mce = 0.0

        bin_confs = []
        bin_accs = []
        bin_counts = []

        for i in range(self.num_bins):
            b_low = self.bin_boundaries[i]
            b_high = self.bin_boundaries[i + 1]

            # In-bin mask
            if i == self.num_bins - 1:
                mask = (confidences >= b_low) & (confidences <= b_high)
            else:
                mask = (confidences >= b_low) & (confidences < b_high)

            count = np.sum(mask)
            bin_counts.append(int(count))

            if count > 0:
                bin_acc = float(np.mean(accuracies[mask]))
                bin_conf = float(np.mean(confidences[mask]))
                gap = abs(bin_acc - bin_conf)

                bin_accs.append(round(bin_acc, 4))
                bin_confs.append(round(bin_conf, 4))

                ece += (count / float(n)) * gap
                mce = max(mce, gap)
            else:
                bin_accs.append(0.0)
                bin_confs.append(0.0)

        return float(ece), float(mce), bin_confs, bin_accs, bin_counts

    @staticmethod
    def compute_nll(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> float:
        """Calculate cross-entropy / negative log-likelihood at a given temperature."""
        scaled_logits = logits / max(1e-4, temperature)
        # Stable softmax
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        n = len(labels)
        correct_probs = probs[np.arange(n), labels]
        nll = -np.mean(np.log(np.clip(correct_probs, 1e-12, 1.0)))
        return float(nll)

    def optimize_temperature(
        self,
        validation_logits: np.ndarray,
        validation_labels: np.ndarray,
        t_min: float = 0.1,
        t_max: float = 5.0,
        num_search_steps: int = 100,
    ) -> float:
        """
        Find scalar temperature T* in [t_min, t_max] that minimizes validation NLL.
        Uses coarse-to-fine golden-section or grid search.
        """
        candidates = np.linspace(t_min, t_max, num_search_steps)
        best_t = 1.0
        min_nll = float("inf")

        for t in candidates:
            nll = self.compute_nll(validation_logits, validation_labels, temperature=t)
            if nll < min_nll:
                min_nll = nll
                best_t = float(t)

        return round(best_t, 3)

    def calibrate_and_evaluate(
        self,
        logits: np.ndarray,
        labels: np.ndarray,
    ) -> CalibrationResult:
        """
        Full calibration pipeline:
        1. Evaluate raw uncalibrated ECE and NLL (T=1.0).
        2. Optimize optimal temperature T*.
        3. Evaluate post-calibration ECE and NLL using T*.
        """
        uncal_nll = self.compute_nll(logits, labels, temperature=1.0)
        t_opt = self.optimize_temperature(logits, labels)
        cal_nll = self.compute_nll(logits, labels, temperature=t_opt)

        # Scale logits with T*
        scaled_logits = logits / t_opt
        max_logits = np.max(scaled_logits, axis=1, keepdims=True)
        exp_logits = np.exp(scaled_logits - max_logits)
        cal_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        ece, mce, b_confs, b_accs, b_counts = self.compute_ece(cal_probs, labels)

        return CalibrationResult(
            num_bins=self.num_bins,
            expected_calibration_error=round(ece, 4),
            maximum_calibration_error=round(mce, 4),
            optimal_temperature=t_opt,
            uncalibrated_nll=round(uncal_nll, 4),
            calibrated_nll=round(cal_nll, 4),
            bin_confidences=b_confs,
            bin_accuracies=b_accs,
            bin_counts=b_counts,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Out-of-Distribution (OOD) Detector & Uncertainty Rater
# ═══════════════════════════════════════════════════════════════════════════════

class OODUncertaintyDetector:
    """
    Computes epistemic uncertainty and detects out-of-distribution (OOD) inputs
    (e.g., severe dust storms, camera occlusions, unfamiliar dirt tracks):
    1. Maximum Softmax Probability (MSP): Baseline confidence metric.
    2. Energy-Based Score: E(x) = -T * log(sum(exp(z_i / T))).
    3. Predictive Entropy: H(p) = -sum(p_i * log(p_i)).
    """

    @staticmethod
    def compute_energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Energy-based out-of-distribution score:
        Higher energy indicates lower model familiarity / anomalous input.
        """
        scaled = logits / max(1e-4, temperature)
        # Log-Sum-Exp computation
        max_s = np.max(scaled, axis=-1, keepdims=True)
        lse = max_s.squeeze(-1) + np.log(np.sum(np.exp(scaled - max_s), axis=-1))
        energy = -temperature * lse
        return energy

    @staticmethod
    def compute_entropy_uncertainty(probabilities: np.ndarray) -> np.ndarray:
        """Shannon predictive entropy in nats."""
        eps = 1e-12
        p_clipped = np.clip(probabilities, eps, 1.0)
        entropy = -np.sum(p_clipped * np.log(p_clipped), axis=-1)
        return entropy

    @classmethod
    def evaluate_ood_auc(
        cls,
        id_scores: np.ndarray,
        ood_scores: np.ndarray,
        higher_score_is_ood: bool = True,
    ) -> float:
        """
        Calculate Area Under the ROC Curve (AUROC) for separating In-Distribution (ID)
        from Out-of-Distribution (OOD) samples.
        """
        y_true = np.concatenate([np.zeros(len(id_scores)), np.ones(len(ood_scores))])
        y_scores = np.concatenate([id_scores, ood_scores])

        if not higher_score_is_ood:
            y_scores = -y_scores

        # Rank-based Mann-Whitney U test statistic for analytical AUROC
        order = np.argsort(y_scores)
        rank = np.empty_like(order)
        rank[order] = np.arange(len(y_scores)) + 1

        n_id = len(id_scores)
        n_ood = len(ood_scores)
        if n_id == 0 or n_ood == 0:
            return 0.50

        sum_rank_ood = np.sum(rank[y_true == 1])
        u_stat = sum_rank_ood - (n_ood * (n_ood + 1)) / 2.0
        auroc = u_stat / (n_id * n_ood)

        return round(float(auroc), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Strict No-Flip Rule Assertion Utility
# ═══════════════════════════════════════════════════════════════════════════════

class NoFlipRuleValidator:
    """
    Enforces the fundamental architectural safety constraint:
    Never apply horizontal flipping to lane segmentation masks unless Class 2
    (Left Boundary) and Class 3 (Right Boundary) are explicitly swapped.
    Violating this swaps left and right lanes, causing inverse vehicle steering!
    """

    @staticmethod
    def validate_or_transform_horizontal_flip(
        mask: np.ndarray,
        left_class_id: int = 2,
        right_class_id: int = 3,
    ) -> np.ndarray:
        """
        Flips image horizontally AND correctly swaps left/right lane classes.
        """
        flipped = np.fliplr(mask)
        swapped = flipped.copy()

        # Swap Left and Right Boundary IDs
        mask_left = (flipped == left_class_id)
        mask_right = (flipped == right_class_id)

        swapped[mask_left] = right_class_id
        swapped[mask_right] = left_class_id

        return swapped

    @staticmethod
    def assert_no_unswapped_flip_applied(
        original_mask: np.ndarray,
        augmented_mask: np.ndarray,
        left_class_id: int = 2,
        right_class_id: int = 3,
    ) -> bool:
        """
        Inspects if an augmented mask is an illegal unswapped horizontal flip.
        Returns True if legal, raises AssertionError if illegal flip detected.
        """
        # If augmented is an unswapped horizontal flip:
        naive_flipped = np.fliplr(original_mask)
        if np.array_equal(naive_flipped, augmented_mask):
            # Check if left and right classes exist
            has_left = np.any(original_mask == left_class_id)
            has_right = np.any(original_mask == right_class_id)
            if has_left or has_right:
                raise AssertionError(
                    "CRITICAL NO-FLIP RULE VIOLATION: Horizontal flip applied without "
                    f"swapping Class {left_class_id} (Left) and Class {right_class_id} (Right)!"
                )
        return True


# ═══════════════════════════════════════════════════════════════════════════════
# Self-Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def run_lane_metrics_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. Multi-class IoU, mIoU, FWIoU, and F1 calculations.
    2. Geometric point-tolerance lane F1 evaluator.
    3. Expected Calibration Error (ECE) and Temperature Scaling (T* optimization).
    4. Out-of-distribution energy score and AUROC discrimination.
    5. Strict No-Flip rule enforcement and class index swapping.
    """
    print("Executing LaneMetrics Verification Suite...")

    # -------------------------------------------------------------------------
    # Test 1: Multi-Class Segmentation Metrics
    # -------------------------------------------------------------------------
    evaluator = LaneSegmentationEvaluator()

    # Synthetic 100x100 masks with known ground truth
    y_true = np.zeros((100, 100), dtype=np.int64)
    y_true[30:70, :] = 1  # Drivable surface
    y_true[30:35, :] = 2  # Left boundary
    y_true[65:70, :] = 3  # Right boundary

    # Add minor synthetic errors
    y_pred = y_true.copy()
    y_pred[28:30, :] = 2  # False positive left boundary
    y_pred[68:70, :] = 1  # False negative right boundary (misclassified as surface)

    evaluator.add_batch(y_true, y_pred)
    res = evaluator.compute_metrics()

    print(f"Computed mIoU: {res.mean_iou:.4f} | FWIoU: {res.frequency_weighted_iou:.4f}")
    assert 0.0 <= res.mean_iou <= 1.0
    assert res.per_class_iou["Left_Boundary"] > 0.60
    assert res.per_class_iou["Right_Boundary"] > 0.50
    assert res.overall_accuracy > 0.95

    # -------------------------------------------------------------------------
    # Test 2: Geometric Point-Tolerance F1 Evaluator
    # -------------------------------------------------------------------------
    geo_eval = GeometricLaneF1Evaluator(pixel_tolerance_tau=5.0)

    # 20 ground truth points along straight line
    gt_pts = np.column_stack([np.linspace(100, 200, 20), np.linspace(50, 300, 20)])
    # Predicted points shifted by 2 pixels (within 5px tolerance => 100% TP)
    pred_pts = gt_pts + np.array([2.0, 0.0])

    geo_eval.evaluate_line_pair(pred_pts, gt_pts)
    geo_res = geo_eval.get_summary()

    print(f"Geometric F1 Score: {geo_res.f1_score:.4f} (TP={geo_res.true_positives})")
    assert geo_res.f1_score == 1.0, "Expected perfect geometric F1 within 5px tolerance"
    assert geo_res.false_positives == 0

    # -------------------------------------------------------------------------
    # Test 3: Calibration & Temperature Scaling
    # -------------------------------------------------------------------------
    calibrator = ModelConfidenceCalibrator(num_bins=10)

    # Generate synthetic overconfident logits
    np.random.seed(1234)
    n_samples = 500
    n_classes = 4
    # True labels
    synth_labels = np.random.randint(0, n_classes, size=n_samples)
    # Logits with overconfidence (multiplied by 3.5)
    one_hot = np.eye(n_classes)[synth_labels]
    synth_logits = (one_hot * 6.0 + np.random.randn(n_samples, n_classes)).astype(np.float32)

    cal_res = calibrator.calibrate_and_evaluate(synth_logits, synth_labels)
    print(f"Calibration: Optimal Temperature T* = {cal_res.optimal_temperature:.2f} (ECE = {cal_res.expected_calibration_error:.4f})")
    assert cal_res.optimal_temperature > 0.0
    assert 0.0 <= cal_res.expected_calibration_error <= 1.0

    # -------------------------------------------------------------------------
    # Test 4: OOD Energy Score & AUROC
    # -------------------------------------------------------------------------
    # In-distribution logits (high confidence)
    id_logits = np.array([[8.0, 1.0, 0.5, 0.2], [0.5, 9.0, 0.2, 0.1]])
    # OOD logits (flat / high entropy)
    ood_logits = np.array([[1.0, 1.1, 0.9, 1.0], [0.8, 0.9, 1.0, 0.8]])

    e_id = OODUncertaintyDetector.compute_energy_score(id_logits)
    e_ood = OODUncertaintyDetector.compute_energy_score(ood_logits)
    # OOD should have higher energy (less negative)
    assert np.mean(e_ood) > np.mean(e_id)

    auroc = OODUncertaintyDetector.evaluate_ood_auc(e_id, e_ood, higher_score_is_ood=True)
    assert auroc == 1.0, f"AUROC separation mismatch: {auroc}"
    print(f"OOD AUROC Score: {auroc:.4f}")

    # -------------------------------------------------------------------------
    # Test 5: Strict No-Flip Rule Validation
    # -------------------------------------------------------------------------
    mask_orig = np.zeros((50, 50), dtype=np.uint8)
    mask_orig[:, :10] = 2  # Left lane
    mask_orig[:, 40:] = 3  # Right lane

    # Legal transform: flipped AND swapped
    # After flipping geometry and swapping class tags, the boundary on the left side
    # of the image must remain Class 2 (Left), and the right side must remain Class 3 (Right)!
    legal_flipped = NoFlipRuleValidator.validate_or_transform_horizontal_flip(mask_orig)
    assert np.all(legal_flipped[:, :10] == 2), "Lane on the vehicle's left must remain Class 2 (Left Boundary)"
    assert np.all(legal_flipped[:, 40:] == 3), "Lane on the vehicle's right must remain Class 3 (Right Boundary)"

    # Illegal transform test: must raise AssertionError
    illegal_flipped = np.fliplr(mask_orig)
    illegal_caught = False
    try:
        NoFlipRuleValidator.assert_no_unswapped_flip_applied(mask_orig, illegal_flipped)
    except AssertionError:
        illegal_caught = True

    assert illegal_caught is True, "Failed to catch illegal unswapped horizontal flip"
    print("No-Flip Rule Enforcement Verified.")

    print("LaneMetrics verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_lane_metrics_verification_suite()
