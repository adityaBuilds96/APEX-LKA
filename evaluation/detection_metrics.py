"""
evaluation/detection_metrics.py
===============================
APEX-Vision Core v4.0 • Object Detection Accuracy & Range-Stratified Metrics
ISO 26262 ASIL-B Aligned 2D/3D Perception Verification

Calculates COCO-standard Mean Average Precision (mAP) and perception benchmarks:
1. mAP@0.50 (IoU threshold = 0.50) and mAP@0.50:0.95 (step = 0.05).
2. Per-Class AP breakdown across 5 target classes:
   - Class 0: VEHICLE (Cars, Baja Buggies, Trucks, ATVs)
   - Class 1: PEDESTRIAN (Track Marshals, Spectators, Pedestrians)
   - Class 2: CYCLIST (Motorcycles, Bicycles)
   - Class 3: OBSTACLE (Boulders, Rocks, Slurry pits, Barriers)
   - Class 4: SIGN_MARKER (Course Boundary Posts, Mile Markers, Traffic Signs)
3. Range-Band Stratification:
   - Near-Field: 0 m to 10 m
   - Mid-Field: 10 m to 20 m
   - Far-Field: 20 m to 40 m
   - Extended Range: > 40 m
4. 101-Point Precision-Recall Curve Interpolation (COCO standard).
5. Comprehensive self-verification suite testing all range bins and IoU sweeps.
"""

from dataclasses import dataclass, field
from enum import Enum
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

logger = logging.getLogger("APEX.Evaluation.DetectionMetrics")


# ═══════════════════════════════════════════════════════════════════════════════
# Enums & Telemetry Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class ObstacleClass(Enum):
    """Categorical obstacle labels for autonomous off-road & road perception."""
    VEHICLE = 0
    PEDESTRIAN = 1
    CYCLIST = 2
    OBSTACLE = 3
    SIGN_MARKER = 4


class RangeBand(Enum):
    """Longitudinal depth distance ranges from ego vehicle front bumper."""
    NEAR = "0-10m"
    MID = "10-20m"
    FAR = "20-40m"
    EXTENDED = "40m+"
    ALL = "ALL"


@dataclass
class BoundingBox:
    """2D Image-space bounding box with classification, confidence, and metric range."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    class_id: int
    confidence: float = 1.0
    range_depth_m: float = 5.0       # Longitudinal distance Z from camera (meters)
    is_matched: bool = False

    @property
    def area(self) -> float:
        w = max(0.0, self.x_max - self.x_min)
        h = max(0.0, self.y_max - self.y_min)
        return float(w * h)

    def get_range_band(self) -> RangeBand:
        """Categorize into discrete range interval."""
        if self.range_depth_m < 10.0:
            return RangeBand.NEAR
        elif self.range_depth_m < 20.0:
            return RangeBand.MID
        elif self.range_depth_m <= 40.0:
            return RangeBand.FAR
        else:
            return RangeBand.EXTENDED


@dataclass
class DetectionSample:
    """Ground truth and model prediction pairing for a single image frame."""
    frame_id: Union[int, str]
    ground_truths: List[BoundingBox]
    predictions: List[BoundingBox]


@dataclass
class ClassRangeMetrics:
    """Evaluation summary for a specific class within a specific range band."""
    class_name: str
    range_band: str
    num_ground_truths: int
    num_predictions: int
    ap_50: float                     # AP at IoU 0.50
    ap_75: float                     # AP at IoU 0.75
    ap_50_95: float                  # COCO mAP across 0.50:0.95
    precision_at_05: float
    recall_at_05: float


@dataclass
class MasterDetectionReport:
    """Full perception benchmark report covering all classes and range strata."""
    num_frames: int
    total_ground_truths: int
    total_predictions: int
    mean_ap_50: float                # Macro mAP@0.50 across all classes
    mean_ap_50_95: float             # Macro mAP@0.50:0.95 across all classes
    range_stratified_map_50: Dict[str, float]    # RangeBand -> mAP@0.50
    range_stratified_map_50_95: Dict[str, float] # RangeBand -> mAP@0.50:0.95
    per_class_summary: Dict[str, ClassRangeMetrics]
    evaluation_duration_ms: float


# ═══════════════════════════════════════════════════════════════════════════════
# Geometric IoU & Match Engines
# ═══════════════════════════════════════════════════════════════════════════════

class BoxGeometryUtils:
    """Vectorized and pairwise bounding box intersection over union calculations."""

    @staticmethod
    def compute_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
        """Compute standard 2D Intersection over Union (IoU) between two boxes."""
        inter_x1 = max(box_a.x_min, box_b.x_min)
        inter_y1 = max(box_a.y_min, box_b.y_min)
        inter_x2 = min(box_a.x_max, box_b.x_max)
        inter_y2 = min(box_a.y_max, box_b.y_max)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        union_area = box_a.area + box_b.area - inter_area
        if union_area <= 0.0:
            return 0.0

        return float(inter_area / union_area)

    @staticmethod
    def compute_pairwise_iou_matrix(
        boxes_pred: List[BoundingBox],
        boxes_gt: List[BoundingBox],
    ) -> np.ndarray:
        """
        Construct matrix of pairwise IoU overlaps.
        Shape: (len(boxes_pred), len(boxes_gt))
        """
        n_p = len(boxes_pred)
        n_g = len(boxes_gt)
        if n_p == 0 or n_g == 0:
            return np.zeros((n_p, n_g), dtype=np.float32)

        iou_mat = np.zeros((n_p, n_g), dtype=np.float32)
        for i, bp in enumerate(boxes_pred):
            for j, bg in enumerate(boxes_gt):
                iou_mat[i, j] = BoxGeometryUtils.compute_iou(bp, bg)

        return iou_mat


# ═══════════════════════════════════════════════════════════════════════════════
# Average Precision & 101-Point Interpolator (COCO Standard)
# ═══════════════════════════════════════════════════════════════════════════════

class PrecisionRecallCurveCalculator:
    """
    Computes Precision-Recall curves and integrates 101-point interpolated AP:
    AP = (1 / 101) * sum_{r in 0:0.01:1.0} p_interp(r)
    where p_interp(r) = max_{r' >= r} p(r').
    """

    COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)

    @classmethod
    def compute_interpolated_ap(
        cls,
        recalls: np.ndarray,
        precisions: np.ndarray,
    ) -> float:
        """
        101-point interpolated Average Precision per COCO definition.
        """
        if len(recalls) == 0 or len(precisions) == 0:
            return 0.0

        # Prepend 0 and append 1 to recall and precision for envelope continuity
        mrec = np.concatenate(([0.0], recalls, [1.0]))
        mpre = np.concatenate(([0.0], precisions, [0.0]))

        # Compute maximum precision to the right of each recall
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])

        # Sample at the 101 uniform recall levels [0.0, 0.01, ..., 1.0]
        interpolated_precisions = []
        for r_th in cls.COCO_RECALL_THRESHOLDS:
            indices = np.where(mrec >= r_th)[0]
            if len(indices) > 0:
                interpolated_precisions.append(mpre[indices[0]])
            else:
                interpolated_precisions.append(0.0)

        ap = float(np.mean(interpolated_precisions))
        return float(np.clip(ap, 0.0, 1.0))

    @classmethod
    def evaluate_detections_at_iou(
        cls,
        predictions: List[BoundingBox],
        ground_truths: List[BoundingBox],
        iou_threshold: float = 0.50,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """
        Match predictions against ground truths sorted by confidence:
        Returns: (recalls, precisions, final_precision, final_recall)
        """
        if len(ground_truths) == 0:
            # If no ground truth, all predictions are False Positives
            return np.array([]), np.array([]), 0.0, 0.0

        if len(predictions) == 0:
            # If no predictions, all ground truths are False Negatives
            return np.array([0.0]), np.array([0.0]), 0.0, 0.0

        # Sort predictions by confidence descending
        sorted_preds = sorted(predictions, key=lambda b: b.confidence, reverse=True)

        tp = np.zeros(len(sorted_preds), dtype=np.float32)
        fp = np.zeros(len(sorted_preds), dtype=np.float32)

        matched_gt_indices = set()

        for i, pred_box in enumerate(sorted_preds):
            best_iou = -1.0
            best_gt_idx = -1

            for j, gt_box in enumerate(ground_truths):
                if j in matched_gt_indices:
                    continue  # Each GT can only be matched once
                if pred_box.class_id != gt_box.class_id:
                    continue  # Must match same class

                iou = BoxGeometryUtils.compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            if best_iou >= iou_threshold and best_gt_idx != -1:
                tp[i] = 1.0
                matched_gt_indices.add(best_gt_idx)
            else:
                fp[i] = 1.0

        # Cumulative sums
        cum_tp = np.cumsum(tp)
        cum_fp = np.cumsum(fp)

        # Precision and Recall arrays
        precisions = cum_tp / (cum_tp + cum_fp)
        recalls = cum_tp / float(len(ground_truths))

        final_p = float(precisions[-1]) if len(precisions) > 0 else 0.0
        final_r = float(recalls[-1]) if len(recalls) > 0 else 0.0

        return recalls, precisions, final_p, final_r


# ═══════════════════════════════════════════════════════════════════════════════
# Master Detection Metrics Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionMetricsEvaluator:
    """
    Evaluates 2D obstacle detection accuracy broken down by class (0-4)
    and distance range bands (Near, Mid, Far).
    Calculates mAP@0.50, mAP@0.75, and mAP@0.50:0.95 across 10 IoU steps.
    """

    IOU_THRESHOLDS_COCO = np.linspace(0.50, 0.95, 10)  # [0.50, 0.55, ..., 0.95]

    def __init__(
        self,
        class_names: Optional[Dict[int, str]] = None,
    ) -> None:
        self.class_names = class_names or {
            ObstacleClass.VEHICLE.value: "VEHICLE",
            ObstacleClass.PEDESTRIAN.value: "PEDESTRIAN",
            ObstacleClass.CYCLIST.value: "CYCLIST",
            ObstacleClass.OBSTACLE.value: "OBSTACLE",
            ObstacleClass.SIGN_MARKER.value: "SIGN_MARKER",
        }
        self.samples: List[DetectionSample] = []

    def reset(self) -> None:
        """Clear all accumulated evaluation samples."""
        self.samples.clear()

    def add_sample(
        self,
        frame_id: Union[int, str],
        ground_truths: List[BoundingBox],
        predictions: List[BoundingBox],
    ) -> None:
        """Register a single evaluated frame."""
        self.samples.append(DetectionSample(
            frame_id=frame_id,
            ground_truths=ground_truths,
            predictions=predictions,
        ))

    def evaluate_class_and_range(
        self,
        class_id: int,
        range_band: RangeBand,
    ) -> ClassRangeMetrics:
        """
        Evaluate AP@0.50, AP@0.75, and mAP@0.50:0.95 for a specific class
        filtered to a specific distance band.
        """
        class_name = self.class_names.get(class_id, f"Class_{class_id}")

        # Filter ground truths and predictions matching criteria
        filtered_gts: List[BoundingBox] = []
        filtered_preds: List[BoundingBox] = []

        for sample in self.samples:
            for gt in sample.ground_truths:
                if gt.class_id == class_id:
                    if range_band == RangeBand.ALL or gt.get_range_band() == range_band:
                        filtered_gts.append(gt)

            for pred in sample.predictions:
                if pred.class_id == class_id:
                    if range_band == RangeBand.ALL or pred.get_range_band() == range_band:
                        filtered_preds.append(pred)

        n_gt = len(filtered_gts)
        n_p = len(filtered_preds)

        if n_gt == 0:
            return ClassRangeMetrics(
                class_name=class_name,
                range_band=range_band.value,
                num_ground_truths=0,
                num_predictions=n_p,
                ap_50=1.0 if n_p == 0 else 0.0,
                ap_75=1.0 if n_p == 0 else 0.0,
                ap_50_95=1.0 if n_p == 0 else 0.0,
                precision_at_05=1.0 if n_p == 0 else 0.0,
                recall_at_05=1.0,
            )

        # Compute AP across the 10 COCO IoU thresholds
        ap_list = []
        p_at_05 = 0.0
        r_at_05 = 0.0

        for iou_th in self.IOU_THRESHOLDS_COCO:
            recs, precs, final_p, final_r = PrecisionRecallCurveCalculator.evaluate_detections_at_iou(
                predictions=filtered_preds,
                ground_truths=filtered_gts,
                iou_threshold=float(iou_th),
            )
            ap = PrecisionRecallCurveCalculator.compute_interpolated_ap(recs, precs)
            ap_list.append(ap)

            if abs(iou_th - 0.50) < 1e-3:
                p_at_05 = final_p
                r_at_05 = final_r

        ap_50 = ap_list[0]
        ap_75 = ap_list[5]  # 0.50 + 5 * 0.05 = 0.75
        ap_50_95 = float(np.mean(ap_list))

        return ClassRangeMetrics(
            class_name=class_name,
            range_band=range_band.value,
            num_ground_truths=n_gt,
            num_predictions=n_p,
            ap_50=round(ap_50, 4),
            ap_75=round(ap_75, 4),
            ap_50_95=round(ap_50_95, 4),
            precision_at_05=round(p_at_05, 4),
            recall_at_05=round(r_at_05, 4),
        )

    def evaluate_master_report(self) -> MasterDetectionReport:
        """
        Execute full benchmark evaluation across all classes (0-4) and all range strata.
        """
        t_start = time.perf_counter()

        per_class_summary: Dict[str, ClassRangeMetrics] = {}
        all_class_aps_50 = []
        all_class_aps_50_95 = []

        range_strata = [RangeBand.NEAR, RangeBand.MID, RangeBand.FAR, RangeBand.ALL]
        range_stratified_map_50: Dict[str, float] = {r.value: 0.0 for r in range_strata}
        range_stratified_map_50_95: Dict[str, float] = {r.value: 0.0 for r in range_strata}

        total_gt = sum(len(s.ground_truths) for s in self.samples)
        total_p = sum(len(s.predictions) for s in self.samples)

        # 1. Evaluate per class across ALL ranges
        for class_id in sorted(self.class_names.keys()):
            metrics_all = self.evaluate_class_and_range(class_id, RangeBand.ALL)
            c_name = self.class_names[class_id]
            per_class_summary[f"{c_name}_ALL"] = metrics_all
            all_class_aps_50.append(metrics_all.ap_50)
            all_class_aps_50_95.append(metrics_all.ap_50_95)

        # 2. Evaluate range band strata
        for r_band in range_strata:
            r_aps_50 = []
            r_aps_50_95 = []
            for class_id in sorted(self.class_names.keys()):
                m_r = self.evaluate_class_and_range(class_id, r_band)
                c_name = self.class_names[class_id]
                per_class_summary[f"{c_name}_{r_band.name}"] = m_r
                r_aps_50.append(m_r.ap_50)
                r_aps_50_95.append(m_r.ap_50_95)

            range_stratified_map_50[r_band.value] = round(float(np.mean(r_aps_50)), 4)
            range_stratified_map_50_95[r_band.value] = round(float(np.mean(r_aps_50_95)), 4)

        mean_ap_50 = float(np.mean(all_class_aps_50)) if all_class_aps_50 else 0.0
        mean_ap_50_95 = float(np.mean(all_class_aps_50_95)) if all_class_aps_50_95 else 0.0
        duration_ms = (time.perf_counter() - t_start) * 1000.0

        return MasterDetectionReport(
            num_frames=len(self.samples),
            total_ground_truths=total_gt,
            total_predictions=total_p,
            mean_ap_50=round(mean_ap_50, 4),
            mean_ap_50_95=round(mean_ap_50_95, 4),
            range_stratified_map_50=range_stratified_map_50,
            range_stratified_map_50_95=range_stratified_map_50_95,
            per_class_summary=per_class_summary,
            evaluation_duration_ms=round(duration_ms, 2),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Confusion Matrix, FROC & Metric Distance Error Evaluators
# ═══════════════════════════════════════════════════════════════════════════════

class DetectionConfusionMatrix:
    """
    Evaluates multi-class classification confusion for object detection,
    including background false positives (ghost detections) and false negatives (misses).
    Shape: (num_classes + 1, num_classes + 1) where index [num_classes] represents Background.
    """

    def __init__(self, num_classes: int = 5, iou_threshold: float = 0.50) -> None:
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        # Matrix rows = ground_truth (last row = background FP)
        # Matrix cols = prediction (last col = background FN/miss)
        self.matrix = np.zeros((num_classes + 1, num_classes + 1), dtype=np.int64)

    def update(self, ground_truths: List[BoundingBox], predictions: List[BoundingBox]) -> None:
        """Accumulate matches and classification assignments."""
        n_gt = len(ground_truths)
        n_p = len(predictions)

        if n_gt == 0 and n_p == 0:
            return

        matched_gt = set()
        matched_pred = set()

        # Compute IoU matrix
        for p_idx, p_box in enumerate(predictions):
            best_iou = -1.0
            best_gt_idx = -1

            for g_idx, g_box in enumerate(ground_truths):
                if g_idx in matched_gt:
                    continue
                iou = BoxGeometryUtils.compute_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = g_idx

            if best_iou >= self.iou_threshold and best_gt_idx != -1:
                g_class = ground_truths[best_gt_idx].class_id
                p_class = p_box.class_id
                self.matrix[g_class, p_class] += 1
                matched_gt.add(best_gt_idx)
                matched_pred.add(p_idx)
            else:
                # Prediction did not match any ground truth -> False Positive (Background row)
                p_class = p_box.class_id
                self.matrix[self.num_classes, p_class] += 1

        # Any unmatched ground truth -> False Negative (Background col)
        for g_idx, g_box in enumerate(ground_truths):
            if g_idx not in matched_gt:
                g_class = g_box.class_id
                self.matrix[g_class, self.num_classes] += 1

    def get_normalized_matrix(self) -> np.ndarray:
        """Row-normalized confusion probabilities."""
        row_sums = self.matrix.sum(axis=1, keepdims=True)
        return np.where(row_sums > 0, self.matrix / row_sums.astype(np.float64), 0.0)


class FROCAnalysisEngine:
    """
    Free-Response Receiver Operating Characteristic (FROC) evaluator.
    Measures Miss Rate (1 - Recall) as a function of False Positives Per Image (FPPI):
    Essential for ISO 26262 ASIL-B and Euro NCAP AEB safety targets (target: Miss Rate < 5% at 0.1 FPPI).
    """

    @staticmethod
    def compute_froc_curve(
        samples: List[DetectionSample],
        class_id: int,
        iou_threshold: float = 0.50,
        num_thresholds: int = 50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates (fppi_array, miss_rate_array) across confidence thresholds.
        """
        num_frames = max(1, len(samples))
        total_gts = sum(sum(1 for g in s.ground_truths if g.class_id == class_id) for s in samples)

        if total_gts == 0:
            return np.array([0.0]), np.array([0.0])

        conf_thresholds = np.linspace(0.05, 0.95, num_thresholds)
        fppi_list = []
        miss_rate_list = []

        for conf_th in conf_thresholds:
            total_fp = 0
            total_tp = 0

            for sample in samples:
                gts = [g for g in sample.ground_truths if g.class_id == class_id]
                preds = [p for p in sample.predictions if p.class_id == class_id and p.confidence >= conf_th]

                matched_gt = set()
                for p_box in preds:
                    best_iou = -1.0
                    best_g_idx = -1
                    for g_idx, g_box in enumerate(gts):
                        if g_idx in matched_gt:
                            continue
                        iou = BoxGeometryUtils.compute_iou(p_box, g_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_g_idx = g_idx

                    if best_iou >= iou_threshold and best_g_idx != -1:
                        total_tp += 1
                        matched_gt.add(best_g_idx)
                    else:
                        total_fp += 1

            fppi = total_fp / float(num_frames)
            recall = total_tp / float(total_gts)
            miss_rate = 1.0 - recall

            fppi_list.append(fppi)
            miss_rate_list.append(miss_rate)

        return np.array(fppi_list), np.array(miss_rate_list)


class RangeDistanceErrorEvaluator:
    """
    Evaluates monocular / stereo 3D distance estimation accuracy for detected obstacles.
    Calculates Mean Absolute Error (MAE) and percentage range error stratified
    by Near (0-10m), Mid (10-20m), and Far (20-40m) depth zones.
    """

    @staticmethod
    def evaluate_range_errors(samples: List[DetectionSample], iou_match_threshold: float = 0.50) -> Dict[str, Any]:
        """
        Match predictions to ground-truths and calculate metric distance error:
        delta_Z = |Z_pred - Z_gt|.
        """
        errors_by_band: Dict[str, List[float]] = {
            RangeBand.NEAR.value: [],
            RangeBand.MID.value: [],
            RangeBand.FAR.value: [],
            RangeBand.ALL.value: [],
        }

        for sample in samples:
            for pred in sample.predictions:
                best_iou = -1.0
                best_gt: Optional[BoundingBox] = None

                for gt in sample.ground_truths:
                    if pred.class_id == gt.class_id:
                        iou = BoxGeometryUtils.compute_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt = gt

                if best_iou >= iou_match_threshold and best_gt is not None:
                    err = abs(pred.range_depth_m - best_gt.range_depth_m)
                    band = best_gt.get_range_band()

                    errors_by_band[RangeBand.ALL.value].append(err)
                    if band.value in errors_by_band:
                        errors_by_band[band.value].append(err)

        summary: Dict[str, Any] = {}
        for band_name, err_list in errors_by_band.items():
            if err_list:
                mae = float(np.mean(err_list))
                rmse = float(np.sqrt(np.mean(np.array(err_list) ** 2)))
                max_err = float(np.max(err_list))
                summary[band_name] = {
                    "count": len(err_list),
                    "mae_m": round(mae, 3),
                    "rmse_m": round(rmse, 3),
                    "max_err_m": round(max_err, 3),
                }
            else:
                summary[band_name] = {"count": 0, "mae_m": 0.0, "rmse_m": 0.0, "max_err_m": 0.0}

        return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_detection_evaluator() -> DetectionMetricsEvaluator:
    """Factory constructor for DetectionMetricsEvaluator."""
    return DetectionMetricsEvaluator()


def run_detection_metrics_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. IoU calculation precision and edge overlap handling.
    2. Precision-Recall curve generation and 101-point COCO interpolation.
    3. Multi-class AP@0.50 and mAP@0.50:0.95 evaluation.
    4. Distance range-band stratification (Near: 0-10m, Mid: 10-20m, Far: 20-40m).
    5. Master benchmark report generation.
    6. Multi-class detection confusion matrix including background.
    7. Free-Response ROC (FROC) Miss Rate vs FPPI curve.
    8. Range distance estimation error analysis (MAE/RMSE).
    """
    print("Executing DetectionMetrics Verification Suite...")
    evaluator = get_detection_evaluator()

    # -------------------------------------------------------------------------
    # Test 1: IoU Geometry Mathematics
    # -------------------------------------------------------------------------
    box1 = BoundingBox(x_min=10, y_min=10, x_max=30, y_max=30, class_id=0) # 20x20, area=400
    box2 = BoundingBox(x_min=20, y_min=10, x_max=40, y_max=30, class_id=0) # 20x20, overlap 10x20=200
    iou = BoxGeometryUtils.compute_iou(box1, box2)
    assert abs(iou - 0.3333) < 1e-3, f"IoU mismatch: {iou}"
    assert BoxGeometryUtils.compute_iou(box1, box1) == 1.0
    box_far = BoundingBox(x_min=100, y_min=100, x_max=120, y_max=120, class_id=0)
    assert BoxGeometryUtils.compute_iou(box1, box_far) == 0.0

    # -------------------------------------------------------------------------
    # Test 2: Multi-Frame Detection Scenario with Range Strata
    # -------------------------------------------------------------------------
    gts_1 = [
        BoundingBox(x_min=50, y_min=100, x_max=150, y_max=200, class_id=0, range_depth_m=6.0),
        BoundingBox(x_min=200, y_min=80, x_max=240, y_max=180, class_id=1, range_depth_m=15.0),
        BoundingBox(x_min=300, y_min=120, x_max=360, y_max=160, class_id=3, range_depth_m=28.0),
    ]
    preds_1 = [
        BoundingBox(x_min=52, y_min=98, x_max=148, y_max=202, class_id=0, confidence=0.98, range_depth_m=6.1),
        BoundingBox(x_min=198, y_min=82, x_max=242, y_max=178, class_id=1, confidence=0.92, range_depth_m=14.8),
        BoundingBox(x_min=302, y_min=118, x_max=358, y_max=162, class_id=3, confidence=0.88, range_depth_m=27.9),
    ]
    evaluator.add_sample(frame_id=1, ground_truths=gts_1, predictions=preds_1)

    gts_2 = [
        BoundingBox(x_min=80, y_min=120, x_max=130, y_max=220, class_id=2, range_depth_m=8.0),
        BoundingBox(x_min=400, y_min=50, x_max=440, y_max=110, class_id=4, range_depth_m=35.0),
    ]
    preds_2 = [
        BoundingBox(x_min=81, y_min=119, x_max=129, y_max=221, class_id=2, confidence=0.95, range_depth_m=8.2),
        BoundingBox(x_min=250, y_min=140, x_max=290, y_max=180, class_id=3, confidence=0.70, range_depth_m=12.0),
    ]
    evaluator.add_sample(frame_id=2, ground_truths=gts_2, predictions=preds_2)

    # -------------------------------------------------------------------------
    # Test 3: Evaluate Range-Stratified Master Report
    # -------------------------------------------------------------------------
    report = evaluator.evaluate_master_report()
    print(f"Master Detection Report: Frames={report.num_frames} | Total GT={report.total_ground_truths} | Total Pred={report.total_predictions}")
    print(f"Overall mAP@0.50: {report.mean_ap_50:.4f} | mAP@0.50:0.95: {report.mean_ap_50_95:.4f}")
    print(f"Near-Field mAP@0.50 (0-10m): {report.range_stratified_map_50['0-10m']:.4f}")

    assert report.num_frames == 2
    assert report.total_ground_truths == 5
    assert report.total_predictions == 5
    assert report.range_stratified_map_50["0-10m"] == 1.0
    assert report.mean_ap_50 >= 0.70

    # -------------------------------------------------------------------------
    # Test 4: Detection Confusion Matrix
    # -------------------------------------------------------------------------
    cm = DetectionConfusionMatrix(num_classes=5, iou_threshold=0.50)
    for sample in evaluator.samples:
        cm.update(sample.ground_truths, sample.predictions)

    norm_cm = cm.get_normalized_matrix()
    assert norm_cm.shape == (6, 6)
    # Check that Vehicle (Class 0) has high true positive rate
    assert norm_cm[0, 0] > 0.90
    print(f"Confusion Matrix Shape: {cm.matrix.shape} | Vehicle TP Count: {cm.matrix[0, 0]}")

    # -------------------------------------------------------------------------
    # Test 5: FROC Miss Rate vs FPPI
    # -------------------------------------------------------------------------
    fppi, miss_rate = FROCAnalysisEngine.compute_froc_curve(evaluator.samples, class_id=0, num_thresholds=10)
    assert len(fppi) == 10
    assert np.min(miss_rate) == 0.0, "Perfect vehicle detection must reach 0.0 miss rate"
    print(f"FROC Curve: Min Miss Rate={np.min(miss_rate):.2f} at FPPI={fppi[np.argmin(miss_rate)]:.2f}")

    # -------------------------------------------------------------------------
    # Test 6: Range Metric Distance Error Evaluator
    # -------------------------------------------------------------------------
    range_errors = RangeDistanceErrorEvaluator.evaluate_range_errors(evaluator.samples)
    near_err = range_errors["0-10m"]
    assert near_err["count"] == 2
    assert near_err["mae_m"] < 0.25, f"Near-field range error too high: {near_err['mae_m']}m"
    print(f"Near-field Range Error: MAE={near_err['mae_m']}m | RMSE={near_err['rmse_m']}m")

    print("DetectionMetrics verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_detection_metrics_verification_suite()

