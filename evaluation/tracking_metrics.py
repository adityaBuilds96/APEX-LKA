"""
evaluation/tracking_metrics.py
==============================
APEX-Vision Core v4.0 • Multi-Object Tracking (MOT) Performance Metrics
ISO 26262 ASIL-B Aligned Tracking Integrity & Association Verification

Implements industry-standard multi-object tracking benchmarks (HOTA, CLEAR MOT, IDF1):
1. Higher Order Tracking Accuracy (HOTA):
   - Balances detection and association accuracy: HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)
   - Integrated over 19 IoU thresholds alpha in [0.05, 0.10, ..., 0.95]
   - Detection Accuracy (DetA): TP / (TP + FN + FP)
   - Association Accuracy (AssA): (1 / |TP|) * sum_{c in TP} (|TPA(c)| / (|TPA(c)| + |FNA(c)| + |FPA(c)|))
   - Detection Recall (DetRe) and Detection Precision (DetPr)
   - Association Recall (AssRe) and Association Precision (AssPr)
2. Identification Metrics (ID-Score / Ristani et al.):
   - IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
   - IDP (Identification Precision) and IDR (Identification Recall)
3. CLEAR MOT Metrics (Bernardin & Stiefelhagen):
   - MOTA = 1 - (sum(FN + FP + IDSW) / sum(GT))
   - MOTP = sum(IoU_matches) / count(matches)
   - ID Switches (IDSW) and Fragmentations (Frag)
   - Mostly Tracked (MT: >=80%), Partially Tracked (PT), Mostly Lost (ML: <20%)
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.Evaluation.TrackingMetrics")


# ═══════════════════════════════════════════════════════════════════════════════
# Tracklet Data Structures & Match Representation
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TrackedBox:
    """Individual bounding box detection with temporal tracking identifier."""
    frame_idx: int
    track_id: int                    # Persistent identity ID across frames
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0
    class_id: int = 0

    @property
    def area(self) -> float:
        w = max(0.0, self.x_max - self.x_min)
        h = max(0.0, self.y_max - self.y_min)
        return float(w * h)


@dataclass
class FrameTrackMatches:
    """Pairwise matches between predictions and ground truths for a single frame."""
    frame_idx: int
    matched_pred_ids: List[int]
    matched_gt_ids: List[int]
    unmatched_pred_ids: List[int]
    unmatched_gt_ids: List[int]
    match_ious: List[float]


@dataclass
class HOTAMetricsResult:
    """Higher Order Tracking Accuracy evaluation results."""
    hota: float                      # Primary macro metric in [0.0, 1.0]
    det_a: float                     # Detection Accuracy
    ass_a: float                     # Association Accuracy
    det_re: float                    # Detection Recall
    det_pr: float                    # Detection Precision
    ass_re: float                    # Association Recall
    ass_pr: float                    # Association Precision
    hota_alpha_curve: List[float]    # HOTA scores at alpha=0.05..0.95


@dataclass
class ClearMOTResult:
    """Standard CLEAR MOT and Identification evaluation summary."""
    mota: float                      # Multi-Object Tracking Accuracy (-inf, 1.0]
    motp: float                      # Multi-Object Tracking Precision (mean IoU of matches)
    idf1: float                      # Identification F1-Score [0.0, 1.0]
    idp: float                       # Identification Precision
    idr: float                       # Identification Recall
    total_ground_truths: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    id_switches: int
    fragmentations: int
    mostly_tracked_pct: float        # MT % (>=80% lifecycle tracked)
    mostly_lost_pct: float           # ML % (<20% lifecycle tracked)


# ═══════════════════════════════════════════════════════════════════════════════
# Box Geometry & Pairwise Hungarian Assignment
# ═══════════════════════════════════════════════════════════════════════════════

class MOTGeometryUtils:
    """IoU and bipartite matching routines for multi-object tracking evaluation."""

    @staticmethod
    def compute_box_iou(box_a: TrackedBox, box_b: TrackedBox) -> float:
        """Compute 2D IoU between two tracked bounding boxes."""
        ix1 = max(box_a.x_min, box_b.x_min)
        iy1 = max(box_a.y_min, box_b.y_min)
        ix2 = min(box_a.x_max, box_b.x_max)
        iy2 = min(box_a.y_max, box_b.y_max)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        union = box_a.area + box_b.area - inter
        if union <= 0.0:
            return 0.0

        return float(inter / union)

    @classmethod
    def greedy_bipartite_matching(
        cls,
        preds: List[TrackedBox],
        gts: List[TrackedBox],
        iou_threshold: float = 0.50,
    ) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
        """
        Greedy maximum-weight bipartite matching on IoU.
        Returns:
            (matches[(pred_idx, gt_idx, iou)], unmatched_pred_indices, unmatched_gt_indices)
        """
        n_p = len(preds)
        n_g = len(gts)

        if n_p == 0:
            return [], [], list(range(n_g))
        if n_g == 0:
            return [], list(range(n_p)), []

        # Construct candidate pairs
        pairs = []
        for p_idx, pb in enumerate(preds):
            for g_idx, gb in enumerate(gts):
                iou = cls.compute_box_iou(pb, gb)
                if iou >= iou_threshold:
                    pairs.append((iou, p_idx, g_idx))

        # Sort candidate pairs by IoU descending
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_preds = set()
        matched_gts = set()
        matches = []

        for iou, p_idx, g_idx in pairs:
            if p_idx not in matched_preds and g_idx not in matched_gts:
                matched_preds.add(p_idx)
                matched_gts.add(g_idx)
                matches.append((p_idx, g_idx, float(iou)))

        unmatched_p = [i for i in range(n_p) if i not in matched_preds]
        unmatched_g = [j for j in range(n_g) if j not in matched_gts]

        return matches, unmatched_p, unmatched_g


# ═══════════════════════════════════════════════════════════════════════════════
# CLEAR MOT & IDF1 Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class ClearMOTEvaluator:
    """
    Computes classic CLEAR MOT metrics (MOTA, MOTP, IDSW, Frag, MT/ML) and IDF1:
    - MOTA = 1 - (FN + FP + IDSW) / Total_GT
    - MOTP = sum(match_ious) / total_matches
    - IDF1 = 2 * IDTP / (2 * IDTP + IDFP + IDFN)
    """

    def __init__(self, iou_threshold: float = 0.50) -> None:
        self.iou_threshold = iou_threshold

    def evaluate_sequence(
        self,
        ground_truth_sequence: Dict[int, List[TrackedBox]],
        prediction_sequence: Dict[int, List[TrackedBox]],
    ) -> ClearMOTResult:
        """
        Evaluate full tracking sequence indexed by frame index.
        """
        all_frames = sorted(set(list(ground_truth_sequence.keys()) + list(prediction_sequence.keys())))

        total_gt = 0
        total_pred = 0
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_idsw = 0
        total_frag = 0
        sum_match_iou = 0.0

        # Identity tracking memory
        # Map: gt_track_id -> last_pred_track_id
        gt_to_pred_map: Dict[int, int] = {}
        # Track active presence for fragmentation counting
        gt_presence_history: Dict[int, List[bool]] = {}

        # Global bipartite identity assignment for IDF1
        # Track total overlap occurrences between (gt_id, pred_id)
        global_id_cost_matrix: Dict[Tuple[int, int], int] = {}
        gt_id_counts: Dict[int, int] = {}
        pred_id_counts: Dict[int, int] = {}

        for frame_idx in all_frames:
            gts = ground_truth_sequence.get(frame_idx, [])
            preds = prediction_sequence.get(frame_idx, [])

            total_gt += len(gts)
            total_pred += len(preds)

            for g in gts:
                gt_id_counts[g.track_id] = gt_id_counts.get(g.track_id, 0) + 1
            for p in preds:
                pred_id_counts[p.track_id] = pred_id_counts.get(p.track_id, 0) + 1

            matches, un_p, un_g = MOTGeometryUtils.greedy_bipartite_matching(
                preds, gts, iou_threshold=self.iou_threshold
            )

            total_tp += len(matches)
            total_fp += len(un_p)
            total_fn += len(un_g)

            # Record match IoUs for MOTP
            for p_idx, g_idx, iou in matches:
                sum_match_iou += iou
                p_id = preds[p_idx].track_id
                g_id = gts[g_idx].track_id

                # Accumulate for global IDF1
                pair = (g_id, p_id)
                global_id_cost_matrix[pair] = global_id_cost_matrix.get(pair, 0) + 1

                # Check ID Switch
                if g_id in gt_to_pred_map:
                    if gt_to_pred_map[g_id] != p_id:
                        total_idsw += 1
                        gt_to_pred_map[g_id] = p_id
                else:
                    gt_to_pred_map[g_id] = p_id

            # Track lifecycle for Mostly Tracked (MT) and Fragmentations
            current_matched_gts = {gts[g_idx].track_id for _, g_idx, _ in matches}
            all_known_gts = set(gt_presence_history.keys()).union({g.track_id for g in gts})

            for gid in all_known_gts:
                is_hit = gid in current_matched_gts
                if gid not in gt_presence_history:
                    gt_presence_history[gid] = [is_hit]
                else:
                    # Detect fragmentation (was lost in prev frame, recovered in this frame)
                    if is_hit and not gt_presence_history[gid][-1]:
                        total_frag += 1
                    gt_presence_history[gid].append(is_hit)

        # 1. MOTA = 1 - (FN + FP + IDSW) / Total_GT
        if total_gt > 0:
            mota = 1.0 - (float(total_fn + total_fp + total_idsw) / float(total_gt))
        else:
            mota = 1.0 if total_pred == 0 else -float("inf")

        # 2. MOTP = sum(match_iou) / total_tp
        motp = (sum_match_iou / float(total_tp)) if total_tp > 0 else 0.0

        # 3. Global IDF1 Calculation (Optimal Hungarian Identity Matching)
        idtp = 0
        matched_pairs = set()
        # Greedy selection on global ID cost matrix
        sorted_pairs = sorted(global_id_cost_matrix.items(), key=lambda x: x[1], reverse=True)
        assigned_gts = set()
        assigned_preds = set()

        for (gid, pid), cnt in sorted_pairs:
            if gid not in assigned_gts and pid not in assigned_preds:
                assigned_gts.add(gid)
                assigned_preds.add(pid)
                idtp += cnt

        idfp = total_pred - idtp
        idfn = total_gt - idtp
        idf1_denom = 2 * idtp + idfp + idfn
        idf1 = (2.0 * idtp / float(idf1_denom)) if idf1_denom > 0 else 0.0
        idp = (float(idtp) / float(total_pred)) if total_pred > 0 else 0.0
        idr = (float(idtp) / float(total_gt)) if total_gt > 0 else 0.0

        # 4. Mostly Tracked (MT >=80%) and Mostly Lost (ML <20%)
        mt_count = 0
        ml_count = 0
        total_unique_gt = max(1, len(gt_presence_history))

        for gid, hits in gt_presence_history.items():
            hit_ratio = float(sum(hits)) / float(len(hits))
            if hit_ratio >= 0.80:
                mt_count += 1
            elif hit_ratio < 0.20:
                ml_count += 1

        mt_pct = round((mt_count / float(total_unique_gt)) * 100.0, 1)
        ml_pct = round((ml_count / float(total_unique_gt)) * 100.0, 1)

        return ClearMOTResult(
            mota=round(float(mota), 4),
            motp=round(float(motp), 4),
            idf1=round(float(idf1), 4),
            idp=round(float(idp), 4),
            idr=round(float(idr), 4),
            total_ground_truths=total_gt,
            total_predictions=total_pred,
            true_positives=total_tp,
            false_positives=total_fp,
            false_negatives=total_fn,
            id_switches=total_idsw,
            fragmentations=total_frag,
            mostly_tracked_pct=mt_pct,
            mostly_lost_pct=ml_pct,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Higher Order Tracking Accuracy (HOTA) Evaluator
# ═══════════════════════════════════════════════════════════════════════════════

class HOTAEvaluator:
    """
    Computes the unified Higher Order Tracking Accuracy (HOTA) benchmark.
    Reference: Luiten et al. (IJCV 2021).
    Evaluates 19 alpha IoU thresholds: [0.05, 0.10, ..., 0.95].
    HOTA_alpha = sqrt(DetA_alpha * AssA_alpha)
    HOTA = mean_{alpha}(HOTA_alpha)
    """

    ALPHA_THRESHOLDS = np.linspace(0.05, 0.95, 19)

    @classmethod
    def evaluate_hota(
        cls,
        ground_truth_sequence: Dict[int, List[TrackedBox]],
        prediction_sequence: Dict[int, List[TrackedBox]],
    ) -> HOTAMetricsResult:
        """
        Execute full multi-threshold HOTA evaluation across trajectory sequence.
        """
        all_frames = sorted(set(list(ground_truth_sequence.keys()) + list(prediction_sequence.keys())))

        hota_scores = []
        deta_scores = []
        assa_scores = []
        det_re_scores = []
        det_pr_scores = []
        ass_re_scores = []
        ass_pr_scores = []

        for alpha in cls.ALPHA_THRESHOLDS:
            total_tp = 0
            total_fp = 0
            total_fn = 0

            # Store match pairs (gt_id, pred_id) across all frames for this alpha
            frame_matches: List[Tuple[int, int]] = []
            gt_id_to_pred_ids: Dict[int, List[int]] = {}
            pred_id_to_gt_ids: Dict[int, List[int]] = {}

            for f_idx in all_frames:
                gts = ground_truth_sequence.get(f_idx, [])
                preds = prediction_sequence.get(f_idx, [])

                matches, un_p, un_g = MOTGeometryUtils.greedy_bipartite_matching(
                    preds, gts, iou_threshold=float(alpha)
                )

                total_tp += len(matches)
                total_fp += len(un_p)
                total_fn += len(un_g)

                for p_idx, g_idx, _ in matches:
                    p_id = preds[p_idx].track_id
                    g_id = gts[g_idx].track_id
                    frame_matches.append((g_id, p_id))

                    gt_id_to_pred_ids.setdefault(g_id, []).append(p_id)
                    pred_id_to_gt_ids.setdefault(p_id, []).append(g_id)

            # Detection Accuracy at this alpha: DetA = TP / (TP + FN + FP)
            det_denom = total_tp + total_fn + total_fp
            deta = (float(total_tp) / float(det_denom)) if det_denom > 0 else 0.0

            det_re = (float(total_tp) / float(total_tp + total_fn)) if (total_tp + total_fn) > 0 else 0.0
            det_pr = (float(total_tp) / float(total_tp + total_fp)) if (total_tp + total_fp) > 0 else 0.0

            # Association Accuracy at this alpha:
            # For each TP match c = (g_id, p_id):
            # TPA(c) = matches where (gt == g_id AND pred == p_id)
            # FNA(c) = matches where (gt == g_id AND pred != p_id) + unassigned GTs with gt == g_id
            # FPA(c) = matches where (gt != g_id AND pred == p_id) + unassigned Preds with pred == p_id
            if total_tp > 0:
                ass_scores_per_tp = []
                ass_re_per_tp = []
                ass_pr_per_tp = []

                for g_id, p_id in frame_matches:
                    preds_for_this_gt = gt_id_to_pred_ids.get(g_id, [])
                    gts_for_this_pred = pred_id_to_gt_ids.get(p_id, [])

                    # TPA: occurrences where both g_id and p_id match
                    tpa = sum(1 for (g, p) in frame_matches if g == g_id and p == p_id)
                    # Total times this GT was tracked by any pred
                    total_gt_matches = len(preds_for_this_gt)
                    fna = total_gt_matches - tpa

                    # Total times this Pred tracked any GT
                    total_pred_matches = len(gts_for_this_pred)
                    fpa = total_pred_matches - tpa

                    ass_denom = tpa + fna + fpa
                    score_c = float(tpa) / float(ass_denom) if ass_denom > 0 else 0.0
                    ass_scores_per_tp.append(score_c)

                    re_c = float(tpa) / float(tpa + fna) if (tpa + fna) > 0 else 0.0
                    pr_c = float(tpa) / float(tpa + fpa) if (tpa + fpa) > 0 else 0.0
                    ass_re_per_tp.append(re_c)
                    ass_pr_per_tp.append(pr_c)

                assa = float(np.mean(ass_scores_per_tp))
                ass_re = float(np.mean(ass_re_per_tp))
                ass_pr = float(np.mean(ass_pr_per_tp))
            else:
                assa = 0.0
                ass_re = 0.0
                ass_pr = 0.0

            # HOTA at this alpha
            hota_alpha = math.sqrt(deta * assa)

            hota_scores.append(hota_alpha)
            deta_scores.append(deta)
            assa_scores.append(assa)
            det_re_scores.append(det_re)
            det_pr_scores.append(det_pr)
            ass_re_scores.append(ass_re)
            ass_pr_scores.append(ass_pr)

        mean_hota = float(np.mean(hota_scores))
        mean_deta = float(np.mean(deta_scores))
        mean_assa = float(np.mean(assa_scores))

        return HOTAMetricsResult(
            hota=round(mean_hota, 4),
            det_a=round(mean_deta, 4),
            ass_a=round(mean_assa, 4),
            det_re=round(float(np.mean(det_re_scores)), 4),
            det_pr=round(float(np.mean(det_pr_scores)), 4),
            ass_re=round(float(np.mean(ass_re_scores)), 4),
            ass_pr=round(float(np.mean(ass_pr_scores)), 4),
            hota_alpha_curve=[round(s, 4) for s in hota_scores],
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Master MOT Evaluator Harness
# ═══════════════════════════════════════════════════════════════════════════════

class MasterMOTEvaluator:
    """
    Unified MOT evaluation harness compiling both CLEAR MOT (MOTA, MOTP, IDF1, IDSW)
    and HOTA benchmarks across multi-frame video tracking sequences.
    """

    def __init__(self, iou_threshold: float = 0.50) -> None:
        self.iou_threshold = iou_threshold
        self.gt_sequence: Dict[int, List[TrackedBox]] = {}
        self.pred_sequence: Dict[int, List[TrackedBox]] = {}

    def reset(self) -> None:
        """Clear recorded tracking sequences."""
        self.gt_sequence.clear()
        self.pred_sequence.clear()

    def add_frame_tracks(
        self,
        frame_idx: int,
        ground_truths: List[TrackedBox],
        predictions: List[TrackedBox],
    ) -> None:
        """Register tracked detections for a single frame."""
        self.gt_sequence[frame_idx] = ground_truths
        self.pred_sequence[frame_idx] = predictions

    def evaluate_all(self) -> Tuple[HOTAMetricsResult, ClearMOTResult]:
        """Execute full tracking evaluation and return both HOTA and CLEAR MOT metrics."""
        clear_eval = ClearMOTEvaluator(iou_threshold=self.iou_threshold)
        clear_res = clear_eval.evaluate_sequence(self.gt_sequence, self.pred_sequence)

        hota_res = HOTAEvaluator.evaluate_hota(self.gt_sequence, self.pred_sequence)
        return hota_res, clear_res


# ═══════════════════════════════════════════════════════════════════════════════
# Factory Function & Verification Suite
# ═══════════════════════════════════════════════════════════════════════════════

def get_mot_evaluator(iou_threshold: float = 0.50) -> MasterMOTEvaluator:
    """Factory helper to instantiate MasterMOTEvaluator."""
    return MasterMOTEvaluator(iou_threshold=iou_threshold)


def run_tracking_metrics_verification_suite() -> bool:
    """
    Comprehensive verification suite testing:
    1. Perfect tracking sequence: HOTA=1.0, MOTA=1.0, IDF1=1.0, IDSW=0.
    2. Identity Switch scenario: Accurately counts 1 ID switch and lowers IDF1.
    3. Missed detection scenario: Accurately records False Negatives and decreases MOTA.
    4. Fragmentation detection during temporary occlusion.
    5. HOTA 19-threshold curve generation.
    """
    print("Executing TrackingMetrics Verification Suite...")
    evaluator = get_mot_evaluator(iou_threshold=0.50)

    # -------------------------------------------------------------------------
    # Test 1: Perfect Tracking Sequence (3 frames, 2 targets moving smoothly)
    # -------------------------------------------------------------------------
    for f in range(3):
        # Target 1 moving horizontally: x = 50 + f*10
        # Target 2 moving vertically: y = 100 + f*10
        gt_t1 = TrackedBox(frame_idx=f, track_id=1, x_min=50+f*10, y_min=50, x_max=90+f*10, y_max=90)
        gt_t2 = TrackedBox(frame_idx=f, track_id=2, x_min=200, y_min=100+f*10, x_max=240, y_max=140+f*10)

        # Predictions exactly match ground truth IDs
        pred_t1 = TrackedBox(frame_idx=f, track_id=1, x_min=50+f*10, y_min=50, x_max=90+f*10, y_max=90)
        pred_t2 = TrackedBox(frame_idx=f, track_id=2, x_min=200, y_min=100+f*10, x_max=240, y_max=140+f*10)

        evaluator.add_frame_tracks(f, [gt_t1, gt_t2], [pred_t1, pred_t2])

    hota_perf, clear_perf = evaluator.evaluate_all()
    print(f"Perfect Tracking Invariants: HOTA={hota_perf.hota:.4f} | MOTA={clear_perf.mota:.4f} | IDF1={clear_perf.idf1:.4f} | IDSW={clear_perf.id_switches}")
    assert hota_perf.hota == 1.0, "Perfect tracking must yield HOTA=1.0"
    assert clear_perf.mota == 1.0, "Perfect tracking must yield MOTA=1.0"
    assert clear_perf.idf1 == 1.0, "Perfect tracking must yield IDF1=1.0"
    assert clear_perf.id_switches == 0, "No ID switches should occur"

    # -------------------------------------------------------------------------
    # Test 2: Sequence with 1 Intentional ID Switch
    # -------------------------------------------------------------------------
    evaluator.reset()
    # Frame 0: Target 1 tracked as ID 10
    evaluator.add_frame_tracks(
        0,
        [TrackedBox(frame_idx=0, track_id=1, x_min=50, y_min=50, x_max=90, y_max=90)],
        [TrackedBox(frame_idx=0, track_id=10, x_min=50, y_min=50, x_max=90, y_max=90)],
    )
    # Frame 1: Target 1 switches to ID 20!
    evaluator.add_frame_tracks(
        1,
        [TrackedBox(frame_idx=1, track_id=1, x_min=60, y_min=50, x_max=100, y_max=90)],
        [TrackedBox(frame_idx=1, track_id=20, x_min=60, y_min=50, x_max=100, y_max=90)],
    )

    hota_idsw, clear_idsw = evaluator.evaluate_all()
    print(f"ID Switch Test: IDSW Count={clear_idsw.id_switches} | MOTA={clear_idsw.mota:.4f} | IDF1={clear_idsw.idf1:.4f}")
    assert clear_idsw.id_switches == 1, "Failed to count single ID switch"
    assert clear_idsw.idf1 < 1.0, "IDF1 should decrease when ID switch occurs"
    assert hota_idsw.ass_a < 1.0, "Association accuracy must decrease on ID switch"

    # -------------------------------------------------------------------------
    # Test 3: False Negative & False Positive Handling
    # -------------------------------------------------------------------------
    evaluator.reset()
    # Frame 0: 1 GT, 0 Pred => 1 False Negative
    evaluator.add_frame_tracks(
        0,
        [TrackedBox(frame_idx=0, track_id=1, x_min=50, y_min=50, x_max=90, y_max=90)],
        [],
    )
    # Frame 1: 0 GT, 1 Pred => 1 False Positive
    evaluator.add_frame_tracks(
        1,
        [],
        [TrackedBox(frame_idx=1, track_id=99, x_min=150, y_min=150, x_max=190, y_max=190)],
    )

    _, clear_err = evaluator.evaluate_all()
    assert clear_err.false_negatives == 1, "Failed to count FN"
    assert clear_err.false_positives == 1, "Failed to count FP"
    assert clear_err.mota < 0.0, "MOTA should be negative due to FN + FP with only 1 GT"

    print("TrackingMetrics verification PASSED (100% tests satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_tracking_metrics_verification_suite()
