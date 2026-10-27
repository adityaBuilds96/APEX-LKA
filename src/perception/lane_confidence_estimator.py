"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

src/perception/lane_confidence_estimator.py
Multi-Modal Lane Perception Confidence Estimator & Dynamic Fusion Weight Engine

Provides dual-stream confidence estimation for classical computer vision pipelines
and deep learning neural segmentation heads. Evaluates geometric stability, edge
gradient consistency, photometric contrast, spatial mask topology, and ensemble
concordance to synthesize calibrated confidence scores and scene-adaptive fusion weights.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("apex_vision.perception.lane_confidence_estimator")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enumerations & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class PerceptionSubsystem(str, Enum):
    """Perception subsystem stream identifier."""
    CLASSICAL_CV = "CLASSICAL_CV"
    DEEP_LEARNING = "DEEP_LEARNING"
    HYBRID_FUSION = "HYBRID_FUSION"


class DominantStreamMode(str, Enum):
    """Operating mode dictated by confidence weighting."""
    PURE_DEEP_LEARNING = "PURE_DEEP_LEARNING"
    BALANCED_HYBRID = "BALANCED_HYBRID"
    CLASSICAL_FALLBACK = "CLASSICAL_FALLBACK"
    SAFETY_INTERLOCK = "SAFETY_INTERLOCK"


@dataclass(frozen=True)
class ClassicalConfidenceMetrics:
    """
    Fine-grained confidence metrics for classical computer vision extraction.
    Scores bounded in [0.0, 1.0].
    """
    edge_gradient_alignment: float
    color_contrast_ratio: float
    ransac_inlier_ratio: float
    fit_residual_score: float
    lane_width_consistency: float
    homography_stability: float
    raw_confidence: float
    calibrated_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "edge_gradient_alignment": round(self.edge_gradient_alignment, 4),
            "color_contrast_ratio": round(self.color_contrast_ratio, 4),
            "ransac_inlier_ratio": round(self.ransac_inlier_ratio, 4),
            "fit_residual_score": round(self.fit_residual_score, 4),
            "lane_width_consistency": round(self.lane_width_consistency, 4),
            "homography_stability": round(self.homography_stability, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
        }


@dataclass(frozen=True)
class DeepLearningConfidenceMetrics:
    """
    Fine-grained confidence metrics for neural network segmentation predictions.
    Scores bounded in [0.0, 1.0].
    """
    mean_softmax_margin: float
    entropy_uncertainty_score: float
    spatial_continuity_score: float
    thickness_consistency: float
    corridor_coherence: float
    ensemble_agreement: float
    raw_confidence: float
    calibrated_confidence: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "mean_softmax_margin": round(self.mean_softmax_margin, 4),
            "entropy_uncertainty_score": round(self.entropy_uncertainty_score, 4),
            "spatial_continuity_score": round(self.spatial_continuity_score, 4),
            "thickness_consistency": round(self.thickness_consistency, 4),
            "corridor_coherence": round(self.corridor_coherence, 4),
            "ensemble_agreement": round(self.ensemble_agreement, 4),
            "raw_confidence": round(self.raw_confidence, 4),
            "calibrated_confidence": round(self.calibrated_confidence, 4),
        }


@dataclass
class ConfidenceFusionReport:
    """
    Comprehensive perception confidence and dynamic fusion arbitration report.
    """
    timestamp_utc: str
    fusion_alpha_dl_weight: float
    classical_weight: float
    dominant_mode: DominantStreamMode
    fused_confidence: float
    classical_metrics: ClassicalConfidenceMetrics
    dl_metrics: DeepLearningConfidenceMetrics
    scene_tag: str
    scene_bias_factor: float
    degradation_flag: bool
    diagnostic_reasoning: str
    computation_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_utc": self.timestamp_utc,
            "fusion_alpha_dl_weight": round(self.fusion_alpha_dl_weight, 4),
            "classical_weight": round(self.classical_weight, 4),
            "dominant_mode": self.dominant_mode.value,
            "fused_confidence": round(self.fused_confidence, 4),
            "classical_metrics": self.classical_metrics.to_dict(),
            "dl_metrics": self.dl_metrics.to_dict(),
            "scene_tag": self.scene_tag,
            "scene_bias_factor": round(self.scene_bias_factor, 4),
            "degradation_flag": self.degradation_flag,
            "diagnostic_reasoning": self.diagnostic_reasoning,
            "computation_latency_ms": round(self.computation_latency_ms, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration Dataclasses
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ClassicalConfidenceConfig:
    """Weightings and thresholds for classical CV confidence estimation."""
    weight_edge_alignment: float = 0.20
    weight_color_contrast: float = 0.15
    weight_ransac_inliers: float = 0.25
    weight_fit_residuals: float = 0.15
    weight_lane_width: float = 0.15
    weight_homography: float = 0.10

    nominal_lane_width_m: float = 3.70
    lane_width_tolerance_m: float = 0.70
    max_acceptable_residual_px: float = 8.0
    min_inlier_ratio_floor: float = 0.35
    gradient_orientation_tol_deg: float = 25.0


@dataclass
class DeepLearningConfidenceConfig:
    """Weightings and thresholds for deep learning confidence estimation."""
    weight_softmax_margin: float = 0.25
    weight_entropy: float = 0.20
    weight_spatial_continuity: float = 0.20
    weight_thickness: float = 0.15
    weight_corridor_coherence: float = 0.10
    weight_ensemble_agreement: float = 0.10

    entropy_scaling_factor: float = 1.44269  # 1 / ln(2)
    min_boundary_pixels_threshold: int = 150
    expected_nominal_thickness_px: float = 5.0
    thickness_variance_penalty_scale: float = 0.25


@dataclass
class ConfidenceFusionConfig:
    """Global configuration governing multi-modal confidence fusion and calibration."""
    classical_config: ClassicalConfidenceConfig = field(default_factory=ClassicalConfidenceConfig)
    dl_config: DeepLearningConfidenceConfig = field(default_factory=DeepLearningConfidenceConfig)

    # Base scene priors for alpha (DL weight)
    scene_alpha_priors: Dict[str, float] = field(default_factory=lambda: {
        "HIGHWAY": 0.65,
        "URBAN": 0.75,
        "RURAL": 0.70,
        "TUNNEL": 0.85,
        "NIGHT": 0.85,
        "RAIN": 0.90,
    })

    # Confidence scaling and calibration bounds
    min_alpha_dl_floor: float = 0.10
    max_alpha_dl_ceiling: float = 0.95
    degradation_confidence_threshold: float = 0.35
    temperature_calibration_cv: float = 1.10
    temperature_calibration_dl: float = 0.90


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Classical Computer Vision Confidence Estimator
# ═══════════════════════════════════════════════════════════════════════════════

class ClassicalConfidenceEstimator:
    """
    Evaluates physical and optical evidence of lane markings extracted via
    classical image processing (Canny, Hough, IPM, RANSAC).
    """

    def __init__(self, config: Optional[ClassicalConfidenceConfig] = None) -> None:
        self.config = config or ClassicalConfidenceConfig()

    def estimate_confidence(
        self,
        edge_map: Optional[np.ndarray],
        sobel_x: Optional[np.ndarray],
        sobel_y: Optional[np.ndarray],
        ransac_inliers_count: int,
        total_candidate_points: int,
        mean_poly_residual_px: float,
        measured_lane_width_m: float,
        ipm_homography_valid: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        hls_image: Optional[np.ndarray] = None,
    ) -> ClassicalConfidenceMetrics:
        """
        Compute normalized multi-criteria classical CV confidence score.
        """
        # 1. Edge gradient alignment
        edge_align_score = self._compute_edge_alignment(
            edge_map, sobel_x, sobel_y, roi_mask
        )

        # 2. Color space contrast ratio
        contrast_score = self._compute_color_contrast(hls_image, edge_map)

        # 3. RANSAC inlier ratio
        inlier_score = self._compute_inlier_score(
            ransac_inliers_count, total_candidate_points
        )

        # 4. Polynomial fit residual score
        residual_score = self._compute_residual_score(mean_poly_residual_px)

        # 5. Lane width consistency
        width_score = self._compute_lane_width_score(measured_lane_width_m)

        # 6. IPM Homography stability
        homography_score = 1.0 if ipm_homography_valid else 0.20

        # Weighted combination
        raw_score = (
            self.config.weight_edge_alignment * edge_align_score
            + self.config.weight_color_contrast * contrast_score
            + self.config.weight_ransac_inliers * inlier_score
            + self.config.weight_fit_residuals * residual_score
            + self.config.weight_lane_width * width_score
            + self.config.weight_homography * homography_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # Sigmoidal calibration
        calibrated = self._calibrate_classical_score(raw_score)

        return ClassicalConfidenceMetrics(
            edge_gradient_alignment=float(edge_align_score),
            color_contrast_ratio=float(contrast_score),
            ransac_inlier_ratio=float(inlier_score),
            fit_residual_score=float(residual_score),
            lane_width_consistency=float(width_score),
            homography_stability=float(homography_score),
            raw_confidence=float(raw_score),
            calibrated_confidence=float(calibrated),
        )

    def _compute_edge_alignment(
        self,
        edge_map: Optional[np.ndarray],
        sobel_x: Optional[np.ndarray],
        sobel_y: Optional[np.ndarray],
        roi_mask: Optional[np.ndarray],
    ) -> float:
        if edge_map is None or sobel_x is None or sobel_y is None:
            return 0.50

        if roi_mask is not None:
            active_edges = (edge_map > 0) & (roi_mask > 0)
        else:
            active_edges = edge_map > 0

        num_edges = int(np.sum(active_edges))
        if num_edges < 30:
            return 0.15

        # Compute gradient orientation at edge pixels
        gx = sobel_x[active_edges].astype(np.float32)
        gy = sobel_y[active_edges].astype(np.float32)
        magnitudes = np.hypot(gx, gy) + 1e-6
        angles_rad = np.arctan2(np.abs(gy), np.abs(gx))  # 0 to pi/2

        # Lane lines in prospective projection are predominantly vertical-leaning
        # Angles near pi/2 (steep dy) or pi/4 are typical; pure horizontal (0) is abnormal
        steepness = angles_rad / (math.pi / 2.0)
        mean_steepness = float(np.mean(steepness))
        mag_coherence = float(np.clip(np.mean(magnitudes) / 128.0, 0.0, 1.0))

        alignment_score = 0.70 * mean_steepness + 0.30 * mag_coherence
        return float(np.clip(alignment_score, 0.0, 1.0))

    def _compute_color_contrast(
        self, hls_image: Optional[np.ndarray], edge_map: Optional[np.ndarray]
    ) -> float:
        if hls_image is None or edge_map is None:
            return 0.50

        l_channel = hls_image[:, :, 1]
        lane_pixels = l_channel[edge_map > 0]
        road_pixels = l_channel[edge_map == 0]

        if len(lane_pixels) < 20 or len(road_pixels) < 100:
            return 0.40

        mean_lane_lum = float(np.mean(lane_pixels))
        mean_road_lum = float(np.mean(road_pixels))

        if mean_road_lum < 1.0:
            mean_road_lum = 1.0

        # Michelson / Weber contrast formulation
        contrast = (mean_lane_lum - mean_road_lum) / (mean_lane_lum + mean_road_lum + 1e-4)
        contrast_score = np.clip((contrast - 0.05) / 0.40, 0.0, 1.0)
        return float(contrast_score)

    def _compute_inlier_score(self, inliers: int, total_candidates: int) -> float:
        if total_candidates <= 0:
            return 0.10
        ratio = float(inliers) / float(total_candidates)
        # Rescale: ratio from 0.30 to 0.85 maps to 0.0 -> 1.0
        score = (ratio - self.config.min_inlier_ratio_floor) / (0.85 - self.config.min_inlier_ratio_floor)
        return float(np.clip(score, 0.0, 1.0))

    def _compute_residual_score(self, residual_px: float) -> float:
        if residual_px <= 0.0:
            return 1.0
        # Exponential decay on fit residual
        decay = math.exp(-residual_px / self.config.max_acceptable_residual_px)
        return float(np.clip(decay, 0.0, 1.0))

    def _compute_lane_width_score(self, measured_width_m: float) -> float:
        if measured_width_m <= 0.0:
            return 0.20
        delta = abs(measured_width_m - self.config.nominal_lane_width_m)
        if delta <= 0.15:
            return 1.0
        score = 1.0 - (delta / self.config.lane_width_tolerance_m)
        return float(np.clip(score, 0.0, 1.0))

    def _calibrate_classical_score(self, score: float) -> float:
        # Standard logistical sigmoid calibration centered at 0.50
        scaled = 1.0 / (1.0 + math.exp(-6.0 * (score - 0.45)))
        return float(np.clip(scaled, 0.02, 0.98))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Deep Learning Confidence Estimator
# ═══════════════════════════════════════════════════════════════════════════════

class DeepLearningConfidenceEstimator:
    """
    Evaluates semantic segmentation tensors (4-class: background, corridor, left, right)
    derived from ErfNet, U-Net, and ensemble outputs.
    """

    def __init__(self, config: Optional[DeepLearningConfidenceConfig] = None) -> None:
        self.config = config or DeepLearningConfidenceConfig()

    def estimate_confidence(
        self,
        probability_tensor: np.ndarray,
        argmax_mask: np.ndarray,
        model_entropy_map: Optional[np.ndarray] = None,
        ensemble_variance_map: Optional[np.ndarray] = None,
    ) -> DeepLearningConfidenceMetrics:
        """
        Evaluate neural output confidence from logits / probability distributions.
        Args:
            probability_tensor: Shape (4, H, W) float32 softmax probabilities.
            argmax_mask: Shape (H, W) int32 class IDs (0..3).
            model_entropy_map: Optional (H, W) float32 entropy map.
            ensemble_variance_map: Optional (H, W) float32 cross-model prediction variance.
        """
        assert probability_tensor.ndim == 3 and probability_tensor.shape[0] == 4, (
            f"Expected probability tensor of shape (4, H, W), got {probability_tensor.shape}"
        )

        # 1. Softmax probability margin between top-1 and top-2
        margin_score = self._compute_softmax_margin(probability_tensor, argmax_mask)

        # 2. Entropy uncertainty score (inverted so low entropy -> high score)
        entropy_score = self._compute_entropy_score(probability_tensor, model_entropy_map)

        # 3. Spatial continuity of left and right boundary masks
        continuity_score = self._compute_spatial_continuity(argmax_mask)

        # 4. Thickness consistency along boundary segments
        thickness_score = self._compute_thickness_consistency(argmax_mask)

        # 5. Drivable corridor coherence (Class 1 vs boundary Classes 2 and 3)
        corridor_score = self._compute_corridor_coherence(argmax_mask)

        # 6. Ensemble agreement score
        ensemble_score = self._compute_ensemble_agreement(ensemble_variance_map)

        # Weighted combination
        raw_score = (
            self.config.weight_softmax_margin * margin_score
            + self.config.weight_entropy * entropy_score
            + self.config.weight_spatial_continuity * continuity_score
            + self.config.weight_thickness * thickness_score
            + self.config.weight_corridor_coherence * corridor_score
            + self.config.weight_ensemble_agreement * ensemble_score
        )
        raw_score = float(np.clip(raw_score, 0.0, 1.0))

        # Calibrated output
        calibrated = self._calibrate_dl_score(raw_score)

        return DeepLearningConfidenceMetrics(
            mean_softmax_margin=float(margin_score),
            entropy_uncertainty_score=float(entropy_score),
            spatial_continuity_score=float(continuity_score),
            thickness_consistency=float(thickness_score),
            corridor_coherence=float(corridor_score),
            ensemble_agreement=float(ensemble_score),
            raw_confidence=float(raw_score),
            calibrated_confidence=float(calibrated),
        )

    def _compute_softmax_margin(
        self, probs: np.ndarray, argmax_mask: np.ndarray
    ) -> float:
        # Focus on lane boundary and corridor pixels (Classes 1, 2, 3)
        lane_pixels = argmax_mask > 0
        if not np.any(lane_pixels):
            return 0.10

        sorted_probs = np.sort(probs[:, lane_pixels], axis=0)  # Shape (4, N)
        top1 = sorted_probs[3, :]
        top2 = sorted_probs[2, :]
        margins = top1 - top2
        mean_margin = float(np.mean(margins))
        return float(np.clip(mean_margin, 0.0, 1.0))

    def _compute_entropy_score(
        self, probs: np.ndarray, entropy_map: Optional[np.ndarray]
    ) -> float:
        if entropy_map is not None:
            mean_ent = float(np.mean(entropy_map))
        else:
            # Vectorized Shannon entropy over the 4 classes: H = -sum(p * log2(p))
            eps = 1e-7
            ent = -np.sum(probs * np.log2(probs + eps), axis=0)  # Shape (H, W), max 2.0 bits
            mean_ent = float(np.mean(ent))

        # Max entropy for 4 uniform classes is log2(4) = 2.0
        normalized_entropy = mean_ent / 2.0
        score = 1.0 - normalized_entropy
        return float(np.clip(score, 0.0, 1.0))

    def _compute_spatial_continuity(self, mask: np.ndarray) -> float:
        left_mask = (mask == 2).astype(np.uint8)
        right_mask = (mask == 3).astype(np.uint8)

        left_pts = int(np.sum(left_mask))
        right_pts = int(np.sum(right_mask))

        if left_pts < self.config.min_boundary_pixels_threshold and right_pts < self.config.min_boundary_pixels_threshold:
            return 0.15

        scores: List[float] = []
        for b_mask, count in [(left_mask, left_pts), (right_mask, right_pts)]:
            if count < self.config.min_boundary_pixels_threshold:
                continue
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(b_mask, connectivity=8)
            if num_labels <= 1:
                scores.append(0.0)
                continue
            # Largest component area vs total area
            largest_comp_area = float(np.max(stats[1:, cv2.CC_STAT_AREA]))
            continuity_ratio = largest_comp_area / float(count)
            # Penalize excessive fragmented components
            fragmentation_penalty = math.exp(-0.25 * max(0, num_labels - 3))
            scores.append(continuity_ratio * fragmentation_penalty)

        if not scores:
            return 0.20
        return float(np.clip(np.mean(scores), 0.0, 1.0))

    def _compute_thickness_consistency(self, mask: np.ndarray) -> float:
        left_mask = (mask == 2).astype(np.uint8)
        right_mask = (mask == 3).astype(np.uint8)
        boundary_mask = left_mask | right_mask

        if np.sum(boundary_mask) < self.config.min_boundary_pixels_threshold:
            return 0.25

        # Row-wise thickness measurement
        row_sums = np.sum(boundary_mask, axis=1)
        active_rows = row_sums[row_sums > 0]

        if len(active_rows) < 10:
            return 0.30

        mean_thick = float(np.mean(active_rows))
        std_thick = float(np.std(active_rows))

        # Deviation from nominal expected thickness
        thickness_error = abs(mean_thick - self.config.expected_nominal_thickness_px)
        stability_score = math.exp(-0.15 * thickness_error) * math.exp(-0.20 * std_thick)
        return float(np.clip(stability_score, 0.0, 1.0))

    def _compute_corridor_coherence(self, mask: np.ndarray) -> float:
        corridor_mask = (mask == 1).astype(np.uint8)
        corridor_pts = int(np.sum(corridor_mask))

        if corridor_pts < 500:
            return 0.30

        # Corridor should be a cohesive polygon without wild voids
        h, w = mask.shape
        row_widths = np.sum(corridor_mask, axis=1)
        valid_rows = row_widths[row_widths > 0]

        if len(valid_rows) < h // 4:
            return 0.40

        # Measure row width variance; in perspective, width should taper smoothly from bottom to top
        smoothness = 1.0 / (1.0 + float(np.std(valid_rows)) / (float(np.mean(valid_rows)) + 1e-4))
        return float(np.clip(smoothness, 0.10, 1.0))

    def _compute_ensemble_agreement(self, variance_map: Optional[np.ndarray]) -> float:
        if variance_map is None:
            return 0.90  # Default high score if single model evaluated
        mean_var = float(np.mean(variance_map))
        score = math.exp(-3.0 * mean_var)
        return float(np.clip(score, 0.05, 1.0))

    def _calibrate_dl_score(self, score: float) -> float:
        # Sharp logistic curve for neural segmentation
        scaled = 1.0 / (1.0 + math.exp(-8.0 * (score - 0.40)))
        return float(np.clip(scaled, 0.01, 0.99))


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Master Confidence Fusion & Arbitration Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LaneConfidenceFusionEstimator:
    """
    Arbitrates between classical computer vision and deep learning perception outputs.
    Calculates dynamic fusion weight alpha:
        hat{L} = alpha * L_{DL} + (1 - alpha) * L_{CV}
    where alpha = f(scene, C_{DL}, C_{CV}).
    """

    def __init__(self, config: Optional[ConfidenceFusionConfig] = None) -> None:
        self.config = config or ConfidenceFusionConfig()
        self.classical_estimator = ClassicalConfidenceEstimator(self.config.classical_config)
        self.dl_estimator = DeepLearningConfidenceEstimator(self.config.dl_config)

    def evaluate_and_fuse(
        self,
        # Classical parameters
        edge_map: Optional[np.ndarray],
        sobel_x: Optional[np.ndarray],
        sobel_y: Optional[np.ndarray],
        ransac_inliers_count: int,
        total_candidate_points: int,
        mean_poly_residual_px: float,
        measured_lane_width_m: float,
        # Deep learning parameters
        dl_probability_tensor: np.ndarray,
        dl_argmax_mask: np.ndarray,
        # Contextual parameters
        scene_tag: str = "HIGHWAY",
        ipm_homography_valid: bool = True,
        roi_mask: Optional[np.ndarray] = None,
        hls_image: Optional[np.ndarray] = None,
        model_entropy_map: Optional[np.ndarray] = None,
        ensemble_variance_map: Optional[np.ndarray] = None,
    ) -> ConfidenceFusionReport:
        """
        Produce a unified confidence arbitration report and dynamic fusion weights.
        """
        t_start = time.perf_counter()

        # 1. Classical confidence estimation
        cv_metrics = self.classical_estimator.estimate_confidence(
            edge_map=edge_map,
            sobel_x=sobel_x,
            sobel_y=sobel_y,
            ransac_inliers_count=ransac_inliers_count,
            total_candidate_points=total_candidate_points,
            mean_poly_residual_px=mean_poly_residual_px,
            measured_lane_width_m=measured_lane_width_m,
            ipm_homography_valid=ipm_homography_valid,
            roi_mask=roi_mask,
            hls_image=hls_image,
        )

        # 2. Deep learning confidence estimation
        dl_metrics = self.dl_estimator.estimate_confidence(
            probability_tensor=dl_probability_tensor,
            argmax_mask=dl_argmax_mask,
            model_entropy_map=model_entropy_map,
            ensemble_variance_map=ensemble_variance_map,
        )

        # 3. Scene-adaptive prior resolution
        prior_alpha = self.config.scene_alpha_priors.get(scene_tag.upper(), 0.70)

        # 4. Compute dynamic alpha fusion weight
        c_cv = cv_metrics.calibrated_confidence
        c_dl = dl_metrics.calibrated_confidence

        # Relative superiority formulation
        eps = 1e-5
        relative_dl_advantage = c_dl / (c_dl + c_cv + eps)

        # Blend prior with current empirical performance
        alpha_raw = 0.40 * prior_alpha + 0.60 * relative_dl_advantage

        # Adverse condition overrides
        if scene_tag.upper() in ("RAIN", "NIGHT", "TUNNEL"):
            # Deep learning is substantially more robust under altered illumination
            alpha_raw = max(alpha_raw, 0.75)

        alpha = float(np.clip(
            alpha_raw,
            self.config.min_alpha_dl_floor,
            self.config.max_alpha_dl_ceiling,
        ))
        classical_w = 1.0 - alpha

        # 5. Composite fused confidence score
        fused_confidence = float(alpha * c_dl + classical_w * c_cv)

        # 6. Mode determination and degradation check
        degradation_flag = fused_confidence < self.config.degradation_confidence_threshold

        if degradation_flag:
            mode = DominantStreamMode.SAFETY_INTERLOCK
            reasoning = f"Fused confidence {fused_confidence:.3f} below ASIL-B threshold 0.35"
        elif alpha >= 0.80:
            mode = DominantStreamMode.PURE_DEEP_LEARNING
            reasoning = f"Deep learning dominance (alpha={alpha:.2f}) under scene {scene_tag}"
        elif alpha <= 0.25:
            mode = DominantStreamMode.CLASSICAL_FALLBACK
            reasoning = f"Classical CV dominance (1-alpha={classical_w:.2f}) due to high edge evidence"
        else:
            mode = DominantStreamMode.BALANCED_HYBRID
            reasoning = f"Hybrid fusion balanced (alpha={alpha:.2f}, 1-alpha={classical_w:.2f})"

        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return ConfidenceFusionReport(
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            fusion_alpha_dl_weight=alpha,
            classical_weight=classical_w,
            dominant_mode=mode,
            fused_confidence=fused_confidence,
            classical_metrics=cv_metrics,
            dl_metrics=dl_metrics,
            scene_tag=scene_tag.upper(),
            scene_bias_factor=prior_alpha,
            degradation_flag=degradation_flag,
            diagnostic_reasoning=reasoning,
            computation_latency_ms=t_elapsed_ms,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Global Singleton Accessor & Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_FUSION_ESTIMATOR: Optional[LaneConfidenceFusionEstimator] = None


def get_lane_confidence_estimator(
    config: Optional[ConfidenceFusionConfig] = None,
) -> LaneConfidenceFusionEstimator:
    """Retrieve or initialize the global LaneConfidenceFusionEstimator singleton."""
    global _GLOBAL_FUSION_ESTIMATOR
    if _GLOBAL_FUSION_ESTIMATOR is None:
        _GLOBAL_FUSION_ESTIMATOR = LaneConfidenceFusionEstimator(config=config)
    return _GLOBAL_FUSION_ESTIMATOR


def run_confidence_estimator_diagnostics(verbose: bool = True) -> bool:
    """
    Execute exhaustive unit validation for LaneConfidenceFusionEstimator across
    ideal, noisy, adverse, and degraded conditions.
    """
    if verbose:
        logger.info("Executing LaneConfidenceFusionEstimator diagnostics...")

    estimator = LaneConfidenceFusionEstimator()

    # 1. Test clean highway scenario
    h, w = 360, 640
    dummy_edge = np.zeros((h, w), dtype=np.uint8)
    dummy_edge[200:350, 200:205] = 255  # Left
    dummy_edge[200:350, 435:440] = 255  # Right

    dummy_sx = np.zeros((h, w), dtype=np.float32)
    dummy_sx[dummy_edge > 0] = 80.0
    dummy_sy = np.zeros((h, w), dtype=np.float32)
    dummy_sy[dummy_edge > 0] = 120.0

    dummy_hls = np.full((h, w, 3), 60, dtype=np.uint8)
    dummy_hls[dummy_edge > 0, 1] = 210  # Bright lines

    # DL synthetic prediction
    dummy_probs = np.full((4, h, w), 0.05, dtype=np.float32)
    dummy_probs[0, :, :] = 0.85  # Background default
    dummy_probs[1, 200:350, 205:435] = 0.80  # Corridor
    dummy_probs[2, 200:350, 198:205] = 0.90  # Left
    dummy_probs[3, 200:350, 435:442] = 0.90  # Right

    dummy_mask = np.argmax(dummy_probs, axis=0).astype(np.int32)

    report_clean = estimator.evaluate_and_fuse(
        edge_map=dummy_edge,
        sobel_x=dummy_sx,
        sobel_y=dummy_sy,
        ransac_inliers_count=280,
        total_candidate_points=320,
        mean_poly_residual_px=1.2,
        measured_lane_width_m=3.70,
        dl_probability_tensor=dummy_probs,
        dl_argmax_mask=dummy_mask,
        scene_tag="HIGHWAY",
        hls_image=dummy_hls,
    )

    assert report_clean.fused_confidence > 0.70, f"Expected high confidence, got {report_clean.fused_confidence}"
    assert report_clean.degradation_flag is False
    assert 0.40 <= report_clean.fusion_alpha_dl_weight <= 0.85

    # 2. Test Rain scenario where classical CV is degraded
    report_rain = estimator.evaluate_and_fuse(
        edge_map=None,
        sobel_x=None,
        sobel_y=None,
        ransac_inliers_count=10,
        total_candidate_points=250,
        mean_poly_residual_px=9.5,
        measured_lane_width_m=2.20,
        dl_probability_tensor=dummy_probs,
        dl_argmax_mask=dummy_mask,
        scene_tag="RAIN",
    )

    assert report_rain.fusion_alpha_dl_weight >= 0.75, "Expected strong DL shift during RAIN"
    assert report_rain.classical_metrics.calibrated_confidence < 0.35

    # 3. Test total degradation interlock
    bad_probs = np.full((4, h, w), 0.25, dtype=np.float32)
    bad_mask = np.zeros((h, w), dtype=np.int32)
    report_bad = estimator.evaluate_and_fuse(
        edge_map=None,
        sobel_x=None,
        sobel_y=None,
        ransac_inliers_count=0,
        total_candidate_points=0,
        mean_poly_residual_px=20.0,
        measured_lane_width_m=0.0,
        dl_probability_tensor=bad_probs,
        dl_argmax_mask=bad_mask,
        scene_tag="TUNNEL",
    )

    assert report_bad.degradation_flag is True, "Expected degradation flag tripped"
    assert report_bad.dominant_mode == DominantStreamMode.SAFETY_INTERLOCK

    if verbose:
        logger.info("All LaneConfidenceFusionEstimator diagnostics passed successfully!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_confidence_estimator_diagnostics(verbose=True)
