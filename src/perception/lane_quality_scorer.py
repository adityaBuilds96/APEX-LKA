"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

src/perception/lane_quality_scorer.py
Real-Time Lane Perception Quality Scorer & Graceful Degradation Interlock

Computes an end-to-end perception quality metric Q in [0, 1] incorporating
pixel density within RANSAC inliers, residual fit error, temporal width
stability against rolling history, scene classifier reliability, and
deep learning ensemble concordance. Manages an ASIL-B aligned graceful
degradation interlock state machine that trips when Q < 0.35 for 5+ consecutive
frames, disengaging steering commands and triggering driver takeover alerts.
"""

from __future__ import annotations

import collections
import logging
import math
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger("apex_vision.perception.lane_quality_scorer")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Enumerations & Data Structures
# ═══════════════════════════════════════════════════════════════════════════════

class QualityDegradationState(str, Enum):
    """Graceful degradation lifecycle states aligned with ISO 26262 ASIL-B."""
    NOMINAL = "NOMINAL"                   # Full LKA steering authority active
    CAUTIONARY = "CAUTIONARY"             # Minor quality dip; maintain tracking
    DEGRADED_ADAPTIVE = "DEGRADED_ADAPTIVE"  # Quality < 0.50; widen ROI, boost smoothing
    INTERLOCK_TRIPPED = "INTERLOCK_TRIPPED"  # Q < 0.35 for 5+ frames; disengage steering
    DRIVER_TAKEOVER = "DRIVER_TAKEOVER"   # Acoustic / visual takeover alert dispatched


class DegradationInterlockAction(str, Enum):
    """Mitigation actions executed by the safety interlock supervisor."""
    NONE = "NONE"
    WIDEN_ROI_WINDOW = "WIDEN_ROI_WINDOW"
    FALLBACK_CLASSICAL_CV = "FALLBACK_CLASSICAL_CV"
    CLAMP_STEERING_TORQUE = "CLAMP_STEERING_TORQUE"
    DISENGAGE_ACTUATION = "DISENGAGE_ACTUATION"
    DISENGAGE_ARDUINO_RELAY = "DISENGAGE_ARDUINO_RELAY"


@dataclass(frozen=True)
class QualitySubMetrics:
    """Detailed breakdown of individual constituent quality scores [0.0, 1.0]."""
    inlier_pixel_density_score: float
    poly_fit_residual_score: float
    lane_width_stability_score: float
    scene_classifier_score: float
    ensemble_agreement_score: float
    raw_composite_quality: float
    calibrated_quality_score: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "inlier_pixel_density_score": round(self.inlier_pixel_density_score, 4),
            "poly_fit_residual_score": round(self.poly_fit_residual_score, 4),
            "lane_width_stability_score": round(self.lane_width_stability_score, 4),
            "scene_classifier_score": round(self.scene_classifier_score, 4),
            "ensemble_agreement_score": round(self.ensemble_agreement_score, 4),
            "raw_composite_quality": round(self.raw_composite_quality, 4),
            "calibrated_quality_score": round(self.calibrated_quality_score, 4),
        }


@dataclass
class LaneQualityAssessment:
    """
    Comprehensive per-frame lane perception quality evaluation and interlock state.
    """
    frame_index: int
    timestamp_utc: str
    quality_score: float
    is_safe_for_control: bool
    degradation_state: QualityDegradationState
    recommended_action: DegradationInterlockAction
    consecutive_low_quality_frames: int
    trip_counter_threshold: int
    sub_metrics: QualitySubMetrics
    measured_width_m: float
    history_mean_width_m: float
    width_standard_deviation_m: float
    takeover_alert_active: bool
    arduino_relay_trip_active: bool
    processing_latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "timestamp_utc": self.timestamp_utc,
            "quality_score": round(self.quality_score, 4),
            "is_safe_for_control": self.is_safe_for_control,
            "degradation_state": self.degradation_state.value,
            "recommended_action": self.recommended_action.value,
            "consecutive_low_quality_frames": self.consecutive_low_quality_frames,
            "trip_counter_threshold": self.trip_counter_threshold,
            "sub_metrics": self.sub_metrics.to_dict(),
            "measured_width_m": round(self.measured_width_m, 3),
            "history_mean_width_m": round(self.history_mean_width_m, 3),
            "width_standard_deviation_m": round(self.width_standard_deviation_m, 3),
            "takeover_alert_active": self.takeover_alert_active,
            "arduino_relay_trip_active": self.arduino_relay_trip_active,
            "processing_latency_ms": round(self.processing_latency_ms, 3),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Configuration Dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class QualityScorerConfig:
    """Parameters governing perception quality scoring and interlock thresholds."""
    # Sub-metric weight distribution (sum = 1.0)
    weight_inlier_density: float = 0.25
    weight_poly_residual: float = 0.20
    weight_width_stability: float = 0.25
    weight_scene_confidence: float = 0.15
    weight_ensemble_agreement: float = 0.15

    # Nominal physical road constants
    nominal_lane_width_m: float = 3.70
    max_acceptable_width_deviation_m: float = 0.85
    min_nominal_inlier_density_px_per_m: float = 12.0
    nominal_corridor_length_m: float = 30.0
    max_residual_error_px: float = 6.5

    # Interlock degradation thresholds
    nominal_quality_threshold: float = 0.70
    cautionary_quality_threshold: float = 0.50
    interlock_trip_threshold: float = 0.35
    consecutive_low_quality_limit: int = 5
    recovery_hysteresis_frames: int = 10
    recovery_quality_threshold: float = 0.60

    # History buffers
    width_history_capacity: int = 30
    quality_history_capacity: int = 60


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Core Quality Scorer & Interlock Engine
# ═══════════════════════════════════════════════════════════════════════════════

class LaneQualityScorer:
    """
    Evaluates multi-source lane perception quality and enforces safety interlocks.

    Standards Compliance:
        - APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
        - Driver-supervised SAE Level 2 execution
        - ISO 26262-style process; ASIL-B aligned; not independently assessed
    """

    def __init__(self, config: Optional[QualityScorerConfig] = None) -> None:
        self.config = config or QualityScorerConfig()
        self._frame_count: int = 0
        self._consecutive_low_frames: int = 0
        self._consecutive_recovery_frames: int = 0
        self._current_state: QualityDegradationState = QualityDegradationState.NOMINAL
        self._relay_tripped: bool = False
        self._takeover_alert: bool = False

        # Rolling history buffers
        self._width_history: Deque[float] = collections.deque(
            maxlen=self.config.width_history_capacity
        )
        self._quality_history: Deque[float] = collections.deque(
            maxlen=self.config.quality_history_capacity
        )

        self._lock = threading.Lock()

    def evaluate_quality(
        self,
        inlier_pixel_count: int,
        mean_polynomial_residual_px: float,
        measured_lane_width_m: float,
        scene_classifier_confidence: float,
        ensemble_agreement_score: float,
        corridor_longitudinal_span_m: float = 30.0,
    ) -> LaneQualityAssessment:
        """
        Evaluate perception quality Q for current frame and update state machine.

        Args:
            inlier_pixel_count: Number of RANSAC inlier pixels supporting polynomials.
            mean_polynomial_residual_px: Fitting residual error in pixels.
            measured_lane_width_m: Distance between left and right polynomials at base.
            scene_classifier_confidence: Softmax confidence of current scene classification.
            ensemble_agreement_score: Multi-model DL ensemble concordance [0, 1].
            corridor_longitudinal_span_m: Evaluated road distance in meters.
        """
        t_start = time.perf_counter()

        with self._lock:
            self._frame_count += 1
            idx = self._frame_count

            # 1. Compute inlier pixel density score
            inlier_density_score = self._compute_inlier_density_score(
                inlier_pixel_count, corridor_longitudinal_span_m
            )

            # 2. Compute polynomial residual score
            residual_score = self._compute_residual_score(mean_polynomial_residual_px)

            # 3. Compute lane width stability score
            width_score, mean_w, std_w = self._compute_lane_width_stability_score(
                measured_lane_width_m
            )

            # 4. Scene classifier reliability
            scene_score = float(np.clip(scene_classifier_confidence, 0.0, 1.0))

            # 5. Ensemble agreement
            ensemble_score = float(np.clip(ensemble_agreement_score, 0.0, 1.0))

            # Composite weighted raw score
            raw_q = (
                self.config.weight_inlier_density * inlier_density_score
                + self.config.weight_poly_residual * residual_score
                + self.config.weight_width_stability * width_score
                + self.config.weight_scene_confidence * scene_score
                + self.config.weight_ensemble_agreement * ensemble_score
            )
            raw_q = float(np.clip(raw_q, 0.0, 1.0))

            # Calibrated Quality Score
            calibrated_q = self._calibrate_quality_score(raw_q)
            self._quality_history.append(calibrated_q)

            # 6. Update Degradation Interlock State Machine
            state, action, low_count, safe = self._update_interlock_state_machine(calibrated_q)

            sub_metrics = QualitySubMetrics(
                inlier_pixel_density_score=float(inlier_density_score),
                poly_fit_residual_score=float(residual_score),
                lane_width_stability_score=float(width_score),
                scene_classifier_score=float(scene_score),
                ensemble_agreement_score=float(ensemble_score),
                raw_composite_quality=float(raw_q),
                calibrated_quality_score=float(calibrated_q),
            )

            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

            return LaneQualityAssessment(
                frame_index=idx,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                quality_score=float(calibrated_q),
                is_safe_for_control=bool(safe),
                degradation_state=state,
                recommended_action=action,
                consecutive_low_quality_frames=int(low_count),
                trip_counter_threshold=self.config.consecutive_low_quality_limit,
                sub_metrics=sub_metrics,
                measured_width_m=float(measured_lane_width_m),
                history_mean_width_m=float(mean_w),
                width_standard_deviation_m=float(std_w),
                takeover_alert_active=bool(self._takeover_alert),
                arduino_relay_trip_active=bool(self._relay_tripped),
                processing_latency_ms=float(t_elapsed_ms),
            )

    # ───────────────────────────────────────────────────────────────────────────
    # Sub-Metric Calculation Helpers
    # ───────────────────────────────────────────────────────────────────────────

    def _compute_inlier_density_score(
        self, inlier_pixels: int, corridor_length_m: float
    ) -> float:
        if inlier_pixels <= 0 or corridor_length_m <= 0.0:
            return 0.05
        density = float(inlier_pixels) / float(corridor_length_m)
        target = self.config.min_nominal_inlier_density_px_per_m
        score = density / (target * 1.5)
        return float(np.clip(score, 0.05, 1.0))

    def _compute_residual_score(self, residual_px: float) -> float:
        if residual_px <= 0.0:
            return 1.0
        # Exponential roll-off
        score = math.exp(-residual_px / self.config.max_residual_error_px)
        return float(np.clip(score, 0.05, 1.0))

    def _compute_lane_width_stability_score(
        self, measured_width_m: float
    ) -> Tuple[float, float, float]:
        if measured_width_m <= 0.5:
            # Degraded or single line mode
            return 0.10, self.config.nominal_lane_width_m, 1.0

        # Absolute deviation from nominal 3.7m
        abs_err = abs(measured_width_m - self.config.nominal_lane_width_m)
        nominal_score = 1.0 - (abs_err / self.config.max_acceptable_width_deviation_m)
        nominal_score = max(0.0, nominal_score)

        # History-based stability check
        if len(self._width_history) >= 3:
            mean_hist = float(np.mean(self._width_history))
            std_hist = float(np.std(self._width_history))
            hist_err = abs(measured_width_m - mean_hist)
            # Rejection if jump > 2-sigma
            jump_penalty = 1.0 if std_hist < 0.05 else math.exp(-hist_err / (2.0 * max(std_hist, 0.10)))
        else:
            mean_hist = measured_width_m
            std_hist = 0.0
            jump_penalty = 1.0

        self._width_history.append(measured_width_m)

        composite = 0.60 * nominal_score + 0.40 * jump_penalty
        return float(np.clip(composite, 0.05, 1.0)), mean_hist, std_hist

    def _calibrate_quality_score(self, raw_score: float) -> float:
        # ASIL-B conservative calibration: penalize ambiguity steeply below 0.50
        if raw_score >= 0.50:
            calibrated = 0.50 + 0.50 * math.tanh(2.5 * (raw_score - 0.50))
        else:
            calibrated = 0.50 * (raw_score / 0.50) ** 1.35
        return float(np.clip(calibrated, 0.01, 0.99))

    # ───────────────────────────────────────────────────────────────────────────
    # Interlock State Machine
    # ───────────────────────────────────────────────────────────────────────────

    def _update_interlock_state_machine(
        self, quality_score: float
    ) -> Tuple[QualityDegradationState, DegradationInterlockAction, int, bool]:
        """
        Manages transition logic across ISO 26262 ASIL-B degradation states.
        """
        if quality_score < self.config.interlock_trip_threshold:
            self._consecutive_low_frames += 1
            self._consecutive_recovery_frames = 0
        else:
            self._consecutive_low_frames = 0
            if self._relay_tripped or self._current_state in (
                QualityDegradationState.INTERLOCK_TRIPPED,
                QualityDegradationState.DRIVER_TAKEOVER,
            ):
                self._consecutive_recovery_frames += 1

        # Check for Interlock Trip Trigger (Q < 0.35 for 5+ frames)
        if self._consecutive_low_frames >= self.config.consecutive_low_quality_limit:
            self._current_state = QualityDegradationState.INTERLOCK_TRIPPED
            self._relay_tripped = True
            self._takeover_alert = True
            action = DegradationInterlockAction.DISENGAGE_ARDUINO_RELAY
            safe_for_actuation = False
            return self._current_state, action, self._consecutive_low_frames, safe_for_actuation

        # If already tripped, enforce recovery hysteresis
        if self._relay_tripped:
            if (
                self._consecutive_recovery_frames >= self.config.recovery_hysteresis_frames
                and quality_score >= self.config.recovery_quality_threshold
            ):
                # Recovery approved
                logger.info("Quality restored above hysteresis threshold; re-engaging LKA authority.")
                self._relay_tripped = False
                self._takeover_alert = False
                self._current_state = QualityDegradationState.CAUTIONARY
                action = DegradationInterlockAction.CLAMP_STEERING_TORQUE
                safe_for_actuation = True
            else:
                self._current_state = QualityDegradationState.DRIVER_TAKEOVER
                action = DegradationInterlockAction.DISENGAGE_ACTUATION
                safe_for_actuation = False
            return self._current_state, action, self._consecutive_low_frames, safe_for_actuation

        # Sub-trip degradation tiers
        if quality_score < self.config.cautionary_quality_threshold:
            self._current_state = QualityDegradationState.DEGRADED_ADAPTIVE
            action = DegradationInterlockAction.WIDEN_ROI_WINDOW
            safe_for_actuation = True
        elif quality_score < self.config.nominal_quality_threshold:
            self._current_state = QualityDegradationState.CAUTIONARY
            action = DegradationInterlockAction.CLAMP_STEERING_TORQUE
            safe_for_actuation = True
        else:
            self._current_state = QualityDegradationState.NOMINAL
            action = DegradationInterlockAction.NONE
            safe_for_actuation = True

        return self._current_state, action, self._consecutive_low_frames, safe_for_actuation

    def reset_interlock(self) -> None:
        """Manual driver reset of the safety interlock and relay disengagement."""
        with self._lock:
            self._relay_tripped = False
            self._takeover_alert = False
            self._consecutive_low_frames = 0
            self._consecutive_recovery_frames = 0
            self._current_state = QualityDegradationState.NOMINAL
            logger.info("Manual reset executed for LaneQualityScorer safety interlock.")

    def is_relay_tripped(self) -> bool:
        """Query if hardware clutch safety relay is currently tripped."""
        with self._lock:
            return self._relay_tripped


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Predictive Quality Trend Forecasting & Journaling
# ═══════════════════════════════════════════════════════════════════════════════

class QualityTrendForecaster:
    """
    Online autoregressive trend forecaster for perception quality Q.
    Predicts lookahead quality Q(t + tau) for tau in {1..10} frames using
    an adaptive linear regression and exponentially weighted moving average (EWMA).
    Alerts proactive path planning before sudden quality dropouts (e.g. tunnel entry).
    """

    def __init__(self, history_capacity: int = 50, smoothing_alpha: float = 0.35) -> None:
        self.capacity = int(history_capacity)
        self.alpha = float(smoothing_alpha)
        self._history: Deque[float] = collections.deque(maxlen=self.capacity)
        self._ewma: float = 0.85
        self._slope: float = 0.0

    def update(self, quality: float) -> None:
        """Update forecaster state with observed quality sample."""
        q = float(np.clip(quality, 0.0, 1.0))
        self._history.append(q)
        # Update EWMA
        self._ewma = self.alpha * q + (1.0 - self.alpha) * self._ewma

        # Compute empirical slope over last 10 frames
        if len(self._history) >= 5:
            recent = list(self._history)[-10:]
            x = np.arange(len(recent), dtype=np.float32)
            y = np.array(recent, dtype=np.float32)
            denom = float(np.sum((x - np.mean(x)) ** 2))
            if denom > 1e-6:
                self._slope = float(np.sum((x - np.mean(x)) * (y - np.mean(y))) / denom)
            else:
                self._slope = 0.0

    def predict_lookahead(self, lookahead_frames: int = 5) -> float:
        """Forecast quality score tau frames into the future."""
        tau = max(1, int(lookahead_frames))
        pred = self._ewma + self._slope * tau
        return float(np.clip(pred, 0.0, 1.0))

    def is_sharp_deterioration_imminent(self, threshold: float = 0.40) -> bool:
        """Query if quality is projected to breach safety threshold within 5 frames."""
        pred = self.predict_lookahead(5)
        return bool(pred < threshold and self._slope < -0.03)


class LaneQualityAuditJournal:
    """
    Thread-safe circular ring buffer journal of perception quality assessments,
    maintaining an ASIL-B compliant flight-recorder trace of degradation transitions.
    """

    def __init__(self, capacity: int = 2000) -> None:
        self.capacity = int(capacity)
        self._journal: Deque[Dict[str, Any]] = collections.deque(maxlen=self.capacity)
        self._incident_log: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

    def record_assessment(self, assessment: LaneQualityAssessment) -> None:
        """Log a frame assessment into the persistent in-memory ring buffer."""
        record = assessment.to_dict()
        with self._lock:
            self._journal.append(record)
            if assessment.degradation_state in (
                QualityDegradationState.INTERLOCK_TRIPPED,
                QualityDegradationState.DRIVER_TAKEOVER,
            ):
                self._incident_log.append(record)
                if len(self._incident_log) > 500:
                    self._incident_log.pop(0)

    def get_recent_history(self, num_frames: int = 100) -> List[Dict[str, Any]]:
        """Retrieve recent frame assessments for telemetry streaming."""
        with self._lock:
            n = min(num_frames, len(self._journal))
            return list(self._journal)[-n:]

    def get_incidents(self) -> List[Dict[str, Any]]:
        """Retrieve recorded degradation interlock incidents."""
        with self._lock:
            return list(self._incident_log)

    def compute_journal_statistics(self) -> Dict[str, Any]:
        """Aggregate statistical performance metrics across the active flight journal."""
        with self._lock:
            if not self._journal:
                return {
                    "total_recorded_frames": 0,
                    "mean_quality": 0.0,
                    "min_quality": 0.0,
                    "p05_quality": 0.0,
                    "incident_count": 0,
                    "nominal_ratio": 1.0,
                }
            qualities = [item["quality_score"] for item in self._journal]
            nominals = sum(
                1 for item in self._journal
                if item["degradation_state"] == QualityDegradationState.NOMINAL.value
            )
            total = len(qualities)
            return {
                "total_recorded_frames": total,
                "mean_quality": round(float(np.mean(qualities)), 4),
                "min_quality": round(float(np.min(qualities)), 4),
                "p05_quality": round(float(np.percentile(qualities, 5)), 4),
                "incident_count": len(self._incident_log),
                "nominal_ratio": round(float(nominals) / float(total), 4),
            }


class StreamlitHUDQualityBridge:
    """
    Prepares serialized telemetry packages formatted for the Streamlit
    web cockpit HUD and digital twin dashboard.
    """

    def __init__(self, audit_journal: Optional[LaneQualityAuditJournal] = None) -> None:
        self.journal = audit_journal or LaneQualityAuditJournal()

    def format_hud_payload(self, assessment: LaneQualityAssessment) -> Dict[str, Any]:
        """Convert an assessment into a rich HUD display payload."""
        q = assessment.quality_score
        # Determine gauge color
        if q >= 0.70:
            status_color = "#10B981"  # Emerald green
            status_text = "OPTIMAL"
        elif q >= 0.50:
            status_color = "#3B82F6"  # Blue
            status_text = "NOMINAL"
        elif q >= 0.35:
            status_color = "#F59E0B"  # Amber
            status_text = "DEGRADED"
        else:
            status_color = "#EF4444"  # Red
            status_text = "TRIPPED"

        return {
            "hud_version": "4.0",
            "timestamp": assessment.timestamp_utc,
            "quality_gauge": {
                "score": round(q, 3),
                "percentage": int(round(q * 100)),
                "status_color": status_color,
                "status_text": status_text,
            },
            "sub_metrics": {
                "inlier_density": round(assessment.sub_metrics.inlier_pixel_density_score, 2),
                "residual_accuracy": round(assessment.sub_metrics.poly_fit_residual_score, 2),
                "width_stability": round(assessment.sub_metrics.lane_width_stability_score, 2),
                "scene_confidence": round(assessment.sub_metrics.scene_classifier_score, 2),
                "dl_agreement": round(assessment.sub_metrics.ensemble_agreement_score, 2),
            },
            "interlock": {
                "state": assessment.degradation_state.value,
                "relay_tripped": assessment.arduino_relay_trip_active,
                "takeover_alert": assessment.takeover_alert_active,
                "consecutive_low_frames": assessment.consecutive_low_quality_frames,
                "recommended_action": assessment.recommended_action.value,
            },
            "physical_geometry": {
                "measured_width_m": assessment.measured_width_m,
                "history_mean_m": assessment.history_mean_width_m,
                "std_deviation_m": assessment.width_standard_deviation_m,
            },
        }

    def format_can_arbitration_frame(self, assessment: LaneQualityAssessment) -> Dict[str, Any]:
        """
        Encode perception quality and interlock status into 8-byte CAN payload
        for arbitration ID 0x148 (Perception Health Status).
        """
        raw_q_byte = int(round(assessment.quality_score * 255.0))
        tripped_bit = 1 if assessment.arduino_relay_trip_active else 0
        takeover_bit = 1 if assessment.takeover_alert_active else 0
        state_code = {
            QualityDegradationState.NOMINAL: 0x00,
            QualityDegradationState.CAUTIONARY: 0x01,
            QualityDegradationState.DEGRADED_ADAPTIVE: 0x02,
            QualityDegradationState.INTERLOCK_TRIPPED: 0x03,
            QualityDegradationState.DRIVER_TAKEOVER: 0x04,
        }.get(assessment.degradation_state, 0xFF)

        payload_bytes = [
            raw_q_byte & 0xFF,
            state_code & 0xFF,
            ((tripped_bit << 1) | takeover_bit) & 0xFF,
            min(255, assessment.consecutive_low_quality_frames) & 0xFF,
            int(round(assessment.measured_width_m * 50.0)) & 0xFF,  # 0.02m resolution
            0x00, 0x00, 0x00
        ]
        return {
            "arbitration_id": 0x148,
            "dlc": 8,
            "payload_hex": "".join(f"{b:02X}" for b in payload_bytes),
            "payload_bytes": payload_bytes,
            "timestamp": assessment.timestamp_utc,
        }



# ═══════════════════════════════════════════════════════════════════════════════
# 5. Global Singleton & Benchmark Test Harness
# ═══════════════════════════════════════════════════════════════════════════════

_GLOBAL_QUALITY_SCORER: Optional[LaneQualityScorer] = None
_GLOBAL_AUDIT_JOURNAL: Optional[LaneQualityAuditJournal] = None
_GLOBAL_TREND_FORECASTER: Optional[QualityTrendForecaster] = None
_GLOBAL_HUD_BRIDGE: Optional[StreamlitHUDQualityBridge] = None


def get_lane_quality_scorer(
    config: Optional[QualityScorerConfig] = None,
) -> LaneQualityScorer:
    """Retrieve or instantiate global LaneQualityScorer singleton."""
    global _GLOBAL_QUALITY_SCORER
    if _GLOBAL_QUALITY_SCORER is None:
        _GLOBAL_QUALITY_SCORER = LaneQualityScorer(config=config)
    return _GLOBAL_QUALITY_SCORER


def get_quality_audit_journal() -> LaneQualityAuditJournal:
    """Retrieve global LaneQualityAuditJournal singleton."""
    global _GLOBAL_AUDIT_JOURNAL
    if _GLOBAL_AUDIT_JOURNAL is None:
        _GLOBAL_AUDIT_JOURNAL = LaneQualityAuditJournal()
    return _GLOBAL_AUDIT_JOURNAL


def get_quality_trend_forecaster() -> QualityTrendForecaster:
    """Retrieve global QualityTrendForecaster singleton."""
    global _GLOBAL_TREND_FORECASTER
    if _GLOBAL_TREND_FORECASTER is None:
        _GLOBAL_TREND_FORECASTER = QualityTrendForecaster()
    return _GLOBAL_TREND_FORECASTER


def get_streamlit_hud_quality_bridge() -> StreamlitHUDQualityBridge:
    """Retrieve global StreamlitHUDQualityBridge singleton."""
    global _GLOBAL_HUD_BRIDGE
    if _GLOBAL_HUD_BRIDGE is None:
        _GLOBAL_HUD_BRIDGE = StreamlitHUDQualityBridge(audit_journal=get_quality_audit_journal())
    return _GLOBAL_HUD_BRIDGE


@dataclass
class QualityBenchmarkReport:
    """Summary of comprehensive stress testing across 500 synthetic frames."""
    total_frames: int
    mean_quality_score: float
    min_quality_score: float
    nominal_frame_count: int
    cautionary_frame_count: int
    degraded_frame_count: int
    interlock_trip_count: int
    mean_processing_latency_ms: float
    passed_asil_b_validation: bool


def run_lane_quality_stress_benchmark(num_frames: int = 500) -> QualityBenchmarkReport:
    """
    Executes a 500-frame simulated lifecycle stress test across varied operational
    conditions: smooth highway, curve jitter, adverse shadow squalls, tunnel dropouts,
    and recovery sequences.
    """
    scorer = LaneQualityScorer()
    journal = LaneQualityAuditJournal()
    forecaster = QualityTrendForecaster()
    latencies: List[float] = []

    nominal_count = 0
    caution_count = 0
    degraded_count = 0
    trip_count = 0

    np.random.seed(42)

    for i in range(num_frames):
        t0 = time.perf_counter()

        # Phase 1: 0..150 Ideal Highway
        if i < 150:
            inliers = int(np.random.normal(450, 20))
            residual = float(np.random.uniform(0.8, 1.5))
            w = float(np.random.normal(3.70, 0.05))
            scene_conf = float(np.random.uniform(0.92, 0.98))
            dl_agree = float(np.random.uniform(0.90, 0.97))
        # Phase 2: 150..220 Shadows & Curves
        elif i < 220:
            inliers = int(np.random.normal(250, 40))
            residual = float(np.random.uniform(2.5, 4.5))
            w = float(np.random.normal(3.85, 0.15))
            scene_conf = float(np.random.uniform(0.70, 0.85))
            dl_agree = float(np.random.uniform(0.65, 0.80))
        # Phase 3: 220..235 Sudden Tunnel Dropout (should trip after 5 frames)
        elif i < 235:
            inliers = int(np.random.normal(15, 5))
            residual = float(np.random.uniform(10.0, 18.0))
            w = float(np.random.uniform(1.2, 2.0))
            scene_conf = float(np.random.uniform(0.20, 0.35))
            dl_agree = float(np.random.uniform(0.15, 0.30))
        # Phase 4: 235..350 Recovery Sequence
        elif i < 350:
            inliers = int(np.random.normal(420, 25))
            residual = float(np.random.uniform(1.0, 2.0))
            w = float(np.random.normal(3.70, 0.06))
            scene_conf = float(np.random.uniform(0.90, 0.96))
            dl_agree = float(np.random.uniform(0.88, 0.95))
        # Phase 5: 350..500 Urban stop and go
        else:
            inliers = int(np.random.normal(350, 30))
            residual = float(np.random.uniform(1.5, 3.0))
            w = float(np.random.normal(3.65, 0.10))
            scene_conf = float(np.random.uniform(0.85, 0.92))
            dl_agree = float(np.random.uniform(0.80, 0.90))

        assessment = scorer.evaluate_quality(
            inlier_pixel_count=max(0, inliers),
            mean_polynomial_residual_px=max(0.1, residual),
            measured_lane_width_m=max(0.1, w),
            scene_classifier_confidence=scene_conf,
            ensemble_agreement_score=dl_agree,
        )

        lat = (time.perf_counter() - t0) * 1000.0
        latencies.append(lat)
        journal.record_assessment(assessment)
        forecaster.update(assessment.quality_score)

        if assessment.degradation_state == QualityDegradationState.NOMINAL:
            nominal_count += 1
        elif assessment.degradation_state == QualityDegradationState.CAUTIONARY:
            caution_count += 1
        elif assessment.degradation_state == QualityDegradationState.DEGRADED_ADAPTIVE:
            degraded_count += 1
        elif assessment.degradation_state in (
            QualityDegradationState.INTERLOCK_TRIPPED,
            QualityDegradationState.DRIVER_TAKEOVER,
        ):
            trip_count += 1

    stats = journal.compute_journal_statistics()
    passed = trip_count > 0 and stats["mean_quality"] > 0.65 and np.mean(latencies) < 2.0

    return QualityBenchmarkReport(
        total_frames=num_frames,
        mean_quality_score=stats["mean_quality"],
        min_quality_score=stats["min_quality"],
        nominal_frame_count=nominal_count,
        cautionary_frame_count=caution_count,
        degraded_frame_count=degraded_count,
        interlock_trip_count=trip_count,
        mean_processing_latency_ms=float(np.mean(latencies)),
        passed_asil_b_validation=bool(passed),
    )


def run_quality_scorer_diagnostics(verbose: bool = True) -> bool:
    """
    Validate LaneQualityScorer multi-criteria scoring, hysteresis recovery,
    and 5-frame ASIL-B degradation interlock tripping.
    """
    if verbose:
        logger.info("Executing LaneQualityScorer diagnostics...")

    scorer = LaneQualityScorer()

    # 1. Ideal frame assessment
    res_ideal = scorer.evaluate_quality(
        inlier_pixel_count=450,
        mean_polynomial_residual_px=1.2,
        measured_lane_width_m=3.70,
        scene_classifier_confidence=0.96,
        ensemble_agreement_score=0.94,
        corridor_longitudinal_span_m=30.0,
    )
    assert res_ideal.quality_score >= 0.75, f"Expected high quality, got {res_ideal.quality_score}"
    assert res_ideal.degradation_state == QualityDegradationState.NOMINAL
    assert res_ideal.is_safe_for_control is True
    assert res_ideal.arduino_relay_trip_active is False

    # 2. Moderate degradation (wide lane + residual error)
    res_mod = scorer.evaluate_quality(
        inlier_pixel_count=180,
        mean_polynomial_residual_px=4.8,
        measured_lane_width_m=4.20,
        scene_classifier_confidence=0.72,
        ensemble_agreement_score=0.68,
        corridor_longitudinal_span_m=30.0,
    )
    assert 0.40 <= res_mod.quality_score < 0.75
    assert res_mod.is_safe_for_control is True

    # 3. Simulate 5 consecutive severely degraded frames (Q < 0.35)
    for i in range(1, 6):
        res_deg = scorer.evaluate_quality(
            inlier_pixel_count=15,
            mean_polynomial_residual_px=15.0,
            measured_lane_width_m=1.20,
            scene_classifier_confidence=0.30,
            ensemble_agreement_score=0.20,
            corridor_longitudinal_span_m=30.0,
        )

    # After 5th frame, interlock must trip
    assert res_deg.consecutive_low_quality_frames == 5
    assert res_deg.degradation_state == QualityDegradationState.INTERLOCK_TRIPPED
    assert res_deg.is_safe_for_control is False
    assert res_deg.arduino_relay_trip_active is True
    assert res_deg.takeover_alert_active is True
    assert res_deg.recommended_action == DegradationInterlockAction.DISENGAGE_ARDUINO_RELAY

    # 4. Test recovery hysteresis: require 10 frames > 0.60 to clear
    for i in range(9):
        res_rec = scorer.evaluate_quality(
            inlier_pixel_count=400,
            mean_polynomial_residual_px=1.5,
            measured_lane_width_m=3.70,
            scene_classifier_confidence=0.95,
            ensemble_agreement_score=0.92,
            corridor_longitudinal_span_m=30.0,
        )
        assert res_rec.arduino_relay_trip_active is True, "Relay should remain tripped before 10 frames"

    # 10th recovery frame clears the trip
    res_recovered = scorer.evaluate_quality(
        inlier_pixel_count=400,
        mean_polynomial_residual_px=1.5,
        measured_lane_width_m=3.70,
        scene_classifier_confidence=0.95,
        ensemble_agreement_score=0.92,
        corridor_longitudinal_span_m=30.0,
    )
    assert res_recovered.arduino_relay_trip_active is False
    assert res_recovered.is_safe_for_control is True

    if verbose:
        logger.info("All LaneQualityScorer diagnostics passed successfully!")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_quality_scorer_diagnostics(verbose=True)
    report = run_lane_quality_stress_benchmark(num_frames=500)
    print("=" * 80)
    print("APEX-Vision Lane Quality Scorer Stress Benchmark Results")
    print(f"Total Frames Tested: {report.total_frames}")
    print(f"Mean Quality Score: {report.mean_quality_score:.4f}")
    print(f"Min Quality Score: {report.min_quality_score:.4f}")
    print(f"Nominal Frames: {report.nominal_frame_count} ({report.nominal_frame_count/report.total_frames*100:.1f}%)")
    print(f"Cautionary Frames: {report.cautionary_frame_count}")
    print(f"Degraded Adaptive Frames: {report.degraded_frame_count}")
    print(f"Interlock Trip Frames: {report.interlock_trip_count}")
    print(f"Mean Latency: {report.mean_processing_latency_ms:.3f} ms")
    print(f"Passed ASIL-B Validation: {report.passed_asil_b_validation}")
    print("=" * 80)

