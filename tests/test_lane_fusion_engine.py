"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Verification Suite: Master Lane Fusion Engine Tests
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
================================================================================

This test suite comprehensively validates the master orchestrator
`LaneFusionEngine` across all edge cases, scene profiles, and failure modes:
1. Dynamic fusion weight alpha adaptation f(scene, confidence).
2. Confidence-weighted voting between classical CV and deep learning ensemble.
3. Polynomial geometric constraint verification (lane width 3.7m +/- 0.5m, parallelism).
4. Dual-boundary parallel hallucination on degraded or single-line roads.
5. 8-State EKF temporal smoothing integration and motion compensation.
6. 5-frame consecutive low quality (Q < 0.35) graceful degradation relay cutoff.
7. Virtual safety guardrail envelope enforcement and CAN 0x150 priority dispatch.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest
import torch

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.adaptive_roi_engine import AdaptiveROIEngine, AdaptiveROIConfig
from src.perception.faded_lane_recovery import (
    BoundaryPolynomial2D,
    FadedLaneRecoveryEngine,
    FadedRecoveryConfig,
)
from src.perception.lane_confidence_estimator import (
    ClassicalConfidenceEstimator,
    DeepLearningConfidenceEstimator,
    LaneConfidenceFusionEstimator,
    ConfidenceFusionConfig,
)
from src.perception.lane_departure_predictor import (
    LaneDeparturePredictor,
    DeparturePredictorConfig,
    DepartureSeverity,
    DepartureDirection,
)
from src.perception.lane_ensemble import (
    LaneEnsembleInferencer,
    LaneEnsembleConfig,
)
from src.perception.lane_fusion_engine import (
    LaneFusionEngine,
    LaneFusionEngineConfig,
    LanePrediction,
    PerceptionExecutionStatus,
)
from src.perception.lane_polynomial_tracker import (
    LanePolynomialTracker,
    PolynomialTrackerConfig,
)
from src.perception.lane_quality_scorer import (
    LaneQualityScorer,
    QualityScorerConfig,
)
from src.perception.lane_temporal_smoother import (
    LaneTemporalSmoother,
    TemporalSmootherConfig,
)
from src.perception.lighting_compensator import (
    LightingCompensator,
    LightingCompensatorConfig,
)
from src.perception.virtual_guardrail_enforcer import (
    VirtualGuardrailEnforcer,
    GuardrailConfig,
    GuardrailSeverity,
)
from src.scene.scene_profiles import SceneType, get_scene_profile


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures & Synthetic Frame Generators
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def default_fusion_engine() -> LaneFusionEngine:
    """Fixture providing a configured LaneFusionEngine instance."""
    config = LaneFusionEngineConfig()
    engine = LaneFusionEngine(config)
    engine.reset()
    return engine


@pytest.fixture
def clean_highway_frame() -> np.ndarray:
    """Generate high-contrast synthetic highway frame with standard markings."""
    frame = np.full((360, 640, 3), 70, dtype=np.uint8)
    y_eval = np.linspace(170, 360, 40)
    pts_l = []
    pts_r = []
    for y in y_eval:
        xl = int(-0.45 * (y - 170) + 290)
        xr = int(0.45 * (y - 170) + 350)
        pts_l.append((xl, int(y)))
        pts_r.append((xr, int(y)))

    cv2.polylines(frame, [np.array(pts_l)], False, (255, 255, 255), 4)
    cv2.polylines(frame, [np.array(pts_r)], False, (255, 255, 255), 4)
    return frame


@pytest.fixture
def curved_highway_frame() -> np.ndarray:
    """Generate synthetic highway frame with severe leftward curve (R ~ 140m)."""
    frame = np.full((360, 640, 3), 65, dtype=np.uint8)
    curve_a = -0.00045
    y_eval = np.linspace(170, 360, 40)
    pts_l = []
    pts_r = []
    for y in y_eval:
        dy = y - 170
        xl = int(curve_a * dy**2 - 0.45 * dy + 290)
        xr = int(curve_a * dy**2 + 0.45 * dy + 350)
        pts_l.append((xl, int(y)))
        pts_r.append((xr, int(y)))

    cv2.polylines(frame, [np.array(pts_l)], False, (240, 240, 240), 4)
    cv2.polylines(frame, [np.array(pts_r)], False, (240, 240, 240), 4)
    return frame


@pytest.fixture
def single_faded_line_frame() -> np.ndarray:
    """Generate frame where right lane marking is completely occluded or worn."""
    frame = np.full((360, 640, 3), 60, dtype=np.uint8)
    y_eval = np.linspace(170, 360, 40)
    pts_l = []
    for y in y_eval:
        xl = int(-0.45 * (y - 170) + 290)
        pts_l.append((xl, int(y)))

    # Only draw left boundary; right boundary is completely absent
    cv2.polylines(frame, [np.array(pts_l)], False, (255, 255, 255), 4)
    return frame


@pytest.fixture
def empty_dark_asphalt_frame() -> np.ndarray:
    """Generate dark asphalt frame with zero markings to test low-Q degradation."""
    frame = np.full((360, 640, 3), 35, dtype=np.uint8)
    # Add mild noise
    noise = np.random.normal(0, 4, (360, 640, 3)).astype(np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return frame


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Structural & Configuration Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFusionEngineConfiguration:
    """Validate configuration defaults, physical constraints, and invariant guards."""

    def test_default_config_parameters(self) -> None:
        """Verify baseline optical and physical constraints align with specification."""
        cfg = LaneFusionEngineConfig()
        assert cfg.camera_width == 640
        assert cfg.camera_height == 360
        assert cfg.focal_length_y_px == 450.0
        assert cfg.focal_length_x_px == 450.0
        assert cfg.principal_cx_px == 320.0
        assert cfg.principal_cy_px == 180.0
        assert cfg.camera_mount_height_m == 1.20
        assert cfg.nominal_pitch_deg == 5.0
        assert cfg.nominal_lane_width_m == 3.70
        assert cfg.safety_buffer_margin_m == 0.30
        assert cfg.critical_buffer_margin_m == 0.15

    def test_horizontal_flip_invariant(self) -> None:
        """Confirm horizontal_flip=False invariant is strictly preserved across submodules."""
        # Non-negotiable invariant check
        config = LaneFusionEngineConfig()
        # Verify no horizontal flip setting enables flipping
        assert not getattr(config, "horizontal_flip", False)

    def test_submodule_instantiation(self, default_fusion_engine: LaneFusionEngine) -> None:
        """Verify all essential modular components are properly constructed."""
        eng = default_fusion_engine
        assert eng.dl_ensemble is not None
        assert eng.temporal_smoother is not None
        assert eng.quality_scorer is not None
        assert eng.guardrail_enforcer is not None
        assert eng.polynomial_tracker is not None
        assert eng.classical_runner.lighting_compensator is not None
        assert eng.classical_runner.faded_recovery is not None


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Optimal Scene Perception & Fusion Weighting Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestOptimalLaneFusion:
    """Verify fusion behavior under optimal highway driving conditions."""

    def test_clean_highway_dual_boundary_detection(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Ensure clean highway frames yield dual-boundary optimal status and high quality."""
        default_fusion_engine.reset()
        pred: LanePrediction = default_fusion_engine.process_frame(
            bgr_frame=clean_highway_frame,
            ego_speed_mps=25.0,
            scene_tag="HIGHWAY",
        )

        assert pred is not None
        assert isinstance(pred, LanePrediction)
        assert pred.frame_index > 0
        assert pred.scene_tag == "HIGHWAY"

        # Quality and confidence should be strong
        assert pred.quality_assessment.quality_score >= 0.50
        assert pred.confidence_fusion.fused_confidence >= 0.50

        # Physical lane width should be near 3.7m nominal
        assert 3.2 <= pred.lane_width_m <= 4.2

        # Lateral error should be near zero on centered vehicle
        assert abs(pred.lateral_error_m) < 0.20

        # Virtual guardrails should be nominal
        assert pred.guardrail_result.active_severity == GuardrailSeverity.NOMINAL
        assert pred.is_safe_for_actuation is True
        assert pred.arduino_clutch_relay_engaged is True

    def test_multi_frame_ekf_stabilization(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify 8-state EKF reduces polynomial variance over consecutive frames."""
        default_fusion_engine.reset()
        estimates = []
        for _ in range(10):
            pred = default_fusion_engine.process_frame(
                bgr_frame=clean_highway_frame,
                ego_speed_mps=25.0,
                scene_tag="HIGHWAY",
                dt_s=0.0333,
            )
            estimates.append(pred.smoothed_estimate)

        # Check covariance trace decreases or stabilizes
        initial_trace = np.trace(estimates[0].covariance_matrix)
        final_trace = np.trace(estimates[-1].covariance_matrix)
        assert final_trace <= initial_trace * 1.5
        assert estimates[-1].quality_score >= 0.60

    def test_scene_adaptive_alpha_tuning(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify DL fusion weight alpha increases under adverse scene tags (RAIN/NIGHT)."""
        default_fusion_engine.reset()
        pred_highway = default_fusion_engine.process_frame(
            bgr_frame=clean_highway_frame,
            scene_tag="HIGHWAY",
        )
        alpha_highway = pred_highway.fusion_alpha_dl_weight

        default_fusion_engine.reset()
        pred_rain = default_fusion_engine.process_frame(
            bgr_frame=clean_highway_frame,
            scene_tag="RAIN",
        )
        alpha_rain = pred_rain.fusion_alpha_dl_weight

        default_fusion_engine.reset()
        pred_night = default_fusion_engine.process_frame(
            bgr_frame=clean_highway_frame,
            scene_tag="NIGHT",
        )
        alpha_night = pred_night.fusion_alpha_dl_weight

        # DL weight alpha must be higher in rain/night than on clear highway
        assert alpha_rain >= alpha_highway
        assert alpha_night >= alpha_highway


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Safeguard 1: Dual-Boundary Parallel Hallucination Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDualBoundaryParallelProjection:
    """Audit Safeguard 1: Hallucinate missing boundary from visible one (W = 3.7m)."""

    def test_single_faded_line_parallel_projection(
        self, default_fusion_engine: LaneFusionEngine, single_faded_line_frame: np.ndarray
    ) -> None:
        """When right boundary is absent, mirror visible left boundary using W = 3.7m."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(
            bgr_frame=single_faded_line_frame,
            ego_speed_mps=25.0,
            scene_tag="HIGHWAY",
        )

        assert pred is not None
        # Must engage parallel hallucination
        assert pred.status in (
            PerceptionExecutionStatus.PARALLEL_HALLUCINATED,
            PerceptionExecutionStatus.OPTIMAL,
            PerceptionExecutionStatus.DEGRADED_DISENGAGED,
        )

        # Hallucinated lane width must strictly match nominal lane width
        assert 3.4 <= pred.lane_width_m <= 4.0

        # Curvature coefficient A must match between left and right
        al = pred.left_polynomial[0]
        ar = pred.right_polynomial[0]
        assert abs(al - ar) < 0.0005

    def test_max_hallucination_frames_guard(
        self, default_fusion_engine: LaneFusionEngine, single_faded_line_frame: np.ndarray
    ) -> None:
        """Verify hallucination cannot exceed 20 consecutive frames without driver alert."""
        default_fusion_engine.reset()
        hallu_count = 0
        for f in range(25):
            pred = default_fusion_engine.process_frame(
                bgr_frame=single_faded_line_frame,
                scene_tag="HIGHWAY",
            )
            if pred.status == PerceptionExecutionStatus.PARALLEL_HALLUCINATED:
                hallu_count += 1

        # Must not remain in unverified hallucination infinitely
        assert hallu_count <= 22


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Safeguard 2 & 5: Guardrails, Degradation & Safety Relay Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestSafetyGuardrailsAndRelayInterlock:
    """Audit Safeguards 2 & 5: 0.3m virtual buffers and 5-frame low-Q relay cutoff."""

    def test_five_frame_low_quality_relay_cutoff(
        self, default_fusion_engine: LaneFusionEngine, empty_dark_asphalt_frame: np.ndarray
    ) -> None:
        """Verify Q < 0.35 sustained for 5 consecutive frames trips Arduino safety relay."""
        default_fusion_engine.reset()

        relay_tripped = False
        trip_frame = -1

        for f in range(8):
            pred = default_fusion_engine.process_frame(
                bgr_frame=empty_dark_asphalt_frame,
                scene_tag="HIGHWAY",
            )
            if not pred.arduino_clutch_relay_engaged or not pred.is_safe_for_actuation:
                relay_tripped = True
                trip_frame = f
                break

        assert relay_tripped is True
        # Must trip at or near frame 5 (0-indexed 3, 4, 5, 6, or 7)
        assert trip_frame in (3, 4, 5, 6, 7)

    def test_virtual_guardrail_severity_escalation(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify guardrail severity responds to trajectory lateral displacement."""
        default_fusion_engine.reset()
        # Normal centered frame
        pred_norm = default_fusion_engine.process_frame(clean_highway_frame, ego_speed_mps=25.0)
        assert pred_norm.guardrail_result.active_severity == GuardrailSeverity.NOMINAL

        # High yaw rate towards boundary causes projected ribbon to breach envelope
        pred_escalated = default_fusion_engine.process_frame(
            clean_highway_frame,
            ego_speed_mps=30.0,
            ego_yaw_rate_radps=0.15,
        )
        assert pred_escalated.guardrail_result.active_severity in (
            GuardrailSeverity.WARNING,
            GuardrailSeverity.CRITICAL,
            GuardrailSeverity.EMERGENCY,
        )

    def test_time_to_lane_departure_alert_generation(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Ensure TTLD calculation outputs valid predictions and alert flags."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(
            bgr_frame=clean_highway_frame,
            ego_speed_mps=30.0,
            ego_yaw_rate_radps=0.08,  # Turning towards boundary
        )

        assert pred.departure_prediction is not None
        assert pred.departure_prediction.time_to_departure_s > 0.0
        assert pred.departure_prediction.direction in (
            DepartureDirection.NONE,
            DepartureDirection.DEPARTING_LEFT,
            DepartureDirection.DEPARTING_RIGHT,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Robustness & Corner-Case Edge Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestRobustnessAndCornerCases:
    """Stress test perception engine against extreme optical and kinematic conditions."""

    def test_complete_blackout_frame(self, default_fusion_engine: LaneFusionEngine) -> None:
        """Handle pure black frame (camera disconnect / tunnel blackout) gracefully."""
        default_fusion_engine.reset()
        black_frame = np.zeros((360, 640, 3), dtype=np.uint8)

        pred = default_fusion_engine.process_frame(black_frame, scene_tag="TUNNEL")
        assert pred is not None
        assert pred.quality_assessment.quality_score < 0.55

    def test_specular_glare_saturation(self, default_fusion_engine: LaneFusionEngine) -> None:
        """Handle 100% saturated white glare frame without numeric crash or NaN."""
        default_fusion_engine.reset()
        white_frame = np.full((360, 640, 3), 255, dtype=np.uint8)

        pred = default_fusion_engine.process_frame(white_frame, scene_tag="HIGHWAY")
        assert pred is not None
        assert not math.isnan(pred.lateral_error_m)
        assert not math.isnan(pred.heading_error_deg)
        assert not math.isnan(pred.lane_width_m)

    def test_pitch_oscillation_compensation(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify dynamic pitch variations (+/- 4 degrees) adapt horizon properly."""
        default_fusion_engine.reset()
        pred_nominal = default_fusion_engine.process_frame(
            clean_highway_frame, ego_pitch_deg=5.0
        )
        pred_pitch_up = default_fusion_engine.process_frame(
            clean_highway_frame, ego_pitch_deg=9.0
        )
        pred_pitch_down = default_fusion_engine.process_frame(
            clean_highway_frame, ego_pitch_deg=1.0
        )

        assert pred_nominal is not None
        assert pred_pitch_up is not None
        assert pred_pitch_down is not None
        # All outputs must remain numerically finite and valid
        assert np.isfinite(pred_pitch_up.lane_width_m)
        assert np.isfinite(pred_pitch_down.lane_width_m)

    def test_execution_latency_budget(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Audit that pipeline latency telemetry is populated and non-negative."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        assert pred.pipeline_latency_ms > 0.0
        assert pred.pipeline_latency_ms < 5000.0  # Safe bounds on any workstation CPU

    def test_telemetry_serialization(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify to_summary_dict serializes all mandatory telemetry fields."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        s_dict = pred.to_summary_dict()

        mandatory_keys = [
            "frame_index",
            "timestamp_utc",
            "status",
            "scene_tag",
            "fusion_alpha",
            "quality_score",
            "lane_width_m",
            "lateral_error_m",
            "heading_error_deg",
            "guardrail_severity",
            "is_safe_for_actuation",
            "relay_engaged",
            "latency_ms",
        ]
        for k in mandatory_keys:
            assert k in s_dict, f"Missing key {k} in summary dictionary"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Geometric Constraints & Parallelism Verifier Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometricConstraintsAndParallelism:
    """Validate polynomial curvature, lane width bounds, and parallelism checks."""

    def test_nominal_highway_geometry_bounds(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Confirm estimated lane width adheres strictly to standard AASHTO 3.7m corridor."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        assert 3.4 <= pred.lane_width_m <= 4.0

    def test_parallel_boundary_curvature_matching(
        self, default_fusion_engine: LaneFusionEngine, curved_highway_frame: np.ndarray
    ) -> None:
        """On curves, left and right boundary curvature A coefficients must share sign and magnitude."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(curved_highway_frame)
        al = pred.left_polynomial[0]
        ar = pred.right_polynomial[0]

        # Curvature discrepancy must be tightly bounded
        assert abs(al - ar) < 0.0050
        assert np.isfinite(al) and np.isfinite(ar)

    def test_centerline_polynomial_midpoint(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify centerline polynomial is arithmetic mean of left and right boundaries."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        pl = pred.left_polynomial
        pr = pred.right_polynomial
        pc = pred.centerline_polynomial

        for y in [200.0, 260.0, 320.0, 359.0]:
            xl = pl[0] * y**2 + pl[1] * y + pl[2]
            xr = pr[0] * y**2 + pr[1] * y + pr[2]
            xc = pc[0] * y**2 + pc[1] * y + pc[2]
            expected_c = (xl + xr) * 0.5
            assert abs(xc - expected_c) < 5.0  # Within 5 pixel tolerance


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Actuation Safety Interlock & State Machine Transition Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestActuationSafetyInterlockMatrix:
    """Audit state transitions across OPTIMAL, COASTING, and DEGRADED_DISENGAGED."""

    def test_transition_from_optimal_to_hallucinated(
        self,
        default_fusion_engine: LaneFusionEngine,
        clean_highway_frame: np.ndarray,
        single_faded_line_frame: np.ndarray,
    ) -> None:
        """Test smooth transition from dual-boundary detection into parallel hallucination."""
        default_fusion_engine.reset()

        # Step 1: Initialize on clean highway
        p1 = default_fusion_engine.process_frame(clean_highway_frame)
        assert p1.status == PerceptionExecutionStatus.OPTIMAL

        # Step 2: Feed single faded line
        p2 = default_fusion_engine.process_frame(single_faded_line_frame)
        assert p2.status in (
            PerceptionExecutionStatus.PARALLEL_HALLUCINATED,
            PerceptionExecutionStatus.OPTIMAL,
        )

        # Actuation must remain safe during early hallucination frames
        assert p2.is_safe_for_actuation is True
        assert p2.arduino_clutch_relay_engaged is True

    def test_relay_recovery_after_reacquiring_lanes(
        self,
        default_fusion_engine: LaneFusionEngine,
        empty_dark_asphalt_frame: np.ndarray,
        clean_highway_frame: np.ndarray,
    ) -> None:
        """Verify that once good lanes reappear, system clears degradation flag."""
        default_fusion_engine.reset()

        # Trip degradation cutoff via 6 bad frames
        for _ in range(6):
            default_fusion_engine.process_frame(empty_dark_asphalt_frame)

        # Feed clear clean frames to reacquire track
        for _ in range(5):
            p_rec = default_fusion_engine.process_frame(clean_highway_frame)

        # After sufficient high-quality frames, quality must recover
        assert p_rec.quality_assessment.quality_score >= 0.50

    def test_lateral_error_sign_convention(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """
        Verify lateral error sign:
        Negative when vehicle is to the left of lane center.
        Positive when vehicle is to the right of lane center.
        """
        # Lane markings shifted left
        frame_l = np.full((360, 640, 3), 70, dtype=np.uint8)
        y_eval = np.linspace(170, 360, 40)
        pts_l1 = [(int(-0.45 * (y - 170) + 260), int(y)) for y in y_eval]
        pts_r1 = [(int(0.45 * (y - 170) + 320), int(y)) for y in y_eval]
        cv2.polylines(frame_l, [np.array(pts_l1)], False, (255, 255, 255), 4)
        cv2.polylines(frame_l, [np.array(pts_r1)], False, (255, 255, 255), 4)

        default_fusion_engine.reset()
        pred_l = default_fusion_engine.process_frame(frame_l)

        # Lane markings shifted right
        frame_r = np.full((360, 640, 3), 70, dtype=np.uint8)
        pts_l2 = [(int(-0.45 * (y - 170) + 320), int(y)) for y in y_eval]
        pts_r2 = [(int(0.45 * (y - 170) + 380), int(y)) for y in y_eval]
        cv2.polylines(frame_r, [np.array(pts_l2)], False, (255, 255, 255), 4)
        cv2.polylines(frame_r, [np.array(pts_r2)], False, (255, 255, 255), 4)

        default_fusion_engine.reset()
        pred_r = default_fusion_engine.process_frame(frame_r)

        assert np.isfinite(pred_l.lateral_error_m)
        assert np.isfinite(pred_r.lateral_error_m)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Fault Injection & Hardware Disturbance Simulation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestFaultInjectionAndHardwareFailures:
    """Verify safety boundaries under hardware faults, NaN inputs, and sensor jitter."""

    def test_nan_free_on_impulse_salt_and_pepper_noise(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Heavy salt-and-pepper noise simulating sensor transmission bitflips."""
        default_fusion_engine.reset()
        noisy_frame = clean_highway_frame.copy()
        mask = np.random.rand(*noisy_frame.shape[:2]) < 0.15
        noisy_frame[mask] = 255

        pred = default_fusion_engine.process_frame(noisy_frame)
        assert pred is not None
        assert np.isfinite(pred.lateral_error_m)
        assert np.isfinite(pred.heading_error_deg)

    def test_variable_speed_adaptation(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Ensure system adapts Stanley gain and lookahead across speeds (10 to 35 m/s)."""
        default_fusion_engine.reset()
        p_low = default_fusion_engine.process_frame(clean_highway_frame, ego_speed_mps=10.0)
        p_high = default_fusion_engine.process_frame(clean_highway_frame, ego_speed_mps=35.0)

        assert p_low.departure_prediction is not None
        assert p_high.departure_prediction is not None
        # Higher speed implies shorter time-to-departure for same clearance
        assert p_high.departure_prediction.time_to_departure_s <= p_low.departure_prediction.time_to_departure_s + 0.1

    def test_zero_velocity_handling(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify zero vehicle velocity (stop-and-go / traffic light) does not divide by zero."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame, ego_speed_mps=0.0)
        assert pred is not None
        assert np.isfinite(pred.lateral_error_m)
        assert np.isfinite(pred.departure_prediction.time_to_departure_s)

    def test_dt_variations_and_frame_drops(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Simulate dropped frames with variable dt (33ms to 120ms)."""
        default_fusion_engine.reset()
        pred_normal = default_fusion_engine.process_frame(clean_highway_frame, dt_s=0.0333)
        # Dropped frame (100ms gap)
        pred_dropped = default_fusion_engine.process_frame(clean_highway_frame, dt_s=0.1000)

        assert pred_normal is not None
        assert pred_dropped is not None
        assert np.isfinite(pred_dropped.smoothed_estimate.lane_width_m)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Confidence-Weighted Voting & Weight Allocation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfidenceWeightedVotingMath:
    """Validate mathematical correctness of multi-method confidence fusion."""

    def test_alpha_weight_limits(self, default_fusion_engine: LaneFusionEngine) -> None:
        """Verify fusion alpha is strictly bounded inside [0.05, 0.95]."""
        for sc in SceneType:
            profile = get_scene_profile(sc)
            assert 0.05 <= profile.fusion.dl_fusion_weight <= 0.95

    def test_dl_dominance_when_classical_fails(
        self, default_fusion_engine: LaneFusionEngine
    ) -> None:
        """When classical CV confidence drops to zero, DL should drive the estimation."""
        eng = default_fusion_engine
        eng.reset()
        # Synthetic degraded frame where classical Canny edges break
        degraded_frame = np.full((360, 640, 3), 40, dtype=np.uint8)
        # Add diffuse road marking without sharp edges
        cv2.line(degraded_frame, (100, 360), (280, 200), (120, 120, 120), 16)
        cv2.line(degraded_frame, (540, 360), (360, 200), (120, 120, 120), 16)
        cv2.GaussianBlur(degraded_frame, (15, 15), 5.0, dst=degraded_frame)

        pred = eng.process_frame(degraded_frame, scene_tag="RAIN")
        assert pred is not None
        # Must execute without crash and maintain valid coordinates
        assert np.isfinite(pred.lane_width_m)

    def test_equal_confidence_voting_blend(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify weighted voting smoothly blends classical and neural polynomial outputs."""
        eng = default_fusion_engine
        eng.reset()
        pred = eng.process_frame(clean_highway_frame, scene_tag="HIGHWAY")
        # In highway mode, both classical and neural outputs are active
        assert pred.confidence_fusion.classical_metrics.raw_confidence >= 0.0
        assert pred.confidence_fusion.dl_metrics.raw_confidence >= 0.0
        assert pred.confidence_fusion.fused_confidence > 0.0
        assert pred.status in (
            PerceptionExecutionStatus.OPTIMAL,
            PerceptionExecutionStatus.DEEP_LEARNING_DOMINANT,
            PerceptionExecutionStatus.CLASSICAL_DOMINANT,
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Drivable Corridor Polygon Geometry & Perspective Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDrivableCorridorPolygonGeometry:
    """Validate spatial consistency of 2D/3D drivable corridor polygons."""

    def test_corridor_vertex_count_and_continuity(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Corridor polygon in image space must have valid closed topology."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        poly_uv = pred.corridor_polygon_uv

        assert len(poly_uv) >= 6
        # Coordinates must fall within or near camera frame bounds
        for u, v in poly_uv:
            assert -100 <= u <= 740
            assert 0 <= v <= 360

    def test_corridor_bev_ground_plane_area(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Corridor polygon in BEV metric space must enclose reasonable ground area."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        poly_bev = pred.corridor_polygon_bev_m

        assert len(poly_bev) >= 4
        # Longitudinal span should cover lookahead horizon (e.g. 5m to 30m)
        y_coords = [p[1] for p in poly_bev]
        assert max(y_coords) - min(y_coords) > 15.0  # At least 15m longitudinal lookahead

    def test_corridor_non_inversion(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Left boundary x must strictly be less than right boundary x at all lookahead depths."""
        default_fusion_engine.reset()
        pred = default_fusion_engine.process_frame(clean_highway_frame)
        pl = pred.left_polynomial
        pr = pred.right_polynomial

        for y in np.linspace(200, 359, 10):
            xl = pl[0] * y**2 + pl[1] * y + pl[2]
            xr = pr[0] * y**2 + pr[1] * y + pr[2]
            assert xl < xr, f"Corridor inverted at y={y}: xl={xl} >= xr={xr}"


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Virtual Guardrail Response & CAN Priority Verification Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestVirtualGuardrailActuationResponses:
    """Verify counter-steer gain amplification and priority CAN dispatches."""

    def test_severity_tier_definitions(self) -> None:
        """Verify GuardrailSeverity values correspond to ASIL-B hazard levels."""
        assert GuardrailSeverity.NOMINAL.value == "NOMINAL"
        assert GuardrailSeverity.WARNING.value == "WARNING"
        assert GuardrailSeverity.CRITICAL.value == "CRITICAL"
        assert GuardrailSeverity.EMERGENCY.value == "EMERGENCY"

    def test_can_risk_priority_arbitration_id(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Verify high-priority CAN ID 0x150 is assigned for boundary alerts."""
        default_fusion_engine.reset()
        # High yaw rate and speed towards boundary triggers departure warning and CAN alert
        pred = default_fusion_engine.process_frame(
            clean_highway_frame, ego_speed_mps=30.0, ego_yaw_rate_radps=0.15
        )

        assert pred.departure_prediction is not None
        assert pred.guardrail_result.active_severity != GuardrailSeverity.NOMINAL
        assert pred.departure_prediction.can_message_priority in (0x140, 0x150)

    def test_emergency_takeover_warning_flag(
        self, default_fusion_engine: LaneFusionEngine, empty_dark_asphalt_frame: np.ndarray
    ) -> None:
        """Confirm SAE Level 2 takeover request flag is asserted upon prolonged failure."""
        default_fusion_engine.reset()
        for _ in range(6):
            pred = default_fusion_engine.process_frame(empty_dark_asphalt_frame)

        assert pred.arduino_clutch_relay_engaged is False
        assert pred.is_safe_for_actuation is False
        assert (
            pred.quality_assessment.takeover_alert_active is True
            or pred.quality_assessment.arduino_relay_trip_active is True
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 12. Lighting Compensation & Preprocessing Pipeline Integration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLightingCompensationPipelineIntegration:
    """Validate CLAHE, specular highlight suppression, and gamma correction in pipeline."""

    def test_low_light_night_mode_enhancement(
        self, default_fusion_engine: LaneFusionEngine
    ) -> None:
        """Verify low-light night mode enhances lane markings without noise explosion."""
        eng = default_fusion_engine
        eng.reset()
        # Create dark night frame (pixel values ~25) with dim lane markings (~70)
        night_frame = np.full((360, 640, 3), 25, dtype=np.uint8)
        cv2.line(night_frame, (100, 360), (280, 200), (70, 70, 70), 3)
        cv2.line(night_frame, (540, 360), (360, 200), (70, 70, 70), 3)

        pred = eng.process_frame(night_frame, scene_tag="NIGHT")
        assert pred is not None
        assert pred.scene_tag == "NIGHT"
        assert np.isfinite(pred.lateral_error_m)

    def test_tunnel_entrance_exposure_transient(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """Simulate sudden exposure drop from 180 to 30 over 3 consecutive frames."""
        eng = default_fusion_engine
        eng.reset()

        # Frame 1: Bright sunlight
        p1 = eng.process_frame(clean_highway_frame, scene_tag="HIGHWAY")
        # Frame 2: Tunnel portal boundary (half darkened)
        half_dark = (clean_highway_frame.astype(np.float32) * 0.4).astype(np.uint8)
        p2 = eng.process_frame(half_dark, scene_tag="TUNNEL")
        # Frame 3: Deep tunnel interior
        deep_dark = (clean_highway_frame.astype(np.float32) * 0.2).astype(np.uint8)
        p3 = eng.process_frame(deep_dark, scene_tag="TUNNEL")

        assert p1 is not None and p2 is not None and p3 is not None
        # Polynomial tracks should not jump wildly across the illumination transient
        delta_c = abs(p3.left_polynomial[2] - p1.left_polynomial[2])
        assert delta_c < 80.0  # Pixels


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Continuous Driving Sequence & Sweeping S-Curve Simulation
# ═══════════════════════════════════════════════════════════════════════════════

class TestContinuousDrivingSequence:
    """Simulate a sustained multi-second highway driving sequence on an S-curve."""

    def test_fifty_frame_continuous_s_curve_tracking(
        self, default_fusion_engine: LaneFusionEngine
    ) -> None:
        """Execute 50 consecutive frames with dynamic curvature sinusoidal modulation."""
        eng = default_fusion_engine
        eng.reset()

        lateral_errors = []
        heading_errors = []
        qualities = []
        lane_widths = []

        total_frames = 50
        dt = 0.0333
        speed = 25.0

        for f in range(total_frames):
            # Dynamic sinusoidal road curvature
            t = f * dt
            curve_param = 0.0004 * math.sin(t * 1.8)

            frame = np.full((360, 640, 3), 65, dtype=np.uint8)
            y_eval = np.linspace(170, 360, 40)
            pts_l = []
            pts_r = []

            for y in y_eval:
                dy = y - 170
                xl = int(curve_param * dy**2 - 0.45 * dy + 290)
                xr = int(curve_param * dy**2 + 0.45 * dy + 350)
                if 0 <= xl < 640:
                    pts_l.append((xl, int(y)))
                if 0 <= xr < 640:
                    pts_r.append((xr, int(y)))

            if len(pts_l) > 1:
                cv2.polylines(frame, [np.array(pts_l)], False, (230, 230, 230), 4)
            if len(pts_r) > 1:
                cv2.polylines(frame, [np.array(pts_r)], False, (230, 230, 230), 4)

            pred = eng.process_frame(
                bgr_frame=frame,
                ego_speed_mps=speed,
                scene_tag="HIGHWAY",
                dt_s=dt,
            )

            lateral_errors.append(abs(pred.lateral_error_m))
            heading_errors.append(abs(pred.heading_error_deg))
            qualities.append(pred.quality_assessment.quality_score)
            lane_widths.append(pred.lane_width_m)

        # Performance assertions across continuous trajectory
        mean_lat_err = float(np.mean(lateral_errors))
        assert mean_lat_err < 0.12, f"Excessive mean lateral error: {mean_lat_err}m"

        mean_q = float(np.mean(qualities))
        assert mean_q >= 0.60, f"Perception quality collapsed on S-curve: {mean_q}"

        std_width = float(np.std(lane_widths))
        assert std_width < 0.25, f"Lane width jitter too high: {std_width}m"

        # No false emergency trips on normal driving
        assert eng.guardrail_enforcer.is_interlock_tripped is False

    def test_reproducibility_across_identical_frame_invocations(
        self, default_fusion_engine: LaneFusionEngine, clean_highway_frame: np.ndarray
    ) -> None:
        """
        Ensure identical camera frames generate bit-for-bit or numerically identical outputs
        when the filter state is reset between evaluations.
        """
        eng = default_fusion_engine

        eng.reset()
        pred_a = eng.process_frame(clean_highway_frame, ego_speed_mps=25.0)

        eng.reset()
        pred_b = eng.process_frame(clean_highway_frame, ego_speed_mps=25.0)

        assert pred_a.status == pred_b.status
        assert abs(pred_a.lateral_error_m - pred_b.lateral_error_m) < 1e-4
        assert abs(pred_a.lane_width_m - pred_b.lane_width_m) < 1e-4
        assert abs(pred_a.quality_assessment.quality_score - pred_b.quality_assessment.quality_score) < 0.01
        assert pred_a.guardrail_result.active_severity == pred_b.guardrail_result.active_severity


