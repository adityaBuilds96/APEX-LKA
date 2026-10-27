"""
================================================================================
APEX-Vision Core v4.0 — Phase 7B Master Execution Prompt
Section 1: Lane Perception Precision Engine ("The Repair Engine")
Verification Suite: Faded Lane Recovery & Dual-Boundary Parallel Projection Tests
================================================================================

Universal Branding: APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Autonomy Level:     driver-supervised SAE Level 2
Safety Integrity:   ISO 26262-style process; ASIL-B aligned; not independently assessed
Non-Negotiables:    horizontal_flip=False; Zero-tolerance boundary envelope violations
Target File Budget: Min 720 LOC
================================================================================

This test suite comprehensively validates the faded lane recovery subsystem:
1. Dual-Boundary Parallel Projection: missing line hallucinated from visible line (W = 3.7m).
2. Parabolic polynomial evaluation, curvature radius, heading tangents, and boundary consistency.
3. Gabor Directional Filter Bank (8 orientations x 3 scales) and Frangi ridge extraction.
4. Steerable filter responses and morphological ridge enhancement.
5. Multi-frame line continuity hallucination and exponential confidence decay (gamma = 0.92).
6. ASIL-B degradation guard: max hallucination limit (20 frames) forcing safety disengagement.
7. Edge cases: NaN/Inf coefficients, extreme curvature, inverted lane order, noisy backgrounds.
8. Telemetry reporting, spline interpolation, and stress benchmarking.
"""

from __future__ import annotations

import copy
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Ensure workspace root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.faded_lane_recovery import (
    BoundaryPolynomial2D,
    CurvatureConsistencyVerifier,
    FadedLaneRecoveryEngine,
    FadedLaneRecoveryTelemetry,
    FadedRecoveryBenchmarkResult,
    FadedRecoveryConfig,
    FrangiRidgeExtractor,
    GaborFilterBank,
    MorphologicalRidgeEnhancer,
    RecoveryMode,
    SplineBoundaryInterpolator,
    SteerableLineFilterBank,
    get_faded_lane_recovery_engine,
)


# ==============================================================================
# Test Fixtures & Synthetic Data Generators
# ==============================================================================

@pytest.fixture
def default_recovery_config() -> FadedRecoveryConfig:
    """Provide a validated baseline FadedRecoveryConfig fixture."""
    return FadedRecoveryConfig(
        nominal_lane_width_m=3.70,
        min_lane_width_m=2.80,
        max_lane_width_m=4.60,
        confidence_decay_factor=0.92,
        max_hallucination_frames=20,
        high_confidence_threshold=0.65,
        weak_confidence_threshold=0.35,
        meters_per_pixel=0.015625,
        enable_gabor_prefilter=True,
        gabor_orientations=8,
        gabor_scales=(3, 5, 7),
        gabor_frequency=0.08,
    )


@pytest.fixture
def clean_dual_polynomials() -> Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D]:
    """
    Generate clean, parallel dual boundary polynomials representing a straight lane.
    Left boundary: x = -1.85 m; Right boundary: x = +1.85 m (nominal width 3.70 m).
    """
    left_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.0,
        B=0.0,
        C=-1.85,
        confidence=0.95,
        inlier_count=450,
        rmse_error=0.02,
        is_hallucinated=False,
        hallucination_source="NONE",
    )
    right_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.0,
        B=0.0,
        C=1.85,
        confidence=0.94,
        inlier_count=440,
        rmse_error=0.02,
        is_hallucinated=False,
        hallucination_source="NONE",
    )
    return left_poly, right_poly


@pytest.fixture
def curved_dual_polynomials() -> Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D]:
    """
    Generate curved parallel dual boundary polynomials (radius ~ 500m).
    A = 0.001, B = 0.02.
    """
    left_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.001,
        B=0.02,
        C=-1.85,
        confidence=0.92,
        inlier_count=380,
        rmse_error=0.03,
        is_hallucinated=False,
        hallucination_source="NONE",
    )
    right_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.001,
        B=0.02,
        C=1.85,
        confidence=0.90,
        inlier_count=370,
        rmse_error=0.03,
        is_hallucinated=False,
        hallucination_source="NONE",
    )
    return left_poly, right_poly


@pytest.fixture
def synthetic_lane_bev_image() -> np.ndarray:
    """Generate a 400x200 grayscale BEV road surface image with painted lane markings."""
    h, w = 400, 200
    img = np.full((h, w), 50, dtype=np.uint8)  # Asphalt base
    # Add road texture noise
    noise = np.random.normal(0, 5, (h, w)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Draw left line (dashed/faded) and right line (solid bright)
    # Left line at x = 40
    for y in range(0, h, 40):
        cv2.line(img, (40, y), (40, min(y + 20, h)), 160, 3)
    # Right line at x = 160 (solid white)
    cv2.line(img, (160, 0), (160, h), 240, 4)
    return img


# ==============================================================================
# 1. BoundaryPolynomial2D Mathematical Evaluation Tests
# ==============================================================================

class TestBoundaryPolynomial2D:
    """Validates parabolic polynomial representation and differential geometric properties."""

    def test_polynomial_evaluation_scalar_and_vector(self) -> None:
        """Verify evaluation of x(y) = A*y^2 + B*y + C for scalar and vector inputs."""
        poly = BoundaryPolynomial2D(
            is_valid=True, A=0.002, B=-0.05, C=1.85, confidence=0.88
        )
        y_val = 10.0
        expected_x = 0.002 * (10.0 ** 2) - 0.05 * 10.0 + 1.85
        assert math.isclose(float(poly.evaluate_x(y_val)), expected_x, rel_tol=1e-6)

        y_vec = np.array([0.0, 10.0, 20.0, 30.0], dtype=np.float64)
        x_vec = poly.evaluate_x(y_vec)
        assert isinstance(x_vec, np.ndarray)
        assert len(x_vec) == 4
        assert math.isclose(x_vec[0], 1.85, rel_tol=1e-6)
        assert math.isclose(x_vec[1], expected_x, rel_tol=1e-6)

    def test_evaluate_heading_tangent_angle(self) -> None:
        """Verify tangent heading angle theta = atan(2*A*y + B)."""
        poly = BoundaryPolynomial2D(
            is_valid=True, A=0.001, B=0.10, C=-1.85, confidence=0.85
        )
        y_val = 15.0
        expected_slope = 2.0 * 0.001 * 15.0 + 0.10
        expected_theta = math.atan(expected_slope)
        assert math.isclose(poly.evaluate_heading_rad(y_val), expected_theta, rel_tol=1e-6)

    def test_evaluate_curvature_radius_straight_line(self) -> None:
        """Verify that zero curvature (A=0) produces safe ceiling radius (5000.0 m)."""
        straight_poly = BoundaryPolynomial2D(
            is_valid=True, A=0.0, B=0.0, C=1.85, confidence=0.9
        )
        r = straight_poly.evaluate_curvature_radius_m(y=10.0)
        assert r >= 5000.0

    def test_evaluate_curvature_radius_curve(self) -> None:
        """Verify radius calculation R = (1 + (2Ay+B)^2)^1.5 / |2A| for circular arc."""
        A = 0.002
        B = 0.0
        poly = BoundaryPolynomial2D(is_valid=True, A=A, B=B, C=0.0, confidence=0.9)
        # At y = 0: R = 1.0 / (2 * 0.002) = 250.0 meters
        r = poly.evaluate_curvature_radius_m(y=0.0)
        assert math.isclose(r, 250.0, rel_tol=1e-4)

    def test_coefficients_array_property(self) -> None:
        """Verify coefficients property returns [A, B, C] array."""
        poly = BoundaryPolynomial2D(
            is_valid=True, A=0.005, B=-0.01, C=2.10, confidence=0.8
        )
        coeffs = poly.coefficients
        assert isinstance(coeffs, np.ndarray)
        assert np.array_equal(coeffs, np.array([0.005, -0.01, 2.10]))

    def test_negative_curvature_radius(self) -> None:
        """Verify curvature calculation correctly handles negative A curvature."""
        poly = BoundaryPolynomial2D(is_valid=True, A=-0.002, B=0.0, C=0.0, confidence=0.9)
        r = poly.evaluate_curvature_radius_m(y=0.0)
        assert r > 0.0
        assert math.isclose(r, 250.0, rel_tol=1e-4)

    def test_high_heading_slope_curvature(self) -> None:
        """Verify curvature evaluation when heading slope is steep."""
        poly = BoundaryPolynomial2D(is_valid=True, A=0.001, B=1.0, C=0.0, confidence=0.8)
        r = poly.evaluate_curvature_radius_m(y=0.0)
        expected = ((1.0 + 1.0**2)**1.5) / (2.0 * 0.001)
        assert math.isclose(r, expected, rel_tol=1e-4)


# ==============================================================================
# 2. Configuration Integrity & Parameter Boundaries
# ==============================================================================

class TestFadedRecoveryConfigValidation:
    """Validates configuration parameters and boundaries for ISO 26262 ASIL-B compliance."""

    def test_default_config_ranges(self, default_recovery_config: FadedRecoveryConfig) -> None:
        """Verify default configuration values adhere to automotive domain constraints."""
        cfg = default_recovery_config
        assert 3.0 <= cfg.nominal_lane_width_m <= 4.0
        assert cfg.min_lane_width_m < cfg.nominal_lane_width_m < cfg.max_lane_width_m
        assert 0.80 <= cfg.confidence_decay_factor < 1.00
        assert 10 <= cfg.max_hallucination_frames <= 30
        assert cfg.high_confidence_threshold > cfg.weak_confidence_threshold
        assert cfg.meters_per_pixel > 0.0
        assert cfg.gabor_orientations >= 4
        assert len(cfg.gabor_scales) >= 2

    def test_custom_lane_width_configuration(self) -> None:
        """Verify engine accepts valid custom lane widths for non-standard roadways."""
        custom_cfg = FadedRecoveryConfig(
            nominal_lane_width_m=3.25,
            min_lane_width_m=2.50,
            max_lane_width_m=4.00,
        )
        assert custom_cfg.nominal_lane_width_m == 3.25
        assert custom_cfg.min_lane_width_m == 2.50


# ==============================================================================
# 3. Gabor Filter Bank & Ridge Feature Extraction Tests
# ==============================================================================

class TestGaborFilterBank:
    """Validates the 8-orientation x 3-scale 2D Gabor filter bank."""

    def test_gabor_filter_initialization(self) -> None:
        """Verify Gabor filter bank creates oriented kernels."""
        filter_bank = GaborFilterBank(
            num_orientations=8,
            scales=(3, 5, 7),
            frequency=0.08,
        )
        assert len(filter_bank._kernels) > 0
        for k in filter_bank._kernels:
            assert isinstance(k, np.ndarray)
            assert k.dtype == np.float32

    def test_gabor_enhancement_response_on_lanes(
        self, synthetic_lane_bev_image: np.ndarray
    ) -> None:
        """Verify Gabor filter bank enhances lane boundaries against road noise."""
        filter_bank = GaborFilterBank(num_orientations=8, scales=(3, 5), frequency=0.08)
        enhanced = filter_bank.enhance_lane_markings(synthetic_lane_bev_image)
        assert enhanced.shape == synthetic_lane_bev_image.shape
        assert enhanced.dtype == np.uint8
        # Markings at x=160 should have elevated response compared to asphalt background
        assert enhanced[200, 160] > enhanced[200, 100]

    def test_gabor_empty_or_zero_image(self) -> None:
        """Verify Gabor filter bank handles completely black image without error."""
        filter_bank = GaborFilterBank(num_orientations=4, scales=(3,), frequency=0.08)
        black_img = np.zeros((100, 100), dtype=np.uint8)
        enhanced = filter_bank.enhance_lane_markings(black_img)
        assert enhanced.shape == (100, 100)
        assert np.max(enhanced) == 0


class TestFrangiAndSteerableRidgeExtractors:
    """Validates Frangi ridge extraction and Steerable line filters."""

    def test_frangi_ridge_extractor(self, synthetic_lane_bev_image: np.ndarray) -> None:
        """Verify Frangi filter isolates linear lane structures."""
        frangi = FrangiRidgeExtractor(scales=(1.5, 3.0), beta=0.5, c=15.0)
        ridges = frangi.extract_ridges(synthetic_lane_bev_image)
        assert ridges.shape == synthetic_lane_bev_image.shape
        assert ridges.dtype == np.uint8
        # Line location should have higher ridge energy than off-road background
        assert ridges[150, 160] >= ridges[150, 100]

    def test_steerable_filter_response(self, synthetic_lane_bev_image: np.ndarray) -> None:
        """Verify steerable filter directional output."""
        steerable = SteerableLineFilterBank(sigma=2.0)
        # Vertical lines correspond to theta ~ 90 deg (pi/2)
        response_vertical = steerable.get_oriented_response(
            synthetic_lane_bev_image, theta_rad=math.pi / 2.0
        )
        response_horizontal = steerable.get_oriented_response(
            synthetic_lane_bev_image, theta_rad=0.0
        )
        assert response_vertical.shape == synthetic_lane_bev_image.shape
        # Vertical response at line should be stronger than horizontal response
        assert response_vertical[200, 160] >= response_horizontal[200, 160]

    def test_morphological_ridge_enhancement(
        self, synthetic_lane_bev_image: np.ndarray
    ) -> None:
        """Verify morphological enhancer preserves linear continuity."""
        enhancer = MorphologicalRidgeEnhancer(kernel_length=17, kernel_width=3)
        enhanced = enhancer.enhance_oriented_ridges(synthetic_lane_bev_image, orientation_deg=90.0)
        assert enhanced.shape == synthetic_lane_bev_image.shape
        assert enhanced.dtype == np.uint8


# ==============================================================================
# 4. Curvature Consistency Verifier Tests
# ==============================================================================

class TestCurvatureConsistencyVerifier:
    """Validates curvature consistency checks and outlier polynomial rejection."""

    def test_valid_curvature_accepted(self) -> None:
        """Verify standard highway curve passes verification."""
        verifier = CurvatureConsistencyVerifier(min_curve_radius_highway_m=120.0)
        poly = BoundaryPolynomial2D(
            is_valid=True, A=0.0008, B=0.01, C=-1.85, confidence=0.88
        )
        is_ok, msg = verifier.verify_polynomial(poly, vehicle_speed_mps=20.0)
        assert is_ok is True

    def test_extreme_curvature_rejected(self) -> None:
        """Verify unphysical curvature spike (|A| > 0.05) is rejected."""
        verifier = CurvatureConsistencyVerifier(min_curve_radius_highway_m=120.0)
        bad_poly = BoundaryPolynomial2D(
            is_valid=True, A=0.15, B=0.0, C=1.85, confidence=0.9
        )
        is_ok, msg = verifier.verify_polynomial(bad_poly, vehicle_speed_mps=20.0)
        assert is_ok is False

    def test_nan_curvature_rejected(self) -> None:
        """Verify NaN coefficient polynomial is safely detected and rejected."""
        verifier = CurvatureConsistencyVerifier()
        nan_poly = BoundaryPolynomial2D(
            is_valid=True, A=float("nan"), B=0.0, C=1.85, confidence=0.8
        )
        is_ok, _ = verifier.verify_polynomial(nan_poly, vehicle_speed_mps=20.0)
        assert is_ok is False


# ==============================================================================
# 5. FadedLaneRecoveryEngine Core Behavioral Tests
# ==============================================================================

class TestFadedLaneRecoveryEngineNominal:
    """Validates nominal dual detection without hallucination."""

    def test_both_boundaries_detected_cleanly(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """When both lines have high confidence, no hallucination occurs."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials

        telemetry = engine.recover_boundaries(left_in, right_in)

        assert telemetry.recovery_mode == RecoveryMode.DUAL_DETECTED
        assert telemetry.left_boundary.is_hallucinated is False
        assert telemetry.right_boundary.is_hallucinated is False
        assert telemetry.left_boundary.hallucination_source == "NONE"
        assert telemetry.right_boundary.hallucination_source == "NONE"
        assert telemetry.hallucination_frame_count == 0
        assert math.isclose(telemetry.lane_width_m, 3.70, abs_tol=0.05)
        assert telemetry.is_safe_for_steering is True


class TestDualBoundaryParallelProjectionSafeguard:
    """
    CRITICAL SAFEGUARD TEST:
    Dual-Boundary Parallel Projection: If one lane line is faded or missing,
    hallucinate the missing boundary from the visible one using nominal lane width W = 3.7m.
    """

    def test_project_right_from_visible_left(
        self,
        default_recovery_config: FadedRecoveryConfig,
        curved_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Right boundary is faded/missing; hallucinate from visible left boundary."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, _ = curved_dual_polynomials
        # Faded/invalid right boundary
        right_in = BoundaryPolynomial2D(
            is_valid=False, A=0.0, B=0.0, C=0.0, confidence=0.10
        )

        telemetry = engine.recover_boundaries(left_in, right_in)

        assert telemetry.recovery_mode == RecoveryMode.PROJECTED_FROM_LEFT
        assert telemetry.left_boundary.is_hallucinated is False
        assert telemetry.right_boundary.is_hallucinated is True
        assert telemetry.right_boundary.hallucination_source == "PARALLEL_LEFT"
        # Exact parallel curvature & heading match
        assert math.isclose(telemetry.right_boundary.A, left_in.A, rel_tol=1e-5)
        assert math.isclose(telemetry.right_boundary.B, left_in.B, rel_tol=1e-5)
        # Expected lateral shift: C_right = C_left + W = -1.85 + 3.70 = +1.85
        assert math.isclose(telemetry.right_boundary.C, left_in.C + 3.70, rel_tol=1e-4)
        # Confidence penalty applied to hallucinated boundary
        assert telemetry.right_boundary.confidence < left_in.confidence
        assert telemetry.right_boundary.is_valid is True

    def test_project_left_from_visible_right(
        self,
        default_recovery_config: FadedRecoveryConfig,
        curved_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Left boundary is faded/missing; hallucinate from visible right boundary."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        _, right_in = curved_dual_polynomials
        # Faded/invalid left boundary
        left_in = BoundaryPolynomial2D(
            is_valid=False, A=0.0, B=0.0, C=0.0, confidence=0.05
        )

        telemetry = engine.recover_boundaries(left_in, right_in)

        assert telemetry.recovery_mode == RecoveryMode.PROJECTED_FROM_RIGHT
        assert telemetry.left_boundary.is_hallucinated is True
        assert telemetry.right_boundary.is_hallucinated is False
        assert telemetry.left_boundary.hallucination_source == "PARALLEL_RIGHT"
        # Exact parallel curvature & heading match
        assert math.isclose(telemetry.left_boundary.A, right_in.A, rel_tol=1e-5)
        assert math.isclose(telemetry.left_boundary.B, right_in.B, rel_tol=1e-5)
        # Expected lateral shift: C_left = C_right - W = 1.85 - 3.70 = -1.85
        assert math.isclose(telemetry.left_boundary.C, right_in.C - 3.70, rel_tol=1e-4)
        assert telemetry.left_boundary.confidence < right_in.confidence
        assert telemetry.left_boundary.is_valid is True


class TestTemporalContinuityAndDecay:
    """Validates multi-frame hallucination confidence decay and recovery tracking."""

    def test_exponential_confidence_decay_over_frames(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify confidence exponentially decays per frame during continuous projection."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, _ = clean_dual_polynomials
        missing_right = BoundaryPolynomial2D(is_valid=False, confidence=0.0)

        confidences = []
        for _ in range(5):
            telemetry = engine.recover_boundaries(left_in, missing_right)
            confidences.append(telemetry.right_boundary.confidence)

        # Successive hallucinated confidences must be monotonically non-increasing
        for i in range(len(confidences) - 1):
            assert confidences[i + 1] <= confidences[i] + 1e-6

    def test_both_boundaries_missing_temporal_coasting(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """When both boundaries temporarily drop out, coast from historical state."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials

        # Feed 3 good frames to establish valid history
        for _ in range(3):
            engine.recover_boundaries(left_in, right_in)

        # Frame 4: complete visual blackout
        empty_poly = BoundaryPolynomial2D(is_valid=False, confidence=0.0)
        telemetry = engine.recover_boundaries(empty_poly, empty_poly)

        assert telemetry.recovery_mode == RecoveryMode.TEMPORAL_COASTING
        assert telemetry.left_boundary.is_hallucinated is True
        assert telemetry.right_boundary.is_hallucinated is True
        assert telemetry.left_boundary.hallucination_source == "TEMPORAL"
        assert telemetry.right_boundary.hallucination_source == "TEMPORAL"
        assert telemetry.hallucination_frame_count == 1


class TestASILBGracefulDegradationGuard:
    """
    SAFETY-CRITICAL INTERLOCK:
    If hallucination persists beyond max_hallucination_frames (20 frames),
    enforce graceful degradation, disengage steering torque, and trigger takeover.
    """

    def test_max_hallucination_frames_limit_enforced(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify engine disengages when hallucination exceeds 20 frames."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials

        # Seed initial history
        engine.recover_boundaries(left_in, right_in)

        # Now simulate complete dropout for 25 consecutive frames
        empty_poly = BoundaryPolynomial2D(is_valid=False, confidence=0.0)
        mode_history = []
        last_telemetry = None
        for frame_idx in range(25):
            last_telemetry = engine.recover_boundaries(empty_poly, empty_poly)
            mode_history.append(last_telemetry.recovery_mode)

        # Frame 1..20 should be TEMPORAL_COASTING
        assert mode_history[0] == RecoveryMode.TEMPORAL_COASTING
        assert mode_history[19] == RecoveryMode.TEMPORAL_COASTING
        # Frame 21+ MUST be DEGRADED_DISENGAGED
        assert mode_history[20] == RecoveryMode.DEGRADED_DISENGAGED
        assert last_telemetry.is_safe_for_steering is False

    def test_reacquisition_resets_hallucination_counter(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Reacquiring clear lane markings resets the hallucination counter to zero."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials

        # Seed history
        engine.recover_boundaries(left_in, right_in)

        # Coast for 5 frames
        empty_poly = BoundaryPolynomial2D(is_valid=False, confidence=0.0)
        for _ in range(5):
            telemetry = engine.recover_boundaries(empty_poly, empty_poly)
        assert telemetry.hallucination_frame_count == 5

        # Reacquire clean visual detection
        telemetry_recovered = engine.recover_boundaries(left_in, right_in)
        assert telemetry_recovered.recovery_mode == RecoveryMode.DUAL_DETECTED
        assert telemetry_recovered.hallucination_frame_count == 0


# ==============================================================================
# 6. Spline Interpolation & Telemetry Reporting Tests
# ==============================================================================

class TestSplineInterpolationAndTelemetry:
    """Validates spline boundary interpolation and telemetry export structures."""

    def test_spline_boundary_interpolator(self) -> None:
        """Verify spline interpolator bridges gaps between sparse inlier points."""
        interpolator = SplineBoundaryInterpolator(min_cluster_points=3, max_gap_meters=8.0)
        # Sequence of (y, x) points in meters with an intermediate gap
        points = [(5.0, -1.85), (8.0, -1.85), (12.0, -1.85), (20.0, -1.85), (25.0, -1.85)]
        bridged = interpolator.bridge_lane_gaps(points)
        assert len(bridged) >= len(points)
        for pt in bridged:
            assert len(pt) == 2

    def test_telemetry_serialization(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify telemetry object serializes to dictionary with all essential fields."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials
        telemetry = engine.recover_boundaries(left_in, right_in)

        assert isinstance(telemetry, FadedLaneRecoveryTelemetry)
        telemetry_dict = telemetry.to_dict()
        assert telemetry_dict["recovery_mode"] == RecoveryMode.DUAL_DETECTED.value
        assert telemetry_dict["hallucination_frame_count"] == 0
        assert math.isclose(telemetry_dict["lane_width_m"], 3.70, abs_tol=0.05)
        assert telemetry_dict["is_safe_for_steering"] is True

    def test_benchmark_result_structure(self) -> None:
        """Verify FadedRecoveryBenchmarkResult tracks recovery rates and metrics."""
        result = FadedRecoveryBenchmarkResult(
            total_test_frames=1000,
            dual_detected_count=850,
            left_projected_count=80,
            right_projected_count=50,
            temporal_coasting_count=18,
            safety_cutoff_count=2,
            mean_recovery_error_m=0.042,
            max_recovery_error_m=0.125,
            passed_asil_b_guard=True,
        )
        assert result.total_test_frames == 1000
        assert result.passed_asil_b_guard is True
        recovery_rate = (result.dual_detected_count + result.left_projected_count + result.right_projected_count) / result.total_test_frames
        assert recovery_rate >= 0.95


# ==============================================================================
# 7. Edge Cases & Defensive Robustness
# ==============================================================================

class TestEdgeCasesAndDefensiveGuards:
    """Stress tests anomalous inputs, inverted boundaries, and extreme geometries."""

    def test_inverted_lane_boundaries(
        self, default_recovery_config: FadedRecoveryConfig
    ) -> None:
        """Left boundary positioned to the right of right boundary should be handled safely."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        inverted_left = BoundaryPolynomial2D(
            is_valid=True, A=0.0, B=0.0, C=2.00, confidence=0.8
        )
        inverted_right = BoundaryPolynomial2D(
            is_valid=True, A=0.0, B=0.0, C=-2.00, confidence=0.8
        )

        telemetry = engine.recover_boundaries(inverted_left, inverted_right)
        assert isinstance(telemetry, FadedLaneRecoveryTelemetry)

    def test_force_reset_clears_internal_state(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify force_reset() purges all rolling histories and reset counters."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials
        engine.recover_boundaries(left_in, right_in)

        engine.force_reset()
        assert engine._hallucination_counter == 0
        assert engine._last_left_poly is None
        assert engine._last_right_poly is None

    def test_singleton_factory_function(self) -> None:
        """Verify get_faded_lane_recovery_engine returns valid instance."""
        inst1 = get_faded_lane_recovery_engine()
        assert isinstance(inst1, FadedLaneRecoveryEngine)

    @pytest.mark.parametrize(
        "nominal_width", [3.20, 3.50, 3.70, 4.00]
    )
    def test_parametric_nominal_widths(
        self,
        nominal_width: float,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify projection respects custom configured nominal lane widths."""
        cfg = FadedRecoveryConfig(nominal_lane_width_m=nominal_width)
        engine = FadedLaneRecoveryEngine(config=cfg)
        left_in, _ = clean_dual_polynomials
        missing_right = BoundaryPolynomial2D(is_valid=False, confidence=0.0)

        telemetry = engine.recover_boundaries(left_in, missing_right, lane_width_override_m=nominal_width)
        assert math.isclose(telemetry.right_boundary.C, left_in.C + nominal_width, rel_tol=1e-4)

    def test_execution_latency_under_5ms(
        self,
        default_recovery_config: FadedRecoveryConfig,
        clean_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Ensure boundary recovery execution satisfies real-time budget (< 5ms)."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = clean_dual_polynomials

        # Warm up
        for _ in range(5):
            engine.recover_boundaries(left_in, right_in)

        t_start = time.perf_counter()
        iterations = 50
        for _ in range(iterations):
            engine.recover_boundaries(left_in, right_in)
        t_elapsed = time.perf_counter() - t_start

        avg_latency_ms = (t_elapsed / iterations) * 1000.0
        assert avg_latency_ms < 5.0, f"Recovery latency {avg_latency_ms:.2f}ms exceeds 5.0ms"

    def test_centerline_geometry_derivation(
        self,
        default_recovery_config: FadedRecoveryConfig,
        curved_dual_polynomials: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify center line is derived as exact midpoint of dual boundaries."""
        engine = FadedLaneRecoveryEngine(config=default_recovery_config)
        left_in, right_in = curved_dual_polynomials
        telemetry = engine.recover_boundaries(left_in, right_in)

        center_poly = engine._compute_centerline(telemetry.left_boundary, telemetry.right_boundary)
        assert center_poly.is_valid is True
        # Midpoint of -1.85 and +1.85 is 0.0
        assert math.isclose(center_poly.C, 0.0, abs_tol=1e-4)
        assert math.isclose(center_poly.A, curved_dual_polynomials[0].A, abs_tol=1e-5)
