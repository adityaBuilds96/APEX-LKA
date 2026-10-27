"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

tests/test_lane_temporal_smoother.py
Comprehensive Verification Suite for 8-State Extended Kalman Filter & Motion Compensator
Validates:
    - 8-State EKF formulation [A_L, B_L, C_L, A_R, B_R, C_R, w_lane, theta_e]^T
    - Kinematic SE(2) ego-motion compensation (v_x, v_y, omega_z)
    - 2-Sigma Mahalanobis outlier gating and Chi-Square innovation testing
    - Dynamic measurement covariance adaptation R(t) = R_0 / (Q^2 + eps)
    - Dual-boundary parallel projection and single-boundary hallucination
    - Physical highway geometric constraints ([2.6m, 4.8m] width, curvature consensus)
    - Coasting prediction modes, max coast frame timeout, and diverged reset
    - Uncertainty ribbon propagation along lookahead evaluation horizons
    - Curvature dynamics analysis and lateral acceleration advisory speed
    - Stanley trajectory steering law coupling with quality-scaled scheduled gain
    - EKF covariance health monitoring (ASIL-B numerical stability)
    - CAN bus arbitration frames (0x160, 0x161, 0x162) bit-accurate encoding
    - Telemetry export and benchmark test harness execution
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pytest

from src.perception.lane_temporal_smoother import (
    ExtendedKalmanFilterCovarianceMonitor,
    LaneCurvatureDynamicsAnalyzer,
    LaneSmootherCANPublisher,
    LaneTemporalSmoother,
    LaneTemporalTelemetryExporter,
    PolynomialBand,
    SmoothedLaneEstimate,
    SmootherBenchmarkResults,
    StanleyTrajectorySteeringCoupler,
    TemporalSmootherConfig,
    TrackingFilterMode,
    get_lane_temporal_smoother,
    get_temporal_telemetry_exporter,
    run_temporal_smoother_diagnostics,
    run_temporal_smoother_stress_benchmark,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def default_config() -> TemporalSmootherConfig:
    """Provide default smoother configuration."""
    return TemporalSmootherConfig()


@pytest.fixture
def smoother(default_config: TemporalSmootherConfig) -> LaneTemporalSmoother:
    """Instantiate a fresh LaneTemporalSmoother for testing."""
    return LaneTemporalSmoother(config=default_config)


@pytest.fixture
def nominal_straight_measurement() -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Return nominal straight highway measurements (left, right) with 3.7m width."""
    left = (0.0, 0.0, -1.85)
    right = (0.0, 0.0, 1.85)
    return left, right


@pytest.fixture
def dynamics_analyzer() -> LaneCurvatureDynamicsAnalyzer:
    """Provide LaneCurvatureDynamicsAnalyzer with 2.0 m/s^2 comfort ceiling."""
    return LaneCurvatureDynamicsAnalyzer(max_lateral_accel_comfort_mps2=2.0)


@pytest.fixture
def stanley_coupler() -> StanleyTrajectorySteeringCoupler:
    """Provide Stanley trajectory steering coupler with nominal parameters."""
    return StanleyTrajectorySteeringCoupler(nominal_gain_k=0.85, softening_speed_mps=1.0)


@pytest.fixture
def cov_monitor() -> ExtendedKalmanFilterCovarianceMonitor:
    """Provide ASIL-B covariance monitor."""
    return ExtendedKalmanFilterCovarianceMonitor(max_allowed_trace=100.0, max_condition_number=1e8)


@pytest.fixture
def can_publisher() -> LaneSmootherCANPublisher:
    """Provide CAN bus publisher."""
    return LaneSmootherCANPublisher()


@pytest.fixture
def telemetry_exporter() -> LaneTemporalTelemetryExporter:
    """Provide telemetry exporter with 4 decimal precision."""
    return LaneTemporalTelemetryExporter(decimal_precision=4)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Initialization and Baseline Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTemporalSmootherInitialization:
    """Verify state vector initialization, dimensions, default matrices, and singletons."""

    def test_default_state_dimensions(self, smoother: LaneTemporalSmoother) -> None:
        """State dimension must be 8 and measurement observation dimension 6."""
        assert smoother.config.state_dim == 8
        assert smoother.config.measurement_dim == 6
        assert smoother._x.shape == (8, 1)
        assert smoother._P.shape == (8, 8)
        assert smoother._Q.shape == (8, 8)
        assert smoother._R0.shape == (6, 6)
        assert smoother._H.shape == (6, 8)

    def test_initial_state_values(self, smoother: LaneTemporalSmoother) -> None:
        """Initial state must represent nominal straight lane centered on ego vehicle."""
        assert smoother._mode == TrackingFilterMode.INITIALIZING
        assert math.isclose(smoother._x[0, 0], 0.0, abs_tol=1e-6)  # A_L
        assert math.isclose(smoother._x[1, 0], 0.0, abs_tol=1e-6)  # B_L
        assert math.isclose(smoother._x[2, 0], -1.85, abs_tol=1e-3)  # C_L
        assert math.isclose(smoother._x[3, 0], 0.0, abs_tol=1e-6)  # A_R
        assert math.isclose(smoother._x[4, 0], 0.0, abs_tol=1e-6)  # B_R
        assert math.isclose(smoother._x[5, 0], 1.85, abs_tol=1e-3)  # C_R
        assert math.isclose(smoother._x[6, 0], 3.70, abs_tol=1e-3)  # w_lane
        assert math.isclose(smoother._x[7, 0], 0.0, abs_tol=1e-6)  # theta_e

    def test_initial_covariance_symmetry_and_positive_definiteness(
        self, smoother: LaneTemporalSmoother, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Initial covariance matrix must be strictly symmetric and positive-definite."""
        health = cov_monitor.check_covariance_health(smoother._P)
        assert health["is_healthy"] is True
        assert health["min_eigenvalue"] > 0.0
        assert health["trace"] < 10.0

    def test_custom_config_injection(self) -> None:
        """Custom configuration parameters must be correctly propagated into smoother."""
        custom_cfg = TemporalSmootherConfig(
            nominal_lane_width_m=3.50,
            max_lane_width_m=4.50,
            min_lane_width_m=2.80,
            history_window_size=15,
            max_coast_frames=20,
        )
        custom_smoother = LaneTemporalSmoother(config=custom_cfg)
        assert math.isclose(custom_smoother._x[6, 0], 3.50, abs_tol=1e-3)
        assert math.isclose(custom_smoother._x[2, 0], -1.75, abs_tol=1e-3)
        assert math.isclose(custom_smoother._x[5, 0], 1.75, abs_tol=1e-3)
        assert custom_smoother.config.history_window_size == 15
        assert custom_smoother.config.max_coast_frames == 20

    def test_singleton_retrieval(self) -> None:
        """Singleton getter functions must return identical instances across multiple calls."""
        s1 = get_lane_temporal_smoother()
        s2 = get_lane_temporal_smoother()
        assert s1 is s2
        t1 = get_temporal_telemetry_exporter()
        t2 = get_temporal_telemetry_exporter()
        assert t1 is t2


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Kinematic Prediction & Ego-Motion Compensation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEKFPredictionAndMotionCompensation:
    """Verify SE(2) polynomial propagation, Jacobian formulation, and covariance growth."""

    def test_zero_motion_propagation(self, smoother: LaneTemporalSmoother) -> None:
        """With zero velocity and yaw rate, state vector should remain invariant while P expands."""
        x_before = smoother._x.copy()
        p_trace_before = float(np.trace(smoother._P))

        motion_active = smoother._predict_step(dt=0.05, vx=0.0, vy=0.0, omega_z=0.0)
        assert motion_active is False
        assert np.allclose(smoother._x, x_before)
        p_trace_after = float(np.trace(smoother._P))
        assert p_trace_after > p_trace_before

    def test_forward_velocity_longitudinal_shift(self, smoother: LaneTemporalSmoother) -> None:
        """
        When vehicle drives forward (vx > 0) on curved road:
        C' = C - delta_x + B*delta_y + A*delta_y^2
        B' = B + 2*A*delta_y - delta_psi
        """
        # Set non-zero curvature and heading on left and right lines
        smoother._x[0, 0] = 0.0010  # A_L
        smoother._x[1, 0] = 0.0200  # B_L
        smoother._x[2, 0] = -1.85   # C_L
        smoother._x[3, 0] = 0.0010  # A_R
        smoother._x[4, 0] = 0.0200  # B_R
        smoother._x[5, 0] = 1.85    # C_R

        dt = 0.1
        vx = 20.0  # 20 m/s -> delta_y = 2.0 m
        motion_active = smoother._predict_step(dt=dt, vx=vx, vy=0.0, omega_z=0.0)
        assert motion_active is True

        delta_y = 2.0
        expected_c_l = -1.85 + 0.0200 * delta_y + 0.0010 * (delta_y ** 2)
        expected_b_l = 0.0200 + 2.0 * 0.0010 * delta_y

        assert math.isclose(smoother._x[2, 0], expected_c_l, abs_tol=1e-5)
        assert math.isclose(smoother._x[1, 0], expected_b_l, abs_tol=1e-5)
        assert math.isclose(smoother._x[0, 0], 0.0010, abs_tol=1e-6)

    def test_lateral_drift_compensation(self, smoother: LaneTemporalSmoother) -> None:
        """Lateral velocity vy should shift lateral offsets C_L and C_R by -delta_x."""
        c_l_init = float(smoother._x[2, 0])
        c_r_init = float(smoother._x[5, 0])

        dt = 0.05
        vy = 0.8  # Drift right at 0.8 m/s -> delta_x = 0.04 m
        smoother._predict_step(dt=dt, vx=10.0, vy=vy, omega_z=0.0)

        delta_x = vy * dt
        assert math.isclose(smoother._x[2, 0], c_l_init - delta_x, abs_tol=1e-4)
        assert math.isclose(smoother._x[5, 0], c_r_init - delta_x, abs_tol=1e-4)

    def test_yaw_rate_heading_tilt_compensation(self, smoother: LaneTemporalSmoother) -> None:
        """Ego yaw rate omega_z rotates vehicle frame, subtracting delta_psi from B and theta_e."""
        dt = 0.05
        omega_z = 0.05  # rad/s -> delta_psi = 0.0025 rad
        smoother._predict_step(dt=dt, vx=15.0, vy=0.0, omega_z=omega_z)

        delta_psi = omega_z * dt
        assert math.isclose(smoother._x[7, 0], -delta_psi, abs_tol=1e-5)
        assert math.isclose(smoother._x[1, 0], -delta_psi, abs_tol=1e-5)
        assert math.isclose(smoother._x[4, 0], -delta_psi, abs_tol=1e-5)

    def test_repeated_prediction_covariance_symmetry(
        self, smoother: LaneTemporalSmoother, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Predicting 15 steps (within coast limit) retains ASIL-B health, and 100 steps retains exact symmetry."""
        for _ in range(15):
            smoother._predict_step(dt=0.0333, vx=25.0, vy=0.1, omega_z=0.02)

        health_15 = cov_monitor.check_covariance_health(smoother._P)
        assert health_15["is_healthy"] is True
        assert health_15["symmetry_max_error"] < 1e-6

        # Continue up to 100 steps: symmetry must remain strictly zero
        for _ in range(85):
            smoother._predict_step(dt=0.0333, vx=25.0, vy=0.1, omega_z=0.02)

        sym_error = float(np.max(np.abs(smoother._P - smoother._P.T)))
        assert sym_error < 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Measurement Preparation & Boundary Hallucination Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestMeasurementPreparationAndHallucination:
    """Verify 6-dimensional measurement vector synthesis and parallel boundary projection."""

    def test_dual_boundary_valid_measurement(self, smoother: LaneTemporalSmoother) -> None:
        """When both left and right boundaries are provided, no hallucination occurs."""
        left = (0.0005, 0.01, -1.80)
        right = (0.0005, 0.01, 1.90)

        z, mask, synthesized = smoother._prepare_measurement_vector(left, right)
        assert synthesized is False
        assert all(mask)
        assert np.allclose(z[0:3, 0], [0.0005, 0.01, -1.80])
        assert np.allclose(z[3:6, 0], [0.0005, 0.01, 1.90])

    def test_left_only_hallucinates_parallel_right(self, smoother: LaneTemporalSmoother) -> None:
        """When only left boundary is visible, right boundary is projected at C_L + w_lane."""
        left = (0.0008, -0.015, -1.85)
        w_current = float(smoother._x[6, 0])

        z, mask, synthesized = smoother._prepare_measurement_vector(left, None)
        assert synthesized is True
        assert mask == [True, True, True, False, False, False]
        # Right line must have identical curvature & heading, and offset shifted right by w_lane
        assert math.isclose(z[3, 0], 0.0008, abs_tol=1e-6)
        assert math.isclose(z[4, 0], -0.015, abs_tol=1e-6)
        assert math.isclose(z[5, 0], -1.85 + w_current, abs_tol=1e-4)

    def test_right_only_hallucinates_parallel_left(self, smoother: LaneTemporalSmoother) -> None:
        """When only right boundary is visible, left boundary is projected at C_R - w_lane."""
        right = (-0.0006, 0.02, 1.85)
        w_current = float(smoother._x[6, 0])

        z, mask, synthesized = smoother._prepare_measurement_vector(None, right)
        assert synthesized is True
        assert mask == [False, False, False, True, True, True]
        # Left line must have identical curvature & heading, and offset shifted left by w_lane
        assert math.isclose(z[0, 0], -0.0006, abs_tol=1e-6)
        assert math.isclose(z[1, 0], 0.02, abs_tol=1e-6)
        assert math.isclose(z[2, 0], 1.85 - w_current, abs_tol=1e-4)

    def test_adaptive_covariance_scales_inversely_with_quality(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Measurement noise covariance R(t) must increase quadratically as quality Q drops."""
        mask = [True] * 6
        r_high_q = smoother._compute_adaptive_measurement_covariance(quality=1.0, active_mask=mask)
        r_low_q = smoother._compute_adaptive_measurement_covariance(quality=0.2, active_mask=mask)

        # Ratio low_q / high_q should be approximately (1.0 / 0.04) = 25
        ratio_diag = r_low_q[2, 2] / r_high_q[2, 2]
        assert 20.0 < ratio_diag < 30.0

    def test_unmeasured_boundary_covariance_penalty(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Synthesized or occluded boundaries receive a 10x measurement noise inflation."""
        mask = [True, True, True, False, False, False]
        r = smoother._compute_adaptive_measurement_covariance(quality=1.0, active_mask=mask)

        # Right line diagonals (indices 3, 4, 5) should be 10x larger than corresponding left line diagonals
        assert math.isclose(r[3, 3], r[0, 0] * 10.0, rel_tol=1e-3)
        assert math.isclose(r[4, 4], r[1, 1] * 10.0, rel_tol=1e-3)
        assert math.isclose(r[5, 5], r[2, 2] * 10.0, rel_tol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Outlier Gating & Chi-Square Innovation Testing
# ═══════════════════════════════════════════════════════════════════════════════

class TestMahalanobisOutlierGating:
    """Verify Chi-Square distance evaluation and lane width jump gating."""

    def test_nominal_measurement_passes_gating(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Nominal measurements must pass Mahalanobis gating with distance < 2.0."""
        left, right = nominal_straight_measurement
        z, mask, _ = smoother._prepare_measurement_vector(left, right)
        r = smoother._compute_adaptive_measurement_covariance(quality=1.0, active_mask=mask)

        is_outlier, d_m = smoother._evaluate_outlier_gating(z, r, mask)
        assert is_outlier is False
        assert d_m < 2.0

    def test_massive_lateral_jump_rejected_by_mahalanobis(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """A sudden 3.0m jump in lateral position must trigger Chi-square gating outlier rejection."""
        # Baseline update to warm up
        left = (0.0, 0.0, -1.85)
        right = (0.0, 0.0, 1.85)
        smoother.update(left, right, perception_quality=0.95)

        # Huge jump measurement
        corrupted_left = (0.0, 0.0, -4.85)
        z, mask, _ = smoother._prepare_measurement_vector(corrupted_left, right)
        r = smoother._compute_adaptive_measurement_covariance(quality=0.95, active_mask=mask)

        is_outlier, d_m = smoother._evaluate_outlier_gating(z, r, mask)
        assert is_outlier is True
        assert d_m > math.sqrt(smoother.config.mahalanobis_gating_threshold_chi2)

    def test_heading_spike_rejected_by_gating(self, smoother: LaneTemporalSmoother) -> None:
        """A heading spike of 0.4 rad (~23 deg) must be rejected as an innovation outlier."""
        for _ in range(5):
            smoother.update((0.0, 0.0, -1.85), (0.0, 0.0, 1.85), perception_quality=0.95)

        heading_spike_left = (0.0, 0.45, -1.85)
        z, mask, _ = smoother._prepare_measurement_vector(heading_spike_left, (0.0, 0.0, 1.85))
        r = smoother._compute_adaptive_measurement_covariance(quality=0.95, active_mask=mask)

        is_outlier, d_m = smoother._evaluate_outlier_gating(z, r, mask)
        assert is_outlier is True
        assert d_m > 3.0

    def test_lane_width_2_sigma_jump_rejection(self, smoother: LaneTemporalSmoother) -> None:
        """When measured width suddenly jumps from 3.7m to 5.2m, 2-sigma width check flags outlier."""
        # Establish consistent width history
        for _ in range(6):
            smoother.update((0.0, 0.0, -1.85), (0.0, 0.0, 1.85), perception_quality=0.95)

        # Provide measurement with 5.5m width (Left at -2.75, Right at +2.75)
        wide_left = (0.0, 0.0, -2.75)
        wide_right = (0.0, 0.0, 2.75)

        res = smoother.update(wide_left, wide_right, perception_quality=0.95)
        assert res.outlier_rejected is True

    def test_rejected_outlier_does_not_corrupt_state(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """When an outlier is rejected, state vector must NOT jump to corrupted measurement."""
        smoother.update((0.0, 0.0, -1.85), (0.0, 0.0, 1.85), perception_quality=0.95)
        c_l_nominal = float(smoother._x[2, 0])

        corrupted_left = (0.05, 0.3, -5.50)
        res = smoother.update(corrupted_left, (0.0, 0.0, 1.85), perception_quality=0.95)

        assert res.outlier_rejected is True
        # Tracked C_L should remain close to nominal (-1.85m), not -5.50m
        assert abs(smoother._x[2, 0] - c_l_nominal) < 0.20


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Kalman Measurement Update & Joseph Form Covariance
# ═══════════════════════════════════════════════════════════════════════════════

class TestKalmanMeasurementUpdate:
    """Verify state convergence, gain properties, and Joseph-form covariance stability."""

    def test_state_convergence_towards_steady_measurement(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Starting from 0 offset, persistent measurements at C_L = -1.70 should pull state there."""
        for _ in range(15):
            smoother.update((0.0, 0.0, -1.70), (0.0, 0.0, 2.00), perception_quality=0.95)

        assert math.isclose(smoother._x[2, 0], -1.70, abs_tol=0.03)
        assert math.isclose(smoother._x[5, 0], 2.00, abs_tol=0.03)

    def test_joseph_form_maintains_symmetry_and_positive_definiteness(
        self, smoother: LaneTemporalSmoother, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Joseph form P = (I - KH)P(I - KH)^T + KRK^T must remain strictly symmetric."""
        left = (0.0002, 0.005, -1.82)
        right = (0.0002, 0.005, 1.88)

        for _ in range(20):
            smoother.update(left, right, perception_quality=0.90)
            health = cov_monitor.check_covariance_health(smoother._P)
            assert health["is_healthy"] is True
            assert health["symmetry_max_error"] < 1e-6

    def test_update_weight_scales_with_quality(self) -> None:
        """High quality measurements produce larger Kalman updates than low quality measurements."""
        smoother_hq = LaneTemporalSmoother()
        smoother_lq = LaneTemporalSmoother()

        # Step measurement from -1.85 to -1.65
        step_meas_l = (0.0, 0.0, -1.65)
        step_meas_r = (0.0, 0.0, 2.05)

        smoother_hq.update(step_meas_l, step_meas_r, perception_quality=1.0)
        smoother_lq.update(step_meas_l, step_meas_r, perception_quality=0.2)

        # Delta moved by HQ smoother should be strictly larger than LQ smoother
        delta_hq = abs(smoother_hq._x[2, 0] - (-1.85))
        delta_lq = abs(smoother_lq._x[2, 0] - (-1.85))
        assert delta_hq > delta_lq * 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Physical Geometric Constraints & Boundary Enforcements
# ═══════════════════════════════════════════════════════════════════════════════

class TestGeometricConstraintsAndBoundaryEnforcement:
    """Verify highway width clamping, curvature consensus, and heading consistency."""

    def test_lane_width_clamped_to_physical_bounds(self, smoother: LaneTemporalSmoother) -> None:
        """Measurements suggesting lane width outside [2.6m, 4.8m] must have tracked width clamped."""
        # Unusually narrow lane (2.2m)
        narrow_l = (0.0, 0.0, -1.10)
        narrow_r = (0.0, 0.0, 1.10)
        for _ in range(10):
            smoother.update(narrow_l, narrow_r, perception_quality=0.85)

        assert smoother._x[6, 0] >= smoother.config.min_lane_width_m

        # Unusually wide lane (5.5m)
        wide_l = (0.0, 0.0, -2.75)
        wide_r = (0.0, 0.0, 2.75)
        for _ in range(10):
            smoother.update(wide_l, wide_r, perception_quality=0.85)

        assert smoother._x[6, 0] <= smoother.config.max_lane_width_m

    def test_curvature_consensus_blending(self, smoother: LaneTemporalSmoother) -> None:
        """Independent left and right curvatures A_L and A_R must blend towards parallel consensus."""
        # Set disparate curvatures
        smoother._x[0, 0] = 0.0020  # A_L
        smoother._x[3, 0] = 0.0004  # A_R

        smoother._enforce_parallel_boundary_constraints()

        # Both curvatures should move towards their mean (0.0012)
        assert abs(smoother._x[0, 0] - 0.0012) < abs(0.0020 - 0.0012)
        assert abs(smoother._x[3, 0] - 0.0012) < abs(0.0004 - 0.0012)

    def test_heading_error_derived_from_centerline(self, smoother: LaneTemporalSmoother) -> None:
        """Centerline heading error theta_e must match average of left and right heading slopes."""
        smoother._x[1, 0] = 0.04  # B_L
        smoother._x[4, 0] = 0.06  # B_R

        smoother._enforce_parallel_boundary_constraints()
        assert math.isclose(smoother._x[7, 0], 0.05, abs_tol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Filter Mode Transitions & Coasting Life Cycle
# ═══════════════════════════════════════════════════════════════════════════════

class TestFilterModeTransitionsAndCoasting:
    """Verify transitions between INITIALIZING, DUAL_TRACKING, COASTING, and DIVERGED_RESET."""

    def test_transition_to_dual_tracking(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Receiving both valid boundaries transitions filter mode to DUAL_TRACKING."""
        left, right = nominal_straight_measurement
        res = smoother.update(left, right, perception_quality=0.95)
        assert res.filter_mode == TrackingFilterMode.DUAL_TRACKING

    def test_transition_to_single_left_coast(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Occluding right boundary transitions mode to SINGLE_LEFT_COAST."""
        left, _ = nominal_straight_measurement
        res = smoother.update(left, None, perception_quality=0.80)
        assert res.filter_mode == TrackingFilterMode.SINGLE_LEFT_COAST

    def test_transition_to_single_right_coast(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Occluding left boundary transitions mode to SINGLE_RIGHT_COAST."""
        _, right = nominal_straight_measurement
        res = smoother.update(None, right, perception_quality=0.80)
        assert res.filter_mode == TrackingFilterMode.SINGLE_RIGHT_COAST

    def test_coasting_prediction_on_total_occlusion(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Occluding both boundaries enters COASTING_PREDICTION mode."""
        res = smoother.update(None, None, perception_quality=0.20)
        assert res.filter_mode == TrackingFilterMode.COASTING_PREDICTION

    def test_max_coast_exceeded_triggers_diverged_reset(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Exceeding max_coast_frames (15) without measurements resets smoother to nominal state."""
        smoother.update((0.0, 0.0, -1.85), (0.0, 0.0, 1.85), perception_quality=0.95)

        # Coast for 16 consecutive frames
        last_res = None
        for _ in range(16):
            last_res = smoother.update(None, None, perception_quality=0.10)

        assert last_res is not None
        assert last_res.filter_mode == TrackingFilterMode.DIVERGED_RESET
        assert math.isclose(smoother._x[6, 0], 3.70, abs_tol=1e-3)

    def test_manual_reset(self, smoother: LaneTemporalSmoother) -> None:
        """Calling reset() restores nominal state vector and clears rolling histories."""
        smoother._x[2, 0] = -3.00
        smoother._history_width.append(4.2)
        smoother._history_confidence.append(0.5)

        smoother.reset()
        assert math.isclose(smoother._x[2, 0], -1.85, abs_tol=1e-3)
        assert len(smoother._history_width) == 0
        assert len(smoother._history_confidence) == 0
        assert smoother._mode == TrackingFilterMode.INITIALIZING


# ═══════════════════════════════════════════════════════════════════════════════
# 8. Uncertainty Ribbon Propagation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestUncertaintyRibbonPropagation:
    """Verify 1-sigma and 2-sigma uncertainty ribbon evaluation along lookahead horizon."""

    def test_ribbon_longitudinal_points_match_config(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Uncertainty ribbons must evaluate at all configured longitudinal points."""
        left, right = nominal_straight_measurement
        res = smoother.update(left, right, perception_quality=0.95)

        expected_pts = smoother.config.eval_longitudinal_points_m
        assert len(res.left_uncertainty_ribbon) == len(expected_pts)
        assert len(res.right_uncertainty_ribbon) == len(expected_pts)

        for band, pt in zip(res.left_uncertainty_ribbon, expected_pts):
            assert math.isclose(band.y_eval_m, pt, abs_tol=1e-3)
            assert band.sigma_2_m == pytest.approx(band.sigma_1_m * 2.0, rel=1e-5)

    def test_uncertainty_increases_with_lookahead_distance(
        self, smoother: LaneTemporalSmoother, nominal_straight_measurement: Tuple[Any, Any]
    ) -> None:
        """Uncertainty sigma_1 must grow monotonically with distance y due to polynomial extrapolation."""
        left, right = nominal_straight_measurement
        res = smoother.update(left, right, perception_quality=0.95)

        sigmas_l = [b.sigma_1_m for b in res.left_uncertainty_ribbon]
        sigmas_r = [b.sigma_1_m for b in res.right_uncertainty_ribbon]

        # Monotonically non-decreasing
        for i in range(len(sigmas_l) - 1):
            assert sigmas_l[i + 1] >= sigmas_l[i]
            assert sigmas_r[i + 1] >= sigmas_r[i]

    def test_evaluated_coordinates_match_polynomial_values(
        self, smoother: LaneTemporalSmoother
    ) -> None:
        """Evaluated ribbon mean x coordinates must exactly match x(y) = A*y^2 + B*y + C."""
        poly_l = (0.0005, -0.01, -1.80)
        poly_r = (0.0005, -0.01, 1.90)

        for _ in range(5):
            res = smoother.update(poly_l, poly_r, perception_quality=0.95)

        a_l, b_l, c_l = res.left_poly_coeffs
        for band in res.left_uncertainty_ribbon:
            y = band.y_eval_m
            expected_x = a_l * (y ** 2) + b_l * y + c_l
            assert math.isclose(band.x_mean_m, expected_x, abs_tol=1e-5)


# ═══════════════════════════════════════════════════════════════════════════════
# 9. Curvature Dynamics Analysis Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCurvatureDynamicsAnalyzer:
    """Verify kappa, radius of curvature, lateral acceleration demand, and advisory speed."""

    def test_straight_road_zero_curvature(
        self, dynamics_analyzer: LaneCurvatureDynamicsAnalyzer
    ) -> None:
        """A straight road (A=0, B=0) must exhibit kappa=0 and radius clamped to 10000m."""
        metrics = dynamics_analyzer.analyze_curvature(
            poly_coeffs=(0.0, 0.0, -1.85),
            eval_distance_m=15.0,
            current_speed_mps=20.0,
        )
        assert metrics["curvature_kappa_radpm"] == 0.0
        assert metrics["radius_of_curvature_m"] == 10000.0
        assert metrics["lateral_accel_demand_mps2"] == 0.0
        assert metrics["exceeds_comfort_limit"] is False

    def test_constant_radius_curve_calculation(
        self, dynamics_analyzer: LaneCurvatureDynamicsAnalyzer
    ) -> None:
        """For x(y) = 0.001*y^2, at y=0, kappa = 2*A = 0.002 -> Radius = 500m."""
        metrics = dynamics_analyzer.analyze_curvature(
            poly_coeffs=(0.001, 0.0, -1.85),
            eval_distance_m=0.0,
            current_speed_mps=20.0,
        )
        assert math.isclose(metrics["curvature_kappa_radpm"], 0.002, abs_tol=1e-5)
        assert math.isclose(metrics["radius_of_curvature_m"], 500.0, abs_tol=1.0)

        # Lateral acceleration at 20 m/s: a_lat = v^2 * kappa = 400 * 0.002 = 0.8 m/s^2
        assert math.isclose(metrics["lateral_accel_demand_mps2"], 0.80, abs_tol=0.02)
        assert metrics["exceeds_comfort_limit"] is False

    def test_sharp_curve_triggers_comfort_limit_and_advisory_speed(
        self, dynamics_analyzer: LaneCurvatureDynamicsAnalyzer
    ) -> None:
        """A sharp curve with radius 125m at 25 m/s (~90 km/h) triggers comfort ceiling exceedance."""
        # A = 0.004 -> kappa ~ 0.008 -> Radius ~ 125m
        metrics = dynamics_analyzer.analyze_curvature(
            poly_coeffs=(0.004, 0.0, -1.85),
            eval_distance_m=0.0,
            current_speed_mps=25.0,
        )
        # a_lat = 625 * 0.008 = 5.0 m/s^2 > 2.0 m/s^2 comfort limit
        assert metrics["exceeds_comfort_limit"] is True
        # Advisory speed for 2.0 m/s^2 limit: sqrt(2.0 / 0.008) = 15.81 m/s (~56.9 km/h)
        assert math.isclose(metrics["advisory_speed_mps"], 15.81, abs_tol=0.1)
        assert metrics["advisory_speed_kph"] < 60.0


# ═══════════════════════════════════════════════════════════════════════════════
# 10. Stanley Trajectory Steering Law Coupling Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStanleyTrajectorySteeringCoupler:
    """Verify Stanley cross-track steering derivation, gain scheduling, and actuator saturation."""

    def test_zero_lateral_error_steers_straight(
        self,
        stanley_coupler: StanleyTrajectorySteeringCoupler,
        smoother: LaneTemporalSmoother,
        nominal_straight_measurement: Tuple[Any, Any],
    ) -> None:
        """When vehicle is centered in straight lane, commanded steer angle is 0 degrees."""
        left, right = nominal_straight_measurement
        est = smoother.update(left, right, perception_quality=0.95)

        cmd = stanley_coupler.compute_steering_command(estimate=est, ego_speed_mps=20.0)
        assert math.isclose(cmd["lateral_error_m"], 0.0, abs_tol=0.05)
        assert math.isclose(cmd["commanded_steer_angle_deg"], 0.0, abs_tol=0.2)
        assert cmd["is_actuator_saturated"] is False

    def test_lateral_offset_triggers_corrective_steering(
        self,
        stanley_coupler: StanleyTrajectorySteeringCoupler,
        smoother: LaneTemporalSmoother,
    ) -> None:
        """If vehicle is shifted left (centerline is at +0.3m), coupler commands right steer."""
        left = (0.0, 0.0, -1.55)
        right = (0.0, 0.0, 2.15)
        for _ in range(5):
            est = smoother.update(left, right, perception_quality=0.95)

        cmd = stanley_coupler.compute_steering_command(estimate=est, ego_speed_mps=15.0)
        # Vehicle is to the left of center -> e_lat = -0.3m
        assert cmd["lateral_error_m"] < -0.20
        # Command must be negative (steer right towards center)
        assert cmd["commanded_steer_angle_rad"] < 0.0

    def test_gain_scheduling_attenuates_under_low_quality(
        self,
        stanley_coupler: StanleyTrajectorySteeringCoupler,
        smoother: LaneTemporalSmoother,
    ) -> None:
        """Scheduled gain k must drop linearly with perception quality Q to prevent jitter."""
        left = (0.0, 0.0, -1.50)
        right = (0.0, 0.0, 2.20)

        # High quality
        est_hq = smoother.update(left, right, perception_quality=1.0)
        cmd_hq = stanley_coupler.compute_steering_command(estimate=est_hq, ego_speed_mps=20.0)

        # Low quality
        smoother.reset()
        est_lq = smoother.update(left, right, perception_quality=0.40)
        cmd_lq = stanley_coupler.compute_steering_command(estimate=est_lq, ego_speed_mps=20.0)

        assert cmd_hq["scheduled_gain_k"] > cmd_lq["scheduled_gain_k"]
        assert math.isclose(cmd_lq["scheduled_gain_k"], stanley_coupler.k_nominal * 0.40, rel_tol=1e-2)

    def test_steer_angle_clipping_and_saturation_flag(
        self,
        stanley_coupler: StanleyTrajectorySteeringCoupler,
        smoother: LaneTemporalSmoother,
    ) -> None:
        """Steering commands exceeding 28 degrees must be clipped and flag saturation."""
        # Align heading error and lateral error so both demand high steer
        smoother._x[7, 0] = math.radians(30.0)
        smoother._x[1, 0] = 0.0
        smoother._x[4, 0] = 0.0
        # Centerline to the left (vehicle to the right): e_lat = +2.0m
        smoother._x[2, 0] = -3.85
        smoother._x[5, 0] = -0.15

        est = smoother._build_output_estimate(False, False, 0.0, 1.0, time.perf_counter())
        cmd = stanley_coupler.compute_steering_command(
            estimate=est, ego_speed_mps=1.0, lookahead_distance_m=0.0
        )

        assert cmd["is_actuator_saturated"] is True
        assert abs(cmd["commanded_steer_angle_rad"]) == pytest.approx(stanley_coupler.max_steer, rel=1e-4)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. Extended Kalman Filter Covariance Monitor Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestExtendedKalmanFilterCovarianceMonitor:
    """Verify ASIL-B numerical integrity, symmetry checks, and condition number thresholds."""

    def test_healthy_covariance_passes(
        self, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Standard diagonal covariance passes all health checks."""
        P = np.diag([1e-4, 1e-2, 0.25, 1e-4, 1e-2, 0.25, 0.05, 1e-2])
        health = cov_monitor.check_covariance_health(P)
        assert health["is_healthy"] is True
        assert health["trace"] < cov_monitor.max_trace

    def test_asymmetric_covariance_fails_health(
        self, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Asymmetric covariance matrix must be detected and fail health check."""
        P = np.diag([1.0] * 8)
        P[0, 1] = 0.5  # Create asymmetry
        P[1, 0] = 0.0
        health = cov_monitor.check_covariance_health(P)
        assert health["is_healthy"] is False
        assert health["symmetry_max_error"] > 1e-3

    def test_negative_eigenvalue_fails_health(
        self, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Non-positive definite matrix with negative eigenvalue fails health check."""
        P = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, -0.05])
        health = cov_monitor.check_covariance_health(P)
        assert health["is_healthy"] is False
        assert health["min_eigenvalue"] < 0.0

    def test_excessive_trace_fails_health(
        self, cov_monitor: ExtendedKalmanFilterCovarianceMonitor
    ) -> None:
        """Covariance with trace exceeding max_allowed_trace fails health check."""
        P = np.diag([20.0] * 8)  # Trace = 160 > 100
        health = cov_monitor.check_covariance_health(P)
        assert health["is_healthy"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# 12. CAN Bus Publishing & Payload Encoding Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneSmootherCANPublisher:
    """Verify bit-accurate CAN frame encoding for IDs 0x160, 0x161, and 0x162."""

    def test_encode_left_boundary_frame_0x160(
        self,
        can_publisher: LaneSmootherCANPublisher,
        smoother: LaneTemporalSmoother,
        nominal_straight_measurement: Tuple[Any, Any],
    ) -> None:
        """0x160 encodes A_L*1e6, B_L*1e3, C_L*100, and Quality*255 into 8 bytes."""
        left, right = nominal_straight_measurement
        est = smoother.update(left, right, perception_quality=0.80)

        frame = can_publisher.encode_left_boundary_frame(est)
        assert frame["arbitration_id"] == 0x160
        assert frame["dlc"] == 8
        assert len(frame["payload_bytes"]) == 8

        # Quality byte is index 6: 0.80 * 255 = 204 = 0xCC
        assert frame["payload_bytes"][6] == 204

        # C_L = -1.85m -> -185 = 0xFF47 (int16 two's complement)
        c_val = (frame["payload_bytes"][4] << 8) | frame["payload_bytes"][5]
        if c_val >= 32768:
            c_val -= 65536
        assert math.isclose(c_val, -185, abs_tol=3)

    def test_encode_right_boundary_frame_0x161(
        self,
        can_publisher: LaneSmootherCANPublisher,
        smoother: LaneTemporalSmoother,
        nominal_straight_measurement: Tuple[Any, Any],
    ) -> None:
        """0x161 encodes right boundary polynomial coefficients and quality."""
        left, right = nominal_straight_measurement
        est = smoother.update(left, right, perception_quality=0.90)

        frame = can_publisher.encode_right_boundary_frame(est)
        assert frame["arbitration_id"] == 0x161
        assert frame["dlc"] == 8

        # C_R = +1.85m -> +185
        c_val = (frame["payload_bytes"][4] << 8) | frame["payload_bytes"][5]
        assert math.isclose(c_val, 185, abs_tol=3)

    def test_encode_lane_geometry_frame_0x162(
        self,
        can_publisher: LaneSmootherCANPublisher,
        smoother: LaneTemporalSmoother,
        nominal_straight_measurement: Tuple[Any, Any],
    ) -> None:
        """0x162 encodes lane width (cm), heading error (0.01 deg), outlier flag, and mode byte."""
        left, right = nominal_straight_measurement
        est = smoother.update(left, right, perception_quality=0.95)

        frame = can_publisher.encode_lane_geometry_frame(est)
        assert frame["arbitration_id"] == 0x162
        assert frame["dlc"] == 8

        # Width in cm: 3.70m -> 370 cm
        w_val = (frame["payload_bytes"][0] << 8) | frame["payload_bytes"][1]
        assert math.isclose(w_val, 370, abs_tol=5)

        # Mode byte for DUAL_TRACKING is 0x01
        assert frame["payload_bytes"][5] == 0x01
        assert frame["payload_bytes"][4] == 0  # outlier rejected flag


# ═══════════════════════════════════════════════════════════════════════════════
# 13. Telemetry Exporter Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestLaneTemporalTelemetryExporter:
    """Verify JSON dictionary export and coordinate structures."""

    def test_export_structure(
        self,
        telemetry_exporter: LaneTemporalTelemetryExporter,
        smoother: LaneTemporalSmoother,
        nominal_straight_measurement: Tuple[Any, Any],
    ) -> None:
        """Export dictionary must contain structured uncertainty ribbons and metadata."""
        left, right = nominal_straight_measurement
        est = smoother.update(left, right, perception_quality=0.95)

        data = telemetry_exporter.to_uncertainty_ribbon_dict(est)
        assert "timestamp" in data
        assert data["filter_mode"] == TrackingFilterMode.DUAL_TRACKING.value
        assert math.isclose(data["lane_width_m"], 3.70, abs_tol=0.05)
        assert len(data["left_band"]) == 6
        assert len(data["right_band"]) == 6

        first_pt = data["left_band"][0]
        assert "y_m" in first_pt
        assert "x_mean_m" in first_pt
        assert "sigma_1_m" in first_pt
        assert "sigma_2_m" in first_pt


# ═══════════════════════════════════════════════════════════════════════════════
# 14. Stress Benchmark & Self-Diagnostics Execution Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestStressBenchmarkAndDiagnostics:
    """Verify built-in diagnostics and stress benchmark execution."""

    def test_run_temporal_smoother_diagnostics(self) -> None:
        """Built-in diagnostics suite must execute cleanly and return True."""
        result = run_temporal_smoother_diagnostics(verbose=False)
        assert result is True

    def test_run_stress_benchmark_short_run(self) -> None:
        """Stress benchmark with 50 frames must complete without numerical divergence."""
        res: SmootherBenchmarkResults = run_temporal_smoother_stress_benchmark(num_frames=50)
        assert res.total_frames == 50
        assert res.mean_latency_ms < 5.0
        assert 3.50 < res.mean_lane_width_m < 3.90
        assert res.std_lane_width_m < 0.20
        assert res.dual_tracking_count > 0
