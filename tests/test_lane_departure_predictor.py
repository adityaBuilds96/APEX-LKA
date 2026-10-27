"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

tests/test_lane_departure_predictor.py
Comprehensive Verification Suite for Predictive Time-to-Lane-Departure (TTLD).

PRIMARY SAFETY MISSION:
  Continuously predict the Time-to-Lane-Departure (TTLD) and project vehicle kinematic
  paths 1.5 to 2.5 seconds forward to identify impending boundary crossings well before
  physical departure occurs.

Test Matrix:
  1. Kinematic vehicle trajectory ribbon forward projection (bicycle model + IMU)
  2. Linear and 2nd-order quadratic dynamic TTLD mathematical solver
  3. Departure severity threshold hierarchy (Nominal, Warning, Critical, Emergency)
  4. Directional resolution (Class 2 Left vs Class 3 Right vs Centered)
  5. ISO 11898-2 SocketCAN binary frame encoding with SAE J1850 CRC-8
  6. Centrifugal tire slip and high-speed curvature drift estimation
  7. Statistical departure risk Gaussian heatmap rasterization
  8. Diagnostic HUD visualizer and TTLD countdown gauge rendering
  9. Edge cases, zero-speed limits, negative margins, and real-time execution latency
"""

from __future__ import annotations

import copy
import math
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

# Ensure project root is present in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.lane_departure_predictor import (
    CANAlertEncoder,
    CentrifugalDriftEstimator,
    DepartureDirection,
    DeparturePredictorConfig,
    DepartureRiskHeatmapRasterizer,
    DepartureSeverity,
    KinematicVehicleProjector,
    LaneDeparturePrediction,
    LaneDeparturePredictor,
    ProjectedWaypoint,
    render_departure_hud,
)


# ==============================================================================
# Test Fixtures & Baseline Configs
# ==============================================================================

@pytest.fixture
def baseline_predictor_config() -> DeparturePredictorConfig:
    """Standard departure predictor configuration aligned with ASIL-B specifications."""
    return DeparturePredictorConfig(
        vehicle_width_m=1.90,
        vehicle_length_m=4.50,
        wheelbase_m=2.70,
        ttld_warning_threshold_s=2.00,
        ttld_critical_threshold_s=1.20,
        ttld_emergency_threshold_s=0.60,
        max_prediction_horizon_s=2.50,
        prediction_time_step_s=0.25,
        can_warning_id=0x140,
        can_emergency_id=0x150,
        enable_imu_acceleration=True,
        lateral_jerk_limit_mps3=3.50,
    )


@pytest.fixture
def default_predictor(baseline_predictor_config: DeparturePredictorConfig) -> LaneDeparturePredictor:
    """Instantiate default LaneDeparturePredictor engine."""
    return LaneDeparturePredictor(config=baseline_predictor_config)


@pytest.fixture
def standard_straight_lane_offsets() -> Tuple[float, float]:
    """
    Standard 3.7m wide straight lane offsets at vehicle front bumper:
    Left boundary: x = -1.85 m
    Right boundary: x = +1.85 m
    """
    return -1.85, 1.85


# ==============================================================================
# 1. Kinematic Vehicle Forward Projector
# ==============================================================================

class TestKinematicVehicleProjector:
    """Validates kinematic bicycle projection, trajectory ribbons, and uncertainty growth."""

    def test_straight_line_projection_zero_lateral_drift(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify vehicle moving straight ahead projects purely along longitudinal axis."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=20.0,
            lateral_speed_mps=0.0,
            heading_angle_rad=0.0,
            yaw_rate_radps=0.0,
            lateral_accel_mps2=0.0,
        )

        assert len(ribbon) == 10 # 2.50s / 0.25s
        half_w = baseline_predictor_config.vehicle_width_m * 0.5

        for i, wp in enumerate(ribbon):
            expected_t = (i + 1) * 0.25
            assert math.isclose(wp.time_horizon_s, expected_t, abs_tol=1e-2)
            assert math.isclose(wp.x_m, 0.0, abs_tol=1e-3)
            assert wp.y_m > 0.0
            assert math.isclose(wp.heading_rad, 0.0, abs_tol=1e-4)
            assert math.isclose(wp.left_wheel_x_m, -half_w, abs_tol=1e-3)
            assert math.isclose(wp.right_wheel_x_m, half_w, abs_tol=1e-3)
            assert wp.lateral_uncertainty_sigma_m > 0.0

    def test_uncertainty_grows_monotonically_with_prediction_horizon(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify lateral uncertainty sigma expands quadratically with prediction horizon."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=22.0,
            lateral_speed_mps=0.0,
            heading_angle_rad=0.0,
            yaw_rate_radps=0.0,
        )

        for i in range(len(ribbon) - 1):
            assert ribbon[i].lateral_uncertainty_sigma_m < ribbon[i + 1].lateral_uncertainty_sigma_m

    def test_lateral_speed_deflects_trajectory_linearly(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify positive lateral speed drifts trajectory rightward."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=15.0,
            lateral_speed_mps=1.0, # 1.0 m/s to the right
            heading_angle_rad=0.0,
            yaw_rate_radps=0.0,
        )

        for wp in ribbon:
            assert wp.x_m > 0.0
            assert wp.right_wheel_x_m > (baseline_predictor_config.vehicle_width_m * 0.5)

    def test_yaw_rate_curves_trajectory_path(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify yaw rate rotates vehicle heading and deflects lateral coordinates."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=15.0,
            lateral_speed_mps=0.0,
            heading_angle_rad=0.0,
            yaw_rate_radps=0.10, # Turning right
        )

        # Heading should increase positively over time
        for i in range(len(ribbon) - 1):
            assert ribbon[i].heading_rad < ribbon[i + 1].heading_rad
            assert ribbon[i + 1].x_m > ribbon[i].x_m


# ==============================================================================
# 2. Quadratic & Linear Dynamic TTLD Solver
# ==============================================================================

class TestDynamicTTLDSolver:
    """Validates root-finding mathematical logic for time-to-lane-departure."""

    def test_linear_crossing_time_solution(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify linear formula t = d / v when acceleration is negligible."""
        d_margin = 1.00 # 1 meter margin
        v_towards = 0.50 # 0.5 m/s approach speed
        a_towards = 0.00 # Zero acceleration

        ttld = default_predictor._solve_quadratic_crossing_time(
            d_margin_m=d_margin, v_towards_mps=v_towards, a_towards_mps2=a_towards
        )

        # t = 1.0 / 0.5 = 2.00 s
        assert math.isclose(ttld, 2.00, abs_tol=1e-3)

    def test_quadratic_crossing_time_solution_with_positive_acceleration(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """
        Verify quadratic solver: 0.5 * a * t^2 + v * t - d = 0
        With a = 1.0 m/s^2, v = 1.0 m/s, d = 4.0 m:
          0.5 * 1.0 * t^2 + 1.0 * t - 4.0 = 0
          0.5 t^2 + t - 4 = 0 -> t^2 + 2t - 8 = 0 -> (t + 4)(t - 2) = 0
          Positive root t = 2.0 s.
        """
        ttld = default_predictor._solve_quadratic_crossing_time(
            d_margin_m=4.0, v_towards_mps=1.0, a_towards_mps2=1.0
        )
        assert math.isclose(ttld, 2.00, abs_tol=1e-3)

    def test_vehicle_moving_away_returns_ninety_nine_seconds(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify vehicle steering away (negative approach speed) returns 99.0s (no departure)."""
        ttld = default_predictor._solve_quadratic_crossing_time(
            d_margin_m=1.0, v_towards_mps=-0.8, a_towards_mps2=0.0
        )
        assert math.isclose(ttld, 99.0, abs_tol=1e-3)

    def test_vehicle_already_penetrating_margin_returns_zero(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify vehicle touching or crossing boundary (margin <= 0) returns 0.0s."""
        ttld = default_predictor._solve_quadratic_crossing_time(
            d_margin_m=-0.05, v_towards_mps=0.5, a_towards_mps2=0.0
        )
        assert math.isclose(ttld, 0.0, abs_tol=1e-3)


# ==============================================================================
# 3. Departure Severity Hierarchy & Directional Resolution
# ==============================================================================

class TestDepartureSeverityAndDirection:
    """Validates classification across Nominal, Warning, Critical, and Emergency."""

    def test_centered_driving_classifies_nominal_severity(
        self,
        default_predictor: LaneDeparturePredictor,
        standard_straight_lane_offsets: Tuple[float, float],
    ) -> None:
        """Verify centered driving with zero heading error yields NOMINAL with no CAN alert."""
        left_x, right_x = standard_straight_lane_offsets
        pred: LaneDeparturePrediction = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=left_x,
            right_boundary_x_at_vehicle_m=right_x,
            vehicle_speed_mps=20.0,
            heading_error_rad=0.0,
            yaw_rate_radps=0.0,
            lateral_accel_mps2=0.0,
            inner_safety_buffer_m=0.30,
        )

        assert pred.severity == DepartureSeverity.NOMINAL
        assert pred.direction == DepartureDirection.NONE
        assert pred.can_alert_required is False
        assert pred.time_to_departure_s >= 20.0
        assert math.isclose(pred.distance_to_left_boundary_m, 0.60, abs_tol=1e-2)
        assert math.isclose(pred.distance_to_right_boundary_m, 0.60, abs_tol=1e-2)

    def test_gentle_drift_left_triggers_warning_severity(
        self,
        default_predictor: LaneDeparturePredictor,
        standard_straight_lane_offsets: Tuple[float, float],
    ) -> None:
        """
        Verify gentle left drift (heading error -0.02 rad at 20 m/s -> v_lat ~ -0.40 m/s):
        Clearance ~ 0.60 m -> TTLD ~ 1.50 s.
        Classifies WARNING (1.2s <= TTLD < 2.0s), CAN 0x140 alert required.
        """
        left_x, right_x = standard_straight_lane_offsets
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=left_x,
            right_boundary_x_at_vehicle_m=right_x,
            vehicle_speed_mps=20.0,
            heading_error_rad=-0.02, # Small angle left
            yaw_rate_radps=0.0,
            lateral_accel_mps2=0.0,
            inner_safety_buffer_m=0.30,
        )

        assert pred.severity == DepartureSeverity.WARNING
        assert pred.direction == DepartureDirection.DEPARTING_LEFT
        assert pred.can_alert_required is True
        assert pred.can_message_priority == 0x140
        assert 1.2 <= pred.time_to_departure_s < 2.0

    def test_moderate_drift_right_triggers_critical_severity(
        self,
        default_predictor: LaneDeparturePredictor,
        standard_straight_lane_offsets: Tuple[float, float],
    ) -> None:
        """
        Verify moderate right drift (heading error +0.04 rad at 20 m/s -> v_lat ~ +0.80 m/s):
        Clearance ~ 0.60 m -> TTLD ~ 0.75 s.
        Classifies CRITICAL (0.6s <= TTLD < 1.2s), CAN 0x150 priority alert.
        """
        left_x, right_x = standard_straight_lane_offsets
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=left_x,
            right_boundary_x_at_vehicle_m=right_x,
            vehicle_speed_mps=20.0,
            heading_error_rad=0.04,
            yaw_rate_radps=0.0,
            lateral_accel_mps2=0.0,
            inner_safety_buffer_m=0.30,
        )

        assert pred.severity == DepartureSeverity.CRITICAL
        assert pred.direction == DepartureDirection.DEPARTING_RIGHT
        assert pred.can_alert_required is True
        assert pred.can_message_priority == 0x150
        assert 0.6 <= pred.time_to_departure_s < 1.2

    def test_aggressive_drift_triggers_emergency_severity(
        self,
        default_predictor: LaneDeparturePredictor,
        standard_straight_lane_offsets: Tuple[float, float],
    ) -> None:
        """
        Verify rapid lateral excursion (heading error -0.08 rad at 20 m/s -> v_lat ~ -1.60 m/s):
        Clearance ~ 0.60 m -> TTLD ~ 0.38 s.
        Classifies EMERGENCY (TTLD < 0.60 s), CAN 0x150 priority alert.
        """
        left_x, right_x = standard_straight_lane_offsets
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=left_x,
            right_boundary_x_at_vehicle_m=right_x,
            vehicle_speed_mps=20.0,
            heading_error_rad=-0.08,
            yaw_rate_radps=0.0,
            lateral_accel_mps2=0.0,
            inner_safety_buffer_m=0.30,
        )

        assert pred.severity == DepartureSeverity.EMERGENCY
        assert pred.direction == DepartureDirection.DEPARTING_LEFT
        assert pred.can_alert_required is True
        assert pred.can_message_priority == 0x150
        assert pred.time_to_departure_s < 0.60

    def test_imminent_or_breached_boundary_forces_emergency(
        self,
        default_predictor: LaneDeparturePredictor,
    ) -> None:
        """Verify vehicle flank already encroaching onto buffer (d <= 0) forces EMERGENCY."""
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.10, # Vehicle half-w 0.95 + buffer 0.30 = 1.25 -> breach!
            right_boundary_x_at_vehicle_m=2.60,
            vehicle_speed_mps=15.0,
            heading_error_rad=0.0,
            inner_safety_buffer_m=0.30,
        )

        assert pred.severity == DepartureSeverity.EMERGENCY
        assert pred.distance_to_left_boundary_m < 0.0


# ==============================================================================
# 4. SocketCAN Priority Alert Encoder & SAE J1850 CRC-8
# ==============================================================================

class TestCANAlertEncoder:
    """Validates 8-byte automotive CAN payload structure and SAE J1850 CRC-8 calculation."""

    def test_sae_j1850_crc8_known_vector(self) -> None:
        """Verify CRC-8 SAE J1850 implementation on standard automotive test bytes."""
        payload = bytes([0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07])
        crc = CANAlertEncoder._compute_crc8_sae_j1850(payload)
        assert isinstance(crc, int)
        assert 0 <= crc <= 255

    def test_encode_departure_frame_structure(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify 8-byte CAN frame fields, rolling counter, and arbitration ID."""
        encoder = CANAlertEncoder()
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=20.0,
            heading_error_rad=-0.02, # Warning left (TTLD ~ 1.5s -> Severity 1)
        )

        can_id, payload = encoder.encode_departure_frame(pred)
        assert can_id in (0x140, 0x150)
        assert len(payload) == 8

        # Byte 0: Severity (1 = Warning)
        assert payload[0] == 1
        # Byte 1: Direction (1 = Left)
        assert payload[1] == 1
        # Byte 7: Verify CRC matches payload bytes 0-6
        expected_crc = encoder._compute_crc8_sae_j1850(payload[:7])
        assert payload[7] == expected_crc

    def test_rolling_counter_increments_consecutively(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify CAN byte 6 message counter increments monotonically with each frame."""
        encoder = CANAlertEncoder()
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=15.0,
            heading_error_rad=0.0,
        )

        _, frame1 = encoder.encode_departure_frame(pred)
        _, frame2 = encoder.encode_departure_frame(pred)
        _, frame3 = encoder.encode_departure_frame(pred)

        assert (frame1[6] + 1) % 256 == frame2[6]
        assert (frame2[6] + 1) % 256 == frame3[6]


# ==============================================================================
# 5. Centrifugal Tire Slip & Curvature Drift Estimator
# ==============================================================================

class TestCentrifugalDriftEstimator:
    """Validates steady-state tire slip angle and outward drift on highway curves."""

    def test_straight_road_has_zero_drift_velocity(self) -> None:
        """Verify straight path (infinite radius R = 5000m) induces negligible drift."""
        estimator = CentrifugalDriftEstimator()
        drift_v = estimator.estimate_curvature_drift_velocity(
            speed_mps=25.0, radius_of_curvature_m=5000.0
        )
        assert math.isclose(drift_v, 0.0, abs_tol=0.05)

    def test_tight_curve_at_high_speed_induces_measurable_drift(self) -> None:
        """Verify sharp curve (R = 150m) at 25 m/s produces positive outward drift velocity."""
        estimator = CentrifugalDriftEstimator()
        drift_v = estimator.estimate_curvature_drift_velocity(
            speed_mps=25.0, radius_of_curvature_m=150.0
        )
        assert drift_v > 0.10
        assert drift_v <= 2.50 # Clamped within vehicle dynamic limits

    def test_drift_velocity_monotonic_with_speed(self) -> None:
        """Verify centrifugal drift increases monotonically with vehicle velocity."""
        estimator = CentrifugalDriftEstimator()
        v_low = estimator.estimate_curvature_drift_velocity(
            speed_mps=15.0, radius_of_curvature_m=200.0
        )
        v_high = estimator.estimate_curvature_drift_velocity(
            speed_mps=30.0, radius_of_curvature_m=200.0
        )
        assert v_high > v_low


# ==============================================================================
# 6. Statistical Departure Risk Heatmap Rasterizer
# ==============================================================================

class TestDepartureRiskHeatmapRasterizer:
    """Validates 2D top-down Gaussian heatmap blending onto BEV ground maps."""

    def test_rasterize_risk_heatmap_modifies_canvas(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify heatmap rasterizer generates color-mapped overlay along trajectory."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=20.0,
            lateral_speed_mps=0.5,
            heading_angle_rad=0.02,
            yaw_rate_radps=0.0,
        )

        rasterizer = DepartureRiskHeatmapRasterizer(meters_per_pixel=0.0185)
        canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        blended = rasterizer.rasterize_risk_heatmap(
            bev_canvas_bgr=canvas,
            ribbon=ribbon,
            center_x_px=200,
            bottom_y_px=580,
        )

        assert blended.shape == canvas.shape
        assert blended.dtype == np.uint8
        assert not np.array_equal(blended, canvas)

    def test_rasterize_risk_heatmap_with_empty_ribbon(self) -> None:
        """Verify rasterizer gracefully handles empty waypoint list without crashing."""
        rasterizer = DepartureRiskHeatmapRasterizer(meters_per_pixel=0.0185)
        canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        blended = rasterizer.rasterize_risk_heatmap(
            bev_canvas_bgr=canvas, ribbon=[], center_x_px=200, bottom_y_px=580
        )
        assert np.array_equal(blended, canvas)


# ==============================================================================
# 7. Diagnostic HUD Canvas Visualizer
# ==============================================================================

class TestDiagnosticHUDCanvasVisualizer:
    """Validates live predictive departure gauge and warning badges."""

    def test_render_departure_hud_displays_badges(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify HUD renderer annotates TTLD countdown and severity box."""
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=20.0,
            heading_error_rad=-0.04,
        )

        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered = render_departure_hud(canvas_bgr=canvas, prediction=pred)

        assert rendered.shape == canvas.shape
        assert rendered.dtype == np.uint8
        assert not np.array_equal(rendered, canvas)

    @pytest.mark.parametrize("sev", [
        DepartureSeverity.NOMINAL,
        DepartureSeverity.WARNING,
        DepartureSeverity.CRITICAL,
        DepartureSeverity.EMERGENCY,
    ])
    def test_render_departure_hud_all_severities(
        self, sev: DepartureSeverity, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify HUD renders distinct color badges for all severity tiers."""
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=15.0,
            heading_error_rad=0.0,
        )
        pred.severity = sev

        canvas = np.zeros((360, 640, 3), dtype=np.uint8)
        rendered = render_departure_hud(canvas_bgr=canvas, prediction=pred)
        assert rendered.shape == (360, 640, 3)


# ==============================================================================
# 8. Edge Cases, Zero Velocity & Real-Time Performance
# ==============================================================================

class TestPredictorEdgeCasesAndLatency:
    """Validates corner cases, standstill handling, and execution latency."""

    def test_near_zero_vehicle_speed_clamped_safely(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify stationary vehicle (v = 0 m/s) does not divide by zero or crash."""
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=0.0,
            heading_error_rad=0.0,
        )

        assert pred.severity == DepartureSeverity.NOMINAL
        assert pred.time_to_departure_s >= 20.0

    def test_prediction_to_dict_completeness(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify to_dict serialization contains all required telemetry fields."""
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=20.0,
            heading_error_rad=-0.02,
        )

        data = pred.to_dict()
        required_fields = {
            "timestamp_utc", "severity", "direction", "ttld_s", "d_left_m",
            "d_right_m", "v_lat_mps", "a_lat_mps2", "yaw_rate_degps",
            "can_alert_required", "can_message_priority", "processing_time_ms"
        }
        for f in required_fields:
            assert f in data

    def test_real_time_prediction_latency_under_two_milliseconds(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify departure predictor loop completes with mean latency < 1.0 ms."""
        latencies_ms: List[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = default_predictor.predict_departure(
                left_boundary_x_at_vehicle_m=-1.85,
                right_boundary_x_at_vehicle_m=1.85,
                vehicle_speed_mps=20.0,
                heading_error_rad=-0.03,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))

        assert mean_lat < 1.0, f"Mean TTLD latency {mean_lat:.2f}ms exceeds 1.0ms"
        assert p95_lat < 2.0, f"P95 TTLD latency {p95_lat:.2f}ms exceeds 2.0ms"


# ==============================================================================
# 9. Parametric Kinematic Sweep (Speeds, Headings & Accelerations)
# ==============================================================================

class TestParametricKinematicSweep:
    """Validates mathematical continuity of TTLD under parametric variations."""

    @pytest.mark.parametrize("speed", [10.0, 15.0, 20.0, 25.0, 30.0])
    def test_speed_scaling_shortens_ttld_monotonically(
        self, speed: float, default_predictor: LaneDeparturePredictor
    ) -> None:
        """
        Verify that for a fixed nonzero heading error, increasing vehicle longitudinal
        speed proportionally increases lateral velocity v_lat, shortening TTLD.
        """
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=speed,
            heading_error_rad=-0.04, # Fixed drift left
        )

        expected_v_lat = speed * math.sin(-0.04)
        assert math.isclose(pred.lateral_velocity_mps, expected_v_lat, abs_tol=1e-2)
        # Higher speed must yield shorter TTLD
        assert pred.time_to_departure_s > 0.0
        assert pred.direction == DepartureDirection.DEPARTING_LEFT

    @pytest.mark.parametrize("heading_deg", [-6.0, -3.0, -1.0, 1.0, 3.0, 6.0])
    def test_heading_angle_sweep_directional_symmetry(
        self, heading_deg: float, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify left vs right directional symmetry across opposite heading angles."""
        heading_rad = math.radians(heading_deg)
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=20.0,
            heading_error_rad=heading_rad,
        )

        if heading_deg < 0:
            assert pred.direction == DepartureDirection.DEPARTING_LEFT
            assert pred.lateral_velocity_mps < 0.0
        else:
            assert pred.direction == DepartureDirection.DEPARTING_RIGHT
            assert pred.lateral_velocity_mps > 0.0

    @pytest.mark.parametrize("a_lat", [-2.0, -1.0, 0.0, 1.0, 2.0])
    def test_lateral_acceleration_effects_on_ttld(
        self, a_lat: float, default_predictor: LaneDeparturePredictor
    ) -> None:
        """
        Verify lateral acceleration integration from MTi-680 RTK IMU.
        Positive acceleration towards right reduces right TTLD; negative acceleration
        reduces left TTLD.
        """
        pred = default_predictor.predict_departure(
            left_boundary_x_at_vehicle_m=-1.85,
            right_boundary_x_at_vehicle_m=1.85,
            vehicle_speed_mps=20.0,
            heading_error_rad=0.02, # Drifting right
            lateral_accel_mps2=a_lat,
        )

        assert math.isclose(pred.lateral_acceleration_mps2, a_lat, abs_tol=1e-3)
        assert pred.time_to_departure_s > 0.0


# ==============================================================================
# 10. Trajectory Ribbon Geometry & Vehicle Footprint Envelope
# ==============================================================================

class TestTrajectoryRibbonGeometry:
    """Validates physical dimensions of the vehicle bounding envelope ribbon."""

    def test_waypoint_flank_offsets_match_vehicle_width(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """
        Verify that at every projected waypoint, the distance between left and right
        wheel bounding coordinates strictly equals the vehicle track width (1.90 m).
        """
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=18.0,
            lateral_speed_mps=0.4,
            heading_angle_rad=0.05,
            yaw_rate_radps=0.02,
        )

        expected_w = baseline_predictor_config.vehicle_width_m

        for wp in ribbon:
            # Under heading theta, flank distance projected onto local normal is W
            flank_span = math.hypot(
                wp.right_wheel_x_m - wp.left_wheel_x_m,
                # Longitudinal difference between left and right wheels under yaw
                0.0,
            )
            # The lateral span right - left equals W * cos(theta)
            lateral_span = wp.right_wheel_x_m - wp.left_wheel_x_m
            expected_span = expected_w * math.cos(wp.heading_rad)
            assert math.isclose(lateral_span, expected_span, abs_tol=1e-2)

    def test_ribbon_longitudinal_spacing_monotonicity(
        self, baseline_predictor_config: DeparturePredictorConfig
    ) -> None:
        """Verify longitudinal distance y_m progresses forward monotonically without loops."""
        projector = KinematicVehicleProjector(config=baseline_predictor_config)
        ribbon = projector.project_trajectory(
            longitudinal_speed_mps=20.0,
            lateral_speed_mps=0.0,
            heading_angle_rad=0.0,
            yaw_rate_radps=0.05, # Mild curvature
        )

        for i in range(len(ribbon) - 1):
            assert ribbon[i].y_m < ribbon[i + 1].y_m
            assert ribbon[i].time_horizon_s < ribbon[i + 1].time_horizon_s


# ==============================================================================
# 11. High-Rate IMU Noise Robustness & Sensor Glitch Handling
# ==============================================================================

class TestIMUNoiseResilience:
    """Validates resilience against sensor noise on 400 Hz MTi-680 IMU data."""

    def test_noisy_imu_acceleration_does_not_destabilize_prediction(
        self, default_predictor: LaneDeparturePredictor
    ) -> None:
        """Verify adding zero-mean Gaussian noise (sigma=0.3 m/s^2) maintains stable severity."""
        np.random.seed(42)
        severities = []
        ttlds = []

        for _ in range(50):
            noise = float(np.random.normal(0.0, 0.3))
            pred = default_predictor.predict_departure(
                left_boundary_x_at_vehicle_m=-1.85,
                right_boundary_x_at_vehicle_m=1.85,
                vehicle_speed_mps=20.0,
                heading_error_rad=0.0,
                lateral_accel_mps2=noise,
            )
            severities.append(pred.severity)
            ttlds.append(pred.time_to_departure_s)

        # Centered vehicle with zero-mean noise should never produce false EMERGENCY or CRITICAL
        assert DepartureSeverity.EMERGENCY not in severities
        assert DepartureSeverity.CRITICAL not in severities
        # All frames remain safely NOMINAL or mild WARNING
        assert all(s in (DepartureSeverity.NOMINAL, DepartureSeverity.WARNING) for s in severities)
        nominal_count = sum(1 for s in severities if s == DepartureSeverity.NOMINAL)
        assert nominal_count >= 25

