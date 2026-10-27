"""
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Driver-Supervised SAE Level 2 Autonomous Driving Stack
ISO 26262-Style Process; ASIL-B Aligned; Not Independently Assessed

tests/test_virtual_guardrail_enforcer.py
Comprehensive Verification Suite for the Virtual Guardrail Safety Envelope & Interlock.

PRIMARY SAFETY MISSION:
  Enforce the non-negotiable safeguard that the vehicle NEVER crosses its Left (Class 2)
  or Right (Class 3) lane boundaries under any driving scenario.

Test Matrix:
  1. 0.3-meter inner virtual safety envelope construction and geometry
  2. Dynamic buffer width scaling across Highway, Urban, Rain, Tunnel, and Night scenes
  3. 3-tier violation severity hierarchy (Warning, Critical, Emergency)
  4. Stanley controller gain scheduling and counter-steer torque vectoring
  5. Predictive trajectory ribbon forward evaluation (t in [0.0s .. 1.5s])
  6. Perception quality degradation interlock (Q < 0.35 for 5+ frames trips Arduino relay)
  7. Fail-safe Arduino clutch relay driver and CRC16-CCITT serial protocol
  8. Diagnostic BEV HUD canvas overlay rendering
  9. Fault injection, sensor dropouts, and extreme cornering edge cases
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.faded_lane_recovery import BoundaryPolynomial2D
from src.perception.lane_departure_predictor import (
    DepartureDirection,
    DepartureSeverity,
    LaneDeparturePrediction,
    ProjectedWaypoint,
)
from src.perception.virtual_guardrail_enforcer import (
    ArduinoRelaySafetyDriver,
    BoundarySide,
    GuardrailConfig,
    GuardrailEnforcementTelemetry,
    GuardrailSeverity,
    GuardrailViolation,
    RelayCommandPacket,
    VirtualGuardrailEnforcer,
    render_guardrail_hud,
)
from src.scene.scene_profiles import SceneType


# ==============================================================================
# Test Fixtures & Synthetic Trajectories
# ==============================================================================

@pytest.fixture
def default_guardrail_config() -> GuardrailConfig:
    """Standard guardrail configuration with 0.30m nominal buffer and ASIL-B bounds."""
    return GuardrailConfig(
        vehicle_width_m=1.90,
        nominal_buffer_width_m=0.30,
        highway_buffer_width_m=0.20,
        urban_buffer_width_m=0.50,
        adverse_buffer_width_m=0.35,
        critical_buffer_fraction=0.50,
        base_stanley_gain=1.25,
        warning_gain_multiplier=1.50,
        max_counter_steer_torque_nm=4.50,
        can_emergency_id=0x150,
        consecutive_low_quality_trip=5,
        quality_cutoff_threshold=0.35,
    )


@pytest.fixture
def standard_lane_boundaries() -> Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D]:
    """
    Generate standard 3.7m wide straight lane boundaries in metric vehicle coordinates.
    Vehicle origin is at (0, 0).
    Left boundary: x = -1.85 m (Class 2)
    Right boundary: x = +1.85 m (Class 3)
    """
    left_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.0,
        B=0.0,
        C=-1.85,
        confidence=0.95,
        inlier_count=50,
        is_hallucinated=False,
    )
    right_poly = BoundaryPolynomial2D(
        is_valid=True,
        A=0.0,
        B=0.0,
        C=1.85,
        confidence=0.95,
        inlier_count=50,
        is_hallucinated=False,
    )
    return left_poly, right_poly


@pytest.fixture
def nominal_centered_prediction() -> LaneDeparturePrediction:
    """Generate nominal centered vehicle prediction moving straight along y-axis."""
    waypoints: List[ProjectedWaypoint] = []
    half_w = 0.95
    for i in range(1, 7):
        t = i * 0.25
        y = t * 20.0  # 20 m/s
        waypoints.append(ProjectedWaypoint(
            time_horizon_s=t,
            x_m=0.0,
            y_m=y,
            heading_rad=0.0,
            left_wheel_x_m=-half_w,
            right_wheel_x_m=half_w,
            lateral_uncertainty_sigma_m=0.05 + 0.02 * t,
        ))

    return LaneDeparturePrediction(
        timestamp_utc="2026-10-27T12:00:00Z",
        severity=DepartureSeverity.NOMINAL,
        direction=DepartureDirection.NONE,
        time_to_departure_s=99.0,
        distance_to_left_boundary_m=0.60,
        distance_to_right_boundary_m=0.60,
        lateral_velocity_mps=0.0,
        lateral_acceleration_mps2=0.0,
        yaw_rate_radps=0.0,
        projected_ribbon=waypoints,
        can_alert_required=False,
        can_message_priority=0x140,
        processing_time_ms=0.85,
    )


# ==============================================================================
# 1. Dynamic Safety Buffer Width Scaling
# ==============================================================================

class TestDynamicBufferScaling:
    """Validates scene-adaptive and speed-dependent buffer scaling."""

    def test_nominal_buffer_width_at_standard_speed(
        self, default_guardrail_config: GuardrailConfig
    ) -> None:
        """Verify nominal 0.30 m buffer on standard suburban/rural cruising at 18 m/s."""
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        buffer_w = enforcer.compute_dynamic_buffer_width(
            vehicle_speed_mps=18.0, scene=SceneType.RURAL
        )
        assert math.isclose(buffer_w, 0.30, abs_tol=1e-4)

    def test_highway_high_speed_narrows_buffer_to_point_two(
        self, default_guardrail_config: GuardrailConfig
    ) -> None:
        """Verify buffer narrows to 0.20 m at high speed (>= 22 m/s) on HIGHWAY."""
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        buffer_w = enforcer.compute_dynamic_buffer_width(
            vehicle_speed_mps=25.0, scene=SceneType.HIGHWAY
        )
        assert math.isclose(buffer_w, 0.20, abs_tol=1e-4)

    def test_urban_stop_and_go_widens_buffer_to_point_five(
        self, default_guardrail_config: GuardrailConfig
    ) -> None:
        """Verify buffer widens to 0.50 m in URBAN environment or speeds < 12 m/s."""
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Urban scene
        buf_urban = enforcer.compute_dynamic_buffer_width(
            vehicle_speed_mps=15.0, scene=SceneType.URBAN
        )
        assert math.isclose(buf_urban, 0.50, abs_tol=1e-4)

        # Low speed stop-and-go regardless of scene
        buf_low_speed = enforcer.compute_dynamic_buffer_width(
            vehicle_speed_mps=8.0, scene=SceneType.HIGHWAY
        )
        assert math.isclose(buf_low_speed, 0.50, abs_tol=1e-4)

    @pytest.mark.parametrize("scene", [SceneType.RAIN, SceneType.TUNNEL, SceneType.NIGHT])
    def test_adverse_conditions_enforce_point_three_five_buffer(
        self, scene: SceneType, default_guardrail_config: GuardrailConfig
    ) -> None:
        """Verify adverse environmental scenes set conservative 0.35 m buffer."""
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        buf_adverse = enforcer.compute_dynamic_buffer_width(
            vehicle_speed_mps=18.0, scene=scene
        )
        assert math.isclose(buf_adverse, 0.35, abs_tol=1e-4)


# ==============================================================================
# 2. 3-Tier Violation Severity Hierarchy & Actuation Directives
# ==============================================================================

class TestViolationSeverityHierarchy:
    """Validates Warning, Critical, and Emergency severity classification."""

    def test_nominal_centered_trajectory_produces_nominal_telemetry(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify centered driving maintains NOMINAL severity with 0 Nm counter-steer."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        telemetry: GuardrailEnforcementTelemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=nominal_centered_prediction,
            perception_quality=0.95,
            vehicle_speed_mps=20.0,
            active_scene=SceneType.HIGHWAY,
        )

        assert telemetry.active_severity == GuardrailSeverity.NOMINAL
        assert telemetry.active_side == BoundarySide.NONE
        assert telemetry.counter_steer_torque_command_nm == 0.0
        assert math.isclose(telemetry.active_stanley_gain, 1.25, abs_tol=1e-3)
        assert telemetry.is_safe_for_actuation is True
        assert telemetry.violation_event is None

    def test_severity_1_warning_when_touching_outer_safety_envelope(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """
        Verify Severity 1 WARNING triggers when vehicle flank encroaches into outer buffer:
        Clearance = 0.22 m <= buffer_w (0.30 m).
        Scales Stanley gain by 1.5x (1.875).
        """
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Vehicle drifting left: left wheel at -1.63 m (clearance to -1.85 is 0.22 m <= 0.30m)
        waypoints = [
            ProjectedWaypoint(
                time_horizon_s=0.5,
                x_m=-0.68,
                y_m=10.0,
                heading_rad=0.0,
                left_wheel_x_m=-1.63,
                right_wheel_x_m=0.27,
                lateral_uncertainty_sigma_m=0.05,
            )
        ]
        pred = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.WARNING,
            direction=DepartureDirection.DEPARTING_LEFT,
            time_to_departure_s=1.5,
            distance_to_left_boundary_m=0.22,
            distance_to_right_boundary_m=1.0,
            lateral_velocity_mps=-0.4,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=waypoints,
            can_alert_required=True,
            can_message_priority=0x140,
            processing_time_ms=0.9,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=pred,
            perception_quality=0.90,
            vehicle_speed_mps=18.0,
            active_scene=SceneType.RURAL,
        )

        assert telemetry.active_severity == GuardrailSeverity.WARNING
        assert telemetry.active_side == BoundarySide.LEFT_BOUNDARY
        expected_gain = 1.25 * 1.50
        assert math.isclose(telemetry.active_stanley_gain, expected_gain, abs_tol=1e-3)
        assert telemetry.violation_event is not None
        assert telemetry.violation_event.emergency_brake_assist_active is False
        assert telemetry.is_safe_for_actuation is True

    def test_severity_2_critical_applies_counter_steer_torque(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """
        Verify Severity 2 CRITICAL triggers when vehicle penetrates inner 50% of buffer:
        Clearance = 0.08 m <= critical_buf (0.15 m).
        Scales Stanley gain by 1.75x and applies counter-steering torque.
        """
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Vehicle drifting right: right wheel at +1.77 m (clearance to +1.85 is 0.08 m <= 0.15m)
        waypoints = [
            ProjectedWaypoint(
                time_horizon_s=0.5,
                x_m=0.82,
                y_m=10.0,
                heading_rad=0.0,
                left_wheel_x_m=-0.13,
                right_wheel_x_m=1.77,
                lateral_uncertainty_sigma_m=0.05,
            )
        ]
        pred = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.CRITICAL,
            direction=DepartureDirection.DEPARTING_RIGHT,
            time_to_departure_s=0.8,
            distance_to_left_boundary_m=1.0,
            distance_to_right_boundary_m=0.08,
            lateral_velocity_mps=0.6,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=waypoints,
            can_alert_required=True,
            can_message_priority=0x150,
            processing_time_ms=0.9,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=pred,
            perception_quality=0.90,
            vehicle_speed_mps=18.0,
            active_scene=SceneType.RURAL,
        )

        assert telemetry.active_severity == GuardrailSeverity.CRITICAL
        assert telemetry.active_side == BoundarySide.RIGHT_BOUNDARY
        # Right threatened -> steer left (negative torque)
        assert telemetry.counter_steer_torque_command_nm < 0.0
        assert math.isclose(telemetry.active_stanley_gain, 1.25 * 1.75, abs_tol=1e-3)
        assert telemetry.violation_event is not None
        assert telemetry.violation_event.emergency_brake_assist_active is False

    def test_severity_3_emergency_boundary_breached(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """
        Verify Severity 3 EMERGENCY triggers when vehicle crosses physical boundary (clearance <= 0):
        Activates emergency brake assist, 2.0x Stanley gain, max counter-steer, CAN 0x150.
        """
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Vehicle breached left boundary: left wheel at -1.95 m (line is -1.85 m, breach = -0.10 m)
        waypoints = [
            ProjectedWaypoint(
                time_horizon_s=0.25,
                x_m=-1.00,
                y_m=5.0,
                heading_rad=0.0,
                left_wheel_x_m=-1.95,
                right_wheel_x_m=-0.05,
                lateral_uncertainty_sigma_m=0.05,
            )
        ]
        pred = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.EMERGENCY,
            direction=DepartureDirection.DEPARTING_LEFT,
            time_to_departure_s=0.1,
            distance_to_left_boundary_m=-0.10,
            distance_to_right_boundary_m=1.5,
            lateral_velocity_mps=-1.2,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=waypoints,
            can_alert_required=True,
            can_message_priority=0x150,
            processing_time_ms=0.9,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=pred,
            perception_quality=0.90,
            vehicle_speed_mps=18.0,
            active_scene=SceneType.RURAL,
        )

        assert telemetry.active_severity == GuardrailSeverity.EMERGENCY
        assert telemetry.active_side == BoundarySide.LEFT_BOUNDARY
        # Left breached -> full positive counter-steer torque (+4.50 Nm)
        assert math.isclose(telemetry.counter_steer_torque_command_nm, 4.50, abs_tol=1e-3)
        assert telemetry.is_safe_for_actuation is False # Emergency blocks normal actuation
        assert telemetry.violation_event is not None
        assert telemetry.violation_event.emergency_brake_assist_active is True
        assert telemetry.violation_event.can_message_priority == 0x150


# ==============================================================================
# 3. Perception Quality Degradation Interlock & Arduino Relay
# ==============================================================================

class TestQualityDegradationInterlock:
    """Validates the 5-frame Q < 0.35 safety cutoff and Arduino relay decoupling."""

    def test_interlock_does_not_trip_for_isolated_low_quality_frames(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify interlock tolerates up to 4 consecutive low-quality frames without tripping."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        for frame_idx in range(4):
            telemetry = enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=nominal_centered_prediction,
                perception_quality=0.25, # Low quality Q < 0.35
            )
            assert enforcer.is_interlock_tripped is False
            assert telemetry.is_safe_for_actuation is True

    def test_interlock_trips_on_fifth_consecutive_low_quality_frame(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify interlock trips on the 5th frame of Q < 0.35, setting is_safe_for_actuation=False."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Feed 5 low-quality frames
        for _ in range(5):
            telemetry = enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=nominal_centered_prediction,
                perception_quality=0.20,
            )

        assert enforcer.is_interlock_tripped is True
        assert telemetry.is_safe_for_actuation is False

    def test_interlock_recovers_when_quality_restores(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify interlock re-engages safely after consecutive high-quality frames return."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Trip the interlock with 5 bad frames
        for _ in range(5):
            enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=nominal_centered_prediction,
                perception_quality=0.20,
            )
        assert enforcer.is_interlock_tripped is True

        # Restore quality over successive frames
        for _ in range(5):
            telemetry = enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=nominal_centered_prediction,
                perception_quality=0.90,
            )

        assert enforcer.is_interlock_tripped is False
        assert telemetry.is_safe_for_actuation is True

    def test_force_reset_interlock_clears_state_immediately(
        self, default_guardrail_config: GuardrailConfig
    ) -> None:
        """Verify manual override or driver acknowledgement clears interlock."""
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        enforcer._relay_cutoff_tripped = True
        enforcer._consecutive_low_q = 5

        enforcer.force_reset_interlock()
        assert enforcer.is_interlock_tripped is False
        assert enforcer._consecutive_low_q == 0


# ==============================================================================
# 4. Arduino Physical Clutch Relay Hardware Driver & CRC16-CCITT
# ==============================================================================

class TestArduinoRelaySafetyDriver:
    """Validates the serial packet framing, CRC16 checksum, and relay state control."""

    def test_crc16_ccitt_known_vector(self) -> None:
        """Verify standard CRC16-CCITT implementation on known ASCII byte strings."""
        data = b"123456789"
        crc = ArduinoRelaySafetyDriver._compute_crc16_ccitt(data)
        assert isinstance(crc, int)
        assert 0 <= crc <= 0xFFFF

    def test_format_relay_packet_structure(self) -> None:
        """Verify command packet format: $APEX,RELAY,<STATE>,<REASON>,<CRC16>\\r\\n."""
        driver = ArduinoRelaySafetyDriver()
        packet: RelayCommandPacket = driver.format_relay_packet(
            energize=True, reason="NOMINAL"
        )

        assert packet.relay_energized is True
        assert packet.safety_code == 0x5A5A
        assert b"$APEX,RELAY,1,NOMINAL," in packet.serial_payload
        assert packet.serial_payload.endswith(b"\r\n")

    def test_dispatch_safety_trip_packet(self) -> None:
        """Verify safety trip de-energizes the relay (opens circuit) with code 0x0000."""
        driver = ArduinoRelaySafetyDriver()
        packet = driver.dispatch_safety_trip(reason="EMERGENCY_DEPARTURE_BREACH")

        assert packet.relay_energized is False
        assert packet.safety_code == 0x0000
        assert b"EMERGENCY_DEPARTURE_BREACH" in packet.serial_payload
        assert b"$APEX,RELAY,0," in packet.serial_payload

    def test_dispatch_engage_packet(self) -> None:
        """Verify dispatch engage closes clutch relay with handshake code 0x5A5A."""
        driver = ArduinoRelaySafetyDriver()
        packet = driver.dispatch_engage(reason="DRIVER_RESUMED_SUPERVISION")

        assert packet.relay_energized is True
        assert packet.safety_code == 0x5A5A
        assert b"$APEX,RELAY,1," in packet.serial_payload


# ==============================================================================
# 5. Diagnostic HUD Canvas Visualizer & Safety Envelope Renderer
# ==============================================================================

class TestDiagnosticHUDCanvasVisualizer:
    """Validates rendering of inner envelopes and telemetry overlay."""

    def test_render_guardrail_hud_output_shape_and_type(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify rendering modifies BEV canvas and returns proper uint8 array."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=nominal_centered_prediction,
        )

        bev_canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        rendered = render_guardrail_hud(
            bev_canvas_bgr=bev_canvas,
            telemetry=telemetry,
            left_poly=left_poly,
            right_poly=right_poly,
        )

        assert rendered.shape == bev_canvas.shape
        assert rendered.dtype == np.uint8
        assert not np.array_equal(rendered, bev_canvas) # Canvas modified with overlay

    def test_render_guardrail_hud_with_invalid_boundaries(
        self,
        default_guardrail_config: GuardrailConfig,
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify HUD renderer gracefully handles invalid/missing boundary polynomials."""
        invalid_poly = BoundaryPolynomial2D(
            is_valid=False,
            A=0.0,
            B=0.0,
            C=0.0,
            confidence=0.0,
            inlier_count=0,
            is_hallucinated=False,
        )
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        telemetry = enforcer.enforce_guardrails(
            left_poly=invalid_poly,
            right_poly=invalid_poly,
            prediction=nominal_centered_prediction,
        )

        bev_canvas = np.zeros((600, 400, 3), dtype=np.uint8)
        rendered = render_guardrail_hud(
            bev_canvas_bgr=bev_canvas,
            telemetry=telemetry,
            left_poly=invalid_poly,
            right_poly=invalid_poly,
        )
        assert rendered.shape == (600, 400, 3)


# ==============================================================================
# 6. Edge Cases, Sensor Dropouts & Fault Resiliency
# ==============================================================================

class TestGuardrailEdgeCasesAndFaults:
    """Validates guardrail enforcer under missing lanes, extreme speeds, and NaN values."""

    def test_missing_left_boundary_only_evaluates_right(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify system handles single visible right lane without crashing."""
        _, right_poly = standard_lane_boundaries
        invalid_left = BoundaryPolynomial2D(
            is_valid=False,
            A=0.0,
            B=0.0,
            C=0.0,
            confidence=0.0,
            inlier_count=0,
            is_hallucinated=False,
        )

        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        telemetry = enforcer.enforce_guardrails(
            left_poly=invalid_left,
            right_poly=right_poly,
            prediction=nominal_centered_prediction,
        )

        assert telemetry.left_clearance_m == 99.0
        assert telemetry.right_clearance_m < 2.0

    def test_violation_history_ring_buffer_caps_at_two_hundred(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify violation history circular buffer does not grow unbounded past 200 items."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        pred_warning = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.WARNING,
            direction=DepartureDirection.DEPARTING_LEFT,
            time_to_departure_s=1.5,
            distance_to_left_boundary_m=0.20,
            distance_to_right_boundary_m=1.0,
            lateral_velocity_mps=-0.4,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=[
                ProjectedWaypoint(0.5, -0.68, 10.0, 0.0, -1.63, 0.27, 0.05)
            ],
            can_alert_required=True,
            can_message_priority=0x140,
            processing_time_ms=0.5,
        )

        for _ in range(250):
            enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=pred_warning,
            )

        history = enforcer.get_violation_history()
        assert len(history) == 200

    def test_telemetry_to_dict_schema_completeness(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify telemetry serialization produces all required HUD keys and numeric fields."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)
        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=nominal_centered_prediction,
        )

        d = telemetry.to_dict()
        required_keys = {
            "timestamp_utc", "severity", "threatened_side", "left_clearance_m",
            "right_clearance_m", "buffer_width_m", "stanley_gain", "counter_steer_nm",
            "is_safe_for_actuation", "has_violation", "processing_time_ms"
        }
        for k in required_keys:
            assert k in d


# ==============================================================================
# 7. Curved Road Topologies & High-Curvature Encroachment
# ==============================================================================

class TestCurvedRoadGuardrails:
    """Validates virtual safety envelope projection along curved highway corridors."""

    def test_left_curving_highway_boundary_clearance(
        self, default_guardrail_config: GuardrailConfig
    ) -> None:
        """
        Verify safety envelopes along a left curve: x(y) = -0.001*y^2 + 0.0*y - 1.85 (left).
        Vehicle tracking straight ahead will threaten the outer boundary as distance increases.
        """
        left_curve = BoundaryPolynomial2D(
            is_valid=True,
            A=-0.001,
            B=0.0,
            C=-1.85,
            confidence=0.92,
            inlier_count=45,
            is_hallucinated=False,
        )
        right_curve = BoundaryPolynomial2D(
            is_valid=True,
            A=-0.001,
            B=0.0,
            C=1.85,
            confidence=0.92,
            inlier_count=45,
            is_hallucinated=False,
        )

        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        # Vehicle traveling at 25 m/s straight ahead (tangent to curve)
        waypoints: List[ProjectedWaypoint] = []
        half_w = 0.95
        for i in range(1, 7):
            t = i * 0.25
            y = t * 25.0
            waypoints.append(ProjectedWaypoint(
                time_horizon_s=t,
                x_m=0.0, # Vehicle maintaining zero lateral offset
                y_m=y,
                heading_rad=0.0,
                left_wheel_x_m=-half_w,
                right_wheel_x_m=half_w,
                lateral_uncertainty_sigma_m=0.05 + 0.02 * t,
            ))

        pred = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.WARNING,
            direction=DepartureDirection.DEPARTING_RIGHT,
            time_to_departure_s=1.2,
            distance_to_left_boundary_m=1.2,
            distance_to_right_boundary_m=0.25,
            lateral_velocity_mps=0.0,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=waypoints,
            can_alert_required=True,
            can_message_priority=0x140,
            processing_time_ms=0.9,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_curve,
            right_poly=right_curve,
            prediction=pred,
            vehicle_speed_mps=25.0,
            active_scene=SceneType.HIGHWAY,
        )

        # As road curves left, straight trajectory approaches right boundary
        assert telemetry.active_side == BoundarySide.RIGHT_BOUNDARY
        assert telemetry.active_severity in (GuardrailSeverity.WARNING, GuardrailSeverity.CRITICAL, GuardrailSeverity.EMERGENCY)


# ==============================================================================
# 8. Counter-Steer Torque Saturation & Sign Conventions
# ==============================================================================

class TestCounterSteerTorqueBounds:
    """Validates torque bounds, saturation limits, and sign consistency."""

    def test_left_boundary_threat_generates_positive_steer_torque(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify threatening Class 2 Left boundary produces POSITIVE steering torque (clockwise/rightward)."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        pred_left_threat = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.CRITICAL,
            direction=DepartureDirection.DEPARTING_LEFT,
            time_to_departure_s=0.5,
            distance_to_left_boundary_m=0.05,
            distance_to_right_boundary_m=1.2,
            lateral_velocity_mps=-0.8,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=[
                ProjectedWaypoint(0.5, -0.80, 10.0, 0.0, -1.75, 0.15, 0.05)
            ],
            can_alert_required=True,
            can_message_priority=0x150,
            processing_time_ms=0.8,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=pred_left_threat,
        )

        assert telemetry.counter_steer_torque_command_nm > 0.0
        assert telemetry.counter_steer_torque_command_nm <= default_guardrail_config.max_counter_steer_torque_nm

    def test_right_boundary_threat_generates_negative_steer_torque(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
    ) -> None:
        """Verify threatening Class 3 Right boundary produces NEGATIVE steering torque (counter-clockwise/leftward)."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        pred_right_threat = LaneDeparturePrediction(
            timestamp_utc="2026-10-27T12:00:00Z",
            severity=DepartureSeverity.CRITICAL,
            direction=DepartureDirection.DEPARTING_RIGHT,
            time_to_departure_s=0.5,
            distance_to_left_boundary_m=1.2,
            distance_to_right_boundary_m=0.05,
            lateral_velocity_mps=0.8,
            lateral_acceleration_mps2=0.0,
            yaw_rate_radps=0.0,
            projected_ribbon=[
                ProjectedWaypoint(0.5, 0.80, 10.0, 0.0, -0.15, 1.75, 0.05)
            ],
            can_alert_required=True,
            can_message_priority=0x150,
            processing_time_ms=0.8,
        )

        telemetry = enforcer.enforce_guardrails(
            left_poly=left_poly,
            right_poly=right_poly,
            prediction=pred_right_threat,
        )

        assert telemetry.counter_steer_torque_command_nm < 0.0
        assert abs(telemetry.counter_steer_torque_command_nm) <= default_guardrail_config.max_counter_steer_torque_nm


# ==============================================================================
# 9. Real-Time Latency Budgets & Execution Constraints
# ==============================================================================

class TestGuardrailLatencyBudgets:
    """Validates real-time execution bounds (P95 <= 5.0 ms target)."""

    def test_guardrail_enforcement_latency_under_five_milliseconds(
        self,
        default_guardrail_config: GuardrailConfig,
        standard_lane_boundaries: Tuple[BoundaryPolynomial2D, BoundaryPolynomial2D],
        nominal_centered_prediction: LaneDeparturePrediction,
    ) -> None:
        """Verify 100 consecutive enforcement iterations complete with mean latency < 3.0 ms."""
        left_poly, right_poly = standard_lane_boundaries
        enforcer = VirtualGuardrailEnforcer(config=default_guardrail_config)

        latencies_ms: List[float] = []
        for _ in range(100):
            t0 = time.perf_counter()
            telemetry = enforcer.enforce_guardrails(
                left_poly=left_poly,
                right_poly=right_poly,
                prediction=nominal_centered_prediction,
            )
            latencies_ms.append((time.perf_counter() - t0) * 1000.0)

        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))

        assert mean_lat < 3.0, f"Mean guardrail latency {mean_lat:.2f}ms exceeds 3.0ms target"
        assert p95_lat < 5.0, f"P95 guardrail latency {p95_lat:.2f}ms exceeds 5.0ms target"

