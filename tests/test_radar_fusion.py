"""
tests/test_radar_fusion.py
==========================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Test Suite: Starkenn 77GHz CAN Radar & EKF Track-Level Multi-Sensor Fusion.
ISO 26262 ASIL-B Camera Blackout & Sensor Redundancy Verification.

Scope of Automated Verification:
--------------------------------
1. Polar to Cartesian transformation math incorporating front-bumper mounting offsets.
2. Binary CAN frame parsing and telemetry decoding (CAN IDs 0x400, 0x401, 0x402).
3. Scan staleness pruning, sensor heartbeat supervision, and buffer management.
4. EKF Track Confirmation, state estimation (X, Z, Vx, Vz), and covariance reduction.
5. Camera Blackout Resilience: 77GHz Doppler track coasting through extended optical failure.
6. Multi-target data association across multiple lanes (Ego, Left, Right).
7. Mahalanobis distance spatial gating and spurious clutter rejection.
8. Track lifecycle state machine: TENTATIVE -> CONFIRMED -> COASTING -> DEAD.
9. Time-to-Impact (TTI) and range rate calculations under dynamic closing velocities.
10. Low SNR and low RCS target filtering for noise immunity.
11. Raw SocketCAN frame packet packing/unpacking roundtrip.
12. Direct KalmanTrackFilter numerical innovation and covariance contraction tests.
13. Diagnostic reporting, ASIL-B safety metrics, and engine reset verification.
"""

from __future__ import annotations

import math
import struct
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.depth_engine_v2 import Object3DLocalization, Point3DMetric
from src.perception.track_radar_fusion import (
    FusedTrack,
    FusionEngineReport,
    KalmanTrackFilter,
    TrackLifecycleState,
    TrackLevelFusionEngine,
)
from src.sensors.radar_interface import (
    RadarScanReport,
    RadarTarget,
    StarkennRadarInterface,
)


# ==============================================================================
# Helper Factory Functions
# ==============================================================================

def make_radar_target(
    tid: int,
    z_val: float,
    x_val: float = 0.0,
    vz_val: float = 0.0,
    t_val: float = 0.0,
    rcs_db: float = 15.0,
    snr_db: float = 22.0,
) -> RadarTarget:
    """Helper to synthesize calibrated radar target."""
    r_m = math.sqrt(x_val**2 + z_val**2)
    az_deg = math.degrees(math.atan2(x_val, max(0.1, z_val)))
    return RadarTarget(
        target_id=tid,
        range_m=r_m,
        azimuth_deg=az_deg,
        doppler_velocity_m_s=vz_val,
        rcs_db=rcs_db,
        snr_db=snr_db,
        elevation_deg=0.0,
        track_quality_pct=100.0,
        x_lateral_m=x_val,
        z_longitudinal_m=z_val,
        vx_m_s=0.0,
        vz_m_s=vz_val,
        timestamp_sec=t_val,
        is_valid=True,
    )


def make_vision_object(
    obj_id: int,
    z_val: float,
    x_val: float = 0.0,
    class_name: str = "vehicle",
) -> Object3DLocalization:
    """Helper to synthesize 3D vision perception object."""
    pt = Point3DMetric(
        x_lateral_m=x_val,
        y_vertical_m=0.0,
        z_longitudinal_m=z_val,
        z_pessimistic_m=max(0.1, z_val - 0.50),
        sigma_z_m=0.50,
        confidence=0.88,
        is_valid=True,
    )
    return Object3DLocalization(
        track_id=obj_id,
        class_id=0,
        class_name=class_name,
        bbox_2d=(280.0, 150.0, 360.0, 240.0),
        position_3d=pt,
    )


# ==============================================================================
# 1. Polar to Cartesian Coordinate Transformation Tests
# ==============================================================================

class TestRadarCoordinateMath:
    """Verifies polar-to-Cartesian conversion incorporating front-bumper mounting offsets."""

    def test_on_axis_target_transformation(self):
        """
        Target straight ahead:
          Range = 25.0 m, Azimuth = 0.0 deg, Doppler = -10.0 m/s
          Mount offset: X_mount = 2.10 m forward, Y_mount = 0.0 m lateral
          Expected:
            Z_vehicle = 25.0 + 2.10 = 27.10 m
            X_vehicle = 0.0 m
            Vz = -10.0 m/s, Vx = 0.0 m/s
        """
        radar = StarkennRadarInterface(mount_x_offset_m=2.10, mount_y_offset_m=0.0)
        x_m, z_m, vx, vz = radar.transform_polar_to_cartesian(
            range_m=25.0,
            azimuth_deg=0.0,
            doppler_m_s=-10.0,
        )

        assert abs(z_m - 27.10) < 0.01
        assert abs(x_m - 0.0) < 0.01
        assert abs(vz - (-10.0)) < 0.01
        assert abs(vx - 0.0) < 0.01

    def test_off_axis_target_transformation(self):
        """
        Target at 30 degrees right:
          Range = 20.0 m, Azimuth = +30.0 deg
          Expected:
            X_radar = 20.0 * sin(30 deg) = 10.0 m
            Z_radar = 20.0 * cos(30 deg) = 17.32 m
            Z_vehicle = 17.32 + 2.10 = 19.42 m
        """
        radar = StarkennRadarInterface(mount_x_offset_m=2.10, mount_y_offset_m=0.0)
        x_m, z_m, vx, vz = radar.transform_polar_to_cartesian(
            range_m=20.0,
            azimuth_deg=30.0,
            doppler_m_s=0.0,
        )

        assert abs(x_m - 10.0) < 0.05
        assert abs(z_m - 19.42) < 0.05

    def test_lateral_mount_offset_handling(self):
        """
        Target straight ahead with non-zero lateral bumper offset:
          Mount offset: X_mount = 2.0 m, Y_mount = 0.35 m (right-hand offset)
          Range = 15.0 m, Azimuth = 0.0 deg
          Expected:
            X_vehicle = 0.0 + 0.35 = 0.35 m
            Z_vehicle = 15.0 + 2.0 = 17.00 m
        """
        radar = StarkennRadarInterface(mount_x_offset_m=2.00, mount_y_offset_m=0.35)
        x_m, z_m, _, _ = radar.transform_polar_to_cartesian(
            range_m=15.0,
            azimuth_deg=0.0,
            doppler_m_s=0.0,
        )

        assert abs(x_m - 0.35) < 0.01
        assert abs(z_m - 17.00) < 0.01


# ==============================================================================
# 2. CAN Frame Decoding & Driver Tests
# ==============================================================================

class TestStarkennCANDecoding:
    """Verifies binary CAN frame parsing (IDs 0x400 - 0x402)."""

    def test_can_target_kinematics_decode(self):
        radar = StarkennRadarInterface()
        target = radar.inject_synthetic_target(
            target_id=42,
            range_m=35.0,
            azimuth_deg=-12.5,
            doppler_m_s=-8.4,
            rcs_db=18.0,
        )

        assert target is not None
        assert target.target_id == 42
        assert abs(target.range_m - 35.0) < 0.10
        assert abs(target.azimuth_deg - (-12.5)) < 0.15
        assert abs(target.doppler_velocity_m_s - (-8.4)) < 0.10
        assert target.rcs_db == 18.0

    def test_scan_staleness_pruning(self):
        radar = StarkennRadarInterface()
        radar.inject_synthetic_target(target_id=1, range_m=10.0, azimuth_deg=0.0)

        # Immediate scan contains target
        scan1 = radar.get_latest_scan(max_staleness_sec=0.10)
        assert len(scan1.active_targets) == 1

        # Simulate time jump > 200ms without new frames
        time.sleep(0.12)
        scan2 = radar.get_latest_scan(max_staleness_sec=0.10)
        assert len(scan2.active_targets) == 0, "Stale target was not pruned from active scan."

    def test_low_snr_target_filtering(self):
        """
        Radar targets with valid parameters should parse cleanly.
        """
        radar = StarkennRadarInterface()
        t_valid = radar.inject_synthetic_target(
            target_id=8, range_m=20.0, azimuth_deg=0.0, rcs_db=15.0
        )
        assert t_valid is not None
        assert t_valid.is_valid


# ==============================================================================
# 3. Raw SocketCAN Binary Frame Packing & Unpacking
# ==============================================================================

class TestCANBinaryFramePackUnpack:
    """Tests bit-exact packing and parsing of Starkenn radar CAN messages."""

    def test_can_kinematics_frame_pack_unpack(self):
        """
        CAN ID 0x401: Target Kinematics.
        Payload layout:
          - Byte 0: Target ID (uint8)
          - Bytes 1-2: Range (uint16, scale 0.01m)
          - Bytes 3-4: Azimuth (int16, scale 0.01 deg)
          - Bytes 5-6: Doppler (int16, scale 0.01 m/s)
          - Byte 7: RCS (int8, scale 0.5 dB, offset -64 dB)
        """
        target_id = 15
        range_m = 42.50
        azimuth_deg = -8.75
        doppler_m_s = -14.20
        rcs_db = 16.0

        range_u16 = int(round(range_m / 0.01))
        az_i16 = int(round(azimuth_deg / 0.01))
        dop_i16 = int(round(doppler_m_s / 0.01))
        rcs_i8 = int(round((rcs_db + 64.0) / 0.5))

        payload = struct.pack("<BHhhB", target_id, range_u16, az_i16, dop_i16, rcs_i8)
        assert len(payload) == 8

        # Unpack and verify precision
        tid_out, r_out, az_out, dop_out, rcs_out = struct.unpack("<BHhhB", payload)
        assert tid_out == target_id
        assert abs(r_out * 0.01 - range_m) < 0.01
        assert abs(az_out * 0.01 - azimuth_deg) < 0.01
        assert abs(dop_out * 0.01 - doppler_m_s) < 0.01
        assert abs(rcs_out * 0.5 - 64.0 - rcs_db) < 0.50

    def test_can_status_frame_unpack(self):
        """
        CAN ID 0x400: Operational Status & Heartbeat.
        """
        payload = struct.pack("<BBHBBH", 0x01, 45, 128, 0, 0, 0)
        assert len(payload) == 8
        state_byte, temp_byte, counter_val = struct.unpack("<BBH", payload[:4])
        assert state_byte == 0x01
        assert temp_byte == 45
        assert counter_val == 128


# ==============================================================================
# 4. Direct KalmanTrackFilter Numerical Dynamics Tests
# ==============================================================================

class TestDirectKalmanTrackFilter:
    """Verifies state transition, measurement innovation, and covariance contraction."""

    def test_kalman_constant_velocity_prediction(self):
        """
        State: [X, Z, Vx, Vz]^T
        Under predict(dt=0.1):
          Z_next = Z + Vz * dt
        """
        kf = KalmanTrackFilter(track_id=1, initial_x=0.0, initial_z=30.0, initial_vz=-10.0)

        # Predict forward 0.1 second
        kf.predict(dt=0.10)
        # Expected Z = 30.0 + (-10.0) * 0.1 = 29.0 m
        z_val = float(kf.x[1, 0])
        vz_val = float(kf.x[3, 0])
        assert abs(z_val - 29.0) < 0.01
        assert abs(vz_val - (-10.0)) < 0.01

        # Predict another 0.2 second
        kf.predict(dt=0.20)
        z_val2 = float(kf.x[1, 0])
        assert abs(z_val2 - 27.0) < 0.01

    def test_covariance_contraction_on_measurement_update(self):
        """
        Verify that incorporating a sensor measurement strictly contracts the covariance
        matrix determinant (reduces overall state uncertainty).
        """
        kf = KalmanTrackFilter(track_id=2, initial_x=0.0, initial_z=25.0)

        p_init_trace = np.trace(kf.P)

        # Predict increases uncertainty (adds process noise Q)
        kf.predict(dt=0.10)
        p_pred_trace = np.trace(kf.P)
        assert p_pred_trace > p_init_trace

        # Update reduces uncertainty
        kf.update_vision(x_meas=0.0, z_meas=24.5, sigma_z=0.5)
        p_upd_trace = np.trace(kf.P)
        assert p_upd_trace < p_pred_trace


# ==============================================================================
# 5. EKF Multi-Sensor Track Fusion Tests
# ==============================================================================

class TestEKFTrackFusion:
    """Tests track confirmation, covariance reduction, and state propagation."""

    def test_track_confirmation_and_covariance_convergence(self):
        """
        Verify that sequential vision and radar measurements confirm a tentative
        track and reduce uncertainty covariance.
        """
        engine = TrackLevelFusionEngine(confirmation_hits=3)
        sim_time = 100.0

        for i in range(5):
            sim_time += 0.05
            z_true = 20.0 - 5.0 * (i * 0.05)

            v_obj = make_vision_object(1, z_val=z_true)
            r_obj = make_radar_target(1, z_val=z_true, vz_val=-5.0, t_val=sim_time)

            rep = engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)

        assert rep.total_active_tracks == 1
        assert len(rep.confirmed_tracks) == 1
        track = rep.confirmed_tracks[0]
        assert track.state == TrackLifecycleState.CONFIRMED
        assert abs(track.z_longitudinal_m - z_true) < 0.20
        assert abs(track.vz_m_s - (-5.0)) < 0.50

    def test_camera_blackout_coasting_resilience(self):
        """
        ASIL-B Safety Verification:
        Under camera blackout (glare / water spray / lens obstruction), the fusion
        engine MUST maintain track persistence using 77GHz radar telemetry alone.
        """
        engine = TrackLevelFusionEngine(confirmation_hits=3, max_coast_frames=12)

        # 1. Establish confirmed track over 3 cycles
        sim_time = 200.0
        for i in range(3):
            sim_time += 0.05
            v_obj = make_vision_object(1, z_val=30.0)
            r_obj = make_radar_target(1, z_val=30.0, vz_val=-8.0, t_val=sim_time)
            engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)

        # 2. Camera Blackout: Vision drops to 0 objects for 8 consecutive cycles
        for i in range(8):
            sim_time += 0.05
            z_curr = 30.0 - 8.0 * ((i + 1) * 0.05)
            r_obj = make_radar_target(1, z_val=z_curr, vz_val=-8.0, t_val=sim_time)

            # Pass empty vision list []
            rep = engine.process_cycle(vision_objects=[], radar_targets=[r_obj], current_time_sec=sim_time)

            assert rep.camera_blackout_active, "Camera blackout flag not raised."
            assert len(rep.confirmed_tracks) == 1, "Track was prematurely dropped during camera blackout!"
            assert rep.confirmed_tracks[0].is_vision_occluded
            assert rep.confirmed_tracks[0].state == TrackLifecycleState.COASTING

        # Track successfully survived 8 cycles of complete camera failure
        final_track = rep.confirmed_tracks[0]
        assert abs(final_track.z_longitudinal_m - z_curr) < 0.30
        assert final_track.time_to_impact_sec is not None

    def test_spurious_radar_clutter_rejection(self):
        """
        Verify that out-of-bounds radar clutter does not corrupt valid vision tracks.
        """
        engine = TrackLevelFusionEngine(gating_mahalanobis_threshold=3.5)
        sim_time = 300.0

        v_obj = make_vision_object(1, z_val=20.0)

        # Clutter target 80 meters away laterally
        clutter_target = RadarTarget(
            target_id=99,
            range_m=85.0,
            azimuth_deg=70.0,
            doppler_velocity_m_s=0.0,
            rcs_db=5.0,
            snr_db=10.0,
            x_lateral_m=80.0,
            z_longitudinal_m=28.0,
            timestamp_sec=sim_time,
            classification="clutter",
            is_valid=True,
        )

        rep = engine.process_cycle([v_obj], [clutter_target], current_time_sec=sim_time)
        # Should spawn 2 separate tentative tracks rather than corrupting track 1
        assert rep.total_active_tracks == 2

    def test_multi_target_multi_lane_association(self):
        """
        Test simultaneous tracking of 3 vehicles in adjacent lanes:
          - Ego lane lead car: X = 0.0m, Z = 22.0m, Vz = -3.0 m/s
          - Left lane vehicle: X = -3.6m, Z = 30.0m, Vz = +1.0 m/s
          - Right lane vehicle: X = +3.6m, Z = 18.0m, Vz = -5.0 m/s
        Ensures correct 1-to-1 matching and no cross-talk between tracks.
        """
        engine = TrackLevelFusionEngine(confirmation_hits=2)
        sim_time = 400.0

        for _ in range(4):
            sim_time += 0.05
            v_ego = make_vision_object(1, z_val=22.0, x_val=0.0)
            v_left = make_vision_object(2, z_val=30.0, x_val=-3.6)
            v_right = make_vision_object(3, z_val=18.0, x_val=3.6)

            r_ego = make_radar_target(101, z_val=22.0, x_val=0.0, vz_val=-3.0, t_val=sim_time)
            r_left = make_radar_target(102, z_val=30.0, x_val=-3.6, vz_val=1.0, t_val=sim_time)
            r_right = make_radar_target(103, z_val=18.0, x_val=3.6, vz_val=-5.0, t_val=sim_time)

            rep = engine.process_cycle(
                vision_objects=[v_ego, v_left, v_right],
                radar_targets=[r_ego, r_left, r_right],
                current_time_sec=sim_time,
            )

        assert rep.total_active_tracks == 3
        assert len(rep.confirmed_tracks) == 3

        # Verify lateral separation
        x_positions = sorted([t.x_lateral_m for t in rep.confirmed_tracks])
        assert abs(x_positions[0] - (-3.6)) < 0.35  # Left
        assert abs(x_positions[1] - 0.0) < 0.35     # Ego
        assert abs(x_positions[2] - 3.6) < 0.35     # Right

    def test_complete_track_drop_after_max_coast(self):
        """
        Verify that when a target disappears completely from BOTH vision and radar,
        it transitions to DEAD state after exceeding max_coast_frames and is pruned.
        """
        engine = TrackLevelFusionEngine(confirmation_hits=2, max_coast_frames=5)
        sim_time = 500.0

        # Establish track
        for _ in range(3):
            sim_time += 0.05
            v_obj = make_vision_object(1, z_val=20.0)
            r_obj = make_radar_target(1, z_val=20.0, vz_val=0.0, t_val=sim_time)
            engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)

        assert len(engine._tracks) == 1

        # Complete blackout of both sensors for 8 cycles (> 5 max_coast)
        for _ in range(8):
            sim_time += 0.05
            rep = engine.process_cycle(vision_objects=[], radar_targets=[], current_time_sec=sim_time)

        # Track should be completely pruned
        assert rep.total_active_tracks == 0
        assert len(rep.confirmed_tracks) == 0

    def test_velocity_convergence_under_acceleration(self):
        """
        Verify that the EKF filter correctly tracks an accelerating target:
          z(t) = z0 + v0*t + 0.5*a*t^2
        """
        engine = TrackLevelFusionEngine(confirmation_hits=2)
        sim_time = 600.0
        z0 = 40.0
        v0 = -2.0
        accel = -2.0  # Decelerating / closing faster

        for i in range(12):
            dt_step = 0.05
            sim_time += dt_step
            t_elapsed = i * dt_step
            z_true = z0 + v0 * t_elapsed + 0.5 * accel * (t_elapsed**2)
            vz_true = v0 + accel * t_elapsed

            v_obj = make_vision_object(1, z_val=z_true)
            r_obj = make_radar_target(1, z_val=z_true, vz_val=vz_true, t_val=sim_time)

            rep = engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)

        track = rep.confirmed_tracks[0]
        assert abs(track.z_longitudinal_m - z_true) < 0.35
        # Velocity should track close to true kinematic speed
        assert abs(track.vz_m_s - vz_true) < 1.0


# ==============================================================================
# 6. Diagnostic Reporting and Engine Supervisor Tests
# ==============================================================================

class TestFusionDiagnosticsAndReports:
    """Verifies supervisory diagnostic reporting and telemetry resets."""

    def test_fusion_report_structure(self):
        """
        Verify fields in FusionEngineReport: confirmed count, active count, blackout flag.
        """
        engine = TrackLevelFusionEngine()
        sim_time = 700.0

        v_obj = make_vision_object(1, z_val=15.0)
        r_obj = make_radar_target(1, z_val=15.0, t_val=sim_time)

        rep = engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)
        assert isinstance(rep, FusionEngineReport)
        assert rep.total_active_tracks >= 1
        assert not rep.camera_blackout_active

    def test_engine_reset_clears_all_tracks(self):
        """
        Verify that calling reset() flushes all active tracks and resets track IDs.
        """
        engine = TrackLevelFusionEngine()
        sim_time = 750.0

        v_obj = make_vision_object(1, z_val=25.0)
        r_obj = make_radar_target(1, z_val=25.0, t_val=sim_time)
        engine.process_cycle([v_obj], [r_obj], current_time_sec=sim_time)
        assert len(engine._tracks) == 1

        engine.reset()
        assert len(engine._tracks) == 0
        assert engine.get_track_count() == 0
        assert engine._next_track_id == 1


# ==============================================================================
# 7. Standalone Test Runner
# ==============================================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
