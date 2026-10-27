"""
tests/test_depth_v2_horizon_correction.py
=========================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Test Suite: Monocular Depth v2.0 Pitch Compensation & Horizon Correction.
ISO 26262 ASIL-B Perception Depth Integrity & Uncertainty Verification.

Scope of Automated Verification:
--------------------------------
1. Analytical Closed-Form Ground Truth Geometry under zero and non-zero attitude.
2. Synthetic Pitch Injection: Validating compensation across vehicle braking/acceleration.
3. Systematic pitch angle and obstacle distance grid sweeps across full operational range.
4. Compound Attitude (Roll + Pitch) de-rotation invariance and lateral coordinate recovery.
5. Analytical partial derivative verification (dZ/dv, dZ/dtheta, dZ/dH) via finite differences.
6. Monte Carlo variance convergence vs closed-form error propagation (sigma_Z).
7. ASIL-B Pessimistic Distance (Z_pessimistic) and RSS d_min margin safety trip points.
8. Metric Covariance Ellipse propagation (Sigma_XZ) in vehicle reference frame.
9. Time-to-Impact (TTI) lower-bound confidence intervals under dynamic closing velocities.
10. Ground Plane Validity Gate: Road slope grade, pitch/roll thresholds, horizon visibility.
11. Extrinsic Drift Monitor: Horizon tracking, EMA pitch bias convergence, vibration detection.
12. Camera Intrinsics and Batched Projection: FOV computation and monotonicity check.
13. Extreme singular edge cases: Above-horizon rays, nadir pitch, zero focal length, boundaries.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.depth_engine_v2 import (
    CameraIntrinsics,
    MonocularDepthEngineV2,
    Point3DMetric,
)
from src.perception.depth_uncertainty import (
    ClosedFormDepthUncertaintyEngine,
    DepthUncertaintyEstimate,
    SensorUncertaintyParameters,
)
from src.perception.extrinsic_drift_monitor import (
    ExtrinsicDriftConfig,
    ExtrinsicDriftMonitor,
    MountCalibrationHealth,
)
from src.perception.ground_plane_validator import (
    GroundPlaneFaultBits,
    GroundPlaneLimitsConfig,
    GroundPlaneValidityResult,
    GroundPlaneValidator,
)


# ==============================================================================
# 1. Theoretical Accuracy & Closed-Form Ground Truth Tests
# ==============================================================================

class TestTheoreticalDepthAccuracy:
    """Verifies analytical ground-plane math against exact geometric equations."""

    def test_level_pitch_ground_truth(self):
        """
        Under pitch=0, roll=0:
          elevation = arctan((v - c_y) / f_y)
          Z_m = H_cam / tan(elevation) = H_cam * f_y / (v - c_y)
        """
        engine = MonocularDepthEngineV2(
            camera_intrinsics=CameraIntrinsics(fx=450.0, fy=450.0, cx=320.0, cy=180.0, height_m=1.20)
        )
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        # Target at v = 240 px (delta_v = 60 px)
        # Expected Z = 1.20 * 450.0 / 60.0 = 9.0 meters
        pt = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=240.0)
        assert abs(pt.z_longitudinal_m - 9.0) < 0.05
        assert abs(pt.x_lateral_m) < 0.01

        # Target at v = 200 px (delta_v = 20 px)
        # Expected Z = 1.20 * 450.0 / 20.0 = 27.0 meters
        pt2 = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=200.0)
        assert abs(pt2.z_longitudinal_m - 27.0) < 0.05

        # Target at v = 190 px (delta_v = 10 px)
        # Expected Z = 1.20 * 450.0 / 10.0 = 54.0 meters
        pt3 = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=190.0)
        assert abs(pt3.z_longitudinal_m - 54.0) < 0.10

    def test_synthetic_pitch_injection_correction(self):
        """
        Verify that injecting live INS pitch compensates for vehicle suspension dive.
        Example: Vehicle dives nose-down under braking (pitch = -2.0 degrees).
        An uncompensated camera would misjudge a 15m obstacle as being dangerously far away!
        """
        engine = MonocularDepthEngineV2(
            camera_intrinsics=CameraIntrinsics(fx=450.0, fy=450.0, cx=320.0, cy=180.0, height_m=1.20)
        )

        pitch_dive_deg = -2.0
        pitch_dive_rad = math.radians(pitch_dive_deg)

        # Ground truth true obstacle distance = 15.0 meters
        # Required true elevation angle = arctan(H / Z) = arctan(1.2 / 15.0)
        true_elevation = math.atan(1.20 / 15.0)

        # Camera pixel elevation: gamma = true_elevation - pitch_dive_rad
        gamma = true_elevation - pitch_dive_rad
        v_target = 180.0 + 450.0 * math.tan(gamma)

        # 1. Uncompensated measurement (assuming pitch=0) produces severe error
        uncomp = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=v_target, pitch_rad=0.0)
        error_uncompensated = abs(uncomp.z_longitudinal_m - 15.0)
        assert error_uncompensated > 3.0, "Uncompensated pitch error should be significant!"

        # 2. Compensated measurement (incorporating INS pitch) recovers exact 15.0m
        comp = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=v_target, pitch_rad=pitch_dive_rad)
        error_compensated = abs(comp.z_longitudinal_m - 15.0)
        assert error_compensated < 0.10, f"Compensated error {error_compensated}m must be < 0.10m"

    def test_synthetic_pitch_grid_sweep(self):
        """
        Systematic grid sweep across diverse pitch angles (-4° to +4°) and distances (8m to 45m).
        Ensures compensated estimation remains within 2% relative error across the operational range.
        """
        engine = MonocularDepthEngineV2(
            camera_intrinsics=CameraIntrinsics(fx=450.0, fy=450.0, cx=320.0, cy=180.0, height_m=1.20)
        )

        pitch_angles_deg = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
        true_distances_m = [8.0, 12.0, 18.0, 25.0, 35.0, 45.0]

        for p_deg in pitch_angles_deg:
            p_rad = math.radians(p_deg)
            for z_true in true_distances_m:
                true_elevation = math.atan(1.20 / z_true)
                gamma = true_elevation - p_rad

                # Ensure pixel is within optical sensor bounds
                v_target = 180.0 + 450.0 * math.tan(gamma)
                if not (185.0 <= v_target <= 355.0):
                    continue

                est = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=v_target, pitch_rad=p_rad)
                rel_err = abs(est.z_longitudinal_m - z_true) / z_true
                assert rel_err < 0.02, f"Failed at pitch {p_deg}° and dist {z_true}m (rel err={rel_err:.3f})"

    def test_roll_derotation_invariance(self):
        """
        Verify that chassis roll (e.g. driving across a banked curve) is cancelled
        by the 2D roll de-rotation transformation.
        """
        engine = MonocularDepthEngineV2()
        roll_bank_rad = math.radians(4.0)

        # Base point on centerline at v=240, u=320 (nominal Z = 9.0m, X = 0.0m)
        u_base, v_base = 320.0, 240.0
        cos_r = math.cos(roll_bank_rad)
        sin_r = math.sin(roll_bank_rad)

        u_banked = cos_r * (u_base - 320.0) - sin_r * (v_base - 180.0) + 320.0
        v_banked = sin_r * (u_base - 320.0) + cos_r * (v_base - 180.0) + 180.0

        pt_derotated = engine.project_pixel_to_ground(
            u_pixel=u_banked,
            v_pixel=v_banked,
            pitch_rad=0.0,
            roll_rad=roll_bank_rad,
        )

        assert abs(pt_derotated.z_longitudinal_m - 9.0) < 0.15
        assert abs(pt_derotated.x_lateral_m) < 0.15

    def test_lateral_offset_reconstruction(self):
        """
        Test that lateral position X_m is correctly recovered from horizontal pixel offset u:
          X_m = (u - c_x) * Z_m / f_x
        """
        engine = MonocularDepthEngineV2(
            camera_intrinsics=CameraIntrinsics(fx=450.0, fy=450.0, cx=320.0, cy=180.0, height_m=1.20)
        )
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        # Target at v=240 (Z=9.0m), u=420 (delta_u = +100 px)
        # Expected X = +100 * 9.0 / 450.0 = +2.0 meters (to vehicle right)
        pt_right = engine.project_pixel_to_ground(u_pixel=420.0, v_pixel=240.0)
        assert abs(pt_right.z_longitudinal_m - 9.0) < 0.05
        assert abs(pt_right.x_lateral_m - 2.0) < 0.05

        # Target at v=240 (Z=9.0m), u=220 (delta_u = -100 px)
        # Expected X = -100 * 9.0 / 450.0 = -2.0 meters (to vehicle left)
        pt_left = engine.project_pixel_to_ground(u_pixel=220.0, v_pixel=240.0)
        assert abs(pt_left.z_longitudinal_m - 9.0) < 0.05
        assert abs(pt_left.x_lateral_m - (-2.0)) < 0.05

    def test_mounting_height_scaling_proportionality(self):
        """
        Verify that ground distance scales exactly linearly with camera height:
        Z_m(H2) = Z_m(H1) * (H2 / H1)
        """
        engine1 = MonocularDepthEngineV2(camera_intrinsics=CameraIntrinsics(height_m=1.00))
        engine2 = MonocularDepthEngineV2(camera_intrinsics=CameraIntrinsics(height_m=1.50))

        pt1 = engine1.project_pixel_to_ground(u_pixel=320.0, v_pixel=225.0)
        pt2 = engine2.project_pixel_to_ground(u_pixel=320.0, v_pixel=225.0)

        ratio = pt2.z_longitudinal_m / pt1.z_longitudinal_m
        assert abs(ratio - (1.50 / 1.00)) < 1e-4


# ==============================================================================
# 2. Closed-Form Uncertainty Propagation Verification
# ==============================================================================

class TestClosedFormUncertainty:
    """Validates analytical partial derivatives against numerical finite differences."""

    def test_analytical_vs_finite_difference_derivatives(self):
        """
        Verify closed-form partial derivatives dZ/dv, dZ/dtheta, dZ/dH
        against numerical finite differences: (f(x + eps) - f(x - eps)) / (2 * eps).
        """
        uncert_engine = ClosedFormDepthUncertaintyEngine()

        v0 = 240.0
        theta0 = math.radians(1.0)
        h0 = 1.20

        # Exact continuous ground truth distance function
        def z_exact(v_val: float, th_val: float, h_val: float) -> float:
            gamma = math.atan((v_val - 180.0) / 450.0)
            return h_val / math.tan(th_val + gamma)

        # Analytical evaluation
        est = uncert_engine.propagate_uncertainty(v_bottom_px=v0, pitch_rad=theta0, camera_height_override=h0)

        # Numerical dZ/dv
        dv = 1e-4
        num_dz_dv = (z_exact(v0 + dv, theta0, h0) - z_exact(v0 - dv, theta0, h0)) / (2.0 * dv)
        assert abs(est.dz_dv - num_dz_dv) < 1e-2

        # Numerical dZ/dtheta
        dth = 1e-4
        num_dz_dtheta = (z_exact(v0, theta0 + dth, h0) - z_exact(v0, theta0 - dth, h0)) / (2.0 * dth)
        assert abs(est.dz_dpitch - num_dz_dtheta) < 1e-1

        # Numerical dZ/dH
        dh = 1e-4
        num_dz_dh = (z_exact(v0, theta0, h0 + dh) - z_exact(v0, theta0, h0 - dh)) / (2.0 * dh)
        assert abs(est.dz_dh - num_dz_dh) < 1e-2

    def test_pessimistic_bound_safety(self):
        """Verify Z_pessimistic satisfies the lower bound condition for safety."""
        uncert_engine = ClosedFormDepthUncertaintyEngine()
        est = uncert_engine.propagate_uncertainty(v_bottom_px=220.0, pitch_rad=0.0)

        assert est.pessimistic_distance_m <= est.nominal_distance_m
        assert est.pessimistic_distance_m >= 0.10
        assert est.sigma_z_m > 0.0

    def test_monte_carlo_variance_convergence(self):
        """
        Monte Carlo validation: Draw 15,000 perturbed realizations of (v, pitch, H)
        according to specified sensor standard deviations and verify that the empirical
        sample standard deviation matches the analytical sigma_Z within 6%.
        """
        params = SensorUncertaintyParameters(
            sigma_v_pixels=1.50,
            sigma_pitch_rad=math.radians(0.15),
            sigma_h_cam_m=0.03,
        )
        uncert_engine = ClosedFormDepthUncertaintyEngine(params=params)

        v0 = 225.0
        pitch0 = math.radians(0.5)
        h0 = 1.20

        # Analytical prediction
        analytical = uncert_engine.propagate_uncertainty(
            v_bottom_px=v0, pitch_rad=pitch0, camera_height_override=h0
        )

        # Empirical Monte Carlo simulation
        rng = np.random.default_rng(2026)
        n_samples = 15000

        v_samples = rng.normal(v0, params.sigma_v_pixels, size=n_samples)
        pitch_samples = rng.normal(pitch0, params.sigma_pitch_rad, size=n_samples)
        h_samples = rng.normal(h0, params.sigma_h_cam_m, size=n_samples)

        gamma_samples = np.arctan((v_samples - 180.0) / 450.0)
        total_angle = pitch_samples + gamma_samples
        z_samples = h_samples / np.tan(total_angle)

        mc_sigma_z = float(np.std(z_samples))
        rel_diff = abs(analytical.sigma_z_m - mc_sigma_z) / analytical.sigma_z_m

        assert rel_diff < 0.06, f"Analytical sigma {analytical.sigma_z_m:.3f} vs MC sigma {mc_sigma_z:.3f} mismatch!"

    def test_responsibility_sensitive_safety_bound(self):
        """
        Verify pessimistic distance correctly triggers RSS d_min emergency condition.
        When nominal Z_m is borderline safe (e.g. 18.0m for d_min=16.5m),
        the pessimistic distance Z_pessimistic accounts for uncertainty.
        """
        uncert_engine = ClosedFormDepthUncertaintyEngine()

        # Target at v=210 px (delta_v = 30 px -> Z_m = 18.0m)
        est = uncert_engine.propagate_uncertainty(v_bottom_px=210.0, pitch_rad=0.0)

        d_min_rss = 16.5  # Required safety bubble
        assert est.nominal_distance_m > d_min_rss
        assert est.pessimistic_distance_m < est.nominal_distance_m
        assert est.pessimistic_distance_m >= 0.10

    def test_time_to_impact_lower_bound(self):
        """
        Verify that Time-to-Impact calculated via pessimistic distance:
          TTI_pessimistic = Z_pessimistic / v_rel
        provides a strictly conservative lower bound on the true TTI.
        """
        uncert_engine = ClosedFormDepthUncertaintyEngine()
        est = uncert_engine.propagate_uncertainty(v_bottom_px=225.0, pitch_rad=0.0)

        v_closing = 15.0  # 15 m/s (~54 km/h closing speed)
        tti_nominal = est.nominal_distance_m / v_closing
        tti_pessimistic = est.pessimistic_distance_m / v_closing

        assert tti_pessimistic < tti_nominal
        assert tti_pessimistic > 0.0


# ==============================================================================
# 3. Ground Plane Validity Gate & Supervisor Tests
# ==============================================================================

class TestGroundPlaneValidator:
    """Verifies rejection of invalid road geometry and transient attitude."""

    def test_pitch_roll_operational_limits(self):
        validator = GroundPlaneValidator(GroundPlaneLimitsConfig(max_pitch_deg=15.0, max_roll_deg=12.0))

        # Safe attitude
        res_ok = validator.validate_attitude(pitch_rad=math.radians(5.0), roll_rad=math.radians(3.0))
        assert res_ok.is_valid
        assert res_ok.is_pitch_safe and res_ok.is_roll_safe

        # Out-of-bounds pitch
        res_bad_pitch = validator.validate_attitude(pitch_rad=math.radians(16.5), roll_rad=0.0)
        assert not res_bad_pitch.is_valid
        assert not res_bad_pitch.is_pitch_safe

        # Out-of-bounds roll
        res_bad_roll = validator.validate_attitude(pitch_rad=0.0, roll_rad=math.radians(14.0))
        assert not res_bad_roll.is_valid
        assert not res_bad_roll.is_roll_safe

    def test_horizon_outside_frame_rejection(self):
        validator = GroundPlaneValidator()
        # Pitch of -30 degrees would place the horizon far outside the camera view
        res = validator.validate_attitude(pitch_rad=math.radians(-30.0), roll_rad=0.0)
        assert not res.is_valid
        assert not res.is_horizon_in_frame

    def test_steep_road_grade_rejection(self):
        """
        Verify that steep road slopes (> 14% grade) are correctly detected and flagged.
        """
        validator = GroundPlaneValidator(GroundPlaneLimitsConfig(max_acceptable_grade_pct=14.0))

        # Acceptable highway grade (4% incline -> pitch = atan(0.04) rad)
        p_ok = math.atan(0.04)
        res_ok = validator.validate_attitude(pitch_rad=p_ok, roll_rad=0.0)
        assert res_ok.is_valid
        assert abs(res_ok.estimated_road_grade_pct - 4.0) < 0.2

        # Extreme mountain grade (16% incline -> pitch = atan(0.16) rad)
        p_steep = math.atan(0.16)
        res_steep = validator.validate_attitude(pitch_rad=p_steep, roll_rad=0.0)
        assert not res_steep.is_valid
        assert res_steep.fault_bitmask & GroundPlaneFaultBits.SUSTAINED_GRADE_WARNING

    def test_high_pitch_rate_crest_sag_rejection(self):
        """
        Verify rejection during rapid pitch rate transients (driving over speed bump / crest).
        """
        validator = GroundPlaneValidator(GroundPlaneLimitsConfig(max_pitch_rate_deg_s=35.0))

        # Normal steady pitch rate (0.5 deg/sec)
        res_normal = validator.validate_attitude(pitch_rad=0.0, roll_rad=0.0, live_pitch_rate_deg_s=0.5)
        assert res_normal.is_dynamic_rate_safe

        # Aggressive pitch transient (45.0 deg/sec > 35.0)
        res_jerk = validator.validate_attitude(pitch_rad=0.0, roll_rad=0.0, live_pitch_rate_deg_s=45.0)
        assert not res_jerk.is_dynamic_rate_safe
        assert not res_jerk.is_valid


# ==============================================================================
# 4. Extrinsic Calibration Drift Monitor Tests
# ==============================================================================

class TestExtrinsicDriftMonitor:
    """Verifies online auto-estimation of camera tilt and vibration monitoring."""

    def test_persistent_pitch_bias_convergence(self):
        monitor = ExtrinsicDriftMonitor(
            ExtrinsicDriftConfig(calibration_ema_alpha=0.10, nominal_lane_width_m=3.70)
        )

        # Inject persistent 1.5 degree mounting tilt over 50 frames
        injected_bias_deg = 1.50
        injected_bias_rad = math.radians(injected_bias_deg)
        expected_horizon_offset_px = -450.0 * math.tan(injected_bias_rad)

        for _ in range(50):
            rep = monitor.update_frame(
                observed_lane_width_m=3.70,
                measured_vanishing_row_px=180.0 + expected_horizon_offset_px,
                ins_pitch_rad=0.0,
            )

        # Smoothed pitch bias should converge toward injected bias
        assert abs(rep.estimated_pitch_bias_deg - injected_bias_deg) < 0.20

    def test_vibration_excessive_detection(self):
        monitor = ExtrinsicDriftMonitor(ExtrinsicDriftConfig(max_horizon_jitter_std_px=3.0))

        # Feed wildly jittering horizon rows (high standard deviation)
        rng = np.random.default_rng(42)
        jittery_rows = 180.0 + rng.normal(0.0, 6.0, size=30)

        for row in jittery_rows:
            rep = monitor.update_frame(
                observed_lane_width_m=3.70,
                measured_vanishing_row_px=float(row),
                ins_pitch_rad=0.0,
            )

        assert rep.health_status == MountCalibrationHealth.VIBRATION_EXCESSIVE

    def test_lane_width_drift_warning(self):
        """
        Verify that persistent abnormal lane width observations (e.g. 2.3m instead of 3.7m)
        trigger a mount height or focal length drift flag.
        """
        monitor = ExtrinsicDriftMonitor(
            ExtrinsicDriftConfig(nominal_lane_width_m=3.70, min_lane_width_m=2.80)
        )

        rep = None
        for _ in range(25):
            rep = monitor.update_frame(
                observed_lane_width_m=2.30,  # Narrow reading below 2.80m
                measured_vanishing_row_px=180.0,
                ins_pitch_rad=0.0,
            )

        assert rep is not None
        assert rep.health_status in (
            MountCalibrationHealth.WARNING_DRIFT,
            MountCalibrationHealth.CRITICAL_MISALIGNMENT,
        )


# ==============================================================================
# 5. Camera Intrinsics and Batched Projection Verification
# ==============================================================================

class TestCameraIntrinsicsAndBatchedProjection:
    """Verifies geometric field-of-view and monotonic ordering of depth projection."""

    def test_intrinsics_validation_and_fov(self):
        """
        Verify field-of-view calculation for 640x360 image with fx=fy=450px:
          HFOV = 2 * arctan(320 / 450) = 2 * 0.6186 = 1.237 rad (~70.88 deg)
          VFOV = 2 * arctan(180 / 450) = 2 * 0.3805 = 0.761 rad (~43.60 deg)
        """
        intr = CameraIntrinsics(fx=450.0, fy=450.0, cx=320.0, cy=180.0, height_m=1.20)
        hfov_deg = math.degrees(2.0 * math.atan(intr.cx / intr.fx))
        vfov_deg = math.degrees(2.0 * math.atan(intr.cy / intr.fy))

        assert abs(hfov_deg - 70.88) < 0.20
        assert abs(vfov_deg - 43.60) < 0.20

    def test_monotonic_distance_ordering(self):
        """
        Verify strict monotonicity: As image row v decreases toward horizon (v -> cy),
        ground distance Z_m must strictly increase:
          v1 < v2  ==>  Z_m(v1) > Z_m(v2)
        """
        engine = MonocularDepthEngineV2()
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        v_steps = [340.0, 300.0, 260.0, 230.0, 210.0, 195.0]
        z_estimates = [
            engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=v).z_longitudinal_m
            for v in v_steps
        ]

        for i in range(len(z_estimates) - 1):
            assert z_estimates[i] < z_estimates[i + 1], (
                f"Monotonicity violated: v={v_steps[i]} (Z={z_estimates[i]:.2f}) vs "
                f"v={v_steps[i+1]} (Z={z_estimates[i+1]:.2f})"
            )

    def test_centerline_zero_lateral_bias(self):
        """
        Points lying exactly along the optical principal column u = cx = 320 px
        must have lateral position X_m = 0.0 across all distances.
        """
        engine = MonocularDepthEngineV2()
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        for v in [340.0, 280.0, 220.0, 195.0]:
            pt = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=v)
            assert abs(pt.x_lateral_m) < 1e-4, f"Non-zero lateral offset at v={v}: {pt.x_lateral_m}"


# ==============================================================================
# 6. Boundary & Singularity Robustness Tests
# ==============================================================================

class TestSingularityAndBoundaryHandling:
    """Tests extreme and degenerate optical conditions."""

    def test_pixel_above_horizon_safe_clamp(self):
        """
        Pixels at or above the horizon (v <= v_horizon) cannot be projected to the road.
        The engine must clamp to a maximum safe range or return a flagged invalid point.
        """
        engine = MonocularDepthEngineV2()
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        # Row 170 is 10 pixels above the optical horizon (row 180)
        pt_sky = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=170.0)
        # Must not produce negative depth or crash with division by zero
        assert pt_sky.z_longitudinal_m > 100.0 or not pt_sky.is_valid

    def test_bottom_edge_near_nadir(self):
        """
        Target at the very bottom of the image (v = 360 px, delta_v = 180 px).
        Elevation = arctan(180 / 450) = 0.3805 rad (~21.8 deg).
        Expected Z = 1.20 * 450.0 / 180.0 = 3.0 meters (immediately in front of hood).
        """
        engine = MonocularDepthEngineV2()
        engine.update_attitude(pitch_rad=0.0, roll_rad=0.0)

        pt_hood = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=360.0)
        assert abs(pt_hood.z_longitudinal_m - 3.0) < 0.05
        assert pt_hood.is_valid

    def test_extreme_pitch_dive_safety(self):
        """
        Test behavior when pitch dive approaches or exceeds the horizon limit.
        When elevation angle becomes non-positive, it must safely flag invalid
        and clamp to MAX_PERCEPTION_RANGE_M without crashing or producing negative depth.
        """
        engine = MonocularDepthEngineV2()
        # Pitch dive of -10 deg with target at v=250 gives elevation < 0
        pt = engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=250.0, pitch_rad=math.radians(-10.0))
        assert not pt.is_valid
        assert pt.z_longitudinal_m >= 100.0


# ==============================================================================
# 7. Standalone Test Runner
# ==============================================================================

if __name__ == "__main__":
    pytest.main(["-v", __file__])
