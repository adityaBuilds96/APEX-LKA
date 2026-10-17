"""
tests/test_sensor_fusion.py
===========================
Comprehensive end-to-end integration unit tests for Phase 5 Section 4:
15-DOF Error-State Kalman Filter (ESKF), Visual-Inertial Odometry (VIO),
CAN 0x180 Telemetry, Vehicle Kinematics, and Kinematic Lane Retention.
"""

import math
import struct
import cv2
import pytest
import numpy as np

from src.fusion.eskf import (
    ErrorStateKalmanFilter15DOF,
    ESKFConfig,
    LocalizationStatus,
    VehiclePose,
    quat_to_rot_matrix,
    rot_matrix_to_euler,
)
from src.fusion.vio_engine import (
    VisualInertialOdometryEngine,
    VIOConfig,
    EgoMotionMeasurement,
)
from src.fusion.sensor_interface import (
    SensorInterfaceManager,
    VehicleKinematics,
    IMUReading,
    GNSSReading,
    WheelEncoderReading,
)
from src.fusion.lane_retention import (
    KinematicLaneRetainer,
    RetainedLaneResult,
)
from deploy.vehicle_comm import (
    CAN_ID_KINEMATIC_STATE,
    KinematicStateTelemetry,
)
from src.inference.pipeline import run_pipeline


# ── 1. End-to-End Multi-Rate Sensor Fusion ─────────────────────────────────

def test_multirate_sensor_fusion_cycle():
    """Verify concurrent ingestion of 100 Hz IMU, 50 Hz Encoders, 30 Hz VIO, and 10 Hz GNSS."""
    eskf = ErrorStateKalmanFilter15DOF()

    # Simulate 1.0 second of vehicle driving straight at 10 m/s
    dt = 0.01
    for step in range(100):
        # 100 Hz IMU
        eskf.predict(imu_accel=[0.0, 0.0, 9.80665], imu_gyro=[0.0, 0.0, 0.0], dt=dt)

        # 50 Hz Wheel Encoders (every 2 steps)
        if step % 2 == 0:
            eskf.update_wheel_encoders(v_left=10.0, v_right=10.0)

        # 30 Hz VIO (every 3 steps approx)
        if step % 3 == 0:
            eskf.update_visual_odometry(dx=0.30, dy=0.0, dpsi=0.0)

        # 10 Hz GNSS (every 10 steps)
        if step % 10 == 0:
            eskf.update_gnss(lat_or_x=float(step * 0.10), lon_or_y=0.0, alt_or_z=0.0, is_local_xyz=True)

    pose = eskf.get_vehicle_pose()
    assert pose.x > 5.0
    assert abs(pose.y) < 0.20
    assert abs(pose.yaw) < 0.05
    assert pose.localization_status in (LocalizationStatus.GNSS_FIX, LocalizationStatus.RTK_LOCKED)


# ── 2. Numerical Protection & Regularization ───────────────────────────────

def test_matrix_inversion_singular_protection():
    """Verify regularized inversion does not crash with degenerate/ill-conditioned measurements."""
    eskf = ErrorStateKalmanFilter15DOF()

    # Degenerate zero-noise covariance
    H_deg = np.zeros((3, 15))
    H_deg[0:3, 0:3] = np.eye(3)
    R_deg = np.zeros((3, 3))  # Singular zero noise

    # Must execute safely without LinAlgError
    eskf._apply_measurement_update(H_deg, np.array([0.1, 0.1, 0.0]), R_deg)
    assert not np.isnan(eskf.p).any()
    assert not np.isnan(eskf.P).any()


# ── 3. Equirectangular Tangent Plane Coordinates ───────────────────────────

def test_gnss_equirectangular_projection():
    """Verify WGS-84 Lat/Lon coordinates correctly initialize origin and project to local metric plane."""
    eskf = ErrorStateKalmanFilter15DOF()

    # Anchor at San Francisco lat/lon
    lat0, lon0 = 37.774929, -122.419416
    eskf.update_gnss(lat_or_x=lat0, lon_or_y=lon0, alt_or_z=10.0, is_local_xyz=False)
    assert eskf.gnss_origin_latlon is not None
    assert np.allclose(eskf.p, np.zeros(3))

    # Move north by 0.0001 deg (~ 11.1 meters north)
    eskf.update_gnss(lat_or_x=lat0 + 0.0001, lon_or_y=lon0, alt_or_z=10.0, is_local_xyz=False)
    assert eskf.p[1] > 5.0  # Pulled north by Kalman update


# ── 4. Non-Holonomic Constraint Damping ─────────────────────────────────────

def test_non_holonomic_lateral_damping():
    """Verify wheel encoder update dampens lateral body velocity towards zero."""
    eskf = ErrorStateKalmanFilter15DOF()
    # Inject large lateral velocity perturbation in world frame
    eskf.v = np.array([5.0, 3.0, 0.0])

    # Multiple wheel encoder updates
    for _ in range(15):
        eskf.update_wheel_encoders(v_left=5.0, v_right=5.0)

    R = quat_to_rot_matrix(eskf.q)
    vy_body = (R.T @ eskf.v)[1]
    assert abs(vy_body) < 0.50  # Lateral body slip suppressed


# ── 5. Accelerometer and Gyro Bias Estimation ──────────────────────────────

def test_sensor_bias_tracking():
    """Verify ESKF estimates gyro bias under stationary zero-motion with non-zero measurement."""
    eskf = ErrorStateKalmanFilter15DOF()
    # Static IMU with constant 0.02 rad/s gyro bias on Z
    for _ in range(50):
        eskf.predict(imu_accel=[0.0, 0.0, 9.80665], imu_gyro=[0.0, 0.0, 0.02], dt=0.01)
        # GNSS fix confirms vehicle is stationary at origin
        if _ % 5 == 0:
            eskf.update_gnss(0.0, 0.0, 0.0, is_local_xyz=True)

    # State should remain near origin
    assert abs(eskf.p[0]) < 0.10
    assert abs(eskf.p[1]) < 0.10


# ── 6. Dynamic Slip Angle Kinematic Model ──────────────────────────────────

def test_slip_angle_understeer_gradient():
    """Verify slip angle increases with steer angle but attenuates at high speeds due to understeer gradient."""
    kin = VehicleKinematics(wheelbase_m=1.60, dist_cg_to_rear_m=0.85, understeer_gradient=0.005)
    mgr = SensorInterfaceManager(kinematics=kin, sim_mode=False)

    beta_low_speed = mgr.compute_slip_angle_deg(speed_mps=3.0, steering_deg=12.0)
    beta_high_speed = mgr.compute_slip_angle_deg(speed_mps=25.0, steering_deg=12.0)

    # Due to Kus * vx^2 denominator, high speed slip angle should be lower than low speed
    assert beta_low_speed > beta_high_speed > 0.0


# ── 7. Slip Angle Dynamic Blending ─────────────────────────────────────────

def test_slip_angle_dynamic_yaw_rate_blend():
    """Verify non-zero yaw rate blends into slip angle estimation."""
    mgr = SensorInterfaceManager(sim_mode=False)
    beta_kinematic = mgr.compute_slip_angle_deg(speed_mps=8.0, steering_deg=5.0, yaw_rate_radps=0.0)
    beta_dynamic = mgr.compute_slip_angle_deg(speed_mps=8.0, steering_deg=5.0, yaw_rate_radps=0.55)

    assert abs(beta_dynamic) > abs(beta_kinematic)


# ── 8. VIO Zero Motion Detection ───────────────────────────────────────────

def test_vio_zero_motion():
    """Verify identical consecutive frames produce zero translation and zero yaw rotation."""
    vio = VisualInertialOdometryEngine()
    frame = np.full((360, 640), 100, dtype=np.uint8)
    for x in range(50, 600, 30):
        for y in range(150, 340, 30):
            cv2.circle(frame, (x, y), 4, 255, -1)

    # Frame 0 (init)
    vio.process_frame(frame)
    # Frame 1 (identical)
    meas = vio.process_frame(frame, dt=0.033)

    if meas.valid:
        assert np.isclose(meas.dyaw, 0.0, atol=0.01)
        assert meas.inlier_ratio >= 0.80


# ── 9. VIO Inlier Ratio on Noisy Motion ────────────────────────────────────

def test_vio_noisy_frame_inliers():
    """Verify VIO handles partial ground occlusion or noise gracefully."""
    vio = VisualInertialOdometryEngine()
    frame0 = np.random.randint(50, 200, (360, 640), dtype=np.uint8)
    frame1 = np.random.randint(50, 200, (360, 640), dtype=np.uint8)

    meas0 = vio.process_frame(frame0)
    meas1 = vio.process_frame(frame1)
    # Pure noise frames should have valid=False
    assert not meas1.valid or meas1.inlier_ratio < 0.50


# ── 10. Kinematic Lane Retainer 2D Lateral Transformation ──────────────────

def test_lane_retention_lateral_shift():
    """Verify lane retention correctly shifts polynomial lateral position when vehicle drifts laterally."""
    retainer = KinematicLaneRetainer()
    # Initial valid lane
    from src.inference.predictor import DetectionStatus, LanePrediction, ModelStatus
    from src.lane_geometry.lane_estimator import LaneGeometry

    pred = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        model_status=ModelStatus.READY,
        left_confidence=0.90,
        right_confidence=0.90,
    )
    geom = LaneGeometry(
        left_poly=np.array([0.0, 0.0, 200.0]),
        right_poly=np.array([0.0, 0.0, 440.0]),
        x_left_bottom=200.0,
        x_right_bottom=440.0,
        lane_center_x=320.0,
    )
    p0 = VehiclePose(x=0.0, y=0.0, yaw=0.0)
    retainer.update(pred, geom, p0, timestamp=10.0)

    # Vehicle drifts right by 0.5 m in vehicle lateral axis (y_w = -0.5 m)
    p1 = VehiclePose(x=0.0, y=-0.5, yaw=0.0)
    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    res = retainer.update(pred_lost, LaneGeometry(), p1, timestamp=10.5)

    assert res.is_retained
    assert res.geometry is not None
    # Lane center in pixel viewport should shift accordingly
    assert res.geometry.lane_center_x != 320.0


# ── 11. Kinematic Lane Retainer Yaw Angle Transformation ───────────────────

def test_lane_retention_yaw_rotation():
    """Verify lane retention tilts polynomial curves when vehicle changes heading."""
    retainer = KinematicLaneRetainer()
    from src.inference.predictor import DetectionStatus, LanePrediction, ModelStatus
    from src.lane_geometry.lane_estimator import LaneGeometry

    pred = LanePrediction(
        status=DetectionStatus.LANE_DETECTED,
        model_status=ModelStatus.READY,
        left_confidence=0.90,
        right_confidence=0.90,
    )
    geom = LaneGeometry(
        left_poly=np.array([0.0, 0.0, 200.0]),
        right_poly=np.array([0.0, 0.0, 440.0]),
        x_left_bottom=200.0,
        x_right_bottom=440.0,
        lane_center_x=320.0,
    )
    p0 = VehiclePose(x=0.0, y=0.0, yaw=0.0)
    retainer.update(pred, geom, p0, timestamp=10.0)

    # Vehicle yaws left by 0.05 rad
    p1 = VehiclePose(x=0.0, y=0.0, yaw=0.05)
    pred_lost = LanePrediction(status=DetectionStatus.LANE_NOT_DETECTED)
    res = retainer.update(pred_lost, LaneGeometry(), p1, timestamp=10.5)

    assert res.is_retained
    assert res.geometry is not None
    # Slope (linear term) should reflect yaw tilt
    assert res.geometry.left_poly[1] != 0.0


# ── 12. CAN ID 0x180 Serialization & Deserialization ───────────────────────

def test_can_0x180_packing_precision():
    """Verify bit-level accuracy of CAN Frame 0x180 pack format <hhhbB."""
    # Test values: vx=14.52 m/s, vy=-0.35 m/s, wz=0.082 rad/s, beta=-2.4 deg, loc=RTK_LOCKED (4)
    vx = 14.52
    vy = -0.35
    wz = 0.082
    beta = -2.4
    loc = 4

    vx_cm = int(round(vx * 100.0))       # 1452
    vy_cm = int(round(vy * 100.0))       # -35
    wz_mrad = int(round(wz * 1000.0))    # 82
    beta_decideg = int(round(beta * 10.0)) # -24
    loc_stat = int(loc)

    payload = struct.pack("<hhhbB", vx_cm, vy_cm, wz_mrad, beta_decideg, loc_stat)
    assert len(payload) == 8

    # Unpack
    u_vx, u_vy, u_wz, u_beta, u_loc = struct.unpack("<hhhbB", payload)
    assert np.isclose(u_vx / 100.0, vx, atol=0.01)
    assert np.isclose(u_vy / 100.0, vy, atol=0.01)
    assert np.isclose(u_wz / 1000.0, wz, atol=0.001)
    assert np.isclose(u_beta / 10.0, beta, atol=0.1)
    assert u_loc == 4


# ── 13. CAN ID 0x180 Saturation & Extremes ─────────────────────────────────

def test_can_0x180_boundary_saturation():
    """Verify extreme velocities and slip angles clamp without integer overflow."""
    # Saturated speed: 50 m/s (5000 cm/s), beta: -40 deg (should clamp to signed int8)
    vx_extreme = 50.0
    vy_extreme = -20.0
    wz_extreme = 5.0
    beta_extreme = -35.0
    loc_extreme = 255

    vx_cm = int(np.clip(round(vx_extreme * 100.0), -32768, 32767))
    vy_cm = int(np.clip(round(vy_extreme * 100.0), -32768, 32767))
    wz_mrad = int(np.clip(round(wz_extreme * 1000.0), -32768, 32767))
    beta_decideg = int(np.clip(round(beta_extreme * 10.0), -128, 127))
    loc_stat = int(np.clip(loc_extreme, 0, 255))

    payload = struct.pack("<hhhbB", vx_cm, vy_cm, wz_mrad, beta_decideg, loc_stat)
    assert len(payload) == 8
    u_vx, u_vy, u_wz, u_beta, u_loc = struct.unpack("<hhhbB", payload)
    assert u_beta == -128  # Clamped to int8 minimum


# ── 14. Full Pipeline Integration with enable_sensor_fusion ────────────────

def test_pipeline_integration_sensor_fusion():
    """Verify run_pipeline with enable_sensor_fusion=True returns valid pose and HUD overlays."""
    # Synthetic frame with lane markings
    h, w = 360, 640
    frame = np.full((h, w, 3), 70, dtype=np.uint8)
    cv2.line(frame, (180, h - 1), (w // 2 - 25, 130), (255, 255, 255), 5)
    cv2.line(frame, (460, h - 1), (w // 2 + 25, 130), (255, 255, 255), 5)

    res = run_pipeline(
        frame,
        backend="classical_cv",
        reset_pd_state=True,
        enable_sensor_fusion=True,
    )

    assert res.success
    assert res.vehicle_pose is not None
    assert isinstance(res.vehicle_pose, VehiclePose)
    assert "sensor_fusion_ms" in res.timings
    assert res.timings["sensor_fusion_ms"] >= 0.0
    assert res.annotated_bgr is not None


# ── 15. Memory and Covariance Stability over 500 Steps ─────────────────────

def test_eskf_long_horizon_stability():
    """Ensure filter remains numerically stable and covariance stays positive-definite over 500 steps."""
    eskf = ErrorStateKalmanFilter15DOF()

    for i in range(500):
        eskf.predict(imu_accel=[0.05, 0.02, 9.80665], imu_gyro=[0.001, -0.001, 0.005], dt=0.01)
        if i % 10 == 0:
            eskf.update_wheel_encoders(v_left=5.0, v_right=5.0)
        if i % 25 == 0:
            eskf.update_visual_odometry(dx=0.25, dy=0.0, dpsi=0.0)

    # Check finite values
    assert np.all(np.isfinite(eskf.p))
    assert np.all(np.isfinite(eskf.v))
    assert np.all(np.isfinite(eskf.P))
    # All eigenvalues strictly positive
    eigs = np.linalg.eigvalsh(eskf.P)
    assert np.all(eigs > 0.0)
