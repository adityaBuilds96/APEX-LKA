"""
tests/test_perception_accuracy.py
==================================
APEX-Vision Core v4.0 • Master Perception Accuracy, Driver & Control Verification Suite
ISO 26262 ASIL-B Aligned Comprehensive Unit & Integration Test Suite

Validates:
1. Classical Computer Vision Precision Stack (Dynamic ROI, Dynamic IPM Homography,
   Multi-Color Space Isolation, Sobel-X, Dual Canny, Hough |m|>=0.35, RANSAC Parabola).
2. Deep Learning Multi-Task Semantic Segmentation & Dave-2 End-to-End Steering.
3. Strict No-Flip Augmentation Rule (swapping Class 2 and Class 3, inverting steering).
4. Off-Road Traversability Head & 2D BEV Costmap Generation.
5. Vehicle Actuation Control Mathematics (Cross-track error e_lat, Stanley Controller,
   PID with anti-windup clamping, Slew rate limiting, and Rollover Governor).
6. Starkenn Orion 77GHz CAN Automotive Radar Driver (CAN IDs 0x400-0x408 track frames).
7. MTi-680 RTK GNSS/INS Driver (400 Hz AHRS + cm-level RTK positioning via XBUS).
8. Hardware Hot-Plug Discovery Engine (HAL) & Event Notifier.
9. Comprehensive Perception & Tracking Evaluation Metrics (IoU, mAP, AbsRel, HOTA).
10. Environmental Stress-Testing Multi-Condition Report Compilation.
"""

import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import pytest
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.perception.color_space_isolator import ColorSpaceIsolator, ColorSpaceProfile
from src.perception.ipm_transformer import IPMTransformer, CameraExtrinsics, CameraIntrinsics
from src.perception.roi_processor import DynamicROIProcessor, DynamicHorizonConfig
from src.perception.edge_hough_pipeline import EdgeHoughPipeline, HoughConfig
from src.perception.cv_precision_stack import CVPrecisionStack, PrecisionStackConfig
from src.models.lane_segmentation import (
    MultiTaskLaneNet,
    Dave2SteeringHead,
    NoFlipRuleValidator,
    BackboneType,
    ModelConfidenceCalibrator,
)
from src.perception.traversability_head import (
    TraversabilityCostmapGenerator,
    TerrainSurfaceClass,
    ASILBTerrainSafetyMonitor,
)
from src.control.lanekeep_controllers import (
    StanleyController,
    PIDController,
    ActuationSaturationLimiter,
    LaneKeepSupervisor,
    VehicleKinematicProfile,
)
from src.hardware.drivers.starkenn_radar import (
    StarkennRadarDriver,
    RadarTrack,
    StarkennCANFrameDecoder,
)
from src.hardware.drivers.mti680_driver import (
    MTi680Driver,
    XsensMessage,
    MTData2Packet,
    MTi680PerceptionBridge,
)
from src.hardware.hal_discovery import (
    HALDiscoveryDaemon,
    HardwareDeviceType,
    HardwareDeviceStatus,
)
from src.hardware.hal_notifier import HALEventNotifier, HALEventType, HALSeverity
from evaluation.lane_metrics import (
    ConfusionMatrixAccumulator,
    GeometricLaneF1Evaluator,
    OODUncertaintyDetector,
)
from evaluation.detection_metrics import (
    BoundingBox2D,
    DetectionMetricsEvaluator,
)
from evaluation.depth_metrics import DepthMetricsEvaluator
from evaluation.tracking_metrics import (
    TrackedBox,
    HOTAEvaluator,
    ClearMOTEvaluator,
)
from evaluation.per_condition_report import (
    EnvironmentalStressEvaluator,
    EnvironmentalCondition,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Test Fixtures
# ═══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_camera_frame() -> np.ndarray:
    """Creates a synthetic 640x360 BGR camera frame with synthetic white and yellow lane boundaries."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    # Ground asphalt
    img[180:, :] = 45
    # Sky
    img[:180, :] = 140

    # Draw synthetic white dashed right line
    cv2.line(img, (400, 200), (580, 355), (240, 240, 240), thickness=6)
    # Draw synthetic yellow solid left line
    cv2.line(img, (240, 200), (60, 355), (0, 220, 240), thickness=6)
    return img


@pytest.fixture
def standard_camera_intrinsics() -> CameraIntrinsics:
    """Returns standard pinhole camera intrinsics for 640x360 sensor."""
    return CameraIntrinsics(
        fx=450.0,
        fy=450.0,
        cx=320.0,
        cy=180.0,
        width=640,
        height=360,
    )


@pytest.fixture
def standard_camera_extrinsics() -> CameraExtrinsics:
    """Returns standard camera mounting extrinsics on BAJA Buggy roll hoop."""
    return CameraExtrinsics(
        mounting_height_m=1.20,
        pitch_rad=math.radians(5.0),
        roll_rad=0.0,
        yaw_rad=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Computer Vision Precision Stack Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestCVPrecisionStack:
    """Unit and integration tests for the Classical CV Precision Stack."""

    def test_color_space_isolator_white_yellow(self, synthetic_camera_frame: np.ndarray) -> None:
        """Verifies multi-color space fusion isolates both white and yellow lanes."""
        isolator = ColorSpaceIsolator(profile=ColorSpaceProfile.ASPHALT_BALANCED)
        feature_mask = isolator.isolate_lane_features(synthetic_camera_frame)

        assert feature_mask.shape == (360, 640)
        assert feature_mask.dtype == np.uint8

        # Left yellow line pixel check (around x=100, y=320)
        left_region = feature_mask[310:330, 90:110]
        assert np.sum(left_region > 0) > 10, "Yellow left boundary was not isolated"

        # Right white line pixel check (around x=540, y=320)
        right_region = feature_mask[310:330, 530:550]
        assert np.sum(right_region > 0) > 10, "White right boundary was not isolated"

    def test_ipm_transformer_metric_scaling(
        self,
        standard_camera_intrinsics: CameraIntrinsics,
        standard_camera_extrinsics: CameraExtrinsics,
    ) -> None:
        """Verifies IPM homography maintains strictly 0.0185 m/px metric scaling and deadband gating."""
        transformer = IPMTransformer(
            intrinsics=standard_camera_intrinsics,
            extrinsics=standard_camera_extrinsics,
            bev_width_px=400,
            bev_height_px=500,
        )

        assert abs(transformer.METRIC_SCALE_M_PER_PX - 0.0185) < 1e-5

        # Test bidirectional coordinate transformation
        u_test, v_test = 200.0, 250.0
        x_world, y_world = transformer.pixel_to_world_bev(u_test, v_test)
        u_recon, v_recon = transformer.world_to_pixel_bev(x_world, y_world)

        assert abs(u_test - u_recon) < 0.1
        assert abs(v_test - v_recon) < 0.1

        # Test pitch deadband: delta < 0.5 deg should NOT trigger homography rebuild
        rebuilt = transformer.update_ahrs_attitude(
            pitch_rad=math.radians(5.2), # delta = 0.2 deg < 0.5 deg
            roll_rad=0.0,
            yaw_rad=0.0,
        )
        assert not rebuilt, "Homography should not rebuild when pitch delta is within deadband"

        # Delta > 0.5 deg SHOULD trigger rebuild
        rebuilt_large = transformer.update_ahrs_attitude(
            pitch_rad=math.radians(6.1), # delta = 1.1 deg > 0.5 deg
            roll_rad=0.0,
            yaw_rad=0.0,
        )
        assert rebuilt_large, "Homography must rebuild when pitch delta exceeds 0.5 deg"

    def test_dynamic_roi_horizon_shift(
        self,
        standard_camera_intrinsics: CameraIntrinsics,
        standard_camera_extrinsics: CameraExtrinsics,
    ) -> None:
        """Verifies dynamic horizon shift formula: Delta y = -fy * tan(pitch)."""
        processor = DynamicROIProcessor(
            intrinsics=standard_camera_intrinsics,
            extrinsics=standard_camera_extrinsics,
        )

        # Baseline horizon at nominal pitch 5.0 deg
        h_nom = processor.compute_dynamic_horizon(pitch_rad=math.radians(5.0))
        assert 130 <= h_nom <= 160

        # Pitching up (+3 deg = 8.0 deg total): horizon should move downward on sensor
        h_up = processor.compute_dynamic_horizon(pitch_rad=math.radians(8.0))
        assert h_up < h_nom, "Pitching up must shift horizon downward in pixel coordinates"

        # Verify multi-scale pyramid sub-windows
        dummy_img = np.zeros((360, 640, 3), dtype=np.uint8)
        pyramid = processor.extract_multiscale_pyramid(dummy_img, pitch_rad=math.radians(5.0))
        assert "near_field" in pyramid
        assert "mid_field" in pyramid
        assert "far_field" in pyramid

    def test_edge_hough_slope_rejection_and_ransac(self) -> None:
        """Verifies Hough transform strictly rejects horizontal guardrails (|m| < 0.35) and fits parabola."""
        pipeline = EdgeHoughPipeline()
        bev_canvas = np.zeros((500, 400), dtype=np.uint8)

        # Draw a horizontal line (|m| = 0.0) -> MUST be rejected
        cv2.line(bev_canvas, (50, 250), (350, 250), 255, thickness=3)

        # Draw a slanted line (|m| = 1.5) -> MUST be accepted
        cv2.line(bev_canvas, (100, 100), (200, 250), 255, thickness=3)

        lines, valid_pts = pipeline.extract_filtered_hough_segments(bev_canvas)

        # Verify slope filter
        for line in lines:
            x1, y1, x2, y2 = line
            slope = abs((y2 - y1) / max(1e-4, (x2 - x1)))
            assert slope >= pipeline.MIN_SLOPE_THRESHOLD, f"Horizontal line with slope {slope} was not rejected"

        # Test RANSAC Parabolic Fitting
        y_pts = np.linspace(50, 450, 50)
        # Parabola: x = 0.0005 * y^2 - 0.2 * y + 200
        x_pts = 0.0005 * (y_pts ** 2) - 0.2 * y_pts + 200.0
        pts = np.column_stack([x_pts, y_pts])

        poly = pipeline.fit_ransac_parabola(pts)
        assert poly is not None
        a, b, c = poly
        assert abs(a - 0.0005) < 0.0002
        assert abs(b - (-0.2)) < 0.05

        # Compute metric radius of curvature
        radius = pipeline.compute_curvature_radius(a, b, y_eval=0.0)
        assert radius >= pipeline.MIN_RADIUS_OF_CURVATURE_M

    def test_cv_precision_stack_orchestration(
        self,
        synthetic_camera_frame: np.ndarray,
        standard_camera_intrinsics: CameraIntrinsics,
        standard_camera_extrinsics: CameraExtrinsics,
    ) -> None:
        """Tests full end-to-end execution of CVPrecisionStack master pipeline."""
        stack = CVPrecisionStack(
            intrinsics=standard_camera_intrinsics,
            extrinsics=standard_camera_extrinsics,
        )

        result = stack.process_frame(
            bgr_frame=synthetic_camera_frame,
            ahrs_pitch_rad=math.radians(5.0),
            ahrs_roll_rad=0.0,
            ahrs_yaw_rad=0.0,
        )

        assert result is not None
        assert result.success or (result.epistemic_uncertainty > 0.5)
        assert result.bev_canvas.shape == (500, 400, 3)
        assert isinstance(result.cross_track_error_m, float)
        assert isinstance(result.radius_of_curvature_m, float)
        assert result.execution_latency_ms > 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Deep Learning Segmentation & End-to-End Steering Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestDeepPerceptionModels:
    """Validates multi-task deep neural architectures and strict No-Flip rule."""

    def test_lane_segmentation_multitask_forward(self) -> None:
        """Verifies forward pass of MultiTaskLaneNet producing 4-class mask and steering angle."""
        model = MultiTaskLaneNet(backbone=BackboneType.MOBILENET_V3_SMALL, num_classes=4)
        model.eval()

        # Input: 1 batch, 3 channels, 360 height, 640 width
        dummy_input = torch.randn(1, 3, 360, 640)
        with torch.no_grad():
            seg_logits, steer_angle = model(dummy_input)

        assert seg_logits.shape == (1, 4, 360, 640)
        assert steer_angle.shape == (1, 1)

        # Steering angle must be within continuous bounded interval [-0.4886, +0.4886] rad (+/- 28 deg)
        pred_steer = steer_angle.item()
        assert -0.489 <= pred_steer <= 0.489

    def test_dave2_steering_angle_range(self) -> None:
        """Verifies Dave2 continuous steering head respects vehicle profile clipping limits."""
        head = Dave2SteeringHead(in_features=576, max_steering_rad=math.radians(28.0))
        dummy_feat = torch.randn(4, 576)
        out = head(dummy_feat)

        assert out.shape == (4, 1)
        max_bound = math.radians(28.0) + 1e-4
        assert torch.all(out <= max_bound)
        assert torch.all(out >= -max_bound)

    def test_strict_no_flip_rule_validation(self) -> None:
        """Verifies NoFlipRuleValidator swaps Class 2 (Left) & Class 3 (Right) and inverts steering."""
        # Create a synthetic 4-class mask with Left Boundary (2) on left, Right Boundary (3) on right
        mask = np.zeros((100, 200), dtype=np.int64)
        mask[30:70, 20:40] = 2   # Left boundary class
        mask[30:70, 160:180] = 3 # Right boundary class

        steering_angle = 0.25 # Radians turning right

        flipped_mask, flipped_steering = NoFlipRuleValidator.apply_safe_horizontal_flip(
            mask=mask,
            steering_angle_rad=steering_angle,
        )

        # Class 2 in original position (20:40) maps to (160:180) in flipped image,
        # but because it is now on the right, it MUST become Class 3 (Right Boundary)
        assert np.all(flipped_mask[30:70, 160:180] == 3), "Class 2 must be swapped to Class 3 upon flip"

        # Class 3 in original position (160:180) maps to (20:40), and MUST become Class 2
        assert np.all(flipped_mask[30:70, 20:40] == 2), "Class 3 must be swapped to Class 2 upon flip"

        # Steering angle must be negated
        assert abs(flipped_steering - (-0.25)) < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Traversability Head & Off-Road Terrain Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestTraversabilityHead:
    """Validates 5-class off-road terrain classifier and ditch dropoff safety monitor."""

    def test_traversability_head_costmap(self) -> None:
        """Verifies 2D BEV costmap generation from multi-class terrain predictions."""
        generator = TraversabilityCostmapGenerator(grid_size=(200, 400), cell_resolution_m=0.10)

        # Create dummy terrain logits: 5 classes (Dirt, Vegetation, Mud, Rocks, Ditch)
        dummy_seg = np.zeros((200, 400), dtype=np.uint8)
        dummy_seg[50:150, 150:250] = TerrainSurfaceClass.DRIVABLE_DIRT.value
        dummy_seg[10:30, 180:220] = TerrainSurfaceClass.ROCKS_BOULDERS.value
        dummy_seg[170:190, 180:220] = TerrainSurfaceClass.DITCH_DROPOFF.value

        costmap = generator.generate_costmap_from_classes(dummy_seg)

        assert costmap.shape == (200, 400)
        assert costmap.dtype == np.float32

        # Dirt should have low cost
        dirt_cost = costmap[100, 200]
        assert dirt_cost < 0.20

        # Rocks should have high cost
        rock_cost = costmap[20, 200]
        assert rock_cost > 0.80

        # Ditch dropoff should have maximum lethal cost
        ditch_cost = costmap[180, 200]
        assert ditch_cost >= 0.99

    def test_asil_b_terrain_safety_monitor(self) -> None:
        """Verifies ASIL-B safety monitor halts vehicle if lethal obstacles intersect path."""
        monitor = ASILBTerrainSafetyMonitor()

        # Safe costmap
        safe_costmap = np.full((200, 400), 0.10, dtype=np.float32)
        safe_pass, reason = monitor.verify_path_safety(safe_costmap, planned_path_y_indices=[180, 190, 200])
        assert safe_pass
        assert reason == "PATH_CLEAR"

        # Hazardous costmap with ditch on planned corridor
        hazard_costmap = np.copy(safe_costmap)
        hazard_costmap[190, 195:205] = 1.00 # Lethal ditch
        hazard_pass, hazard_reason = monitor.verify_path_safety(hazard_costmap, planned_path_y_indices=[180, 190, 200])
        assert not hazard_pass
        assert "LETHAL_TERRAIN_OBSTACLE" in hazard_reason


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Control Mathematics & Actuation Safety Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestControlMathematics:
    """Unit tests for Stanley, PID with anti-windup, and Actuation Saturation Limiting."""

    def test_cross_track_error_lookahead(self) -> None:
        """Verifies speed-scheduled lookahead: L_d(v) = max(L_min, k_speed * v)."""
        profile = VehicleKinematicProfile(
            max_steering_angle_deg=28.0,
            max_steering_rate_deg_per_s=40.0,
            wheelbase_m=1.65,
            lookahead_min_m=2.5,
            lookahead_speed_gain_s=0.35,
        )

        supervisor = LaneKeepSupervisor(profile=profile)

        # At v = 0 m/s: lookahead should be L_min = 2.5m
        ld_0 = supervisor.compute_lookahead_distance(velocity_mps=0.0)
        assert abs(ld_0 - 2.50) < 1e-4

        # At v = 10.0 m/s: lookahead should be 0.35 * 10 = 3.5m > 2.5m
        ld_10 = supervisor.compute_lookahead_distance(velocity_mps=10.0)
        assert abs(ld_10 - 3.50) < 1e-4

    def test_stanley_controller_dynamics(self) -> None:
        """Verifies front-axle Stanley formula: delta = theta_e + arctan(k * e / (v + eps))."""
        stanley = StanleyController(k_gain=1.2, epsilon_v=0.5)

        # Case 1: Zero error -> zero steering
        delta_zero = stanley.compute_steering_command(heading_error_rad=0.0, cross_track_error_m=0.0, velocity_mps=5.0)
        assert abs(delta_zero) < 1e-5

        # Case 2: Positive cross-track error (vehicle right of lane center) -> negative steering (turn left)
        delta_left = stanley.compute_steering_command(heading_error_rad=0.0, cross_track_error_m=0.5, velocity_mps=5.0)
        # Expected: arctan(1.2 * 0.5 / (5.0 + 0.5)) = arctan(0.6 / 5.5) = arctan(0.10909) approx 0.1086 rad
        expected = math.atan((1.2 * 0.5) / 5.5)
        assert abs(delta_left - expected) < 1e-4

    def test_pid_controller_anti_windup(self) -> None:
        """Verifies PID error integration clamps at anti-windup ceiling |I| <= 0.15 rad."""
        pid = PIDController(kp=0.8, ki=0.2, kd=0.05, anti_windup_limit=0.15)

        # Accumulate large steady-state error over 50 steps
        for _ in range(50):
            pid.compute(error=1.0, dt=0.05)

        assert abs(pid.integrator_state) <= 0.15001, "Integrator must clamp to anti-windup limit"

    def test_actuation_saturation_limiter_and_rollover(self) -> None:
        """Verifies steering is hard clamped to 28.0 deg, 40.0 deg/s slew, and rollover governor."""
        limiter = ActuationSaturationLimiter(
            max_angle_deg=28.0,
            max_rate_deg_per_s=40.0,
            wheelbase_m=1.65,
            rollover_accel_limit_g=0.65,
        )

        dt = 0.05 # 50 ms step
        max_step = math.radians(40.0 * dt) # 2.0 degrees max per step

        # Test 1: Slew rate limiting from 0 to 10 degrees in one step
        cmd_rad = math.radians(10.0)
        filtered = limiter.filter_command(target_angle_rad=cmd_rad, current_angle_rad=0.0, dt=dt, velocity_mps=2.0)
        assert abs(filtered - max_step) < 1e-4, "Slew rate limiter must clamp step change"

        # Test 2: Absolute steering saturation clamp (attempting 35 degrees)
        cmd_35 = math.radians(35.0)
        # Steady state at current=27.5 deg
        clamped = limiter.filter_command(target_angle_rad=cmd_35, current_angle_rad=math.radians(27.5), dt=dt, velocity_mps=2.0)
        assert clamped <= math.radians(28.0) + 1e-5, "Command must be clamped to 28.0 degrees max"

        # Test 3: Rollover governor at high speed (v = 12.5 m/s = 45 km/h)
        # Rollover limit: delta_limit = arctan(1.65 * 6.37 / 156.25) = arctan(0.0673) = 3.85 degrees
        rollover_clamped = limiter.filter_command(
            target_angle_rad=math.radians(20.0),
            current_angle_rad=math.radians(3.0),
            dt=dt,
            velocity_mps=12.5,
        )
        assert rollover_clamped <= math.radians(4.5), "High-speed rollover governor must attenuate steering angle"


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Hardware Drivers: Starkenn 77GHz Radar & MTi-680 GNSS/INS Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHardwareDrivers:
    """Verifies Starkenn 77GHz CAN Radar and MTi-680 RTK GNSS/INS protocol parsing."""

    def test_starkenn_radar_driver_can_decoding(self) -> None:
        """Verifies decoding CAN ID 0x400-0x408 track frames into 3D Cartesian coordinates."""
        driver = StarkennRadarDriver()

        # Construct synthetic CAN frame for Track 7:
        # Range = 25.00m (2500 counts = 0x09C4)
        # Azimuth = +10.00 deg (1000 counts + 9000 offset = 10000 = 0x2710)
        # Velocity = -5.00 m/s (-500 / 0.02 + 5000 = 4750 counts = 0x128E)
        # RCS = 15.0 dBsm (15.0 / 0.5 + 100 = 130 counts = 0x82)
        track_id = 7
        range_high, range_low = 0x09, 0xC4
        az_high, az_low = 0x27, 0x10
        vel_high, vel_low = 0x12, 0x8E
        rcs_val = 130

        payload = bytes([track_id, range_high, range_low, az_high, az_low, vel_high, vel_low, rcs_val])
        track = driver.decode_can_track_frame(can_id=0x400, data=payload)

        assert track is not None
        assert track.track_id == 7
        assert abs(track.range_m - 25.00) < 0.02
        assert abs(track.azimuth_deg - 10.00) < 0.02
        assert abs(track.radial_velocity_mps - (-5.00)) < 0.05

        # Verify Cartesian projection: X = R * sin(az), Z = R * cos(az)
        expected_x = 25.00 * math.sin(math.radians(10.0))
        expected_z = 25.00 * math.cos(math.radians(10.0))
        assert abs(track.x_cartesian_m - expected_x) < 0.05
        assert abs(track.z_cartesian_m - expected_z) < 0.05

    def test_starkenn_radar_blockage_and_timeout(self) -> None:
        """Verifies track aging clears stale obstacles after 250 ms timeout."""
        driver = StarkennRadarDriver()
        payload = bytes([1, 0x03, 0xE8, 0x23, 0x28, 0x13, 0x88, 100])
        driver.decode_can_track_frame(can_id=0x400, data=payload)

        assert len(driver.get_active_tracks()) == 1

        # Simulate time jump > 250 ms
        driver.tracks[1].last_updated_timestamp -= 0.30
        active_after_timeout = driver.get_active_tracks()
        assert len(active_after_timeout) == 0, "Stale track was not cleared after 250 ms timeout"

    def test_mti680_driver_xbus_parsing(self) -> None:
        """Verifies MTi-680 binary XBUS protocol packet framing and checksum verification."""
        driver = MTi680Driver()

        # Construct a synthetic MTData2 packet:
        # Preamble: 0xFA, BusId: 0xFF, MsgId: 0x36 (MTData2), Length: 4, Payload: [0x10, 0x20, 0x01, 0x02]
        # Checksum: Two's complement sum
        preamble = 0xFA
        bus_id = 0xFF
        msg_id = 0x36
        length = 4
        payload = bytes([0x10, 0x20, 0x01, 0x02])
        chksum = (-(bus_id + msg_id + length + sum(payload))) & 0xFF

        packet_bytes = bytes([preamble, bus_id, msg_id, length]) + payload + bytes([chksum])

        messages = driver.feed_serial_bytes(packet_bytes)
        assert len(messages) == 1
        assert messages[0].message_id == 0x36
        assert messages[0].valid_checksum

    def test_mti680_ipm_bridge_deadband(self) -> None:
        """Verifies MTi-680 bridge triggers IPM update only when pitch delta > 0.5 degrees."""
        bridge = MTi680PerceptionBridge()

        # Initial baseline pitch 5.0 deg
        bridge.update_attitude(pitch_deg=5.0, roll_deg=0.0, yaw_deg=0.0)

        # Small jitter (0.2 deg) -> NO trigger
        trig_small = bridge.check_ipm_recompute_needed(current_pitch_deg=5.2)
        assert not trig_small

        # Large pitch dive under heavy braking (6.0 deg) -> TRIGGER
        trig_large = bridge.check_ipm_recompute_needed(current_pitch_deg=6.0)
        assert trig_large


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Hardware Hot-Plug Discovery & Notifier Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestHALDiscoveryAndNotifier:
    """Verifies background hardware discovery daemon and Streamlit event push notifier."""

    def test_hal_discovery_registry(self) -> None:
        """Verifies hardware registry registers and audits hardware inventory."""
        daemon = HALDiscoveryDaemon()
        daemon.register_device(
            device_id="cam_front_0",
            device_type=HardwareDeviceType.CAMERA,
            interface_name="/dev/video0",
            status=HardwareDeviceStatus.ONLINE,
        )

        inventory = daemon.audit_hardware_inventory()
        assert len(inventory.devices) == 1
        assert inventory.devices[0].device_id == "cam_front_0"
        assert inventory.devices[0].status == HardwareDeviceStatus.ONLINE

    def test_hal_notifier_event_push(self, tmp_path: Path) -> None:
        """Verifies JSONL event audit logger and event queue push."""
        jsonl_path = tmp_path / "test_hal_events.jsonl"
        notifier = HALEventNotifier(audit_jsonl_path=jsonl_path)

        notifier.push_event(
            event_type=HALEventType.DEVICE_CONNECTED,
            device_id="starkenn_radar_can0",
            message="Starkenn Orion 77GHz Radar connected on can0",
            severity=HALSeverity.INFO,
        )

        assert notifier.get_pending_event_count() == 1
        events = notifier.consume_events()
        assert len(events) == 1
        assert events[0].device_id == "starkenn_radar_can0"

        # Check JSONL persistence
        assert jsonl_path.exists()
        content = jsonl_path.read_text()
        assert "starkenn_radar_can0" in content


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Perception & Tracking Metrics Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════════════

class TestEvaluationMetrics:
    """Verifies IoU, mAP, AbsRel, HOTA, and environmental stress reporting modules."""

    def test_lane_metrics_iou_and_confidence(self) -> None:
        """Verifies confusion matrix accumulation, per-class IoU, and temperature scaling."""
        acc = ConfusionMatrixAccumulator(num_classes=4)

        # Ground truth: 100 pixels of class 1, 50 of class 2
        y_true = np.zeros(200, dtype=np.int64)
        y_true[0:100] = 1
        y_true[100:150] = 2

        # Prediction: 90 true positives for class 1, 10 false positives
        y_pred = np.zeros(200, dtype=np.int64)
        y_pred[0:90] = 1
        y_pred[90:100] = 0 # False negative
        y_pred[100:140] = 2 # 40 TP
        y_pred[140:150] = 3 # 10 misclassified as 3

        acc.accumulate(y_true, y_pred)
        ious = acc.compute_per_class_iou()

        assert 1 in ious
        assert 0.80 <= ious[1] <= 0.95
        assert 0.60 <= ious[2] <= 0.85

    def test_detection_metrics_map(self) -> None:
        """Verifies 2D bounding box mAP computation across distance strata."""
        evaluator = DetectionMetricsEvaluator(num_classes=5)

        # Ground truth box at 8.0 meters (Near field)
        gt_box = BoundingBox2D(x1=50, y1=50, x2=150, y2=150, class_id=0, distance_m=8.0)
        # Predicted box with high overlap (IoU > 0.8)
        pred_box = BoundingBox2D(x1=52, y1=52, x2=148, y2=148, class_id=0, score=0.92, distance_m=8.1)

        evaluator.add_frame_predictions(ground_truths=[gt_box], predictions=[pred_box])
        map50 = evaluator.compute_map(iou_threshold=0.50)

        assert map50 >= 0.90, "mAP@0.50 should be >= 0.90 for high-overlap prediction"

    def test_depth_metrics_abs_rel(self) -> None:
        """Verifies monocular depth AbsRel and Delta_1 evaluation formulas."""
        evaluator = DepthMetricsEvaluator()

        d_gt = np.array([5.0, 10.0, 20.0, 30.0], dtype=np.float32)
        d_pred = np.array([5.2, 9.5, 21.0, 28.0], dtype=np.float32)

        metrics = evaluator.compute_depth_metrics(ground_truth=d_gt, predicted=d_pred)

        assert "abs_rel" in metrics
        assert "delta_1" in metrics
        assert metrics["abs_rel"] < 0.10, "AbsRel error should be < 0.10 for close predictions"
        assert metrics["delta_1"] >= 0.99

    def test_tracking_metrics_hota_clear_mot(self) -> None:
        """Verifies Clear MOT and HOTA tracking metrics."""
        mot_eval = ClearMOTEvaluator()
        t1_gt = TrackedBox(track_id=1, x1=10, y1=10, x2=50, y2=50, frame_id=1)
        t1_pr = TrackedBox(track_id=101, x1=11, y1=11, x2=49, y2=49, frame_id=1)

        mot_eval.update(frame_id=1, gt_boxes=[t1_gt], pred_boxes=[t1_pr])
        summary = mot_eval.compute_mot_metrics()

        assert summary["id_switches"] == 0
        assert summary["mota"] >= 0.90

    def test_per_condition_benchmark_report(self) -> None:
        """Verifies multi-condition stress test compilation and ASIL-B boundary gating."""
        evaluator = EnvironmentalStressEvaluator()

        # 1. Day clear (Baseline)
        evaluator.record_condition_metrics(
            condition=EnvironmentalCondition.DAY_CLEAR,
            sample_count=500,
            lane_miou=0.78,
            lane_boundary_f1=0.82,
            detection_map_50=0.80,
            detection_map_50_95=0.58,
            depth_abs_rel=0.09,
            depth_delta_1=0.88,
            tracking_hota=0.72,
            tracking_idsw_rate_per_min=1.0,
            mean_latency_ms=19.5,
        )

        # 2. Dust storm (Lethal degradation)
        evaluator.record_condition_metrics(
            condition=EnvironmentalCondition.AIRBORNE_DUST_SMOKE,
            sample_count=200,
            lane_miou=0.51, # Below 0.55 ASIL-B threshold
            lane_boundary_f1=0.53,
            detection_map_50=0.48, # Below 0.50 ASIL-B threshold
            detection_map_50_95=0.32,
            depth_abs_rel=0.24, # Above 0.22 limit
            depth_delta_1=0.70,
            tracking_hota=0.45,
            tracking_idsw_rate_per_min=8.0,
            mean_latency_ms=23.0,
        )

        report = evaluator.generate_master_report()
        assert report.total_samples == 700
        assert report.most_severe_degradation_condition == "AIRBORNE_DUST_SMOKE"
        assert len(report.critical_odd_restrictions) >= 1
        assert not report.condition_matrix["AIRBORNE_DUST_SMOKE"].asil_b_safety_pass


# ═══════════════════════════════════════════════════════════════════════════════
# Main Test Execution Summary
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
