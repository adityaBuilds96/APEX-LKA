"""
src/perception/ipm_transformer.py
=================================
APEX-Vision Core v4.0 • Dynamic Inverse Perspective Mapping (IPM) Engine
ISO 26262 ASIL-B Aligned Geometric Perspective Transformation

Transforms front-facing monocular camera perspective frames into a metric
top-down Bird's Eye View (BEV) ground plane representation.

Mathematical & Calibration Specifications:
------------------------------------------
1. Metric Resolution: Strictly calibrated to 0.0185 meters/pixel (1.85 cm/px,
   or ~54.05 pixels/meter).
2. Camera Intrinsic Defaults (640x360 RGB Camera):
   - Focal lengths: f_x = 450.0 px, f_y = 450.0 px
   - Principal point (optical center): c_x = 320.0 px, c_y = 180.0 px
3. Camera Extrinsic Defaults (BAJA SAE Buggy Profile):
   - Mounting height: H_cam = 1.20 meters
   - Nominal pitch: theta_pitch = 5.0 degrees (downward tilt)
   - Nominal roll: phi_roll = 0.0 degrees
   - Nominal yaw: psi_yaw = 0.0 degrees
4. Dynamic AHRS Angle Adaptation:
   - Ingests real-time 400 Hz pitch and roll angles from the MTi-680 RTK GNSS/INS.
   - Enforces dynamic homography recomputation when |delta_pitch| > 0.5 deg or
     |delta_roll| > 0.5 deg to maintain metric fidelity over rough terrain.
5. Bidirectional Coordinate Mapping:
   - Image Space (u_img, v_img) <-> BEV Pixel Space (u_bev, v_bev) <-> Vehicle Ground Plane (X_m, Y_m)
"""

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG

logger = logging.getLogger("APEX.IPMTransformer")


@dataclass
class CameraIntrinsics:
    """Pinhole camera intrinsic parameters."""
    fx: float = 450.0
    fy: float = 450.0
    cx: float = 320.0
    cy: float = 180.0
    image_width: int = 640
    image_height: int = 360
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0

    @property
    def K_matrix(self) -> np.ndarray:
        """Return 3x3 camera calibration matrix K."""
        return np.array([
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)


@dataclass
class CameraExtrinsics:
    """Camera mounting pose relative to vehicle front axle reference frame."""
    mounting_height_m: float = 1.20  # H_cam in meters
    nominal_pitch_deg: float = 5.0   # Downward pitch
    nominal_roll_deg: float = 0.0
    nominal_yaw_deg: float = 0.0
    longitudinal_offset_m: float = 0.20  # Forward of front axle


@dataclass
class IPMConfig:
    """Configuration for Inverse Perspective Mapping metric grid."""
    # Metric Scaling: strictly 0.0185 meters/pixel
    meters_per_pixel: float = 0.0185

    # Target BEV Output Grid Dimensions
    bev_width: int = 400   # 400 px * 0.0185 m/px = 7.40 meters lateral span
    bev_height: int = 600  # 600 px * 0.0185 m/px = 11.10 meters longitudinal span

    # Ground Plane Metric Boundaries relative to camera ground projection
    x_lateral_min_m: float = -3.70   # Left boundary (meters)
    x_lateral_max_m: float = +3.70   # Right boundary (meters)
    y_longitudinal_min_m: float = 1.00   # Near-field ground cutoff (meters)
    y_longitudinal_max_m: float = 12.10  # Far-field ground limit (meters)

    # Recomputation Deadband Threshold
    angle_recompute_threshold_deg: float = 0.50

    # Interpolation Flags
    warp_interpolation: int = cv2.INTER_LINEAR
    warp_border_mode: int = cv2.BORDER_CONSTANT


@dataclass
class IPMTransformResult:
    """Result telemetry from IPM warp pass."""
    bev_image: np.ndarray
    homography_matrix: np.ndarray
    inv_homography_matrix: np.ndarray
    pitch_angle_deg: float
    roll_angle_deg: float
    recomputed: bool
    warp_latency_ms: float
    meters_per_pixel: float = 0.0185


class IPMTransformer:
    """
    Constructs and applies dynamic Inverse Perspective Mapping (IPM) homographies
    to project camera perspective views into metric top-down coordinates.
    """

    def __init__(
        self,
        intrinsics: Optional[CameraIntrinsics] = None,
        extrinsics: Optional[CameraExtrinsics] = None,
        config: Optional[IPMConfig] = None,
    ):
        self.intrinsics = intrinsics or CameraIntrinsics()
        self.extrinsics = extrinsics or CameraExtrinsics()
        self.config = config or IPMConfig()

        # Active state angles
        self.current_pitch_deg = self.extrinsics.nominal_pitch_deg
        self.current_roll_deg = self.extrinsics.nominal_roll_deg

        # Cached Homography Matrices
        self._H_ipm: Optional[np.ndarray] = None
        self._H_inv: Optional[np.ndarray] = None
        self._last_recomputed_pitch = self.current_pitch_deg
        self._last_recomputed_roll = self.current_roll_deg

        # Compute initial homography
        self.recompute_homography(self.current_pitch_deg, self.current_roll_deg)

    def recompute_homography(self, pitch_deg: float, roll_deg: float) -> np.ndarray:
        """
        Recompute 3x3 IPM Homography Matrix based on current pitch and roll angles.
        Derives source perspective trapezoid and target rectangular BEV metric rectangle.
        """
        self.current_pitch_deg = pitch_deg
        self.current_roll_deg = roll_deg

        # Compute rotation matrix R = R_yaw * R_pitch * R_roll
        pitch_rad = math.radians(pitch_deg)
        roll_rad = math.radians(roll_deg)

        # Ground plane limits in meters
        x_min = self.config.x_lateral_min_m
        x_max = self.config.x_lateral_max_m
        y_min = self.config.y_longitudinal_min_m
        y_max = self.config.y_longitudinal_max_m
        h_cam = self.extrinsics.mounting_height_m

        # 4 Ground Plane Metric Points (in camera frame coordinates: X right, Y down, Z forward)
        # Note: In standard camera frame, ground is at Y = +h_cam
        # P = [X_ground, h_cam, Z_ground]^T
        pts_ground = [
            np.array([x_min, h_cam, y_min]),  # Bottom-Left
            np.array([x_max, h_cam, y_min]),  # Bottom-Right
            np.array([x_max, h_cam, y_max]),  # Top-Right
            np.array([x_min, h_cam, y_max]),  # Top-Left
        ]

        # Rotation angles: rotate around X by pitch, rotate around Z by roll
        # R_pitch = [[1, 0, 0], [0, cos(p), -sin(p)], [0, sin(p), cos(p)]]
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)

        R_pitch = np.array([
            [1.0, 0.0, 0.0],
            [0.0, cos_p, -sin_p],
            [0.0, sin_p, cos_p]
        ], dtype=np.float64)

        R_roll = np.array([
            [cos_r, -sin_r, 0.0],
            [sin_r, cos_r, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)

        R_total = R_roll @ R_pitch
        K = self.intrinsics.K_matrix

        # Project 4 ground points into Image Pixel Coordinates (u, v)
        src_pts = []
        for P_g in pts_ground:
            # Rotate into camera optical frame
            P_cam = R_total @ P_g
            if P_cam[2] <= 0.1:
                P_cam[2] = 0.1
            # Project via K
            p_img = K @ P_cam
            u = p_img[0] / p_img[2]
            v = p_img[1] / p_img[2]
            src_pts.append([u, v])

        src_arr = np.array(src_pts, dtype=np.float32)

        # Target BEV Pixel Coordinates (u_bev, v_bev)
        # Origin (0,0) at top-left of BEV, v_bev points forward down
        w_bev = self.config.bev_width
        h_bev = self.config.bev_height

        dst_arr = np.array([
            [0.0, h_bev - 1.0],       # Bottom-Left (near)
            [w_bev - 1.0, h_bev - 1.0], # Bottom-Right (near)
            [w_bev - 1.0, 0.0],       # Top-Right (far)
            [0.0, 0.0],               # Top-Left (far)
        ], dtype=np.float32)

        # Calculate perspective transformation matrix H_ipm
        H = cv2.getPerspectiveTransform(src_arr, dst_arr)
        self._H_ipm = H
        self._H_inv = np.linalg.inv(H)
        self._last_recomputed_pitch = pitch_deg
        self._last_recomputed_roll = roll_deg

        logger.debug(f"IPM Homography recomputed for pitch={pitch_deg:.2f} deg, roll={roll_deg:.2f} deg")
        return H

    def update_ahrs_angles(self, pitch_deg: float, roll_deg: float) -> bool:
        """
        Ingest current MTi-680 AHRS pitch and roll angles.
        Recomputes homography if delta exceeds angle_recompute_threshold_deg.
        """
        delta_p = abs(pitch_deg - self._last_recomputed_pitch)
        delta_r = abs(roll_deg - self._last_recomputed_roll)

        if (delta_p > self.config.angle_recompute_threshold_deg or
            delta_r > self.config.angle_recompute_threshold_deg):
            self.recompute_homography(pitch_deg, roll_deg)
            return True
        return False

    def transform_image_to_bev(
        self,
        image: np.ndarray,
        pitch_deg: Optional[float] = None,
        roll_deg: Optional[float] = None,
    ) -> IPMTransformResult:
        """
        Warp an input perspective camera frame into top-down metric BEV space.
        """
        t_start = time.perf_counter()
        recomputed = False

        if pitch_deg is not None and roll_deg is not None:
            recomputed = self.update_ahrs_angles(pitch_deg, roll_deg)

        if self._H_ipm is None:
            self.recompute_homography(self.current_pitch_deg, self.current_roll_deg)
            recomputed = True

        bev_size = (self.config.bev_width, self.config.bev_height)
        bev_image = cv2.warpPerspective(
            image,
            self._H_ipm,
            bev_size,
            flags=self.config.warp_interpolation,
            borderMode=self.config.warp_border_mode,
        )

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        return IPMTransformResult(
            bev_image=bev_image,
            homography_matrix=self._H_ipm.copy(),
            inv_homography_matrix=self._H_inv.copy(),
            pitch_angle_deg=self.current_pitch_deg,
            roll_angle_deg=self.current_roll_deg,
            recomputed=recomputed,
            warp_latency_ms=elapsed_ms,
            meters_per_pixel=self.config.meters_per_pixel,
        )

    def warp_bev_to_perspective(self, bev_image: np.ndarray) -> np.ndarray:
        """
        Unwarp a BEV image or mask back into original perspective camera view.
        """
        if self._H_inv is None:
            self.recompute_homography(self.current_pitch_deg, self.current_roll_deg)

        img_size = (self.intrinsics.image_width, self.intrinsics.image_height)
        return cv2.warpPerspective(
            bev_image,
            self._H_inv,
            img_size,
            flags=self.config.warp_interpolation,
            borderMode=self.config.warp_border_mode,
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # Coordinate Transformations (Image <-> BEV <-> Ground Plane Meters)
    # ═══════════════════════════════════════════════════════════════════════════

    def image_to_bev_pixel(self, u_img: float, v_img: float) -> Tuple[float, float]:
        """Convert camera image coordinates (u, v) into BEV pixel coordinates."""
        if self._H_ipm is None:
            self.recompute_homography(self.current_pitch_deg, self.current_roll_deg)
        pt = np.array([u_img, v_img, 1.0], dtype=np.float64)
        p_bev = self._H_ipm @ pt
        w = p_bev[2] if abs(p_bev[2]) > 1e-6 else 1e-6
        return float(p_bev[0] / w), float(p_bev[1] / w)

    def bev_pixel_to_image(self, u_bev: float, v_bev: float) -> Tuple[float, float]:
        """Convert BEV pixel coordinates (u, v) into camera image coordinates."""
        if self._H_inv is None:
            self.recompute_homography(self.current_pitch_deg, self.current_roll_deg)
        pt = np.array([u_bev, v_bev, 1.0], dtype=np.float64)
        p_img = self._H_inv @ pt
        w = p_img[2] if abs(p_img[2]) > 1e-6 else 1e-6
        return float(p_img[0] / w), float(p_img[1] / w)

    def bev_pixel_to_ground_metric(self, u_bev: float, v_bev: float) -> Tuple[float, float]:
        """
        Convert BEV pixel coordinates into vehicle ground plane metric coordinates:
        - X_ground: lateral displacement (meters, positive = vehicle right)
        - Y_ground: longitudinal displacement (meters, positive = forward)
        """
        # Lateral: center of BEV corresponds to X = 0.0 meters
        u_center = self.config.bev_width / 2.0
        x_m = (u_bev - u_center) * self.config.meters_per_pixel

        # Longitudinal: bottom of BEV corresponds to y_longitudinal_min_m
        # v_bev = h_bev - 1 corresponds to near-field
        v_bottom = self.config.bev_height - 1.0
        dist_from_bottom_px = v_bottom - v_bev
        y_m = self.config.y_longitudinal_min_m + (dist_from_bottom_px * self.config.meters_per_pixel)

        return float(x_m), float(y_m)

    def ground_metric_to_bev_pixel(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """
        Convert vehicle ground plane metric coordinates (X_m, Y_m) into BEV pixel coordinates.
        """
        u_center = self.config.bev_width / 2.0
        u_bev = u_center + (x_m / self.config.meters_per_pixel)

        v_bottom = self.config.bev_height - 1.0
        dist_from_bottom_px = (y_m - self.config.y_longitudinal_min_m) / self.config.meters_per_pixel
        v_bev = v_bottom - dist_from_bottom_px

        return float(u_bev), float(v_bev)

    def image_to_ground_metric(self, u_img: float, v_img: float) -> Tuple[float, float]:
        """Direct transformation from camera image pixel to vehicle ground plane meters."""
        u_bev, v_bev = self.image_to_bev_pixel(u_img, v_img)
        return self.bev_pixel_to_ground_metric(u_bev, v_bev)

    def ground_metric_to_image(self, x_m: float, y_m: float) -> Tuple[float, float]:
        """Direct transformation from vehicle ground plane meters to camera image pixel."""
        u_bev, v_bev = self.ground_metric_to_bev_pixel(x_m, y_m)
        return self.bev_pixel_to_image(u_bev, v_bev)

    # ═══════════════════════════════════════════════════════════════════════════
    # Diagnostic Overlays & Grid Rendering
    # ═══════════════════════════════════════════════════════════════════════════

    def draw_metric_grid_overlay(
        self,
        bev_image: np.ndarray,
        grid_interval_m: float = 1.0,
        major_interval_m: float = 5.0,
    ) -> np.ndarray:
        """
        Draw a metric distance grid (1m minor lines, 5m major lines) on top of BEV image.
        """
        annotated = bev_image.copy()
        h, w = annotated.shape[:2]

        # 1. Longitudinal grid lines (every grid_interval_m)
        y_curr = self.config.y_longitudinal_min_m
        while y_curr <= self.config.y_longitudinal_max_m:
            _, v_px = self.ground_metric_to_bev_pixel(0.0, y_curr)
            v_int = int(round(v_px))
            if 0 <= v_int < h:
                is_major = (abs(y_curr % major_interval_m) < 0.1) or (abs(y_curr) < 0.1)
                color = (0, 255, 255) if is_major else (80, 80, 80)
                thickness = 2 if is_major else 1
                cv2.line(annotated, (0, v_int), (w - 1, v_int), color, thickness)
                # Label distance
                cv2.putText(annotated, f"{y_curr:.0f}m", (10, v_int - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_curr += grid_interval_m

        # 2. Lateral grid lines (every 1.0 meter from center)
        x_curr = -3.0
        while x_curr <= 3.0:
            u_px, _ = self.ground_metric_to_bev_pixel(x_curr, 5.0)
            u_int = int(round(u_px))
            if 0 <= u_int < w:
                is_center = abs(x_curr) < 0.05
                color = (0, 0, 255) if is_center else (80, 80, 80)
                thickness = 2 if is_center else 1
                cv2.line(annotated, (u_int, 0), (u_int, h - 1), color, thickness)
                if abs(x_curr) > 0.05:
                    cv2.putText(annotated, f"{x_curr:+.0f}m", (u_int + 2, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            x_curr += grid_interval_m

        return annotated


    def unwarp_bev_polynomial_to_image(
        self,
        poly_coeffs: np.ndarray,
        y_eval_range_m: Tuple[float, float] = (1.5, 12.0),
        num_points: int = 50,
    ) -> np.ndarray:
        """
        Unwarps a 2nd-degree polynomial curve x = A*y^2 + B*y + C evaluated in BEV
        ground plane metric coordinates back into an array of (u, v) image coordinates.
        Ideal for HUD rendering and projection verification.
        """
        y_vals_m = np.linspace(y_eval_range_m[0], y_eval_range_m[1], num_points)
        A, B, C = poly_coeffs[0], poly_coeffs[1], poly_coeffs[2]
        x_vals_m = A * (y_vals_m ** 2) + B * y_vals_m + C

        image_points = []
        for x_m, y_m in zip(x_vals_m, y_vals_m):
            u_img, v_img = self.ground_metric_to_image(x_m, y_m)
            # Clip within camera boundary with safety margin
            if -50 <= u_img <= self.intrinsics.image_width + 50 and -50 <= v_img <= self.intrinsics.image_height + 50:
                image_points.append([u_img, v_img])

        return np.array(image_points, dtype=np.int32)

    def generate_drivable_corridor_bev_polygon(
        self,
        left_poly_coeffs: np.ndarray,
        right_poly_coeffs: np.ndarray,
        y_range_m: Tuple[float, float] = (1.5, 11.0),
        num_points: int = 40,
    ) -> np.ndarray:
        """
        Generates a closed polygon representing the drivable lane corridor in BEV pixel space
        formed between left and right boundary polynomials.
        """
        y_vals_m = np.linspace(y_range_m[0], y_range_m[1], num_points)

        # Left boundary pixels
        left_pts = []
        for y_m in y_vals_m:
            x_m = left_poly_coeffs[0] * (y_m ** 2) + left_poly_coeffs[1] * y_m + left_poly_coeffs[2]
            u_px, v_px = self.ground_metric_to_bev_pixel(x_m, y_m)
            left_pts.append([u_px, v_px])

        # Right boundary pixels (reversed order to close polygon)
        right_pts = []
        for y_m in reversed(y_vals_m):
            x_m = right_poly_coeffs[0] * (y_m ** 2) + right_poly_coeffs[1] * y_m + right_poly_coeffs[2]
            u_px, v_px = self.ground_metric_to_bev_pixel(x_m, y_m)
            right_pts.append([u_px, v_px])

        corridor_polygon = np.array(left_pts + right_pts, dtype=np.int32)
        return corridor_polygon


# ═══════════════════════════════════════════════════════════════════════════
# Factory Function & Self-Verification
# ═══════════════════════════════════════════════════════════════════════════

def get_ipm_transformer(
    intrinsics: Optional[CameraIntrinsics] = None,
    extrinsics: Optional[CameraExtrinsics] = None,
    config: Optional[IPMConfig] = None,
) -> IPMTransformer:
    """Factory constructor for IPMTransformer."""
    return IPMTransformer(intrinsics=intrinsics, extrinsics=extrinsics, config=config)


def run_unit_tests() -> bool:
    """Validate IPM coordinate scaling, polynomial unwarping, and round-trip accuracy."""
    print("Testing IPMTransformer coordinate mathematics...")
    transformer = get_ipm_transformer()

    # Invariant 1: Center line pixel converts to 0.0m lateral offset
    u_center = transformer.config.bev_width / 2.0
    x_m, y_m = transformer.bev_pixel_to_ground_metric(u_center, 300.0)
    assert abs(x_m) < 1e-4, f"Center BEV u coordinate must map to 0.0m, got {x_m}"

    # Invariant 2: Metric resolution is strictly 0.0185 m/px
    x_m1, _ = transformer.bev_pixel_to_ground_metric(u_center + 100.0, 300.0)
    expected_delta = 100.0 * 0.0185
    assert abs(x_m1 - expected_delta) < 1e-4, f"Scaling mismatch: expected {expected_delta}, got {x_m1}"

    # Invariant 3: Round-trip Image -> BEV -> Image mapping consistency
    u_orig, v_orig = 320.0, 250.0
    u_bev, v_bev = transformer.image_to_bev_pixel(u_orig, v_orig)
    u_back, v_back = transformer.bev_pixel_to_image(u_bev, v_bev)
    assert abs(u_orig - u_back) < 0.5 and abs(v_orig - v_back) < 0.5, "Round-trip projection drift exceeded 0.5px"

    # Invariant 4: Dynamic AHRS recomputation triggering
    recomputed = transformer.update_ahrs_angles(pitch_deg=5.8, roll_deg=0.0)
    assert recomputed is True, "Expected homography recompute for delta pitch > 0.5 deg"

    # Invariant 5: Small pitch drift does NOT trigger recomputation (hysteresis)
    recomputed_small = transformer.update_ahrs_angles(pitch_deg=5.9, roll_deg=0.0)
    assert recomputed_small is False, "Small delta <= 0.5 deg should not trigger recompute"

    # Invariant 6: Polynomial unwarping returns valid pixel coordinates
    synthetic_poly = np.array([0.001, 0.0, -1.75])  # Left lane at -1.75m
    img_pts = transformer.unwarp_bev_polynomial_to_image(synthetic_poly)
    assert len(img_pts) > 0, "Failed to unwarp polynomial to image plane"

    # Invariant 7: Corridor polygon generation
    right_poly = np.array([0.001, 0.0, 1.75])
    poly = transformer.generate_drivable_corridor_bev_polygon(synthetic_poly, right_poly)
    assert len(poly) > 10, "Failed to generate closed drivable corridor polygon"

    print("IPMTransformer self-verification PASSED (100% geometric invariants satisfied).")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_unit_tests()

