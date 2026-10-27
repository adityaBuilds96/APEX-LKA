"""
src/perception/depth_engine_v2.py
=================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
Pitch/Roll-Compensated Monocular Depth & 3D Spatial Localization Engine v2.0.
ISO 26262 ASIL-B Camera Depth Integrity & Uncertainty Propagation Standard.

Mathematical Ground-Plane Projection (MANDATORY SPECIFICATION):
--------------------------------------------------------------
Given:
  - Camera Resolution : 640x360
  - Intrinsics        : f_x = 450.0 px, f_y = 450.0 px, c_x = 320.0 px, c_y = 180.0 px
  - Camera Height     : H_cam = 1.20 m
  - Live Chassis Pitch: theta_pitch from MTi-680 INS
  - Live Chassis Roll : phi_roll from MTi-680 INS

1. Roll De-Rotation (Image Plane Coordinate Alignment):
   u_rel = u - c_x
   v_rel = v - c_y
   u_derot =  cos(phi) * u_rel + sin(phi) * v_rel + c_x
   v_derot = -sin(phi) * u_rel + cos(phi) * v_rel + c_y

2. Longitudinal Distance (Z_m):
   elevation_angle = theta_pitch + atan2(v_derot - c_y, f_y)
   Z_m = H_cam / tan(elevation_angle)

3. Lateral Coordinate (X_m):
   X_m = (u_derot - c_x) * Z_m / f_x

4. Closed-Form Uncertainty & Pessimistic Safety Margins:
   sigma_Z computed via analytical error propagation of (sigma_v, sigma_theta, sigma_H)
   Z_pessimistic = max(0.1, Z_m - 1.96 * sigma_Z)
   Used for Time-To-Impact (TTI) and Responsibility-Sensitive Safety (RSS) margins.

5. Dynamic Latency Compensation:
   theta_compensated = theta_raw + theta_rate * tau_latency
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.perception.depth_uncertainty import (
    ClosedFormDepthUncertaintyEngine,
    DepthUncertaintyEstimate,
    SensorUncertaintyParameters,
    UncertaintyEllipse2D,
)
from src.perception.extrinsic_drift_monitor import (
    ExtrinsicDriftConfig,
    ExtrinsicDriftMonitor,
    ExtrinsicHealthReport,
    MountCalibrationHealth,
)
from src.perception.ground_plane_validator import (
    GroundPlaneFaultBits,
    GroundPlaneLimitsConfig,
    GroundPlaneValidator,
    GroundPlaneValidityResult,
)

logger = logging.getLogger("APEX.DepthEngineV2")

EPSILON: float = 1e-6
MIN_SAFE_DISTANCE_M: float = 0.10
MAX_PERCEPTION_RANGE_M: float = 160.0


# ──────────────────────────────────────────────────────────────────────────────
# 1. Coordinate Dataclasses & Calibration Specs
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class CameraIntrinsics:
    """Standardized pinhole camera intrinsic parameters."""
    width: int = 640
    height: int = 360
    fx: float = 450.0
    fy: float = 450.0
    cx: float = 320.0
    cy: float = 180.0
    height_m: float = 1.20
    pitch_offset_rad: float = 0.0
    roll_offset_rad: float = 0.0
    yaw_offset_rad: float = 0.0

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        """Construct 3x3 camera intrinsic matrix K."""
        return np.array([
            [self.fx,  0.0,     self.cx],
            [0.0,     self.fy,  self.cy],
            [0.0,     0.0,     1.0],
        ], dtype=np.float64)

    @property
    def inverse_intrinsic_matrix(self) -> np.ndarray:
        """Compute K^-1 analytically for performance."""
        return np.array([
            [1.0 / self.fx, 0.0,           -self.cx / self.fx],
            [0.0,           1.0 / self.fy, -self.cy / self.fy],
            [0.0,           0.0,           1.0],
        ], dtype=np.float64)


@dataclass
class Point3DMetric:
    """3D coordinates in the ISO vehicle coordinate frame (X: Right, Y: Up, Z: Forward)."""
    x_lateral_m: float
    y_vertical_m: float
    z_longitudinal_m: float
    z_pessimistic_m: float
    sigma_z_m: float
    confidence: float
    is_valid: bool
    sigma_x_m: float = 0.0
    elevation_angle_rad: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.array([self.x_lateral_m, self.y_vertical_m, self.z_longitudinal_m], dtype=np.float64)


@dataclass
class BoundingBox3DExtrusion:
    """
    Oriented 3D bounding box corners and geometric properties.
    """
    center_3d: Point3DMetric
    length_m: float = 4.50
    width_m: float = 1.80
    height_m: float = 1.50
    heading_rad: float = 0.0
    corners_3d: List[Tuple[float, float, float]] = field(default_factory=list)
    projected_corners_2d: List[Tuple[float, float]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["center_3d"] = self.center_3d.to_dict()
        return d


@dataclass
class Object3DLocalization:
    """Complete 3D spatial bounding box localization result."""
    track_id: int
    class_id: int
    class_name: str
    bbox_2d: Tuple[float, float, float, float] # (x1, y1, x2, y2)
    position_3d: Point3DMetric
    time_to_impact_sec: Optional[float] = None
    is_in_ego_path: bool = False
    lateral_warning: bool = False
    extruded_box_3d: Optional[BoundingBox3DExtrusion] = None
    uncertainty_ellipse: Optional[UncertaintyEllipse2D] = None
    timestamp_utc: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["position_3d"] = self.position_3d.to_dict()
        if self.extruded_box_3d:
            d["extruded_box_3d"] = self.extruded_box_3d.to_dict()
        if self.uncertainty_ellipse:
            d["uncertainty_ellipse"] = self.uncertainty_ellipse.to_dict()
        return d


# ──────────────────────────────────────────────────────────────────────────────
# 2. Inverse Perspective Mapping (IPM) & Homography Transformer
# ──────────────────────────────────────────────────────────────────────────────

class PlanarHomographyTransformer:
    """
    Synthesizes and caches metric ground-to-image and image-to-ground homographies
    incorporating live chassis pitch and roll angles from the MTi-680 INS.
    """

    def __init__(self, cam: CameraIntrinsics) -> None:
        self.cam = cam
        self.K = cam.intrinsic_matrix
        self.K_inv = cam.inverse_intrinsic_matrix
        self._cached_pitch: float = -999.0
        self._cached_roll: float = -999.0
        self._H_ground_to_image: np.ndarray = np.eye(3, dtype=np.float64)
        self._H_image_to_ground: np.ndarray = np.eye(3, dtype=np.float64)

    def compute_rotation_matrix(self, pitch_rad: float, roll_rad: float) -> np.ndarray:
        """
        Compute vehicle-to-camera rotation matrix (roll around Z_cam, pitch around X_cam).
        """
        cp = math.cos(pitch_rad)
        sp = math.sin(pitch_rad)
        cr = math.cos(roll_rad)
        sr = math.sin(roll_rad)

        # R_pitch (rotation around X)
        R_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0,  cp, -sp],
            [0.0,  sp,  cp],
        ], dtype=np.float64)

        # R_roll (rotation around Z optical axis)
        R_z = np.array([
            [ cr, -sr, 0.0],
            [ sr,  cr, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        return R_z @ R_x

    def update_homographies(self, pitch_rad: float, roll_rad: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute H_img2ground and H_ground2img given dynamic vehicle attitude.
        """
        if abs(pitch_rad - self._cached_pitch) < 1e-5 and abs(roll_rad - self._cached_roll) < 1e-5:
            return self._H_image_to_ground, self._H_ground_to_image

        R = self.compute_rotation_matrix(pitch_rad, roll_rad)
        # Columns of rotation for planar projection (X ground and Z ground, with Y ground = 0)
        r1 = R[:, 0]
        r3 = R[:, 2]
        t = np.array([0.0, -self.cam.height_m, 0.0], dtype=np.float64)

        H_g2i = self.K @ np.column_stack((r1, r3, t))
        if abs(H_g2i[2, 2]) > 1e-9:
            H_g2i /= H_g2i[2, 2]

        try:
            H_i2g = np.linalg.inv(H_g2i)
            if abs(H_i2g[2, 2]) > 1e-9:
                H_i2g /= H_i2g[2, 2]
        except np.linalg.LinAlgError:
            H_i2g = np.eye(3, dtype=np.float64)

        self._H_ground_to_image = H_g2i
        self._H_image_to_ground = H_i2g
        self._cached_pitch = pitch_rad
        self._cached_roll = roll_rad

        return H_i2g, H_g2i


# ──────────────────────────────────────────────────────────────────────────────
# 3. Master Depth Engine V2
# ──────────────────────────────────────────────────────────────────────────────

class MonocularDepthEngineV2:
    """
    ISO 26262 ASIL-B Compliant Monocular 3D Depth Engine.
    Natively integrates closed-form uncertainty propagation, ground-plane validity
    supervision, and online extrinsic drift auto-compensation.
    """

    def __init__(
        self,
        camera_intrinsics: Optional[CameraIntrinsics] = None,
        uncertainty_params: Optional[SensorUncertaintyParameters] = None,
        enable_latency_compensation: bool = True,
        sensor_latency_sec: float = 0.0125, # 12.5 ms
    ) -> None:
        self.cam = camera_intrinsics or CameraIntrinsics()
        self.enable_latency_compensation = enable_latency_compensation
        self.sensor_latency_sec = sensor_latency_sec

        # Uncertainty propagation engine
        uncert_params = uncertainty_params or SensorUncertaintyParameters(
            focal_length_x_px=self.cam.fx,
            focal_length_y_px=self.cam.fy,
            principal_point_x_px=self.cam.cx,
            principal_point_y_px=self.cam.cy,
            camera_height_m=self.cam.height_m,
        )
        self.uncertainty_engine = ClosedFormDepthUncertaintyEngine(params=uncert_params)

        # Ground plane attitude validator
        self.plane_validator = GroundPlaneValidator(
            limits=GroundPlaneLimitsConfig(
                focal_length_y_px=self.cam.fy,
                principal_point_y_px=self.cam.cy,
                image_height_px=self.cam.height,
            )
        )

        # Online extrinsic drift monitor
        self.drift_monitor = ExtrinsicDriftMonitor(
            config=ExtrinsicDriftConfig(
                focal_length_y_px=self.cam.fy,
                principal_point_y_px=self.cam.cy,
                camera_height_m=self.cam.height_m,
            )
        )

        # Homography transformer
        self.homography_transformer = PlanarHomographyTransformer(cam=self.cam)

        # State cache
        self.current_pitch_rad: float = 0.0
        self.current_roll_rad: float = 0.0
        self.current_pitch_rate_rad_s: float = 0.0
        self.last_validity_report: Optional[GroundPlaneValidityResult] = None
        self.last_drift_report: Optional[ExtrinsicHealthReport] = None
        self._total_projections: int = 0

    # ──────────────────────────────────────────────────────────────────────────
    # Attitude State Telemetry Updates & Latency Compensation
    # ──────────────────────────────────────────────────────────────────────────

    def update_attitude(
        self,
        pitch_rad: float,
        roll_rad: float,
        timestamp_sec: Optional[float] = None,
        pitch_rate_rad_s: Optional[float] = None,
        observed_lane_width_m: Optional[float] = None,
        vanishing_row_px: Optional[float] = None,
    ) -> GroundPlaneValidityResult:
        """
        Feed live MTi-680 IMU attitude telemetry into depth engine and supervisory gates.
        Applies transport latency forward-projection if configured.
        """
        # Latency forward projection
        if self.enable_latency_compensation and pitch_rate_rad_s is not None:
            effective_pitch = pitch_rad + pitch_rate_rad_s * self.sensor_latency_sec
            self.current_pitch_rate_rad_s = pitch_rate_rad_s
        else:
            effective_pitch = pitch_rad

        # Incorporate online calibration correction offset
        corrected_pitch = effective_pitch + self.drift_monitor.smoothed_pitch_bias_rad
        self.current_pitch_rad = corrected_pitch
        self.current_roll_rad = roll_rad

        # 1. Validate planar assumption
        validity = self.plane_validator.validate_attitude(
            pitch_rad=corrected_pitch,
            roll_rad=roll_rad,
            timestamp_sec=timestamp_sec,
            live_pitch_rate_deg_s=math.degrees(pitch_rate_rad_s) if pitch_rate_rad_s is not None else None,
        )
        self.last_validity_report = validity

        # 2. Update extrinsic drift monitor
        drift_report = self.drift_monitor.update_frame(
            observed_lane_width_m=observed_lane_width_m,
            measured_vanishing_row_px=vanishing_row_px,
            ins_pitch_rad=pitch_rad,
        )
        self.last_drift_report = drift_report

        # 3. Update planar homographies
        self.homography_transformer.update_homographies(
            pitch_rad=self.current_pitch_rad,
            roll_rad=self.current_roll_rad,
        )

        return validity

    # ──────────────────────────────────────────────────────────────────────────
    # 2D-to-3D Geometric Ground-Plane Projection
    # ──────────────────────────────────────────────────────────────────────────

    def derotate_pixel(self, u: float, v: float, roll_rad: float) -> Tuple[float, float]:
        """
        Apply 2D coordinate de-rotation around camera optical center to cancel chassis roll:
          [u'] = [ cos(phi)  sin(phi)] * [u - c_x] + [c_x]
          [v']   [-sin(phi)  cos(phi)]   [v - c_y]   [c_y]
        """
        if abs(roll_rad) < 1e-5:
            return (u, v)

        cos_r = math.cos(roll_rad)
        sin_r = math.sin(roll_rad)

        u_rel = u - self.cam.cx
        v_rel = v - self.cam.cy

        u_derot = cos_r * u_rel + sin_r * v_rel + self.cam.cx
        v_derot = -sin_r * u_rel + cos_r * v_rel + self.cam.cy

        return (u_derot, v_derot)

    def project_pixel_to_ground(
        self,
        u_pixel: float,
        v_pixel: float,
        pitch_rad: Optional[float] = None,
        roll_rad: Optional[float] = None,
    ) -> Point3DMetric:
        """
        Calculate true metric 3D ground coordinates (X_m, Y_m, Z_m) for a camera pixel.
        Applies live roll de-rotation, pitch compensation, and closed-form variance propagation.
        """
        self._total_projections += 1
        p_rad = pitch_rad if pitch_rad is not None else self.current_pitch_rad
        r_rad = roll_rad if roll_rad is not None else self.current_roll_rad

        # De-rotate for chassis roll
        u_derot, v_derot = self.derotate_pixel(u_pixel, v_pixel, r_rad)

        # Propagate closed-form depth uncertainty
        uncert = self.uncertainty_engine.propagate_uncertainty(
            v_bottom_px=v_derot,
            pitch_rad=p_rad,
            u_center_px=u_derot,
            camera_height_override=self.cam.height_m,
        )

        if not uncert.is_valid_ground_plane:
            return Point3DMetric(
                x_lateral_m=0.0,
                y_vertical_m=0.0,
                z_longitudinal_m=MAX_PERCEPTION_RANGE_M,
                z_pessimistic_m=MAX_PERCEPTION_RANGE_M,
                sigma_z_m=999.0,
                confidence=0.0,
                is_valid=False,
                sigma_x_m=999.0,
                elevation_angle_rad=0.0,
            )

        z_m = uncert.nominal_distance_m
        z_pessimistic = uncert.pessimistic_distance_m

        # Lateral metric coordinate: X_m = (u_derot - c_x) * Z_m / f_x
        x_m = (u_derot - self.cam.cx) * z_m / self.cam.fx
        y_m = 0.0 # By flat-ground assumption, points on the ground have vertical offset 0.0

        confidence = 1.0 - min(1.0, uncert.relative_uncertainty_pct / 100.0)
        elevation = p_rad + math.atan2(v_derot - self.cam.cy, self.cam.fy)

        return Point3DMetric(
            x_lateral_m=round(x_m, 3),
            y_vertical_m=round(y_m, 3),
            z_longitudinal_m=round(z_m, 3),
            z_pessimistic_m=round(z_pessimistic, 3),
            sigma_z_m=uncert.sigma_z_m,
            confidence=round(confidence, 3),
            is_valid=True,
            sigma_x_m=uncert.sigma_x_m,
            elevation_angle_rad=round(elevation, 5),
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Object 3D Localization, Extrusion & Safety Margins
    # ──────────────────────────────────────────────────────────────────────────

    def extrude_3d_bounding_box(
        self,
        base_center: Point3DMetric,
        length_m: float = 4.50,
        width_m: float = 1.80,
        height_m: float = 1.50,
        heading_rad: float = 0.0,
    ) -> BoundingBox3DExtrusion:
        """
        Extrude an 8-corner 3D bounding box from base center contact position.
        """
        dx = width_m / 2.0
        dz = length_m / 2.0
        dy = height_m

        # Relative corner coordinates in vehicle frame
        rel_corners = [
            (-dx, 0.0, -dz),
            ( dx, 0.0, -dz),
            ( dx, 0.0,  dz),
            (-dx, 0.0,  dz),
            (-dx,  dy, -dz),
            ( dx,  dy, -dz),
            ( dx,  dy,  dz),
            (-dx,  dy,  dz),
        ]

        cos_h = math.cos(heading_rad)
        sin_h = math.sin(heading_rad)

        corners_3d: List[Tuple[float, float, float]] = []
        corners_2d: List[Tuple[float, float]] = []

        for cx, cy, cz in rel_corners:
            rx = cos_h * cx + sin_h * cz + base_center.x_lateral_m
            rz = -sin_h * cx + cos_h * cz + base_center.z_longitudinal_m
            ry = cy

            corners_3d.append((round(rx, 3), round(ry, 3), round(rz, 3)))

            # Re-project to 2D
            if rz > 0.1:
                u_proj = self.cam.cx + (rx * self.cam.fx) / rz
                v_proj = self.cam.cy - (ry * self.cam.fy) / rz
                corners_2d.append((round(u_proj, 1), round(v_proj, 1)))

        return BoundingBox3DExtrusion(
            center_3d=base_center,
            length_m=length_m,
            width_m=width_m,
            height_m=height_m,
            heading_rad=heading_rad,
            corners_3d=corners_3d,
            projected_corners_2d=corners_2d,
        )

    def localize_bounding_box(
        self,
        bbox_xyxy: Tuple[float, float, float, float],
        track_id: int = -1,
        class_id: int = 0,
        class_name: str = "vehicle",
        closing_velocity_m_s: Optional[float] = None,
        ego_lane_width_m: float = 3.70,
        extrude_3d: bool = False,
    ) -> Object3DLocalization:
        """
        Estimate 3D position of an obstacle from its 2D bounding box contact point (x_mid, y_bottom).
        Computes spatial uncertainty ellipse, pessimistic TTI, and ego-path intersection.
        """
        x1, y1, x2, y2 = bbox_xyxy
        u_center = (x1 + x2) / 2.0
        v_bottom = y2

        pos_3d = self.project_pixel_to_ground(u_center, v_bottom)

        tti = None
        if closing_velocity_m_s and closing_velocity_m_s > 0.5:
            tti = round(pos_3d.z_pessimistic_m / closing_velocity_m_s, 2)

        # Check if in ego vehicle travel path (within half lane width)
        half_lane = ego_lane_width_m / 2.0
        in_ego_path = abs(pos_3d.x_lateral_m) <= (half_lane + 0.30)
        lateral_warning = abs(pos_3d.x_lateral_m) <= (half_lane + 0.90)

        # Extrude 3D box if requested
        extrusion = None
        if extrude_3d and pos_3d.is_valid:
            extrusion = self.extrude_3d_bounding_box(pos_3d)

        # Compute uncertainty ellipse
        ellipse = None
        if pos_3d.is_valid:
            uncert_est = self.uncertainty_engine.propagate_uncertainty(
                v_bottom_px=v_bottom,
                pitch_rad=self.current_pitch_rad,
                u_center_px=u_center,
            )
            ellipse = self.uncertainty_engine.compute_spatial_uncertainty_ellipse(uncert_est)

        return Object3DLocalization(
            track_id=track_id,
            class_id=class_id,
            class_name=class_name,
            bbox_2d=bbox_xyxy,
            position_3d=pos_3d,
            time_to_impact_sec=tti,
            is_in_ego_path=in_ego_path,
            lateral_warning=lateral_warning,
            extruded_box_3d=extrusion,
            uncertainty_ellipse=ellipse,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Bird's-Eye-View (BEV) Polynomial Projection & Metric Warping
    # ──────────────────────────────────────────────────────────────────────────

    def project_polyline_to_bev(
        self,
        pixel_points: Sequence[Tuple[float, float]],
    ) -> List[Tuple[float, float]]:
        """
        Convert pixel coordinates into metric ground-plane coordinates (Z_forward, X_lateral).
        """
        bev_points: List[Tuple[float, float]] = []
        for u, v in pixel_points:
            pt3d = self.project_pixel_to_ground(u, v)
            if pt3d.is_valid:
                # Store as (Z_forward, X_lateral)
                bev_points.append((pt3d.z_longitudinal_m, pt3d.x_lateral_m))
        return bev_points

    def warp_image_to_bev(
        self,
        image_bgr: np.ndarray,
        output_resolution_m_per_pixel: float = 0.05,
        forward_range_m: float = 50.0,
        lateral_range_m: float = 10.0,
    ) -> np.ndarray:
        """
        Synthesize a top-down metric Bird's-Eye-View (BEV) orthographic raster image.
        """
        bev_height = int(forward_range_m / output_resolution_m_per_pixel)
        bev_width = int((2.0 * lateral_range_m) / output_resolution_m_per_pixel)

        # Build metric pixel grid to ground coordinate map
        H_i2g, _ = self.homography_transformer.update_homographies(
            self.current_pitch_rad,
            self.current_roll_rad,
        )

        # Scale and translation matrix from metric to BEV image pixels
        M_metric_to_bev = np.array([
            [1.0 / output_resolution_m_per_pixel, 0.0, lateral_range_m / output_resolution_m_per_pixel],
            [0.0, -1.0 / output_resolution_m_per_pixel, forward_range_m / output_resolution_m_per_pixel],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

        H_cam_to_bev = M_metric_to_bev @ H_i2g

        try:
            H_bev_to_cam = np.linalg.inv(H_cam_to_bev)
            warped = cv2.warpPerspective(
                image_bgr,
                H_bev_to_cam,
                (bev_width, bev_height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
            )
            return warped
        except Exception as err:
            logger.warning(f"Error executing BEV warp: {err}")
            return np.zeros((bev_height, bev_width, 3), dtype=np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Lens Distortion Rectification & Sagittal Road Profile Reconstruction
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LensDistortionCoefficients:
    """
    Standard Brown-Conrady 5-parameter radial and tangential lens distortion model:
      (k1, k2, p1, p2, k3)
    """
    k1: float = -0.050
    k2: float = 0.010
    p1: float = 0.000
    p2: float = 0.000
    k3: float = 0.000

    def to_numpy(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float64)


class LensDistortionRectifier:
    """
    Applies inverse radial distortion mapping to raw optical coordinates
    prior to metric flat-ground projection.
    """

    def __init__(self, cam: CameraIntrinsics, dist: Optional[LensDistortionCoefficients] = None) -> None:
        self.cam = cam
        self.dist = dist or LensDistortionCoefficients()
        self.dist_coeffs = self.dist.to_numpy()

    def undistort_pixel(self, u_dist: float, v_dist: float) -> Tuple[float, float]:
        """
        Iteratively compute undistorted pixel coordinate using Brown-Conrady model.
        """
        # Normalized camera coordinates
        x = (u_dist - self.cam.cx) / self.cam.fx
        y = (v_dist - self.cam.cy) / self.cam.fy

        # Gauss-Newton / fixed-point inverse iteration
        x_undist, y_undist = x, y
        for _ in range(5):
            r2 = x_undist ** 2 + y_undist ** 2
            radial = 1.0 + self.dist.k1 * r2 + self.dist.k2 * (r2 ** 2) + self.dist.k3 * (r2 ** 3)
            delta_x = 2.0 * self.dist.p1 * x_undist * y_undist + self.dist.p2 * (r2 + 2.0 * (x_undist ** 2))
            delta_y = self.dist.p1 * (r2 + 2.0 * (y_undist ** 2)) + 2.0 * self.dist.p2 * x_undist * y_undist

            x_undist = (x - delta_x) / max(1e-6, radial)
            y_undist = (y - delta_y) / max(1e-6, radial)

        u_undist = x_undist * self.cam.fx + self.cam.cx
        v_undist = y_undist * self.cam.fy + self.cam.cy

        return (u_undist, v_undist)


class SagittalRoadProfileEstimator:
    """
    Estimates non-planar road elevation profiles y(z) = a * z^2 + b * z
    for crests, dips, and vertical road curvature using multi-point lane boundary depth.
    """

    def __init__(self, min_points_for_fit: int = 5) -> None:
        self.min_points = min_points_for_fit
        self.vertical_curvature_kappa: float = 0.0
        self.grade_slope: float = 0.0

    def fit_vertical_profile(
        self,
        longitudinal_z_array: Sequence[float],
        vertical_elevation_y_array: Sequence[float],
    ) -> Tuple[float, float, float]:
        """
        Fit 2nd-degree vertical polynomial y(z) = a * z^2 + b * z + c:
        Returns:
            (vertical_curvature_a, grade_slope_b, intercept_c)
        """
        if len(longitudinal_z_array) < self.min_points:
            return 0.0, 0.0, 0.0

        z = np.array(longitudinal_z_array, dtype=np.float64)
        y = np.array(vertical_elevation_y_array, dtype=np.float64)

        # 2nd-degree polynomial regression
        try:
            poly_coeffs = np.polyfit(z, y, deg=2)
            self.vertical_curvature_kappa = float(poly_coeffs[0])
            self.grade_slope = float(poly_coeffs[1])
            return float(poly_coeffs[0]), float(poly_coeffs[1]), float(poly_coeffs[2])
        except (np.linalg.LinAlgError, ValueError):
            return 0.0, 0.0, 0.0

    def get_corrected_height_at_distance(self, nominal_z_m: float, base_height_m: float) -> float:
        """
        Compute effective camera-to-road vertical clearance incorporating vertical curvature:
          H_eff(z) = H_cam - (a * z^2 + b * z)
        """
        delta_y = (
            self.vertical_curvature_kappa * (nominal_z_m ** 2)
            + self.grade_slope * nominal_z_m
        )
        return max(0.20, base_height_m - delta_y)


# ──────────────────────────────────────────────────────────────────────────────
# 5. Diagnostic Telemetry & Performance Profiler
# ──────────────────────────────────────────────────────────────────────────────

class DepthEngineDiagnostics:
    """
    Real-time ASIL-B telemetry and performance metrics for monocular depth estimation.
    """

    def __init__(self) -> None:
        self.total_projections: int = 0
        self.out_of_bounds_count: int = 0
        self.pessimistic_interventions: int = 0
        self.processing_latency_ms_history: List[float] = []

    def record_projection(self, is_valid: bool, was_pessimistic_used: bool, latency_ms: float) -> None:
        self.total_projections += 1
        if not is_valid:
            self.out_of_bounds_count += 1
        if was_pessimistic_used:
            self.pessimistic_interventions += 1
        self.processing_latency_ms_history.append(latency_ms)
        if len(self.processing_latency_ms_history) > 500:
            self.processing_latency_ms_history.pop(0)

    def summary(self) -> Dict[str, Any]:
        avg_lat = (
            float(np.mean(self.processing_latency_ms_history))
            if self.processing_latency_ms_history else 0.0
        )
        return {
            "total_projections": self.total_projections,
            "out_of_bounds_count": self.out_of_bounds_count,
            "pessimistic_interventions": self.pessimistic_interventions,
            "mean_projection_latency_ms": round(avg_lat, 4),
        }


# ──────────────────────────────────────────────────────────────────────────────
# 6. Redundant Stereo Baseline Triangulator (ASIL-B Diversity Channel)
# ──────────────────────────────────────────────────────────────────────────────

class StereoBaselineTriangulator:
    """
    Independent stereoscopic distance solver used as a secondary diversity check
    against monocular flat-ground projection failures:
      Z_stereo = (f_x * Baseline_m) / disparity_px
    """

    def __init__(self, focal_length_x_px: float = 450.0, baseline_m: float = 0.22) -> None:
        self.fx = focal_length_x_px
        self.baseline_m = baseline_m

    def triangulate(self, disparity_px: float, sigma_disparity_px: float = 0.50) -> Tuple[float, float]:
        """
        Compute metric depth and analytical standard deviation from stereo disparity:
          sigma_Z = (Z^2 / (f * B)) * sigma_d
        """
        if disparity_px <= 0.2:
            return MAX_PERCEPTION_RANGE_M, 999.0

        z = (self.fx * self.baseline_m) / disparity_px
        sigma_z = ((z ** 2) / (self.fx * self.baseline_m)) * sigma_disparity_px

        return round(z, 3), round(sigma_z, 3)

    def cross_validate_monocular(
        self,
        z_mono_m: float,
        disparity_px: float,
        tolerance_pct: float = 25.0,
    ) -> Tuple[bool, float, float]:
        """
        Cross-check monocular flat-ground depth against stereo triangulation.
        Returns:
            (is_consistent, z_stereo_m, divergence_pct)
        """
        z_stereo, _ = self.triangulate(disparity_px)
        divergence = abs(z_mono_m - z_stereo) / max(0.1, z_stereo) * 100.0
        is_consistent = divergence <= tolerance_pct
        return is_consistent, z_stereo, round(divergence, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

def run_depth_engine_self_test() -> bool:
    """Execute complete ASIL-B self-test for MonocularDepthEngineV2."""
    print("Executing DepthEngineV2 ISO 26262 ASIL-B self-check...")
    depth_engine = MonocularDepthEngineV2()

    # Update attitude with small pitch (1.0 degree)
    validity = depth_engine.update_attitude(pitch_rad=math.radians(1.0), roll_rad=0.0)
    assert validity.is_valid, "Level attitude validation failed!"

    # Project point at bottom center (u=320, v=240)
    pt3d = depth_engine.project_pixel_to_ground(u_pixel=320.0, v_pixel=240.0)
    print(f"3D Position: X={pt3d.x_lateral_m:.2f}m, Z={pt3d.z_longitudinal_m:.2f}m (pessimistic={pt3d.z_pessimistic_m:.2f}m)")
    assert pt3d.is_valid
    assert abs(pt3d.x_lateral_m) < 0.05, "Center pixel should have ~0 lateral offset"
    assert pt3d.z_longitudinal_m > 0.0

    # Test Bounding Box Localization with 3D extrusion
    loc = depth_engine.localize_bounding_box(
        bbox_xyxy=(280.0, 190.0, 360.0, 240.0),
        closing_velocity_m_s=12.0,
        extrude_3d=True,
    )
    assert loc.position_3d.is_valid
    assert loc.is_in_ego_path
    assert loc.extruded_box_3d is not None
    assert len(loc.extruded_box_3d.corners_3d) == 8
    print(f"Obstacle localization passed. TTI: {loc.time_to_impact_sec}s | Extruded Corners: 8")

    # Test BEV polyline projection
    pixels = [(320.0, 220.0), (320.0, 250.0), (320.0, 280.0)]
    bev_pts = depth_engine.project_polyline_to_bev(pixels)
    assert len(bev_pts) == 3
    assert bev_pts[0][0] > bev_pts[1][0] > bev_pts[2][0], "Closer pixel rows should yield smaller Z distances!"
    print(f"BEV Polyline Projection passed ({len(bev_pts)} points).")

    print("DepthEngineV2 verification passed successfully.")
    return True


if __name__ == "__main__":
    run_depth_engine_self_test()
