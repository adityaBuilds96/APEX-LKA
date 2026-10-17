"""
src/inference/distance_estimator.py
===================================
Monocular Camera Geometry & Real-World Metric Distance Estimator for APEX-LKA.

Maps 2D image pixel coordinates (x_px, y_px) to 3D vehicle ground coordinates (X_m, Z_m)
using camera mounting height (H_cam), focal length (f_y), pitch angle (theta_pitch),
and vanishing horizon geometry.

Formulas:
---------
1. Longitudinal Distance Z (meters):
   Z_meters(y) = (H_cam * f_y) / ((y - y_horizon) * cos(theta_pitch))
2. Lateral Offset X (meters):
   X_meters(x, y) = ((x - c_x) * Z_meters(y)) / f_x
3. Metric Road / Corridor Width:
   Width_meters(y) = ((x_right - x_left) * Z_meters(y)) / f_x
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.config import cfg


@dataclass
class DistanceProfile:
    """Metric distance telemetry across lookahead horizons and boundaries."""
    closest_boundary_distance_m: float = 0.0
    width_at_2m: float = 0.0
    width_at_5m: float = 0.0
    width_at_10m: float = 0.0
    horizon_y_px: int = 126
    lookahead_y_map: Dict[float, int] = field(default_factory=dict)


class MonocularDistanceEstimator:
    """
    Real-time monocular 3D metric coordinate and ground distance estimator.
    Guaranteed zero-division safety, bounds checking, and NaN/Inf guards.
    """

    def __init__(
        self,
        mounting_height_m: Optional[float] = None,
        pitch_angle_deg: Optional[float] = None,
        focal_length_px: Optional[float] = None,
        horizon_y_px: Optional[int] = None,
        image_width: int = 640,
        image_height: int = 360,
    ) -> None:
        sp_cfg = cfg.get("scene_perception", {})
        cam_cfg = sp_cfg.get("camera_calibration", {})

        # Mounting parameters
        self.h_cam = float(
            mounting_height_m
            if mounting_height_m is not None
            else cam_cfg.get("mounting_height_m", 1.25)
        )
        self.pitch_deg = float(
            pitch_angle_deg
            if pitch_angle_deg is not None
            else cam_cfg.get("pitch_angle_deg", 0.0)
        )
        self.pitch_rad = math.radians(self.pitch_deg)
        self.cos_pitch = max(abs(math.cos(self.pitch_rad)), 1e-4)

        # Optical parameters
        self.fy = float(
            focal_length_px
            if focal_length_px is not None
            else cam_cfg.get("focal_length_px", 500.0)
        )
        self.fx = self.fy  # Square pixel assumption unless configured otherwise

        self.w = int(image_width)
        self.h = int(image_height)
        self.cx = self.w / 2.0
        self.cy = float(
            horizon_y_px
            if horizon_y_px is not None
            else cam_cfg.get("horizon_y_px", int(0.35 * self.h))
        )
        self.horizon_y = int(self.cy)

        # Precompute constants
        self.h_fy = self.h_cam * self.fy

    def pixel_to_meters(
        self,
        x_px: Union[float, int, np.ndarray],
        y_px: Union[float, int, np.ndarray],
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert pixel coordinates (x_px, y_px) to ground coordinates (X_m, Z_m)
        relative to the vehicle front bumper center (0, 0).

        Parameters
        ----------
        x_px : float or np.ndarray
            Horizontal pixel coordinate(s).
        y_px : float or np.ndarray
            Vertical pixel coordinate(s).

        Returns
        -------
        Tuple[float or np.ndarray, float or np.ndarray]
            (X_m, Z_m) where:
            - X_m: Lateral distance from vehicle centerline (+ = right, - = left).
            - Z_m: Longitudinal distance forward ahead of vehicle in meters.
        """
        is_array = isinstance(x_px, np.ndarray) or isinstance(y_px, np.ndarray)
        x = np.asarray(x_px, dtype=np.float32)
        y = np.asarray(y_px, dtype=np.float32)

        # Longitudinal distance Z_m:
        # Prevent division by zero and negative distances above the horizon
        delta_y = y - self.horizon_y
        valid_mask = delta_y > 0.5

        safe_delta_y = np.where(valid_mask, delta_y, 0.5)
        z_m = (self.h_fy / (safe_delta_y * self.cos_pitch))

        # Clamp max range to 100 meters for points at or above horizon
        z_m = np.where(valid_mask, np.clip(z_m, 0.1, 100.0), 100.0)

        # Lateral distance X_m:
        # X = ((x - cx) * Z) / fx
        x_m = ((x - self.cx) * z_m) / max(self.fx, 1e-4)

        if not is_array:
            return float(x_m), float(z_m)
        return x_m, z_m

    def meters_to_pixel(
        self,
        x_m: Union[float, int, np.ndarray],
        z_m: Union[float, int, np.ndarray],
    ) -> Tuple[Union[int, np.ndarray], Union[int, np.ndarray]]:
        """
        Inverse projection: Convert ground coordinates (X_m, Z_m) to image pixel (x_px, y_px).

        Parameters
        ----------
        x_m : float or np.ndarray
            Lateral coordinate in meters.
        z_m : float or np.ndarray
            Longitudinal coordinate in meters (> 0.0).

        Returns
        -------
        Tuple[int or np.ndarray, int or np.ndarray]
            (x_px, y_px) pixel coordinates.
        """
        is_array = isinstance(x_m, np.ndarray) or isinstance(z_m, np.ndarray)
        xm = np.asarray(x_m, dtype=np.float32)
        zm = np.maximum(np.asarray(z_m, dtype=np.float32), 0.1)

        # y = horizon_y + (H_cam * fy) / (Z * cos(pitch))
        y_px = self.horizon_y + (self.h_fy / (zm * self.cos_pitch))
        # x = cx + (X * fx) / Z
        x_px = self.cx + (xm * self.fx) / zm

        if not is_array:
            return int(round(float(x_px))), int(round(float(y_px)))
        return np.round(x_px).astype(np.int32), np.round(y_px).astype(np.int32)

    def compute_lookahead_y_rows(
        self,
        lookaheads_m: Tuple[float, ...] = (2.0, 5.0, 10.0),
    ) -> Dict[float, int]:
        """
        Compute image row y corresponding to each lookahead distance Z in meters.
        """
        y_map: Dict[float, int] = {}
        for z in lookaheads_m:
            _, y_px = self.meters_to_pixel(0.0, z)
            # Bound within frame
            clamped_y = int(np.clip(y_px, self.horizon_y + 1, self.h - 1))
            y_map[float(z)] = clamped_y
        return y_map

    def compute_path_widths_at_lookaheads(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        lookaheads_m: Tuple[float, ...] = (2.0, 5.0, 10.0),
    ) -> Dict[float, float]:
        """
        Calculate metric drivable path width at specific lookahead distances.

        Parameters
        ----------
        left_poly : Optional[np.ndarray]
            Left polynomial coefficients [a, b, c] for x = a*y^2 + b*y + c.
        right_poly : Optional[np.ndarray]
            Right polynomial coefficients [a, b, c].
        lookaheads_m : Tuple[float, ...]
            Lookahead distances forward in meters.

        Returns
        -------
        Dict[float, float]
            Mapping from lookahead distance to metric width in meters.
        """
        widths: Dict[float, float] = {}
        y_map = self.compute_lookahead_y_rows(lookaheads_m)

        for z, y in y_map.items():
            if left_poly is not None and right_poly is not None:
                try:
                    xl = float(np.polyval(left_poly, y))
                    xr = float(np.polyval(right_poly, y))
                    px_width = max(0.0, xr - xl)
                    # Convert pixel width to metric width at distance z
                    width_m = (px_width * z) / max(self.fx, 1e-4)
                    widths[z] = round(float(np.clip(width_m, 0.5, 12.0)), 2)
                    continue
                except (ValueError, TypeError):
                    pass

            # Safe geometric default if boundaries not fitted (standard lane ~3.5 m)
            widths[z] = 3.5

        return widths

    def compute_distance_to_closest_boundary(
        self,
        boundary_pts: np.ndarray,
    ) -> float:
        """
        Compute minimum ground distance in meters to detected boundary points or obstacles.

        Parameters
        ----------
        boundary_pts : np.ndarray
            (N, 2) array of [x, y] coordinates.

        Returns
        -------
        float
            Distance in meters to closest boundary intercept ahead of vehicle.
        """
        if boundary_pts is None or len(boundary_pts) == 0:
            return 50.0

        pts = np.asarray(boundary_pts, dtype=np.float32)
        valid = pts[:, 1] > self.horizon_y + 1
        if not np.any(valid):
            return 50.0

        valid_pts = pts[valid]
        xs = valid_pts[:, 0]
        ys = valid_pts[:, 1]

        xm, zm = self.pixel_to_meters(xs, ys)
        # Euclidean ground distance: sqrt(X^2 + Z^2)
        distances = np.sqrt(np.maximum(0.0, xm ** 2 + zm ** 2))
        closest_m = float(np.min(distances)) if len(distances) > 0 else 50.0
        return round(float(np.clip(closest_m, 0.0, 100.0)), 2)

    def compute_distance_profile(
        self,
        left_poly: Optional[np.ndarray],
        right_poly: Optional[np.ndarray],
        boundary_pts: Optional[np.ndarray] = None,
    ) -> DistanceProfile:
        """
        Generate unified metric distance telemetry.
        """
        y_map = self.compute_lookahead_y_rows((2.0, 5.0, 10.0))
        widths = self.compute_path_widths_at_lookaheads(left_poly, right_poly, (2.0, 5.0, 10.0))

        if boundary_pts is not None and len(boundary_pts) > 0:
            closest_dist = self.compute_distance_to_closest_boundary(boundary_pts)
        else:
            closest_dist = 50.0

        return DistanceProfile(
            closest_boundary_distance_m=closest_dist,
            width_at_2m=widths.get(2.0, 3.5),
            width_at_5m=widths.get(5.0, 3.5),
            width_at_10m=widths.get(10.0, 3.5),
            horizon_y_px=self.horizon_y,
            lookahead_y_map=y_map,
        )
