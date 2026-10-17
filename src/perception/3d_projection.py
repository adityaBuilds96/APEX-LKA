"""
src/perception/3d_projection.py
===============================
3D Bounding Box Projection and Spatial Localization Engine for APEX-LKA:
- Fuses 2D object detection bounding boxes with dense monocular metric depth maps.
- Extracts median depth Z_med from central 50% ROI of each 2D box (filtering background clutter).
- Back-projects 2D image coordinates to 3D metric vehicle coordinates:
    X_m = (u_c - c_x) * Z_med / f_x
    Y_m = (v_c - c_y) * Z_med / f_y
    Z_m = Z_med
- Estimates physical object dimensions (W_m, H_m, L_m) using optical angular span and class priors.
- Guards against zero-division, NaN, Inf, and out-of-frame boundaries.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from src.perception.object_detector import Detection, DetectionResult, OBJECT_CLASSES

log = logging.getLogger("apex_3d_projection")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 3D Physical Dimension Priors: (Width, Height, Length) in meters
CLASS_3D_PRIORS: Dict[str, Tuple[float, float, float]] = {
    "vehicle":       (1.80, 1.50, 4.20),
    "pedestrian":    (0.60, 1.70, 0.60),
    "cyclist":       (0.70, 1.60, 1.70),
    "traffic_cone":  (0.35, 0.70, 0.35),
    "barrier":       (1.50, 1.00, 0.50),
    "animal":        (0.60, 0.80, 1.00),
    "debris":        (0.40, 0.30, 0.40),
    "traffic_sign":  (0.60, 0.80, 0.10),
    "traffic_light": (0.35, 0.90, 0.30),
}
DEFAULT_PRIOR = (1.00, 1.00, 1.00)


@dataclass
class Object3D:
    """3D Bounding Box representation in vehicle / camera frame."""
    x: float               # Lateral offset (meters, + = right, - = left)
    y: float               # Vertical offset (meters, + = down, - = up)
    z: float               # Longitudinal depth (meters forward)
    width: float           # Extent across X (meters)
    height: float          # Extent across Y (meters)
    length: float          # Extent across Z (meters)
    class_id: int
    class_name: str
    confidence: float
    box_2d: np.ndarray     # Original 2D bounding box [x1, y1, x2, y2]

    @property
    def distance_m(self) -> float:
        """Euclidean distance from camera origin."""
        return float(math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z))

    @property
    def lateral_distance_m(self) -> float:
        """Absolute lateral offset from vehicle centerline."""
        return abs(self.x)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "x": round(self.x, 3),
            "y": round(self.y, 3),
            "z": round(self.z, 3),
            "width": round(self.width, 3),
            "height": round(self.height, 3),
            "length": round(self.length, 3),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": round(self.confidence, 3),
            "distance_m": round(self.distance_m, 3),
            "box_2d": [round(float(v), 1) for v in self.box_2d],
        }


@dataclass
class ProjectionResult:
    """Collection of projected 3D objects for a single frame."""
    objects_3d: List[Object3D] = field(default_factory=list)
    nearest_object: Optional[Object3D] = None
    projection_ms: float = 0.0

    def __len__(self) -> int:
        return len(self.objects_3d)

    @property
    def is_empty(self) -> bool:
        return len(self.objects_3d) == 0


class BoundingBox3DProjector:
    """
    Projects 2D detections into 3D metric spatial coordinates using monocular depth.
    """

    def __init__(
        self,
        fx: float = 500.0,
        fy: float = 500.0,
        cx: float = 320.0,
        cy: float = 180.0,
        mount_height_m: float = 1.20,
        pitch_rad: float = 0.08726646,
        roi_central_fraction: float = 0.50,
        min_depth_m: float = 0.2,
        max_depth_m: float = 80.0,
    ) -> None:
        self.fx = max(10.0, float(fx))
        self.fy = max(10.0, float(fy))
        self.cx = float(cx)
        self.cy = float(cy)
        self.mount_height_m = float(mount_height_m)
        self.pitch_rad = float(pitch_rad)
        self.roi_fraction = float(np.clip(roi_central_fraction, 0.1, 1.0))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

    def update_intrinsics(self, fx: float, fy: float, cx: float, cy: float) -> None:
        """Update intrinsic calibration parameters."""
        self.fx = max(10.0, float(fx))
        self.fy = max(10.0, float(fy))
        self.cx = float(cx)
        self.cy = float(cy)

    def project_2d_to_3d(
        self,
        detections: Union[DetectionResult, List[Detection]],
        metric_depth_map: np.ndarray,
    ) -> ProjectionResult:
        """
        Projects each 2D bounding box to 3D vehicle space.

        Parameters:
            detections: DetectionResult or list of Detection items.
            metric_depth_map: 2D Float32 array containing depth in meters (H, W).

        Returns:
            ProjectionResult containing list of Object3D.
        """
        t0 = time.perf_counter()

        if metric_depth_map is None or metric_depth_map.size == 0:
            return ProjectionResult()

        h_img, w_img = metric_depth_map.shape[:2]

        # Extract list of detections
        if isinstance(detections, DetectionResult):
            dets = detections.get_detections()
        elif isinstance(detections, list):
            dets = detections
        else:
            dets = []

        if not dets:
            return ProjectionResult(projection_ms=round((time.perf_counter() - t0) * 1000.0, 2))

        projected: List[Object3D] = []

        for det in dets:
            box = det.box
            if len(box) < 4:
                continue

            x1, y1, x2, y2 = [float(v) for v in box[:4]]
            w_box = max(1.0, x2 - x1)
            h_box = max(1.0, y2 - y1)
            uc = (x1 + x2) / 2.0
            vc = (y1 + y2) / 2.0

            # 1. Extract central ROI (default 50% central window)
            margin_x = (1.0 - self.roi_fraction) / 2.0
            margin_y = (1.0 - self.roi_fraction) / 2.0

            rx1 = int(round(x1 + margin_x * w_box))
            rx2 = int(round(x2 - margin_x * w_box))
            ry1 = int(round(y1 + margin_y * h_box))
            ry2 = int(round(y2 - margin_y * h_box))

            # Guard against invalid ROI bounds
            rx1 = max(0, min(w_img - 1, rx1))
            rx2 = max(rx1 + 1, min(w_img, rx2))
            ry1 = max(0, min(h_img - 1, ry1))
            ry2 = max(ry1 + 1, min(h_img, ry2))

            roi_depth = metric_depth_map[ry1:ry2, rx1:rx2]

            # 2. Compute median depth Z_med (robust against background clutter)
            valid_mask = np.isfinite(roi_depth) & (roi_depth > 0.0)
            if np.any(valid_mask):
                z_med = float(np.median(roi_depth[valid_mask]))
            else:
                # Fallback to bottom-edge ground projection
                y_horizon = self.cy - self.fy * math.tan(self.pitch_rad)
                dy = max(5.0, y2 - y_horizon)
                z_med = (self.fy * self.mount_height_m) / dy

            z_med = float(np.clip(z_med, self.min_depth_m, self.max_depth_m))

            # 3. Back-project box center (u_c, v_c) to 3D vehicle space:
            #    X_m = (u_c - c_x) * Z_med / f_x
            #    Y_m = (v_c - c_y) * Z_med / f_y
            #    Z_m = Z_med
            x_m = float((uc - self.cx) * z_med / self.fx)
            y_m = float((vc - self.cy) * z_med / self.fy)
            z_m = z_med

            # 4. Estimate physical dimensions (W_m, H_m, L_m)
            w_optical = max(0.1, float(w_box * z_med / self.fx))
            h_optical = max(0.1, float(h_box * z_med / self.fy))

            prior_w, prior_h, prior_l = CLASS_3D_PRIORS.get(det.class_name, DEFAULT_PRIOR)

            # Fuse optical angular extent with class priors (60% optical, 40% prior, clamped)
            w_fused = float(np.clip(0.6 * w_optical + 0.4 * prior_w, 0.4 * prior_w, 2.5 * prior_w))
            h_fused = float(np.clip(0.6 * h_optical + 0.4 * prior_h, 0.4 * prior_h, 2.5 * prior_h))
            l_fused = float(prior_l)

            obj_3d = Object3D(
                x=x_m,
                y=y_m,
                z=z_m,
                width=w_fused,
                height=h_fused,
                length=l_fused,
                class_id=int(det.class_id),
                class_name=str(det.class_name),
                confidence=float(det.score),
                box_2d=np.array([x1, y1, x2, y2], dtype=np.float32),
            )
            projected.append(obj_3d)

        # Identify nearest obstacle along forward axis Z
        nearest = None
        if projected:
            nearest = min(projected, key=lambda o: o.z)

        p_ms = (time.perf_counter() - t0) * 1000.0
        return ProjectionResult(
            objects_3d=projected,
            nearest_object=nearest,
            projection_ms=round(p_ms, 2),
        )

    @staticmethod
    def compute_tti_ms(
        obj: Object3D,
        ego_speed_mps: float,
        min_speed_threshold_mps: float = 0.2,
    ) -> int:
        """
        Calculate Collision Time-to-Impact (TTI) in milliseconds:
          TTI = Z_m / v_mps
        Returns:
          TTI in milliseconds [0, 65535]. Returns 65535 if ego is stationary or moving away.
        """
        if ego_speed_mps < min_speed_threshold_mps or obj.z <= 0.0:
            return 65535

        tti_sec = obj.z / ego_speed_mps
        tti_ms = int(round(tti_sec * 1000.0))
        return int(np.clip(tti_ms, 0, 65535))
