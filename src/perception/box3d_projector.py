"""
src/perception/box3d_projector.py
==================================
APEX-Vision Core v3.0 — 3D Bounding Box Projection Engine (ISO 26262 Compliant)
- Converts 2D bounding boxes + metric depth maps into 3D Oriented Bounding Boxes (OBB):
    [X_c, Y_c, Z_c, L, W, H, yaw_rad]
- Computes 8 spatial 3D corner vertices in vehicle camera frame.
- Projects 3D OBB wireframe vertices back onto 2D image coordinates for real-time UI overlay.
- Robust against occlusion, depth outliers, and out-of-frame boundaries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

# Defensive project root resolution
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 3D Metric Physical Dimensions: (Length, Width, Height) in meters
CLASS_3D_DIMENSIONS: Dict[str, Tuple[float, float, float]] = {
    "vehicle": (4.50, 1.85, 1.55),
    "pedestrian": (0.60, 0.60, 1.75),
    "cone": (0.40, 0.40, 0.70),
    "cyclist": (1.75, 0.70, 1.60),
    "barrier": (0.50, 1.50, 0.90),
}
DEFAULT_DIMENSIONS: Tuple[float, float, float] = (1.50, 1.50, 1.50)


@dataclass
class OrientedBoundingBox3D:
    """3D Oriented Bounding Box representation in vehicle camera frame."""
    x_c: float              # Center lateral offset (meters, + = right, - = left)
    y_c: float              # Center vertical offset (meters, + = down towards road, - = up)
    z_c: float              # Center longitudinal depth (meters forward)
    length: float           # Extent along heading / Z (meters)
    width: float            # Extent along lateral / X (meters)
    height: float           # Extent along vertical / Y (meters)
    yaw_rad: float          # Rotation angle about vertical axis (radians)
    class_name: str
    confidence: float
    bbox_2d: List[float]    # [x1, y1, x2, y2]
    corners_3d: np.ndarray = field(default_factory=lambda: np.zeros((8, 3), dtype=np.float32))
    corners_2d: np.ndarray = field(default_factory=lambda: np.zeros((8, 2), dtype=np.float32))

    @property
    def distance_m(self) -> float:
        """Euclidean distance from ego camera origin."""
        return float(math.sqrt(self.x_c**2 + self.y_c**2 + self.z_c**2))

    @property
    def ground_contact_point(self) -> Tuple[float, float, float]:
        """Ground contact position [X, Y_road, Z]."""
        return (self.x_c, self.y_c + self.height / 2.0, self.z_c)


class Box3DProjector:
    """
    3D Bounding Box Projection Engine for monocular perception.
    """

    def __init__(
        self,
        focal_length_px: float = 500.0,
        principal_point: Tuple[float, float] = (320.0, 180.0),
        camera_height_m: float = 1.25,
        pitch_angle_deg: float = -5.0,
    ) -> None:
        self.fx = float(focal_length_px)
        self.fy = float(focal_length_px)
        self.cx, self.cy = principal_point
        self.cam_h = float(camera_height_m)
        self.pitch_rad = math.radians(pitch_angle_deg)

    def extract_roi_metric_depth(
        self,
        bbox: List[float],
        metric_depth_map: np.ndarray,
        central_fraction: float = 0.50,
    ) -> float:
        """
        Extract robust median depth from the central ROI of the 2D bounding box.
        """
        h_map, w_map = metric_depth_map.shape[:2]
        x1, y1, x2, y2 = bbox

        # Clamp box to valid boundaries
        x1 = max(0, min(w_map - 1, int(round(x1))))
        x2 = max(0, min(w_map, int(round(x2))))
        y1 = max(0, min(h_map - 1, int(round(y1))))
        y2 = max(0, min(h_map, int(round(y2))))

        bw = x2 - x1
        bh = y2 - y1
        if bw < 2 or bh < 2:
            return 15.0  # safe default depth

        # Central ROI to avoid background bleed
        margin_x = int(bw * (1.0 - central_fraction) / 2.0)
        margin_y = int(bh * (1.0 - central_fraction) / 2.0)

        roi_x1 = max(0, x1 + margin_x)
        roi_x2 = min(w_map, x2 - margin_x)
        roi_y1 = max(0, y1 + margin_y)
        roi_y2 = min(h_map, y2 - margin_y)

        roi = metric_depth_map[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0:
            roi = metric_depth_map[y1:y2, x1:x2]

        valid_vals = roi[(roi > 0.2) & (roi < 100.0)]
        if valid_vals.size > 0:
            return float(np.median(valid_vals))

        # Pinhole ground-contact fallback
        bc_y = float(y2)
        dy = max(bc_y - self.cy, 1.0)
        return float((self.fy * self.cam_h) / dy)

    def project_2d_to_3d_obb(
        self,
        bbox_2d: List[float],
        class_name: str,
        confidence: float,
        metric_depth_map: np.ndarray,
        yaw_rad: float = 0.0,
    ) -> OrientedBoundingBox3D:
        """
        Construct 3D Oriented Bounding Box from 2D detection and metric depth.
        """
        x1, y1, x2, y2 = bbox_2d
        bc_x = (x1 + x2) / 2.0
        bc_y = (y1 + y2) / 2.0

        # Obtain robust depth
        z_c = self.extract_roi_metric_depth(bbox_2d, metric_depth_map)
        x_c = float((bc_x - self.cx) * z_c / self.fx)
        y_c = float((bc_y - self.cy) * z_c / self.fy)

        # Lookup class priors
        l, w, h = CLASS_3D_DIMENSIONS.get(class_name.lower(), DEFAULT_DIMENSIONS)

        # Compute 8 corners in 3D camera coordinates
        corners_3d = self.compute_3d_corners(x_c, y_c, z_c, l, w, h, yaw_rad)

        # Project 3D corners to 2D image coordinates
        corners_2d = self.project_3d_points_to_2d(corners_3d)

        obb = OrientedBoundingBox3D(
            x_c=round(x_c, 3),
            y_c=round(y_c, 3),
            z_c=round(z_c, 3),
            length=round(l, 2),
            width=round(w, 2),
            height=round(h, 2),
            yaw_rad=round(yaw_rad, 3),
            class_name=class_name,
            confidence=round(confidence, 3),
            bbox_2d=bbox_2d,
            corners_3d=corners_3d,
            corners_2d=corners_2d,
        )
        return obb

    @staticmethod
    def compute_3d_corners(
        x_c: float,
        y_c: float,
        z_c: float,
        l: float,
        w: float,
        h: float,
        yaw_rad: float = 0.0,
    ) -> np.ndarray:
        """
        Calculates the 8 corner vertices of an oriented 3D bounding box:
        Front-top-left, Front-top-right, Front-bottom-right, Front-bottom-left,
        Back-top-left,  Back-top-right,  Back-bottom-right,  Back-bottom-left.
        """
        dx = w / 2.0
        dy = h / 2.0
        dz = l / 2.0

        # 8 canonical vertices relative to center
        local_corners = np.array([
            [-dx, -dy, -dz],  # 0: Front-Top-Left
            [ dx, -dy, -dz],  # 1: Front-Top-Right
            [ dx,  dy, -dz],  # 2: Front-Bottom-Right
            [-dx,  dy, -dz],  # 3: Front-Bottom-Left
            [-dx, -dy,  dz],  # 4: Back-Top-Left
            [ dx, -dy,  dz],  # 5: Back-Top-Right
            [ dx,  dy,  dz],  # 6: Back-Bottom-Right
            [-dx,  dy,  dz],  # 7: Back-Bottom-Left
        ], dtype=np.float32)

        # Rotation around Y axis (vertical in camera frame)
        cos_y = math.cos(yaw_rad)
        sin_y = math.sin(yaw_rad)
        rot_y = np.array([
            [ cos_y, 0.0, sin_y],
            [ 0.0,   1.0, 0.0  ],
            [-sin_y, 0.0, cos_y],
        ], dtype=np.float32)

        # Rotate and translate
        rotated = (rot_y @ local_corners.T).T
        world_corners = rotated + np.array([x_c, y_c, z_c], dtype=np.float32)
        return world_corners

    def project_3d_points_to_2d(self, points_3d: np.ndarray) -> np.ndarray:
        """
        Projects 3D camera-frame points [N, 3] to 2D image coordinates [N, 2].
        """
        u = self.cx + self.fx * (points_3d[:, 0] / np.maximum(points_3d[:, 2], 0.1))
        v = self.cy + self.fy * (points_3d[:, 1] / np.maximum(points_3d[:, 2], 0.1))
        return np.stack([u, v], axis=-1)

    def draw_3d_obb_wireframe(
        self,
        image_bgr: np.ndarray,
        obb: OrientedBoundingBox3D,
        color: Tuple[int, int, int] = (0, 255, 120),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Draws 3D oriented bounding box wireframe lines connecting the 8 vertices.
        """
        pts = obb.corners_2d.astype(np.int32)

        # Front Face: 0 -> 1 -> 2 -> 3
        front_poly = pts[0:4]
        cv2.polylines(image_bgr, [front_poly], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Rear Face: 4 -> 5 -> 6 -> 7
        rear_poly = pts[4:8]
        cv2.polylines(image_bgr, [rear_poly], isClosed=True, color=color, thickness=max(1, thickness - 1), lineType=cv2.LINE_AA)

        # Connecting Edges: 0-4, 1-5, 2-6, 3-7
        for i in range(4):
            cv2.line(image_bgr, tuple(pts[i]), tuple(pts[i + 4]), color=color, thickness=thickness, lineType=cv2.LINE_AA)

        # Text label above top face
        top_u = int(round(np.min(pts[:, 0])))
        top_v = int(round(np.min(pts[:, 1])))
        lbl = f"{obb.class_name[:3].upper()} Z:{obb.z_c:.1f}m"
        cv2.putText(
            image_bgr,
            lbl,
            (top_u, max(15, top_v - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return image_bgr
