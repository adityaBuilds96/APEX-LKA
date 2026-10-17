"""
APEX-Vision Core v3.0 — Priority Threat Risk Assessor
=====================================================
ISO 26262 ASIL-B Compliant Threat Assessment Engine:
- Projects 2D bounding boxes onto 3D road coordinates using camera extrinsics.
- Computes multi-factor Risk Score (0.0 - 1.0) based on:
  1. Estimated Distance (Ground contact point projection)
  2. Object Class Weight (Pedestrian = 1.0, Vehicle = 0.70, Cone = 0.50)
  3. Lateral Offset / In-Lane Proximity (Gaussian corridor weighting)
- Selects Primary Target (highest risk) for CAN ID 0x150 broadcast.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from src.models.perception_core import Detection2D


@dataclass
class ThreatTarget:
    """Target object evaluated with 3D spatial pose and collision risk score."""
    target_id: int
    class_name: str
    class_enum: int  # 1: Pedestrian, 2: Vehicle, 3: Cone, 0: Unknown
    bbox: List[float]  # [x1, y1, x2, y2]
    distance_m: float
    distance_mm: int
    lateral_offset_m: float
    is_in_lane: bool
    risk_score: float  # [0.0, 1.0]
    stop_flag: bool    # True if emergency braking required


@dataclass
class RiskEngineReport:
    """Consolidated assessment report containing all threats and primary target."""
    targets: List[ThreatTarget] = field(default_factory=list)
    primary_target: Optional[ThreatTarget] = None
    max_risk_score: float = 0.0
    emergency_stop_required: bool = False
    processing_time_ms: float = 0.0


class PriorityRiskAssessor:
    """
    Priority Risk Assessment Engine for real-time ADAS threat prioritization.
    """

    # Class Risk Weight Matrix
    CLASS_WEIGHTS: Dict[str, float] = {
        "pedestrian": 1.0,  # Critical vulnerable road user
        "vehicle":    0.70, # Mass threat, protected occupant
        "cone":       0.50, # Channelization boundary / stationary obstacle
    }

    # CAN Bus Class Enums
    CLASS_ENUMS: Dict[str, int] = {
        "pedestrian": 1,
        "vehicle":    2,
        "cone":       3,
    }

    def __init__(
        self,
        camera_height_m: float = 1.25,
        pitch_angle_deg: float = -5.0,
        focal_length_px: float = 500.0,
        principal_point: Tuple[float, float] = (320.0, 180.0),
        lane_width_m: float = 3.2,
        max_assessment_distance_m: float = 40.0,
        emergency_brake_threshold: float = 0.70,
    ):
        self.cam_h = camera_height_m
        self.pitch_rad = math.radians(pitch_angle_deg)
        self.fy = focal_length_px
        self.fx = focal_length_px
        self.cx, self.cy = principal_point
        self.lane_half_width_m = lane_width_m / 2.0
        self.max_dist_m = max_assessment_distance_m
        self.emergency_brake_threshold = emergency_brake_threshold

    def estimate_3d_coordinates(
        self, bbox: List[float], img_w: int = 640, img_h: int = 360
    ) -> Tuple[float, float]:
        """
        Estimate ground-plane longitudinal distance Z and lateral offset X (in meters)
        from the bottom-center of the 2D bounding box.
        """
        x1, y1, x2, y2 = bbox
        bc_x = (x1 + x2) / 2.0
        bc_y = y2

        # Delta from optical center
        dy = bc_y - self.cy
        dx = bc_x - self.cx

        # Angle of depression relative to optical axis
        alpha_y = math.atan2(dy, self.fy)
        total_pitch = self.pitch_rad + alpha_y

        # Avoid singularity near or above the horizon
        if total_pitch > 0.02:
            dist_z = self.cam_h / math.tan(total_pitch)
        else:
            # Fallback estimation for distant or horizon-level targets
            norm_y = max(0.01, min(1.0, (bc_y - self.cy) / (img_h - self.cy)))
            dist_z = self.cam_h / (norm_y * math.tan(abs(self.pitch_rad) + 0.1) + 1e-3)

        dist_z = max(0.5, min(self.max_dist_m, dist_z))

        # Lateral displacement X = (dx / fx) * Z
        lateral_x = (dx / self.fx) * dist_z

        return dist_z, lateral_x

    def compute_risk_score(
        self,
        class_name: str,
        dist_z: float,
        lateral_x: float,
    ) -> Tuple[float, bool]:
        """
        Calculate threat Risk Score [0.0, 1.0] and In-Lane presence flag.
        """
        c_weight = self.CLASS_WEIGHTS.get(class_name.lower(), 0.40)

        # Distance proximity factor: 1.0 at 0m, decaying to 0.0 at max_dist_m
        prox_factor = max(0.0, min(1.0, 1.0 - (dist_z / self.max_dist_m)))

        # Lateral factor: Gaussian bell curve centered on vehicle path (X=0)
        # Objects in the center lane have factor ~1.0; objects in adjacent lanes decay sharply.
        sigma_lat = 0.90  # 90cm lateral spread
        lat_factor = math.exp(-0.5 * (lateral_x / sigma_lat) ** 2)

        # In-lane check
        is_in_lane = abs(lateral_x) <= self.lane_half_width_m

        # Combined threat score
        raw_risk = c_weight * (0.45 * prox_factor + 0.55 * lat_factor)
        risk_score = max(0.0, min(1.0, float(raw_risk)))

        return risk_score, is_in_lane

    def evaluate(
        self,
        detections: List[Detection2D],
        img_w: int = 640,
        img_h: int = 360,
    ) -> RiskEngineReport:
        """
        Evaluate all detected objects and determine the primary target.
        """
        targets: List[ThreatTarget] = []

        for idx, det in enumerate(detections):
            dist_z, lat_x = self.estimate_3d_coordinates(det.bbox, img_w=img_w, img_h=img_h)
            risk, in_lane = self.compute_risk_score(det.class_name, dist_z, lat_x)

            # Emergency Stop Flag: high risk and within stopping distance
            stop_flag = (risk >= self.emergency_brake_threshold and dist_z <= 18.0) or (in_lane and dist_z <= 3.5)

            dist_mm = int(round(dist_z * 1000.0))
            class_enum = self.CLASS_ENUMS.get(det.class_name.lower(), 0)

            target = ThreatTarget(
                target_id=idx + 1,
                class_name=det.class_name,
                class_enum=class_enum,
                bbox=det.bbox,
                distance_m=round(dist_z, 2),
                distance_mm=dist_mm,
                lateral_offset_m=round(lat_x, 2),
                is_in_lane=in_lane,
                risk_score=round(risk, 3),
                stop_flag=stop_flag,
            )
            targets.append(target)

        # Sort targets by risk_score descending
        targets.sort(key=lambda t: t.risk_score, reverse=True)

        primary = targets[0] if targets else None
        max_risk = primary.risk_score if primary else 0.0
        estop = any(t.stop_flag for t in targets)

        return RiskEngineReport(
            targets=targets,
            primary_target=primary,
            max_risk_score=max_risk,
            emergency_stop_required=estop,
        )
