"""
src/perception/risk_assessment.py
=================================
Priority Threat Risk Assessor for APEX-LKA:
- Evaluates collision risk and threat severity for all detected objects.
- Analyzes:
  1. Normalized Proximity Score S_prox = y2 / H
  2. Drivable Corridor Polygon Intersection / Centerline Proximity
  3. Class-Specific Vulnerability & Threat Severity Multiplier
- Threat Risk Score:
  Risk = clamp(100 * (0.5 * S_prox + 0.3 * InCorridor + 0.2 * Severity), 0, 100)
- Triggers Emergency Brake Flag if Risk > 85 and S_prox > 0.75.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from src.perception.object_detector import Detection, DetectionResult, OBJECT_CLASSES

log = logging.getLogger("apex_risk_assessment")

# Class Severity Multipliers (0.0 to 1.0)
CLASS_SEVERITY_MAP: Dict[str, float] = {
    "pedestrian":    1.0,  # Critical vulnerable road user
    "animal":        1.0,  # Critical dynamic obstacle
    "cyclist":       0.9,  # High vulnerable road user
    "vehicle":       0.8,  # High mass collision threat
    "barrier":       0.8,  # High rigid collision hazard
    "traffic_cone":  0.5,  # Medium channelization marker
    "debris":        0.5,  # Medium road surface obstacle
    "traffic_sign":  0.1,  # Low informational infrastructure
    "traffic_light": 0.1,  # Low informational infrastructure
}


@dataclass
class AssessedObject:
    """Individual object with computed threat and risk telemetry."""
    detection: Detection
    proximity_score: float
    in_corridor: bool
    severity_multiplier: float
    risk_score: float
    distance_y_px: float
    distance_mm: int


@dataclass
class RiskAssessmentReport:
    """Aggregated scene risk assessment report."""
    assessed_objects: List[AssessedObject] = field(default_factory=list)
    highest_risk_object: Optional[AssessedObject] = None
    highest_risk_score: float = 0.0
    highest_risk_class_id: int = 0
    highest_risk_distance_mm: int = 0
    emergency_brake_flag: bool = False
    assessment_ms: float = 0.0

    @property
    def has_threat(self) -> bool:
        return self.highest_risk_score >= 30.0

    @property
    def is_critical(self) -> bool:
        return self.highest_risk_score >= 70.0


class PriorityRiskAssessor:
    """
    Automotive-grade threat evaluation engine for vehicle collision avoidance.
    """

    def __init__(
        self,
        emergency_brake_risk_thresh: float = 85.0,
        proximity_brake_thresh: float = 0.75,
        corridor_padding_px: float = 20.0,
        camera_height_mm: float = 1200.0,
        camera_fy_px: float = 612.0,
        horizon_y_px: float = 126.0,
    ) -> None:
        self.emergency_brake_risk_thresh = emergency_brake_risk_thresh
        self.proximity_brake_thresh = proximity_brake_thresh
        self.corridor_padding_px = corridor_padding_px
        self.camera_height_mm = camera_height_mm
        self.camera_fy_px = camera_fy_px
        self.horizon_y_px = horizon_y_px

    def estimate_distance_mm(self, y2_px: float) -> int:
        """
        Estimate ground-plane distance in millimeters using pinhole camera projection.
        Distance Z = (fy * H_cam) / (y2 - y_horizon)
        """
        dy = max(5.0, y2_px - self.horizon_y_px)
        z_mm = (self.camera_fy_px * self.camera_height_mm) / dy
        return int(np.clip(z_mm, 500.0, 65535.0))

    def is_in_corridor(
        self,
        x_bc: float,
        y2: float,
        img_w: int,
        img_h: int,
        corridor_poly: Optional[np.ndarray] = None,
    ) -> bool:
        """
        Checks if the object's bottom-center coordinate lies inside the drivable corridor.
        """
        if corridor_poly is not None and len(corridor_poly) >= 3:
            poly = np.asarray(corridor_poly, dtype=np.float32)
            if poly.ndim == 3 and poly.shape[1] == 1:
                poly = poly[:, 0, :]
            # Scale polygon to image size if needed
            dist = cv2.pointPolygonTest(poly, (float(x_bc), float(y2)), measureDist=True)
            return bool(dist >= -self.corridor_padding_px)

        # Default dynamic trapezoidal corridor from ego vehicle position
        # Bottom width ~280px, horizon width ~80px centered at W/2
        center_x = img_w / 2.0
        y_frac = np.clip(y2 / float(max(1, img_h)), 0.0, 1.0)
        half_width = 40.0 + 100.0 * y_frac + self.corridor_padding_px
        return bool(abs(x_bc - center_x) <= half_width)

    def assess(
        self,
        detections: Union[DetectionResult, List[Detection]],
        drivable_corridor_poly: Optional[np.ndarray] = None,
        vehicle_speed_mps: float = 0.0,
        frame_shape: Tuple[int, int] = (360, 640),  # (H, W)
    ) -> RiskAssessmentReport:
        """
        Computes threat risk scores for all detected entities.
        """
        t_start = time.perf_counter()
        img_h, img_w = frame_shape[:2]

        if isinstance(detections, DetectionResult):
            det_list = detections.get_detections()
        else:
            det_list = detections or []

        if not det_list:
            return RiskAssessmentReport(
                assessment_ms=(time.perf_counter() - t_start) * 1000.0
            )

        assessed: List[AssessedObject] = []
        highest_obj: Optional[AssessedObject] = None
        max_risk: float = -1.0

        for det in det_list:
            x1, y1, x2, y2 = det.box
            x_bc = (x1 + x2) / 2.0

            # 1. Proximity Score S_prox = y2 / H (objects near bottom are closer)
            s_prox = float(np.clip(y2 / float(max(1, img_h)), 0.0, 1.0))

            # 2. Corridor intersection
            in_corr = self.is_in_corridor(
                x_bc=x_bc,
                y2=y2,
                img_w=img_w,
                img_h=img_h,
                corridor_poly=drivable_corridor_poly,
            )
            corr_val = 1.0 if in_corr else 0.0

            # 3. Class severity multiplier
            c_name = det.class_name.lower().replace(" ", "_")
            severity = CLASS_SEVERITY_MAP.get(c_name, 0.5)

            # 4. Threat Risk Score Equation
            # Risk = clamp(100 * (0.5 * S_prox + 0.3 * InCorridor + 0.2 * Severity), 0, 100)
            raw_risk = 100.0 * (0.5 * s_prox + 0.3 * corr_val + 0.2 * severity)
            # Modulate by detection confidence
            risk_score = float(np.clip(raw_risk * det.score, 0.0, 100.0))

            dist_mm = self.estimate_distance_mm(y2)

            assessed_item = AssessedObject(
                detection=det,
                proximity_score=s_prox,
                in_corridor=in_corr,
                severity_multiplier=severity,
                risk_score=risk_score,
                distance_y_px=float(y2),
                distance_mm=dist_mm,
            )
            assessed.append(assessed_item)

            if risk_score > max_risk:
                max_risk = risk_score
                highest_obj = assessed_item

        # Emergency brake condition
        emergency_brake = False
        if highest_obj is not None:
            if (
                highest_obj.risk_score >= self.emergency_brake_risk_thresh
                and highest_obj.proximity_score >= self.proximity_brake_thresh
            ):
                emergency_brake = True

        return RiskAssessmentReport(
            assessed_objects=assessed,
            highest_risk_object=highest_obj,
            highest_risk_score=max_risk if highest_obj is not None else 0.0,
            highest_risk_class_id=highest_obj.detection.class_id if highest_obj else 0,
            highest_risk_distance_mm=highest_obj.distance_mm if highest_obj else 0,
            emergency_brake_flag=emergency_brake,
            assessment_ms=(time.perf_counter() - t_start) * 1000.0,
        )
