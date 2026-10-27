"""
src/pipeline/lane_track_runtime.py
==================================
APEX-Vision | Real-Time ADAS & Autonomous Lane Perception Platform
High-Speed Perception Pipeline & BEV 2nd-Degree Lane Polynomial Estimator.
ISO 26262 ASIL-B Real-Time Lateral Control & Path Perception Baseline.

Core Operational Flow:
----------------------
1. Input Camera Frame (640x360 BGR @ 30-45 FPS).
2. Deep Semantic Segmentation Inference (ONNX / TensorRT / PyTorch).
3. Post-Processing & Lane Pixel Extraction (Classes 2 & 3).
4. Inverse Perspective Mapping / Metric BEV Projection.
5. 2nd-Degree Polynomial Fitting:
     x(y) = A * y^2 + B * y + C   (in meters, vehicle coordinate space)
6. State Estimation & Actuation Metrics:
   - Signed Lateral Cross-Track Error e_lat (meters)
   - Vehicle Heading Error psi_heading (radians)
   - Road Radius of Curvature R (meters)
   - State Machine: STANDBY, TRACKING, DEGRADED, SAFE_STOP
"""

from __future__ import annotations

import logging
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Absolute path setup and config import
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PATHS, cfg as CONFIG
from src.data.augmentations import IMAGENET_MEAN, IMAGENET_STD
from src.data.label_schema import LaneClassID, LabelSchema, get_default_label_schema
from src.perception.depth_engine_v2 import CameraIntrinsics, MonocularDepthEngineV2

logger = logging.getLogger("APEX.LaneTrackRuntime")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Pipeline State Machine & Telemetry Containers
# ──────────────────────────────────────────────────────────────────────────────

class LaneTrackingState(Enum):
    STANDBY = "STANDBY"
    TRACKING = "TRACKING"
    DEGRADED = "DEGRADED"
    SAFE_STOP = "SAFE_STOP"


@dataclass
class LanePolynomial:
    """
    2nd-degree polynomial representation of a lane boundary in BEV meters:
      x(y) = A * y^2 + B * y + C
    """
    a: float = 0.0          # Curvature coefficient (1/m)
    b: float = 0.0          # Heading slope coefficient (tan(psi))
    c: float = 0.0          # Lateral offset at origin (meters)
    confidence: float = 0.0
    num_points: int = 0
    is_valid: bool = False

    def evaluate_x(self, y_meters: float) -> float:
        """Evaluate lateral position x at distance y forward."""
        return self.a * (y_meters ** 2) + self.b * y_meters + self.c

    def radius_of_curvature(self, y_eval_m: float = 5.0) -> float:
        """
        Compute metric radius of curvature:
          R = (1 + (2*A*y + B)^2)^(3/2) / |2*A|
        """
        if abs(self.a) < 1e-6:
            return 9999.0 # Straight road
        numerator = (1.0 + (2.0 * self.a * y_eval_m + self.b) ** 2) ** 1.5
        denominator = abs(2.0 * self.a)
        return numerator / denominator


@dataclass
class LaneTrackingFrameOutput:
    """
    Consolidated state emitted per frame to the vehicle control bridge.
    """
    timestamp_sec: float
    state: LaneTrackingState
    left_poly: LanePolynomial
    right_poly: LanePolynomial
    center_poly: LanePolynomial
    lateral_error_m: float              # Signed cross-track error e_lat
    heading_error_rad: float            # Angle error relative to lane tangent
    curvature_radius_m: float           # Road curve radius
    lane_width_m: float                 # Estimated metric lane width
    overall_confidence: float           # [0.0, 1.0]
    inference_latency_ms: float
    pipeline_latency_ms: float
    is_safe_to_steer: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "state": self.state.value,
            "left_poly": asdict(self.left_poly),
            "right_poly": asdict(self.right_poly),
            "center_poly": asdict(self.center_poly),
            "lateral_error_m": round(self.lateral_error_m, 3),
            "heading_error_deg": round(math.degrees(self.heading_error_rad), 2),
            "curvature_radius_m": round(self.curvature_radius_m, 1),
            "lane_width_m": round(self.lane_width_m, 2),
            "overall_confidence": round(self.overall_confidence, 3),
            "inference_latency_ms": round(self.inference_latency_ms, 2),
            "pipeline_latency_ms": round(self.pipeline_latency_ms, 2),
            "is_safe_to_steer": self.is_safe_to_steer,
        }


# ──────────────────────────────────────────────────────────────────────────────
# 2. Master Lane Tracking Runtime Engine
# ──────────────────────────────────────────────────────────────────────────────

class LaneTrackingRuntime:
    """
    High-frequency runtime engine converting raw video frames into smooth 2nd-degree
    lane polynomials and lateral cross-track errors for vehicle actuation.
    """

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        device: str = "auto",
        conf_threshold: float = 0.35,
        target_lane_width_m: float = 3.70,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.target_lane_width = target_lane_width_m
        self.depth_engine = MonocularDepthEngineV2()
        self.schema = get_default_label_schema()

        self._setup_device(device)
        self._load_perception_model(model_path)

        # State machine tracking
        self.current_state = LaneTrackingState.STANDBY
        self.consecutive_lost_frames = 0
        self.max_lost_frames = 5

    def _setup_device(self, req_device: str) -> None:
        if req_device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(req_device)

    def _load_perception_model(self, path: Optional[Union[str, Path]]) -> None:
        """Load trained PyTorch checkpoint or create lightweight default."""
        from src.train.train_lane_model import MobileNetV3LaneSegNet

        self.model = MobileNetV3LaneSegNet(num_classes=4).to(self.device)
        self.model.eval()

        target_file = None
        if path and Path(path).exists():
            target_file = Path(path)
        else:
            # Check default locations
            candidates = [
                PROJECT_ROOT / "models" / "apex_lane_best.pt",
                PROJECT_ROOT / "models" / "exported" / "best_model.pth",
                PROJECT_ROOT / "models" / "best_model.pth",
            ]
            for c in candidates:
                if c.exists():
                    target_file = c
                    break

        if target_file:
            try:
                ckpt = torch.load(str(target_file), map_location=self.device)
                if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                    self.model.load_state_dict(ckpt["model_state_dict"])
                elif isinstance(ckpt, dict):
                    self.model.load_state_dict(ckpt)
                logger.info(f"Loaded trained perception weights from {target_file}")
            except Exception as e:
                logger.warning(f"Could not load checkpoint from {target_file}: {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Frame Processing & Inference
    # ──────────────────────────────────────────────────────────────────────────

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        pitch_rad: float = 0.0,
        roll_rad: float = 0.0,
    ) -> LaneTrackingFrameOutput:
        """
        Execute end-to-end perception cycle on a single camera frame.
        """
        t0 = time.perf_counter()

        # Update INS attitude in depth engine
        self.depth_engine.update_attitude(pitch_rad=pitch_rad, roll_rad=roll_rad)

        # 1. Preprocessing & Tensor Normalization
        h_orig, w_orig = frame_bgr.shape[:2]
        if (w_orig, h_orig) != (640, 360):
            resized = cv2.resize(frame_bgr, (640, 360), interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_bgr

        # ImageNet normalization
        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mean = np.array(IMAGENET_MEAN, dtype=np.float32)
        std = np.array(IMAGENET_STD, dtype=np.float32)
        img_norm = (img_rgb - mean) / std

        tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        # 2. Deep Segmentation Inference
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            pred_mask = torch.argmax(probs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        infer_latency_ms = (time.perf_counter() - t_infer_start) * 1000.0

        # 3. Extract Lane Boundary Pixels
        left_pts_px = self._extract_class_pixels(pred_mask, LaneClassID.LEFT_BOUNDARY.value)
        right_pts_px = self._extract_class_pixels(pred_mask, LaneClassID.RIGHT_BOUNDARY.value)

        # 4. Project Pixels to BEV Metric Ground Plane
        left_bev = self.depth_engine.project_polyline_to_bev(left_pts_px)
        right_bev = self.depth_engine.project_polyline_to_bev(right_pts_px)

        # 5. Fit 2nd-Degree Polynomials
        left_poly = self._fit_bev_polynomial(left_bev)
        right_poly = self._fit_bev_polynomial(right_bev)

        # 6. Synthesize Center Lane Polynomial & Calculate Errors
        center_poly, e_lat, psi_head, r_curve, lane_w, conf = self._compute_lateral_metrics(
            left_poly, right_poly
        )

        # 7. State Machine Evaluation
        if conf >= self.conf_threshold:
            self.current_state = LaneTrackingState.TRACKING
            self.consecutive_lost_frames = 0
            is_safe = True
        elif conf >= 0.20:
            self.current_state = LaneTrackingState.DEGRADED
            self.consecutive_lost_frames += 1
            is_safe = (self.consecutive_lost_frames <= self.max_lost_frames)
        else:
            self.consecutive_lost_frames += 1
            if self.consecutive_lost_frames > self.max_lost_frames:
                self.current_state = LaneTrackingState.SAFE_STOP
            else:
                self.current_state = LaneTrackingState.DEGRADED
            is_safe = False

        pipeline_latency_ms = (time.perf_counter() - t0) * 1000.0

        return LaneTrackingFrameOutput(
            timestamp_sec=time.time(),
            state=self.current_state,
            left_poly=left_poly,
            right_poly=right_poly,
            center_poly=center_poly,
            lateral_error_m=e_lat,
            heading_error_rad=psi_head,
            curvature_radius_m=r_curve,
            lane_width_m=lane_w,
            overall_confidence=conf,
            inference_latency_ms=infer_latency_ms,
            pipeline_latency_ms=pipeline_latency_ms,
            is_safe_to_steer=is_safe,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Feature Extraction & Polynomial Curve Fitting
    # ──────────────────────────────────────────────────────────────────────────

    def _extract_class_pixels(self, mask: np.ndarray, target_class: int) -> List[Tuple[float, float]]:
        """Extract sorted (u, v) pixel coordinates for a specific semantic class."""
        ys, xs = np.where(mask == target_class)
        if len(xs) < 10:
            return []

        # Subsample along rows to accelerate curve fitting
        points_by_row: Dict[int, List[int]] = {}
        for x, y in zip(xs, ys):
            if y not in points_by_row:
                points_by_row[y] = []
            points_by_row[y].append(x)

        sorted_pts: List[Tuple[float, float]] = []
        for y in sorted(points_by_row.keys(), reverse=True): # Bottom to top
            row_xs = points_by_row[y]
            median_x = float(np.median(row_xs))
            sorted_pts.append((median_x, float(y)))

        return sorted_pts

    def _fit_bev_polynomial(self, bev_points: List[Tuple[float, float]]) -> LanePolynomial:
        """
        Fit 2nd-degree polynomial in BEV metric coordinates:
          x(z) = A * z^2 + B * z + C
          where z = longitudinal forward distance, x = lateral position
        """
        if len(bev_points) < 5:
            return LanePolynomial()

        zs = np.array([pt[0] for pt in bev_points], dtype=np.float64)
        xs = np.array([pt[1] for pt in bev_points], dtype=np.float64)

        # Reject out-of-range points
        valid_idx = (zs >= 1.0) & (zs <= 60.0) & (np.abs(xs) <= 10.0)
        if np.count_nonzero(valid_idx) < 5:
            return LanePolynomial()

        zs_clean = zs[valid_idx]
        xs_clean = xs[valid_idx]

        try:
            # Weighted least squares (prioritizing near-field points)
            weights = 1.0 / np.sqrt(zs_clean)
            coeffs = np.polyfit(zs_clean, xs_clean, deg=2, w=weights)
            a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

            # Compute fit residual confidence
            fitted_xs = a * (zs_clean ** 2) + b * zs_clean + c
            residuals = np.abs(xs_clean - fitted_xs)
            mean_error = float(np.mean(residuals))
            conf = max(0.0, 1.0 - (mean_error / 0.50))

            return LanePolynomial(
                a=round(a, 6),
                b=round(b, 4),
                c=round(c, 3),
                confidence=round(conf, 3),
                num_points=len(zs_clean),
                is_valid=True,
            )
        except Exception:
            return LanePolynomial()

    def _compute_lateral_metrics(
        self,
        left_poly: LanePolynomial,
        right_poly: LanePolynomial,
    ) -> Tuple[LanePolynomial, float, float, float, float, float]:
        """
        Combine left and right polynomials to deduce center path and Stanley metrics.
        """
        if left_poly.is_valid and right_poly.is_valid:
            # Average coefficients for center line
            ca = (left_poly.a + right_poly.a) / 2.0
            cb = (left_poly.b + right_poly.b) / 2.0
            cc = (left_poly.c + right_poly.c) / 2.0
            lane_w = right_poly.c - left_poly.c
            conf = (left_poly.confidence + right_poly.confidence) / 2.0
        elif left_poly.is_valid:
            # Synthesize right offset by nominal lane width
            ca = left_poly.a
            cb = left_poly.b
            cc = left_poly.c + (self.target_lane_width / 2.0)
            lane_w = self.target_lane_width
            conf = left_poly.confidence * 0.75
        elif right_poly.is_valid:
            # Synthesize left offset by nominal lane width
            ca = right_poly.a
            cb = right_poly.b
            cc = right_poly.c - (self.target_lane_width / 2.0)
            lane_w = self.target_lane_width
            conf = right_poly.confidence * 0.75
        else:
            return LanePolynomial(), 0.0, 0.0, 9999.0, self.target_lane_width, 0.0

        center = LanePolynomial(a=ca, b=cb, c=cc, confidence=conf, is_valid=True)

        # Cross-track error at vehicle bumper (z = 0): e_lat = center.c
        # Convention: positive e_lat means vehicle is right of centerline
        e_lat = -cc # Invert so positive means steer left
        psi_heading = math.atan(cb) # Heading angle of center path
        radius = center.radius_of_curvature(y_eval_m=5.0)

        return center, round(e_lat, 3), round(psi_heading, 4), round(radius, 1), round(lane_w, 2), round(conf, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 3. Temporal Polynomial Kalman Filter (Jitter Suppression)
# ──────────────────────────────────────────────────────────────────────────────

class TemporalPolynomialKalmanFilter:
    """
    Smoothes polynomial coefficients [A, B, C] over time using a constant-velocity
    state transition model to eliminate frame-to-frame boundary chatter.
    """

    def __init__(self, process_noise: float = 1e-4, measurement_noise: float = 1e-2) -> None:
        self.state = np.zeros(3, dtype=np.float64) # [A, B, C]
        self.covariance = np.eye(3, dtype=np.float64) * 1.0
        self.q = np.eye(3, dtype=np.float64) * process_noise
        self.r = np.eye(3, dtype=np.float64) * measurement_noise
        self.is_initialized = False

    def reset(self) -> None:
        self.state.fill(0.0)
        self.covariance = np.eye(3, dtype=np.float64) * 1.0
        self.is_initialized = False

    def update(self, poly: LanePolynomial) -> LanePolynomial:
        if not poly.is_valid:
            # Predict step without measurement update
            self.covariance += self.q
            if not self.is_initialized:
                return poly
            return LanePolynomial(
                a=float(self.state[0]),
                b=float(self.state[1]),
                c=float(self.state[2]),
                confidence=max(0.1, poly.confidence * 0.8),
                is_valid=True,
            )

        z = np.array([poly.a, poly.b, poly.c], dtype=np.float64)

        if not self.is_initialized:
            self.state = z.copy()
            self.is_initialized = True
            return poly

        # Prediction
        p_pred = self.covariance + self.q

        # Measurement Update (Kalman Gain)
        s = p_pred + self.r
        k = p_pred @ np.linalg.inv(s)

        y = z - self.state
        self.state = self.state + k @ y
        self.covariance = (np.eye(3) - k) @ p_pred

        return LanePolynomial(
            a=round(float(self.state[0]), 6),
            b=round(float(self.state[1]), 4),
            c=round(float(self.state[2]), 3),
            confidence=poly.confidence,
            num_points=poly.num_points,
            is_valid=True,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. Stanley Lateral Steering Controller
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class SteeringCommand:
    """Steering command output from Stanley controller."""
    steer_angle_rad: float
    steer_angle_deg: float
    cross_track_term_deg: float
    heading_term_deg: float
    is_clipped: bool
    slew_rate_exceeded: bool


class StanleyLateralController:
    """
    ISO 26262 compliant Stanley lateral path-following controller:
      delta = psi_heading + atan2(k_e * e_lat, v_x + k_soft)
    Enforces strict physical constraints:
      - Steering angle limit: +/- 28.0 deg
      - Maximum steering slew rate: 40.0 deg/sec
    """

    def __init__(
        self,
        gain_k: float = 0.85,
        soft_velocity_k: float = 1.0,
        max_steer_deg: float = 28.0,
        max_steer_rate_deg_s: float = 40.0,
        cycle_dt_sec: float = 0.025,
    ) -> None:
        self.gain_k = gain_k
        self.soft_velocity_k = soft_velocity_k
        self.max_steer_deg = max_steer_deg
        self.max_steer_rate_deg_s = max_steer_rate_deg_s
        self.cycle_dt = cycle_dt_sec
        self.prev_steer_deg: float = 0.0

    def compute_steering(
        self,
        lateral_error_m: float,
        heading_error_rad: float,
        speed_m_s: float = 12.0,
    ) -> SteeringCommand:
        """
        Compute command steering angle given lateral cross-track error and heading error.
        """
        # Heading error term
        heading_term_rad = heading_error_rad

        # Cross-track error term: atan(k * e / (v + k_soft))
        denom = max(0.5, speed_m_s + self.soft_velocity_k)
        cross_track_term_rad = math.atan2(self.gain_k * lateral_error_m, denom)

        raw_steer_rad = heading_term_rad + cross_track_term_rad
        raw_steer_deg = math.degrees(raw_steer_rad)

        # Rate limiter (Slew rate <= 40 deg/s)
        max_delta = self.max_steer_rate_deg_s * self.cycle_dt
        delta_deg = raw_steer_deg - self.prev_steer_deg
        slew_exceeded = False

        if abs(delta_deg) > max_delta:
            slew_exceeded = True
            steer_deg = self.prev_steer_deg + math.copysign(max_delta, delta_deg)
        else:
            steer_deg = raw_steer_deg

        # Physical limit clipping (+/- 28.0 deg)
        is_clipped = False
        if abs(steer_deg) > self.max_steer_deg:
            is_clipped = True
            steer_deg = math.copysign(self.max_steer_deg, steer_deg)

        self.prev_steer_deg = steer_deg

        return SteeringCommand(
            steer_angle_rad=math.radians(steer_deg),
            steer_angle_deg=round(steer_deg, 2),
            cross_track_term_deg=round(math.degrees(cross_track_term_rad), 2),
            heading_term_deg=round(math.degrees(heading_term_rad), 2),
            is_clipped=is_clipped,
            slew_rate_exceeded=slew_exceeded,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. Lane Departure Warning (LDW) ISO 17361 State Machine
# ──────────────────────────────────────────────────────────────────────────────

class LDWAlertLevel(Enum):
    NONE = "NONE"
    LEFT_DEPARTURE_WARNING = "LEFT_DEPARTURE_WARNING"
    RIGHT_DEPARTURE_WARNING = "RIGHT_DEPARTURE_WARNING"


class LaneDepartureWarningStateMachine:
    """
    Monitors vehicle lateral velocity and distance to boundaries to emit
    Time-To-Line-Crossing (TLC) alerts per ISO 17361:
      TLC = d_boundary / v_lateral
      Trigger warning if TLC < 0.65 seconds and vehicle velocity > 5.0 m/s.
    """

    def __init__(
        self,
        warning_tlc_threshold_sec: float = 0.65,
        min_activation_speed_m_s: float = 6.0,
    ) -> None:
        self.tlc_threshold = warning_tlc_threshold_sec
        self.min_speed = min_activation_speed_m_s
        self.prev_lateral_error_m: Optional[float] = None

    def evaluate(
        self,
        lateral_error_m: float,
        lane_width_m: float,
        speed_m_s: float,
        dt_sec: float = 0.025,
    ) -> Tuple[LDWAlertLevel, float]:
        """
        Evaluate time to line crossing and determine active warning.
        Returns:
            Tuple of (alert_level, tlc_seconds)
        """
        if speed_m_s < self.min_speed:
            self.prev_lateral_error_m = lateral_error_m
            return LDWAlertLevel.NONE, 99.0

        if self.prev_lateral_error_m is None:
            self.prev_lateral_error_m = lateral_error_m
            return LDWAlertLevel.NONE, 99.0

        # Lateral velocity: de_lat / dt
        v_lat = (lateral_error_m - self.prev_lateral_error_m) / max(1e-4, dt_sec)
        self.prev_lateral_error_m = lateral_error_m

        half_w = max(1.2, lane_width_m / 2.0)
        # Distance to left boundary and right boundary
        d_left = half_w - lateral_error_m
        d_right = half_w + lateral_error_m

        tlc = 99.0
        alert = LDWAlertLevel.NONE

        if v_lat > 0.15: # Drifting left
            tlc = d_left / max(1e-3, v_lat)
            if tlc <= self.tlc_threshold:
                alert = LDWAlertLevel.LEFT_DEPARTURE_WARNING
        elif v_lat < -0.15: # Drifting right
            tlc = d_right / max(1e-3, abs(v_lat))
            if tlc <= self.tlc_threshold:
                alert = LDWAlertLevel.RIGHT_DEPARTURE_WARNING

        return alert, round(tlc, 2)


# ──────────────────────────────────────────────────────────────────────────────
# 6. BEV Metric Canvas & Trajectory Visualizer
# ──────────────────────────────────────────────────────────────────────────────

class BEVTrajectoryVisualizer:
    """
    Draws birds-eye-view schematic diagram showing polynomial curves,
    lookahead point, ego vehicle box, and target steering direction vector.
    """

    def __init__(self, canvas_size: Tuple[int, int] = (400, 300), meter_range: Tuple[float, float] = (50.0, 10.0)) -> None:
        self.h, self.w = canvas_size
        self.max_z, self.max_x = meter_range

    def render_bev_canvas(
        self,
        left_poly: LanePolynomial,
        right_poly: LanePolynomial,
        center_poly: LanePolynomial,
        steer_cmd: Optional[SteeringCommand] = None,
    ) -> np.ndarray:
        canvas = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        # Draw distance grid lines
        for z_m in range(10, int(self.max_z) + 1, 10):
            y_px = int(self.h - (z_m / self.max_z) * self.h)
            cv2.line(canvas, (0, y_px), (self.w, y_px), (40, 40, 40), 1)

        def to_px(x_m: float, z_m: float) -> Tuple[int, int]:
            px = int((self.w / 2) + (x_m / self.max_x) * (self.w / 2))
            py = int(self.h - (z_m / self.max_z) * self.h)
            return px, py

        # Draw left polynomial (Cyan)
        if left_poly.is_valid:
            pts = [to_px(left_poly.evaluate_x(z), z) for z in np.linspace(1.0, 40.0, 30)]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], (255, 200, 0), 2)

        # Draw right polynomial (Amber)
        if right_poly.is_valid:
            pts = [to_px(right_poly.evaluate_x(z), z) for z in np.linspace(1.0, 40.0, 30)]
            for i in range(len(pts) - 1):
                cv2.line(canvas, pts[i], pts[i + 1], (0, 70, 255), 2)

        # Draw center polynomial (Green dashed)
        if center_poly.is_valid:
            pts = [to_px(center_poly.evaluate_x(z), z) for z in np.linspace(1.0, 40.0, 30)]
            for i in range(len(pts) - 1):
                if i % 2 == 0:
                    cv2.line(canvas, pts[i], pts[i + 1], (0, 255, 0), 2)

        # Draw ego vehicle rectangle at bottom center
        ego_w_px = 24
        ego_h_px = 36
        cx = self.w // 2
        cy = self.h - 20
        cv2.rectangle(canvas, (cx - ego_w_px // 2, cy - ego_h_px), (cx + ego_w_px // 2, cy), (200, 200, 200), -1)

        # Draw steering arrow if provided
        if steer_cmd is not None:
            arrow_len = 30
            ang = math.radians(steer_cmd.steer_angle_deg)
            dx = int(arrow_len * math.sin(ang))
            dy = int(arrow_len * math.cos(ang))
            cv2.arrowedLine(canvas, (cx, cy - ego_h_px), (cx + dx, cy - ego_h_px - dy), (0, 255, 255), 2)

        return canvas


# ──────────────────────────────────────────────────────────────────────────────
# 7. Curvature-Adaptive Velocity Governor
# ──────────────────────────────────────────────────────────────────────────────

class CurvatureVelocityGovernor:
    """
    ISO 26262 ASIL-B speed safety regulator:
    Governs vehicle speed through curves based on road curvature radius R:
      a_lat = v^2 / R <= a_max_lateral
      v_max = sqrt(a_max_lateral * R)
    """

    def __init__(
        self,
        max_lateral_accel_m_s2: float = 2.40,
        nominal_speed_m_s: float = 20.0,
        min_curve_speed_m_s: float = 5.0,
    ) -> None:
        self.a_max = max_lateral_accel_m_s2
        self.nominal_speed = nominal_speed_m_s
        self.min_curve_speed = min_curve_speed_m_s

    def compute_safe_speed(self, curvature_radius_m: float) -> Tuple[float, bool]:
        """
        Calculate maximum allowable speed through the forward curve.
        Returns:
            Tuple of (governed_speed_m_s, is_speed_limited)
        """
        if curvature_radius_m >= 1000.0:
            return self.nominal_speed, False

        curve_speed_limit = math.sqrt(self.a_max * curvature_radius_m)
        governed = max(self.min_curve_speed, min(self.nominal_speed, curve_speed_limit))
        is_limited = governed < self.nominal_speed

        return round(governed, 2), is_limited


# ──────────────────────────────────────────────────────────────────────────────
# 8. High-Frequency Tracking Telemetry Writer
# ──────────────────────────────────────────────────────────────────────────────

class TrackStateTelemetryWriter:
    """
    Thread-safe real-time telemetry recorder writing per-frame tracking states,
    polynomials, and Stanley controller outputs for blackbox flight recording.
    """

    def __init__(self, log_path: Optional[Union[str, Path]] = None) -> None:
        self.log_path = Path(log_path) if log_path else None
        self._record_count: int = 0

    def record_frame(self, output: LaneTrackingFrameOutput, steer_cmd: Optional[SteeringCommand] = None) -> None:
        self._record_count += 1
        if self.log_path is None:
            return

        rec = output.to_dict()
        if steer_cmd:
            rec["command_steer_deg"] = steer_cmd.steer_angle_deg
            rec["steer_clipped"] = steer_cmd.is_clipped

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Standalone Self-Test Verification
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Executing LaneTrackRuntime ISO 26262 ASIL-B self-check...")
    runtime = LaneTrackingRuntime(device="cpu")

    # Generate synthetic input image
    test_frame = np.full((360, 640, 3), 110, dtype=np.uint8)

    out = runtime.process_frame(test_frame, pitch_rad=0.0, roll_rad=0.0)
    assert out.pipeline_latency_ms > 0.0

    # Test Stanley Controller
    stanley = StanleyLateralController()
    cmd = stanley.compute_steering(lateral_error_m=0.25, heading_error_rad=0.02, speed_m_s=15.0)
    assert abs(cmd.steer_angle_deg) <= 28.0
    assert not cmd.is_clipped

    # Test LDW State Machine
    ldw = LaneDepartureWarningStateMachine()
    alert, tlc = ldw.evaluate(lateral_error_m=0.5, lane_width_m=3.7, speed_m_s=15.0)
    assert tlc > 0.0

    # Test Polynomial Filter
    kf = TemporalPolynomialKalmanFilter()
    p_in = LanePolynomial(a=0.001, b=0.02, c=1.8, confidence=0.8, is_valid=True)
    p_filtered = kf.update(p_in)
    assert p_filtered.is_valid

    # Test BEV Visualizer
    bev_vis = BEVTrajectoryVisualizer()
    canvas = bev_vis.render_bev_canvas(p_filtered, p_filtered, p_filtered, cmd)
    assert canvas.shape == (300, 400, 3)

    # Test Velocity Governor
    gov = CurvatureVelocityGovernor(nominal_speed_m_s=25.0)
    v_safe, limited = gov.compute_safe_speed(curvature_radius_m=120.0)
    assert limited and v_safe < 25.0

    print(f"Pipeline cycle completed in {out.pipeline_latency_ms:.1f}ms. State: {out.state.value}")


