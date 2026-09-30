"""
src/inference/classical_cv.py
==============================
High-Accuracy Classical Computer-Vision Lane Detection Engine.

Key Features & Enhancements
---------------------------
1. Dynamic Horizon & Pavement Detection:
   - Evaluates dark/light intensity gradients in the upper portion of the frame.
   - Constrains Region of Interest (ROI) strictly below estimated horizon line,
     suppressing sky, clouds, overhead signs, and elevated bridges.
2. Guardrail & Barrier Rejection:
   - Precise HLS color gating:
     * Yellow lane: H in [15, 35], L in [30, 204], S in [115, 255]
     * White lane:  H in [0, 180], L in [190, 255], S in [0, 255]
     * Filters metallic grey/silver guardrails and diffuse asphalt reflections.
   - Slope & spatial filtering:
     * Rejects near-vertical lines (|slope| > 2.0) and near-horizontal noise (|slope| < 0.30).
     * Enforces lane envelope boundary intercepts.
3. 2nd-Degree Polynomial Fitting & Metric Offset:
   - Fits quadratic curves x = a*y^2 + b*y + c for curved highway geometries.
   - Derives bottom lane intercepts at y = H - 1.
   - Computes metric lateral offset assuming standard 3.7m highway lane width (0.0185 m/px).
4. Unified Lane Accuracy Metric:
   - Accuracy = 0.4 * Conf_left + 0.4 * Conf_right + 0.2 * GeometryPlausibility
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import (
    BaseLanePredictor,
    DetectionStatus,
    LanePrediction,
    ModelStatus,
)
from src.inference.preprocessing import PreprocessResult

# Conversion constant: standard 3.7m lane width / ~200px at model resolution (640x360)
METERS_PER_PIXEL_DEFAULT = 0.0185


# ═══════════════════════════════════════════════════════════════════════════
# Accuracy Score Calculator
# ═══════════════════════════════════════════════════════════════════════════

def compute_lane_accuracy(
    left_conf: float,
    right_conf: float,
    geom_plausibility: float,
) -> float:
    """
    Unified Lane Accuracy Metric [0.0, 1.0]:
        Accuracy = 0.4 * Conf_left + 0.4 * Conf_right + 0.2 * GeometryPlausibility
    """
    acc = 0.4 * float(left_conf) + 0.4 * float(right_conf) + 0.2 * float(geom_plausibility)
    return float(np.clip(acc, 0.0, 1.0))


# ═══════════════════════════════════════════════════════════════════════════
# Classical CV Predictor
# ═══════════════════════════════════════════════════════════════════════════

class ClassicalCVPredictor(BaseLanePredictor):
    """
    High-accuracy classical computer-vision lane detector with horizon detection,
    guardrail suppression, and 2nd-degree polynomial curve fitting.
    """

    LABEL = "CLASSICAL CV BASELINE — NOT THE FINAL ML MODEL"

    @property
    def model_status(self) -> ModelStatus:
        return ModelStatus.CLASSICAL_CV

    @property
    def backend_name(self) -> str:
        return "Classical CV (OpenCV)"

    def predict(self, preprocessed: PreprocessResult) -> LanePrediction:
        """Run the classical CV pipeline on preprocessed input."""
        if not preprocessed.valid:
            return LanePrediction(
                status        = DetectionStatus.INFERENCE_ERROR,
                model_status  = ModelStatus.CLASSICAL_CV,
                backend_name  = self.backend_name,
                error_message = f"Invalid preprocessed input: {preprocessed.error}",
            )

        t0 = time.perf_counter()
        h, w = preprocessed.model_h, preprocessed.model_w

        try:
            img = preprocessed.resized_bgr   # uint8 BGR (360x640)

            # ── 1. Dynamic Horizon Detection ──────────────────────────────
            horizon_y = detect_horizon(img)

            # ── 2. Precise HLS Color Mask (White & Yellow Lanes) ─────────
            colour_mask = _colour_mask(img)

            # ── 3. Region of Interest Mask Below Horizon ──────────────────
            roi_mask = _roi_mask(h, w, horizon_y=horizon_y)
            masked = cv2.bitwise_and(colour_mask, roi_mask)

            # ── 4. Edge Detection ─────────────────────────────────────────
            edges = cv2.Canny(masked, 50, 150)

            # ── 5. Probabilistic Hough Transform ──────────────────────────
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=25, minLineLength=15, maxLineGap=80,
            )

            # ── 6. Guardrail Rejection & Left/Right Line Separation ───────
            left_pts, right_pts = _separate_lines(lines, w, h, horizon_y)

            # ── 7. 2nd-Degree Polynomial Fitting & Mask Rasterization ────
            left_mask, left_conf, left_intercept = _fit_and_rasterise(
                left_pts, h, w, horizon_y, side="left"
            )
            right_mask, right_conf, right_intercept = _fit_and_rasterise(
                right_pts, h, w, horizon_y, side="right"
            )

            # ── 8. Geometry Plausibility & Metric Offset ──────────────────
            geom_plausibility = 0.0
            if left_intercept is not None and right_intercept is not None:
                lane_width_px = right_intercept - left_intercept
                # Expected lane width at bottom is ~200-450 px on a 640px wide frame
                if 0.28 * w <= lane_width_px <= 0.82 * w:
                    geom_plausibility = 1.0
                elif 0.20 * w <= lane_width_px <= 0.90 * w:
                    geom_plausibility = 0.7
                else:
                    geom_plausibility = 0.3
            elif left_intercept is not None or right_intercept is not None:
                geom_plausibility = 0.5

            accuracy_score = compute_lane_accuracy(left_conf, right_conf, geom_plausibility)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            both = (left_mask is not None and left_mask.any() and
                    right_mask is not None and right_mask.any())
            one  = (left_mask is not None and left_mask.any() or
                    right_mask is not None and right_mask.any())

            det_status = (
                DetectionStatus.LANE_DETECTED         if both else
                DetectionStatus.PARTIAL_LANE_DETECTED if one  else
                DetectionStatus.LANE_NOT_DETECTED
            )

            return LanePrediction(
                status                = det_status,
                model_status          = ModelStatus.CLASSICAL_CV,
                backend_name          = self.backend_name,
                left_mask             = left_mask if left_mask is not None else np.zeros((h, w), np.uint8),
                right_mask            = right_mask if right_mask is not None else np.zeros((h, w), np.uint8),
                left_confidence       = left_conf,
                right_confidence      = right_conf,
                model_h               = h,
                model_w               = w,
                inference_ms          = elapsed_ms,
                accuracy_score        = accuracy_score,
                geometry_plausibility = geom_plausibility,
            )

        except Exception as exc:
            return LanePrediction(
                status        = DetectionStatus.INFERENCE_ERROR,
                model_status  = ModelStatus.CLASSICAL_CV,
                backend_name  = self.backend_name,
                error_message = f"Classical CV inference error: {exc}",
                inference_ms  = (time.perf_counter() - t0) * 1000.0,
            )


# ═══════════════════════════════════════════════════════════════════════════
# Internal CV Algorithms
# ═══════════════════════════════════════════════════════════════════════════

def detect_horizon(img_bgr: np.ndarray) -> int:
    """
    Detect the horizon line by gradient analysis across the upper 60% of the image.
    Returns the horizon Y-coordinate in pixels.
    """
    h, w = img_bgr.shape[:2]
    search_h = int(0.60 * h)
    gray = cv2.cvtColor(img_bgr[:search_h, :], cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (11, 11), 0)

    # Compute vertical gradient (Sobel along Y)
    grad_y = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    row_grads = np.mean(np.abs(grad_y), axis=1)

    # Search window: typically 25% to 50% from top
    y_start = int(0.25 * h)
    y_end = int(0.50 * h)

    if y_end > y_start:
        peak_offset = int(np.argmax(row_grads[y_start:y_end]))
        detected_y = y_start + peak_offset
        # Bound within reasonable dashcam limits
        return int(np.clip(detected_y, int(0.30 * h), int(0.48 * h)))

    return int(0.35 * h)


def _colour_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract lane markings using strict HLS color bounds.
    Rejects low-saturation grey metallic guardrails.
    """
    hls = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)

    # ── Yellow lanes: H: 15–35, L: 30–204, S: 115–255 ──────────────────────
    yellow_lo = np.array([15, 30, 115], dtype=np.uint8)
    yellow_hi = np.array([35, 204, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, yellow_lo, yellow_hi)

    # ── White lanes: H: 0–180, L: 190–255, S: 0–255 ────────────────────────
    white_lo = np.array([0, 190, 0], dtype=np.uint8)
    white_hi = np.array([180, 255, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hls, white_lo, white_hi)

    # Fallback grayscale high-contrast pass for faint markings
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, high_white = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

    combined = cv2.bitwise_or(white_mask, yellow_mask)
    combined = cv2.bitwise_or(combined, high_white)

    # Morphological opening to purge small isolated noise specs
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    return cleaned


def _roi_mask(h: int, w: int, horizon_y: int = 126) -> np.ndarray:
    """
    Trapezoidal Region of Interest strictly constrained below the horizon.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    top_y = max(horizon_y, int(0.32 * h))

    pts = np.array([[
        (int(0.06 * w), h),
        (int(0.42 * w), top_y),
        (int(0.58 * w), top_y),
        (int(0.94 * w), h),
    ]], dtype=np.int32)

    cv2.fillPoly(mask, pts, 255)
    return mask


def _separate_lines(
    lines: Optional[np.ndarray],
    w: int,
    h: int,
    horizon_y: int,
) -> Tuple[list, list]:
    """
    Split Hough lines into left and right lane candidates with guardrail suppression.

    Filters applied:
    - Rejects near-vertical edges (|slope| > 2.0: barrier posts, signs, columns).
    - Rejects near-horizontal edges (|slope| < 0.30: crosswalks, shadows).
    - Left lines must have negative slope and bottom intercept in [0.05*w, 0.50*w].
    - Right lines must have positive slope and bottom intercept in [0.50*w, 0.95*w].
    """
    mid_x = w / 2.0
    left_pts: list[Tuple[int, int]] = []
    right_pts: list[Tuple[int, int]] = []

    if lines is None:
        return left_pts, right_pts

    for line in lines:
        seg = line.flatten()
        if len(seg) != 4:
            continue
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        if x2 == x1:
            continue  # Perfectly vertical: guardrail post / column

        slope = (y2 - y1) / float(x2 - x1)

        # ── Guardrail / Noise slope rejection ──────────────────────────────
        if abs(slope) < 0.30 or abs(slope) > 2.0:
            continue

        # Project line to image bottom (y = h) to verify road envelope intercept
        bottom_intercept_x = x1 + (h - y1) / slope

        if slope < 0:  # Left lane candidate
            if 0.05 * w <= bottom_intercept_x <= 0.50 * w and max(x1, x2) < mid_x + 30:
                left_pts.extend([(x1, y1), (x2, y2)])
        elif slope > 0:  # Right lane candidate
            if 0.50 * w <= bottom_intercept_x <= 0.95 * w and min(x1, x2) > mid_x - 30:
                right_pts.extend([(x1, y1), (x2, y2)])

    return left_pts, right_pts


def _fit_and_rasterise(
    pts: list,
    h: int,
    w: int,
    horizon_y: int,
    side: str,
    line_thickness: int = 6,
) -> Tuple[Optional[np.ndarray], float, Optional[float]]:
    """
    Fit 2nd-degree polynomial x = a*y^2 + b*y + c and rasterise to binary mask.

    Returns
    -------
    mask : uint8 binary mask or None
    confidence : float in [0.0, 1.0]
    bottom_intercept_x : float or None
    """
    if len(pts) < 4:
        return None, 0.0, None

    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)

    deg = 2 if len(pts) >= 6 else 1
    try:
        coeffs = np.polyfit(ys, xs, deg=deg)
    except (np.linalg.LinAlgError, ValueError):
        return None, 0.0, None

    y_top = max(horizon_y, int(0.35 * h))
    y_bottom = h - 1
    y_vals = np.linspace(y_top, y_bottom, num=50, dtype=np.float32)
    x_vals = np.polyval(coeffs, y_vals)

    # Valid points check
    valid = (x_vals >= 0) & (x_vals < w)
    if valid.sum() < 3:
        return None, 0.0, None

    y_vals_clipped = y_vals[valid].astype(np.int32)
    x_vals_clipped = x_vals[valid].astype(np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    pts_arr = np.column_stack([x_vals_clipped, y_vals_clipped])
    cv2.polylines(mask, [pts_arr], isClosed=False, color=255, thickness=line_thickness)

    # Intercept at image bottom (y = H - 1)
    bottom_intercept_x = float(np.polyval(coeffs, h - 1))

    # Normalized confidence heuristic
    conf = min(len(pts) / 45.0, 0.98)
    return mask, float(conf), bottom_intercept_x
