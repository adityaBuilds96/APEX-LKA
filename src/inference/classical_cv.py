"""
src/inference/classical_cv.py
==============================
Classical computer-vision lane detection baseline using OpenCV.

!! IMPORTANT LABEL !!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CLASSICAL CV BASELINE — NOT THE FINAL ML MODEL
  Results from this module are produced by rule-based image
  processing, NOT by a trained neural network.
  Use it to verify the pipeline works end-to-end before the
  ML model is trained.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Algorithm
---------
1. Convert image to HLS colour space
2. Extract white + yellow lane markings via colour thresholding
3. Apply ROI (trapezoid crop — road area only)
4. Canny edge detection
5. Probabilistic Hough line transform
6. Separate lines into left/right groups by slope sign
7. Fit first-degree polynomials to each group
8. Rasterise fitted lines back to binary masks

Confidence is the ratio of the number of edge pixels found
in a lane region vs the expected maximum — a rough heuristic,
NOT a probability from a learned model.
"""

import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.predictor import (
    BaseLanePredictor,
    DetectionStatus,
    LanePrediction,
    ModelStatus,
)
from src.inference.preprocessing import PreprocessResult


# ═══════════════════════════════════════════════════════════════════════════
# Classical CV Predictor
# ═══════════════════════════════════════════════════════════════════════════

class ClassicalCVPredictor(BaseLanePredictor):
    """
    Always available — no training required.

    Suitable for:
      * Verifying the complete inference → geometry → visualization pipeline
      * Generating plausible lane overlays on clear road images

    NOT suitable for:
      * Poor lighting, rain, faded markings, complex urban scenes
      * Any safety-critical evaluation
    """

    LABEL = "CLASSICAL CV BASELINE — NOT THE FINAL ML MODEL"

    @property
    def model_status(self) -> ModelStatus:
        return ModelStatus.CLASSICAL_CV

    @property
    def backend_name(self) -> str:
        return "Classical CV (OpenCV)"

    def predict(self, preprocessed: PreprocessResult) -> LanePrediction:
        """Run the classical CV pipeline. Never raises — errors returned as LanePrediction."""
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
            img = preprocessed.resized_bgr   # uint8 BGR at model resolution

            # ── 1. Colour thresholding → combined lane mask ────────────────
            colour_mask = _colour_mask(img)

            # ── 2. ROI ─────────────────────────────────────────────────────
            roi_mask = _roi_mask(h, w)
            masked   = cv2.bitwise_and(colour_mask, roi_mask)

            # ── 3. Canny edges ─────────────────────────────────────────────
            edges = cv2.Canny(masked, 50, 150)

            # ── 4. Hough lines ─────────────────────────────────────────────
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=30, minLineLength=20, maxLineGap=100,
            )

            # ── 5. Separate into left / right ──────────────────────────────
            left_pts, right_pts = _separate_lines(lines, w, h)

            # ── 6. Fit polynomials → rasterise to masks ────────────────────
            left_mask,  left_conf  = _fit_and_rasterise(left_pts,  h, w, side="left")
            right_mask, right_conf = _fit_and_rasterise(right_pts, h, w, side="right")

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
                status           = det_status,
                model_status     = ModelStatus.CLASSICAL_CV,
                backend_name     = self.backend_name,
                left_mask        = left_mask  if left_mask  is not None else np.zeros((h, w), np.uint8),
                right_mask       = right_mask if right_mask is not None else np.zeros((h, w), np.uint8),
                left_confidence  = left_conf,
                right_confidence = right_conf,
                model_h          = h,
                model_w          = w,
                inference_ms     = elapsed_ms,
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
# Internal CV functions
# ═══════════════════════════════════════════════════════════════════════════

def _colour_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    Extract white and yellow pixels — typical lane marking colours.
    Returns a single-channel uint8 binary mask.
    """
    # ── White lanes ────────────────────────────────────────────────────────
    gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, white_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # ── Yellow lanes ──────────────────────────────────────────────────────
    hls  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HLS)
    yellow_lo = np.array([15,  38,  90], dtype=np.uint8)
    yellow_hi = np.array([35, 204, 255], dtype=np.uint8)
    yellow_mask = cv2.inRange(hls, yellow_lo, yellow_hi)

    combined = cv2.bitwise_or(white_mask, yellow_mask)

    # Slight blur to connect nearby fragments
    combined = cv2.GaussianBlur(combined, (5, 5), 0)
    _, combined = cv2.threshold(combined, 1, 255, cv2.THRESH_BINARY)
    return combined


def _roi_mask(h: int, w: int) -> np.ndarray:
    """
    Trapezoid ROI that focuses on the road area.

    The trapezoid is defined as a fraction of image dimensions:
      top_y    = 35% from top  (below horizon)
      top_x    = 40–60% of width (lane ahead narrows)
      bottom_y = 98% of height
      bottom_x = 0–100% of width

    These values work well for a dashcam-style forward-facing camera.
    Adjust in configs/project_config.yaml if needed.
    """
    mask = np.zeros((h, w), dtype=np.uint8)
    roi_y_start = 0.35
    pts = np.array([[
        (int(0.10 * w), h),
        (int(0.42 * w), int(roi_y_start * h)),
        (int(0.58 * w), int(roi_y_start * h)),
        (int(0.90 * w), h),
    ]], dtype=np.int32)
    cv2.fillPoly(mask, pts, 255)
    return mask


def _separate_lines(
    lines: Optional[np.ndarray],
    w: int,
    h: int,
) -> Tuple[list, list]:
    """
    Split Hough lines into left-lane and right-lane groups.

    Left  lane: negative slope (line goes up-right), x < midpoint
    Right lane: positive slope (line goes up-left),  x > midpoint

    Lines with near-zero slope (|slope| < 0.3) are noise → discarded.

    Compatible with OpenCV 4.x (shape N,1,4) and OpenCV 5.x (shape N,4).
    """
    mid_x = w / 2
    left_pts:  list[Tuple[int, int]] = []
    right_pts: list[Tuple[int, int]] = []

    if lines is None:
        return left_pts, right_pts

    for line in lines:
        # OpenCV 5.x: line shape is (4,); OpenCV 4.x: (1, 4)
        seg = line.flatten()
        if len(seg) != 4:
            continue
        x1, y1, x2, y2 = int(seg[0]), int(seg[1]), int(seg[2]), int(seg[3])
        if x2 == x1:
            continue   # vertical line — skip
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue   # near-horizontal — noise

        if slope < 0 and x1 < mid_x and x2 < mid_x:
            left_pts.extend([(x1, y1), (x2, y2)])
        elif slope > 0 and x1 > mid_x and x2 > mid_x:
            right_pts.extend([(x1, y1), (x2, y2)])

    return left_pts, right_pts


def _fit_and_rasterise(
    pts: list,
    h: int,
    w: int,
    side: str,
    line_thickness: int = 8,
) -> Tuple[Optional[np.ndarray], float]:
    """
    Fit a 1st-degree polynomial to (x, y) points and draw it on a mask.

    Returns
    -------
    mask : uint8 ndarray (h, w)  or None if not enough points
    confidence : float [0, 1]
    """
    MIN_POINTS = 4
    if len(pts) < MIN_POINTS:
        return None, 0.0

    xs = np.array([p[0] for p in pts], dtype=np.float32)
    ys = np.array([p[1] for p in pts], dtype=np.float32)

    # Fit x = a*y + b  (more stable for near-vertical lines)
    try:
        coeffs = np.polyfit(ys, xs, deg=1)
    except np.linalg.LinAlgError:
        return None, 0.0

    # Evaluate from horizon (35% down) to bottom of image
    y_bottom = h
    y_top    = int(0.35 * h)
    y_vals   = np.linspace(y_top, y_bottom, num=40, dtype=np.float32)
    x_vals   = np.polyval(coeffs, y_vals)

    # Clip to valid image bounds
    valid = (x_vals >= 0) & (x_vals < w)
    if valid.sum() < 2:
        return None, 0.0

    y_vals = y_vals[valid].astype(np.int32)
    x_vals = x_vals[valid].astype(np.int32)

    # Rasterise onto mask
    mask = np.zeros((h, w), dtype=np.uint8)
    pts_arr = np.column_stack([x_vals, y_vals])
    cv2.polylines(mask, [pts_arr], isClosed=False,
                  color=255, thickness=line_thickness)

    # Confidence heuristic: ratio of points found vs expected max
    # Normalised to [0, 1]; capped at 0.99 (we don't claim certainty)
    conf = min(len(pts) / 60.0, 0.99)
    return mask, float(conf)
