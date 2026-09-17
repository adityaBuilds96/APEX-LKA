"""
src/lane_geometry/lane_estimator.py
=====================================
Fit polynomial curves to lane point arrays and estimate lane geometry.

Inputs
------
PostprocessedLanes (left_pts, right_pts) in model-space coordinates.

Outputs
-------
LaneGeometry — curves, bottom x-positions, lane center, validity flags.

Coordinate system (EXPLICIT)
-----------------------------
  Origin: top-left corner of image
  X: increases rightward  (pixel columns)
  Y: increases downward   (pixel rows)

  Left  lane x < image_center_x (typically)
  Right lane x > image_center_x (typically)

  lane_center_x = (x_left_bottom + x_right_bottom) / 2

Curve fitting
-------------
We fit   x = f(y)   i.e. x as a function of y.
This is more stable than y = f(x) for near-vertical lane lines.
Polynomial degree: configurable (default 2 = quadratic = handles curves).
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import cfg
from src.inference.postprocessing import PostprocessedLanes


# Polynomial degree from config
_POLY_DEG = cfg["lane_geometry"]["poly_degree"]          # default: 2
_MIN_PTS  = cfg["lane_geometry"]["min_lane_pixels"]      # default: 50


# ═══════════════════════════════════════════════════════════════════════════
# Result dataclass
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LaneGeometry:
    """
    Geometric description of the detected lanes in model-space pixels.

    Polynomial coefficients
    -----------------------
    left_poly, right_poly : np.ndarray | None
        Coefficients for  x = poly(y)  (np.polyfit convention, highest power first).
        None if that lane was not fitted.

    Bottom-of-image x positions (most important for offset)
    --------------------------------------------------------
    x_left_bottom, x_right_bottom : float | None
        x coordinate of each lane at the BOTTOM of the image (y = model_h).
        This is where the car is relative to the lanes — used for offset calc.

    Lane center
    -----------
    lane_center_x : float | None
        (x_left_bottom + x_right_bottom) / 2
        None if at least one lane is not detected.

    one_lane_estimated : bool
        True if lane_center was estimated from one visible lane + lane-width assumption.
        Mark this clearly in the UI.

    curvature_radius : float | None
        Approximate radius of curvature (metres) at y=model_h.
        None if quadratic fit is not available or lane is straight.
        Note: pixel-to-metre conversion is approximate without camera calibration.

    Validity
    --------
    has_left, has_right : bool
    geometry_valid : bool — True if at least one lane was fitted
    """
    # Polynomial coefficients
    left_poly:          Optional[np.ndarray] = None
    right_poly:         Optional[np.ndarray] = None

    # Bottom-of-image x positions
    x_left_bottom:      Optional[float] = None
    x_right_bottom:     Optional[float] = None

    # Lane center
    lane_center_x:      Optional[float] = None
    one_lane_estimated: bool = False

    # Curvature
    curvature_left_m:  Optional[float] = None
    curvature_right_m: Optional[float] = None

    # Model dimensions
    model_h:            int = 360
    model_w:            int = 640

    # Timing
    estimate_ms:        float = 0.0

    # Error
    error:              Optional[str] = None

    @property
    def has_left(self) -> bool:
        return self.left_poly is not None

    @property
    def has_right(self) -> bool:
        return self.right_poly is not None

    @property
    def geometry_valid(self) -> bool:
        return self.has_left or self.has_right


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

# Assumed lane width in pixels at model resolution.
# Used ONLY when estimating the missing lane from one visible lane.
# 640 wide image: typical highway lane = ~200 px at bottom of frame.
_ASSUMED_LANE_WIDTH_PX = 200


def estimate_geometry(
    lanes: PostprocessedLanes,
    model_h: int,
    model_w: int,
) -> LaneGeometry:
    """
    Fit polynomials to lane point arrays and compute geometry.

    Parameters
    ----------
    lanes : PostprocessedLanes
    model_h, model_w : int — image dimensions in model space

    Returns
    -------
    LaneGeometry
    """
    t0 = time.perf_counter()

    # ── Fit left lane ──────────────────────────────────────────────────────
    left_poly  = _fit_poly(lanes.left_pts)
    right_poly = _fit_poly(lanes.right_pts)

    # ── Evaluate at bottom of image ────────────────────────────────────────
    y_bottom = float(model_h)
    x_left   = float(np.polyval(left_poly,  y_bottom)) if left_poly  is not None else None
    x_right  = float(np.polyval(right_poly, y_bottom)) if right_poly is not None else None

    # ── Lane center estimation ─────────────────────────────────────────────
    one_estimated = False
    if x_left is not None and x_right is not None:
        lane_center = (x_left + x_right) / 2.0
    elif x_left is not None:
        # Only left visible: estimate right from lane width
        lane_center   = x_left + _ASSUMED_LANE_WIDTH_PX / 2.0
        x_right       = x_left + _ASSUMED_LANE_WIDTH_PX
        one_estimated = True
    elif x_right is not None:
        # Only right visible: estimate left from lane width
        lane_center   = x_right - _ASSUMED_LANE_WIDTH_PX / 2.0
        x_left        = x_right - _ASSUMED_LANE_WIDTH_PX
        one_estimated = True
    else:
        lane_center = None

    # ── Curvature (pixels → approximate metres) ────────────────────────────
    # Without calibration, pixel-to-metre is ~0.003 m/px (rough).
    # Only compute for quadratic fit (degree >= 2).
    curv_left  = _curvature_radius(left_poly,  y_bottom) if left_poly  is not None else None
    curv_right = _curvature_radius(right_poly, y_bottom) if right_poly is not None else None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return LaneGeometry(
        left_poly          = left_poly,
        right_poly         = right_poly,
        x_left_bottom      = x_left,
        x_right_bottom     = x_right,
        lane_center_x      = lane_center,
        one_lane_estimated = one_estimated,
        curvature_left_m   = curv_left,
        curvature_right_m  = curv_right,
        model_h            = model_h,
        model_w            = model_w,
        estimate_ms        = elapsed_ms,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _fit_poly(pts: np.ndarray) -> Optional[np.ndarray]:
    """
    Fit x = f(y) polynomial to (x, y) point array.

    Returns polynomial coefficients (highest power first) or None if
    insufficient points.
    """
    if pts is None or len(pts) < _MIN_PTS:
        return None
    ys = pts[:, 1].astype(np.float32)
    xs = pts[:, 0].astype(np.float32)
    try:
        coeffs = np.polyfit(ys, xs, deg=_POLY_DEG)
        return coeffs
    except (np.linalg.LinAlgError, ValueError):
        return None


def _curvature_radius(
    poly: np.ndarray,
    y_eval: float,
    px_per_m: float = 333.0,   # rough: 1 m ≈ 333 px at model resolution
) -> Optional[float]:
    """
    Approximate radius of curvature at y=y_eval in metres.

    For x = a*y^2 + b*y + c  (quadratic),
    R = (1 + (dx/dy)^2)^(3/2) / |d^2x/dy^2|

    Only meaningful for degree-2 fits.
    """
    if poly is None or len(poly) < 3:
        return None
    a = poly[0]   # y^2 coefficient
    b = poly[1]   # y^1 coefficient
    if abs(a) < 1e-7:
        return None   # effectively straight → infinite radius

    dx_dy   =  2 * a * y_eval + b
    d2x_dy2 =  2 * a

    try:
        R_px = ((1 + dx_dy**2) ** 1.5) / abs(d2x_dy2)
        R_m  = R_px / px_per_m
        return round(float(R_m), 1)
    except (ZeroDivisionError, OverflowError):
        return None


def eval_poly_y_range(
    poly: np.ndarray,
    y_top: int,
    y_bottom: int,
    num_points: int = 40,
) -> np.ndarray:
    """
    Evaluate a lane polynomial over a y range.

    Returns ndarray of shape (N, 2) with (x, y) columns, integer, clipped
    to be used for drawing.
    """
    y_vals = np.linspace(y_top, y_bottom, num=num_points)
    x_vals = np.polyval(poly, y_vals)
    pts    = np.column_stack([x_vals, y_vals]).astype(np.int32)
    return pts
