"""
src/inference/postprocessing.py
================================
Post-process raw model/CV output into clean, structured lane data.

At this stage "post-processing" means:
  1. Morphological cleanup of binary masks (remove noise, fill gaps)
  2. Extracting ordered point arrays from masks
  3. Computing mask-level quality metrics

This module is the bridge between LanePrediction (raw masks) and
lane_geometry (fitted curves and geometric measurements).

Usage
-----
    from src.inference.postprocessing import clean_masks, extract_lane_points

    clean_left, clean_right = clean_masks(prediction)
    left_pts, right_pts = extract_lane_points(clean_left, clean_right)
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.inference.predictor import LanePrediction


# ═══════════════════════════════════════════════════════════════════════════
# Result type
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class PostprocessedLanes:
    """
    Cleaned and structured lane data from raw masks.

    left_pts, right_pts : np.ndarray of shape (N, 2), dtype int
        Ordered (x, y) pixel coordinates in MODEL-space.
        Ordered top-to-bottom (increasing y).
        Empty array (shape (0, 2)) if that lane is not detected.

    clean_left_mask, clean_right_mask : np.ndarray uint8
        Morphologically cleaned binary masks (255 = lane).

    postprocess_ms : float
    """
    left_pts:          np.ndarray
    right_pts:         np.ndarray
    clean_left_mask:   np.ndarray
    clean_right_mask:  np.ndarray
    postprocess_ms:    float = 0.0

    @property
    def has_left(self) -> bool:
        return len(self.left_pts) > 0

    @property
    def has_right(self) -> bool:
        return len(self.right_pts) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════

def postprocess(prediction: LanePrediction) -> PostprocessedLanes:
    """
    Full postprocessing pipeline.

    1. Morphological clean-up of both masks
    2. Extract ordered (x, y) point arrays

    Parameters
    ----------
    prediction : LanePrediction
        Raw output from any BaseLanePredictor.

    Returns
    -------
    PostprocessedLanes
    """
    t0 = time.perf_counter()

    h, w = prediction.model_h, prediction.model_w
    empty = np.empty((0, 2), dtype=np.int32)

    # ── Clean masks ────────────────────────────────────────────────────────
    left_clean  = _clean_mask(prediction.left_mask,  h, w)
    right_clean = _clean_mask(prediction.right_mask, h, w)

    # ── Extract ordered point arrays ───────────────────────────────────────
    left_pts  = _mask_to_points(left_clean)
    right_pts = _mask_to_points(right_clean)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    return PostprocessedLanes(
        left_pts         = left_pts,
        right_pts        = right_pts,
        clean_left_mask  = left_clean,
        clean_right_mask = right_clean,
        postprocess_ms   = elapsed_ms,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _clean_mask(
    mask: Optional[np.ndarray],
    h: int,
    w: int,
) -> np.ndarray:
    """
    Morphological clean-up of a binary lane mask.

    Operations (in order):
    1. Ensure correct shape and dtype
    2. Closing  — fill small gaps in lane markings
    3. Opening  — remove tiny isolated noise blobs
    4. Connected-component filtering — keep only the largest component

    Returns uint8 mask of shape (h, w).
    """
    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    # Ensure correct shape
    if mask.shape != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    m = mask.copy()
    if m.dtype != np.uint8:
        m = (m > 0).astype(np.uint8) * 255

    if not m.any():
        return m

    # ── Morphological operations ───────────────────────────────────────────
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 15))
    kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 5))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel_close)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel_open)

    # ── Keep only largest connected component ──────────────────────────────
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num_labels <= 1:
        return m

    # Component 0 is always background; skip it
    component_areas = stats[1:, cv2.CC_STAT_AREA]
    if len(component_areas) == 0:
        return np.zeros_like(m)

    largest_idx = int(np.argmax(component_areas)) + 1  # +1 because we skipped 0
    m = ((labels == largest_idx).astype(np.uint8)) * 255

    return m


def _mask_to_points(mask: np.ndarray) -> np.ndarray:
    """
    Convert a binary mask to an ordered array of (x, y) points.

    Strategy: for each row that has mask pixels, take the horizontal
    centroid as the representative x coordinate.  This gives one
    (x, y) point per row, ordered top-to-bottom.

    Returns ndarray shape (N, 2) int32, or (0, 2) if mask is empty.
    """
    if mask is None or not mask.any():
        return np.empty((0, 2), dtype=np.int32)

    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return np.empty((0, 2), dtype=np.int32)

    # Group by row, compute centroid x
    rows   = np.unique(ys)
    result = []
    for y in rows:
        x_coords = xs[ys == y]
        x_center = int(np.round(x_coords.mean()))
        result.append((x_center, int(y)))

    pts = np.array(result, dtype=np.int32)
    # Sort top-to-bottom (increasing y)
    order = np.argsort(pts[:, 1])
    return pts[order]
