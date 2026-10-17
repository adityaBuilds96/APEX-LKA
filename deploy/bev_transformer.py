"""
deploy/bev_transformer.py
=========================
Metric Bird's Eye View (BEV) Inverse Perspective Mapping (IPM) Engine:
- Transforms perspective image into high-resolution metric top-down plane
- Provides bidirectional coordinate mapping: Image Pixel <-> BEV Pixel <-> Metric (X_m, Z_m)
- Implements RANSAC + Weighted Least Squares (WLS) polynomial curve fitting in metric space (X_m = f(Z_m))
- Hardened with homography singularity rejection and numerical NaN/Inf guards
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
import warnings
if hasattr(np, 'exceptions') and hasattr(np.exceptions, 'RankWarning'):
    warnings.simplefilter("ignore", np.exceptions.RankWarning)
elif hasattr(np, 'RankWarning'):
    warnings.simplefilter("ignore", getattr(np, 'RankWarning'))



DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

log = logging.getLogger("bev_transformer")


class BEVTransformer:
    """
    Bidirectional Bird's Eye View (BEV) metric space transformer.
    Vehicle frame:
        X: lateral [m], positive right, negative left, 0 = vehicle centerline
        Z: longitudinal [m], positive forward from front bumper/axle
    """

    def __init__(self, config: Optional[Union[str, Path, Dict[str, Any]]] = None) -> None:
        self.config_data = self._load_config(config)

        self.x_range_m: Tuple[float, float] = tuple(self.config_data.get("x_range_m", [-5.0, 5.0]))
        self.z_range_m: Tuple[float, float] = tuple(self.config_data.get("z_range_m", [1.0, 20.0]))
        self.meters_per_pixel: float = float(self.config_data.get("meters_per_pixel", 0.025))

        self.nominal_track_width_m: float = float(self.config_data.get("nominal_track_width_m", 1.80))
        self.track_width_range_m: Tuple[float, float] = tuple(
            self.config_data.get("track_width_range_m", [1.0, 4.5])
        )

        # Compute BEV grid dimensions
        dx_m = self.x_range_m[1] - self.x_range_m[0]
        dz_m = self.z_range_m[1] - self.z_range_m[0]
        self.bev_width_px = int(round(dx_m / self.meters_per_pixel))
        self.bev_height_px = int(round(dz_m / self.meters_per_pixel))

        # Ground and image calibration points
        self.src_points = np.array(self.config_data.get("src_points", [
            [120.0, 350.0],
            [520.0, 350.0],
            [395.0, 190.0],
            [245.0, 190.0],
        ]), dtype=np.float32)

        self.ground_points_m = np.array(self.config_data.get("ground_points_m", [
            [-1.75, 2.0],
            [ 1.75, 2.0],
            [ 1.75, 14.0],
            [-1.75, 14.0],
        ]), dtype=np.float32)

        # Compute Homography
        self.H, self.H_inv = self._compute_homography()

    def _load_config(self, config: Optional[Union[str, Path, Dict[str, Any]]]) -> Dict[str, Any]:
        if isinstance(config, dict):
            return config
        if config is not None:
            p = Path(config)
            if not p.is_absolute():
                p = PROJECT_ROOT / p
            if p.exists():
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}

        # Default fallback
        default_p = PROJECT_ROOT / "configs" / "bev.yaml"
        if default_p.exists():
            with open(default_p, "r") as f:
                return yaml.safe_load(f) or {}

        return {}

    def _compute_homography(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute forward and inverse homography matrices with singularity checks."""
        # Convert ground points (X_m, Z_m) to BEV pixel coordinates (u, v)
        dst_pts = np.zeros_like(self.ground_points_m, dtype=np.float32)
        for i in range(len(self.ground_points_m)):
            xm, zm = self.ground_points_m[i]
            u, v = self.metric_to_pixel(xm, zm)
            dst_pts[i] = [u, v]

        H = cv2.getPerspectiveTransform(self.src_points, dst_pts)
        det_H = float(np.linalg.det(H))

        if abs(det_H) < 1e-7:
            raise ValueError(f"Singular homography matrix encountered (det = {det_H:.2e})")

        H_inv = np.linalg.inv(H)
        return H, H_inv

    # ═══════════════════════════════════════════════════════════════════════════
    # Coordinate Conversion API
    # ═══════════════════════════════════════════════════════════════════════════

    def pixel_to_metric(
        self, x_px: Union[float, np.ndarray], y_px: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert BEV pixel coordinates to metric vehicle coordinates (X_m, Z_m).
        x_px: Column index [0, bev_width_px - 1]
        y_px: Row index [0, bev_height_px - 1] (row 0 = z_max, row bottom = z_min)
        """
        X_m = self.x_range_m[0] + x_px * self.meters_per_pixel
        Z_m = self.z_range_m[1] - y_px * self.meters_per_pixel
        return X_m, Z_m

    def metric_to_pixel(
        self, X_m: Union[float, np.ndarray], Z_m: Union[float, np.ndarray]
    ) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
        """
        Convert metric vehicle coordinates (X_m, Z_m) to BEV pixel coordinates (x_px, y_px).
        """
        x_px = (X_m - self.x_range_m[0]) / self.meters_per_pixel
        y_px = (self.z_range_m[1] - Z_m) / self.meters_per_pixel
        return x_px, y_px

    def image_to_bev_points(self, pts_img: np.ndarray) -> np.ndarray:
        """
        Transform (N, 2) pixel points from camera perspective to BEV pixel plane.
        """
        if len(pts_img) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_in = np.asarray(pts_img, dtype=np.float32).reshape(-1, 1, 2)
        pts_warped = cv2.perspectiveTransform(pts_in, self.H).reshape(-1, 2)
        return pts_warped

    def bev_to_image_points(self, pts_bev: np.ndarray) -> np.ndarray:
        """
        Transform (N, 2) pixel points from BEV plane back to camera perspective.
        """
        if len(pts_bev) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_in = np.asarray(pts_bev, dtype=np.float32).reshape(-1, 1, 2)
        pts_unwarped = cv2.perspectiveTransform(pts_in, self.H_inv).reshape(-1, 2)
        return pts_unwarped

    def unwarp_points(self, pts_bev: np.ndarray) -> np.ndarray:
        """Alias for bev_to_image_points satisfying Functional Requirement 3."""
        return self.bev_to_image_points(pts_bev)

    def image_to_metric(self, pts_img: np.ndarray) -> np.ndarray:
        """
        Transform (N, 2) perspective image pixels directly to metric (N, 2) coordinates (X_m, Z_m).
        """
        pts_bev = self.image_to_bev_points(pts_img)
        if len(pts_bev) == 0:
            return np.empty((0, 2), dtype=np.float32)
        X_m, Z_m = self.pixel_to_metric(pts_bev[:, 0], pts_bev[:, 1])
        return np.column_stack((X_m, Z_m)).astype(np.float32)

    def metric_to_image(self, pts_m: np.ndarray) -> np.ndarray:
        """
        Transform (N, 2) metric (X_m, Z_m) points back to perspective image pixels.
        """
        if len(pts_m) == 0:
            return np.empty((0, 2), dtype=np.float32)
        pts_m = np.asarray(pts_m, dtype=np.float32)
        u, v = self.metric_to_pixel(pts_m[:, 0], pts_m[:, 1])
        pts_bev = np.column_stack((u, v)).astype(np.float32)
        return self.bev_to_image_points(pts_bev)

    # ═══════════════════════════════════════════════════════════════════════════
    # Image Warping API
    # ═══════════════════════════════════════════════════════════════════════════

    def warp(self, image: np.ndarray, flags: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Warp camera perspective image into top-down metric BEV view.
        Output dimensions: (bev_height_px, bev_width_px).
        """
        warped = cv2.warpPerspective(
            image,
            self.H,
            (self.bev_width_px, self.bev_height_px),
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.nan_to_num(warped, nan=0.0, posinf=0.0, neginf=0.0)

    def unwarp(
        self, bev_image: np.ndarray, orig_size: Tuple[int, int], flags: int = cv2.INTER_LINEAR
    ) -> np.ndarray:
        """
        Unwarp BEV image back to original perspective size (width, height).
        """
        unwarped = cv2.warpPerspective(
            bev_image,
            self.H_inv,
            orig_size,
            flags=flags,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return np.nan_to_num(unwarped, nan=0.0, posinf=0.0, neginf=0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Metric Polynomial Curve Fitting (RANSAC + Weighted Least Squares)
    # ═══════════════════════════════════════════════════════════════════════════

    def fit_metric_polynomial(
        self,
        points_m: np.ndarray,
        weights: Optional[np.ndarray] = None,
        degree: int = 2,
        ransac: bool = True,
        ransac_iters: int = 50,
        inlier_tol_m: float = 0.08,
    ) -> Optional[np.ndarray]:
        """
        Fit lateral position as a function of longitudinal distance: X_m = f(Z_m).
        Returns polynomial coefficients [a, b, c] (highest power first) or None.
        """
        if len(points_m) < degree + 1:
            return None

        # Guard against dense point clouds (caps RANSAC iterations to sub-millisecond execution)
        if len(points_m) > 200:
            idx_sub = np.linspace(0, len(points_m) - 1, 200, dtype=int)
            points_m = points_m[idx_sub]
            if weights is not None and len(weights) > 200:
                weights = weights[idx_sub]

        Z = points_m[:, 1]  # Independent longitudinal coordinate
        X = points_m[:, 0]  # Dependent lateral coordinate
        N = len(Z)


        w = np.ones(N, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
        w = np.clip(w, 1e-4, 1.0)

        inlier_mask = np.ones(N, dtype=bool)

        if ransac and N >= 6:
            best_inliers = 0
            best_mask = inlier_mask
            rng = np.random.default_rng(42)

            for _ in range(ransac_iters):
                idx = rng.choice(N, size=degree + 1, replace=False)
                try:
                    p = np.polyfit(Z[idx], X[idx], degree)
                    res = np.abs(X - np.polyval(p, Z))
                    curr_mask = res < inlier_tol_m
                    inlier_cnt = int(np.sum(curr_mask))
                    if inlier_cnt > best_inliers:
                        best_inliers = inlier_cnt
                        best_mask = curr_mask
                except Exception:
                    continue

            if best_inliers >= degree + 1:
                inlier_mask = best_mask

        # Weighted Least Squares on inliers
        Z_in = Z[inlier_mask]
        X_in = X[inlier_mask]
        w_in = w[inlier_mask]

        if len(Z_in) < degree + 1:
            return None

        try:
            # Weighted polynomial fit: minimize sum(w_i * (X_i - f(Z_i))^2)
            coeffs = np.polyfit(Z_in, X_in, degree, w=w_in)
            return coeffs
        except Exception as e:
            log.debug("BEV polynomial fit failed: %s", e)
            return None


def update_bev_scaling(
    config_path: Path,
    measured_width_m: float,
    ground_truth_width_m: float,
    tolerance_pct: float = 3.0,
) -> Tuple[bool, float]:
    """
    Auto-tunes meters_per_pixel if the measured BEV track width differs from ground truth by > 3%.
    """
    if ground_truth_width_m <= 0.0 or measured_width_m <= 0.0:
        return False, 0.0

    err_pct = abs(measured_width_m - ground_truth_width_m) / ground_truth_width_m * 100.0
    if err_pct <= tolerance_pct:
        log.info("BEV track width error %.2f%% is within acceptable tolerance (<= %.1f%%).", err_pct, tolerance_pct)
        return False, err_pct

    log.warning("BEV track width error %.2f%% exceeds %.1f%% tolerance. Auto-tuning meters_per_pixel...", err_pct, tolerance_pct)

    with open(config_path, "r") as f:
        data = yaml.safe_load(f) or {}

    current_res = float(data.get("meters_per_pixel", 0.025))
    scale_correction = ground_truth_width_m / measured_width_m
    new_res = round(current_res * scale_correction, 5)

    data["meters_per_pixel"] = new_res
    with open(config_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)

    log.info("Updated meters_per_pixel from %.5f to %.5f", current_res, new_res)
    return True, err_pct
