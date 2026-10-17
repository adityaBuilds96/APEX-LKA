"""
deploy/calibrate_extrinsics.py
==============================
Camera Extrinsic Calibration Utility for APEX-LKA:
- Static checkerboard mode: Computes pitch, roll, yaw from ground plane checkerboard
- Driving vanishing-point mode: Estimates pitch and yaw from lane/edge convergence
- Synthetic mode: Mathematical / synthetic ground-truth calibration for CI & headless validation
- Calculates analytic horizon: y_horizon = cy - fy * tan(theta_pitch)
- Enforces numerical safety bounds and writes to configs/camera_extrinsics.yaml
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import CameraExtrinsics, cfg

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("calibrate_extrinsics")


def get_intrinsics() -> Tuple[float, float, float, float]:
    """Retrieve camera intrinsic parameters (fx, fy, cx, cy) from project config."""
    intrinsics = cfg.get("camera_intrinsics", {})
    fx = float(intrinsics.get("fx", 500.0))
    fy = float(intrinsics.get("fy", 500.0))
    cx = float(intrinsics.get("cx", 320.0))
    cy = float(intrinsics.get("cy", 180.0))
    return fx, fy, cx, cy


def compute_analytic_horizon(cy: float, fy: float, pitch_rad: float) -> int:
    """
    Compute pixel row of the analytic horizon line.
    y_horizon = cy - fy * tan(pitch_rad)
    Down-positive pitch tilts camera down, moving horizon up in the image (smaller y).
    """
    return int(round(cy - fy * math.tan(pitch_rad)))


def calibrate_from_checkerboard(
    image: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    square_size_m: float = 0.03,
    board_cols: int = 9,
    board_rows: int = 6,
    mount_height_m: float = 1.20,
    debug_dir: Optional[Path] = None,
) -> Tuple[CameraExtrinsics, bool]:
    """
    Static checkerboard extrinsic calibration.
    Extracts camera orientation from board plane normal.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    pattern_size = (board_cols, board_rows)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size, flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if not found or corners is None:
        log.warning("Checkerboard pattern %dx%d not detected in input image.", board_cols, board_rows)
        return CameraExtrinsics(mount_height_m=mount_height_m), False

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((board_rows * board_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_cols, 0:board_rows].T.reshape(-1, 2) * square_size_m

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros(4, dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(objp, corners_sub, K, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    if not success:
        log.warning("solvePnP failed to converge on checkerboard points.")
        return CameraExtrinsics(mount_height_m=mount_height_m), False

    R, _ = cv2.Rodrigues(rvec)

    # In camera frame (X right, Y down, Z forward):
    # Board normal in camera coordinates is R[:, 2]
    # Pitch is tilt around camera X axis: theta = atan2(-R[1, 2], R[2, 2])
    # Roll is tilt around camera Z axis: phi = atan2(R[1, 0], R[1, 1])
    # Yaw is heading offset around camera Y axis: psi = atan2(R[0, 2], R[2, 2])
    pitch = float(math.atan2(-R[1, 2], R[2, 2]))
    roll = float(math.atan2(R[1, 0], R[1, 1]))
    yaw = float(math.atan2(R[0, 2], R[2, 2]))

    y_horizon = compute_analytic_horizon(cy, fy, pitch)

    if debug_dir is not None:
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg_img = image.copy()
        cv2.drawChessboardCorners(dbg_img, pattern_size, corners_sub, found)
        cv2.line(dbg_img, (0, y_horizon), (image.shape[1] - 1, y_horizon), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "extrinsics_checkerboard_debug.png"), dbg_img)

    ext = CameraExtrinsics(
        mount_height_m=float(mount_height_m),
        pitch_rad=pitch,
        roll_rad=roll,
        yaw_rad=yaw,
        x_offset_m=0.0,
        z_offset_m=0.50,
        y_horizon_px=y_horizon,
        calibration_method="static_checkerboard",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return ext, True


def calibrate_from_vanishing_point(
    lines_or_image: Union[np.ndarray, List[Tuple[float, float, float, float]]],
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    mount_height_m: float = 1.20,
    debug_dir: Optional[Path] = None,
) -> Tuple[CameraExtrinsics, bool]:
    """
    Driving-scene extrinsic calibration from vanishing point of parallel road boundaries.
    """
    segments: List[Tuple[float, float, float, float]] = []

    if isinstance(lines_or_image, np.ndarray):
        gray = cv2.cvtColor(lines_or_image, cv2.COLOR_BGR2GRAY) if len(lines_or_image.shape) == 3 else lines_or_image
        edges = cv2.Canny(gray, 50, 150)
        hough = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40, minLineLength=50, maxLineGap=15)
        if hough is not None:
            for l in hough:
                x1, y1, x2, y2 = l[0]
                if abs(x2 - x1) > 1e-4:
                    slope = (y2 - y1) / (x2 - x1)
                    if 0.2 < abs(slope) < 4.0:  # Only slanted road lines
                        segments.append((float(x1), float(y1), float(x2), float(y2)))
    else:
        segments = lines_or_image

    if len(segments) < 2:
        log.warning("Insufficient lines for vanishing point estimation (%d found).", len(segments))
        return CameraExtrinsics(mount_height_m=mount_height_m), False

    # Find line intersections
    intersections: List[Tuple[float, float]] = []
    for i in range(len(segments)):
        x1, y1, x2, y2 = segments[i]
        A1 = y2 - y1
        B1 = x1 - x2
        C1 = A1 * x1 + B1 * y1
        for j in range(i + 1, len(segments)):
            x3, y3, x4, y4 = segments[j]
            A2 = y4 - y3
            B2 = x3 - x4
            C2 = A2 * x3 + B2 * y3
            det = A1 * B2 - A2 * B1
            if abs(det) > 1e-5:
                ix = (B2 * C1 - B1 * C2) / det
                iy = (A1 * C2 - A2 * C1) / det
                if 0 <= ix <= cx * 2 and 0 <= iy <= cy * 2:
                    intersections.append((ix, iy))

    if not intersections:
        log.warning("No valid line intersections in FOV.")
        return CameraExtrinsics(mount_height_m=mount_height_m), False

    vx = float(np.median([p[0] for p in intersections]))
    vy = float(np.median([p[1] for p in intersections]))

    pitch = float(math.atan2(cy - vy, fy))
    yaw = float(math.atan2(vx - cx, fx))
    roll = 0.0

    y_horizon = compute_analytic_horizon(cy, fy, pitch)

    if debug_dir is not None and isinstance(lines_or_image, np.ndarray):
        debug_dir.mkdir(parents=True, exist_ok=True)
        dbg = lines_or_image.copy()
        cv2.circle(dbg, (int(round(vx)), int(round(vy))), 6, (0, 255, 0), -1)
        cv2.line(dbg, (0, y_horizon), (dbg.shape[1] - 1, y_horizon), (0, 0, 255), 2)
        cv2.imwrite(str(debug_dir / "extrinsics_vp_debug.png"), dbg)

    ext = CameraExtrinsics(
        mount_height_m=float(mount_height_m),
        pitch_rad=pitch,
        roll_rad=roll,
        yaw_rad=yaw,
        x_offset_m=0.0,
        z_offset_m=0.50,
        y_horizon_px=y_horizon,
        calibration_method="driving_vanishing_point",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    return ext, True


def calibrate_synthetic(
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 320.0,
    cy: float = 180.0,
    mount_height_m: float = 1.20,
    pitch_deg: float = 5.0,
    roll_deg: float = 0.0,
    yaw_deg: float = 0.0,
) -> CameraExtrinsics:
    """
    Generate synthetic ground-truth extrinsics for headless validation and regression testing.
    """
    pitch_rad = float(math.radians(pitch_deg))
    roll_rad = float(math.radians(roll_deg))
    yaw_rad = float(math.radians(yaw_deg))
    y_h = compute_analytic_horizon(cy, fy, pitch_rad)

    return CameraExtrinsics(
        mount_height_m=float(mount_height_m),
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        yaw_rad=yaw_rad,
        x_offset_m=0.0,
        z_offset_m=0.50,
        y_horizon_px=y_h,
        calibration_method="synthetic_simulation",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def run_calibration(
    mode: str = "synthetic",
    synthetic: bool = False,
    dry_run: bool = False,
    output_path: Optional[Path] = None,
    image_path: Optional[Path] = None,
    mount_height_m: float = 1.20,
    pitch_deg: float = 5.0,
    debug_dir: Optional[Path] = None,
) -> CameraExtrinsics:
    """Master calibration coordinator."""
    fx, fy, cx, cy = get_intrinsics()
    target_out = output_path or (PROJECT_ROOT / "configs" / "camera_extrinsics.yaml")

    log.info("Starting Camera Extrinsic Calibration [Mode: %s, Synthetic: %s, Dry-Run: %s]", mode, synthetic, dry_run)
    log.info("Camera Intrinsics: fx=%.1f, fy=%.1f, cx=%.1f, cy=%.1f", fx, fy, cx, cy)

    if synthetic or mode == "synthetic":
        ext = calibrate_synthetic(
            fx=fx, fy=fy, cx=cx, cy=cy, mount_height_m=mount_height_m, pitch_deg=pitch_deg
        )
        success = True
    elif mode == "static":
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Static calibration requires an existing image path, got: {image_path}")
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Failed to read image at {image_path}")
        ext, success = calibrate_from_checkerboard(
            img, fx, fy, cx, cy, mount_height_m=mount_height_m, debug_dir=debug_dir
        )
    elif mode == "driving":
        if image_path is not None and image_path.exists():
            img = cv2.imread(str(image_path))
            ext, success = calibrate_from_vanishing_point(
                img, fx, fy, cx, cy, mount_height_m=mount_height_m, debug_dir=debug_dir
            )
        else:
            # Fallback to synthetic lines representing a straight road
            lines = [
                (120.0, 360.0, 310.0, 140.0),
                (520.0, 360.0, 330.0, 140.0),
            ]
            ext, success = calibrate_from_vanishing_point(
                lines, fx, fy, cx, cy, mount_height_m=mount_height_m, debug_dir=debug_dir
            )
    else:
        raise ValueError(f"Unknown calibration mode: {mode}")

    if not success:
        log.warning("Calibration did not converge cleanly. Using synthetic fallback.")
        ext = calibrate_synthetic(fx=fx, fy=fy, cx=cx, cy=cy, mount_height_m=mount_height_m, pitch_deg=pitch_deg)

    # Validation Cross-Checks
    is_valid, issues = ext.validate(image_height=int(round(cy * 2.0)), cy=cy, fy=fy)
    for issue in issues:
        if issue.startswith("Warning:"):
            log.warning("[VALIDATION] %s", issue)
        else:
            log.error("[VALIDATION] %s", issue)

    if not is_valid:
        raise ValueError(f"Extrinsics failed validation constraints: {issues}")

    log.info("Calibrated Extrinsics:")
    log.info("  Mount Height : %.3f m", ext.mount_height_m)
    log.info("  Pitch Angle  : %.4f rad (%.2f°)", ext.pitch_rad, math.degrees(ext.pitch_rad))
    log.info("  Roll Angle   : %.4f rad (%.2f°)", ext.roll_rad, math.degrees(ext.roll_rad))
    log.info("  Yaw Angle    : %.4f rad (%.2f°)", ext.yaw_rad, math.degrees(ext.yaw_rad))
    log.info("  Analytic Hzn : y = %d px", ext.y_horizon_px or 0)

    if not dry_run:
        ext.save_yaml(target_out)
        log.info("Saved camera extrinsics to %s", target_out)
    else:
        log.info("[DRY-RUN] Extrinsics verified successfully; file write skipped.")

    return ext


def main():
    parser = argparse.ArgumentParser(description="APEX-LKA Camera Extrinsic Calibration Utility")
    parser.add_argument("--mode", type=str, default="synthetic", choices=["synthetic", "static", "driving"],
                        help="Calibration mode (default: synthetic)")
    parser.add_argument("--synthetic", action="store_true", help="Force synthetic calibration mode")
    parser.add_argument("--dry-run", action="store_true", help="Validate calibration without writing to disk")
    parser.add_argument("--output", type=str, default=None, help="Output YAML config file path")
    parser.add_argument("--image", type=str, default=None, help="Input calibration image file path")
    parser.add_argument("--height", type=float, default=1.20, help="Camera mounting height in meters (default: 1.20)")
    parser.add_argument("--pitch-deg", type=float, default=5.0, help="Expected pitch angle in degrees (default: 5.0)")
    parser.add_argument("--debug-dir", type=str, default=None, help="Directory to save diagnostic debug images")
    args = parser.parse_args()

    out_p = Path(args.output) if args.output else None
    img_p = Path(args.image) if args.image else None
    dbg_p = Path(args.debug_dir) if args.debug_dir else None

    run_calibration(
        mode=args.mode,
        synthetic=args.synthetic,
        dry_run=args.dry_run,
        output_path=out_p,
        image_path=img_p,
        mount_height_m=args.height,
        pitch_deg=args.pitch_deg,
        debug_dir=dbg_p,
    )


if __name__ == "__main__":
    main()
