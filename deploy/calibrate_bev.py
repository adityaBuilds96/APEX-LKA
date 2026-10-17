"""
deploy/calibrate_bev.py
=======================
BEV Metric Scaling & Track Width Calibration Utility:
- Validates pixel-to-meter scaling by projecting a known track width at known distance
- Auto-tunes meters_per_pixel if geometric error exceeds 3.0%
- Supports --synthetic and --dry-run modes for CI testing and headless Jetson execution
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Optional, Tuple

import cv2
import numpy as np
import yaml

DEPLOY_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.bev_transformer import BEVTransformer, update_bev_scaling

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
log = logging.getLogger("calibrate_bev")


def simulate_synthetic_track_measurement(
    transformer: BEVTransformer,
    known_width_m: float = 1.80,
    eval_distance_z_m: float = 6.0,
) -> float:
    """
    Project ground-truth track boundaries at eval_distance_z_m, warp them,
    and measure recovered distance in BEV space.
    """
    # 1. Known metric points in vehicle frame
    left_m = np.array([[-known_width_m / 2.0, eval_distance_z_m]], dtype=np.float32)
    right_m = np.array([[known_width_m / 2.0, eval_distance_z_m]], dtype=np.float32)

    # 2. Project to camera perspective image pixels
    left_img = transformer.metric_to_image(left_m)
    right_img = transformer.metric_to_image(right_m)

    # 3. Transform image pixels back to metric space through BEV transformer
    left_recovered = transformer.image_to_metric(left_img)
    right_recovered = transformer.image_to_metric(right_img)

    measured_width = float(abs(right_recovered[0, 0] - left_recovered[0, 0]))
    return measured_width


def calibrate_bev_scaling(
    config_path: Path,
    known_track_width_m: float = 1.80,
    eval_z_m: float = 6.0,
    synthetic: bool = True,
    dry_run: bool = False,
    tolerance_pct: float = 3.0,
) -> Tuple[float, float, bool]:
    """Master BEV metric scaling calibration procedure."""
    log.info("Starting BEV Metric Calibration [Config: %s, Synthetic: %s, Dry-Run: %s]", config_path.name, synthetic, dry_run)
    log.info("Target Ground-Truth Track Width: %.3f m at Z = %.1f m", known_track_width_m, eval_z_m)

    transformer = BEVTransformer(config_path)

    measured_width_m = simulate_synthetic_track_measurement(
        transformer=transformer,
        known_width_m=known_track_width_m,
        eval_distance_z_m=eval_z_m,
    )

    err_pct = abs(measured_width_m - known_track_width_m) / known_track_width_m * 100.0
    log.info("Measured BEV Track Width: %.4f m (Error: %.2f%%)", measured_width_m, err_pct)

    needs_update = err_pct > tolerance_pct

    if needs_update and not dry_run:
        updated, new_err = update_bev_scaling(
            config_path=config_path,
            measured_width_m=measured_width_m,
            ground_truth_width_m=known_track_width_m,
            tolerance_pct=tolerance_pct,
        )
        if updated:
            # Re-verify with updated config
            transformer_updated = BEVTransformer(config_path)
            measured_width_m = simulate_synthetic_track_measurement(
                transformer_updated, known_track_width_m, eval_z_m
            )
            err_pct = abs(measured_width_m - known_track_width_m) / known_track_width_m * 100.0
            log.info("Re-verification after auto-tuning: Measured: %.4f m (Error: %.2f%%)", measured_width_m, err_pct)

    passed = err_pct <= tolerance_pct
    if passed:
        log.info("[OK] BEV Metric Scaling successfully verified within <= %.1f%% tolerance.", tolerance_pct)
    else:
        log.warning("[FAIL] BEV Metric Scaling error %.2f%% exceeds %.1f%% tolerance.", err_pct, tolerance_pct)

    return measured_width_m, err_pct, passed


def main():
    parser = argparse.ArgumentParser(description="APEX-LKA BEV Metric Scaling Calibration Utility")
    parser.add_argument("--config", type=str, default="configs/bev.yaml", help="BEV YAML config file path")
    parser.add_argument("--track-width", type=float, default=1.80, help="Known ground-truth track width in meters")
    parser.add_argument("--eval-distance", type=float, default=6.0, help="Forward distance in meters to evaluate width")
    parser.add_argument("--synthetic", action="store_true", help="Run synthetic calibration benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Validate without modifying configuration file")
    parser.add_argument("--tol", type=float, default=3.0, help="Acceptable error tolerance percentage (default: 3.0)")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path

    _, _, passed = calibrate_bev_scaling(
        config_path=cfg_path,
        known_track_width_m=args.track_width,
        eval_z_m=args.eval_distance,
        synthetic=args.synthetic or True,
        dry_run=args.dry_run,
        tolerance_pct=args.tol,
    )

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
