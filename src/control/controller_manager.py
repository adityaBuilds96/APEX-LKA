"""
src/control/controller_manager.py
=================================
Master Vehicle Lateral Controller Manager:
- Manages Stanley (default for straights & gentle curves) and Pure Pursuit (tight curves)
- Hysteresis switching based on curvature kappa with smooth sigmoid/linear blending
- Perception-aware steering attenuation based on real-time track_quality
- Global kinematic saturation (+/- 30 deg) and rate limiting (120 deg/s)
- Externalized configuration from configs/project_config.yaml
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

DEPLOY_DIR = Path(__file__).resolve().parent.parent.parent
if str(DEPLOY_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_DIR))

from src.config import cfg
from src.control.stanley import StanleyController
from src.control.pure_pursuit import PurePursuitController

log = logging.getLogger("controller_manager")


class ControllerManager:
    """
    Unified lateral controller manager providing perception-aware steering arbitration.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        ctrl_cfg = config or cfg.get("control", {})

        # Geometry & Kinematics limits
        self.wheelbase_m = float(ctrl_cfg.get("wheelbase_m", 1.60))
        limits_cfg = ctrl_cfg.get("limits", {})
        self.delta_max_deg = float(limits_cfg.get("delta_max_deg", 30.0))
        self.delta_rate_max_dps = float(limits_cfg.get("delta_rate_max_dps", 120.0))

        # Stanley sub-controller
        stanley_cfg = ctrl_cfg.get("stanley", {})
        self.stanley = StanleyController(
            k=float(stanley_cfg.get("k", 1.1)),
            epsilon_v=float(stanley_cfg.get("epsilon_v", 0.3)),
            delta_max_deg=self.delta_max_deg,
            delta_rate_max_dps=self.delta_rate_max_dps,
        )

        # Pure pursuit sub-controller
        pp_cfg = ctrl_cfg.get("pure_pursuit", {})
        self.pure_pursuit = PurePursuitController(
            wheelbase_m=self.wheelbase_m,
            L_min=float(pp_cfg.get("L_min", 1.5)),
            L_max=float(pp_cfg.get("L_max", 6.0)),
            k_speed=float(pp_cfg.get("k_speed", 0.4)),
            delta_max_deg=self.delta_max_deg,
            delta_rate_max_dps=self.delta_rate_max_dps,
        )

        # Switching and hysteresis parameters
        switch_cfg = ctrl_cfg.get("switching", {})
        self.kappa_switch = float(switch_cfg.get("kappa_switch", 0.08))
        self.hysteresis = float(switch_cfg.get("hysteresis", 0.02))

        # Perception quality scaling
        qual_cfg = ctrl_cfg.get("quality_scaling", {})
        self.min_scale = float(qual_cfg.get("min_scale", 0.3))
        self.full_scale_above = float(qual_cfg.get("full_scale_above", 0.75))

        # Controller state
        self.active_mode = "stanley"
        self._prev_delta_deg: Optional[float] = None

    def reset(self) -> None:
        """Reset internal controller and rate limiter states."""
        self.active_mode = "stanley"
        self._prev_delta_deg = None
        self.stanley.reset()
        self.pure_pursuit.reset()

    def step(
        self,
        e_m: float,
        theta_e: float,
        kappa: float,
        v: float = 2.5,
        track_quality: float = 1.0,
        dt: float = 0.033,
        center_poly_bev: Optional[np.ndarray] = None,
    ) -> float:
        """
        Execute one control cycle and compute recommended steering angle in degrees.

        Parameters
        ----------
        e_m : float
            Lateral offset error in meters (positive = steer right).
        theta_e : float
            Heading error in radians (positive = steer right).
        kappa : float
            Path curvature (1/m).
        v : float
            Vehicle forward velocity in m/s (default: 2.5 m/s).
        track_quality : float
            Perception confidence score in [0.0, 1.0].
        dt : float
            Loop cycle duration in seconds (default: 0.033s for 30 Hz).
        center_poly_bev : np.ndarray | None
            Optional metric polynomial coefficients for exact trajectory evaluation.

        Returns
        -------
        float
            Calibrated steering angle command in degrees within [-delta_max_deg, +delta_max_deg].
        """
        abs_kappa = abs(kappa)

        # 1. Hysteresis Mode Switching
        upper_threshold = self.kappa_switch + self.hysteresis  # e.g., 0.10
        lower_threshold = self.kappa_switch - self.hysteresis  # e.g., 0.06

        if self.active_mode == "stanley":
            if abs_kappa > upper_threshold:
                self.active_mode = "pure_pursuit"
        elif self.active_mode == "pure_pursuit":
            if abs_kappa < lower_threshold:
                self.active_mode = "stanley"

        # 2. Compute candidate steering outputs
        delta_stanley = self.stanley.compute(e_m=e_m, theta_e=theta_e, v=v, dt=dt)
        delta_pp = self.pure_pursuit.compute(
            e_m=e_m, theta_e=theta_e, kappa=kappa, v=v, dt=dt, center_poly_bev=center_poly_bev
        )

        # 3. Smooth Blending across hysteresis zone
        if abs_kappa <= lower_threshold:
            raw_delta = delta_stanley
        elif abs_kappa >= upper_threshold:
            raw_delta = delta_pp
        else:
            # Linear transition weight w: 0.0 at lower_threshold, 1.0 at upper_threshold
            blend_w = (abs_kappa - lower_threshold) / max(1e-4, (upper_threshold - lower_threshold))
            raw_delta = (1.0 - blend_w) * delta_stanley + blend_w * delta_pp

        # 4. Perception-Aware Control Attenuation
        # If track quality drops below full_scale_above, authority is smoothly attenuated down to min_scale
        q = float(np.clip(track_quality, 0.0, 1.0))
        if q >= self.full_scale_above:
            qual_scale = 1.0
        else:
            ratio = q / max(1e-4, self.full_scale_above)
            qual_scale = self.min_scale + (1.0 - self.min_scale) * ratio

        scaled_delta = raw_delta * qual_scale

        # 5. Global Kinematic Saturation
        sat_delta = float(np.clip(scaled_delta, -self.delta_max_deg, self.delta_max_deg))

        # 6. Global Slew-Rate Limiter
        if self._prev_delta_deg is not None and dt > 0.0:
            max_step = self.delta_rate_max_dps * dt
            diff = sat_delta - self._prev_delta_deg
            clamped_diff = float(np.clip(diff, -max_step, max_step))
            final_delta = self._prev_delta_deg + clamped_diff
        else:
            final_delta = sat_delta

        self._prev_delta_deg = final_delta
        return round(final_delta, 3)
