"""
src/control/pure_pursuit.py
===========================
Curvature-Adaptive Pure Pursuit Lateral Controller.
Formula:
    delta_pp = atan2(2 * L * sin(alpha), L_d)
Equipped with:
- Curvature-adaptive lookahead distance L_d(v, kappa)
- Wheelbase L geometry
- Dynamic rate limiter and hard kinematic saturation
"""

from __future__ import annotations

import math
from typing import Optional, Union

import numpy as np


class PurePursuitController:
    """
    Curvature-adaptive geometric Pure Pursuit lateral controller.
    """

    def __init__(
        self,
        wheelbase_m: float = 1.60,
        L_min: float = 1.5,
        L_max: float = 6.0,
        k_speed: float = 0.4,
        delta_max_deg: float = 30.0,
        delta_rate_max_dps: float = 120.0,
    ) -> None:
        self.wheelbase_m = float(wheelbase_m)
        self.L_min = float(L_min)
        self.L_max = float(L_max)
        self.k_speed = float(k_speed)
        self.delta_max_deg = float(delta_max_deg)
        self.delta_rate_max_dps = float(delta_rate_max_dps)
        self._prev_delta_deg: Optional[float] = None

    def reset(self) -> None:
        """Reset internal rate limiter state."""
        self._prev_delta_deg = None

    def compute_lookahead(self, v: float, kappa: float) -> float:
        """
        Adaptive lookahead distance:
        L_d(v, kappa) = clamp(k_speed * v + L_min / sqrt(|kappa| + eps), L_min, L_max)
        Tightens on sharp curves (high kappa), extends on straights.
        """
        eps = 1e-4
        raw_ld = self.k_speed * v + (self.L_min / math.sqrt(abs(kappa) + eps))
        return float(np.clip(raw_ld, self.L_min, self.L_max))

    def compute(
        self,
        e_m: float,
        theta_e: float,
        kappa: float,
        v: float = 2.5,
        dt: float = 0.033,
        center_poly_bev: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute Pure Pursuit steering command in degrees.

        Parameters
        ----------
        e_m : float
            Lateral offset in meters (positive when path is to right of vehicle).
        theta_e : float
            Heading error in radians.
        kappa : float
            Path curvature at lookahead (1/m).
        v : float
            Vehicle speed in m/s.
        dt : float
            Time step elapsed in seconds.
        center_poly_bev : np.ndarray | None
            Optional fitted polynomial X_m = f(Z_m) for exact lookahead evaluation.

        Returns
        -------
        float
            Steering command in degrees clamped to [-delta_max_deg, +delta_max_deg].
        """
        # 1. Compute adaptive lookahead distance
        L_d = self.compute_lookahead(v, kappa)

        # 2. Compute lookahead point (X_d, Z_d) in vehicle frame
        Z_d = L_d
        if center_poly_bev is not None and len(center_poly_bev) >= 2:
            X_d = float(np.polyval(center_poly_bev, Z_d))
        else:
            # 2nd-order Taylor expansion from local offset, heading, and curvature
            X_d = e_m + Z_d * math.tan(theta_e) + 0.5 * kappa * (Z_d ** 2)

        # 3. Angle alpha from vehicle heading (Z-axis) to lookahead point
        alpha = math.atan2(X_d, Z_d)

        # 4. Pure Pursuit steering curvature
        delta_rad = math.atan2(2.0 * self.wheelbase_m * math.sin(alpha), L_d)

        # 5. Convert to degrees, saturate, and rate-limit
        raw_delta_deg = math.degrees(delta_rad)
        sat_delta_deg = float(np.clip(raw_delta_deg, -self.delta_max_deg, self.delta_max_deg))

        if self._prev_delta_deg is not None and dt > 0.0:
            max_step = self.delta_rate_max_dps * dt
            delta_diff = sat_delta_deg - self._prev_delta_deg
            clamped_diff = float(np.clip(delta_diff, -max_step, max_step))
            final_delta_deg = self._prev_delta_deg + clamped_diff
        else:
            final_delta_deg = sat_delta_deg

        self._prev_delta_deg = final_delta_deg
        return round(final_delta_deg, 3)
