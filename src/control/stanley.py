"""
src/control/stanley.py
======================
Stanley Lateral Tracking Controller for Autonomous Steering.
Formula:
    delta_stanley = theta_e + atan2(k * e_m, v + epsilon_v)
Equipped with:
- Speed softening constant epsilon_v
- Configurable crosstrack gain k
- Dynamic rate limiter (delta_rate_max_dps)
- Hard kinematic saturation (delta_max_deg)
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np


class StanleyController:
    """
    Kinematic Stanley lateral path tracking controller.
    """

    def __init__(
        self,
        k: float = 1.1,
        epsilon_v: float = 0.3,
        delta_max_deg: float = 30.0,
        delta_rate_max_dps: float = 120.0,
    ) -> None:
        self.k = float(k)
        self.epsilon_v = float(epsilon_v)
        self.delta_max_deg = float(delta_max_deg)
        self.delta_rate_max_dps = float(delta_rate_max_dps)
        self._prev_delta_deg: Optional[float] = None

    def reset(self) -> None:
        """Reset internal rate limiter state."""
        self._prev_delta_deg = None

    def compute(
        self,
        e_m: float,
        theta_e: float,
        v: float = 2.5,
        dt: float = 0.033,
    ) -> float:
        """
        Compute Stanley steering angle command in degrees.

        Parameters
        ----------
        e_m : float
            Signed cross-track lateral error in meters.
            Positive when path is to the right of vehicle centerline (steer right).
        theta_e : float
            Heading error in radians relative to lane tangent.
            Positive when path points right of vehicle heading.
        v : float
            Vehicle forward speed in m/s (default: 2.5 m/s).
        dt : float
            Time step elapsed since previous control cycle in seconds.

        Returns
        -------
        float
            Steering command in degrees clamped to [-delta_max_deg, +delta_max_deg].
        """
        # 1. Stanley steering angle calculation in radians
        softened_speed = max(0.01, v + self.epsilon_v)
        crosstrack_steering = math.atan2(self.k * e_m, softened_speed)
        delta_rad = theta_e + crosstrack_steering

        # 2. Convert to degrees and apply kinematic limits
        raw_delta_deg = math.degrees(delta_rad)
        sat_delta_deg = float(np.clip(raw_delta_deg, -self.delta_max_deg, self.delta_max_deg))

        # 3. Rate limiting (slew-rate constraint)
        if self._prev_delta_deg is not None and dt > 0.0:
            max_step = self.delta_rate_max_dps * dt
            delta_diff = sat_delta_deg - self._prev_delta_deg
            clamped_diff = float(np.clip(delta_diff, -max_step, max_step))
            final_delta_deg = self._prev_delta_deg + clamped_diff
        else:
            final_delta_deg = sat_delta_deg

        self._prev_delta_deg = final_delta_deg
        return round(final_delta_deg, 3)
