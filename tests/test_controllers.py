"""
tests/test_controllers.py
=========================
Unit tests for Task 5.3: Steering Kinematics Tuning (Stanley + Pure Pursuit):
1. Stanley zero error -> zero steering command
2. Stanley pure lateral offset -> correct steering sign
3. Stanley heading error -> correct steering sign
4. Pure Pursuit adaptive lookahead shrinking on high curvature
5. ControllerManager hysteresis switching between Stanley and Pure Pursuit
6. Global slew-rate limiter clamping step changes
7. Perception-aware attenuation scaling down steering command on low track_quality
"""

import math
from pathlib import Path
import sys
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.control.stanley import StanleyController
from src.control.pure_pursuit import PurePursuitController
from src.control.controller_manager import ControllerManager


def test_stanley_zero_error():
    """Verify Stanley outputs exactly 0.0 deg steering when lateral and heading errors are zero."""
    ctrl = StanleyController(k=1.1, epsilon_v=0.3, delta_max_deg=30.0)
    delta = ctrl.compute(e_m=0.0, theta_e=0.0, v=2.5)
    assert delta == 0.0


def test_stanley_pure_lateral_offset():
    """Verify steering sign: positive e_m (path to right) -> positive steering (steer right)."""
    ctrl = StanleyController(k=1.1, epsilon_v=0.3)
    # Path is 0.5m to the right
    delta_right = ctrl.compute(e_m=0.5, theta_e=0.0, v=2.5)
    assert delta_right > 0.0

    # Path is 0.5m to the left
    ctrl.reset()
    delta_left = ctrl.compute(e_m=-0.5, theta_e=0.0, v=2.5)
    assert delta_left < 0.0

    # Symmetry
    assert math.isclose(delta_right, -delta_left, abs_tol=1e-3)


def test_stanley_heading_error():
    """Verify steering sign with heading angle."""
    ctrl = StanleyController(k=1.1, epsilon_v=0.3)
    # Path heads 0.1 rad to the right
    delta = ctrl.compute(e_m=0.0, theta_e=0.10, v=2.5)
    assert delta > 0.0
    assert math.isclose(delta, math.degrees(0.10), abs_tol=0.1)


def test_stanley_saturation_and_rate_limiting():
    """Verify hard kinematic clamping and slew-rate limiting."""
    ctrl = StanleyController(delta_max_deg=30.0, delta_rate_max_dps=120.0)

    # 1. Extreme error saturated at 30 deg
    delta_sat = ctrl.compute(e_m=10.0, theta_e=0.0, v=1.0, dt=1.0)
    assert delta_sat == 30.0

    # 2. Sudden reversal: step to -30 deg in dt = 0.1s
    # Max change = 120 * 0.1 = 12 deg
    # Expected output = 30.0 - 12.0 = 18.0 deg
    delta_slewed = ctrl.compute(e_m=-10.0, theta_e=0.0, v=1.0, dt=0.1)
    assert math.isclose(delta_slewed, 18.0, abs_tol=0.1)


def test_pure_pursuit_adaptive_lookahead():
    """Verify lookahead distance shrinks on sharp curves and expands on straights."""
    pp = PurePursuitController(L_min=1.5, L_max=6.0, k_speed=0.4)

    # Straight road (kappa ~ 0)
    ld_straight = pp.compute_lookahead(v=2.5, kappa=0.0)
    # ld = 0.4 * 2.5 + 1.5 / sqrt(1e-4) = 1.0 + 150 -> clamped to 6.0
    assert ld_straight == 6.0

    # Moderate curve (kappa = 0.20)
    # ld = 1.0 + 1.5 / sqrt(0.2) = 1.0 + 3.35 = 4.35 m
    ld_curve = pp.compute_lookahead(v=2.5, kappa=0.20)

    # Very sharp curve (kappa = 1.50)
    # ld = 1.0 + 1.5 / sqrt(1.5) = 1.0 + 1.22 = 2.22 m
    ld_sharp = pp.compute_lookahead(v=2.5, kappa=1.50)

    assert ld_straight > ld_curve > ld_sharp
    assert ld_sharp >= 1.5


def test_pure_pursuit_steering():
    """Verify Pure Pursuit outputs correct steering sign towards lookahead target."""
    pp = PurePursuitController(wheelbase_m=1.60, L_min=1.5, L_max=6.0)

    # Lookahead target to the right
    delta_right = pp.compute(e_m=0.3, theta_e=0.05, kappa=0.05, v=2.5)
    assert delta_right > 0.0

    pp.reset()
    # Lookahead target to the left
    delta_left = pp.compute(e_m=-0.3, theta_e=-0.05, kappa=-0.05, v=2.5)
    assert delta_left < 0.0


def test_controller_manager_switching_and_blending():
    """Verify ControllerManager switches from Stanley to Pure Pursuit on high curvature with hysteresis."""
    mgr = ControllerManager()

    # 1. Straight road: kappa = 0.02 -> Stanley active
    mgr.step(e_m=0.1, theta_e=0.0, kappa=0.02)
    assert mgr.active_mode == "stanley"

    # 2. Curvature increases to 0.07 (below upper threshold 0.10) -> stays Stanley
    mgr.step(e_m=0.1, theta_e=0.0, kappa=0.07)
    assert mgr.active_mode == "stanley"

    # 3. Curvature exceeds 0.10 (0.08 + 0.02) -> switches to Pure Pursuit
    mgr.step(e_m=0.1, theta_e=0.0, kappa=0.12)
    assert mgr.active_mode == "pure_pursuit"

    # 4. Curvature drops to 0.08 (above lower threshold 0.06) -> stays Pure Pursuit due to hysteresis!
    mgr.step(e_m=0.1, theta_e=0.0, kappa=0.08)
    assert mgr.active_mode == "pure_pursuit"

    # 5. Curvature drops below 0.06 -> switches back to Stanley
    mgr.step(e_m=0.1, theta_e=0.0, kappa=0.04)
    assert mgr.active_mode == "stanley"


def test_controller_manager_quality_scaling():
    """Verify perception-aware control scales down steering authority when track_quality drops."""
    mgr = ControllerManager()

    # Full quality (1.0)
    delta_full = mgr.step(e_m=0.5, theta_e=0.1, kappa=0.02, track_quality=1.0, dt=1.0)

    mgr.reset()
    # High quality (0.80 > 0.75 full_scale_above) -> unattenuated
    delta_high = mgr.step(e_m=0.5, theta_e=0.1, kappa=0.02, track_quality=0.80, dt=1.0)
    assert math.isclose(delta_full, delta_high, abs_tol=1e-2)

    mgr.reset()
    # Degraded quality (0.35) -> attenuated
    delta_degraded = mgr.step(e_m=0.5, theta_e=0.1, kappa=0.02, track_quality=0.35, dt=1.0)
    assert abs(delta_degraded) < abs(delta_full)

    mgr.reset()
    # Zero quality -> attenuated to minimum scale factor (0.30)
    delta_zero_qual = mgr.step(e_m=0.5, theta_e=0.1, kappa=0.02, track_quality=0.0, dt=1.0)
    expected_zero = delta_full * 0.30
    assert math.isclose(delta_zero_qual, expected_zero, abs_tol=0.2)
