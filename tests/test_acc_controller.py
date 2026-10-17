"""
tests/test_acc_controller.py
=============================
Unit tests for Task 5.3: Adaptive Cruise Control (ACC) & Speed Profiler.
"""

import math
import pytest
import numpy as np

from src.planning.acc_controller import (
    AdaptiveCruiseController,
    ACCConfig,
    ACCControlOutput,
)


def test_target_gap_computation():
    """Verify headway gap expands linearly with ego velocity."""
    acc = AdaptiveCruiseController()
    gap_0 = acc.compute_target_gap(0.0)
    gap_2 = acc.compute_target_gap(2.0)
    gap_4 = acc.compute_target_gap(4.0)

    # d_standstill = 2.0, t_gap = 1.5
    assert gap_0 == pytest.approx(2.0, abs=1e-3)
    assert gap_2 == pytest.approx(2.0 + 1.5 * 2.0, abs=1e-3)
    assert gap_4 == pytest.approx(2.0 + 1.5 * 4.0, abs=1e-3)


def test_free_road_acceleration_towards_set_speed():
    """Verify open-road scenario commands positive acceleration up to set speed."""
    acc = AdaptiveCruiseController()
    out = acc.update(ego_speed_mps=1.0, v_set_mps=4.17, lead_distance_m=None)

    assert out.commanded_accel_mpss > 0.0
    assert out.target_speed_mps > 1.0
    assert not out.is_lead_detected


def test_curvature_speed_limiting_on_bends():
    """Verify sharp curvature clamps desired set speed to comfort lateral acceleration."""
    acc = AdaptiveCruiseController()
    # High curvature bend: kappa = 0.20 rad/m (radius 5m)
    # v_bend = sqrt(2.5 / 0.2) = 3.53 m/s < 4.17 m/s
    v_bend = acc.compute_curvature_speed_limit(curvature=0.20)
    assert v_bend < 4.17
    assert v_bend == pytest.approx(math.sqrt(2.5 / 0.20), rel=1e-2)

    out = acc.update(ego_speed_mps=4.0, v_set_mps=4.17, path_curvature=0.20)
    assert out.curvature_speed_limit_mps < 4.17
    assert out.target_speed_mps <= out.curvature_speed_limit_mps + 0.1


def test_car_following_deceleration_behind_close_lead():
    """Verify lead vehicle within target gap triggers negative acceleration / braking."""
    acc = AdaptiveCruiseController()
    # Ego at 3.0 m/s (target gap = 2.0 + 1.5 * 3.0 = 6.5m). Lead is at 3.5m (much closer).
    out = acc.update(
        ego_speed_mps=3.0,
        v_set_mps=4.17,
        lead_distance_m=3.5,
        lead_speed_mps=1.5,
        dt=0.033,
    )

    assert out.is_lead_detected is True
    assert out.commanded_accel_mpss < 0.0  # Must brake
    assert out.gap_error_m is not None
    assert out.gap_error_m < 0.0  # Actual gap < target gap


def test_jerk_slew_rate_limiting():
    """Verify sudden braking commands are rate-limited to max jerk (2.0 m/s^3)."""
    acc = AdaptiveCruiseController()
    dt = 0.05

    # Initial state accel = 0.0
    # Suddenly introduce emergency close vehicle requiring -3.0 m/s^2 deceleration
    out = acc.update(
        ego_speed_mps=4.0,
        v_set_mps=4.17,
        lead_distance_m=1.0,
        lead_speed_mps=0.0,
        dt=dt,
    )

    # In a single step of dt=0.05s, max accel drop is max_jerk * dt = 2.0 * 0.05 = 0.10 m/s^2
    assert abs(out.commanded_accel_mpss) <= (2.0 * dt + 1e-4)
    assert out.jerk_limited is True


def test_acc_reset():
    """Verify reset() clears prior acceleration memory."""
    acc = AdaptiveCruiseController()
    acc.update(ego_speed_mps=2.0, v_set_mps=4.0, lead_distance_m=None)
    assert acc._last_accel_cmd != 0.0

    acc.reset()
    assert acc._last_accel_cmd == 0.0
