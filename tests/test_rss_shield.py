"""
tests/test_rss_shield.py
========================
Unit tests for Task 5.4: Mobileye Responsibility-Sensitive Safety (RSS) Shield.
"""

import time
import pytest
import numpy as np

from src.planning.rss_shield import (
    MobileyeRSSShield,
    RSSConfig,
    RSSSafetyCheckResult,
)
from src.perception.tracker import TrackedObject


def test_rss_safe_distance_speed_scaling():
    """Verify minimum safe distance increases quadratically/monotonically with vehicle speed."""
    shield = MobileyeRSSShield()

    d_low = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=2.0, lead_speed_mps=2.0)
    d_mid = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=4.17, lead_speed_mps=4.17)
    d_high = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=8.0, lead_speed_mps=8.0)

    assert d_low < d_mid < d_high
    assert d_low >= shield.config.standstill_clearance_m


def test_rss_lead_stationary_requires_larger_distance():
    """Verify closing on a stationary lead vehicle requires greater stopping distance than matching speed."""
    shield = MobileyeRSSShield()

    # Following moving car vs closing on stopped obstacle at same ego speed
    d_moving_lead = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=4.0, lead_speed_mps=4.0)
    d_static_lead = shield.compute_min_safe_longitudinal_distance(ego_speed_mps=4.0, lead_speed_mps=0.0)

    assert d_static_lead > d_moving_lead


def test_rss_evaluation_safe_condition():
    """Verify safe clearance returns is_safe=True and override_active=False."""
    shield = MobileyeRSSShield()
    res = shield.evaluate_threat(
        ego_speed_mps=4.0,
        lead_distance_m=20.0,
        lead_speed_mps=3.5,
        lateral_offset_m=0.0,
    )

    assert res.is_safe is True
    assert res.override_active is False
    assert res.safety_status == "SAFE"
    assert res.commanded_accel_mpss is None


def test_rss_evaluation_override_condition():
    """Verify unsafe distance triggers immediate RSS override with -4.5 m/s^2 commanded braking."""
    shield = MobileyeRSSShield()
    # At 4.0 m/s closing on static obstacle at 3.0 m, safe distance is ~6m
    res = shield.evaluate_threat(
        ego_speed_mps=4.0,
        lead_distance_m=3.0,
        lead_speed_mps=0.0,
        lateral_offset_m=0.2,
    )

    assert res.is_safe is False
    assert res.override_active is True
    assert res.safety_status == "OVERRIDE_BRAKE"
    assert res.commanded_accel_mpss == -4.5
    assert res.violation_margin_m > 0.0


def test_rss_lateral_corridor_filtering():
    """Verify obstacles outside drivable lateral corridor do not trigger longitudinal override."""
    shield = MobileyeRSSShield()
    # Obstacle is very close (2.0m) longitudinally, but located 3.5m to the right (outside corridor)
    res = shield.evaluate_threat(
        ego_speed_mps=4.0,
        lead_distance_m=2.0,
        lead_speed_mps=0.0,
        lateral_offset_m=3.5,
    )

    assert res.override_active is False
    assert res.is_safe is True


def test_rss_batch_objects_evaluation():
    """Verify evaluate_objects identifies the critical threat among multiple entities."""
    shield = MobileyeRSSShield()

    # Object 1: Far away in lane
    obj1 = TrackedObject(track_id=1, bbox_3d=np.array([0.0, 0.0, 18.0]))
    # Object 2: Close on sidewalk (outside corridor)
    obj2 = TrackedObject(track_id=2, bbox_3d=np.array([3.0, 0.0, 2.5]))
    # Object 3: Close directly ahead in lane (CRITICAL)
    obj3 = TrackedObject(track_id=3, bbox_3d=np.array([0.1, 0.0, 2.8]))

    worst = shield.evaluate_objects(ego_speed_mps=4.0, tracked_objects=[obj1, obj2, obj3])
    assert worst.override_active is True
    assert worst.lead_track_id == 3


def test_rss_latency_budget():
    """Verify RSS calculation executes in under 1.0 ms."""
    shield = MobileyeRSSShield()
    obj = TrackedObject(track_id=1, bbox_3d=np.array([0.2, 0.0, 5.0]))

    latencies = []
    for _ in range(50):
        t0 = time.perf_counter()
        res = shield.evaluate_objects(ego_speed_mps=4.0, tracked_objects=[obj])
        latencies.append((time.perf_counter() - t0) * 1000.0)

    p99 = float(np.percentile(latencies, 99))
    assert p99 < 1.0, f"Expected < 1.0 ms, got {p99:.3f} ms"
