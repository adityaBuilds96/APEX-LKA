"""
tests/test_digital_twin.py
==========================
Unit tests for Digital Twin loader, OpenDRIVE XML parsing,
and 8-preset environmental weather simulation.
"""

from pathlib import Path
import numpy as np
import pytest

from src.simulation.digital_twin import (
    DigitalTwinLoader,
    DigitalTwinMap,
    RoadWaypoint,
)


@pytest.fixture
def dt_loader():
    """Provides an initialized DigitalTwinLoader instance."""
    return DigitalTwinLoader(maps_dir="simulation/maps")


def test_opendrive_college_campus_loading(dt_loader):
    """Verify college_campus.xodr is parsed into valid roads and waypoints."""
    campus_map = dt_loader.load_map("college_campus")
    assert isinstance(campus_map, DigitalTwinMap)
    assert campus_map.name == "college_campus"
    assert campus_map.roads_count >= 1
    assert campus_map.total_length_m > 50.0
    assert len(campus_map.waypoints) > 10

    wp = campus_map.waypoints[0]
    assert isinstance(wp, RoadWaypoint)
    assert isinstance(wp.x, float)
    assert isinstance(wp.y, float)


def test_opendrive_baja_track_loading(dt_loader):
    """Verify baja_track_2027.xodr is parsed with elevation and off-road width."""
    baja_map = dt_loader.load_map("baja_track_2027")
    assert isinstance(baja_map, DigitalTwinMap)
    assert baja_map.name == "baja_track_2027"
    assert baja_map.total_length_m > 100.0
    assert len(baja_map.waypoints) > 10


def test_closest_waypoint_lookup(dt_loader):
    """Verify nearest-neighbor waypoint queries return expected coordinates."""
    dt_loader.load_map("college_campus")
    closest = dt_loader.get_closest_waypoint(x=105.0, y=50.0)
    assert closest is not None
    assert abs(closest.x - 105.0) < 5.0
    assert abs(closest.y - 50.0) < 5.0


def test_all_8_weather_presets(dt_loader):
    """Verify all 8 industrial weather presets return valid physical parameters."""
    for preset in DigitalTwinLoader.WEATHER_PRESETS:
        params = dt_loader.set_weather(preset)
        assert isinstance(params, dict)
        assert "cloudiness" in params
        assert "precipitation" in params
        assert "wind_intensity" in params
        assert "sun_altitude_angle" in params
        assert dt_loader.current_weather == preset


def test_synthetic_weather_filters(dt_loader):
    """Verify camera image post-processing filters for different weather variants."""
    dummy_frame = np.full((360, 640, 3), 128, dtype=np.uint8)

    # Test foggy filter
    fog_frame = dt_loader.apply_synthetic_weather_filter(dummy_frame, "fog_dense")
    assert fog_frame.shape == (360, 640, 3)
    assert fog_frame.dtype == np.uint8
    # Fog should increase brightness towards haze color
    assert np.mean(fog_frame) > np.mean(dummy_frame)

    # Test night pitch black filter
    night_frame = dt_loader.apply_synthetic_weather_filter(dummy_frame, "night_pitch_black")
    assert night_frame.shape == (360, 640, 3)
    # Periphery should be darker than center
    assert np.mean(night_frame) < np.mean(dummy_frame)

    # Test rain filter
    rain_frame = dt_loader.apply_synthetic_weather_filter(dummy_frame, "rain_moderate")
    assert rain_frame.shape == (360, 640, 3)
