"""
src/simulation/digital_twin.py
==============================
APEX-LKA Phase 5 Section 7: Digital Twin Virtual Track Reconstruction.
Parses custom OpenDRIVE (.xodr) road networks and Unreal Engine (.umap) level assets,
manages track waypoints and lane boundaries, and controls environmental weather variants.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

try:
    import carla
    HAS_CARLA = True
except ImportError:
    carla = None
    HAS_CARLA = False

log = logging.getLogger("apex_digital_twin")


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class RoadWaypoint:
    x: float
    y: float
    z: float = 0.0
    heading_rad: float = 0.0
    lane_width_m: float = 3.5
    road_id: str = "0"
    s_m: float = 0.0


@dataclass
class DigitalTwinMap:
    name: str
    xodr_path: Path
    umap_path: Path
    total_length_m: float = 0.0
    speed_limit_kmh: float = 30.0
    waypoints: List[RoadWaypoint] = field(default_factory=list)
    junctions_count: int = 0
    roads_count: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Digital Twin Loader
# ═══════════════════════════════════════════════════════════════════════════

class DigitalTwinLoader:
    """
    Manages loading, parsing, and environmental configuration for digital twin tracks.
    Supports College Campus and BAJA SAE 2027 circuits.
    """

    WEATHER_PRESETS = [
        "clear_noon",
        "cloudy_afternoon",
        "rain_moderate",
        "fog_dense",
        "dusk_golden",
        "night_streetlights",
        "night_pitch_black",
        "dust_cloud_heavy",
    ]

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/simulation.yaml",
        maps_dir: Union[str, Path] = "simulation/maps",
    ) -> None:
        self.config_path = Path(config_path)
        self.maps_dir = Path(maps_dir)
        self.cfg: Dict[str, Any] = {}
        self._load_config()

        self.loaded_maps: Dict[str, DigitalTwinMap] = {}
        self.active_map_name: str = "college_campus"
        self.current_weather: str = "clear_noon"

    def _load_config(self) -> None:
        """Loads simulation configuration YAML."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.cfg = yaml.safe_load(f) or {}
            except Exception as e:
                log.warning("Could not read %s: %s", self.config_path, e)
                self.cfg = {}

    def load_map(self, map_name: str) -> DigitalTwinMap:
        """
        Loads and parses OpenDRIVE (.xodr) and UE4 (.umap) files for the given track.
        """
        map_key = map_name.lower().replace(".xodr", "").replace(".umap", "")
        if map_key in self.loaded_maps:
            self.active_map_name = map_key
            return self.loaded_maps[map_key]

        xodr_file = self.maps_dir / f"{map_key}.xodr"
        umap_file = self.maps_dir / f"{map_key}.umap"

        # Check config for alternate paths
        avail = self.cfg.get("digital_twin", {}).get("available_maps", {}).get(map_key, {})
        if avail:
            if "xodr_path" in avail:
                xodr_file = Path(avail["xodr_path"])
            if "umap_path" in avail:
                umap_file = Path(avail["umap_path"])

        parsed_map = self._parse_opendrive(xodr_file, umap_file, map_key)
        self.loaded_maps[map_key] = parsed_map
        self.active_map_name = map_key
        log.info(
            "Digital Twin map '%s' loaded (roads: %d, length: %.1fm, waypoints: %d).",
            map_key,
            parsed_map.roads_count,
            parsed_map.total_length_m,
            len(parsed_map.waypoints),
        )
        return parsed_map

    def _parse_opendrive(self, xodr_path: Path, umap_path: Path, name: str) -> DigitalTwinMap:
        """
        Parses OpenDRIVE XML road geometries, generating discrete centerline waypoints.
        """
        waypoints: List[RoadWaypoint] = []
        total_len = 0.0
        roads_count = 0
        junctions_count = 0

        if not xodr_path.exists():
            log.warning("OpenDRIVE file %s does not exist; synthesizing procedural geometry.", xodr_path)
            return self._synthesize_procedural_track(name, xodr_path, umap_path)

        try:
            tree = ET.parse(xodr_path)
            root = tree.getroot()

            # Iterate through roads
            for road in root.findall("road"):
                roads_count += 1
                r_id = road.get("id", str(roads_count))
                r_len = float(road.get("length", 100.0))
                total_len += r_len

                plan_view = road.find("planView")
                if plan_view is not None:
                    for geom in plan_view.findall("geometry"):
                        s = float(geom.get("s", 0.0))
                        x = float(geom.get("x", 0.0))
                        y = float(geom.get("y", 0.0))
                        hdg = float(geom.get("hdg", 0.0))
                        length = float(geom.get("length", 10.0))

                        # Sample points along geometry
                        step_m = 2.0
                        num_pts = max(2, int(length / step_m))
                        for i in range(num_pts):
                            dist = i * (length / num_pts)
                            px = x + dist * math.cos(hdg)
                            py = y + dist * math.sin(hdg)
                            waypoints.append(
                                RoadWaypoint(
                                    x=round(px, 3),
                                    y=round(py, 3),
                                    z=0.0,
                                    heading_rad=round(hdg, 4),
                                    road_id=r_id,
                                    s_m=round(s + dist, 3),
                                )
                            )

            for _ in root.findall("junction"):
                junctions_count += 1

        except Exception as exc:
            log.error("Error parsing OpenDRIVE file %s: %s; falling back to synthesis.", xodr_path, exc)
            return self._synthesize_procedural_track(name, xodr_path, umap_path)

        return DigitalTwinMap(
            name=name,
            xodr_path=xodr_path,
            umap_path=umap_path,
            total_length_m=round(total_len, 2),
            speed_limit_kmh=40.0 if "baja" in name else 30.0,
            waypoints=waypoints,
            junctions_count=junctions_count,
            roads_count=roads_count,
        )

    def _synthesize_procedural_track(self, name: str, xodr_path: Path, umap_path: Path) -> DigitalTwinMap:
        """Generates procedural road circuit if .xodr file is not on disk."""
        waypoints: List[RoadWaypoint] = []
        is_baja = "baja" in name.lower()
        num_pts = 120 if is_baja else 80
        total_len = 850.0 if is_baja else 520.0

        for i in range(num_pts):
            theta = (2.0 * math.pi * i) / num_pts
            if is_baja:
                # Off-road irregular loop with hairpin
                r = 100.0 + 35.0 * math.sin(3.0 * theta) + 15.0 * math.cos(5.0 * theta)
                x = 120.0 + r * math.cos(theta)
                y = 50.0 + r * math.sin(theta)
                z = 1.5 * math.sin(2.0 * theta)  # elevation variations
            else:
                # Campus perimeter loop
                rx = 110.0
                ry = 65.0
                x = 120.0 + rx * math.cos(theta)
                y = 50.0 + ry * math.sin(theta)
                z = 0.0

            hdg = theta + math.pi / 2.0
            waypoints.append(
                RoadWaypoint(
                    x=round(x, 3),
                    y=round(y, 3),
                    z=round(z, 3),
                    heading_rad=round(hdg, 4),
                    road_id="proc_loop",
                    s_m=round(i * (total_len / num_pts), 3),
                )
            )

        return DigitalTwinMap(
            name=name,
            xodr_path=xodr_path,
            umap_path=umap_path,
            total_length_m=total_len,
            speed_limit_kmh=40.0 if is_baja else 30.0,
            waypoints=waypoints,
            junctions_count=2,
            roads_count=4,
        )

    # ───────────────────────────────────────────────────────────────────────
    # Environmental Weather Variants
    # ───────────────────────────────────────────────────────────────────────

    def set_weather(
        self,
        weather_name: str,
        carla_world: Optional[Any] = None,
    ) -> Dict[str, float]:
        """
        Applies one of the 8 industrial weather presets to CARLA world or internal state.
        Presets:
          - clear_noon
          - cloudy_afternoon
          - rain_moderate
          - fog_dense
          - dusk_golden
          - night_streetlights
          - night_pitch_black
          - dust_cloud_heavy
        """
        w_key = weather_name.lower().strip()
        if w_key not in self.WEATHER_PRESETS:
            log.warning("Unknown weather '%s'; defaulting to clear_noon.", weather_name)
            w_key = "clear_noon"

        self.current_weather = w_key
        presets_cfg = self.cfg.get("digital_twin", {}).get("weather_presets", {})
        params = presets_cfg.get(w_key, self._get_fallback_weather_params(w_key))

        if carla_world is not None and HAS_CARLA and carla is not None:
            try:
                wp = carla.WeatherParameters(
                    cloudiness=params.get("cloudiness", 10.0),
                    precipitation=params.get("precipitation", 0.0),
                    precipitation_deposits=params.get("precipitation_deposits", 0.0),
                    wind_intensity=params.get("wind_intensity", 5.0),
                    sun_azimuth_angle=params.get("sun_azimuth_angle", 0.0),
                    sun_altitude_angle=params.get("sun_altitude_angle", 75.0),
                    fog_density=params.get("fog_density", 0.0),
                    fog_distance=params.get("fog_distance", 1000.0),
                    wetness=params.get("wetness", 0.0),
                )
                if hasattr(wp, "dust_storm") and "dust_storm" in params:
                    wp.dust_storm = params["dust_storm"]

                carla_world.set_weather(wp)
                log.info("Applied CARLA WeatherParameters for '%s'.", w_key)
            except Exception as e:
                log.debug("Failed setting CARLA weather: %s", e)

        return params

    def _get_fallback_weather_params(self, key: str) -> Dict[str, float]:
        """Built-in physics parameters for all 8 environment variants."""
        table: Dict[str, Dict[str, float]] = {
            "clear_noon": {
                "cloudiness": 10.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 5.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": 75.0,
                "fog_density": 0.0, "fog_distance": 1000.0, "wetness": 0.0,
            },
            "cloudy_afternoon": {
                "cloudiness": 70.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 15.0, "sun_azimuth_angle": 45.0, "sun_altitude_angle": 40.0,
                "fog_density": 5.0, "fog_distance": 500.0, "wetness": 0.0,
            },
            "rain_moderate": {
                "cloudiness": 90.0, "precipitation": 45.0, "precipitation_deposits": 60.0,
                "wind_intensity": 35.0, "sun_azimuth_angle": 90.0, "sun_altitude_angle": 25.0,
                "fog_density": 15.0, "fog_distance": 200.0, "wetness": 80.0,
            },
            "fog_dense": {
                "cloudiness": 80.0, "precipitation": 5.0, "precipitation_deposits": 20.0,
                "wind_intensity": 2.0, "sun_azimuth_angle": 180.0, "sun_altitude_angle": 30.0,
                "fog_density": 65.0, "fog_distance": 50.0, "wetness": 30.0,
            },
            "dusk_golden": {
                "cloudiness": 25.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 10.0, "sun_azimuth_angle": 260.0, "sun_altitude_angle": 8.0,
                "fog_density": 4.0, "fog_distance": 800.0, "wetness": 0.0,
            },
            "night_streetlights": {
                "cloudiness": 20.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 5.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": -45.0,
                "fog_density": 2.0, "fog_distance": 800.0, "wetness": 0.0,
            },
            "night_pitch_black": {
                "cloudiness": 85.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 8.0, "sun_azimuth_angle": 0.0, "sun_altitude_angle": -80.0,
                "fog_density": 0.0, "fog_distance": 1000.0, "wetness": 0.0,
            },
            "dust_cloud_heavy": {
                "cloudiness": 50.0, "precipitation": 0.0, "precipitation_deposits": 0.0,
                "wind_intensity": 85.0, "sun_azimuth_angle": 120.0, "sun_altitude_angle": 45.0,
                "fog_density": 50.0, "fog_distance": 35.0, "wetness": 0.0, "dust_storm": 75.0,
            },
        }
        return table.get(key, table["clear_noon"])

    def apply_synthetic_weather_filter(self, frame_bgr: np.ndarray, weather_name: Optional[str] = None) -> np.ndarray:
        """
        Applies photorealistic camera sensor artifacts matching the weather preset
        (e.g., fog attenuation, sun glare, rain streaks, night exposure).
        """
        w_name = (weather_name or self.current_weather).lower()
        img = frame_bgr.astype(np.float32)

        if w_name == "fog_dense":
            # Blend with uniform white/grey haze (exponential atmospheric scattering)
            haze = np.full_like(img, 215.0)
            img = cv2.addWeighted(img, 0.40, haze, 0.60, 0)

        elif w_name == "rain_moderate":
            # Darker wet road + reflections
            img[:, :, :] *= 0.85
            # Add subtle vertical streaks
            noise = np.random.normal(0, 15, img.shape[:2]).astype(np.float32)
            noise_blur = cv2.GaussianBlur(noise, (1, 15), 0)
            img[:, :, 0] = np.clip(img[:, :, 0] + noise_blur * 0.5, 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] + noise_blur * 0.5, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] + noise_blur * 0.5, 0, 255)

        elif w_name == "dusk_golden":
            # Warm orange/golden sunlight bias
            img[:, :, 0] *= 0.70  # Reduce blue
            img[:, :, 1] *= 0.95  # Neutral green
            img[:, :, 2] = np.clip(img[:, :, 2] * 1.30, 0, 255)  # Boost red/orange

        elif w_name == "night_pitch_black":
            # Headlights illumination cone: center-bottom illuminated, periphery dark
            h, w = img.shape[:2]
            y_indices, x_indices = np.indices((h, w))
            dist_sq = (x_indices - w // 2) ** 2 + ((y_indices - h) * 1.8) ** 2
            cone = np.clip(1.0 - dist_sq / (w * 0.7) ** 2, 0.08, 1.0)
            img *= cone[:, :, np.newaxis]

        elif w_name == "night_streetlights":
            # Low light overall with amber street ambient
            img *= 0.45
            img[:, :, 1] = np.clip(img[:, :, 1] * 1.15, 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * 1.25, 0, 255)

        elif w_name == "dust_cloud_heavy":
            # Sepia / brownish dust storm attenuation
            dust_mask = np.full_like(img, (90.0, 140.0, 180.0))  # BGR dust tint
            img = cv2.addWeighted(img, 0.45, dust_mask, 0.55, 0)

        elif w_name == "cloudy_afternoon":
            # Desaturate and reduce contrast slightly
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            gray_3c = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)
            img = cv2.addWeighted(img, 0.75, gray_3c, 0.25, 0)

        return np.clip(img, 0, 255).astype(np.uint8)

    # ───────────────────────────────────────────────────────────────────────
    # Waypoint Tracking & Nearest Neighbor Search
    # ───────────────────────────────────────────────────────────────────────

    def get_closest_waypoint(self, x: float, y: float, map_name: Optional[str] = None) -> Optional[RoadWaypoint]:
        """Finds closest road centerline waypoint to (x, y) coordinates."""
        m = self.loaded_maps.get((map_name or self.active_map_name).lower(), None)
        if not m or not m.waypoints:
            return None

        best_wp = None
        min_dist_sq = float("inf")
        for wp in m.waypoints:
            d_sq = (wp.x - x) ** 2 + (wp.y - y) ** 2
            if d_sq < min_dist_sq:
                min_dist_sq = d_sq
                best_wp = wp

        return best_wp
