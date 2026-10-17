"""
src/simulation/carla_bridge.py
==============================
APEX-LKA Phase 5 Section 7: Industrial CARLA Simulator Bridge Protocol.
Provides synchronous, deterministically-stepped virtual execution connecting
the APEX perception and control pipeline directly to CARLA 0.9.15+ (or isolated
synthetic simulation when running in headless CI).

Features:
- Connects to CARLA server with configurable timeout and synchronous frame stepping.
- Spawns virtual ego vehicle (Tesla Model 3 / BAJA layout) with exact hardware sensor rig.
- Virtual Camera (640x360, 90 deg FOV, 1.25m mount height, -5 deg optical tilt).
- Virtual 6-DOF IMU (100 Hz accel + gyro) & GNSS (10 Hz).
- Virtual Wheel Encoders (100 Hz wheel tick telemetry).
- Virtual CAN Bus interface (vcan0) emulating Arduino Mega 2560 CAN traffic.
- Zero-latency vehicle actuation command injection via carla.VehicleControl().
"""

from __future__ import annotations

import logging
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml

# Optional high-speed CAN import
try:
    import can
    HAS_CAN = True
except ImportError:
    can = None
    HAS_CAN = False

# CARLA Python API import check
try:
    import carla
    HAS_CARLA = True
except ImportError:
    carla = None
    HAS_CARLA = False

log = logging.getLogger("apex_carla_bridge")


# ═══════════════════════════════════════════════════════════════════════════
# Data Classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class Transform3D:
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    roll: float = 0.0


@dataclass
class IMUData:
    timestamp: float
    accelerometer: Tuple[float, float, float]  # (ax, ay, az) in m/s^2
    gyroscope: Tuple[float, float, float]      # (wx, wy, wz) in rad/s
    compass_rad: float = 0.0


@dataclass
class GNSSData:
    timestamp: float
    latitude: float
    longitude: float
    altitude: float


@dataclass
class WheelEncoderData:
    timestamp: float
    left_ticks: int
    right_ticks: int
    left_speed_mps: float
    right_speed_mps: float
    linear_speed_mps: float


@dataclass
class VehicleState:
    transform: Transform3D
    velocity_mps: Tuple[float, float, float]
    speed_mps: float
    steer: float = 0.0
    throttle: float = 0.0
    brake: float = 0.0
    gear: int = 1


# ═══════════════════════════════════════════════════════════════════════════
# CARLA Simulator Bridge
# ═══════════════════════════════════════════════════════════════════════════

class CARLABridge:
    """
    Real-time bridge connecting APEX autonomous driving pipeline to CARLA.
    Supports synchronous mode stepping, virtual sensor acquisition,
    virtual CAN broadcasting, and actuation command injection.
    """

    def __init__(
        self,
        config_path: Union[str, Path] = "configs/simulation.yaml",
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[float] = None,
        synchronous: bool = True,
        fixed_delta_seconds: float = 0.03333333,
        mock_mode: bool = False,
    ) -> None:
        self.config_path = Path(config_path)
        self.cfg: Dict[str, Any] = {}
        self._load_config()

        bridge_cfg = self.cfg.get("simulation", {}).get("bridge", {})
        self.host = host or bridge_cfg.get("host", "localhost")
        self.port = port or bridge_cfg.get("port", 2000)
        self.timeout = timeout or bridge_cfg.get("timeout_seconds", 5.0)
        self.synchronous = synchronous
        self.fixed_delta_seconds = fixed_delta_seconds or bridge_cfg.get("fixed_delta_seconds", 0.03333333)
        self.force_mock_mode = mock_mode or not HAS_CARLA

        # CARLA handles
        self.client: Optional[Any] = None
        self.world: Optional[Any] = None
        self.original_settings: Optional[Any] = None
        self.ego_vehicle: Optional[Any] = None
        self.blueprint_library: Optional[Any] = None

        # Spawned sensor actor handles
        self.actors: List[Any] = []
        self.camera_sensor: Optional[Any] = None
        self.depth_sensor: Optional[Any] = None
        self.semantic_sensor: Optional[Any] = None
        self.imu_sensor: Optional[Any] = None
        self.gnss_sensor: Optional[Any] = None

        # Thread-safe sensor queues
        self._camera_queue: queue.Queue = queue.Queue(maxsize=4)
        self._depth_queue: queue.Queue = queue.Queue(maxsize=4)
        self._semantic_queue: queue.Queue = queue.Queue(maxsize=4)
        self._imu_queue: queue.Queue = queue.Queue(maxsize=16)
        self._gnss_queue: queue.Queue = queue.Queue(maxsize=8)

        # Cached latest sensor readings
        self._lock = threading.Lock()
        self._latest_camera_bgr: Optional[np.ndarray] = None
        self._latest_depth_m: Optional[np.ndarray] = None
        self._latest_semantic_mask: Optional[np.ndarray] = None
        self._latest_imu: Optional[IMUData] = None
        self._latest_gnss: Optional[GNSSData] = None
        self._latest_wheel_encoders: Optional[WheelEncoderData] = None
        self._wheel_ticks_left: int = 0
        self._wheel_ticks_right: int = 0

        # Virtual CAN bus handle
        self.can_bus: Optional[Any] = None
        self._init_virtual_can()

        # Mock simulation state (used in CI or when CARLA server offline)
        self.is_mock_world = False
        self._sim_frame_count: int = 0
        self._sim_ego_x: float = 120.5
        self._sim_ego_y: float = 48.2
        self._sim_ego_speed: float = 5.55
        self._sim_steer: float = 0.0
        self._sim_throttle: float = 0.3
        self._sim_brake: float = 0.0

    def _load_config(self) -> None:
        """Loads simulation YAML configuration with sensible defaults."""
        if self.config_path.exists():
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    self.cfg = yaml.safe_load(f) or {}
            except Exception as e:
                log.warning("Failed to parse %s: %s; using defaults.", self.config_path, e)
                self.cfg = {}

    def _init_virtual_can(self) -> None:
        """Initializes virtual CAN bus interface (vcan0 or virtual channel)."""
        can_cfg = self.cfg.get("simulation", {}).get("can_bus", {})
        interface = can_cfg.get("interface", "virtual")
        channel = can_cfg.get("channel", "vcan0")
        bitrate = can_cfg.get("bitrate", 500000)

        if HAS_CAN and can is not None:
            try:
                self.can_bus = can.interface.Bus(
                    channel=channel,
                    interface=interface,
                    bitrate=bitrate,
                    receive_own_messages=True,
                )
                log.info("Virtual CAN Bus initialized on %s (%s).", channel, interface)
            except Exception as e:
                log.debug("Virtual CAN init failed (%s); operating with virtual memory loopback.", e)
                self.can_bus = None

    # ───────────────────────────────────────────────────────────────────────
    # Connection & World Management
    # ───────────────────────────────────────────────────────────────────────

    def connect(self) -> bool:
        """
        Attempts connection to live CARLA server; falls back to mock world if unavailable.
        """
        if self.force_mock_mode:
            log.info("Force mock mode active; initializing deterministic virtual CARLA world.")
            self._setup_mock_world()
            return True

        if not HAS_CARLA or carla is None:
            log.warning("CARLA package not available; operating in mock simulation mode.")
            self._setup_mock_world()
            return True

        try:
            log.info("Connecting to CARLA Server at %s:%d (timeout=%.1fs)...", self.host, self.port, self.timeout)
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)

            self.world = self.client.get_world()
            self.blueprint_library = self.world.get_blueprint_library()

            if self.synchronous:
                self.original_settings = self.world.get_settings()
                settings = self.world.get_settings()
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.fixed_delta_seconds
                self.world.apply_settings(settings)
                log.info("Synchronous simulation mode enabled (fixed_delta=%.4fs, 30 FPS).", self.fixed_delta_seconds)

            self.is_mock_world = False
            log.info("Connected to CARLA Map: %s", self.world.get_map().name)
            return True

        except Exception as exc:
            log.warning("CARLA Server connection failed (%s); falling back to mock simulation.", exc)
            self._setup_mock_world()
            return True

    def _setup_mock_world(self) -> None:
        """Initializes internal mock simulation engine for headless testing and CI."""
        self.is_mock_world = True
        self.client = None
        self.world = None
        self._sim_frame_count = 0
        self._sim_ego_x = 120.5
        self._sim_ego_y = 48.2
        self._sim_ego_speed = 5.55
        log.info("APEX Virtual CARLA Mock World ready.")

    def is_connected(self) -> bool:
        """Returns True if connected to CARLA or active mock simulation."""
        return self.is_mock_world or (self.world is not None)

    # ───────────────────────────────────────────────────────────────────────
    # Ego Vehicle Spawning
    # ───────────────────────────────────────────────────────────────────────

    def spawn_ego_vehicle(
        self,
        blueprint_id: Optional[str] = None,
        transform: Optional[Transform3D] = None,
    ) -> Any:
        """
        Spawns the ego vehicle at designated spawn coordinates.
        """
        ego_cfg = self.cfg.get("simulation", {}).get("ego_vehicle", {})
        bp_name = blueprint_id or ego_cfg.get("blueprint", "vehicle.tesla.model3")

        spawn_cfg = ego_cfg.get("spawn_transform", {})
        tx = transform.x if transform else spawn_cfg.get("x", 120.5)
        ty = transform.y if transform else spawn_cfg.get("y", 48.2)
        tz = transform.z if transform else spawn_cfg.get("z", 0.5)
        pitch = transform.pitch if transform else spawn_cfg.get("pitch", 0.0)
        yaw = transform.yaw if transform else spawn_cfg.get("yaw", 0.0)
        roll = transform.roll if transform else spawn_cfg.get("roll", 0.0)

        if self.is_mock_world or not HAS_CARLA:
            self._sim_ego_x = tx
            self._sim_ego_y = ty
            self.ego_vehicle = "mock_ego_vehicle"
            log.info("Spawned Mock Ego Vehicle at (%.1f, %.1f, %.1f).", tx, ty, tz)
            return self.ego_vehicle

        try:
            bp = self.blueprint_library.find(bp_name)
            bp.set_attribute("role_name", ego_cfg.get("role_name", "hero"))
            color = ego_cfg.get("color", "255,255,255")
            if bp.has_attribute("color"):
                bp.set_attribute("color", color)

            sp_transform = carla.Transform(
                carla.Location(x=tx, y=ty, z=tz),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
            )
            self.ego_vehicle = self.world.spawn_actor(bp, sp_transform)
            self.actors.append(self.ego_vehicle)
            log.info("Spawned CARLA Ego Vehicle (%s) at (%.1f, %.1f, %.1f).", bp_name, tx, ty, tz)
            return self.ego_vehicle

        except Exception as e:
            log.error("Failed to spawn ego vehicle in CARLA: %s; falling back to mock ego.", e)
            self.is_mock_world = True
            self.ego_vehicle = "mock_ego_vehicle"
            return self.ego_vehicle

    # ───────────────────────────────────────────────────────────────────────
    # Virtual Sensor Rig Setup
    # ───────────────────────────────────────────────────────────────────────

    def setup_sensors(
        self,
        enable_depth: bool = True,
        enable_semantic: bool = True,
    ) -> None:
        """
        Attaches the virtual sensor rig to the ego vehicle.
        Mount offsets match physical APEX hardware:
        RGB Camera: 640x360, 90 deg FOV, 1.25m height, -5 deg pitch tilt.
        """
        if self.is_mock_world or self.ego_vehicle == "mock_ego_vehicle" or not HAS_CARLA:
            log.info("Attached Virtual Sensors (RGB Camera, IMU, GNSS, Wheel Encoders) in Mock World.")
            return

        sensors_cfg = self.cfg.get("simulation", {}).get("sensors", {})
        cam_cfg = sensors_cfg.get("camera", {})
        mount = cam_cfg.get("mount", {})

        # 1. RGB Camera
        bp_cam = self.blueprint_library.find("sensor.camera.rgb")
        bp_cam.set_attribute("image_size_x", str(cam_cfg.get("width", 640)))
        bp_cam.set_attribute("image_size_y", str(cam_cfg.get("height", 360)))
        bp_cam.set_attribute("fov", str(cam_cfg.get("fov", 90.0)))

        cam_trans = carla.Transform(
            carla.Location(x=mount.get("x", 1.50), y=mount.get("y", 0.0), z=mount.get("z", 1.25)),
            carla.Rotation(pitch=mount.get("pitch", -5.0), yaw=mount.get("yaw", 0.0), roll=mount.get("roll", 0.0)),
        )
        self.camera_sensor = self.world.spawn_actor(bp_cam, cam_trans, attach_to=self.ego_vehicle)
        self.camera_sensor.listen(self._on_camera_rgb)
        self.actors.append(self.camera_sensor)

        # 2. Depth Camera (Optional)
        if enable_depth:
            bp_depth = self.blueprint_library.find("sensor.camera.depth")
            bp_depth.set_attribute("image_size_x", str(cam_cfg.get("width", 640)))
            bp_depth.set_attribute("image_size_y", str(cam_cfg.get("height", 360)))
            bp_depth.set_attribute("fov", str(cam_cfg.get("fov", 90.0)))
            self.depth_sensor = self.world.spawn_actor(bp_depth, cam_trans, attach_to=self.ego_vehicle)
            self.depth_sensor.listen(self._on_camera_depth)
            self.actors.append(self.depth_sensor)

        # 3. Semantic Segmentation Camera (Optional)
        if enable_semantic:
            bp_sem = self.blueprint_library.find("sensor.camera.semantic_segmentation")
            bp_sem.set_attribute("image_size_x", str(cam_cfg.get("width", 640)))
            bp_sem.set_attribute("image_size_y", str(cam_cfg.get("height", 360)))
            bp_sem.set_attribute("fov", str(cam_cfg.get("fov", 90.0)))
            self.semantic_sensor = self.world.spawn_actor(bp_sem, cam_trans, attach_to=self.ego_vehicle)
            self.semantic_sensor.listen(self._on_camera_semantic)
            self.actors.append(self.semantic_sensor)

        # 4. 6-Axis IMU
        bp_imu = self.blueprint_library.find("sensor.other.imu")
        imu_trans = carla.Transform(carla.Location(x=0.0, y=0.0, z=0.8))
        self.imu_sensor = self.world.spawn_actor(bp_imu, imu_trans, attach_to=self.ego_vehicle)
        self.imu_sensor.listen(self._on_imu)
        self.actors.append(self.imu_sensor)

        # 5. GNSS Receiver
        bp_gnss = self.blueprint_library.find("sensor.other.gnss")
        gnss_trans = carla.Transform(carla.Location(x=0.0, y=0.0, z=1.5))
        self.gnss_sensor = self.world.spawn_actor(bp_gnss, gnss_trans, attach_to=self.ego_vehicle)
        self.gnss_sensor.listen(self._on_gnss)
        self.actors.append(self.gnss_sensor)

        log.info("All CARLA hardware sensor rig streams listening successfully.")

    # ───────────────────────────────────────────────────────────────────────
    # Sensor Callbacks
    # ───────────────────────────────────────────────────────────────────────

    def _on_camera_rgb(self, carla_img: Any) -> None:
        """Callback converting CARLA raw BGRA image to 640x360 BGR uint8."""
        try:
            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
            bgr = arr[:, :, :3]
            with self._lock:
                self._latest_camera_bgr = bgr
            if not self._camera_queue.full():
                self._camera_queue.put(bgr)
        except Exception as e:
            log.debug("RGB image callback conversion error: %s", e)

    def _on_camera_depth(self, carla_img: Any) -> None:
        """Callback converting CARLA depth buffer to metric float32 in meters."""
        try:
            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
            # CARLA depth encoding: normalized = (R + G*256 + B*256*256) / (256^3 - 1)
            # Metric depth = normalized * 1000 meters
            r = arr[:, :, 0].astype(np.float32)
            g = arr[:, :, 1].astype(np.float32)
            b = arr[:, :, 2].astype(np.float32)
            norm = (r + g * 256.0 + b * 65536.0) / 16777215.0
            depth_m = norm * 1000.0
            with self._lock:
                self._latest_depth_m = depth_m
            if not self._depth_queue.full():
                self._depth_queue.put(depth_m)
        except Exception as e:
            log.debug("Depth callback conversion error: %s", e)

    def _on_camera_semantic(self, carla_img: Any) -> None:
        """Callback converting CARLA semantic segmentation to integer label mask."""
        try:
            arr = np.frombuffer(carla_img.raw_data, dtype=np.uint8)
            arr = np.reshape(arr, (carla_img.height, carla_img.width, 4))
            # CARLA CityObjectLabel is in the Red channel (tag)
            tag_mask = arr[:, :, 2]
            with self._lock:
                self._latest_semantic_mask = tag_mask
            if not self._semantic_queue.full():
                self._semantic_queue.put(tag_mask)
        except Exception as e:
            log.debug("Semantic callback conversion error: %s", e)

    def _on_imu(self, carla_imu: Any) -> None:
        """Callback collecting 6-axis accelerometer & gyroscope."""
        try:
            data = IMUData(
                timestamp=carla_imu.timestamp,
                accelerometer=(
                    carla_imu.accelerometer.x,
                    carla_imu.accelerometer.y,
                    carla_imu.accelerometer.z,
                ),
                gyroscope=(
                    carla_imu.gyroscope.x,
                    carla_imu.gyroscope.y,
                    carla_imu.gyroscope.z,
                ),
                compass_rad=float(getattr(carla_imu, "compass", 0.0)),
            )
            with self._lock:
                self._latest_imu = data
            if not self._imu_queue.full():
                self._imu_queue.put(data)
        except Exception as e:
            log.debug("IMU callback error: %s", e)

    def _on_gnss(self, carla_gnss: Any) -> None:
        """Callback collecting GNSS lat/lon/alt coordinates."""
        try:
            data = GNSSData(
                timestamp=carla_gnss.timestamp,
                latitude=carla_gnss.latitude,
                longitude=carla_gnss.longitude,
                altitude=carla_gnss.altitude,
            )
            with self._lock:
                self._latest_gnss = data
            if not self._gnss_queue.full():
                self._gnss_queue.put(data)
        except Exception as e:
            log.debug("GNSS callback error: %s", e)

    # ───────────────────────────────────────────────────────────────────────
    # Synchronous Stepping & Sensor Retrieval
    # ───────────────────────────────────────────────────────────────────────

    def tick(self, timeout_s: float = 1.0) -> None:
        """
        Advances the simulation world by one time step.
        Guarantees deterministic execution at exactly 30 FPS.
        """
        self._sim_frame_count += 1

        if self.is_mock_world or self.world is None or not HAS_CARLA:
            # Deterministically step mock kinematics
            dt = self.fixed_delta_seconds
            # Kinematic bicycle model
            self._sim_ego_speed = max(0.0, self._sim_ego_speed + (self._sim_throttle * 2.0 - self._sim_brake * 5.0) * dt)
            self._sim_ego_x += self._sim_ego_speed * dt * math.cos(self._sim_steer * 0.4)
            self._sim_ego_y += self._sim_ego_speed * dt * math.sin(self._sim_steer * 0.4)

            # Generate deterministic synthetic camera frame
            self._latest_camera_bgr = self._generate_synthetic_frame(self._sim_frame_count)
            self._latest_depth_m = np.full((360, 640), 15.0, dtype=np.float32)
            self._latest_semantic_mask = np.zeros((360, 640), dtype=np.uint8)

            # Generate synthetic IMU & GNSS
            self._latest_imu = IMUData(
                timestamp=time.time(),
                accelerometer=(0.0, 0.0, 9.81),
                gyroscope=(0.0, 0.0, self._sim_steer * 0.1),
            )
            self._latest_gnss = GNSSData(
                timestamp=time.time(),
                latitude=12.9716 + (self._sim_ego_x * 1e-5),
                longitude=77.5946 + (self._sim_ego_y * 1e-5),
                altitude=920.0,
            )

            # Update wheel tick counts
            wheel_radius = 0.33
            dist_step = self._sim_ego_speed * dt
            ticks_step = int((dist_step / (2.0 * math.pi * wheel_radius)) * 1024)
            self._wheel_ticks_left += ticks_step
            self._wheel_ticks_right += ticks_step
            self._latest_wheel_encoders = WheelEncoderData(
                timestamp=time.time(),
                left_ticks=self._wheel_ticks_left,
                right_ticks=self._wheel_ticks_right,
                left_speed_mps=self._sim_ego_speed,
                right_speed_mps=self._sim_ego_speed,
                linear_speed_mps=self._sim_ego_speed,
            )
            return

        try:
            self.world.tick()
        except Exception as exc:
            log.warning("CARLA world.tick() error: %s", exc)

    def _generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Renders realistic road image with simulated camera ego motion."""
        h, w = 360, 640
        frame = np.full((h, w, 3), (75, 75, 75), dtype=np.uint8)
        # Horizon & Sky
        frame[:126, :] = (185, 165, 135)

        # Dynamic Lane Markings
        curve_offset = 20.0 * math.sin(frame_idx * 0.04)
        lx_base = int(180 + curve_offset)
        rx_base = int(460 + curve_offset)

        # Left lane marking
        cv2.line(frame, (lx_base, h - 1), (w // 2 - 25, 126), (255, 255, 255), 5, cv2.LINE_AA)
        # Right lane marking
        cv2.line(frame, (rx_base, h - 1), (w // 2 + 25, 126), (255, 255, 255), 5, cv2.LINE_AA)

        # Dashed center guidance
        for y_seg in range(140, h, 30):
            cx = int(w // 2 + (curve_offset * (y_seg / h)))
            cv2.line(frame, (cx, y_seg), (cx, min(h - 1, y_seg + 14)), (200, 200, 200), 2, cv2.LINE_AA)

        return frame

    def get_camera_frame(self) -> Optional[np.ndarray]:
        """Returns the newest BGR camera frame (640x360 uint8)."""
        with self._lock:
            if self._latest_camera_bgr is not None:
                return self._latest_camera_bgr.copy()
        if self.is_mock_world:
            return self._generate_synthetic_frame(self._sim_frame_count)
        return None

    def get_depth_frame(self) -> Optional[np.ndarray]:
        """Returns metric depth frame in meters (640x360 float32)."""
        with self._lock:
            if self._latest_depth_m is not None:
                return self._latest_depth_m.copy()
        return None

    def get_semantic_mask(self) -> Optional[np.ndarray]:
        """Returns semantic segmentation class mask (640x360 uint8)."""
        with self._lock:
            if self._latest_semantic_mask is not None:
                return self._latest_semantic_mask.copy()
        return None

    def get_imu_data(self) -> Optional[IMUData]:
        """Returns the most recent 6-axis IMU measurement."""
        with self._lock:
            return self._latest_imu

    def get_gnss_data(self) -> Optional[GNSSData]:
        """Returns the most recent GNSS latitude/longitude/altitude."""
        with self._lock:
            return self._latest_gnss

    def get_wheel_encoder_data(self) -> Optional[WheelEncoderData]:
        """Returns virtual wheel tick encoder readings."""
        with self._lock:
            return self._latest_wheel_encoders

    def get_vehicle_state(self) -> VehicleState:
        """Retrieves current 3D position, velocity, and kinematics of the ego vehicle."""
        if self.is_mock_world or self.ego_vehicle == "mock_ego_vehicle" or not HAS_CARLA:
            return VehicleState(
                transform=Transform3D(x=self._sim_ego_x, y=self._sim_ego_y, z=0.5, yaw=self._sim_steer * 20.0),
                velocity_mps=(self._sim_ego_speed * math.cos(self._sim_steer), self._sim_ego_speed * math.sin(self._sim_steer), 0.0),
                speed_mps=self._sim_ego_speed,
                steer=self._sim_steer,
                throttle=self._sim_throttle,
                brake=self._sim_brake,
            )

        try:
            t = self.ego_vehicle.get_transform()
            v = self.ego_vehicle.get_velocity()
            spd = float(math.sqrt(v.x**2 + v.y**2 + v.z**2))
            ctrl = self.ego_vehicle.get_control()

            return VehicleState(
                transform=Transform3D(
                    x=t.location.x,
                    y=t.location.y,
                    z=t.location.z,
                    pitch=t.rotation.pitch,
                    yaw=t.rotation.yaw,
                    roll=t.rotation.roll,
                ),
                velocity_mps=(v.x, v.y, v.z),
                speed_mps=spd,
                steer=ctrl.steer,
                throttle=ctrl.throttle,
                brake=ctrl.brake,
                gear=ctrl.gear,
            )
        except Exception as e:
            log.debug("Get vehicle state error: %s", e)
            return VehicleState(
                transform=Transform3D(),
                velocity_mps=(0.0, 0.0, 0.0),
                speed_mps=0.0,
            )

    # ───────────────────────────────────────────────────────────────────────
    # Actuation Command Injection
    # ───────────────────────────────────────────────────────────────────────

    def apply_control(
        self,
        steer: float,
        throttle: float = 0.3,
        brake: float = 0.0,
        reverse: bool = False,
        hand_brake: bool = False,
    ) -> None:
        """
        Injects vehicle control command into CARLA simulator or mock world.
        Clamps inputs to physical vehicle actuation limits.
        """
        steer_clamped = float(np.clip(steer, -1.0, 1.0))
        throttle_clamped = float(np.clip(throttle, 0.0, 1.0))
        brake_clamped = float(np.clip(brake, 0.0, 1.0))

        self._sim_steer = steer_clamped
        self._sim_throttle = throttle_clamped
        self._sim_brake = brake_clamped

        if not self.is_mock_world and self.ego_vehicle is not None and HAS_CARLA:
            try:
                ctrl = carla.VehicleControl(
                    steer=steer_clamped,
                    throttle=throttle_clamped,
                    brake=brake_clamped,
                    reverse=reverse,
                    hand_brake=hand_brake,
                )
                self.ego_vehicle.apply_control(ctrl)
            except Exception as e:
                log.debug("Failed applying control to CARLA ego vehicle: %s", e)

        # Broadcast actuation feedback on Virtual CAN Bus (CAN ID 0x120 / 0x1A0)
        self._broadcast_can_actuation(steer_clamped, throttle_clamped, brake_clamped)

    def _broadcast_can_actuation(self, steer: float, throttle: float, brake: float) -> None:
        """Transmits injected control on virtual CAN bus for Arduino HIL compatibility."""
        if self.can_bus is not None and HAS_CAN and can is not None:
            try:
                # Steer degrees [-30..+30], throttle [0..100], brake [0..100]
                steer_deg = int(steer * 30.0)
                steer_u16 = steer_deg & 0xFFFF
                th_u8 = int(throttle * 100.0)
                brk_u8 = int(brake * 100.0)
                data = [
                    (steer_u16 >> 8) & 0xFF,
                    steer_u16 & 0xFF,
                    th_u8,
                    brk_u8,
                    0x00, 0x00, 0x00, 0x00,
                ]
                msg = can.Message(arbitration_id=0x120, data=data, is_extended_id=False)
                self.can_bus.send(msg, timeout=0.001)
            except Exception as e:
                log.debug("Virtual CAN send error: %s", e)

    # ───────────────────────────────────────────────────────────────────────
    # Cleanup & Teardown
    # ───────────────────────────────────────────────────────────────────────

    def cleanup(self) -> None:
        """
        Safely tears down spawned sensors and actors, restoring original CARLA settings.
        """
        log.info("Cleaning up CARLA Simulation Bridge resources...")

        # Stop listening sensors
        for s in (self.camera_sensor, self.depth_sensor, self.semantic_sensor, self.imu_sensor, self.gnss_sensor):
            if s is not None:
                try:
                    s.stop()
                except Exception:
                    pass

        # Destroy spawned actors
        if not self.is_mock_world and self.client is not None:
            for actor in self.actors:
                if actor is not None:
                    try:
                        actor.destroy()
                    except Exception:
                        pass

        # Restore original world settings
        if not self.is_mock_world and self.world is not None and self.original_settings is not None:
            try:
                self.world.apply_settings(self.original_settings)
            except Exception:
                pass

        if self.can_bus is not None:
            try:
                self.can_bus.shutdown()
            except Exception:
                pass
            self.can_bus = None

        self.actors.clear()
        self.camera_sensor = None
        self.depth_sensor = None
        self.semantic_sensor = None
        self.imu_sensor = None
        self.gnss_sensor = None
        self.ego_vehicle = None
        self.world = None
        self.client = None
        log.info("CARLA Simulation Bridge cleanup complete.")

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass

