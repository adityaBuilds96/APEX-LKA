"""
APEX-LKA v4.0 — Hardware Auto-Discovery HAL Subsystem
======================================================
Autonomous hardware hot-plug discovery engine running on background async thread.

Scans system topology every 500 ms:
  - /dev/v4l/by-id/ and /sys/class/video4linux/   → USB/CSI cameras
  - /sys/class/net/                                → SocketCAN interfaces
  - /dev/serial/by-id/ and /dev/ttyACM*/ttyUSB*   → Arduino / MTi-680
  - /sys/class/thermal/ + tegrastats              → Jetson thermal/power

State Machine per device:
  CONNECTED → DEGRADED → DISCONNECTED → UNCONFIGURED

Publishes real-time JSON events and Streamlit HUD alerts.
Automatically re-initializes device drivers without restarting main process.
"""

from __future__ import annotations

import asyncio
import glob
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

__all__ = [
    "DeviceState",
    "DeviceType",
    "DeviceRecord",
    "HardwareEvent",
    "HardwareEventType",
    "AutoDiscoveryConfig",
    "AutoDiscoveryEngine",
]

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCAN_INTERVAL_S        = 0.500       # 500 ms scan cycle
THERMAL_LIMIT_C        = 80.0        # Jetson CPU/GPU thermal limit
THERMAL_WARNING_C      = 70.0        # Warning threshold
STALE_THRESHOLD_S      = 2.0         # Device considered stale after 2 missed scans
DEGRADED_THRESHOLD_S   = 5.0         # Device degraded after 5 s of issues
LOST_THRESHOLD_S       = 15.0        # Device lost after 15 s
ARDUINO_BAUD_RATE      = 921600
MTI680_USB_BAUD        = 115200

# Known vendor:product ID pairs
KNOWN_DEVICES: Dict[Tuple[str, str], str] = {
    ("2341", "0042"): "arduino_mega_2560",
    ("2341", "0010"): "arduino_mega_2560",
    ("1A86", "7523"): "arduino_ch340",
    ("0403", "6001"): "arduino_ftdi",
    ("10C4", "EA60"): "mti680_cp210x",
    ("1B4F", "0010"): "camera_usb_generic",
}

# CAN interface names to look for
SOCKETCAN_NAMES = ("can0", "can1", "vcan0", "slcan0", "slcan1")

# Camera device patterns
CAMERA_PATTERNS = [
    "/dev/video*",
    "/dev/v4l/by-id/*",
]

# Serial device patterns
SERIAL_PATTERNS = [
    "/dev/ttyACM*",
    "/dev/ttyUSB*",
    "/dev/serial/by-id/*",
]

# Thermal zone patterns
THERMAL_PATTERNS = [
    "/sys/class/thermal/thermal_zone*/temp",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class DeviceState(Enum):
    UNCONFIGURED = "UNCONFIGURED"
    CONNECTED    = "CONNECTED"
    DEGRADED     = "DEGRADED"
    DISCONNECTED = "DISCONNECTED"


class DeviceType(Enum):
    CAMERA_USB     = "camera_usb"
    CAMERA_CSI     = "camera_csi"
    SOCKETCAN      = "socketcan"
    ARDUINO_PRIMARY   = "arduino_primary"
    ARDUINO_SECONDARY = "arduino_secondary"
    MTI680         = "mti680"
    THERMAL        = "thermal"
    UNKNOWN        = "unknown"


class HardwareEventType(Enum):
    CONNECTED    = "connected"
    DISCONNECTED = "disconnected"
    DEGRADED     = "degraded"
    RECOVERED    = "recovered"
    THERMAL_WARN = "thermal_warning"
    THERMAL_CRIT = "thermal_critical"
    DRIVER_REINIT = "driver_reinit"
    STATE_CHANGE = "state_change"


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------

@dataclass
class DeviceRecord:
    """Full device record maintained in device registry."""
    device_id: str           # Unique device identifier
    device_type: DeviceType
    state: DeviceState = DeviceState.UNCONFIGURED
    path: str = ""           # Device filesystem path
    vendor_id: str = ""
    product_id: str = ""
    serial_number: str = ""
    description: str = ""

    # Timing
    first_seen_time: float = 0.0
    last_seen_time: float = 0.0
    last_state_change: float = 0.0
    connected_since: float = 0.0

    # Health metrics
    error_count: int = 0
    reinit_count: int = 0
    thermal_temp_c: float = 0.0

    # CAN metrics (if applicable)
    can_message_ids: List[int] = field(default_factory=list)
    expected_sample_rate_hz: float = 0.0
    measured_sample_rate_hz: float = 0.0

    @property
    def age_s(self) -> float:
        """Seconds since last seen."""
        return time.monotonic() - self.last_seen_time if self.last_seen_time > 0 else float("inf")

    @property
    def is_stale(self) -> bool:
        return self.age_s > STALE_THRESHOLD_S

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "type": self.device_type.value,
            "state": self.state.value,
            "path": self.path,
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "description": self.description,
            "age_s": round(self.age_s, 3),
            "error_count": self.error_count,
            "reinit_count": self.reinit_count,
            "thermal_temp_c": self.thermal_temp_c,
        }


@dataclass
class HardwareEvent:
    """Hardware topology change event."""
    event_type: HardwareEventType
    device_id: str
    device_type: DeviceType
    device_path: str
    timestamp: float
    old_state: Optional[DeviceState] = None
    new_state: Optional[DeviceState] = None
    message: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "event": self.event_type.value,
            "device_id": self.device_id,
            "device_type": self.device_type.value,
            "path": self.device_path,
            "timestamp": self.timestamp,
            "old_state": self.old_state.value if self.old_state else None,
            "new_state": self.new_state.value if self.new_state else None,
            "message": self.message,
            "metadata": self.metadata,
        })

    def to_hud_alert(self) -> Dict[str, Any]:
        """Format as Streamlit HUD notification."""
        severity = {
            HardwareEventType.THERMAL_CRIT: "error",
            HardwareEventType.DISCONNECTED: "warning",
            HardwareEventType.DEGRADED: "warning",
            HardwareEventType.THERMAL_WARN: "info",
            HardwareEventType.CONNECTED: "success",
            HardwareEventType.RECOVERED: "success",
            HardwareEventType.DRIVER_REINIT: "info",
            HardwareEventType.STATE_CHANGE: "info",
        }.get(self.event_type, "info")

        return {
            "severity": severity,
            "title": f"Hardware: {self.event_type.value.upper()}",
            "body": self.message or f"{self.device_type.value} at {self.device_path}",
            "device_id": self.device_id,
            "timestamp": self.timestamp,
            "dismissible": True,
        }


@dataclass
class ThermalStatus:
    """Jetson thermal and power budget status."""
    cpu_temp_c: float = 0.0
    gpu_temp_c: float = 0.0
    soc_temp_c: float = 0.0
    board_temp_c: float = 0.0
    cpu_power_mw: float = 0.0
    gpu_power_mw: float = 0.0
    total_power_mw: float = 0.0
    is_throttled: bool = False
    thermal_limit_c: float = THERMAL_LIMIT_C
    timestamp: float = 0.0

    @property
    def max_temp_c(self) -> float:
        return max(self.cpu_temp_c, self.gpu_temp_c, self.soc_temp_c)

    @property
    def is_critical(self) -> bool:
        return self.max_temp_c >= self.thermal_limit_c

    @property
    def is_warning(self) -> bool:
        return self.max_temp_c >= THERMAL_WARNING_C and not self.is_critical


@dataclass
class AutoDiscoveryConfig:
    """Configuration for AutoDiscoveryEngine."""
    scan_interval_s: float = SCAN_INTERVAL_S
    thermal_limit_c: float = THERMAL_LIMIT_C
    thermal_warning_c: float = THERMAL_WARNING_C
    enable_camera_scan: bool = True
    enable_can_scan: bool = True
    enable_serial_scan: bool = True
    enable_thermal_scan: bool = True
    enable_driver_reinit: bool = True
    hud_notification_queue_size: int = 100
    event_log_size: int = 1000
    required_devices: List[str] = field(default_factory=list)
    # Vendor:product pairs for Arduino and MTi-680
    arduino_vid_pids: List[Tuple[str, str]] = field(
        default_factory=lambda: [("2341", "0042"), ("2341", "0010"), ("1A86", "7523")]
    )
    mti680_vid_pids: List[Tuple[str, str]] = field(
        default_factory=lambda: [("10C4", "EA60"), ("0403", "6001")]
    )


# ---------------------------------------------------------------------------
# Filesystem Scanner
# ---------------------------------------------------------------------------

class FilesystemScanner:
    """
    Low-level filesystem scanner for Linux device topology.
    Uses /sys and /dev virtual filesystems.
    """

    @staticmethod
    def scan_cameras() -> List[Dict[str, str]]:
        """Scan /dev/video* and /sys/class/video4linux/ for cameras."""
        cameras = []
        # v4l2 devices
        for dev in sorted(glob.glob("/dev/video*")):
            device_id = os.path.basename(dev)
            # Try to get camera name from sysfs
            sys_path = f"/sys/class/video4linux/{device_id}/name"
            name = ""
            try:
                with open(sys_path) as f:
                    name = f.read().strip()
            except OSError:
                pass
            cameras.append({
                "path": dev,
                "device_id": device_id,
                "description": name or "USB/CSI Camera",
                "type": "csi" if "csi" in name.lower() else "usb",
            })

        # by-id symlinks
        for dev in sorted(glob.glob("/dev/v4l/by-id/*")):
            real_path = os.path.realpath(dev)
            cameras.append({
                "path": real_path,
                "device_id": os.path.basename(dev),
                "description": "Camera by-id",
                "type": "usb",
            })

        return cameras

    @staticmethod
    def scan_socketcan() -> List[Dict[str, str]]:
        """Scan /sys/class/net/ for SocketCAN interfaces."""
        can_ifaces = []
        net_dir = Path("/sys/class/net")
        if not net_dir.exists():
            return []

        for iface in net_dir.iterdir():
            iface_name = iface.name
            if not any(iface_name.startswith(prefix)
                       for prefix in ("can", "vcan", "slcan")):
                continue

            # Check if it's a CAN interface
            type_file = iface / "type"
            try:
                with open(type_file) as f:
                    if_type = int(f.read().strip())
                # Type 280 = ARPHRD_CAN in Linux
                if if_type != 280:
                    continue
            except (OSError, ValueError):
                pass  # Some virtual CAN interfaces may not have type file

            # Check link state
            state_file = iface / "operstate"
            state = "unknown"
            try:
                with open(state_file) as f:
                    state = f.read().strip()
            except OSError:
                pass

            can_ifaces.append({
                "path": f"/sys/class/net/{iface_name}",
                "device_id": iface_name,
                "description": f"SocketCAN {iface_name}",
                "operstate": state,
                "bitrate": FilesystemScanner._get_can_bitrate(iface_name),
            })

        return can_ifaces

    @staticmethod
    def _get_can_bitrate(iface_name: str) -> str:
        """Attempt to read CAN bitrate from ip link show."""
        try:
            result = subprocess.run(
                ["ip", "-details", "link", "show", iface_name],
                capture_output=True, text=True, timeout=1.0,
            )
            for line in result.stdout.splitlines():
                if "bitrate" in line:
                    m = re.search(r"bitrate (\d+)", line)
                    if m:
                        return m.group(1)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            pass
        return "unknown"

    @staticmethod
    def scan_serial_devices() -> List[Dict[str, str]]:
        """Scan /dev/ttyACM*, /dev/ttyUSB*, /dev/serial/by-id/ for serial devices."""
        devices = []
        seen_paths: Set[str] = set()

        # ttyACM and ttyUSB
        for pattern in ("/dev/ttyACM*", "/dev/ttyUSB*"):
            for dev in sorted(glob.glob(pattern)):
                if dev not in seen_paths:
                    seen_paths.add(dev)
                    vid, pid, serial, desc = FilesystemScanner._get_usb_info(dev)
                    devices.append({
                        "path": dev,
                        "device_id": os.path.basename(dev),
                        "vendor_id": vid,
                        "product_id": pid,
                        "serial_number": serial,
                        "description": desc,
                    })

        # by-id symlinks
        for dev in sorted(glob.glob("/dev/serial/by-id/*")):
            real_path = os.path.realpath(dev)
            if real_path not in seen_paths:
                seen_paths.add(real_path)
                vid, pid, serial, desc = FilesystemScanner._get_usb_info(real_path)
                devices.append({
                    "path": real_path,
                    "device_id": os.path.basename(dev),
                    "vendor_id": vid,
                    "product_id": pid,
                    "serial_number": serial,
                    "description": desc,
                })

        return devices

    @staticmethod
    def _get_usb_info(tty_path: str) -> Tuple[str, str, str, str]:
        """Extract USB vendor/product/serial from sysfs for a tty device."""
        vid = pid = serial = desc = ""
        try:
            tty_name = os.path.basename(os.path.realpath(tty_path))
            # Walk sysfs to find parent USB device
            sys_tty = Path(f"/sys/class/tty/{tty_name}")
            real_sys = sys_tty.resolve()
            parent = real_sys.parent
            for _ in range(6):
                vid_file = parent / "idVendor"
                pid_file = parent / "idProduct"
                if vid_file.exists():
                    vid = vid_file.read_text().strip().upper()
                    pid = pid_file.read_text().strip().upper() if pid_file.exists() else ""
                    mfr = (parent / "manufacturer").read_text().strip() if (parent / "manufacturer").exists() else ""
                    prod = (parent / "product").read_text().strip() if (parent / "product").exists() else ""
                    serial_f = parent / "serial"
                    serial = serial_f.read_text().strip() if serial_f.exists() else ""
                    desc = f"{mfr} {prod}".strip()
                    break
                parent = parent.parent
        except (OSError, ValueError, FileNotFoundError):
            pass
        return vid, pid, serial, desc

    @staticmethod
    def scan_thermal() -> List[Dict[str, Any]]:
        """Scan /sys/class/thermal/ for temperature zones."""
        zones = []
        for zone_path in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
            zone_name = os.path.basename(zone_path)
            temp_c = None
            zone_type = ""
            try:
                with open(f"{zone_path}/temp") as f:
                    temp_c = int(f.read().strip()) / 1000.0
                with open(f"{zone_path}/type") as f:
                    zone_type = f.read().strip()
            except (OSError, ValueError):
                pass
            if temp_c is not None:
                zones.append({
                    "path": zone_path,
                    "zone_id": zone_name,
                    "type": zone_type,
                    "temp_c": temp_c,
                })
        return zones

    @staticmethod
    def read_tegrastats() -> Dict[str, Any]:
        """Read Jetson Orin power/thermal metrics from tegrastats (if available)."""
        result = {
            "cpu_power_mw": 0.0,
            "gpu_power_mw": 0.0,
            "total_power_mw": 0.0,
            "cpu_freq_mhz": 0,
            "gpu_freq_mhz": 0,
            "is_throttled": False,
        }
        # Try reading from /sys/bus/i2c/drivers/ina3221x sysfs or tegrastats pipe
        try:
            sys_power = Path("/sys/bus/i2c/drivers/ina3221x")
            if sys_power.exists():
                # Find rail power channels
                for rail in sys_power.rglob("in*_input"):
                    try:
                        val_mv = float(rail.read_text().strip())
                        if "vdd_cpu" in str(rail):
                            result["cpu_power_mw"] = val_mv
                        elif "vdd_gpu" in str(rail):
                            result["gpu_power_mw"] = val_mv
                    except (OSError, ValueError):
                        pass
        except OSError:
            pass
        return result


# ---------------------------------------------------------------------------
# Device State Machine
# ---------------------------------------------------------------------------

class DeviceStateMachine:
    """
    Per-device state machine with hysteresis.
    UNCONFIGURED → CONNECTED → DEGRADED → DISCONNECTED → (reconnect) → CONNECTED
    """

    def __init__(self, device: DeviceRecord) -> None:
        self._device = device
        self._missing_scans = 0

    def on_device_seen(self, now: float) -> Optional[DeviceState]:
        """
        Call when device is found in a scan.
        Returns new state if state changed, else None.
        """
        self._missing_scans = 0
        old_state = self._device.state
        self._device.last_seen_time = now

        if old_state in (DeviceState.UNCONFIGURED, DeviceState.DISCONNECTED):
            self._device.state = DeviceState.CONNECTED
            self._device.connected_since = now
            self._device.last_state_change = now
            return DeviceState.CONNECTED

        if old_state == DeviceState.DEGRADED:
            self._device.state = DeviceState.CONNECTED
            self._device.last_state_change = now
            return DeviceState.CONNECTED

        return None  # No state change

    def on_device_missing(self, now: float) -> Optional[DeviceState]:
        """
        Call when device NOT found in a scan.
        Returns new state if state changed, else None.
        """
        self._missing_scans += 1
        old_state = self._device.state

        if old_state == DeviceState.UNCONFIGURED:
            return None

        miss_s = self._missing_scans * SCAN_INTERVAL_S

        if miss_s >= LOST_THRESHOLD_S and old_state != DeviceState.DISCONNECTED:
            self._device.state = DeviceState.DISCONNECTED
            self._device.last_state_change = now
            return DeviceState.DISCONNECTED

        if miss_s >= DEGRADED_THRESHOLD_S and old_state == DeviceState.CONNECTED:
            self._device.state = DeviceState.DEGRADED
            self._device.last_state_change = now
            return DeviceState.DEGRADED

        return None


# ---------------------------------------------------------------------------
# Driver Reinit Registry
# ---------------------------------------------------------------------------

class DriverReinitRegistry:
    """
    Maintains a registry of driver init callbacks per device type.
    Called automatically when a device is reconnected.
    """

    def __init__(self) -> None:
        self._handlers: Dict[DeviceType, List[Callable[[DeviceRecord], None]]] = {}
        self._lock = threading.Lock()

    def register(self, device_type: DeviceType, handler: Callable[[DeviceRecord], None]) -> None:
        with self._lock:
            if device_type not in self._handlers:
                self._handlers[device_type] = []
            self._handlers[device_type].append(handler)
        log.debug("Driver reinit handler registered for %s.", device_type.value)

    def trigger(self, device: DeviceRecord) -> int:
        """Trigger all handlers for device type. Returns count of handlers called."""
        with self._lock:
            handlers = list(self._handlers.get(device.device_type, []))
        count = 0
        for handler in handlers:
            try:
                handler(device)
                count += 1
            except Exception as exc:
                log.error(
                    "Driver reinit handler error for %s: %s",
                    device.device_id, exc,
                )
        return count


# ---------------------------------------------------------------------------
# Auto-Discovery Engine
# ---------------------------------------------------------------------------

class AutoDiscoveryEngine:
    """
    Autonomous hardware hot-plug discovery engine.

    Scans system topology every 500 ms and maintains a live device registry.
    Fires callbacks on device connect/disconnect/degrade events.
    Publishes JSON events to Streamlit HUD notification queue.
    Automatically reinitializes device drivers on reconnection.

    Thread-safe: scan thread runs independently of consumer threads.
    """

    def __init__(self, config: Optional[AutoDiscoveryConfig] = None) -> None:
        self._cfg = config or AutoDiscoveryConfig()
        self._scanner = FilesystemScanner()
        self._reinit_registry = DriverReinitRegistry()

        # Device registry: device_id → DeviceRecord
        self._registry: Dict[str, DeviceRecord] = {}
        self._state_machines: Dict[str, DeviceStateMachine] = {}
        self._registry_lock = threading.RLock()

        # Event system
        self._event_callbacks: List[Callable[[HardwareEvent], None]] = []
        self._event_log: List[HardwareEvent] = []
        self._hud_queue: "queue.Queue[Dict]" = __import__("queue").Queue(
            maxsize=self._cfg.hud_notification_queue_size
        )
        self._event_lock = threading.Lock()

        # Thermal status
        self._thermal_status = ThermalStatus()
        self._thermal_lock = threading.Lock()

        # Scan thread
        self._running = False
        self._scan_thread: Optional[threading.Thread] = None
        self._scan_count = 0
        self._last_scan_time = 0.0

        # Track previously seen device sets for delta detection
        self._prev_can_ifaces: Set[str] = set()
        self._prev_serial_devs: Set[str] = set()
        self._prev_camera_devs: Set[str] = set()

        log.info(
            "AutoDiscoveryEngine initialized: scan_interval=%.3f s, "
            "thermal_limit=%.1f°C.",
            self._cfg.scan_interval_s, self._cfg.thermal_limit_c,
        )

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def start(self) -> None:
        """Start background scan thread."""
        if self._running:
            return
        self._running = True
        self._scan_thread = threading.Thread(
            target=self._scan_loop,
            name="AutoDiscovery",
            daemon=True,
        )
        self._scan_thread.start()
        log.info("AutoDiscoveryEngine scan thread started.")

    def stop(self) -> None:
        """Stop background scan thread gracefully."""
        self._running = False
        if self._scan_thread and self._scan_thread.is_alive():
            self._scan_thread.join(timeout=5.0)
        log.info("AutoDiscoveryEngine stopped.")

    @property
    def is_running(self) -> bool:
        """Return True if background discovery loop is running."""
        return self._running

    def run_once(self) -> Dict[str, Any]:
        """Run a single scan cycle and return results. Useful for CLI testing."""
        return self._run_scan()

    def register_event_callback(self, callback: Callable[[HardwareEvent], None]) -> None:
        """Register callback for hardware events."""
        self._event_callbacks.append(callback)

    def register_driver_reinit(
        self,
        device_type: DeviceType,
        handler: Callable[[DeviceRecord], None],
    ) -> None:
        """Register driver reinitialization handler for a device type."""
        self._reinit_registry.register(device_type, handler)

    def get_device(self, device_id: str) -> Optional[DeviceRecord]:
        """Get current state of a specific device."""
        with self._registry_lock:
            return self._registry.get(device_id)

    def get_all_devices(self) -> Dict[str, DeviceRecord]:
        """Return snapshot of all devices in registry."""
        with self._registry_lock:
            return dict(self._registry)

    def get_devices_by_type(self, device_type: DeviceType) -> List[DeviceRecord]:
        """Return all devices of a given type."""
        with self._registry_lock:
            return [d for d in self._registry.values() if d.device_type == device_type]

    def get_connected_can_interfaces(self) -> List[str]:
        """Return names of currently connected SocketCAN interfaces."""
        return [
            d.device_id.replace("can_", "", 1) if d.device_id.startswith("can_") else d.device_id
            for d in self.get_devices_by_type(DeviceType.SOCKETCAN)
            if d.state == DeviceState.CONNECTED
        ]

    def get_connected_arduinos(self) -> List[DeviceRecord]:
        """Return currently connected Arduino devices."""
        return [
            d for d in self._registry.values()
            if d.device_type in (DeviceType.ARDUINO_PRIMARY, DeviceType.ARDUINO_SECONDARY)
            and d.state == DeviceState.CONNECTED
        ]

    def get_thermal_status(self) -> ThermalStatus:
        """Return current thermal status."""
        with self._thermal_lock:
            return self._thermal_status

    def get_hud_notification(self, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Dequeue one HUD notification (non-blocking if timeout=0).
        Returns None if no notification available.
        """
        try:
            if timeout <= 0.0:
                return self._hud_queue.get_nowait()
            return self._hud_queue.get(timeout=timeout)
        except Exception:
            return None

    def get_event_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return recent hardware events as JSON-serializable dicts."""
        with self._event_lock:
            events = self._event_log[-limit:]
        return [
            {
                "event": e.event_type.value,
                "device_id": e.device_id,
                "device_type": e.device_type.value,
                "path": e.device_path,
                "timestamp": e.timestamp,
                "message": e.message,
            }
            for e in events
        ]

    def get_status_summary(self) -> Dict[str, Any]:
        """Return full status summary for dashboard/HUD."""
        with self._registry_lock:
            all_devices = list(self._registry.values())

        thermal = self.get_thermal_status()
        connected = [d for d in all_devices if d.state == DeviceState.CONNECTED]
        degraded  = [d for d in all_devices if d.state == DeviceState.DEGRADED]
        missing   = [d for d in all_devices if d.state == DeviceState.DISCONNECTED]

        return {
            "scan_count": self._scan_count,
            "last_scan_time": self._last_scan_time,
            "total_devices": len(all_devices),
            "connected": len(connected),
            "connected_devices": len(connected),
            "degraded": len(degraded),
            "degraded_devices": len(degraded),
            "disconnected": len(missing),
            "disconnected_devices": len(missing),
            "socketcan_interfaces": [d.device_id for d in connected
                                     if d.device_type == DeviceType.SOCKETCAN],
            "can_interfaces": self.get_connected_can_interfaces(),
            "arduinos_connected": len([d for d in connected
                                       if d.device_type in (
                                           DeviceType.ARDUINO_PRIMARY,
                                           DeviceType.ARDUINO_SECONDARY)]),
            "arduino_primary_connected": any(
                d.device_type == DeviceType.ARDUINO_PRIMARY and d.state == DeviceState.CONNECTED
                for d in all_devices
            ),
            "arduino_secondary_connected": any(
                d.device_type == DeviceType.ARDUINO_SECONDARY and d.state == DeviceState.CONNECTED
                for d in all_devices
            ),
            "cameras_connected": len([d for d in connected
                                      if d.device_type in (
                                          DeviceType.CAMERA_USB,
                                          DeviceType.CAMERA_CSI)]),
            "mti680_connected": any(d.device_type == DeviceType.MTI680
                                    and d.state == DeviceState.CONNECTED
                                    for d in all_devices),
            "is_running": self._running,
            "max_temp_c": thermal.max_temp_c,
            "thermal_warning": thermal.is_warning,
            "thermal_critical": thermal.is_critical,
            "devices": {d.device_id: d.to_dict() for d in all_devices},
            "thermal": {
                "max_temp_c": thermal.max_temp_c,
                "cpu_c": thermal.cpu_temp_c,
                "gpu_c": thermal.gpu_temp_c,
                "is_throttled": thermal.is_throttled,
                "is_critical": thermal.is_critical,
                "is_warning": thermal.is_warning,
            },
        }

    # -----------------------------------------------------------------------
    # Background scan loop
    # -----------------------------------------------------------------------

    def _scan_loop(self) -> None:
        log.debug("AutoDiscovery scan loop started.")
        while self._running:
            try:
                self._run_scan()
            except Exception as exc:
                log.error("AutoDiscovery scan error: %s", exc, exc_info=True)
            time.sleep(self._cfg.scan_interval_s)
        log.debug("AutoDiscovery scan loop exited.")

    def _run_scan(self) -> Dict[str, Any]:
        """Execute one full topology scan. Returns scan result summary."""
        now = time.monotonic()
        self._scan_count += 1
        self._last_scan_time = now

        results: Dict[str, Any] = {
            "scan_id": self._scan_count,
            "timestamp": now,
            "cameras": [],
            "can_interfaces": [],
            "serial_devices": [],
            "thermal_zones": [],
        }

        # Scan cameras
        if self._cfg.enable_camera_scan:
            cameras = FilesystemScanner.scan_cameras()
            results["cameras"] = cameras
            self._process_camera_scan(cameras, now)

        # Scan SocketCAN
        if self._cfg.enable_can_scan:
            can_ifaces = FilesystemScanner.scan_socketcan()
            results["can_interfaces"] = can_ifaces
            self._process_can_scan(can_ifaces, now)

        # Scan serial devices
        if self._cfg.enable_serial_scan:
            serial_devs = FilesystemScanner.scan_serial_devices()
            results["serial_devices"] = serial_devs
            self._process_serial_scan(serial_devs, now)

        # Scan thermal
        if self._cfg.enable_thermal_scan:
            thermal_zones = FilesystemScanner.scan_thermal()
            results["thermal_zones"] = thermal_zones
            self._process_thermal_scan(thermal_zones, now)

        # Mark devices not seen in this scan as potentially degraded
        self._age_missing_devices(now)

        return results

    # -----------------------------------------------------------------------
    # Scan processors
    # -----------------------------------------------------------------------

    def _process_camera_scan(self, cameras: List[Dict], now: float) -> None:
        current_paths = {c["path"] for c in cameras}

        for cam in cameras:
            path = cam["path"]
            device_id = f"camera_{cam.get('device_id', path)}"
            dtype = DeviceType.CAMERA_CSI if cam.get("type") == "csi" else DeviceType.CAMERA_USB
            self._update_device(device_id, dtype, path, cam, now)

        # Detect missing cameras
        with self._registry_lock:
            cam_devs = [
                d for d in self._registry.values()
                if d.device_type in (DeviceType.CAMERA_USB, DeviceType.CAMERA_CSI)
                and d.state != DeviceState.DISCONNECTED
            ]
        for d in cam_devs:
            if d.path not in current_paths:
                self._mark_device_missing(d.device_id, now)

        self._prev_camera_devs = current_paths

    def _process_can_scan(self, can_ifaces: List[Dict], now: float) -> None:
        current_names = {c["device_id"] for c in can_ifaces}

        for iface in can_ifaces:
            device_id = f"can_{iface['device_id']}"
            self._update_device(device_id, DeviceType.SOCKETCAN, iface["path"], iface, now)

        # Detect missing CAN interfaces
        with self._registry_lock:
            can_devs = [
                d for d in self._registry.values()
                if d.device_type == DeviceType.SOCKETCAN
                and d.state != DeviceState.DISCONNECTED
            ]
        for d in can_devs:
            iface_name = d.device_id.replace("can_", "", 1) if d.device_id.startswith("can_") else d.device_id
            if iface_name not in current_names:
                self._mark_device_missing(d.device_id, now)

        self._prev_can_ifaces = current_names

    def _process_serial_scan(self, serial_devs: List[Dict], now: float) -> None:
        current_paths = {d["path"] for d in serial_devs}
        arduino_count = 0

        for dev in serial_devs:
            path = dev["path"]
            vid = dev.get("vendor_id", "").upper()
            pid = dev.get("product_id", "").upper()
            desc = dev.get("description", "")

            # Classify device type
            is_arduino = any(
                vid == v and pid == p
                for v, p in self._cfg.arduino_vid_pids
            )
            is_mti680 = any(
                vid == v and pid == p
                for v, p in self._cfg.mti680_vid_pids
            )
            # Also check description for MTi-680
            if not is_mti680 and "mti" in desc.lower():
                is_mti680 = True
            if not is_arduino and "arduino" in desc.lower():
                is_arduino = True

            if is_arduino:
                arduino_count += 1
                dtype = DeviceType.ARDUINO_PRIMARY if arduino_count <= 1 else DeviceType.ARDUINO_SECONDARY
                device_id = f"arduino_{arduino_count}_{os.path.basename(path)}"
            elif is_mti680:
                dtype = DeviceType.MTI680
                device_id = f"mti680_{os.path.basename(path)}"
            else:
                dtype = DeviceType.UNKNOWN
                device_id = f"serial_{os.path.basename(path)}"

            self._update_device(device_id, dtype, path, dev, now)

        # Detect missing serial devices
        with self._registry_lock:
            ser_devs = [
                d for d in self._registry.values()
                if d.device_type in (
                    DeviceType.ARDUINO_PRIMARY,
                    DeviceType.ARDUINO_SECONDARY,
                    DeviceType.MTI680,
                    DeviceType.UNKNOWN,
                )
                and d.state != DeviceState.DISCONNECTED
            ]
        for d in ser_devs:
            if d.path not in current_paths:
                self._mark_device_missing(d.device_id, now)

        self._prev_serial_devs = current_paths

    def _process_thermal_scan(self, zones: List[Dict], now: float) -> None:
        """Update thermal status from zone readings."""
        cpu_temp = 0.0
        gpu_temp = 0.0
        soc_temp = 0.0
        board_temp = 0.0
        max_temp = 0.0
        throttled = False

        for zone in zones:
            temp = zone.get("temp_c", 0.0)
            zone_type = (zone.get("type") or zone.get("zone") or "").lower()
            max_temp = max(max_temp, temp)

            if "cpu" in zone_type:
                cpu_temp = max(cpu_temp, temp)
            elif "gpu" in zone_type:
                gpu_temp = max(gpu_temp, temp)
            elif "soc" in zone_type:
                soc_temp = max(soc_temp, temp)
            elif "board" in zone_type:
                board_temp = max(board_temp, temp)

            if temp >= self._cfg.thermal_limit_c:
                throttled = True

        if cpu_temp == 0.0 and gpu_temp == 0.0 and soc_temp == 0.0 and max_temp > 0.0:
            cpu_temp = max_temp

        with self._thermal_lock:
            old_temp = self._thermal_status.max_temp_c
            self._thermal_status = ThermalStatus(
                cpu_temp_c=cpu_temp,
                gpu_temp_c=gpu_temp,
                soc_temp_c=soc_temp,
                board_temp_c=board_temp,
                is_throttled=throttled,
                thermal_limit_c=self._cfg.thermal_limit_c,
                timestamp=now,
            )

        # Fire thermal events
        if max_temp >= self._cfg.thermal_limit_c:
            self._fire_event(HardwareEvent(
                event_type=HardwareEventType.THERMAL_CRIT,
                device_id="jetson_thermal",
                device_type=DeviceType.THERMAL,
                device_path="/sys/class/thermal",
                timestamp=now,
                message=f"CRITICAL: Max temperature {max_temp:.1f}°C >= {self._cfg.thermal_limit_c}°C!",
                metadata={"cpu_c": cpu_temp, "gpu_c": gpu_temp, "max_c": max_temp},
            ))
        elif max_temp >= self._cfg.thermal_warning_c and old_temp < self._cfg.thermal_warning_c:
            self._fire_event(HardwareEvent(
                event_type=HardwareEventType.THERMAL_WARN,
                device_id="jetson_thermal",
                device_type=DeviceType.THERMAL,
                device_path="/sys/class/thermal",
                timestamp=now,
                message=f"WARNING: Temperature {max_temp:.1f}°C >= {self._cfg.thermal_warning_c}°C",
                metadata={"cpu_c": cpu_temp, "gpu_c": gpu_temp},
            ))

    # -----------------------------------------------------------------------
    # Device registry management
    # -----------------------------------------------------------------------

    def _update_device(
        self,
        device_id: str,
        device_type: DeviceType,
        path: str,
        info: Dict[str, Any],
        now: float,
    ) -> None:
        """Update or create device record. Fire events on state changes."""
        with self._registry_lock:
            if device_id not in self._registry:
                # New device
                record = DeviceRecord(
                    device_id=device_id,
                    device_type=device_type,
                    path=path,
                    vendor_id=info.get("vendor_id", ""),
                    product_id=info.get("product_id", ""),
                    serial_number=info.get("serial_number", ""),
                    description=info.get("description", ""),
                    first_seen_time=now,
                    last_seen_time=now,
                )
                self._registry[device_id] = record
                self._state_machines[device_id] = DeviceStateMachine(record)
                log.info("New device discovered: %s (%s) at %s", device_id, device_type.value, path)
            else:
                record = self._registry[device_id]
                record.last_seen_time = now

            sm = self._state_machines[device_id]
            old_state = record.state
            new_state = sm.on_device_seen(now)

        if new_state is not None and new_state != old_state:
            evt_type = (HardwareEventType.RECOVERED
                        if old_state in (DeviceState.DEGRADED, DeviceState.DISCONNECTED)
                        else HardwareEventType.CONNECTED)

            self._fire_event(HardwareEvent(
                event_type=evt_type,
                device_id=device_id,
                device_type=device_type,
                device_path=path,
                timestamp=now,
                old_state=old_state,
                new_state=new_state,
                message=f"{device_type.value} {evt_type.value}: {path}",
            ))

            # Trigger driver reinitialization
            if self._cfg.enable_driver_reinit:
                with self._registry_lock:
                    rec = self._registry.get(device_id)
                if rec:
                    count = self._reinit_registry.trigger(rec)
                    if count > 0:
                        rec.reinit_count += 1
                        self._fire_event(HardwareEvent(
                            event_type=HardwareEventType.DRIVER_REINIT,
                            device_id=device_id,
                            device_type=device_type,
                            device_path=path,
                            timestamp=now,
                            message=f"Driver reinitialized ({count} handlers) for {device_id}",
                        ))

    def _mark_device_missing(self, device_id: str, now: float) -> None:
        """Mark a device as missing in this scan cycle."""
        with self._registry_lock:
            record = self._registry.get(device_id)
            sm = self._state_machines.get(device_id)
            if record is None or sm is None:
                return
            old_state = record.state
            new_state = sm.on_device_missing(now)

        if new_state is not None and new_state != old_state:
            evt_type = (HardwareEventType.DISCONNECTED
                        if new_state == DeviceState.DISCONNECTED
                        else HardwareEventType.DEGRADED)
            self._fire_event(HardwareEvent(
                event_type=evt_type,
                device_id=device_id,
                device_type=record.device_type,
                device_path=record.path,
                timestamp=now,
                old_state=old_state,
                new_state=new_state,
                message=f"{record.device_type.value} {evt_type.value}: {record.path}",
            ))

    def _age_missing_devices(self, now: float) -> None:
        """Check all registered devices and mark those not seen recently."""
        with self._registry_lock:
            device_ids = list(self._registry.keys())
        for device_id in device_ids:
            with self._registry_lock:
                record = self._registry.get(device_id)
            if record and record.is_stale and record.state == DeviceState.CONNECTED:
                self._mark_device_missing(device_id, now)

    # -----------------------------------------------------------------------
    # Event system
    # -----------------------------------------------------------------------

    def _fire_event(self, event: HardwareEvent) -> None:
        """Fire a hardware event to all registered callbacks and HUD queue."""
        with self._event_lock:
            self._event_log.append(event)
            if len(self._event_log) > self._cfg.event_log_size:
                self._event_log = self._event_log[-self._cfg.event_log_size:]

        # Push HUD notification (non-blocking)
        try:
            self._hud_queue.put_nowait(event.to_hud_alert())
        except Exception:
            pass  # HUD queue full — skip

        # Call registered callbacks
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception as exc:
                log.error("Event callback error: %s", exc)

        log.info("[HW-EVENT] %s: %s", event.event_type.value, event.message)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def _main_cli() -> None:
    """CLI test mode: run one scan cycle and print results."""
    import argparse
    import json

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="APEX-LKA Auto-Discovery Engine")
    parser.add_argument("--once", action="store_true", help="Run one scan cycle and exit.")
    parser.add_argument("--continuous", action="store_true", help="Run continuous scanning.")
    parser.add_argument("--json", action="store_true", help="Output results as JSON.")
    args = parser.parse_args()

    engine = AutoDiscoveryEngine()

    if args.once:
        results = engine.run_once()
        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            print(f"Scan complete: {results['scan_id']} | "
                  f"CAN: {len(results.get('can_interfaces', []))} | "
                  f"Serial: {len(results.get('serial_devices', []))} | "
                  f"Cameras: {len(results.get('cameras', []))} | "
                  f"Thermal: {len(results.get('thermal_zones', []))} zones")
            for zone in results.get("thermal_zones", []):
                print(f"  Thermal: {zone['zone_id']} ({zone['type']}): {zone['temp_c']:.1f}°C")
            for can in results.get("can_interfaces", []):
                print(f"  CAN: {can['device_id']} ({can.get('operstate', '?')})")
            for s in results.get("serial_devices", []):
                print(f"  Serial: {s['path']} VID={s['vendor_id']} PID={s['product_id']}")
        return

    if args.continuous:
        engine.start()
        try:
            while True:
                time.sleep(1.0)
                summary = engine.get_status_summary()
                print(f"\r[Scan #{engine._scan_count}] "
                      f"Devices: {summary['connected']}✓ {summary['degraded']}⚠ "
                      f"{summary['disconnected']}✗ | "
                      f"Max Temp: {summary['thermal']['max_temp_c']:.1f}°C", end="")
        except KeyboardInterrupt:
            engine.stop()
            print("\nStopped.")


if __name__ == "__main__":
    _main_cli()
