"""
APEX-LKA v4.0 — Hardware Auto-Discovery HAL Test Suite
======================================================
Comprehensive test suite validating:
  - Filesystem scanner mocking for /dev and /sys topology
  - Dynamic device connection, degradation, disconnection, and recovery state machines
  - Event callback propagation and Streamlit HUD notification formatting
  - Driver reinitialization on device hot-plug
  - Serial-to-SocketCAN fallback transitions and dual Arduino management
  - Jetson Orin thermal warning and throttling event triggers
  - Non-blocking background thread lifecycle and thread safety
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, PropertyMock, mock_open, patch

import pytest

from src.hardware.auto_discovery import (
    AutoDiscoveryConfig,
    AutoDiscoveryEngine,
    DeviceRecord,
    DeviceState,
    DeviceStateMachine,
    DeviceType,
    DriverReinitRegistry,
    FilesystemScanner,
    HardwareEvent,
    HardwareEventType,
    ThermalStatus,
    SCAN_INTERVAL_S,
    STALE_THRESHOLD_S,
    DEGRADED_THRESHOLD_S,
    LOST_THRESHOLD_S,
    THERMAL_LIMIT_C,
    THERMAL_WARNING_C,
)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def base_config() -> AutoDiscoveryConfig:
    return AutoDiscoveryConfig(
        scan_interval_s=0.05,
        thermal_limit_c=80.0,
        thermal_warning_c=70.0,
        enable_camera_scan=True,
        enable_can_scan=True,
        enable_serial_scan=True,
        enable_thermal_scan=True,
        enable_driver_reinit=True,
        hud_notification_queue_size=50,
        event_log_size=100,
    )


@pytest.fixture
def engine(base_config: AutoDiscoveryConfig) -> AutoDiscoveryEngine:
    return AutoDiscoveryEngine(config=base_config)


@pytest.fixture
def sample_device_record() -> DeviceRecord:
    return DeviceRecord(
        device_id="test_can0",
        device_type=DeviceType.SOCKETCAN,
        state=DeviceState.UNCONFIGURED,
        path="/sys/class/net/can0",
        vendor_id="10EC",
        product_id="8168",
        description="Mock SocketCAN Interface",
    )


# ===========================================================================
# Section 1: DeviceRecord & HardwareEvent Unit Tests
# ===========================================================================

class TestDeviceRecord:
    """Validates DeviceRecord initialization, state changes, and age calculation."""

    def test_default_values(self):
        rec = DeviceRecord(device_id="cam1", device_type=DeviceType.CAMERA_USB)
        assert rec.device_id == "cam1"
        assert rec.device_type == DeviceType.CAMERA_USB
        assert rec.state == DeviceState.UNCONFIGURED
        assert rec.error_count == 0
        assert rec.reinit_count == 0
        assert rec.thermal_temp_c == 0.0

    def test_age_calculation(self):
        now = time.monotonic()
        rec = DeviceRecord(
            device_id="ard1",
            device_type=DeviceType.ARDUINO_PRIMARY,
            last_seen_time=now - 5.0,
        )
        assert abs(rec.age_s - 5.0) < 0.2

    def test_age_when_never_seen(self):
        rec = DeviceRecord(device_id="dev0", device_type=DeviceType.UNKNOWN)
        assert rec.age_s == float("inf")

    def test_is_stale_flag(self):
        now = time.monotonic()
        fresh = DeviceRecord(
            device_id="fresh",
            device_type=DeviceType.SOCKETCAN,
            last_seen_time=now - 0.5,
        )
        assert not fresh.is_stale

        stale = DeviceRecord(
            device_id="stale",
            device_type=DeviceType.SOCKETCAN,
            last_seen_time=now - (STALE_THRESHOLD_S + 1.0),
        )
        assert stale.is_stale

    def test_to_dict_structure(self):
        rec = DeviceRecord(
            device_id="mti_1",
            device_type=DeviceType.MTI680,
            state=DeviceState.CONNECTED,
            path="/dev/ttyUSB0",
            vendor_id="10C4",
            product_id="EA60",
            description="MTi-680 GNSS/INS",
            last_seen_time=time.monotonic(),
        )
        d = rec.to_dict()
        assert d["device_id"] == "mti_1"
        assert d["type"] == "mti680"
        assert d["state"] == "CONNECTED"
        assert d["path"] == "/dev/ttyUSB0"
        assert d["vendor_id"] == "10C4"
        assert "age_s" in d


class TestHardwareEvent:
    """Validates JSON event serialization and Streamlit HUD notification formatting."""

    def test_to_json_valid_json(self):
        evt = HardwareEvent(
            event_type=HardwareEventType.CONNECTED,
            device_id="can0",
            device_type=DeviceType.SOCKETCAN,
            device_path="/sys/class/net/can0",
            timestamp=1729271182.456,
            old_state=DeviceState.UNCONFIGURED,
            new_state=DeviceState.CONNECTED,
            message="CAN link up at 500 kbps",
            metadata={"bitrate": "500000", "operstate": "up"},
        )
        payload = evt.to_json()
        parsed = json.loads(payload)
        assert parsed["event"] == "connected"
        assert parsed["device_id"] == "can0"
        assert parsed["old_state"] == "UNCONFIGURED"
        assert parsed["new_state"] == "CONNECTED"
        assert parsed["metadata"]["bitrate"] == "500000"

    def test_hud_alert_severity_mapping(self):
        cases = [
            (HardwareEventType.THERMAL_CRIT, "error"),
            (HardwareEventType.DISCONNECTED, "warning"),
            (HardwareEventType.DEGRADED, "warning"),
            (HardwareEventType.THERMAL_WARN, "info"),
            (HardwareEventType.CONNECTED, "success"),
            (HardwareEventType.RECOVERED, "success"),
            (HardwareEventType.DRIVER_REINIT, "info"),
        ]
        for evt_type, expected_sev in cases:
            evt = HardwareEvent(
                event_type=evt_type,
                device_id="dummy",
                device_type=DeviceType.UNKNOWN,
                device_path="/dev/null",
                timestamp=0.0,
            )
            alert = evt.to_hud_alert()
            assert alert["severity"] == expected_sev, f"Failed for {evt_type}"

    def test_hud_alert_dismissible_and_title(self):
        evt = HardwareEvent(
            event_type=HardwareEventType.RECOVERED,
            device_id="arduino_1",
            device_type=DeviceType.ARDUINO_PRIMARY,
            device_path="/dev/ttyACM0",
            timestamp=50.0,
            message="Arduino link restored",
        )
        alert = evt.to_hud_alert()
        assert alert["dismissible"] is True
        assert "RECOVERED" in alert["title"]
        assert alert["body"] == "Arduino link restored"


class TestThermalStatus:
    """ThermalStatus max_temp, warning, and critical threshold properties."""

    def test_max_temp_selection(self):
        ts = ThermalStatus(cpu_temp_c=65.0, gpu_temp_c=72.5, soc_temp_c=68.0)
        assert ts.max_temp_c == 72.5

    def test_warning_flag(self):
        normal = ThermalStatus(cpu_temp_c=55.0, gpu_temp_c=60.0)
        assert not normal.is_warning
        assert not normal.is_critical

        warning = ThermalStatus(cpu_temp_c=71.0, gpu_temp_c=68.0)
        assert warning.is_warning
        assert not warning.is_critical

    def test_critical_flag(self):
        critical = ThermalStatus(cpu_temp_c=81.0, gpu_temp_c=75.0)
        assert critical.is_critical
        assert not critical.is_warning


# ===========================================================================
# Section 2: DeviceStateMachine Hysteresis & Transition Tests
# ===========================================================================

class TestDeviceStateMachine:
    """Validates the UNCONFIGURED → CONNECTED → DEGRADED → DISCONNECTED lifecycle."""

    def test_unconfigured_to_connected_on_seen(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        new_state = sm.on_device_seen(now)
        assert new_state == DeviceState.CONNECTED
        assert sample_device_record.state == DeviceState.CONNECTED
        assert sample_device_record.connected_since == now

    def test_stay_connected_on_subsequent_seen(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        sm.on_device_seen(now)
        new_state = sm.on_device_seen(now + 0.5)
        assert new_state is None
        assert sample_device_record.state == DeviceState.CONNECTED

    def test_transition_to_degraded_after_missing_threshold(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        sm.on_device_seen(now)

        # Missing scans below degraded threshold (5 seconds / 0.5s = 10 scans)
        for i in range(1, 10):
            res = sm.on_device_missing(now + i * SCAN_INTERVAL_S)
            assert res is None
            assert sample_device_record.state == DeviceState.CONNECTED

        # Trigger degraded state at scan 10 (10 * 0.5s = 5.0s)
        res = sm.on_device_missing(now + 10 * SCAN_INTERVAL_S)
        assert res == DeviceState.DEGRADED
        assert sample_device_record.state == DeviceState.DEGRADED

    def test_transition_to_disconnected_after_lost_threshold(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        sm.on_device_seen(now)

        # 30 missing scans = 15.0 seconds
        for i in range(1, 30):
            sm.on_device_missing(now + i * SCAN_INTERVAL_S)

        res = sm.on_device_missing(now + 30 * SCAN_INTERVAL_S)
        assert res == DeviceState.DISCONNECTED
        assert sample_device_record.state == DeviceState.DISCONNECTED

    def test_recovery_from_degraded_to_connected(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        sm.on_device_seen(now)

        # Drive to DEGRADED
        for i in range(1, 11):
            sm.on_device_missing(now + i * SCAN_INTERVAL_S)
        assert sample_device_record.state == DeviceState.DEGRADED

        # Device seen again
        new_state = sm.on_device_seen(now + 6.0)
        assert new_state == DeviceState.CONNECTED
        assert sample_device_record.state == DeviceState.CONNECTED

    def test_recovery_from_disconnected_to_connected(self, sample_device_record: DeviceRecord):
        sm = DeviceStateMachine(sample_device_record)
        now = time.monotonic()
        sm.on_device_seen(now)

        # Drive to DISCONNECTED
        for i in range(1, 31):
            sm.on_device_missing(now + i * SCAN_INTERVAL_S)
        assert sample_device_record.state == DeviceState.DISCONNECTED

        # Reconnection
        new_state = sm.on_device_seen(now + 20.0)
        assert new_state == DeviceState.CONNECTED
        assert sample_device_record.state == DeviceState.CONNECTED

    def test_unconfigured_missing_does_not_transition(self, sample_device_record: DeviceRecord):
        sample_device_record.state = DeviceState.UNCONFIGURED
        sm = DeviceStateMachine(sample_device_record)
        res = sm.on_device_missing(time.monotonic())
        assert res is None
        assert sample_device_record.state == DeviceState.UNCONFIGURED


# ===========================================================================
# Section 3: DriverReinitRegistry Unit Tests
# ===========================================================================

class TestDriverReinitRegistry:
    """Validates registration and invocation of hot-plug driver reinit handlers."""

    def test_register_and_trigger(self, sample_device_record: DeviceRecord):
        reg = DriverReinitRegistry()
        calls = []

        def handler1(dev: DeviceRecord):
            calls.append(f"h1:{dev.device_id}")

        def handler2(dev: DeviceRecord):
            calls.append(f"h2:{dev.device_id}")

        reg.register(DeviceType.SOCKETCAN, handler1)
        reg.register(DeviceType.SOCKETCAN, handler2)

        count = reg.trigger(sample_device_record)
        assert count == 2
        assert "h1:test_can0" in calls
        assert "h2:test_can0" in calls

    def test_trigger_unregistered_type_returns_zero(self, sample_device_record: DeviceRecord):
        reg = DriverReinitRegistry()
        sample_device_record.device_type = DeviceType.CAMERA_CSI
        count = reg.trigger(sample_device_record)
        assert count == 0

    def test_handler_exception_isolated(self, sample_device_record: DeviceRecord):
        reg = DriverReinitRegistry()
        good_called = []

        def faulty_handler(dev: DeviceRecord):
            raise RuntimeError("Driver init hardware crash!")

        def safe_handler(dev: DeviceRecord):
            good_called.append(dev.device_id)

        reg.register(DeviceType.SOCKETCAN, faulty_handler)
        reg.register(DeviceType.SOCKETCAN, safe_handler)

        count = reg.trigger(sample_device_record)
        assert count == 1
        assert "test_can0" in good_called


# ===========================================================================
# Section 4: FilesystemScanner Mocked Unit Tests
# ===========================================================================

class TestFilesystemScannerMocked:
    """Mocks Linux sysfs / devfs hierarchy to test camera, CAN, serial, and thermal scanner."""

    @patch("glob.glob")
    def test_scan_cameras_csi_detected(self, mock_glob):
        mock_glob.side_effect = lambda pat: ["/dev/video0"] if "video" in pat else []

        with patch("builtins.open", mock_open(read_data="imx219_csi_camera\n")):
            cameras = FilesystemScanner.scan_cameras()

        assert len(cameras) >= 1
        cam = cameras[0]
        assert cam["path"] == "/dev/video0"
        assert cam["type"] == "csi"
        assert "imx219" in cam["description"]

    @patch("glob.glob")
    @patch("os.path.realpath", side_effect=lambda p: "/dev/video1" if "by-id" in p else p)
    def test_scan_cameras_by_id_usb(self, mock_realpath, mock_glob):
        mock_glob.side_effect = lambda pat: (
            ["/dev/v4l/by-id/usb-ArduCam_USB_Camera-video-index0"] if "by-id" in pat else []
        )
        cameras = FilesystemScanner.scan_cameras()
        assert len(cameras) == 1
        assert cameras[0]["type"] == "usb"
        assert "usb-ArduCam" in cameras[0]["device_id"]

    def test_scan_socketcan_active_can0(self):
        can0_mock = Path("/sys/class/net/can0")

        def mock_open_impl(p, *args, **kwargs):
            p_str = str(p).replace("\\", "/")
            if "type" in p_str:
                return mock_open(read_data="280\n")()
            if "operstate" in p_str:
                return mock_open(read_data="up\n")()
            return mock_open(read_data="")()

        with patch("pathlib.Path.exists", return_value=True), \
             patch("pathlib.Path.iterdir", return_value=[can0_mock]), \
             patch("builtins.open", side_effect=mock_open_impl), \
             patch.object(FilesystemScanner, "_get_can_bitrate", return_value="500000"):
            can_ifaces = FilesystemScanner.scan_socketcan()

        assert len(can_ifaces) == 1
        assert can_ifaces[0]["device_id"] == "can0"
        assert can_ifaces[0]["operstate"] == "up"
        assert can_ifaces[0]["bitrate"] == "500000"

    @patch("glob.glob")
    @patch.object(FilesystemScanner, "_get_usb_info")
    def test_scan_serial_devices_arduino_and_mti(self, mock_usb_info, mock_glob):
        mock_glob.side_effect = lambda pat: (
            ["/dev/ttyACM0", "/dev/ttyUSB0"] if "tty" in pat else []
        )

        def usb_info_impl(dev_path):
            if "ttyACM0" in dev_path:
                return {
                    "vendor_id": "2341",
                    "product_id": "0042",
                    "serial_number": "ARDUINO_MEGA_PRIMARY",
                    "description": "Arduino Mega 2560",
                }
            elif "ttyUSB0" in dev_path:
                return {
                    "vendor_id": "10C4",
                    "product_id": "EA60",
                    "serial_number": "MTI680_INS",
                    "description": "Xsens MTi-680",
                }
            return {}

        mock_usb_info.side_effect = usb_info_impl
        serials = FilesystemScanner.scan_serial_devices()

        assert len(serials) == 2
        paths = [s["path"] for s in serials]
        assert "/dev/ttyACM0" in paths
        assert "/dev/ttyUSB0" in paths

    @patch("glob.glob")
    def test_scan_thermal_zones(self, mock_glob):
        mock_glob.return_value = [
            "/sys/class/thermal/thermal_zone0/temp",
            "/sys/class/thermal/thermal_zone1/temp",
        ]

        def open_thermal_impl(path, *args, **kwargs):
            if "zone0" in str(path):
                return mock_open(read_data="65000\n")()
            elif "zone1" in str(path):
                return mock_open(read_data="72500\n")()
            return mock_open(read_data="0\n")()

        with patch("builtins.open", side_effect=open_thermal_impl):
            zones = FilesystemScanner.scan_thermal()

        assert len(zones) == 2
        assert zones[0]["temp_c"] == 65.0
        assert zones[1]["temp_c"] == 72.5


# ===========================================================================
# Section 5: Dynamic Device Connection & Disconnection Engine Tests
# ===========================================================================

class TestEngineEventCallbacks:
    """Verifies callbacks, event logs, and status summaries across dynamic hot-plugs."""

    def test_device_connection_fires_callback(self, engine: AutoDiscoveryEngine):
        events_received = []
        engine.register_event_callback(lambda e: events_received.append(e))

        fake_cameras = [{
            "path": "/dev/video0",
            "device_id": "video0",
            "description": "USB Cam",
            "type": "usb",
        }]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=fake_cameras), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        assert len(events_received) >= 1
        connect_evt = [e for e in events_received if e.event_type == HardwareEventType.CONNECTED][0]
        assert connect_evt.device_id == "camera_video0"
        assert connect_evt.new_state == DeviceState.CONNECTED

    def test_device_disconnection_fires_event(self, engine: AutoDiscoveryEngine):
        events_received = []
        engine.register_event_callback(lambda e: events_received.append(e))

        fake_can = [{
            "path": "/sys/class/net/can0",
            "device_id": "can0",
            "description": "SocketCAN can0",
            "operstate": "up",
            "bitrate": "500000",
        }]

        # 1. Connect can0
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=fake_can), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        dev = engine.get_device("can_can0")
        assert dev is not None
        assert dev.state == DeviceState.CONNECTED

        # 2. Simulate unplugging can0 across scans to drive state machine to DISCONNECTED
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            for _ in range(35):
                engine.run_once()

        disconnect_evts = [e for e in events_received if e.event_type == HardwareEventType.DISCONNECTED]
        assert len(disconnect_evts) >= 1
        assert disconnect_evts[0].device_id == "can_can0"
        assert disconnect_evts[0].new_state == DeviceState.DISCONNECTED

    def test_hud_queue_notification_delivery(self, engine: AutoDiscoveryEngine):
        fake_serials = [{
            "path": "/dev/ttyACM0",
            "device_id": "ttyACM0",
            "vendor_id": "2341",
            "product_id": "0042",
            "serial_number": "SN_MEGA",
            "description": "Arduino Mega 2560",
        }]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_serials), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        alert = engine.get_hud_notification()
        assert alert is not None
        assert alert["severity"] == "success"
        assert "CONNECTED" in alert["title"]
        assert alert["device_id"].startswith("arduino_1_")

    def test_driver_reinit_invoked_on_reconnection(self, engine: AutoDiscoveryEngine):
        reinit_invoked = []
        engine.register_driver_reinit(
            DeviceType.ARDUINO_PRIMARY,
            lambda dev: reinit_invoked.append(dev.device_id),
        )

        fake_arduino = [{
            "path": "/dev/ttyACM0",
            "device_id": "ttyACM0",
            "vendor_id": "2341",
            "product_id": "0042",
            "serial_number": "A1",
            "description": "Arduino Mega 2560",
        }]

        # Connect
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_arduino), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        assert len(reinit_invoked) == 1

        # Disconnect over 35 cycles
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            for _ in range(35):
                engine.run_once()

        # Reconnect
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_arduino), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        assert len(reinit_invoked) == 2


# ===========================================================================
# Section 6: Serial-to-SocketCAN Fallback & Dual Arduino Topology
# ===========================================================================

class TestSerialToCANFallback:
    """Validates SocketCAN primary mode and seamless fallback to dual serial Arduinos."""

    def test_dual_mode_active_topology(self, engine: AutoDiscoveryEngine):
        fake_can = [{
            "path": "/sys/class/net/can0",
            "device_id": "can0",
            "description": "SocketCAN can0",
            "operstate": "up",
            "bitrate": "500000",
        }]
        fake_serials = [
            {
                "path": "/dev/ttyACM0",
                "device_id": "ttyACM0",
                "vendor_id": "2341",
                "product_id": "0042",
                "serial_number": "ARD_PRI",
                "description": "Arduino Mega Primary",
            },
            {
                "path": "/dev/ttyACM1",
                "device_id": "ttyACM1",
                "vendor_id": "2341",
                "product_id": "0042",
                "serial_number": "ARD_SEC",
                "description": "Arduino Mega Secondary",
            },
        ]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=fake_can), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_serials), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        can_ifaces = engine.get_connected_can_interfaces()
        assert "can0" in can_ifaces

        arduinos = engine.get_connected_arduinos()
        assert len(arduinos) == 2
        types = {a.device_type for a in arduinos}
        assert DeviceType.ARDUINO_PRIMARY in types
        assert DeviceType.ARDUINO_SECONDARY in types

    def test_socketcan_down_fallback_to_serial_available(self, engine: AutoDiscoveryEngine):
        fake_serials = [{
            "path": "/dev/ttyACM0",
            "device_id": "ttyACM0",
            "vendor_id": "2341",
            "product_id": "0042",
            "serial_number": "PRI_1",
            "description": "Arduino Mega",
        }]

        # Run with CAN disabled
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_serials), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        can_ifaces = engine.get_connected_can_interfaces()
        assert len(can_ifaces) == 0

        arduinos = engine.get_connected_arduinos()
        assert len(arduinos) == 1
        assert arduinos[0].device_type == DeviceType.ARDUINO_PRIMARY
        assert arduinos[0].state == DeviceState.CONNECTED

    def test_dual_arduino_primary_and_secondary_classification(self, engine: AutoDiscoveryEngine):
        fake_serials = [
            {
                "path": "/dev/ttyACM0",
                "device_id": "ttyACM0",
                "vendor_id": "2341",
                "product_id": "0042",
                "description": "Arduino",
            },
            {
                "path": "/dev/ttyACM1",
                "device_id": "ttyACM1",
                "vendor_id": "2341",
                "product_id": "0042",
                "description": "Arduino",
            },
            {
                "path": "/dev/ttyUSB0",
                "device_id": "ttyUSB0",
                "vendor_id": "10C4",
                "product_id": "EA60",
                "description": "Xsens MTi",
            },
        ]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=fake_serials), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.run_once()

        summary = engine.get_status_summary()
        assert summary["arduino_primary_connected"] is True
        assert summary["arduino_secondary_connected"] is True
        assert summary["mti680_connected"] is True


# ===========================================================================
# Section 7: Thermal Safety & Budget Limit Tests
# ===========================================================================

class TestThermalSafety:
    """Tests Jetson Orin NX thermal monitoring and threshold event dispatches."""

    def test_thermal_warning_event_fires_at_threshold(self, engine: AutoDiscoveryEngine):
        events_received = []
        engine.register_event_callback(lambda e: events_received.append(e))

        zones = [
            {"path": "/sys/class/thermal/thermal_zone0/temp", "zone": "thermal_zone0", "temp_c": 72.0},
            {"path": "/sys/class/thermal/thermal_zone1/temp", "zone": "thermal_zone1", "temp_c": 68.0},
        ]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=zones):
            engine.run_once()

        warn_evts = [e for e in events_received if e.event_type == HardwareEventType.THERMAL_WARN]
        assert len(warn_evts) == 1
        assert "72.0°C" in warn_evts[0].message

        ts = engine.get_thermal_status()
        assert ts.is_warning
        assert not ts.is_critical

    def test_thermal_critical_event_and_throttling(self, engine: AutoDiscoveryEngine):
        events_received = []
        engine.register_event_callback(lambda e: events_received.append(e))

        zones = [
            {"path": "/sys/class/thermal/thermal_zone0/temp", "zone": "thermal_zone0", "temp_c": 83.5},
            {"path": "/sys/class/thermal/thermal_zone1/temp", "zone": "thermal_zone1", "temp_c": 81.0},
        ]

        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=zones):
            engine.run_once()

        crit_evts = [e for e in events_received if e.event_type == HardwareEventType.THERMAL_CRIT]
        assert len(crit_evts) == 1
        assert "CRITICAL" in crit_evts[0].message

        ts = engine.get_thermal_status()
        assert ts.is_critical


# ===========================================================================
# Section 8: Background Thread Lifecycle & HUD Queue Operation
# ===========================================================================

class TestBackgroundThreadLifecycle:
    """Verifies non-blocking background daemon thread start, stop, and queue retrieval."""

    def test_start_and_stop_daemon(self, engine: AutoDiscoveryEngine):
        with patch.object(FilesystemScanner, "scan_cameras", return_value=[]), \
             patch.object(FilesystemScanner, "scan_socketcan", return_value=[]), \
             patch.object(FilesystemScanner, "scan_serial_devices", return_value=[]), \
             patch.object(FilesystemScanner, "scan_thermal", return_value=[]):
            engine.start()
            assert engine.is_running
            time.sleep(0.12)
            engine.stop()
            assert not engine.is_running

    def test_status_summary_schema(self, engine: AutoDiscoveryEngine):
        summary = engine.get_status_summary()
        required_keys = [
            "total_devices",
            "connected_devices",
            "degraded_devices",
            "disconnected_devices",
            "max_temp_c",
            "thermal_warning",
            "thermal_critical",
            "arduino_primary_connected",
            "arduino_secondary_connected",
            "mti680_connected",
            "can_interfaces",
            "is_running",
            "devices",
        ]
        for k in required_keys:
            assert k in summary, f"Missing key in summary: {k}"

    def test_event_log_capacity_cap(self, base_config: AutoDiscoveryConfig):
        base_config.event_log_size = 5
        eng = AutoDiscoveryEngine(config=base_config)

        for i in range(10):
            eng._fire_event(HardwareEvent(
                event_type=HardwareEventType.STATE_CHANGE,
                device_id=f"dev_{i}",
                device_type=DeviceType.UNKNOWN,
                device_path="/dev/null",
                timestamp=float(i),
            ))

        log_events = eng.get_event_log(limit=20)
        assert len(log_events) <= 5

    def test_hud_queue_non_blocking_empty_returns_none(self, engine: AutoDiscoveryEngine):
        # Non-blocking get from empty queue must return None immediately
        item = engine.get_hud_notification(timeout=0.0)
        assert item is None
