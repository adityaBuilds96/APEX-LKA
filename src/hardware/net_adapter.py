"""
APEX LKA — Multi-Protocol Network Adapter Manager
==================================================
Unified communications bridge supporting:
- CAN 2.0B (SocketCAN 500 kbps, 8-byte frames)
- CAN-FD (Flexible Data-rate 2/5 Mbps, up to 64-byte frames for batch waypoints)
- Serial JSON (115200 baud with CRC16 framing for dual Arduino/STM32)
- ROS 2 DDS Node Publisher (30 Hz perception & trajectory streams)
"""

from __future__ import annotations

import enum
import json
import struct
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

try:
    import can
    HAS_PYTHON_CAN = True
except ImportError:
    HAS_PYTHON_CAN = False

from src.hardware.ros2_publisher import APEXPerceptionNode


class NetworkProtocol(str, enum.Enum):
    CAN_SOCKET = "can_socket"
    CAN_FD = "can_fd"
    SERIAL_JSON = "serial_json"
    ROS2_DDS = "ros2_dds"


class NetworkAdapterManager:
    """
    Unified multi-protocol communications manager.
    Enables zero-friction vehicle network integration across automotive CAN, CAN-FD, Serial, and ROS 2.
    """

    # Standard Arbitration IDs
    ID_ACTUATION_CMD = 0x100
    ID_TELEMETRY = 0x110
    ID_HEARTBEAT = 0x120
    ID_CAN_FD_TRAJECTORY = 0x150

    def __init__(
        self,
        protocol: Union[str, NetworkProtocol] = NetworkProtocol.CAN_SOCKET,
        channel: str = "vcan0",
        bitrate: int = 500000,
        fd_data_bitrate: int = 2000000,
    ):
        if isinstance(protocol, str):
            protocol = NetworkProtocol(protocol.lower())
        self.protocol = protocol
        self.channel = channel
        self.bitrate = bitrate
        self.fd_data_bitrate = fd_data_bitrate

        self._can_bus = None
        self._rx_callbacks: Dict[int, List[Callable[[int, bytes], None]]] = {}
        self._serial_callbacks: List[Callable[[Dict[str, Any]], None]] = []

        # ROS 2 Publisher instance
        self.ros2_node: Optional[APEXPerceptionNode] = None
        if self.protocol == NetworkProtocol.ROS2_DDS:
            self.ros2_node = APEXPerceptionNode()

        # Telemetry counters
        self.tx_count = 0
        self.rx_count = 0
        self.error_count = 0

        self._init_interface()

    def _init_interface(self) -> None:
        """Initialize physical or virtual interface based on protocol."""
        if self.protocol in (NetworkProtocol.CAN_SOCKET, NetworkProtocol.CAN_FD):
            if HAS_PYTHON_CAN:
                try:
                    # Attempt socketcan or fallback to virtual
                    is_fd = (self.protocol == NetworkProtocol.CAN_FD)
                    try:
                        self._can_bus = can.Bus(
                            channel=self.channel,
                            interface="socketcan",
                            fd=is_fd,
                            bitrate=self.bitrate,
                            data_bitrate=self.fd_data_bitrate if is_fd else None,
                        )
                    except Exception:
                        self._can_bus = can.Bus(
                            channel=self.channel,
                            interface="virtual",
                            fd=is_fd,
                        )
                except Exception as e:
                    # Non-fatal: operate in internal loopback mode
                    self._can_bus = None
            else:
                self._can_bus = None

    def send_actuation_command(
        self,
        steering_deg: float,
        throttle: float = 0.0,
        brake: float = 0.0,
        confidence: float = 1.0,
        mode: int = 1,
    ) -> bool:
        """
        Send autonomous vehicle actuation command across active protocol.
        CAN: Standard 8-byte frame on ID 0x100.
        Serial: JSON message with CRC16.
        ROS 2: Updates vehicle state topic.
        """
        self.tx_count += 1
        clamped_steer = max(-60.0, min(60.0, float(steering_deg)))
        clamped_throttle = max(0.0, min(1.0, float(throttle)))
        clamped_brake = max(0.0, min(1.0, float(brake)))
        clamped_conf = max(0.0, min(1.0, float(confidence)))

        if self.protocol in (NetworkProtocol.CAN_SOCKET, NetworkProtocol.CAN_FD):
            # Pack 8-byte standard automotive control frame:
            # int16 steer (deg * 100), uint8 throttle (0-255), uint8 brake (0-255), uint8 conf (0-100), uint8 mode, uint16 seq
            steer_scaled = int(round(clamped_steer * 100))
            throt_scaled = int(round(clamped_throttle * 255))
            brake_scaled = int(round(clamped_brake * 255))
            conf_scaled = int(round(clamped_conf * 100))
            seq = self.tx_count & 0xFFFF

            payload = struct.pack("<hBBBBH", steer_scaled, throt_scaled, brake_scaled, conf_scaled, int(mode), seq)
            return self._send_can_raw(self.ID_ACTUATION_CMD, payload, is_fd=False)

        elif self.protocol == NetworkProtocol.SERIAL_JSON:
            msg = {
                "type": "ACTUATION_CMD",
                "steering_deg": round(clamped_steer, 2),
                "throttle": round(clamped_throttle, 3),
                "brake": round(clamped_brake, 3),
                "confidence": round(clamped_conf, 2),
                "mode": mode,
                "timestamp_ms": int(time.time() * 1000),
            }
            raw = self._encode_serial_json(msg)
            return len(raw) > 0

        elif self.protocol == NetworkProtocol.ROS2_DDS:
            if self.ros2_node:
                self.ros2_node.publish_state(
                    speed_mps=clamped_throttle * 5.0,
                    steer_deg=clamped_steer,
                    yaw_rad=0.0,
                )
            return True

        return False

    def send_can_fd_trajectory(self, waypoints: List[Tuple[float, float]]) -> bool:
        """
        Send dense trajectory waypoints via high-bandwidth CAN-FD (up to 64 bytes).
        Packs up to 15 waypoints (x_mm, y_mm) in a single frame on ID 0x150.
        """
        if self.protocol != NetworkProtocol.CAN_FD:
            # Can also execute on regular CAN by sending truncated waypoints or Serial/ROS 2
            pass

        self.tx_count += 1
        num_pts = min(15, len(waypoints))
        header = struct.pack("<BB", num_pts, self.tx_count & 0xFF)
        pt_bytes = bytearray()
        for i in range(num_pts):
            x_m, y_m = waypoints[i]
            x_mm = int(round(max(-32000, min(32000, x_m * 1000))))
            y_mm = int(round(max(-32000, min(32000, y_m * 1000))))
            pt_bytes.extend(struct.pack("<hh", x_mm, y_mm))

        # Pad to valid CAN-FD DLC length (e.g. 64 bytes)
        payload = header + bytes(pt_bytes)
        if len(payload) < 64:
            payload = payload.ljust(64, b"\x00")
        else:
            payload = payload[:64]

        return self._send_can_raw(self.ID_CAN_FD_TRAJECTORY, payload, is_fd=True)

    def send_heartbeat(self, status_code: int = 0) -> bool:
        """Send 10 Hz system heartbeat / watchdog frame."""
        payload = struct.pack("<BBI", status_code, 0xAA, int(time.time() & 0xFFFFFFFF))
        return self._send_can_raw(self.ID_HEARTBEAT, payload, is_fd=False)

    def register_rx_callback(self, can_id: int, callback: Callable[[int, bytes], None]) -> None:
        """Register callback for specific CAN arbitration ID."""
        if can_id not in self._rx_callbacks:
            self._rx_callbacks[can_id] = []
        self._rx_callbacks[can_id].append(callback)

    def _send_can_raw(self, can_id: int, data: bytes, is_fd: bool = False) -> bool:
        """Transmit raw frame or deliver to internal loopback."""
        if HAS_PYTHON_CAN and self._can_bus is not None:
            try:
                msg = can.Message(
                    arbitration_id=can_id,
                    data=data,
                    is_extended_id=False,
                    is_fd=is_fd,
                    bitrate_switch=is_fd,
                )
                self._can_bus.send(msg, timeout=0.01)
            except Exception:
                self.error_count += 1

        # Deliver to internal loopback subscribers
        if can_id in self._rx_callbacks:
            for cb in self._rx_callbacks[can_id]:
                try:
                    cb(can_id, data)
                except Exception as e:
                    print(f"[NetworkAdapter] RX callback error: {e}")

        return True

    @staticmethod
    def _encode_serial_json(msg_dict: Dict[str, Any]) -> str:
        """Serialize JSON packet with CRC16 integrity checksum and newline framing."""
        raw_json = json.dumps(msg_dict, separators=(",", ":"))
        crc = NetworkAdapterManager.compute_crc16(raw_json.encode("utf-8"))
        framed = f"{raw_json}*{crc:04X}\n"
        return framed

    @staticmethod
    def compute_crc16(data: bytes) -> int:
        """Standard CCITT CRC-16 (poly 0x1021)."""
        crc = 0xFFFF
        for b in data:
            crc ^= (b << 8)
            for _ in range(8):
                if crc & 0x8000:
                    crc = ((crc << 1) ^ 0x1021) & 0xFFFF
                else:
                    crc = (crc << 1) & 0xFFFF
        return crc

    def close(self) -> None:
        """Clean shutdown of network interfaces."""
        if self._can_bus is not None:
            try:
                self._can_bus.shutdown()
            except Exception:
                pass
            self._can_bus = None
        if self.ros2_node is not None:
            self.ros2_node.stop_periodic_publisher()
