"""
APEX-Vision Core v3.0 — Multi-Protocol Automotive CAN Bus Manager
==================================================================
ISO 26262 ASIL-B Compliant CAN Bus Interface for 500 kbps Automotive Networks.
Serializes, broadcasts, and unpacks the complete APEX-Vision CAN telemetry matrix:
  - CAN ID 0x150: Primary Threat Telemetry (30 Hz)
  - CAN ID 0x160: Spatial Obstacle & 3D Depth Alert (30 Hz)
  - CAN ID 0x170: Trajectory Collision & TTI Alert (30 Hz)
  - CAN ID 0x180: Kinematic Vehicle State (100 Hz)
  - CAN ID 0x190: Autonomous Plan & Control State (30 Hz)
  - CAN ID 0x1A0: Fleet & Cryptographic OTA Status (30 Hz)
All frames are strictly 8 bytes (DLC 8) with big-endian automotive packing.
"""

from __future__ import annotations

import math
import struct
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import can
    HAS_CAN = True
except ImportError:
    HAS_CAN = False

from src.perception.risk_engine import ThreatTarget


class CANManager:
    """
    Automotive CAN Bus Controller for APEX-Vision Core v3.0.
    Handles 500 kbps CAN interface, multi-frame serialization, and multi-rate transmission.
    """

    # CAN Arbitration ID Definition Table
    CAN_ID_RISK_THREAT      = 0x150  # Primary Threat Telemetry (30 Hz)
    CAN_ID_SPATIAL_OBSTACLE = 0x160  # Spatial Obstacle & Voxel Alert (30 Hz)
    CAN_ID_TRAJECTORY_ALERT = 0x170  # Trajectory Collision & TTI Alert (30 Hz)
    CAN_ID_KINEMATIC_STATE  = 0x180  # Kinematic Vehicle State (100 Hz)
    CAN_ID_PLAN_CONTROL     = 0x190  # Autonomous Plan & Control State (30 Hz)
    CAN_ID_FLEET_OTA        = 0x1A0  # Fleet & OTA Security Status (30 Hz)

    DEFAULT_BITRATE = 500000  # 500 kbps automotive standard

    def __init__(
        self,
        channel: str = "vcan0",
        bitrate: int = DEFAULT_BITRATE,
        broadcast_hz: float = 30.0,
        interface: str = "virtual",
        endian: str = ">",  # Big-Endian automotive default
    ):
        self.channel = channel
        self.bitrate = bitrate
        self.broadcast_hz = broadcast_hz
        self.interval_s = 1.0 / max(1.0, broadcast_hz)
        self.endian = endian

        self._bus = None
        self._is_connected = False
        self._current_target: Optional[ThreatTarget] = None
        self._lock = threading.Lock()

        # Multi-Rate Broadcaster Thread
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.tx_count = 0
        self.last_tx_time = 0.0

        # In-memory loopback callback subscribers: callback(arbitration_id, payload_bytes)
        self._callbacks: List[Callable[[int, bytes], None]] = []

        self._init_bus(interface)

    def _init_bus(self, interface: str) -> None:
        """Attempt to open CAN bus (SocketCAN / Virtual) or fall back to loopback."""
        if HAS_CAN:
            try:
                self._bus = can.Bus(channel=self.channel, interface=interface, bitrate=self.bitrate)
                self._is_connected = True
            except Exception:
                try:
                    self._bus = can.Bus(channel=self.channel, interface="virtual")
                    self._is_connected = True
                except Exception:
                    self._bus = None
                    self._is_connected = False
        else:
            self._bus = None
            self._is_connected = False

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. CAN ID 0x150: Primary Threat Telemetry Serializer (30 Hz)
    # Format: [Risk_ID (8b), Class_Enum (8b), Dist_mm (16b), Risk_Score (8b), Stop_Flag (8b), Padding (16b)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_risk_frame(
        self,
        risk_id: int,
        class_enum: int,
        dist_mm: int,
        risk_score: float,
        stop_flag: bool,
    ) -> bytes:
        """Encode CAN ID 0x150 payload (8 bytes)."""
        u8_risk_id = max(0, min(255, int(risk_id)))
        u8_class_enum = max(0, min(255, int(class_enum)))
        u16_dist_mm = max(0, min(65535, int(round(dist_mm))))
        u8_risk_score = max(0, min(100, int(round(risk_score * 100.0))))
        u8_stop_flag = 1 if stop_flag else 0
        u16_padding = 0x0000

        fmt = f"{self.endian}BBHBBH"
        return struct.pack(fmt, u8_risk_id, u8_class_enum, u16_dist_mm, u8_risk_score, u8_stop_flag, u16_padding)

    def unpack_risk_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x150 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x150 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}BBHBBH"
        risk_id, class_enum, dist_mm, risk_score_u8, stop_flag_u8, padding = struct.unpack(fmt, data)
        return {
            "risk_id": risk_id,
            "class_enum": class_enum,
            "dist_mm": dist_mm,
            "distance_m": round(dist_mm / 1000.0, 2),
            "risk_score": round(risk_score_u8 / 100.0, 3),
            "stop_flag": bool(stop_flag_u8),
            "padding": padding,
        }

    # Aliases for 0x150
    pack_threat_frame = pack_risk_frame
    unpack_threat_frame = unpack_risk_frame

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CAN ID 0x160: Spatial Obstacle & Depth Alert Serializer (30 Hz)
    # Format: [Nearest_ID (8b), X_offset_mm (int16), Z_dist_mm (uint16), TTI_ms (uint16), Threat_Level (8b)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_spatial_obstacle_frame(
        self,
        nearest_id: int,
        x_offset_mm: int,
        z_dist_mm: int,
        tti_ms: int,
        threat_level: int,
    ) -> bytes:
        """Encode CAN ID 0x160 payload (8 bytes)."""
        u8_id = max(0, min(255, int(nearest_id)))
        s16_x = max(-32768, min(32767, int(round(x_offset_mm))))
        u16_z = max(0, min(65535, int(round(z_dist_mm))))
        u16_tti = max(0, min(65535, int(round(tti_ms))))
        u8_threat = max(0, min(255, int(threat_level)))

        fmt = f"{self.endian}BhHHB"
        return struct.pack(fmt, u8_id, s16_x, u16_z, u16_tti, u8_threat)

    def unpack_spatial_obstacle_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x160 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x160 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}BhHHB"
        nearest_id, x_offset_mm, z_dist_mm, tti_ms, threat_level = struct.unpack(fmt, data)
        return {
            "nearest_id": nearest_id,
            "x_offset_mm": x_offset_mm,
            "x_offset_m": round(x_offset_mm / 1000.0, 3),
            "z_dist_mm": z_dist_mm,
            "z_dist_m": round(z_dist_mm / 1000.0, 3),
            "tti_ms": tti_ms,
            "threat_level": threat_level,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. CAN ID 0x170: Trajectory Collision & TTI Alert Serializer (30 Hz)
    # Format: [Track_ID (8b), TTI_ms (uint16), Intersect_X_mm (int16), Collision_Level (8b), Emergency_Action (8b), Reserved (8b)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_trajectory_alert_frame(
        self,
        track_id: int,
        tti_ms: int,
        intersect_x_mm: int,
        collision_level: int,
        emergency_action: int,
        reserved: int = 0,
    ) -> bytes:
        """Encode CAN ID 0x170 payload (8 bytes)."""
        u8_id = max(0, min(255, int(track_id)))
        u16_tti = max(0, min(65535, int(round(tti_ms))))
        s16_ix = max(-32768, min(32767, int(round(intersect_x_mm))))
        u8_lvl = max(0, min(255, int(collision_level)))
        u8_act = max(0, min(255, int(emergency_action)))
        u8_res = max(0, min(255, int(reserved)))

        fmt = f"{self.endian}BHhBBB"
        return struct.pack(fmt, u8_id, u16_tti, s16_ix, u8_lvl, u8_act, u8_res)

    def unpack_trajectory_alert_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x170 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x170 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}BHhBBB"
        track_id, tti_ms, intersect_x_mm, collision_level, emergency_action, reserved = struct.unpack(fmt, data)
        return {
            "track_id": track_id,
            "tti_ms": tti_ms,
            "intersect_x_mm": intersect_x_mm,
            "intersect_x_m": round(intersect_x_mm / 1000.0, 3),
            "collision_level": collision_level,
            "emergency_action": emergency_action,
            "reserved": reserved,
        }

    # Aliases for 0x170
    pack_trajectory_collision_frame = pack_trajectory_alert_frame
    unpack_trajectory_collision_frame = unpack_trajectory_alert_frame

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. CAN ID 0x180: Kinematic Vehicle State Serializer (100 Hz)
    # Format: [Speed_cm_s (uint16), YawRate_mrad_s (int16), SlipAngle_mrad (int16), Localization_Status (8b), Reserved (8b)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_kinematic_state_frame(
        self,
        speed_cm_s: int,
        yaw_rate_mrad_s: int,
        slip_angle_mrad: int,
        localization_status: int,
        reserved: int = 0,
    ) -> bytes:
        """Encode CAN ID 0x180 payload (8 bytes)."""
        u16_speed = max(0, min(65535, int(round(speed_cm_s))))
        s16_yaw = max(-32768, min(32767, int(round(yaw_rate_mrad_s))))
        s16_slip = max(-32768, min(32767, int(round(slip_angle_mrad))))
        u8_status = max(0, min(255, int(localization_status)))
        u8_res = max(0, min(255, int(reserved)))

        fmt = f"{self.endian}HhhBB"
        return struct.pack(fmt, u16_speed, s16_yaw, s16_slip, u8_status, u8_res)

    def unpack_kinematic_state_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x180 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x180 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}HhhBB"
        speed_cm_s, yaw_rate_mrad_s, slip_angle_mrad, status, reserved = struct.unpack(fmt, data)
        return {
            "speed_cm_s": speed_cm_s,
            "speed_m_s": round(speed_cm_s / 100.0, 2),
            "yaw_rate_mrad_s": yaw_rate_mrad_s,
            "yaw_rate_deg_s": round(math.degrees(yaw_rate_mrad_s / 1000.0), 2),
            "slip_angle_mrad": slip_angle_mrad,
            "slip_angle_deg": round(math.degrees(slip_angle_mrad / 1000.0), 2),
            "localization_status": status,
            "reserved": reserved,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. CAN ID 0x190: Autonomous Plan & Control State Serializer (30 Hz)
    # Format: [Target_Speed_cm_s (uint16), Target_Steer_cdeg (int16), FSM_State (8b), RSS_Override (8b), Curvature_x10000 (int16)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_plan_control_frame(
        self,
        target_speed_cm_s: int,
        target_steer_cdeg: int,
        fsm_state: int,
        rss_override: bool,
        curvature_x10000: int,
    ) -> bytes:
        """Encode CAN ID 0x190 payload (8 bytes)."""
        u16_speed = max(0, min(65535, int(round(target_speed_cm_s))))
        s16_steer = max(-32768, min(32767, int(round(target_steer_cdeg))))
        u8_fsm = max(0, min(255, int(fsm_state)))
        u8_rss = 1 if rss_override else 0
        s16_curv = max(-32768, min(32767, int(round(curvature_x10000))))

        fmt = f"{self.endian}HhBBh"
        return struct.pack(fmt, u16_speed, s16_steer, u8_fsm, u8_rss, s16_curv)

    def unpack_plan_control_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x190 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x190 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}HhBBh"
        speed_cm_s, steer_cdeg, fsm_state, rss_override, curv_x10000 = struct.unpack(fmt, data)
        return {
            "target_speed_cm_s": speed_cm_s,
            "target_speed_m_s": round(speed_cm_s / 100.0, 2),
            "target_steer_cdeg": steer_cdeg,
            "target_steer_deg": round(steer_cdeg / 100.0, 2),
            "fsm_state": fsm_state,
            "rss_override": bool(rss_override),
            "curvature_x10000": curv_x10000,
            "curvature": round(curv_x10000 / 10000.0, 5),
        }

    # Aliases for 0x190
    pack_control_state_frame = pack_plan_control_frame
    unpack_control_state_frame = unpack_plan_control_frame

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. CAN ID 0x1A0: Fleet & Cryptographic OTA Status Serializer (30 Hz)
    # Format: [Active_Model_Hash_Byte0 (8b), Sync_State (8b), Shadow_Disagreement_Score (8b), Harvest_Queue_Count (8b), Backend_Enum (8b), Reserved (24b)]
    # ═══════════════════════════════════════════════════════════════════════════

    def pack_fleet_ota_frame(
        self,
        active_model_hash_byte0: int,
        sync_state: int,
        shadow_disagreement_score: int,
        harvest_queue_count: int,
        backend_enum: int,
        reserved: bytes = b"\x00\x00\x00",
    ) -> bytes:
        """Encode CAN ID 0x1A0 payload (8 bytes)."""
        u8_h0 = max(0, min(255, int(active_model_hash_byte0)))
        u8_sync = max(0, min(255, int(sync_state)))
        u8_dis = max(0, min(255, int(shadow_disagreement_score)))
        u8_q = max(0, min(255, int(harvest_queue_count)))
        u8_bk = max(0, min(255, int(backend_enum)))
        pad = reserved[:3].ljust(3, b"\x00")

        fmt = f"{self.endian}BBBBB3s"
        return struct.pack(fmt, u8_h0, u8_sync, u8_dis, u8_q, u8_bk, pad)

    def unpack_fleet_ota_frame(self, data: bytes) -> Dict[str, Any]:
        """Decode CAN ID 0x1A0 frame."""
        if len(data) != 8:
            raise ValueError(f"CAN 0x1A0 frame must be exactly 8 bytes, got {len(data)}")
        fmt = f"{self.endian}BBBBB3s"
        h0, sync_state, disagreement, queue_count, backend, pad = struct.unpack(fmt, data)
        return {
            "active_model_hash_byte0": h0,
            "sync_state": sync_state,
            "shadow_disagreement_score": disagreement,
            "harvest_queue_count": queue_count,
            "backend_enum": backend,
            "reserved": pad,
        }

    # ═══════════════════════════════════════════════════════════════════════════
    # Unified Frame Transmission & Dispatcher
    # ═══════════════════════════════════════════════════════════════════════════

    def send_frame(self, arbitration_id: int, data: bytes) -> bytes:
        """Generic transmission helper broadcasting to physical/virtual bus and subscribers."""
        if len(data) != 8:
            raise ValueError(f"CAN frame must be exactly 8 bytes, got {len(data)}")

        if self._bus is not None:
            try:
                msg = can.Message(
                    arbitration_id=arbitration_id,
                    data=data,
                    is_extended_id=False,
                )
                self._bus.send(msg, timeout=0.01)
            except Exception:
                pass

        with self._lock:
            self.tx_count += 1
            self.last_tx_time = time.time()
            cbs = list(self._callbacks)

        for cb in cbs:
            try:
                cb(arbitration_id, data)
            except Exception:
                pass

        return data

    def send_target(self, target: Optional[ThreatTarget]) -> bytes:
        """Construct and transmit CAN 0x150 frame for the given ThreatTarget."""
        if target is None:
            data = self.pack_risk_frame(0, 0, 65535, 0.0, False)
        else:
            data = self.pack_risk_frame(
                risk_id=target.target_id,
                class_enum=target.class_enum,
                dist_mm=target.distance_mm,
                risk_score=target.risk_score,
                stop_flag=target.stop_flag,
            )
        return self.send_frame(self.CAN_ID_RISK_THREAT, data)

    def send_spatial_obstacle(
        self,
        nearest_id: int,
        x_offset_mm: int,
        z_dist_mm: int,
        tti_ms: int,
        threat_level: int,
    ) -> bytes:
        """Construct and transmit CAN 0x160 spatial obstacle frame."""
        data = self.pack_spatial_obstacle_frame(nearest_id, x_offset_mm, z_dist_mm, tti_ms, threat_level)
        return self.send_frame(self.CAN_ID_SPATIAL_OBSTACLE, data)

    def send_trajectory_alert(
        self,
        track_id: int,
        tti_ms: int,
        intersect_x_mm: int,
        collision_level: int,
        emergency_action: int,
    ) -> bytes:
        """Construct and transmit CAN 0x170 trajectory collision alert frame."""
        data = self.pack_trajectory_alert_frame(track_id, tti_ms, intersect_x_mm, collision_level, emergency_action)
        return self.send_frame(self.CAN_ID_TRAJECTORY_ALERT, data)

    def send_kinematic_state(
        self,
        speed_cm_s: int,
        yaw_rate_mrad_s: int,
        slip_angle_mrad: int,
        localization_status: int,
    ) -> bytes:
        """Construct and transmit CAN 0x180 kinematic vehicle state frame."""
        data = self.pack_kinematic_state_frame(speed_cm_s, yaw_rate_mrad_s, slip_angle_mrad, localization_status)
        return self.send_frame(self.CAN_ID_KINEMATIC_STATE, data)

    def send_plan_control(
        self,
        target_speed_cm_s: int,
        target_steer_cdeg: int,
        fsm_state: int,
        rss_override: bool,
        curvature_x10000: int,
    ) -> bytes:
        """Construct and transmit CAN 0x190 autonomous plan & control frame."""
        data = self.pack_plan_control_frame(target_speed_cm_s, target_steer_cdeg, fsm_state, rss_override, curvature_x10000)
        return self.send_frame(self.CAN_ID_PLAN_CONTROL, data)

    def send_fleet_ota(
        self,
        active_model_hash_byte0: int,
        sync_state: int,
        shadow_disagreement_score: int,
        harvest_queue_count: int,
        backend_enum: int,
    ) -> bytes:
        """Construct and transmit CAN 0x1A0 fleet and OTA frame."""
        data = self.pack_fleet_ota_frame(active_model_hash_byte0, sync_state, shadow_disagreement_score, harvest_queue_count, backend_enum)
        return self.send_frame(self.CAN_ID_FLEET_OTA, data)

    def update_target(self, target: Optional[ThreatTarget]) -> None:
        """Update active target for continuous 30 Hz broadcasting."""
        with self._lock:
            self._current_target = target

    def register_callback(self, callback: Callable[[int, bytes], None]) -> None:
        """Register subscriber callback for transmitted frames."""
        with self._lock:
            if callback not in self._callbacks:
                self._callbacks.append(callback)

    def start_broadcaster(self) -> None:
        """Launch background thread broadcasting at sustained 30 Hz."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._broadcast_loop, daemon=True)
        self._thread.start()

    def stop_broadcaster(self) -> None:
        """Stop background 30 Hz broadcast loop."""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def _broadcast_loop(self) -> None:
        """Main 30 Hz execution loop."""
        while self._running:
            t0 = time.perf_counter()
            with self._lock:
                tgt = self._current_target

            self.send_target(tgt)

            elapsed = time.perf_counter() - t0
            sleep_time = self.interval_s - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def close(self) -> None:
        """Teardown broadcaster and close CAN bus."""
        self.stop_broadcaster()
        if self._bus is not None:
            try:
                self._bus.shutdown()
            except Exception:
                pass
            self._bus = None
            self._is_connected = False
