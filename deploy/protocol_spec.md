# APEX-LKA: Hardware Communication Protocol Specification
### Jetson Orin Nano ↔ 2× Arduino Mega 2560 (CAN Bus & USB Serial Fallback)
**Document Version:** 2.0  
**Vehicle Platform:** SAE India BAJA 2027 Autonomous Category  
**Architecture:** Distributed Dual-Node Real-Time Microcontroller Network with High-Level Linux Host

---

## 1. Network Overview & Topology

The vehicle control architecture distributes time-critical physical actuation across two **Arduino Mega 2560** microcontrollers linked to the primary **NVIDIA Jetson Orin Nano** via high-speed **CAN Bus (500 kbps)** with a redundant **USB Serial (115200 / 250000 baud)** fallback layer.

```
 +-------------------------------------------------------------------------+
 |                       NVIDIA Jetson Orin Nano                           |
 |         (Autonomous Perception, Lane Geometry, Steering Planner)        |
 +-------------------------------------------------------------------------+
        | (can0 - SocketCAN)                            | (USB Serial /dev/ttyACM*)
        | 500 kbps                                      | 115200 baud
        v                                               v
+-------------------------------+               +-------------------------------+
|         CAN Bus Line          |               |      USB Serial Hub / Bus     |
+-------------------------------+               +-------------------------------+
        |                       |                       |                       |
        v                       v                       v                       v
+---------------+       +---------------+       +---------------+       +---------------+
|   MCP2515     |       |   MCP2515     |       | Arduino Mega  |       | Arduino Mega  |
|  Transceiver  |       |  Transceiver  |       |    Node 1     |       |    Node 2     |
+---------------+       +---------------+       +---------------+       +---------------+
        |                       |                   (Steering)             (Throttle/Brake)
        v                       v
+---------------+       +---------------+
| Arduino Mega  |       | Arduino Mega  |
|    Node 1     |       |    Node 2     |
| (Primary Steer|       | (Secondary Act|
|  & Actuation) |       |  & Sensors)   |
+---------------+       +---------------+
```

---

## 2. Serial Communication Protocol (UART / USB)

When CAN Bus is offline or during diagnostic bench testing, high-speed bidirectional Serial streaming provides resilient, line-delimited ASCII JSON data interchange.

- **Port Mapping:**
  - Node 1: Auto-discovered or `/dev/ttyACM0` (Windows: `COM3`)
  - Node 2: Auto-discovered or `/dev/ttyACM1` (Windows: `COM4`)
- **Baudrate:** `115200` (default) or `250000` (low-latency high-throughput mode)
- **Framing:** JSON string terminated with LF (`\n`, `0x0A`)
- **Transmission Rate:** 30 Hz nominal

### 2.1 Jetson → Arduino Command Packet
```json
{
  "node": 1,
  "cmd_a": 12.50,
  "cmd_b": 0.00,
  "offset_m": -0.15,
  "confidence": 0.95,
  "mode": "AUTO",
  "ts": 418290
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `node` | Integer | Destination node identifier (`1` = Steering/Actuation, `2` = Throttle/Brake/Aux) |
| `cmd_a` | Float | Primary control signal (Node 1: Target Steering Rack Angle $-30.0^\circ$ to $+30.0^\circ$; Node 2: Throttle $0.0\%$ to $100.0\%$) |
| `cmd_b` | Float | Secondary control signal (Node 1: Actuator damping/trim; Node 2: Brake pressure $0.0\%$ to $100.0\%$) |
| `offset_m` | Float | Physical lateral offset from lane centerline in meters ($-3.00\text{ m}$ to $+3.00\text{ m}$) |
| `confidence` | Float | Perception model confidence score ($0.00$ to $1.00$) |
| `mode` | String | Vehicle operating mode: `"MANUAL"`, `"AUTO"`, or `"EMERGENCY_STOP"` |
| `ts` | Integer | Monotonic millisecond timestamp from Jetson boot |

### 2.2 Arduino → Jetson Acknowledgment / Feedback Packet
```json
{
  "node": 1,
  "status": "OK",
  "ack_ts": 418290,
  "fb_a": 12.45,
  "fb_b": 0.00,
  "override": 0,
  "fault": 0
}
```

| Field | Type | Description |
| :--- | :--- | :--- |
| `node` | Integer | Originating node identifier (`1` or `2`) |
| `status` | String | Health status: `"OK"`, `"WARN"`, or `"FAULT"` |
| `ack_ts` | Integer | Mirrored timestamp of last processed command |
| `fb_a` | Float | Measured physical feedback A (Node 1: Actual steering angle from encoder; Node 2: Wheel speed in m/s) |
| `fb_b` | Float | Measured physical feedback B (Node 1: Actuator motor current; Node 2: Brake line pressure) |
| `override` | Integer | Driver physical takeover detected: `0` = Autonomous active, `1` = Manual driver override |
| `fault` | Integer | Bitmask of active node hardware faults |

---

## 3. CAN Bus Protocol Specification

- **Physical Layer:** ISO 11898-2 High-Speed CAN with $120\ \Omega$ termination at both ends.
- **Bitrate:** `500 kbps`
- **Identifier Format:** 11-bit Standard Base Frame (CAN 2.0A)
- **Endianness:** Little-Endian (`<`) for all multi-byte fields.

### 3.1 CAN Message Matrix

| CAN ID (Hex) | Transmit Node | Receive Node | Name | Rate (Hz) | DLC |
| :---: | :---: | :---: | :--- | :---: | :---: |
| `0x050` | Jetson | All Nodes | Master Heartbeat | 10 | 4 |
| `0x100` | Jetson | Arduino Node 1 | Node 1 Control Command | 30 | 8 |
| `0x150` | Jetson | All Nodes | Obstacle Threat Alert | 30 | 8 |
| `0x160` | Jetson | All Nodes | 3D Obstacle Spatial Alert | 30 | 8 |
| `0x170` | Jetson | All Nodes | Trajectory Collision Alert | 30 | 8 |
| `0x180` | Jetson | All Nodes | Kinematic Vehicle State Broadcast | 100 | 8 |
| `0x190` | Jetson | All Nodes | Autonomous Plan & Control State | 30 | 8 |
| `0x1A0` | Jetson | All Nodes | Fleet Remote Status & OTA Broadcast | 30 | 8 |
| `0x200` | Jetson | Arduino Node 2 | Node 2 Control Command | 30 | 8 |
| `0x300` | Arduino Node 1 | Jetson | Node 1 Sensor Feedback | 30 | 8 |
| `0x301` | Arduino Node 2 | Jetson | Node 2 Sensor Feedback | 30 | 8 |

---

### 3.2 Message Frame Definitions

#### CAN ID `0x050` — Master Heartbeat Broadcast
Broadcast by the Jetson every 100 ms to confirm autonomous stack health.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0 | `master_alive` | `uint8` | 1 | 0xAA (fixed) | Master synchronization token |
| 1 | `health_code` | `uint8` | 1 | 0 to 255 | 0=NOMINAL, 1=DEGRADED, 2=CAMERA_LOSS, 3=ESTOP |
| 2-3 | `cycle_counter` | `uint16` | 1 | 0 to 65535 | Incremental sequence heartbeat |

```python
# Packing format (4 bytes):
data = struct.pack("<BBH", 0xAA, health_code, cycle_counter)
```

---

#### CAN ID `0x150` — Obstacle Threat Alert Broadcast
Broadcast by Jetson at 30 Hz containing telemetry for the highest-threat detected entity.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `distance_mm` | `uint16` | 1 mm / LSB | 0 to 65535 | Distance to highest-threat obstacle in millimeters |
| 2 | `class_enum` | `uint8` | 1 | 0 to 8 | 0=Vehicle, 1=Pedestrian, 2=Cyclist, 3=Cone, 4=Barrier, 5=Animal, 6=Debris, 7=Sign, 8=Light |
| 3 | `risk_score` | `uint8` | 1 / LSB | 0 to 100 | Threat Risk Score: clamp(100*(0.5*S_prox + 0.3*InCorridor + 0.2*Severity)) |
| 4 | `estop_flag` | `uint8` | 1 | 0 or 1 | 0 = Clear, 1 = BRAKE_IMMINENT (Emergency Stop Triggered) |
| 5-7 | `seq_counter` | `uint24` | 1 / LSB | 0 to $2^{24}-1$ | Sequence counter and frame verification tag |

```python
# Packing format (8 bytes):
data = struct.pack("<HBBBBH", dist_mm, class_enum, risk_score, 1 if emergency_stop else 0, seq & 0xFF, (seq >> 8) & 0xFFFF)
```

---

#### CAN ID `0x160` — 3D Obstacle Spatial Alert Broadcast
Broadcast by Jetson at 30 Hz containing 3D metric spatial coordinates and Time-to-Impact (TTI) for the nearest detected obstacle.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `x_mm` | `int16` | 1 mm / LSB | -32768 to 32767 | Lateral offset relative to vehicle centerline (- = left, + = right) |
| 2-3 | `z_mm` | `uint16` | 1 mm / LSB | 0 to 65535 | Longitudinal distance along forward optical axis |
| 4-5 | `tti_ms` | `uint16` | 1 ms / LSB | 0 to 65535 | Collision Time-to-Impact in milliseconds (65535 = no impact / stopped) |
| 6 | `class_enum` | `uint8` | 1 | 0 to 8 | Detected obstacle class enum |
| 7 | `threat_level` | `uint8` | 1 | 0 to 3 | 0=CLEAR, 1=CAUTION, 2=WARNING, 3=CRITICAL_STOP |

```python
# Packing format (8 bytes):
data = struct.pack("<hHHBB", x_mm, z_mm, tti_ms, class_enum, threat_level)
```

---

#### CAN ID `0x170` — Trajectory Collision Alert Broadcast
Broadcast by Jetson at 30 Hz containing critical object trajectory, projected cross-path lateral intersection, Time-to-Impact (TTI), risk level, and emergency evasive action flag.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `track_id` | `uint16` | 1 | 0 to 65535 | Persistent Track ID of highest collision risk object |
| 2-3 | `min_tti_ms` | `uint16` | 1 ms / LSB | 0 to 65535 | Minimum Time-To-Impact in milliseconds (65535 = no collision) |
| 4-5 | `cross_x_mm` | `int16` | 1 mm / LSB | -32768 to 32767 | Projected path intersection lateral offset relative to vehicle center (mm) |
| 6 | `risk_level` | `uint8` | 1 | 0 to 3 | 0=CLEAR, 1=MONITOR, 2=WARNING, 3=IMMINENT_COLLISION |
| 7 | `action_flag` | `uint8` | 1 | 0 to 3 | 0=NONE, 1=BRAKE_NOW, 2=SWERVE_LEFT, 3=SWERVE_RIGHT |

```python
# Packing format (8 bytes):
data = struct.pack("<HHhBB", track_id, min_tti_ms, cross_x_mm, risk_level, action_flag)
```

---

#### CAN ID `0x180` — Kinematic Vehicle State Broadcast
Broadcast by Jetson at 100 Hz containing high-frequency estimated vehicle state from the 15-DOF Error-State Kalman Filter (ESKF).

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `vx_cm` | `int16` | 1 cm/s / LSB | -32768 to 32767 | Longitudinal velocity $v_x \times 100$ (cm/s) |
| 2-3 | `vy_cm` | `int16` | 1 cm/s / LSB | -32768 to 32767 | Lateral velocity $v_y \times 100$ (cm/s) |
| 4-5 | `yaw_rate_mrad` | `int16` | 1 mrad/s / LSB | -32768 to 32767 | Body yaw rate $\omega_z \times 1000$ (mrad/s) |
| 6 | `slip_angle_decideg` | `int8` | 0.1° / LSB | -128 to 127 | Dynamic vehicle side-slip angle $\beta \times 10$ (deci-degrees) |
| 7 | `localization_status`| `uint8` | 1 | 0 to 4 | 0=INIT, 1=DEAD_RECKONING, 2=VIO_ACTIVE, 3=GNSS_FIX, 4=RTK_LOCKED |

```python
# Packing format (8 bytes):
data = struct.pack("<hhhbB", int(vx * 100), int(vy * 100), int(wz * 1000), int(beta * 10), loc_status)
```

---

#### CAN ID `0x190` — Autonomous Plan & Control State Broadcast
Broadcast by Jetson at 30 Hz containing target speed, planned road-wheel steering angle, behavior FSM state, Mobileye RSS safety shield override status, and path curvature.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `v_target_cm` | `uint16` | 1 cm/s / LSB | 0 to 65535 | Target Vehicle Speed $v_{\text{target}} \times 100$ (cm/s) |
| 2-3 | `delta_target_cdeg` | `int16` | $0.01^\circ$ / LSB | -32768 to 32767 | Target Steering Angle $\delta_{\text{target}} \times 100$ (centi-degrees) |
| 4 | `behavior_state` | `uint8` | 1 | 0 to 5 | FSM Mode: 0=LANE_FOLLOWING, 1=ADAPTIVE_CRUISE, 2=YIELD, 3=OVERTAKE, 4=ESTOP, 5=STOP |
| 5 | `rss_override` | `uint8` | 1 | 0 or 1 | Mobileye RSS Safety Shield: 0=Normal, 1=RSS_OVERRIDE_BRAKE |
| 6-7 | `curvature_scaled` | `int16` | $10^{-4}\text{ m}^{-1}$ / LSB | -32768 to 32767 | Planned Path Curvature $\kappa \times 10000$ |

```python
# Packing format (8 bytes):
data = struct.pack("<HhBBh", int(v_target * 100), int(delta_target * 100), fsm_state, 1 if rss_override else 0, int(kappa * 10000))
```

---

#### CAN ID `0x1A0` — Fleet Remote Status & OTA Broadcast
Broadcast by Jetson at 30 Hz containing active model version hash, OTA sync state, shadow mode discrepancy score, harvested edge package queue count, and active inference backend.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `model_hash_u16` | `uint16` | 1 | 0 to 65535 | Active Model Version SHA-256 Lower 16 bits |
| 2 | `ota_sync_state` | `uint8` | 1 | 0 to 6 | 0=IDLE, 1=CHECKING, 2=DOWNLOADING, 3=VERIFYING, 4=HOTSWAPPING, 5=SYNC_OK, 6=ERROR |
| 3 | `shadow_discrepancy` | `uint8` | 1 | 0 to 100 | Silent Shadow Model Discrepancy Score ($0=$ Complete Match, $100=$ Total Divergence) |
| 4 | `harvest_queue_count`| `uint8` | 1 | 0 to 255 | Number of packaged failure incident snapshots pending fleet ingestion |
| 5 | `active_backend` | `uint8` | 1 | 0 to 3 | Active Inference Backend: 0=CLASSICAL, 1=PYTORCH, 2=ONNX, 3=TENSORRT |
| 6-7 | `reserved` | `uint16` | 1 | 0 | Reserved for future expansion |

```python
# Packing format (8 bytes):
data = struct.pack("<HBBBBH", model_hash_u16, ota_sync_state, shadow_disc_score, harvest_queue_count, active_backend, 0)
```

---

#### CAN ID `0x100` — Control Command Node 1 (Steering & Actuation)
Sent by Jetson to Node 1 at 30 Hz.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `cmd_a_scaled` | `int16` | $0.01^\circ$ / LSB | -3000 to +3000 | Target Steering Rack Angle ($-30.00^\circ$ to $+30.00^\circ$) |
| 2-3 | `cmd_b_scaled` | `int16` | $0.01$ / LSB | -10000 to +10000 | Actuator rate limit / Damping parameter |
| 4 | `confidence` | `uint8` | 1 / 255 | 0 to 255 | Perception Model Confidence ($0.00$ to $1.00$) |
| 5 | `mode_enum` | `uint8` | 1 | 0 to 2 | 0 = MANUAL, 1 = AUTO, 2 = EMERGENCY_STOP |
| 6-7 | `seq_timestamp` | `uint16` | 1 ms / LSB | 0 to 65535 | Timestamp modulo 65536 ms |

```python
# Packing format (8 bytes):
data = struct.pack("<hhBBH", int(steer_deg * 100), int(cmd_b * 100), int(conf * 255), mode_enum, ts_ms % 65536)
```

---

#### CAN ID `0x200` — Control Command Node 2 (Throttle, Brake, Auxiliary)
Sent by Jetson to Node 2 at 30 Hz.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `cmd_a_scaled` | `int16` | $0.01\%$ / LSB | 0 to 10000 | Throttle Command ($0.00\%$ to $100.00\%$) |
| 2-3 | `cmd_b_scaled` | `int16` | $0.01\%$ / LSB | 0 to 10000 | Brake Command ($0.00\%$ to $100.00\%$) |
| 4 | `mode_enum` | `uint8` | 1 | 0 to 2 | 0 = MANUAL, 1 = AUTO, 2 = EMERGENCY_STOP |
| 5 | `aux_flags` | `uint8` | Bitfield | 0 to 255 | Bit 0: Headlights, Bit 1: Horn, Bit 2: Gear (0=Fwd, 1=Rev) |
| 6-7 | `seq_timestamp` | `uint16` | 1 ms / LSB | 0 to 65535 | Timestamp modulo 65536 ms |

```python
# Packing format (8 bytes):
data = struct.pack("<hhBBH", int(throttle_pct * 100), int(brake_pct * 100), mode_enum, aux_flags, ts_ms % 65536)
```

---

#### CAN ID `0x300` — Feedback Node 1 (Steering Angle & Motor Telemetry)
Transmitted by Node 1 to Jetson at 30 Hz.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `fb_steering` | `int16` | $0.01^\circ$ / LSB | -3000 to +3000 | Measured Absolute Steering Angle from Wheel Encoder |
| 2-3 | `fb_current` | `int16` | $10\text{ mA}$ / LSB | -32768 to +32767 | Steering Motor Driver Current Draw |
| 4 | `override_state`| `uint8` | 1 | 0 or 1 | 0 = Autonomous Active, 1 = Driver Physical Override |
| 5 | `fault_flags` | `uint8` | Bitfield | 0 to 255 | Bit 0: Sensor Err, Bit 1: Overcurrent, Bit 2: Watchdog Timeout |
| 6-7 | `node_uptime_ms`| `uint16` | 1 ms / LSB | 0 to 65535 | Microcontroller millis() % 65536 |

```python
# Unpacking format (8 bytes):
fb_steer_raw, fb_curr_raw, override, fault, uptime_ms = struct.unpack("<hhBBH", data)
fb_steering_deg = fb_steer_raw / 100.0
```

---

#### CAN ID `0x301` — Feedback Node 2 (Wheel Speed & Brake Pressure)
Transmitted by Node 2 to Jetson at 30 Hz.

| Byte | Field | Type | Scaling | Range | Description |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 0-1 | `fb_speed` | `int16` | $0.01\text{ m/s}$ / LSB | 0 to 3000 | Ground Speed from Optical/Hall Wheel Encoders |
| 2-3 | `fb_brake_psi` | `int16` | $0.1\text{ psi}$ / LSB | 0 to 5000 | Hydraulic Line Brake Pressure |
| 4 | `override_state`| `uint8` | 1 | 0 or 1 | 0 = Nominal, 1 = Driver Brake Pedal Depressed |
| 5 | `fault_flags` | `uint8` | Bitfield | 0 to 255 | Bit 0: Encoder Err, Bit 1: Pressure Loss, Bit 2: Watchdog Timeout |
| 6-7 | `node_uptime_ms`| `uint16` | 1 ms / LSB | 0 to 65535 | Microcontroller millis() % 65536 |

```python
# Unpacking format (8 bytes):
fb_speed_raw, fb_brake_raw, override, fault, uptime_ms = struct.unpack("<hhBBH", data)
speed_mps = fb_speed_raw / 100.0
```

---

## 4. Hardware Safety Watchdog & Failsafe Transitions

### 4.1 500ms Loss-of-Communication Watchdog
Both Arduino microcontrollers run hardware timer interrupts verifying reception of valid commands:
$$\Delta t = t_{\text{now}} - t_{\text{last\_valid\_cmd}}$$
- If $\Delta t \le 500\text{ ms}$: Nominal control execution.
- If $\Delta t > 500\text{ ms}$:
  - Assert `WATCHDOG_TIMEOUT` fault flag.
  - Ramp target steering rack angle smoothly to $0.0^\circ$ (Neutral Center).
  - Cut throttle to $0\%$ and apply proportional brake pressure to stop vehicle.
  - Emit serial diagnostic warning packet.

### 4.2 Driver Physical Override Interrupt
If the driver manually turns the steering wheel or depresses the brake pedal while the system is in `AUTO` mode:
- Hardware limit switch or current sensor on Pin 2/18 fires an interrupt.
- Arduino immediately decouples autonomous drive commands.
- Feedback packet sets `override_state = 1`.
- Jetson state machine transitions from `AUTO` to `MANUAL` until explicitly re-engaged.
