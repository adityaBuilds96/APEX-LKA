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
