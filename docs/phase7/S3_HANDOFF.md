# APEX-LKA Phase 7 Section 3 — Hardware Framing Integrity & Auto-Discovery HAL
## Technical Architecture Handoff Report · S3_HANDOFF.md

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [CAN Bus Network Architecture](#2-can-bus-network-architecture)
3. [DBC Message Layout Reference](#3-dbc-message-layout-reference)
4. [Framing Integrity Specification](#4-framing-integrity-specification)
5. [Hardware Driver Architecture](#5-hardware-driver-architecture)
6. [Dual Arduino Bridge Protocol](#6-dual-arduino-bridge-protocol)
7. [Hardware Failure Behavior](#7-hardware-failure-behavior)
8. [Recovery State Machine](#8-recovery-state-machine)
9. [Auto-Discovery HAL Protocol](#9-auto-discovery-hal-protocol)
10. [Thermal Safety & Power Budget](#10-thermal-safety--power-budget)
11. [Test Coverage & Verification](#11-test-coverage--verification)
12. [Known Limitations & Open Items](#12-known-limitations--open-items)

---

## 1. Executive Summary

Phase 7 Section 3 delivers the complete **CAN Hardware Framing Integrity** and **Hardware Auto-Discovery HAL** subsystem for APEX-LKA v4.0. This section replaces the ad-hoc serial communication layer from Phase 6 with a production-hardened, ISO 26262 ASIL-B aware CAN bus stack.

### Key Deliverables

| Component | File | LOC | Status |
|-----------|------|-----|--------|
| DBC Database | `configs/can/apex_v4.dbc` | 280+ | ✅ |
| DBC Generator/Validator | `scripts/gen_dbc.py` | 350+ | ✅ |
| CAN Integrity Engine | `src/hardware/can_integrity.py` | 850+ | ✅ |
| Starkenn Radar Driver | `src/hardware/drivers/starkenn_can_driver.py` | 650+ | ✅ |
| MTi-680 INS Driver | `src/hardware/drivers/mti680_can_driver.py` | 600+ | ✅ |
| Arduino Firmware | `deploy/arduino_can_bridge/arduino_can_bridge.ino` | 450+ | ✅ |
| Arduino Bridge Host | `src/hardware/arduino_bridge.py` | 750+ | ✅ |
| Auto-Discovery Engine | `src/hardware/auto_discovery.py` | 950+ | ✅ |
| Hardware Config | `configs/hardware/auto_discovery.yaml` | 160+ | ✅ |

### Architecture Highlights

- **500 kbps CAN bus** with 5 network nodes and 19 message definitions
- **CRC-8 SAE J1850** (polynomial 0x1D) lookup-table driven — O(1) per byte
- **4-bit rolling sequence counter** with anomaly classification (VALID / OUT_OF_ORDER / FRAME_DROP / RESET)
- **Automatic serial fallback** from SocketCAN to Arduino bridge if corruption > 5%
- **500ms scan cycle** hardware auto-discovery with hot-plug support
- **Thermal safety monitoring** with 80°C CPU/GPU limit

---

## 2. CAN Bus Network Architecture

### 2.1 Physical Layer

```
CAN Bus Topology (500 kbps, ISO 11898-2)
═══════════════════════════════════════════════════════════════
  ┌──────────────────┐
  │   JETSON ORIN NX │  Master Node (ID=0x01)
  │   APEX-LKA v4.0  │  Transmits: 0x050, 0x100, 0x150,
  │   192.168.2.1    │             0x160, 0x170, 0x180, 0x190
  └────────┬─────────┘
           │ CAN H/L (120Ω termination)
    ───────┼────────────────────────────────────────────────
           │              │              │              │
  ┌────────┴───┐  ┌───────┴────┐  ┌─────┴──────┐  ┌──┴──────────┐
  │ ARDUINO    │  │ ARDUINO    │  │ RADAR      │  │ INS MTi-680 │
  │ PRIMARY    │  │ SECONDARY  │  │ STARKENN   │  │ GNSS/INS    │
  │ (ID=0x02)  │  │ (ID=0x03)  │  │ (ID=0x04)  │  │ (ID=0x05)   │
  │ Steering   │  │ Redundant  │  │ 77 GHz     │  │ RTK GPS     │
  │ Relay      │  │ Safety     │  │ Radar      │  │ 100 Hz      │
  └────────────┘  └────────────┘  └────────────┘  └─────────────┘
```

### 2.2 Node Address Map

| Node Name | CAN Node ID | Station Address | Primary Role |
|-----------|-------------|-----------------|--------------|
| `JETSON_ORIN` | 0x01 | 1 | Master compute node, command issuer |
| `ARDUINO_PRIMARY` | 0x02 | 2 | EPS steering actuation, safety relay |
| `ARDUINO_SECONDARY` | 0x03 | 3 | Redundant safety interlock |
| `RADAR_STARKENN` | 0x04 | 4 | Forward obstacle detection, 8 tracks |
| `INS_MTI680` | 0x05 | 5 | GNSS/INS navigation, 100 Hz Euler |

### 2.3 Bus Load Budget

| Message Group | Hz | Frames/s | Bits/frame | Load% |
|--------------|-----|----------|------------|-------|
| Heartbeat (0x050) | 100 | 100 | 111 | 2.22% |
| Steering (0x100) | 30 | 30 | 111 | 0.67% |
| Risk/Spatial/Traj/Plan (4×30Hz) | 30×4 | 120 | 111 | 2.66% |
| Kinematics (0x180) | 100 | 100 | 111 | 2.22% |
| MTi-680 (2×100Hz) | 100×2 | 200 | 111 | 4.44% |
| Radar Header (0x400) | 20 | 20 | 111 | 0.44% |
| Radar Tracks (8×20Hz) | 20×8 | 160 | 111 | 3.56% |
| **Total** | | **730** | | **≈16.2%** |

> **Note:** 16.2% nominal bus load is well within the 70% practical limit for reliable ISO 11898 operation.

---

## 3. DBC Message Layout Reference

### 3.1 Frame Structure (All Messages)

All APEX-LKA v4.0 CAN frames use **8-byte DLC** with the following canonical layout:

```
Byte  0123456 7
      [DATA  ][CRC8][CNTR]
                    └─ 4-bit Counter (bits 56-59)
               └─── CRC-8 SAE J1850 over bytes 0-5
      └─────── Signal payload (signals packed little-endian)
```

### 3.2 Complete Message Table

| CAN ID | Name | Hz | DLC | Transmitter | Key Signals |
|--------|------|----|-----|-------------|-------------|
| 0x050 | Heartbeat | 100 | 8 | JETSON_ORIN | NodeID, SystemState[0-5], ErrorFlags[16-bit] |
| 0x100 | Steering_Primary | 30 | 8 | JETSON_ORIN | TargetAngle[±28°@0.001°], AngVelLim, Torque[±10Nm] |
| 0x150 | Priority_Risk | 30 | 8 | JETSON_ORIN | RiskLevel[0-255], TTC[0-10s@0.001], EmergBrake |
| 0x160 | Spatial_State | 30 | 8 | JETSON_ORIN | HeadingErr[±180°], LateralErr[±16m], Curvature |
| 0x170 | Trajectory_Poly | 30 | 8 | JETSON_ORIN | PolyA/B/C[±0.5], PolyD[±5m] |
| 0x180 | Vehicle_Kinematics | 100 | 8 | JETSON_ORIN | WheelSpeed[0-250km/h], YawRate, Accel×2 |
| 0x190 | Plan_Control_Status | 30 | 8 | JETSON_ORIN | ControllerMode[0-4], Health[20-bit], LatencyUs |
| 0x320 | MTi680_Orientation | 100 | 8 | INS_MTI680 | Roll[±180°], Pitch[±90°], Yaw[0-360°@0.01°] |
| 0x321 | MTi680_Velocity | 100 | 8 | INS_MTI680 | VelEast/North[±328m/s], VelDown, GnssFixType |
| 0x400 | Starkenn_Radar_Header | 20 | 8 | RADAR_STARKENN | TargetCount[0-8], BusLoad%, StatusFlags |
| 0x401-0x408 | Starkenn_Radar_Track1-8 | 20 | 8 | RADAR_STARKENN | RangeX/Y[0.05m], RelVelX/Y[0.01m/s], RCS, Conf |

### 3.3 Signal Encoding Details

#### Steering Angle (0x100, bits 0-15)
```
Physical = Raw × 0.001 + (-28.0)
Range: -28.000° to +28.000° (56° total sweep)
LSB: 1 millidegree
Raw 0     → -28.000° (full left)
Raw 28000 → 0.000° (center)
Raw 56000 → +27.999° (full right)
```

#### TTC — Time to Collision (0x150, bits 8-23)
```
Physical = Raw × 0.001 + 0.0
Range: 0.000 s to 10.000 s
LSB: 1 millisecond
Raw 0     → 0.000 s (contact)
Raw 10000 → 10.000 s (safe)
Raw 65535 → INVALID (no target detected)
```

#### Radar Range X (0x401-0x408, bits 4-15)
```
Physical = Raw × 0.05 + (-50.0)
Range: -50.00 m to +154.75 m
LSB: 5 cm resolution
Negative = behind vehicle (reverse)
```

---

## 4. Framing Integrity Specification

### 4.1 CRC-8 Algorithm (SAE J1850)

**Polynomial:** `0x1D` (reflected: `0xB8`)  
**Initial value:** `0xFF`  
**Final XOR:** `0xFF`  
**Byte coverage:** Bytes 0–5 of each 8-byte frame  
**Result location:** Byte 6 of frame  

```python
# Reference implementation (Python)
def compute_crc8(data: bytes) -> int:
    crc = 0xFF
    for byte in data[:6]:
        crc = CRC8_TABLE[crc ^ byte]
    return crc ^ 0xFF
```

**Reference test vectors:**

| Input (hex) | Expected CRC-8 |
|-------------|----------------|
| `00 00 00 00 00 00` | `0xFF` |
| `01 02 03 04 05 06` | computed per table |
| `FF FF FF FF FF FF` | computed per table |

> The lookup table is precomputed at startup (256 entries) providing O(1) per-byte computation.

### 4.2 Sequence Counter Protocol

Each CAN message contains a 4-bit rolling counter in bits 56–59 (byte 7, lower nibble), incrementing 0→1→...→15→0.

**Classification Rules:**

| Condition | Classification | Action |
|-----------|---------------|--------|
| `C_new == (C_old + 1) % 16` | `VALID` | Normal operation |
| `C_new == C_old` | `OUT_OF_ORDER` | Log, increment OOO counter |
| `1 < gap < 8` | `FRAME_DROP` | Log warning, increment missed count |
| `C_new == 0 AND C_old ∉ {14,15}` | `RESET` | Log info, accept new sequence |
| Large backward jump | `OUT_OF_ORDER` | Log, do not panic |

**Frame Drop Rate Calculation:**
```
drop_rate = dropped_frames / total_frames  (rolling)
alert threshold: drop_rate > 0.01 (1%)
```

### 4.3 Message Staleness Detection

Each received message timestamp is recorded. The `LineStatusTracker` checks message age every cycle:

```
age = t_now - t_last_rx
stale if age > 100 ms (configurable)
```

On staleness:
- Raise `STALE` flag for affected message ID
- Fire safety callback to set associated subsystem to DEGRADED
- Log `WARNING` with CAN ID and age

**Safety-critical staleness timeouts:**

| Message | Stale Timeout | Action |
|---------|--------------|--------|
| 0x050 Heartbeat | 15 ms | DEGRADED (Arduino opens relay after 200 ms) |
| 0x100 Steering | 50 ms | Arduino relay opens (200 ms firmware timeout) |
| 0x320/0x321 MTi-680 | 25 ms | Switch INS to dead reckoning |
| 0x400 Radar Header | 100 ms | Radar data marked invalid, perception adapts |

---

## 5. Hardware Driver Architecture

### 5.1 Starkenn Radar Driver

```
StarkennCANDriver
├── ZeroCopyRingBuffer (256 slots)
│   └── push(can_id, payload, ts) → O(1), non-blocking
├── CANIntegrityEngine callbacks
│   ├── _on_header_frame(0x400) → RadarScanFrame assembly
│   └── _on_track_frame(0x401-0x408) → RadarTrack decode + filter
├── Track Filters
│   ├── RCS filter: rcs_flag must be SET (>-15 dBm)
│   ├── Confidence filter: conf_flag must be SET (>60%)
│   └── Range filter: 0.5 m ≤ range_x ≤ 150 m
└── Output: RadarScanFrame.valid_tracks → TrackRadarFusion
```

**RadarTrack fields for fusion engine:**

| Field | Type | Notes |
|-------|------|-------|
| `range_x_m` | float | Longitudinal range, 0.05m LSB |
| `range_y_m` | float | Lateral range, 0.05m LSB |
| `rel_vel_x_mps` | float | Closing speed, 0.01m/s LSB |
| `closing_speed_mps` | property | = -rel_vel_x_mps |
| `range_m` | property | Euclidean range |
| `azimuth_deg` | property | atan2(y,x) |
| `is_valid_for_fusion` | property | All filters passed |

### 5.2 MTi-680 Driver

```
MTi680CANDriver
├── CANIntegrityEngine callbacks
│   ├── _on_orient_frame(0x320) → roll/pitch/yaw + INS status
│   └── _on_velocity_frame(0x321) → vel_E/N/D + GNSS fix type
├── Dead Reckoning State Machine
│   ├── ACTIVE when InsStatus == DEAD_RECK
│   ├── Propagates heading using yaw rate
│   └── Inflates variance: scale = 1 + duration_s × 2.0
├── HorizonCalibration (for DepthEngineV2)
│   ├── EMA pitch filter (α=0.05)
│   ├── Only samples when speed < 2 m/s
│   └── Outputs pitch_bias_deg + variance
└── GNSS Transition Detection
    ├── RTK loss: prev.is_rtk AND NOT new.is_rtk
    └── Recovery: NOT prev.is_rtk AND new.is_rtk
```

**INS Status Variance Scaling:**

| Status | Variance Scale | Meaning |
|--------|---------------|---------|
| INIT (0) | 1,000,000× | Not usable |
| COARSE_ALIGN (1) | 1,000× | Very rough |
| FINE_ALIGN (2) | 10× | Acceptable |
| RTK_FLOAT (3) | 2× | Sub-meter |
| RTK_FIXED (4) | 1× | 2 cm accuracy |
| DEAD_RECK (5) | 10× (grows) | IMU-only drift |

---

## 6. Dual Arduino Bridge Protocol

### 6.1 COBS Serial Frame Format

```
Serial Packet (host ↔ Arduino):
  [0x00][COBS_BLOCK][0x00]

COBS_BLOCK decodes to:
  [CAN_ID_H][CAN_ID_L][DLC][D0][D1][D2][D3][D4][D5][D6][D7][PKT_CRC8][CTR]
  Byte:       0          1    2   3   4   5   6   7   8   9   10       11   12

CRC-8 coverage: bytes 0-10 (13-byte packet header + data)
Inner frame CRC: D6 = CRC-8 over D0-D5 (as per DBC spec)
```

### 6.2 Arduino Firmware State Machine

```
Arduino Startup:
  WDT enable (500 ms) → MCP2515 init → Serial bridge → Relay OPEN
    │
    ▼ (on successful MCP2515 init)
  Relay evaluation loop (every loop() call):
    ├── Check last_steer_rx_ms
    │   ├── age > 200 ms → OPEN relay (failsafe)
    │   └── age ≤ 200 ms → CLOSE relay (active)
    ├── Forward CAN → Serial (all frames)
    ├── Parse Serial → CAN (verified frames only)
    └── WDT reset every iteration
```

### 6.3 Bridge Mode Priority

```
Priority 1: SocketCAN (can0)         ← Primary mode
Priority 2: Serial /dev/ttyACM0      ← Primary Arduino fallback
Priority 3: Serial /dev/ttyACM1      ← Secondary Arduino fallback
Priority 4: DISCONNECTED             ← Full failsafe

Fallback Trigger Conditions:
  A. can0 interface not available (ifup fails)
  B. Frame corruption rate > 5% over 100-frame window
  C. No frames received for > 500 ms
```

### 6.4 Latency Budget

| Stage | Target | Maximum |
|-------|--------|---------|
| Arduino CAN RX → Serial TX | 0.5 ms | 2 ms |
| Host serial RX → decode | 0.3 ms | 1 ms |
| Host decode → callback | 0.2 ms | 0.5 ms |
| **End-to-end overhead** | **< 1.5 ms** | **< 5 ms** |

---

## 7. Hardware Failure Behavior

### 7.1 Failure Mode Catalog

| Failure | Detection | Response | Recovery |
|---------|-----------|----------|---------|
| CAN bus electrical fault | No frames on can0 | Switch to serial bridge | Re-scan every 2 s |
| Arduino Primary power loss | Relay opens (firmware WDT) | Switch to Arduino Secondary | Reconnect serial port |
| Arduino Secondary power loss | Serial port disappears | Continue on Primary only | Re-scan every 2 s |
| Radar CRC errors > 5% | CRC counter in LineStatusTracker | Mark radar DEGRADED | Frames with valid CRC still used |
| Radar no data > 100 ms | Staleness timer | Set radar_stale flag | Wait for recovery |
| MTi-680 RTK loss | GnssFixType transition | Dead reckoning mode | Auto-recover on RTK fix |
| MTi-680 CAN absence | Staleness timer > 25 ms | Use last known heading | Hard timeout → stop |
| Jetson thermal > 80°C | Thermal zone monitor | Alert HUD, log | Auto-throttle (kernel) |
| Jetson thermal > 95°C | Emergency threshold | SHUTDOWN command | Manual restart required |

### 7.2 Steering Safety Chain

```
Jetson Orin (0x100 @ 30 Hz)
    │
    ▼ CAN bus / serial bridge
Arduino Mega 2560 (firmware)
    │
    ├─ Valid frame received? → Relay CLOSED (steering enabled)
    │
    └─ No frame for 200 ms? → Relay OPEN (neutral safe state)
                                    │
                                    └─ EPS returns to center (passive safety)
```

> The Arduino 500 ms hardware watchdog ensures the relay opens even if the Arduino firmware itself freezes.

---

## 8. Recovery State Machine

### 8.1 Device State Transitions

```
              ┌──────────────────┐
  startup ───►│  UNCONFIGURED    │◄── device_id not in registry
              └────────┬─────────┘
                       │ first_seen
                       ▼
              ┌──────────────────┐
      ────────►│    CONNECTED     │◄── device seen in scan
              └────────┬─────────┘
                       │ missed 5 s worth of scans (10 scans)
                       ▼
              ┌──────────────────┐
              │    DEGRADED      │
              └────────┬─────────┘
                       │ missed 15 s worth (30 scans)
                       ▼
              ┌──────────────────┐
              │  DISCONNECTED    │─── driver reinit callbacks cleared
              └────────┬─────────┘
                       │ device reappears in scan
                       ▼
                  CONNECTED ──► driver reinit callbacks fired
```

### 8.2 Auto-Discovery Reinit Flow

```
Scan detects device (previously DISCONNECTED → now found):
    1. State machine: DISCONNECTED → CONNECTED
    2. Fire HardwareEvent(RECOVERED, device_id)
    3. Push HUD notification (severity=success)
    4. DriverReinitRegistry.trigger(device_record)
       ├── Radar handler → StarkennCANDriver.start()
       ├── INS handler   → MTi680CANDriver re-subscribe
       └── Arduino handler → ArduinoBridge.reconnect()
    5. Log "Driver reinitialized" event
```

### 8.3 CAN Bus Fallback Sequence

```
Step 1: SocketCAN health check fails
    → Attempt ip link set can0 up (3 retries, 500 ms apart)
    → If still down: trigger fallback

Step 2: Fallback to Serial Primary (/dev/ttyACM0)
    → Open serial at 921600 baud
    → Wait for first valid COBS frame (timeout 2 s)
    → If timeout: try 115200 baud fallback
    → If still failing: try Secondary

Step 3: Fallback to Serial Secondary (/dev/ttyACM1)
    → Same connection sequence as Step 2

Step 4: DISCONNECTED
    → Fire HardwareEvent(DISCONNECTED) for can interface
    → Set all safety-critical flags to STALE
    → Exponential backoff reconnect: 2s, 4s, 8s, 16s, 30s max
```

---

## 9. Auto-Discovery HAL Protocol

### 9.1 Scan Cycle Architecture

```
AutoDiscoveryEngine (background thread, 500 ms cycle)
├── FilesystemScanner.scan_cameras()
│   └── glob /dev/video* + /dev/v4l/by-id/* + /sys/class/video4linux/
├── FilesystemScanner.scan_socketcan()
│   └── iterate /sys/class/net/ → filter type==280 (ARPHRD_CAN)
├── FilesystemScanner.scan_serial_devices()
│   └── glob /dev/ttyACM* + /dev/ttyUSB* + /dev/serial/by-id/*
│       └── classify by USB VID:PID → Arduino vs MTi-680 vs unknown
├── FilesystemScanner.scan_thermal()
│   └── glob /sys/class/thermal/thermal_zone*/temp
└── Per-device DeviceStateMachine.on_device_seen() or on_device_missing()
```

### 9.2 Event JSON Schema

```json
{
  "event": "connected|disconnected|degraded|recovered|thermal_warning|driver_reinit",
  "device_id": "can_can0",
  "device_type": "socketcan|arduino_primary|mti680|camera_usb|...",
  "path": "/sys/class/net/can0",
  "timestamp": 1729271182.456,
  "old_state": "disconnected",
  "new_state": "connected",
  "message": "socketcan connected: /sys/class/net/can0",
  "metadata": {
    "bitrate": "500000",
    "operstate": "up"
  }
}
```

### 9.3 HUD Notification Format

```python
{
    "severity": "success|warning|error|info",
    "title": "Hardware: CONNECTED",
    "body": "socketcan at /sys/class/net/can0",
    "device_id": "can_can0",
    "timestamp": 1729271182.456,
    "dismissible": True,
}
```

### 9.4 Device Classification Logic

```
Serial device classification priority:
  1. USB VID:PID match in arduino_vid_pids list → Arduino
  2. USB VID:PID match in mti680_vid_pids list  → MTi-680
  3. Description contains "arduino" (case-insensitive) → Arduino
  4. Description contains "mti" (case-insensitive)  → MTi-680
  5. Unknown → DeviceType.UNKNOWN

Arduino assignment:
  First Arduino found → ARDUINO_PRIMARY
  Second Arduino found → ARDUINO_SECONDARY
  Additional → UNKNOWN (not expected)
```

---

## 10. Thermal Safety & Power Budget

### 10.1 Jetson Orin NX Thermal Zones

| Zone ID | sysfs Path | Type | Limit |
|---------|-----------|------|-------|
| thermal_zone0 | `/sys/class/thermal/thermal_zone0/temp` | CPU | 80°C |
| thermal_zone1 | `/sys/class/thermal/thermal_zone1/temp` | GPU | 80°C |
| thermal_zone2 | `/sys/class/thermal/thermal_zone2/temp` | SoC | 85°C |
| thermal_zone3 | `/sys/class/thermal/thermal_zone3/temp` | CV Core | 80°C |
| thermal_zone4 | `/sys/class/thermal/thermal_zone4/temp` | Board | 60°C |

### 10.2 Thermal Response Policy

```
temp < 70°C   → NORMAL operation
70°C ≤ temp < 80°C → WARNING: log + HUD alert (yellow)
80°C ≤ temp < 95°C → CRITICAL: throttle + error alert (red)
temp ≥ 95°C   → EMERGENCY: initiate safe shutdown sequence
```

### 10.3 Power Budget (Jetson Orin NX 15W)

| Consumer | Nominal Power | Peak Power |
|----------|--------------|-----------|
| CPU (6× Cortex-A78AE) | 4W | 8W |
| GPU (1024 CUDA cores) | 6W | 10W |
| Deep Learning Accelerators | 2W | 4W |
| Memory (LPDDR5) | 1.5W | 2W |
| CAN interface / SPI | 0.1W | 0.2W |
| **Total System** | **≈13.6W** | **≤25W** |

---

## 11. Test Coverage & Verification

### 11.1 test_can_integrity.py Coverage

| Test Case | Description |
|-----------|-------------|
| `test_crc8_table_size` | Verifies 256-entry LUT completeness |
| `test_crc8_known_vectors` | Reference value validation |
| `test_crc8_all_zeros` | Edge case: all-zero payload |
| `test_crc8_all_ones` | Edge case: 0xFF payload |
| `test_sequence_valid` | Normal counter increment |
| `test_sequence_out_of_order` | Repeated counter detection |
| `test_sequence_frame_drop` | Gap > 1 frame detection |
| `test_sequence_reset` | Counter reset to 0 detection |
| `test_counter_4bit_wrap` | 15→0 normal wrap |
| `test_payload_encode_decode` | Round-trip accuracy |
| `test_nan_guard` | NaN/Inf input replacement |
| `test_signal_clamping` | Out-of-range physical clamping |
| `test_stale_threshold` | 100ms age trigger |
| `test_throughput_benchmark` | > 50,000 msg/sec target |

### 11.2 test_auto_discovery.py Coverage

| Test Case | Description |
|-----------|-------------|
| `test_scan_cameras_mock` | Mock /dev/video* filesystem |
| `test_scan_socketcan_mock` | Mock /sys/class/net/can0 |
| `test_scan_serial_mock` | Mock /dev/ttyACM0 with Arduino VID:PID |
| `test_device_connected_event` | New device → CONNECTED event |
| `test_device_degraded_transition` | Missing scans → DEGRADED |
| `test_device_disconnected_transition` | Long absence → DISCONNECTED |
| `test_device_recovery` | Reappearance → RECOVERED + reinit |
| `test_serial_to_socketcan_fallback` | can0 down → serial switch |
| `test_socketcan_to_serial_fallback` | SocketCAN → serial fallback trigger |
| `test_hud_notification_format` | JSON event + HUD alert format |
| `test_background_thread_nonblocking` | Thread starts without blocking |
| `test_thermal_warning_event` | 70°C threshold event |
| `test_thermal_critical_event` | 80°C threshold event |

### 11.3 Verification Commands

```bash
# 1. Validate & regenerate DBC
python scripts/gen_dbc.py --validate --input configs/can/apex_v4.dbc

# 2. Run all hardware integrity tests
pytest tests/test_can_integrity.py tests/test_auto_discovery.py -v --tb=short

# 3. Test auto-discovery scan cycle
python -m src.hardware.auto_discovery --once

# 4. Benchmark parser throughput
pytest tests/test_can_integrity.py::TestCANThroughput -v -s

# 5. Full test suite with coverage
pytest tests/test_can_integrity.py tests/test_auto_discovery.py \
       --cov=src.hardware --cov-report=term-missing
```

---

## 12. Known Limitations & Open Items

### 12.1 Current Limitations

| Item | Description | Mitigation |
|------|-------------|-----------|
| Windows `auto_discovery.py` | `/sys` and `/dev` paths are Linux-specific | Mock filesystem layer provided for testing |
| SocketCAN on Windows | `python-can` SocketCAN backend requires Linux | Serial bridge mode works on all platforms |
| `tegrastats` power reading | Requires Jetson hardware | Graceful fallback to zero values |
| MCP2515 clock sensitivity | 16 MHz crystal required for 500 kbps | Documented in firmware header |
| COBS frame synchronization | Recovery after serial corruption may take 1-2 frames | Buffer flush on frame error implemented |

### 12.2 Open Items (Phase 8 Targets)

| ID | Item | Priority |
|----|------|---------|
| HW-001 | CAN-FD migration (2 Mbps data phase) for radar burst frames | Medium |
| HW-002 | AUTOSAR ComStack integration for production deployment | Low |
| HW-003 | CAN bus traffic replay system for regression testing | High |
| HW-004 | Hardware-in-loop (HIL) auto-discovery integration | High |
| HW-005 | Encrypted serial transport (AES-128) for security-critical commands | Medium |

---

## Appendix A — File Inventory

```
configs/can/
  apex_v4.dbc                    CAN database (DBC format)

configs/hardware/
  auto_discovery.yaml            Hardware inventory and rules

scripts/
  gen_dbc.py                     DBC validator and generator

src/hardware/
  can_integrity.py               CRC-8 engine + sequence tracker + codec
  arduino_bridge.py              Dual Arduino host coordinator
  auto_discovery.py              Hardware auto-discovery HAL

src/hardware/drivers/
  starkenn_can_driver.py         Starkenn 77GHz radar decoder
  mti680_can_driver.py           Xsens MTi-680 GNSS/INS decoder

deploy/arduino_can_bridge/
  arduino_can_bridge.ino         Arduino Mega 2560 firmware

tests/
  test_can_integrity.py          CRC, sequence, payload, throughput tests
  test_auto_discovery.py         Mock filesystem, fallback, HUD tests

docs/phase7/
  S3_HANDOFF.md                  This document
```

---

*APEX-LKA Phase 7 Section 3 — Generated: 2026-10-18 | Author: APEX Engineering*
