# APEX-LKA: NVIDIA Jetson Orin Nano Deployment Guide
### SAE India BAJA 2027 Autonomous Lane Perception Platform

This document describes the deployment of the ultra-lightweight **APEX-LKA** inference engine on the **NVIDIA Jetson Orin Nano 4GB** developer kit connected to an **Arduino Mega 2560** steering controller.

---

## 1. System Architecture

```
                    +------------------------------------+
                    |       CSI / USB Camera (30 FPS)    |
                    +------------------------------------+
                                      |
                                      v (GStreamer / V4L2 zero-buffer)
+-----------------------------------------------------------------------------+
| NVIDIA Jetson Orin Nano 4GB                                                 |
|                                                                             |
|  [deploy/camera_interface.py] ---> Threaded Non-blocking Ring Buffer        |
|                                                  |                          |
|  [TensorRT Engine (.plan)]    <--- Normalized Float32 (1x3x360x640)         |
|        FP16 Precision                            |                          |
|        Latency < 10ms                            v                          |
|                                    4-Class Segmentation Logits              |
|                                                  |                          |
|  [deploy/jetson_run.py]       <--- Softmax + 2nd Order Polynomial Fitting   |
|        Steering Angle (-30° to +30°)             |                          |
|        Lateral Offset (meters)                   |                          |
|        Emergency Stop Watchdog                   |                          |
+-----------------------------------------------------------------------------+
                      |                                   |
                      | Serial (115200 baud)              | CAN Bus (500 kbps)
                      v                                   v
        +---------------------------+        +---------------------------+
        |   Arduino Mega 2560       |        |   Vehicle CAN Network     |
        |   Steering Actuator PID   |        |   Telemetry & E-STOP      |
        +---------------------------+        +---------------------------+
```

---

## 2. Hardware Wiring & Pinout

### A. Arduino Mega 2560 Serial Connection
Connect the Arduino Mega to one of the Jetson USB ports via standard USB Type-B cable:
- Port: `/dev/ttyUSB0` (or `/dev/ttyACM0`)
- Baudrate: `115200`
- Handshake: None, non-blocking asynchronous transmission

### B. CAN Bus Transceiver (SN65HVD230 / Waveshare CAN)
Jetson Orin Nano has native CAN controller pins on the 40-pin expansion header:
- **PIN 29 (CAN0_DIN / RX)** $\rightarrow$ Transceiver **RXD**
- **PIN 31 (CAN0_DOUT / TX)** $\rightarrow$ Transceiver **TXD**
- **PIN 2 or 4 (5V or 3.3V)** $\rightarrow$ Transceiver **VCC**
- **PIN 6 or 9 (GND)** $\rightarrow$ Transceiver **GND**
- Transceiver **CAN_H** and **CAN_L** connect to vehicle CAN bus with a $120\ \Omega$ terminating resistor.

### C. CSI Camera Installation (Sony IMX219 / IMX477)
- Connect 15-pin/22-pin ribbon cable to **CAM0 (J13)** connector on Jetson Orin Nano.
- Ensure the blue side of the ribbon faces outward away from the heatsink.

---

## 3. One-Command Setup

Run the automated installation script on your Jetson:
```bash
cd /home/jetson/apex-lka/deploy
chmod +x install_jetson.sh
sudo ./install_jetson.sh
```

The script will:
1. Install GStreamer, V4L2, and SocketCAN utilities.
2. Grant current user permissions for `video`, `dialout`, and `can`.
3. Install minimal edge Python dependencies (`deploy/requirements_edge.txt`).
4. Configure SocketCAN `can0` to auto-initialize at 500 kbps.
5. Install and enable `apex-lka.service` systemd service for boot-time startup.

---

## 4. TensorRT Compilation for Jetson Orin Nano

To compile the universal ONNX model into a Jetson-native TensorRT engine (.plan):
```bash
# Export ONNX model from repository
python3 run.py export

# Build TensorRT engine with FP16 and 512MB workspace budget
trtexec --onnx=models/exported/best_model.onnx \
        --saveEngine=models/exported/best_model.plan \
        --fp16 \
        --memPoolSize=workspace:512
```

---

## 5. Running the Perception Engine

### Manual Execution (Debug Window Enabled)
```bash
python3 deploy/jetson_run.py --config deploy/deploy_config.yaml --debug
```

### Headless Production Run
```bash
python3 deploy/jetson_run.py
```

### Video Recording Mode (Saves overlays to SD card)
```bash
python3 deploy/jetson_run.py --record
```

---

## 6. Communication Protocols

### Serial NMEA-Style Telemetry Packet
Sent every frame at 30 Hz:
```
$APEX,<STEERING_DEG>,<LATERAL_OFFSET_M>,<CONFIDENCE>,<STATUS>,<ESTOP>*<CHECKSUM>\n
```
Example:
```
$APEX,+04.25,-0.18,0.95,BOTH,0*3F
```
- `<STEERING_DEG>`: Target wheel steering angle ($-30.00^\circ$ to $+30.00^\circ$).
- `<LATERAL_OFFSET_M>`: Lateral distance from vehicle centerline to lane center in meters.
- `<CONFIDENCE>`: Perception confidence score ($0.00$ to $1.00$).
- `<STATUS>`: `BOTH`, `LEFT_ONLY`, `RIGHT_ONLY`, or `NONE`.
- `<ESTOP>`: `1` if safety threshold exceeded, `0` during nominal driving.
- `<CHECKSUM>`: Standard XOR hex checksum.

### CAN Bus Frame Specification (Arbitration ID `0x100`)
| Byte | Data Field | Format | Scaling | Range |
| :--- | :--- | :--- | :--- | :--- |
| 0-1 | Steering Angle | Signed 16-bit int | $0.01^\circ$ / LSB | -3000 to +3000 |
| 2-3 | Lateral Offset | Signed 16-bit int | 1 mm / LSB | -3000 to +3000 mm |
| 4 | Confidence | Unsigned 8-bit int | 1% / LSB | 0 to 100% |
| 5 | E-STOP Flag | Unsigned 8-bit int | Boolean | 0 (OK), 1 (STOP) |
| 6 | Status Code | Unsigned 8-bit int | Enum | 0: NONE, 1: LEFT, 2: RIGHT, 3: BOTH |
| 7 | Counter | Unsigned 8-bit int | Rolling counter | 0 to 255 |

---

## 7. Systemd Service Lifecycle

Manage the background autonomous service:
```bash
# Check service health and live FPS
sudo systemctl status apex-lka.service

# View live log output
journalctl -u apex-lka.service -f

# Stop perception
sudo systemctl stop apex-lka.service

# Restart perception
sudo systemctl restart apex-lka.service
```

---

## 8. Failsafe Emergency Stop & Watchdog

- **Catastrophic Failure Watchdog:** If lane markings are lost (`confidence < 0.20` or status `NONE`) for more than 30 consecutive frames (~1.0 second), the system asserts `ESTOP = 1`, zeroes the steering angle, and signals the Arduino to apply brakes.
- **Hardware Fault Isolation:** If camera or serial cables disconnect during driving, auto-reconnect routines attempt reconnection without terminating the process or crashing the OS.
