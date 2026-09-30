# APEX-LKA NVIDIA Jetson Orin Nano Deployment & Integration Manual

## 1. System Architecture Overview

The autonomous vehicle setup utilizes an **NVIDIA Jetson Orin Nano (4GB or 8GB)** as the central perception and decision-making engine, interfaced with **2× Arduino Mega 2560 microcontrollers** over an automotive differential CAN Bus (500 kbps) with redundant USB Serial fallback.

```
                      ┌────────────────────────────┐
                      │  NVIDIA Jetson Orin Nano   │
                      │  - TensorRT INT8/FP16 Engine│
                      │  - Watchdog Supervisor     │
                      │  - VehicleCommunicator     │
                      └─────────────┬──────────────┘
                                    │
          ┌─────────────────────────┴────────────────────────┐
          │ CAN Bus (500 kbps, MCP2515) & USB Serial         │
          ▼                                                  ▼
┌───────────────────────────┐                      ┌───────────────────────────┐
│   Arduino Mega Node 1     │                      │   Arduino Mega Node 2     │
│   (Steering & Actuation)  │                      │ (Throttle, Brake & Speed) │
│ - Slew Rate Limiter (120°/s)│                    │ - Optical Wheel Counter   │
│ - 500ms Safety Watchdog   │                      │ - Hydraulic Brake Press.  │
│ - Manual Override ISR     │                      │ - 500ms Throttle Cutoff   │
└───────────────────────────┘                      └───────────────────────────┘
```

---

## 2. Jetson Orin Nano Environment Setup

### Prerequisites
- NVIDIA JetPack 5.1.2+ or 6.x (Ubuntu 20.04/22.04 LTS aarch64)
- Python 3.8+ / 3.10+
- Linux SocketCAN enabled (`can0`)

### Installation Steps
```bash
# 1. Clone or copy deployment package
git clone https://github.com/adityaBuilds96/APEX-LKA.git ~/APEX-LKA
cd ~/APEX-LKA

# 2. Install edge runtime dependencies (minimal footprint)
pip install -r requirements_edge.txt

# 3. Configure CAN interface (can0 at 500 kbps)
sudo ip link set can0 type can bitrate 500000
sudo ip link set up can0

# 4. Add current user to dialout group for USB Serial access
sudo usermod -aG dialout $USER
```

---

## 3. Hardware Wiring & Pinout Guide

For detailed wiring diagrams and noise suppression, see [arduino/WIRING_GUIDE.md](file:///c:/Users/chavh/OneDrive/LKA.MAIN/arduino/WIRING_GUIDE.md).

### MCP2515 CAN Transceiver SPI Connections
| MCP2515 Pin | Arduino Mega 2560 Pin | Function |
| :--- | :--- | :--- |
| `VCC` | `5V` | Power supply |
| `GND` | `GND` | Ground reference |
| `CS` | `Pin 53` | SPI Chip Select |
| `MOSI` | `Pin 51` | Master Out Slave In |
| `MISO` | `Pin 50` | Master In Slave Out |
| `SCK` | `Pin 52` | SPI Clock |
| `INT` | `Pin 2` | External Interrupt |

### Bus Termination Rules
- Exactly **$120\,\Omega$ termination resistors** must be bridged between `CAN_H` and `CAN_L` at both extreme physical ends of the bus (one on Node 1, one on Node 2).
- Intermediate stubs must NOT have termination enabled.

---

## 4. Communication Protocol & Telemetry

### CAN Bus Frame Matrix
- **`0x050` Heartbeat Broadcast:** Master alive token (`0xA5`), health code, and cycle counter.
- **`0x100` Steering Command:** Signed steering angle ($\pm 35.00^\circ$), speed ($\text{m/s}$), perception confidence, and mode (`AUTO`/`MANUAL`/`ESTOP`).
- **`0x200` Throttle/Brake Command:** Throttle percentage ($0\text{--}100\%$), brake percentage ($0\text{--}100\%$).
- **`0x300` Node 1 Feedback:** Actual steering angle, motor current (mA), override state, fault flags.
- **`0x301` Node 2 Feedback:** Measured vehicle wheel speed ($\text{m/s}$), brake pressure (PSI).

---

## 5. Running Standalone Inference & Auto-Recovery

### Standalone Perception Engine
Execute the ultra-lightweight inference runner (< 30ms latency, sustained 30 FPS):
```bash
python deploy/jetson_run.py --config deploy/deploy_config.yaml
```

### Hardware Bench Verification
Test communications prior to vehicle integration:
```bash
# Verify CAN bus reception and transmission
python deploy/comm_test.py --test-can --can-channel can0

# Verify USB serial connection
python deploy/comm_test.py --test-serial --serial-port /dev/ttyACM0
```

### Autonomous Systemd Watchdog Service
Enable the supervisor watchdog to ensure uninterrupted operation:
```bash
sudo cp deploy/apex-watchdog.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now apex-watchdog.service

# Check status
sudo systemctl status apex-watchdog.service
```

If the main perception loop becomes unresponsive for $> 3.0$ seconds, the watchdog terminates and restarts the process. If $\ge 3$ crashes occur within 60 seconds, it enters **SAFE MODE** and commands vehicle emergency braking.
