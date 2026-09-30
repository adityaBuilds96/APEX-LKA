# APEX-LKA: Hardware Wiring & CAN Bus Setup Guide
### NVIDIA Jetson Orin Nano & Dual Arduino Mega 2560 Node Network
**SAE India BAJA 2027 Autonomous Category**

---

## 1. System Physical Wiring Overview

```
                      +---------------------------------------+
                      |         24V / 12V LiFePO4 Battery      |
                      +---------------------------------------+
                                          |
                      +---------------------------------------+
                      | Isolated DC-DC Step-Down Converters    |
                      |   [12V 5A -> Jetson]   [5V 5A -> Mega] |
                      +---------------------------------------+
                                          |
                        +-----------------+-----------------+
                        | (COMMON CHASSIS GROUND STAR-POINT)|
                        +-----------------+-----------------+
                                          |
          +-------------------------------+-------------------------------+
          |                               |                               |
          v                               v                               v
+-----------------------+       +-----------------------+       +-----------------------+
| Jetson Orin Nano      |       | Arduino Mega Node 1   |       | Arduino Mega Node 2   |
| (Perception & Plan)   |       | (Steering Actuation)  |       | (Throttle / Brake)    |
|                       |       |                       |       |                       |
| CAN Transceiver       |       | MCP2515 SPI Module    |       | MCP2515 SPI Module    |
| [120Ω TERMINATION]    |       | [STUB < 0.3m]         |       | [120Ω TERMINATION]    |
+-----------------------+       +-----------------------+       +-----------------------+
            |                               |                               |
CAN_H ======+===============================+===============================+====== CAN_H
CAN_L ======+===============================+===============================+====== CAN_L
```

---

## 2. Arduino Mega 2560 to MCP2515 CAN Transceiver Pinout

Both Arduino Mega 2560 nodes connect to their respective **MCP2515 CAN Bus Controller (with TJA1050 Transceiver)** via hardware SPI:

| MCP2515 Pin | Arduino Mega 2560 Pin | Function | Notes |
| :---: | :---: | :---: | :--- |
| **VCC** | **5V** | Logic & Transceiver Power | Requires stable 5.0V regulated power |
| **GND** | **GND** | Signal Ground | Connect to star-ground reference |
| **CS** | **Pin 53** | Hardware SPI Slave Select (SS) | Dedicated hardware chip select |
| **MOSI** | **Pin 51** | Master Out Slave In | Hardware SPI Master Out |
| **MISO** | **Pin 50** | Master In Slave Out | Hardware SPI Master In |
| **SCK** | **Pin 52** | SPI Serial Clock | Hardware SPI Clock |
| **INT** | **Pin 2** | External Interrupt 0 | Active-low frame received interrupt |

> **IMPORTANT:** Ensure the crystal on your MCP2515 module matches the code setting:
> - Most standard blue breakout boards use a **16.0 MHz** crystal (`MCP_16MHZ`).
> - Some compact modules use an **8.0 MHz** crystal (`MCP_8MHZ`).
> Verify the label printed on the silver crystal oscillator can on the board.

---

## 3. Node 1 & Node 2 Sensor / Actuator Pinout

### Node 1 (Steering & Actuator Controller)
- **Pin 9 (PWM):** Steering Actuator Motor Driver PWM (Cytron / BTS7960 / H-Bridge).
- **Pin 8 (DIR):** Steering Actuator Motor Driver Direction.
- **Pin 18 (INT 5):** Driver Manual Takeover Limit Switch (Normally High, pulled to GND on driver grab).
- **Pin A0 (Analog):** Steering Rack Absolute Position Feedback (Bourns Potentiometer / Hall sensor).
- **Pin A1 (Analog):** Actuator Motor Current Shunt Sensor (ACS712 / Allegro sensor).

### Node 2 (Throttle, Brake & Speed Controller)
- **Pin 6 (PWM):** Electronic Throttle Control (RC Servo / Motorized throttle DAC).
- **Pin 5 (PWM):** Brake Hydraulic Linear Actuator PWM.
- **Pin 4 (DIR):** Brake Linear Actuator Direction.
- **Pin 3 (INT 1):** Wheel Speed Optical/Hall Effect Pulse Sensor on front axle.
- **Pin 19 (INT 4):** Driver Brake Pedal Override Pressure Switch.
- **Pin A2 (Analog):** Hydraulic Brake Line Pressure Transducer (0-1500 PSI, 0.5V-4.5V output).

---

## 4. CAN Bus Bus-Topology & Termination Rules

1. **Strict Linear Bus (Daisy Chain):**
   - The CAN bus must form a continuous linear transmission line without star or ring branches.
   - Any stub connecting an Arduino node to the main trunk must be **shorter than 0.3 meters (12 inches)**.
2. **Termination Resistors ($120\ \Omega$):**
   - High-speed CAN ($500\text{ kbps}$) requires exactly **two $120\ \Omega$ terminating resistors** across `CAN_H` and `CAN_L`.
   - Place one termination resistor at the physical beginning (Jetson Orin Nano / Transceiver).
   - Place the second termination resistor at the physical end (Arduino Mega Node 2).
   - Node 1 is an intermediate node in the middle of the bus and must have its termination jumper **REMOVED** (or switch OPEN).
   - **Verification:** Measure resistance between `CAN_H` and `CAN_L` with system powered off: it must read **$60\ \Omega \pm 4\ \Omega$** (two $120\ \Omega$ resistors in parallel).

---

## 5. Grounding & Noise Suppression (Critical for SAE BAJA)

High-power brushless steering motors and petrol engine ignition coils create severe inductive EMF spikes. To guarantee zero communication dropouts:

1. **Star-Grounding Point:**
   - Establish a single brass star-grounding bolt on the vehicle chassis.
   - Ground the Jetson, both Arduino Megas, the CAN transceivers, and the motor drivers directly to this point.
   - Never daisy-chain high-current motor grounds through logic grounds.
2. **Shielded Twisted Pair (STP):**
   - Use dedicated shielded twisted-pair cabling for `CAN_H` and `CAN_L`.
   - Ground the cable shield at **one end only** (at the Jetson chassis ground) to prevent ground loops.
3. **Decoupling Capacitors:**
   - Place a $100\ \mu\text{F}$ electrolytic and a $0.1\ \mu\text{F}$ ceramic capacitor across the $5\text{V}$ and $\text{GND}$ rails of each MCP2515 module.
