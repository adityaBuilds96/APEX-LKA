# APEX LKA • System Architecture & Technical Diagrams

This document collects formal architectural blueprints, dataflow diagrams, and state machines documenting the APEX LKA autonomous vehicle platform.

---

## 1. End-to-End System Dataflow

```mermaid
flowchart LR
    subgraph SENSORS["1. SENSING LAYER"]
        CAM["CSI Camera<br/>640x360 @ 30 FPS"]
        IMU["6-Axis IMU<br/>100 Hz"]
        GNSS["GNSS RTK<br/>10 Hz"]
        ENC["Wheel Encoders<br/>Tick Rates"]
    end

    subgraph PERCEPTION["2. PERCEPTION & DEPTH LAYER"]
        HAL["Hardware Abstraction Layer<br/>(TensorRT / Edge TPU / OpenVINO)"]
        SEG["DeepLabV3+ Lane Head<br/>Road & Lanes"]
        OBJ["YOLO Multi-Task Head<br/>Obstacles & Pedestrians"]
        DEP["Monocular Metric Depth<br/>0-15m Range"]
        OCC["Occupancy Grid Mapping<br/>0.1m Resolution"]
    end

    subgraph ESTIMATION["3. FUSION & STATE ESTIMATION"]
        ESKF["Error-State Kalman Filter<br/>15-State Fusion"]
        TRACK["ByteTrack Multi-Object<br/>3D Velocities"]
        HOLD["Centerline Retention<br/>Dead Reckoning"]
    end

    subgraph DECISION["4. PLANNING & SAFETY LAYER"]
        BEH["Behavior State Machine<br/>Lane Follow / Yield"]
        PATH["Quintic Spline Planner<br/>Obstacle Evasion"]
        RSS["RSS Safety Shield<br/>Decoupled Safety Headway"]
    end

    subgraph ACTUATION["5. VEHICLE CONTROL & ACTUATION"]
        CTRL["Stanley & Pure Pursuit<br/>Dual Kinematic Steering"]
        NET["Network Adapter<br/>CAN 2.0B / CAN-FD / ROS 2"]
        ARD["Dual Arduino Mega 2560<br/>PWM & Mechanical Spring Brakes"]
    end

    CAM --> HAL
    HAL --> SEG & OBJ & DEP
    DEP --> OCC
    IMU & GNSS & ENC --> ESKF
    SEG --> HOLD
    OBJ --> TRACK
    ESKF & HOLD & OCC & TRACK --> BEH
    BEH --> PATH
    PATH --> RSS
    RSS --> CTRL
    CTRL --> NET
    NET --> ARD
```

---

## 2. Hardware Interconnect & Wiring Blueprint

```
+----------------------------------------------------------------------------------------------------+
|                                    NVIDIA JETSON ORIN NANO 4GB                                     |
|                                                                                                    |
|   +--------------------------+    +--------------------------+    +----------------------------+   |
|   | MIPI-CSI2 Camera (IMX219)|    | USB 3.2 Type-A           |    | Pin 8/10 (UART0) / USB     |   |
|   | 640x360 @ 30 FPS         |    | Vector/Peak CAN-to-USB   |    | Dual Arduino Mega Serial   |   |
|   +--------------------------+    +--------------------------+    +----------------------------+   |
+-------------------------------------------------|--------------------------------|-----------------+
                                                  |                                |
                                          CAN High / CAN Low               115200 Baud UART
                                                  |                                |
                                                  v                                v
+----------------------------------------------------------------------------------------------------+
|                                    DUAL ARDUINO ACTUATION RIG                                      |
|                                                                                                    |
|   +---------------------------------------------------+      +---------------------------------+   |
|   |            PRIMARY CONTROLLER (MEGA 2560)         |      |    SECONDARY SAFETY SUPERVISOR  |   |
|   |                                                   |      |                                 |   |
|   |  - MCP2515 CAN SPI Module (500 kbps)              |      |  - Steering Potentiometer (ADC) |   |
|   |  - Stepper Direction & Step Pins (PWM)            |----->|  - Discrepancy Delta Checker    |   |
|   |  - Throttle DAC Output (0-5V)                     |      |  - Optoisolated Cutout Relay    |   |
|   |  - 100ms Hardware Watchdog Timer                  |      |  - Manual E-Stop Pushbutton     |   |
|   +---------------------------------------------------+      +---------------------------------+   |
|                             |                                                |                     |
+-----------------------------|------------------------------------------------|---------------------+
                              v                                                v
               +-----------------------------+                  +-----------------------------+
               | High-Torque Steering Servo  |                  | Normally-Closed Brake Valve |
               | Actuator (BAJA Tie-Rods)    |                  | Mechanical Spring Actuator  |
               +-----------------------------+                  +-----------------------------+
```

---

## 3. Failsafe State Machine (ISO 26262 ASIL-B)

```mermaid
stateDiagram-v2
    [*] --> INITIALIZING: System Power-On
    INITIALIZING --> NORMAL_AUTONOMOUS: Self-Tests Passed & Heartbeat Active
    INITIALIZING --> HARDWARE_FAULT: Driver / CAN / Sensor Init Failed

    state NORMAL_AUTONOMOUS {
        [*] --> LANE_KEEPING
        LANE_KEEPING --> YIELDING: Obstacle in Corridor (TTC < 2.5s)
        YIELDING --> LANE_KEEPING: Obstacle Cleared
    }

    NORMAL_AUTONOMOUS --> DEGRADED_HOLD: Transient Sensor Drop (Blur / Mud Splatter)
    DEGRADED_HOLD --> NORMAL_AUTONOMOUS: Perception Confirmed Restored (< 3.0s)
    DEGRADED_HOLD --> CONTROLLED_SAFE_STOP: Sensor Dropout Exceeds 3.0s

    NORMAL_AUTONOMOUS --> EMERGENCY_BRAKING: RSS Shield Violation (TTC < 1.8s)
    EMERGENCY_BRAKING --> NORMAL_AUTONOMOUS: Hazard Removed & Path Clear
    EMERGENCY_BRAKING --> CONTROLLED_SAFE_STOP: Hazard Persists at Standstill

    NORMAL_AUTONOMOUS --> HARDWARE_FAULT: CAN Bus-Off / Heartbeat Timeout (> 150ms)
    DEGRADED_HOLD --> HARDWARE_FAULT: ESKF Divergence (> 0.5m)
    CONTROLLED_SAFE_STOP --> FAIL_SAFE_LOCK: Vehicle Speed == 0.0 m/s
    HARDWARE_FAULT --> FAIL_SAFE_LOCK: Immediate Power Relay De-energize

    state FAIL_SAFE_LOCK {
        [*] --> MECHANICAL_SPRING_BRAKE_LOCKED
        MECHANICAL_SPRING_BRAKE_LOCKED --> OPERATOR_MANUAL_RESET: Physical Key Turn
    }
```

---

## 4. Multi-Hardware Abstraction Layer (HAL) Class Hierarchy

```mermaid
classDiagram
    class HardwareBackend {
        <<abstract>>
        +String name
        +String backend_type
        +is_available() bool*
        +initialize() bool*
        +load_model(path, task) Any*
        +predict(tensor) Dict*
        +get_system_telemetry() Dict*
        +unload() void*
    }

    class JetsonBackend {
        -bool _is_jetson
        -bool _has_cuda
        +predict(tensor) Dict
        +get_system_telemetry() Dict
    }

    class CoralTPUBackend {
        -bool _tpu_present
        +predict(tensor) Dict
        +get_system_telemetry() Dict
    }

    class OpenVINOBackend {
        -bool _openvino_available
        +predict(tensor) Dict
        +get_system_telemetry() Dict
    }

    class CPUGenericBackend {
        +predict(tensor) Dict
        +get_system_telemetry() Dict
    }

    class HardwareAbstractionLayer {
        -HardwareBackend _active_backend
        -FunctionPointer _fast_predict_fn
        +predict(tensor) Dict
        +set_backend(name) bool
        +get_telemetry() Dict
        +measure_dispatch_overhead_us() float
    }

    HardwareBackend <|-- JetsonBackend
    HardwareBackend <|-- CoralTPUBackend
    HardwareBackend <|-- OpenVINOBackend
    HardwareBackend <|-- CPUGenericBackend
    HardwareAbstractionLayer o-- HardwareBackend
```
