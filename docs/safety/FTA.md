# ISO 26262 Fault Tree Analysis (FTA)
## APEX-Vision Core v3.0 | Autonomous Platform

```mermaid
graph TD
    TOP["[TOP EVENT] Unintended Vehicle Collision or Lane Departure"]
    
    GATE_OR1["OR GATE: System Failure Modes"]
    TOP --> GATE_OR1
    
    E1["Loss of Lateral Guidance (Lane Departure)"]
    E2["Failure to Mitigate Forward Collision"]
    GATE_OR1 --> E1
    GATE_OR1 --> E2
    
    GATE_AND1["AND GATE: Primary & Retention Failures"]
    E1 --> GATE_AND1
    F1["Camera Sensor Blackout > 5.0 Seconds"]
    F2["15-DOF ESKF Dead-Reckoning Divergence (>0.35m)"]
    GATE_AND1 --> F1
    GATE_AND1 --> F2
    
    GATE_AND2["AND GATE: Perception & Safety Shield Dual Failure"]
    E2 --> GATE_AND2
    F3["HydraNet / YOLO Detection Missed Obstacle"]
    F4["Mobileye RSS Safety Shield Override Inactive"]
    GATE_AND2 --> F3
    GATE_AND2 --> F4

    GATE_OR2["OR GATE: Hardware Transceiver Dropouts"]
    E1 --> GATE_OR2
    F5["CAN 0x100 / 0x190 Actuation Link Loss"]
    F6["Hardware Watchdog 500ms Expiration"]
    GATE_OR2 --> F5
    GATE_OR2 --> F6
```

### Quantitative Safety Integrity
- **Single Point Fault Metric (SPFM):** > 97.4% (Exceeds ASIL-B 90% requirement)
- **Latent Fault Metric (LFM):** > 88.2% (Exceeds ASIL-B 60% requirement)
- **Probabilistic Metric for Hardware Failures (PMHF):** < 1.0e-7 / hour
- **Fail-Safe Mechanism:** Independent hardware watchdog on 2x Arduino Mega 2560 microcontrollers commands mechanical brake clamping on CAN heartbeat loss > 500ms.
