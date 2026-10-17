# APEX-Vision Core v4.0 • Operational Design Domain (ODD) Specification
## Multi-Profile Operational Boundaries, Environmental Limits & Minimal Risk Conditions
**Standard Alignment:** Developed following an ISO 26262-style process; ASIL-B-aligned; not independently assessed  
**Taxonomy Standard:** ISO 34503 / BSI PAS 1883 Operational Design Domain Taxonomy  
**Autonomy Classification:** Driver-supervised SAE Level 2  
**Target Hardware:** NVIDIA Jetson Orin Nano 4GB + Dual Arduino Mega 2560 Watchdogs  
**Date of Issue:** 2026-10-17  
**Status:** FROZEN SPECIFICATION — PHASE 7 BASELINE  

---

## 1. Executive Overview & Scope

This Operational Design Domain (ODD) specification establishes the formal operating conditions, environmental boundaries, dynamic constraints, and sensor operational envelopes within which the **APEX-Vision Core v4.0** autonomous driving platform is designed to function safely.

In accordance with **SAE J3016** (Driver-Supervised Level 2), the driver remains permanently responsible for monitoring the driving environment, supervising system actions, and executing manual steering or braking takeovers whenever the vehicle approaches or exceeds the ODD boundaries specified herein.

The platform provides polymorphic vehicle abstraction via runtime YAML configurations supporting four distinct operational profiles:
1. **Primary Target:** BAJA SAE Off-Road Competition Buggy (`configs/profiles/baja_2027.yaml`)
2. **Secondary Profile 1:** Low-Speed Autonomous Campus Shuttle (`configs/profiles/campus_shuttle.yaml`)
3. **Secondary Profile 2:** Industrial Warehouse Automated Guided Vehicle (`configs/profiles/agv_warehouse.yaml`)
4. **Secondary Profile 3:** Standard Production Passenger EV (`configs/profiles/passenger_car.yaml`)

---

## 2. Profile 1: BAJA SAE Buggy (Primary Target Profile)

### 2.1 Vehicle Kinematics & Physical Geometry
The BAJA SAE Buggy profile represents a single-seat, ruggedized off-road vehicle operating on unpaved dirt, gravel trails, and competition courses with high mechanical shock and severe chassis vibration.

| Parameter | Value | Metric Units | Engineering Specification & Rationales |
|:---|:---:|:---:|:---|
| **Wheelbase ($L$)** | $1.60$ | $\text{m}$ | Compact wheelbase for agile cornering on tight dirt circuits |
| **Track Width ($W_{\text{track}}$)** | $1.40$ | $\text{m}$ | Symmetrical front/rear track width |
| **Overall Length** | $2.25$ | $\text{m}$ | Front bumper to rear engine frame |
| **Overall Width** | $1.62$ | $\text{m}$ | Outer tire tread sidewalls |
| **Curb Weight** | $220.0$ | $\text{kg}$ | Dry vehicle weight without driver |
| **Gross Vehicle Weight** | $310.0$ | $\text{kg}$ | Loaded with 90 kg driver and instrumentation |
| **Center of Gravity Height ($h_{\text{cg}}$)**| $0.48$ | $\text{m}$ | Low CG configuration with roll cage |
| **Distance COG to Rear Axle ($l_r$)** | $0.85$ | $\text{m}$ | Rear weight bias (53.1% rear distribution) |
| **Distance COG to Front Axle ($l_f$)** | $0.75$ | $\text{m}$ | Front weight bias (46.9% front distribution) |
| **Steering Ratio** | $1.0 : 1$ | ratio | Direct mechanical rack-and-pinion steering without boost |
| **Max Physical Steering Angle ($\delta_{\max}$)** | $\pm 28.0^\circ$ | degrees | Hard tie-rod mechanical stops at steering knuckles |
| **Max Steering Slew Rate ($\dot{\delta}_{\max}$)** | $40.0^\circ/\text{s}$ | $\text{deg/s}$ | High-torque servo actuator safe operating velocity |
| **Max Autonomous Velocity ($v_{\max}$)** | $12.5$ ($45.0\,\text{km/h}$) | $\text{m/s}$ | Competition course maximum operating ceiling |
| **Max Longitudinal Acceleration** | $+3.5$ | $\text{m/s}^2$ | Briggs & Stratton single-cylinder engine torque limit |
| **Max Service Deceleration** | $-4.5$ | $\text{m/s}^2$ | Hydraulic disc brakes on all four wheels |
| **Emergency Braking Deceleration** | $-7.0$ | $\text{m/s}^2$ | Hard lockup decel on dry compacted clay |
| **Max Suspension Travel** | $0.25$ ($250\,\text{mm}$) | $\text{m}$ | Long-travel dual A-arm front / trailing arm rear |
| **Dynamic Pitch Excursion** | $\pm 8.0^\circ$ | degrees | Braking dive, acceleration squat, and bump cresting |
| **Dynamic Roll Excursion** | $\pm 6.0^\circ$ | degrees | High-speed rut cornering roll compliance |

### 2.2 Operational Terrain & Roadway Constraints
* **Surface Materials:** Compacted dirt, clay, loose gravel, dry sand, crushed stone, grass edges, and shallow mud.
* **Prohibited Surfaces:** Deep standing water ($> 0.15\,\text{m}$ depth), boulder fields with step obstacles $> 0.20\,\text{m}$, uncompacted sand dunes with slope $> 25^\circ$.
* **Lane Markings:** Entirely unpainted, non-standardized. Path boundaries are established via dirt-to-vegetation contrast, tire ruts, earth berms, and trail edges.
* **Track Width:** Minimum allowable navigable corridor width is $2.20\,\text{m}$ ($1.57\times$ vehicle track).
* **Maximum Grade / Incline:** Longitudinal slope up to $\pm 20^\circ$ ($36.4\%$). Lateral cross-slope / banking up to $\pm 12^\circ$ ($21.3\%$).

### 2.3 Environmental & Weather Envelopes
* **Ambient Temperature:** $-5^\circ\text{C}$ to $+45^\circ\text{C}$ operating range.
* **Optical Illumination:** $500\,\text{lux}$ (overcast twilight) to $120,000\,\text{lux}$ (direct midday desert sun). Night operation permitted ONLY with vehicle forward LED light bar ($\ge 2500\,\text{lumens}$).
* **Atmospheric Particulates:** Airborne dust plumes, gravel spray, low-density splash droplets.
* **Precipitation:** Light to moderate rain ($\le 10\,\text{mm/hr}$). Severe monsoon downpours or heavy snowfall exceed ODD.
* **Lens Occlusion Limit:** Mud or dust coating covering $> 25\%$ of camera active sensor area requires immediate fallback transition to Degraded Hold.

---

## 3. Profile 2: Autonomous Campus Shuttle

### 3.1 Kinematics & Actuation Envelope
The Campus Shuttle profile represents a low-speed, multi-passenger electric shuttle transporting students along university paved paths and pedestrian plazas.

| Parameter | Value | Metric Units | Description |
|:---|:---:|:---:|:---|
| **Wheelbase ($L$)** | $2.80$ | $\text{m}$ | Extended wheelbase for passenger cabin packaging |
| **Track Width ($W_{\text{track}}$)** | $1.85$ | $\text{m}$ | Low-speed passenger chassis track |
| **Gross Vehicle Weight** | $1850.0$ | $\text{kg}$ | Loaded with 8 passengers and lithium battery pack |
| **Steering Ratio** | $14.5 : 1$ | ratio | Power-assisted electric steering column |
| **Max Steering Angle** | $\pm 32.0^\circ$ | degrees | High maneuverability for tight campus pathways |
| **Max Steering Slew Rate** | $25.0^\circ/\text{s}$ | $\text{deg/s}$ | Smooth passenger comfort rate limiter |
| **Max Autonomous Velocity** | $6.94$ ($25.0\,\text{km/h}$) | $\text{m/s}$ | Low-speed campus safety zone speed limit |
| **Max Comfort Deceleration** | $-1.8$ | $\text{m/s}^2$ | Nominal passenger comfort braking threshold |
| **Emergency Braking Deceleration** | $-5.5$ | $\text{m/s}^2$ | Full regenerative + hydraulic brake emergency stop |
| **Camera Mount Height** | $2.10$ | $\text{m}$ | Roofline panoramic forward mount |
| **Nominal Camera Pitch** | $7.5^\circ$ | degrees | Downward angle optimized for near-field pedestrians |

### 3.2 Operating Environment & Pedestrian Interactions
* **Surface Materials:** Smooth asphalt, concrete pavers, interlocking brick walkways, painted crosswalks.
* **Traffic Context:** Mixed pedestrian traffic, bicycles, electric scooters, low-speed service vehicles.
* **Minimum Path Width:** $3.20\,\text{m}$ clear drivable width.
* **Pedestrian Safety Margin:** Emergency stop commanded if any pedestrian penetrates within lateral envelope $\pm 1.2\,\text{m}$ with Time-to-Impact (TTI) $< 2.0\,\text{s}$.

---

## 4. Profile 3: Industrial Warehouse AGV

### 4.1 Kinematics & Indoor Constraints
The Warehouse AGV profile represents an industrial material handling platform operating in indoor logistics fulfillment centers.

| Parameter | Value | Metric Units | Description |
|:---|:---:|:---:|:---|
| **Wheelbase ($L$)** | $1.10$ | $\text{m}$ | Short wheelbase for narrow aisle turning |
| **Track Width ($W_{\text{track}}$)** | $0.95$ | $\text{m}$ | Ultra-compact indoor footprint |
| **Gross Vehicle Weight** | $650.0$ | $\text{kg}$ | Including 300 kg pallet payload |
| **Steering Mechanism** | Differential / Ackermann | dual | Supported steering kinematics |
| **Max Steering Angle** | $\pm 45.0^\circ$ | degrees | Extreme cornering capability |
| **Max Autonomous Velocity** | $4.17$ ($15.0\,\text{km/h}$) | $\text{m/s}$ | OSHA indoor warehouse speed regulation |
| **Max Service Deceleration** | $-2.5$ | $\text{m/s}^2$ | Pallet load stability constraint |
| **Emergency Deceleration** | $-4.0$ | $\text{m/s}^2$ | Magnetic drum fail-safe braking |
| **Camera Mount Height** | $0.85$ | $\text{m}$ | Forward mast mounting position |
| **Nominal Camera Pitch** | $3.0^\circ$ | degrees | Level horizon targeting floor markings |

### 4.2 Facility Infrastructure Constraints
* **Surface Materials:** Polished industrial concrete, epoxy-coated warehouse floors. Zero potholes or suspension deflection.
* **Lane & Guidance Markings:** High-contrast yellow/white painted boundary lines, floor barcodes, magnetic floor strips.
* **Illumination:** $150\,\text{lux}$ to $1,000\,\text{lux}$ artificial LED indoor industrial lighting. Zero solar glare.
* **Aisle Clearance:** Operates within high-density racking aisles with lateral clearance down to $\pm 0.25\,\text{m}$.

---

## 5. Profile 4: Production Passenger Car / EV

### 5.1 Kinematics & High-Speed Highway Envelope
The Passenger Car profile models a standard highway-capable passenger vehicle operating on marked multi-lane freeways and municipal roadways.

| Parameter | Value | Metric Units | Description |
|:---|:---:|:---:|:---|
| **Wheelbase ($L$)** | $2.70$ | $\text{m}$ | Mid-size passenger sedan standard wheelbase |
| **Track Width ($W_{\text{track}}$)** | $1.58$ | $\text{m}$ | Front/rear passenger tread width |
| **Gross Vehicle Weight** | $1750.0$ | $\text{kg}$ | Typical electric passenger vehicle curb weight |
| **Steering Ratio** | $15.2 : 1$ | ratio | Electronic Power Steering (EPS) rack ratio |
| **Max Steering Angle** | $\pm 35.0^\circ$ | degrees | Parking maneuverability limit |
| **Max Steering Slew Rate** | $120.0^\circ/\text{s}$ | $\text{deg/s}$ | High-speed emergency evasive steer rate |
| **Max Autonomous Velocity** | $22.2$ ($80.0\,\text{km/h}$) | $\text{m/s}$ | Highway lane keep assistance ceiling |
| **Max Service Deceleration** | $-3.5$ | $\text{m/s}^2$ | Standard ADAS ACC deceleration bound |
| **Emergency Deceleration** | $-9.0$ | $\text{m/s}^2$ | ABS-modulated dry asphalt emergency stop |
| **Camera Mount Height** | $1.35$ | $\text{m}$ | Windshield interior rear-view mirror mount |
| **Nominal Camera Pitch** | $4.5^\circ$ | degrees | Looking forward through swept windshield wiper area |

### 5.2 Infrastructure & Environmental Limits
* **Surface Materials:** Asphalt concrete, Portland cement concrete.
* **Lane Markings:** Standard solid/dashed white and yellow painted markings with retroreflective glass beads.
* **Minimum Lane Width:** $3.0\,\text{m}$ to $3.75\,\text{m}$.
* **Maximum Road Curvature:** Minimum radius of curvature $R \ge 150\,\text{m}$ at $80\,\text{km/h}$.

---

## 6. Environmental & Operational Boundaries Matrix

The following matrix formally defines the Go / No-Go operational thresholds across all environmental dimensions:

```
+---------------------------------------------------------------------------------------------------------+
| ENVIRONMENTAL PARAMETER       | NOMINAL (GO)            | DEGRADED (CAUTION)      | OUT OF ODD (NO-GO)      |
+-------------------------------+-------------------------+-------------------------+-------------------------+
| Ambient Illumination          | 1,000 - 80,000 lux      | 300 - 1,000 lux         | < 300 lux (no aux lamp) |
| Direct Optical Sun Glare      | Off-axis (> 15 deg)     | Low sun angle (5-15 deg)| Direct into lens (<5 deg)|
| Precipitation                 | Clear / Overcast        | Light rain <= 5 mm/hr   | Heavy rain > 10 mm/hr   |
| Fog / Optical Obscuration     | Visibility > 100 m      | Visibility 30 - 100 m   | Visibility < 30 m       |
| Airborne Dust (Off-Road)      | Light trail dust        | Intermittent dust cloud | Sustained blackout plume|
| Ambient Temperature           | 0°C to +40°C            | -10°C to 0°C / 40-50°C  | < -10°C or > +50°C      |
| Road Surface Water Depth      | Damp / Dry surface      | Standing water < 25 mm  | Standing water >= 50 mm |
| Lens Mud Splatter Area        | < 5% surface occluded   | 5% - 25% occluded       | > 25% active FOV covered|
| Maximum Track Cross-Slope     | <= 6.0 deg (10.5%)      | 6.0 - 12.0 deg          | > 12.0 deg (rollover)   |
| Maximum Track Gradient        | <= 10.0 deg (17.6%)     | 10.0 - 20.0 deg         | > 20.0 deg              |
| Radar Ground Clutter (77GHz)  | SNR > 15 dB             | SNR 6 - 15 dB           | SNR < 6 dB (blinded)    |
| GNSS RTK Solution Status      | RTK Fixed (cm-level)    | RTK Float (dm-level)    | Uncorrected GPS / Lost  |
| Max Host Edge Core Temp       | <= 75°C                 | 75°C - 85°C (throttled) | > 85°C (thermal shutdown)|
+---------------------------------------------------------------------------------------------------------+
```

---

## 7. Sensor Degradation Hierarchy & Minimal Risk Maneuver (MRM)

When operating conditions transition from **NOMINAL** to **DEGRADED** or **OUT OF ODD**, the platform executes a deterministic 6-level degradation hierarchy to guarantee passenger and vehicle safety:

```
[LEVEL 0: FULL NOMINAL OPERATION]
   │  All sensors (Camera, Radar, GNSS/INS) active; perception confidence >= 0.75
   ▼  (Drop in optical sharpness / partial lens occlusion / low confidence)
[LEVEL 1: TIER-1 OPTICAL ENHANCEMENT]
   │  CLAHE tone mapping clipLimit boosted from 1.5 to 4.0; auto-exposure ROI adapts
   ▼  (Optical perception temporarily occluded or mud burst)
[LEVEL 2: KINEMATIC DEAD-RECKONING HOLD]
   │  ESKF fuses 400Hz AHRS + wheel ticks; holds lane trajectory for <= 5.0 seconds (<0.35m drift)
   ▼  (Optical blackout exceeds 5.0 seconds OR persistent radar target within corridor)
[LEVEL 3: SENSOR-FUSED RADAR-CENTRIC SAFE-HOLD]
   │  Starkenn 77GHz CAN radar tracks primary corridor obstacles; speed restricted to 50%
   ▼  (Severe sensor failure / loss of CAN heartbeat / compute host brownout)
[LEVEL 4: GRACEFUL CONTROLLED DECELERATION]
   │  Vehicle commands service deceleration (-2.5 m/s²), aligns steering to center, activates hazards
   ▼  (Imminent collision TTI < 1.0s OR manual E-stop triggered)
[LEVEL 5: EMERGENCY FAIL-SAFE STOP (MRM)]
      Hard hydraulic brake lockup (-4.5 to -7.0 m/s²); dual Arduino watchdogs drop power relay
```

### 7.1 Minimal Risk Maneuver (MRM) Execution Rules
1. **Trigger Condition:** Any unrecoverable sensor fault exceeding Fault Tolerant Time Interval (FTTI = 500 ms), total loss of optical and radar perception, or human driver E-stop depression.
2. **Steering Policy:** Steering command is locked to $0.0^\circ$ (neutral straight-ahead) to prevent erratic departure into ditches or oncoming traffic.
3. **Deceleration Policy:** Commands maximum available profile deceleration until measured wheel odometry confirms velocity $v = 0.0\,\text{m/s}$.
4. **Mechanical Lockout:** Secondary safety supervisor microcontroller de-energizes the main actuation power relay, causing normally-closed spring-actuated brake valves to mechanically lock the wheels.

---

---

## 9. Sensor Suite Operating Envelopes & Physical Mounting Geometry

To ensure uncompromised perception coverage while respecting vehicle packaging constraints, the physical mounting geometry and operational sensing envelopes are defined across all four profiles:

### 9.1 Camera Subsystem Specifications (640x360 @ 30 FPS RGB)
- **Sensor Type:** Sony IMX219 / IMX477 Rolling Shutter CMOS (or industrial equivalent global shutter USB3).
- **Resolution:** $640 \times 360$ pixels active array.
- **Horizontal Field of View ($\text{HFOV}$):** $70.0^\circ$ ($\approx 1.2217\,\text{rad}$).
- **Vertical Field of View ($\text{VFOV}$):** $43.6^\circ$ ($\approx 0.761\,\text{rad}$).
- **Effective Focal Length:** $f_x \approx 450.0\,\text{px}, f_y \approx 450.0\,\text{px}$.
- **Principal Center Point:** $c_x = 320.0\,\text{px}, c_y = 180.0\,\text{px}$.
- **Near-Field Blind Zone:** Ground region directly in front of vehicle front bumper ($0.0\,\text{m}$ to $1.8\,\text{m}$) occluded by vehicle bonnet/hood.
- **Far-Field Range Horizon:** Usable metric road boundary resolution extends up to $40.0\,\text{m}$. Beyond $40.0\,\text{m}$, pixel ground resolution falls below $5.0\,\text{cm/px}$, degrading curve fitting.

### 9.2 Starkenn Orion 77GHz Automotive CAN Radar
- **Operating Frequency Band:** $76.0\,\text{GHz} - 77.0\,\text{GHz}$ FMCW.
- **Azimuth Field of View:** $\pm 45.0^\circ$ ($90.0^\circ$ total sweep) in Near-Field mode; $\pm 10.0^\circ$ ($20.0^\circ$ sweep) in Far-Field mode.
- **Maximum Detection Range:** $0.5\,\text{m}$ to $120.0\,\text{m}$ for passenger car targets ($RCS \ge 10\,\text{dBsm}$); $0.5\,\text{m}$ to $50.0\,\text{m}$ for pedestrian/debris targets ($RCS \ge 0.5\,\text{dBsm}$).
- **Range Accuracy:** $\pm 0.15\,\text{m}$ at distances $< 30.0\,\text{m}$.
- **Radial Velocity Measurement Range:** $-70.0\,\text{m/s}$ to $+70.0\,\text{m/s}$ ($\pm 252\,\text{km/h}$) with accuracy $\pm 0.10\,\text{m/s}$.
- **CAN Bus Protocol:** Transmits up to 32 target tracks over CAN IDs `0x400` through `0x408` at $20\,\text{Hz}$ ($50.0\,\text{ms}$ refresh rate).
- **Mounting Position:** Front centerline bumper at $H_{\text{radar}} = 0.55\,\text{m}$ above ground level, zero nominal tilt ($\theta_{\text{radar}} = 0.0^\circ$).

### 9.3 MTi-680 RTK GNSS/INS Sensor Kit
- **AHRS Orientation Update Rate:** $400.0\,\text{Hz}$ strapdown integration ($2.5\,\text{ms}$ cycle).
- **Roll / Pitch Dynamic Accuracy:** $0.2^\circ\,\text{RMS}$ under aggressive off-road cornering and chassis vibration.
- **Heading / Yaw Dynamic Accuracy:** $0.5^\circ\,\text{RMS}$ fusing dual-antenna GNSS and 3-axis magnetometer.
- **RTK Position Accuracy:** $0.01\,\text{m} + 1\,\text{ppm}$ horizontal CEP in RTK Fixed mode; $0.40\,\text{m}$ in RTK Float mode; $1.5\,\text{m}$ in autonomous SBAS mode.
- **Protocol & Interfaces:** XBUS binary framing transmitted over RS-232 / USB Serial at $115,200\,\text{baud}$ or SocketCAN at $500\,\text{kbps}$.
- **Mounting Position:** Rigidly affixed near vehicle center of gravity ($l_r = 0.85\,\text{m}$ behind front axle, $H_{\text{imu}} = 0.45\,\text{m}$ above ground).

---

## 10. Actuation Bounds, Slew-Rate Limiting & Safety Transfer Dynamics

The vehicle actuation controller (`lanekeep_controllers.py`) enforces strict profile-specific saturation boundaries to eliminate actuator wind-up, prevent mechanical tie-rod bind, and suppress high-frequency resonant chassis oscillation:

```
+----------------------------------------------------------------------------------------------------+
| PARAMETER                       | BAJA BUGGY        | CAMPUS SHUTTLE    | AGV WAREHOUSE     | PASSENGER CAR     |
+---------------------------------+-------------------+-------------------+-------------------+-------------------+
| Max Left Steering Angle (deg)   | -28.0°            | -32.0°            | -45.0°            | -35.0°            |
| Max Right Steering Angle (deg)  | +28.0°            | +32.0°            | +45.0°            | +35.0°            |
| Max Steering Slew Rate (deg/s)  | 40.0°/s           | 25.0°/s           | 60.0°/s           | 120.0°/s          |
| Deadband Compensation (deg)     | ± 0.75°           | ± 0.50°           | ± 0.20°           | ± 0.25°           |
| Lookahead Distance Min (m)      | 2.50 m            | 3.00 m            | 1.00 m            | 4.00 m            |
| Lookahead Distance Max (m)      | 15.00 m           | 20.00 m           | 6.00 m            | 40.00 m           |
| Speed-Scheduled Gain k_speed    | 0.55 s            | 0.70 s            | 0.40 s            | 0.85 s            |
| Max Permissible Lateral Error   | 0.60 m            | 0.40 m            | 0.15 m            | 0.35 m            |
| Safe-Stop Deceleration (m/s²)   | -4.5 m/s²         | -2.5 m/s²         | -2.0 m/s²         | -3.5 m/s²         |
| E-Stop Deceleration (m/s²)      | -7.0 m/s²         | -5.5 m/s²         | -4.0 m/s²         | -9.0 m/s²         |
+----------------------------------------------------------------------------------------------------+
```

### 10.1 Mathematical Actuation Clamping Formulations
At each discrete time step $k$ with sampling interval $\Delta t = 0.0333\,\text{s}$:
1. **Slew Rate Saturation:**
   $$\Delta \delta_{\text{raw}} = \delta_{\text{cmd}}[k] - \delta_{\text{actual}}[k-1]$$
   $$\Delta \delta_{\text{clamped}} = \text{clip}\left(\Delta \delta_{\text{raw}}, -\dot{\delta}_{\max} \cdot \Delta t, +\dot{\delta}_{\max} \cdot \Delta t\right)$$
   $$\delta_{\text{rate\_limited}}[k] = \delta_{\text{actual}}[k-1] + \Delta \delta_{\text{clamped}}$$
2. **Absolute Angle Saturation:**
   $$\delta_{\text{final}}[k] = \text{clip}\left(\delta_{\text{rate\_limited}}[k], -\delta_{\max}, +\delta_{\max}\right)$$
3. **Anti-Windup Integral Clamping (PID Controller):**
   $$I_{\text{accum}}[k] = \text{clip}\left(I_{\text{accum}}[k-1] + e_{\text{lat}}[k] \cdot \Delta t, -\frac{I_{\max}}{K_i}, +\frac{I_{\max}}{K_i}\right)$$

---

## 11. Profile-by-Profile Environmental Limits Comparison Table

| Environmental Condition | BAJA SAE Buggy (Profile 1) | Campus Shuttle (Profile 2) | Warehouse AGV (Profile 3) | Passenger EV (Profile 4) |
|:---|:---|:---|:---|:---|
| **Primary Ground Texture** | Dirt, gravel, clay, shallow mud | Smooth asphalt, concrete pavers | Polished epoxy concrete floor | Bituminous asphalt, highway concrete |
| **Pavement Markings** | None (unmarked trail boundaries) | Crosswalks, curbs, bike lanes | High-contrast yellow/white tapes | MUTCD dashed/solid white/yellow |
| **Max Wind Speed** | $60\,\text{km/h}$ ($16.7\,\text{m/s}$) | $45\,\text{km/h}$ ($12.5\,\text{m/s}$) | Indoor ($0\,\text{km/h}$ draft) | $80\,\text{km/h}$ ($22.2\,\text{m/s}$) |
| **Max Standing Water** | $150\,\text{mm}$ (shallow splash) | $25\,\text{mm}$ (wet pavement) | Zero (indoor dry floor) | $20\,\text{mm}$ (hydroplane limit) |
| **Permitted Dust / Spray** | High (trail dust plumes, ruts) | Low (urban ambient air) | Cleanroom / warehouse grade | Low to moderate spray |
| **Direct Sun Glare Policy** | CLAHE + Polarizer active | CLAHE tone mapping | N/A (artificial interior LED) | Sun visor + CLAHE active |
| **Max Track Cross-Slope** | $12.0^\circ$ ($21.3\%$) | $4.0^\circ$ ($7.0\%$) | $1.0^\circ$ ($1.75\%$) | $6.0^\circ$ ($10.5\%$) |
| **Max Longitudinal Slope**| $20.0^\circ$ ($36.4\%$) | $8.0^\circ$ ($14.0\%$) | $3.0^\circ$ ($5.2\%$) | $12.0^\circ$ ($21.3\%$) |
| **Ambient Temp Range** | $-5^\circ\text{C}$ to $+45^\circ\text{C}$ | $0^\circ\text{C}$ to $+40^\circ\text{C}$ | $+5^\circ\text{C}$ to $+35^\circ\text{C}$ | $-20^\circ\text{C}$ to $+50^\circ\text{C}$ |

---

## 12. ODD Boundary Exit & Takeover Transition Protocol

Whenever the vehicle approaches the boundary of its Operational Design Domain, the system executes an alert-and-transition sequence aligned with ISO 26262 ASIL-B safety requirements:

```
[TIME t = 0.0s : ODD EXCURSION DETECTED]
   │  Condition: Camera blur exceeds threshold OR track slope > 12° OR GNSS lost > 3.0s
   ▼
[TIME t = 0.05s : AUDITORY & VISUAL TAKEOVER REQUEST (TOR)]
   │  Dashboard HUD displays flashing red banner: "ODD BOUNDARY EXCEEDED — TAKE OVER IMMEDIATELY"
   │  Audible buzzer sounds on vehicle cabin CAN node; primary controller clamps velocity to 50%
   ▼
[TIME t = 0.05s to 2.5s : DRIVER SUPERVISION WINDOW]
   │  If human driver intervenes (manual steering torque > 3.5 Nm OR brake pedal depression > 15%):
   │     --> System immediately relinquishes control to MANUAL_OVERRIDE (Transition time < 15 ms)
   ▼
[TIME t = 2.5s : NO HUMAN TAKEOVER DETECTED]
   │  Autonomous stack initiates Minimal Risk Maneuver (MRM)
   ▼
[TIME t = 2.5s to 4.5s : CONTROLLED BRUSH DECELERATION]
   │  Vehicle applies controlled service braking (-2.5 m/s²), straightens steering rack (0.0°),
   │  and pulses hazard warning strobes over CAN ID 0x190
   ▼
[TIME t = 4.5s : SAFE STOP COMPLETE]
      Vehicle achieves full standstill (v = 0.0 m/s); transmission placed in Park / Neutral;
      Microcontroller watchdog drops actuation relay, engaging mechanical fail-safe spring brakes.
```

---

## 14. Formal Hazard Mitigations & ASIL-B Safety Goals Traceability

To fulfill ISO 26262-3:2018 requirements, the operational boundaries established in this ODD are directly allocated to top-level Safety Goals (SG-01 to SG-05):

| Safety Goal ID | Description | Target ASIL | Allocated ODD Boundary Constraint | Verification Test Link |
|:---|:---|:---:|:---|:---|
| **SG-01** | Prevent unintended departure from drivable path | **ASIL-B** | Max lateral offset $e_{\text{lat}} \le 0.60\,\text{m}$ (BAJA) / $0.35\,\text{m}$ (Car); steering rate $\le 40^\circ/\text{s}$ | `tests/test_controllers.py` |
| **SG-02** | Guarantee emergency braking before obstacle impact | **ASIL-B** | Mobileye RSS safety headway $d_{\min}$; Radar track range $0.5 - 120\,\text{m}$ | `tests/test_planner_rss.py` |
| **SG-03** | Maintain continuous lane tracking across dropouts | **ASIL-A** | ESKF dead-reckoning hold $\le 5.0\,\text{s}$ with drift $\le 0.35\,\text{m}$ | `tests/test_eskf_localization.py` |
| **SG-04** | Command safe state upon watchdog timeout | **ASIL-B** | Max transition time to mechanical safe stop $\le 500\,\text{ms}$ | `deploy/fault_injector.py` |
| **SG-05** | Reject corrupted OTA updates prior to execution | **ASIL-B** | Ed25519 digital signature validation before pointer swap | `tests/test_ota_harvester.py` |

### 14.1 Longitudinal Safety Headway Formulation (Mobileye RSS)
Longitudinal safety spacing within this ODD is enforced dynamically via the RSS safety inequality:
$$d_{\min}(v_{\text{ego}}, v_{\text{front}}) = \left[ v_{\text{ego}} \cdot \rho + \frac{1}{2} a_{\max, \text{accel}} \cdot \rho^2 + \frac{(v_{\text{ego}} + \rho \cdot a_{\max, \text{accel}})^2}{2 \cdot a_{\min, \text{brake}}} - \frac{v_{\text{front}}^2}{2 \cdot a_{\max, \text{front\_brake}}} \right]_+$$
Where the profile-specific dynamic constants are defined as:
- Response Time ($\rho$): $0.50\,\text{s}$ (Autonomous compute + CAN actuation latency).
- Maximum Acceleration ($a_{\max, \text{accel}}$): $3.5\,\text{m/s}^2$ (BAJA Buggy) / $2.5\,\text{m/s}^2$ (Shuttle).
- Minimum Braking Deceleration ($a_{\min, \text{brake}}$): $4.5\,\text{m/s}^2$ (BAJA Buggy) / $3.5\,\text{m/s}^2$ (Car).
- Assumed Leading Vehicle Braking ($a_{\max, \text{front\_brake}}$): $7.0\,\text{m/s}^2$ (Conservative maximum).

---

## 15. Dynamic Lateral Headway & Boundary Margin Formulations

In addition to longitudinal headway, the vehicle enforces profile-specific lateral clearance boundaries:
$$d_{\text{lat, min}} = c_{\text{ego}} + c_{\text{obstacle}} + \mu_{\text{lat}}$$
Where:
- $c_{\text{ego}} = \frac{W_{\text{track}}}{2}$ is half vehicle track width ($0.70\,\text{m}$ for BAJA Buggy).
- $c_{\text{obstacle}}$ is the half-width of the detected 3D bounding box from `box3d_projector.py`.
- $\mu_{\text{lat}}$ is the dynamic lateral safety buffer scheduled with vehicle velocity:
  $$\mu_{\text{lat}}(v) = \max\left(0.30\,\text{m}, 0.05 \cdot v\right)$$
If the predicted trajectory approaches within $d_{\text{lat, min}}$ of any detected obstacle or road boundary berm, the local path planner rejects the candidate trajectory and commands evasive swerving or emergency deceleration.

---

## 17. ODD Testing Scenarios & Edge-Case Validation Matrix

The following operational scenarios from the 101 Golden Scenario library validate the boundaries of the ODD:

| Scenario ID | Test Name | Operational Profile | Boundary Condition Tested | Expected Autonomous Behavior |
|:---|:---|:---:|:---|:---|
| **SCN-01** | Midday High-Angle Dust Burst | BAJA Buggy | 3-second camera occlusion ($>25\%$ mud) | ESKF dead-reckoning hold engaged; lateral drift $<0.35\,\text{m}$ |
| **SCN-02** | Extreme Rutting Tie-Rod Impact | BAJA Buggy | Front wheel shock perturbation ($15^\circ$ angle kick) | Stanley saturation clipping clamps command to $\pm 28.0^\circ$ |
| **SCN-03** | Steep Crest Blind Drop | BAJA Buggy | Dynamic pitch $+7.5^\circ$ over crest | IPM dynamically adjusts horizon row; avoids false phantom lines |
| **SCN-04** | Dusk Glare on Interlocking Pavers | Campus Shuttle | Direct sun glare ($< 10^\circ$ elevation angle) | CLAHE adaptive tone mapping activates; retains path tracking |
| **SCN-05** | Pedestrian Crosswalk Cut-In | Campus Shuttle | Pedestrian crosses within $2.0\,\text{m}$ corridor | TTI calculation triggers emergency braking at $-5.5\,\text{m/s}^2$ |
| **SCN-06** | Narrow Racking Corridors | Warehouse AGV | Aisle lateral clearance down to $\pm 0.25\,\text{m}$ | Lateral offset error maintained within $\pm 0.08\,\text{m}$ |
| **SCN-07** | High-Speed Highway Hairpin | Passenger Car | Road curvature radius $R = 180\,\text{m}$ at $80\,\text{km/h}$ | Stanley controller tracks centerline with error $\le 0.15\,\text{m}$ |
| **SCN-08** | CAN Bus Electrical Fault Storm | All Profiles | 50 consecutive malformed CAN frames | Watchdog safe-stop engages mechanical spring brakes in $< 56.2\,\text{ms}$ |
| **SCN-09** | Thermal Throttling Load | All Profiles | Host edge GPU core temperature $> 85^\circ\text{C}$ | Automatic graceful cooldown mode engaged; speed clamped to 50% |
| **SCN-10** | Loss of GNSS RTK Lock | All Profiles | Transition from RTK Fixed to GPS Lost in canyon | ESKF shifts weight to visual odometry + wheel tick integration |

---

## 18. Final Engineering Sign-Off & Specification Approval

* **Process Standard:** ISO 26262:2018 ASIL-B Aligned Engineering Process
* **Autonomy Level:** Driver-Supervised SAE Level 2
* **Baseline Status:** SEALED AND FROZEN FOR PHASE 7 SECTION 1
* **Audit Key:** `ENG-AV-ODD-ASILB-20261017`
* **Repository Authority:** `adityaBuilds96/APEX-LKA` (Branch `main`)

---

## 19. Document Modification History & Safety Approvals

| Revision | Release Date | Author / Engineering Role | Primary Scope of Revision | Approval Status |
|:---:|:---:|:---|:---|:---:|
| **v1.0** | 2026-09-15 | Lead Autonomous Architect | Initial draft for Phase 3 prototype | Obsolete |
| **v2.0** | 2026-10-03 | Systems Safety Engineer | Expanded for Phase 4 HIL & Field Validation | Superseded |
| **v3.0** | 2026-10-11 | Autonomous Platform Lead | Multi-profile expansion (Baja, Shuttle, AGV, Car) | Superseded |
| **v4.0** | 2026-10-17 | Principal AV Systems Engineer | Complete Phase 7 truth pass and ASIL-B alignment | **APPROVED & FROZEN** |

*This document is under strict Git configuration management. Unauthorized modification voids functional safety verification.*
*Document Verification ID: `ISO-26262-ODD-V4-CERT-SEAL`*
*All rights reserved by APEX Autonomous Vehicle Systems Engineering.*





