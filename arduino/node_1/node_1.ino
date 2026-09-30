/*
  ==============================================================================
  APEX-LKA: Arduino Mega 2560 Firmware - Node 1 (Steering & Actuation)
  Platform: SAE India BAJA 2027 Autonomous Category
  ------------------------------------------------------------------------------
  Hardware Connections:
    - Microcontroller: Arduino Mega 2560 (ATmega2560 @ 16 MHz)
    - CAN Controller: MCP2515 SPI Module (CS: Pin 53, INT: Pin 2, MOSI: 51, MISO: 50, SCK: 52)
    - Manual Override Sensor: Pin 18 (Hardware Interrupt 5)
    - Steering Actuator PWM: Pin 9 / Direction: Pin 8
    - Steering Angle Feedback Encoder: Analog Pin A0 or Optical SPI
    - Motor Current Sense: Analog Pin A1
  ==============================================================================
*/

#include <SPI.h>
#include <mcp_can.h>

// --- Pin Assignments ---
const int PIN_CAN_CS       = 53;  // Mega Hardware SS
const int PIN_CAN_INT      = 2;   // External Interrupt 0
const int PIN_OVERRIDE_SW  = 18;  // Driver physical takeover limit switch
const int PIN_ACTUATOR_PWM = 9;   // Motor Driver PWM output
const int PIN_ACTUATOR_DIR = 8;   // Motor Driver Direction
const int PIN_ENCODER_IN   = A0;  // Steering position feedback potentiometer/encoder
const int PIN_CURRENT_IN   = A1;  // Motor driver current shunt amplifier

// --- Communication & Rate Constants ---
const long CAN_BAUDRATE             = CAN_500KBPS;
const byte CAN_CRYSTAL_FREQ         = MCP_16MHZ; // Standard 16MHz crystal on MCP2515
const unsigned long SERIAL_BAUDRATE = 115200;

const unsigned long WATCHDOG_TIMEOUT_MS = 500;  // Return to safe neutral if no cmd in 500ms
const unsigned long TELEMETRY_RATE_MS   = 33;   // ~30 Hz feedback loop

const float MAX_STEER_RATE_DEG_PER_SEC = 60.0;  // Slew-rate limiter: prevents current spikes
const float MAX_STEER_ANGLE_DEG        = 30.0;

// --- CAN Identifiers ---
const unsigned long CAN_ID_HEARTBEAT   = 0x050;
const unsigned long CAN_ID_CMD_NODE1   = 0x100;
const unsigned long CAN_ID_FEEDBACK_N1 = 0x300;

// --- State Enums ---
enum VehicleMode {
  MODE_MANUAL = 0,
  MODE_AUTO = 1,
  MODE_EMERGENCY_STOP = 2
};

// --- Fault Bitfield ---
const byte FAULT_NONE             = 0x00;
const byte FAULT_SENSOR_ERR       = 0x01;
const byte FAULT_OVERCURRENT      = 0x02;
const byte FAULT_WATCHDOG_TIMEOUT = 0x04;
const byte FAULT_CAN_BUS_OFF      = 0x08;

// --- Global Variables ---
MCP_CAN CAN0(PIN_CAN_CS);

volatile bool g_manual_override = false;
unsigned long g_last_valid_cmd_ms = 0;
unsigned long g_last_telemetry_ms = 0;

float g_target_steering_deg  = 0.0;
float g_current_steering_deg = 0.0;
float g_measured_steering_deg = 0.0;
float g_measured_current_amps = 0.0;

VehicleMode g_active_mode = MODE_AUTO;
byte g_fault_flags        = FAULT_NONE;
unsigned long g_last_seq_ts = 0;

// --- Forward Declarations ---
void processSerialInput();
void processCANInput();
void updateActuation(float dt);
void sendTelemetry();
void isrManualOverride();

// ==============================================================================
// Setup
// ==============================================================================
void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  while (!Serial && millis() < 2000); // Wait up to 2s for USB Serial on boot

  pinMode(PIN_ACTUATOR_PWM, OUTPUT);
  pinMode(PIN_ACTUATOR_DIR, OUTPUT);
  analogWrite(PIN_ACTUATOR_PWM, 0);
  digitalWrite(PIN_ACTUATOR_DIR, LOW);

  pinMode(PIN_OVERRIDE_SW, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_OVERRIDE_SW), isrManualOverride, CHANGE);

  // Initialize MCP2515 CAN Controller
  if (CAN0.begin(MCP_ANY, CAN_BAUDRATE, CAN_CRYSTAL_FREQ) == CAN_OK) {
    CAN0.setMode(MCP_NORMAL);
    pinMode(PIN_CAN_INT, INPUT);
    Serial.println(F("{\"node\":1,\"status\":\"CAN_OK\",\"baud\":500000}"));
  } else {
    g_fault_flags |= FAULT_CAN_BUS_OFF;
    Serial.println(F("{\"node\":1,\"status\":\"CAN_FAIL\",\"msg\":\"Check MCP2515 wiring\"}"));
  }

  g_last_valid_cmd_ms = millis();
}

// ==============================================================================
// Main Loop
// ==============================================================================
void loop() {
  unsigned long now = millis();
  static unsigned long last_loop_time = 0;
  float dt = (now - last_loop_time) / 1000.0;
  if (dt <= 0.0 || dt > 0.1) dt = 0.01;
  last_loop_time = now;

  // 1. Process Incoming Communication
  processCANInput();
  processSerialInput();

  // 2. Read Sensors (Position & Current)
  int raw_enc = analogRead(PIN_ENCODER_IN);
  // Map 0-1023 analog range to -35.0 deg to +35.0 deg
  g_measured_steering_deg = ((raw_enc - 512) / 512.0) * 35.0;

  int raw_curr = analogRead(PIN_CURRENT_IN);
  g_measured_current_amps = (raw_curr / 1023.0) * 20.0; // 0-20A range

  if (g_measured_current_amps > 15.0) {
    g_fault_flags |= FAULT_OVERCURRENT;
  } else {
    g_fault_flags &= ~FAULT_OVERCURRENT;
  }

  // 3. Safety Watchdog Check
  if (now - g_last_valid_cmd_ms > WATCHDOG_TIMEOUT_MS) {
    g_fault_flags |= FAULT_WATCHDOG_TIMEOUT;
    g_target_steering_deg = 0.0; // Fail-safe to neutral center
  } else {
    g_fault_flags &= ~FAULT_WATCHDOG_TIMEOUT;
  }

  // 4. Update Actuator Output with Slew-Rate Limiting
  updateActuation(dt);

  // 5. Periodic Telemetry Transmission (30 Hz)
  if (now - g_last_telemetry_ms >= TELEMETRY_RATE_MS) {
    g_last_telemetry_ms = now;
    sendTelemetry();
  }
}

// ==============================================================================
// Actuator Control with Ramp Interpolation
// ==============================================================================
void updateActuation(float dt) {
  if (g_manual_override || g_active_mode == MODE_MANUAL) {
    // Cut motor power to permit human driver steering
    analogWrite(PIN_ACTUATOR_PWM, 0);
    return;
  }

  // Rate-limit target angle to prevent mechanical shock & current spikes
  float max_step = MAX_STEER_RATE_DEG_PER_SEC * dt;
  float error = g_target_steering_deg - g_current_steering_deg;

  if (error > max_step) {
    g_current_steering_deg += max_step;
  } else if (error < -max_step) {
    g_current_steering_deg -= max_step;
  } else {
    g_current_steering_deg = g_target_steering_deg;
  }

  // Simple closed-loop proportional driving to position
  float pos_err = g_current_steering_deg - g_measured_steering_deg;
  int pwm_val = constrain((int)(abs(pos_err) * 25.0), 0, 255);

  digitalWrite(PIN_ACTUATOR_DIR, pos_err > 0 ? HIGH : LOW);
  analogWrite(PIN_ACTUATOR_PWM, pwm_val);
}

// ==============================================================================
// Communication Handlers
// ==============================================================================
void processCANInput() {
  if (digitalRead(PIN_CAN_INT) == LOW) { // Frame received in MCP2515 buffer
    long unsigned int rx_id;
    unsigned char len = 0;
    unsigned char rx_buf[8];

    CAN0.readMsgBuf(&rx_id, &len, rx_buf);

    if (rx_id == CAN_ID_CMD_NODE1 && len >= 8) {
      // Unpack: <hhBBH
      int16_t steer_scaled = (int16_t)(rx_buf[0] | (rx_buf[1] << 8));
      int16_t trim_scaled  = (int16_t)(rx_buf[2] | (rx_buf[3] << 8));
      uint8_t conf         = rx_buf[4];
      uint8_t mode         = rx_buf[5];
      uint16_t ts          = (uint16_t)(rx_buf[6] | (rx_buf[7] << 8));

      g_target_steering_deg = constrain(steer_scaled / 100.0, -MAX_STEER_ANGLE_DEG, MAX_STEER_ANGLE_DEG);
      g_active_mode = (VehicleMode)mode;
      g_last_seq_ts = ts;
      g_last_valid_cmd_ms = millis();
    }
    else if (rx_id == CAN_ID_HEARTBEAT && len >= 4) {
      if (rx_buf[0] == 0xAA) {
        g_last_valid_cmd_ms = millis();
      }
    }
  }
}

void processSerialInput() {
  if (!Serial.available()) return;

  String line = Serial.readStringUntil('\n');
  line.trim();
  if (line.length() < 10 || line.charAt(0) != '{') return;

  // Simple JSON parsing: look for "cmd_a" and "node"
  if (line.indexOf("\"node\":1") != -1 || line.indexOf("\"node\": 1") != -1) {
    int idx = line.indexOf("\"cmd_a\":");
    if (idx != -1) {
      float steer = line.substring(idx + 8).toFloat();
      g_target_steering_deg = constrain(steer, -MAX_STEER_ANGLE_DEG, MAX_STEER_ANGLE_DEG);
      g_last_valid_cmd_ms = millis();
    }
  }
}

void sendTelemetry() {
  unsigned long now = millis();

  // 1. Send CAN Frame (ID 0x300)
  int16_t steer_s = (int16_t)(g_measured_steering_deg * 100);
  int16_t curr_s  = (int16_t)(g_measured_current_amps * 100);
  uint8_t ovr     = g_manual_override ? 1 : 0;
  uint8_t flt     = g_fault_flags;
  uint16_t uptime = (uint16_t)(now % 65536);

  byte tx_buf[8];
  tx_buf[0] = (byte)(steer_s & 0xFF);
  tx_buf[1] = (byte)((steer_s >> 8) & 0xFF);
  tx_buf[2] = (byte)(curr_s & 0xFF);
  tx_buf[3] = (byte)((curr_s >> 8) & 0xFF);
  tx_buf[4] = ovr;
  tx_buf[5] = flt;
  tx_buf[6] = (byte)(uptime & 0xFF);
  tx_buf[7] = (byte)((uptime >> 8) & 0xFF);

  CAN0.sendMsgBuf(CAN_ID_FEEDBACK_N1, 0, 8, tx_buf);

  // 2. Send Serial JSON Acknowledgment
  Serial.print(F("{\"node\":1,\"status\":\"OK\",\"ack_ts\":"));
  Serial.print(g_last_seq_ts);
  Serial.print(F(",\"fb_a\":"));
  Serial.print(g_measured_steering_deg, 2);
  Serial.print(F(",\"fb_b\":"));
  Serial.print(g_measured_current_amps, 2);
  Serial.print(F(",\"override\":"));
  Serial.print(ovr);
  Serial.print(F(",\"fault\":"));
  Serial.print(flt);
  Serial.println(F("}"));
}

void isrManualOverride() {
  g_manual_override = (digitalRead(PIN_OVERRIDE_SW) == LOW);
}
