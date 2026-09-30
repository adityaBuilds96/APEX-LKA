/*
  ==============================================================================
  APEX-LKA: Arduino Mega 2560 Firmware - Node 2 (Throttle, Brake & Sensors)
  Platform: SAE India BAJA 2027 Autonomous Category
  ------------------------------------------------------------------------------
  Hardware Connections:
    - Microcontroller: Arduino Mega 2560 (ATmega2560 @ 16 MHz)
    - CAN Controller: MCP2515 SPI Module (CS: Pin 53, INT: Pin 2, MOSI: 51, MISO: 50, SCK: 52)
    - Wheel Speed Optical/Hall Encoder: Pin 3 (External Interrupt 1)
    - Electronic Throttle DAC/PWM: Pin 6
    - Electronic Brake Actuator PWM: Pin 5 / Direction: Pin 4
    - Brake Hydraulic Line Pressure Sensor: Analog Pin A2
    - Driver Brake Pedal Override Switch: Pin 19 (Hardware Interrupt 4)
  ==============================================================================
*/

#include <SPI.h>
#include <mcp_can.h>

// --- Pin Assignments ---
const int PIN_CAN_CS          = 53;  // Mega Hardware SS
const int PIN_CAN_INT         = 2;   // External Interrupt 0
const int PIN_WHEEL_SPEED_INT = 3;   // External Interrupt 1 (Hall/Optical)
const int PIN_BRAKE_PEDAL_INT = 19;  // Hardware Interrupt 4
const int PIN_THROTTLE_PWM    = 6;   // Throttle control output
const int PIN_BRAKE_PWM       = 5;   // Brake actuator PWM
const int PIN_BRAKE_DIR       = 4;   // Brake actuator direction
const int PIN_PRESSURE_IN     = A2;  // Hydraulic pressure transducer

// --- Communication & Rate Constants ---
const long CAN_BAUDRATE             = CAN_500KBPS;
const byte CAN_CRYSTAL_FREQ         = MCP_16MHZ;
const unsigned long SERIAL_BAUDRATE = 115200;

const unsigned long WATCHDOG_TIMEOUT_MS = 500;  // Cut throttle if no cmd in 500ms
const unsigned long TELEMETRY_RATE_MS   = 33;   // ~30 Hz feedback loop

const float WHEEL_DIAMETER_METERS = 0.584; // 23-inch BAJA tire
const int PULSES_PER_REV          = 20;    // Encoder pulse count per revolution

// --- CAN Identifiers ---
const unsigned long CAN_ID_HEARTBEAT   = 0x050;
const unsigned long CAN_ID_CMD_NODE2   = 0x200;
const unsigned long CAN_ID_FEEDBACK_N2 = 0x301;

// --- State Enums ---
enum VehicleMode {
  MODE_MANUAL = 0,
  MODE_AUTO = 1,
  MODE_EMERGENCY_STOP = 2
};

// --- Fault Bitfield ---
const byte FAULT_NONE             = 0x00;
const byte FAULT_SENSOR_ERR       = 0x01;
const byte FAULT_PRESSURE_LOSS    = 0x02;
const byte FAULT_WATCHDOG_TIMEOUT = 0x04;
const byte FAULT_CAN_BUS_OFF      = 0x08;

// --- Global Variables ---
MCP_CAN CAN0(PIN_CAN_CS);

volatile unsigned long g_encoder_pulses = 0;
volatile bool g_brake_pedal_override   = false;

unsigned long g_last_valid_cmd_ms = 0;
unsigned long g_last_telemetry_ms = 0;
unsigned long g_last_speed_calc_ms = 0;

float g_target_throttle_pct = 0.0;
float g_target_brake_pct    = 0.0;
float g_current_throttle_pct= 0.0;

float g_measured_speed_mps  = 0.0;
float g_measured_pressure_psi = 0.0;

VehicleMode g_active_mode = MODE_AUTO;
byte g_fault_flags        = FAULT_NONE;
unsigned long g_last_seq_ts = 0;

// --- Forward Declarations ---
void processCANInput();
void processSerialInput();
void updateDriveOutputs(float dt);
void calculateWheelSpeed();
void sendTelemetry();
void isrWheelPulse();
void isrBrakeOverride();

// ==============================================================================
// Setup
// ==============================================================================
void setup() {
  Serial.begin(SERIAL_BAUDRATE);
  while (!Serial && millis() < 2000);

  pinMode(PIN_THROTTLE_PWM, OUTPUT);
  pinMode(PIN_BRAKE_PWM, OUTPUT);
  pinMode(PIN_BRAKE_DIR, OUTPUT);

  analogWrite(PIN_THROTTLE_PWM, 0);
  analogWrite(PIN_BRAKE_PWM, 0);
  digitalWrite(PIN_BRAKE_DIR, LOW);

  pinMode(PIN_WHEEL_SPEED_INT, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_WHEEL_SPEED_INT), isrWheelPulse, RISING);

  pinMode(PIN_BRAKE_PEDAL_INT, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(PIN_BRAKE_PEDAL_INT), isrBrakeOverride, CHANGE);

  // Initialize MCP2515 CAN Controller
  if (CAN0.begin(MCP_ANY, CAN_BAUDRATE, CAN_CRYSTAL_FREQ) == CAN_OK) {
    CAN0.setMode(MCP_NORMAL);
    pinMode(PIN_CAN_INT, INPUT);
    Serial.println(F("{\"node\":2,\"status\":\"CAN_OK\",\"baud\":500000}"));
  } else {
    g_fault_flags |= FAULT_CAN_BUS_OFF;
    Serial.println(F("{\"node\":2,\"status\":\"CAN_FAIL\",\"msg\":\"Check MCP2515 wiring\"}"));
  }

  g_last_valid_cmd_ms = millis();
  g_last_speed_calc_ms = millis();
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

  // 1. Process Communication
  processCANInput();
  processSerialInput();

  // 2. Read Sensors (Pressure & Speed)
  int raw_press = analogRead(PIN_PRESSURE_IN);
  g_measured_pressure_psi = (raw_press / 1023.0) * 1500.0; // 0-1500 psi sensor

  if (now - g_last_speed_calc_ms >= 50) { // Calculate speed every 50ms
    calculateWheelSpeed();
    g_last_speed_calc_ms = now;
  }

  // 3. Safety Watchdog Check
  if (now - g_last_valid_cmd_ms > WATCHDOG_TIMEOUT_MS) {
    g_fault_flags |= FAULT_WATCHDOG_TIMEOUT;
    g_target_throttle_pct = 0.0; // Cut throttle immediately
    g_target_brake_pct    = 40.0; // Apply safety braking
  } else {
    g_fault_flags &= ~FAULT_WATCHDOG_TIMEOUT;
  }

  // 4. Update Drive Outputs
  updateDriveOutputs(dt);

  // 5. Periodic Telemetry Transmission (30 Hz)
  if (now - g_last_telemetry_ms >= TELEMETRY_RATE_MS) {
    g_last_telemetry_ms = now;
    sendTelemetry();
  }
}

// ==============================================================================
// Throttle & Brake Control
// ==============================================================================
void updateDriveOutputs(float dt) {
  if (g_brake_pedal_override || g_active_mode == MODE_MANUAL) {
    analogWrite(PIN_THROTTLE_PWM, 0);
    analogWrite(PIN_BRAKE_PWM, 0);
    return;
  }

  // Smooth throttle ramp (rate-limited to 50% per second)
  float max_step = 50.0 * dt;
  float err = g_target_throttle_pct - g_current_throttle_pct;
  if (err > max_step) g_current_throttle_pct += max_step;
  else if (err < -max_step) g_current_throttle_pct -= max_step;
  else g_current_throttle_pct = g_target_throttle_pct;

  int throt_pwm = (int)constrain((g_current_throttle_pct / 100.0) * 255.0, 0, 255);
  analogWrite(PIN_THROTTLE_PWM, throt_pwm);

  // Brake output
  int brake_pwm = (int)constrain((g_target_brake_pct / 100.0) * 255.0, 0, 255);
  digitalWrite(PIN_BRAKE_DIR, brake_pwm > 0 ? HIGH : LOW);
  analogWrite(PIN_BRAKE_PWM, brake_pwm);
}

void calculateWheelSpeed() {
  noInterrupts();
  unsigned long pulses = g_encoder_pulses;
  g_encoder_pulses = 0;
  interrupts();

  float revs_per_sec = (float)pulses / (float)PULSES_PER_REV / 0.050;
  g_measured_speed_mps = revs_per_sec * (3.14159 * WHEEL_DIAMETER_METERS);
}

// ==============================================================================
// Communication Handlers
// ==============================================================================
void processCANInput() {
  if (digitalRead(PIN_CAN_INT) == LOW) {
    long unsigned int rx_id;
    unsigned char len = 0;
    unsigned char rx_buf[8];

    CAN0.readMsgBuf(&rx_id, &len, rx_buf);

    if (rx_id == CAN_ID_CMD_NODE2 && len >= 8) {
      int16_t throt_s = (int16_t)(rx_buf[0] | (rx_buf[1] << 8));
      int16_t brake_s = (int16_t)(rx_buf[2] | (rx_buf[3] << 8));
      uint8_t mode    = rx_buf[4];
      uint16_t ts     = (uint16_t)(rx_buf[6] | (rx_buf[7] << 8));

      g_target_throttle_pct = constrain(throt_s / 100.0, 0.0, 100.0);
      g_target_brake_pct    = constrain(brake_s / 100.0, 0.0, 100.0);
      g_active_mode         = (VehicleMode)mode;
      g_last_seq_ts         = ts;
      g_last_valid_cmd_ms   = millis();
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

  if (line.indexOf("\"node\":2") != -1 || line.indexOf("\"node\": 2") != -1) {
    int idx_a = line.indexOf("\"cmd_a\":");
    int idx_b = line.indexOf("\"cmd_b\":");

    if (idx_a != -1) {
      g_target_throttle_pct = constrain(line.substring(idx_a + 8).toFloat(), 0.0, 100.0);
    }
    if (idx_b != -1) {
      g_target_brake_pct = constrain(line.substring(idx_b + 8).toFloat(), 0.0, 100.0);
    }
    g_last_valid_cmd_ms = millis();
  }
}

void sendTelemetry() {
  unsigned long now = millis();

  // 1. Send CAN Frame (ID 0x301)
  int16_t speed_s = (int16_t)(g_measured_speed_mps * 100);
  int16_t press_s = (int16_t)(g_measured_pressure_psi * 10);
  uint8_t ovr     = g_brake_pedal_override ? 1 : 0;
  uint8_t flt     = g_fault_flags;
  uint16_t uptime = (uint16_t)(now % 65536);

  byte tx_buf[8];
  tx_buf[0] = (byte)(speed_s & 0xFF);
  tx_buf[1] = (byte)((speed_s >> 8) & 0xFF);
  tx_buf[2] = (byte)(press_s & 0xFF);
  tx_buf[3] = (byte)((press_s >> 8) & 0xFF);
  tx_buf[4] = ovr;
  tx_buf[5] = flt;
  tx_buf[6] = (byte)(uptime & 0xFF);
  tx_buf[7] = (byte)((uptime >> 8) & 0xFF);

  CAN0.sendMsgBuf(CAN_ID_FEEDBACK_N2, 0, 8, tx_buf);

  // 2. Send Serial JSON Acknowledgment
  Serial.print(F("{\"node\":2,\"status\":\"OK\",\"ack_ts\":"));
  Serial.print(g_last_seq_ts);
  Serial.print(F(",\"fb_a\":"));
  Serial.print(g_measured_speed_mps, 2);
  Serial.print(F(",\"fb_b\":"));
  Serial.print(g_measured_pressure_psi, 1);
  Serial.print(F(",\"override\":"));
  Serial.print(ovr);
  Serial.print(F(",\"fault\":"));
  Serial.print(flt);
  Serial.println(F("}"));
}

void isrWheelPulse() {
  g_encoder_pulses++;
}

void isrBrakeOverride() {
  g_brake_pedal_override = (digitalRead(PIN_BRAKE_PEDAL_INT) == LOW);
}
