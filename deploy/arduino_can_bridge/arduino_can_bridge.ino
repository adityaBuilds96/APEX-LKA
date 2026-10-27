/*
 * APEX-LKA v4.0 — Arduino Mega 2560 CAN Bridge Firmware
 * =======================================================
 * SPI MCP2515 CAN controller, 16 MHz oscillator, 500 kbps CAN bus.
 * Bidirectional USB Serial bridge (921600 baud) with COBS framing.
 *
 * Features:
 *   - 500 kbps CAN bus via SPI MCP2515 (CS=PIN 10, INT=PIN 2)
 *   - 500 ms hardware watchdog via avr/wdt.h
 *   - COBS-encoded serial packets: [0x00][CAN_ID_H][CAN_ID_L][LEN][DATA...][CRC8][COUNTER][0x00]
 *   - Fail-Safe Output: Pin 13 relay interlock → LOW if no valid Jetson
 *     steering payload (0x100) within 200 ms
 *   - CRC-8 SAE J1850 (poly=0x1D) on all frames
 *   - 4-bit rolling sequence counter per CAN ID
 *   - Bidirectional: CAN→Serial and Serial→CAN forwarding
 *   - Primary mode: Real CAN bus
 *   - Pass-through mode: relay all CAN frames to Serial for host
 *
 * Hardware Connections (Arduino Mega 2560):
 *   MCP2515 CS   → Pin 10 (SPI SS)
 *   MCP2515 INT  → Pin 2  (INT0)
 *   MCP2515 SCK  → Pin 52 (SPI SCK)
 *   MCP2515 SI   → Pin 51 (SPI MOSI)
 *   MCP2515 SO   → Pin 50 (SPI MISO)
 *   Relay/LED    → Pin 13 (Safety interlock output)
 *   Status LED   → Pin 12
 *
 * Dependencies:
 *   - mcp_can library (Cory Fowler): github.com/coryjfowler/MCP_CAN_lib
 *   - avr/wdt.h (built-in AVR toolchain)
 */

#include <SPI.h>
#include <mcp_can.h>
#include <avr/wdt.h>

// ============================================================
// Configuration Constants
// ============================================================

#define CAN_CS_PIN          10
#define CAN_INT_PIN          2
#define RELAY_PIN           13      // Safety interlock — HIGH=safe, LOW=failsafe
#define STATUS_LED_PIN      12

#define SERIAL_BAUD_RATE    921600UL
#define CAN_BIT_RATE        CAN_500KBPS
#define MCP2515_CLOCK       MCP_16MHz

#define STEERING_CAN_ID     0x100   // Jetson → Arduino steering command
#define HEARTBEAT_CAN_ID    0x050   // Jetson → All nodes heartbeat
#define RISK_CAN_ID         0x150   // Jetson → Risk priority

#define STEER_TIMEOUT_MS    200UL   // Relay opens if no steering frame within 200 ms
#define WATCHDOG_TIMEOUT    WDTO_500MS  // 500 ms hardware watchdog

#define CRC8_POLY           0x1D    // SAE J1850
#define CRC8_INIT           0xFF
#define CRC8_FINAL_XOR      0xFF
#define COUNTER_MODULUS     16      // 4-bit counter
#define MAX_DLC             8
#define SERIAL_BUF_SIZE     128

// COBS framing constants
#define COBS_FRAME_START    0x00
#define COBS_FRAME_END      0x00
#define COBS_OVERHEAD       5       // [START][ID_H][ID_L][LEN] ... [CRC8][CTR][END]

// ============================================================
// Global Objects
// ============================================================

MCP_CAN CAN(CAN_CS_PIN);

// CRC-8 lookup table (SAE J1850, poly=0x1D)
static uint8_t crc8_table[256];
static bool crc8_table_ready = false;

// State
static unsigned long last_steer_rx_ms = 0;
static bool relay_active = true;  // true = relay closed (safe, output enabled)
static uint8_t tx_counter[256] = {0};  // Per-CAN-ID TX sequence counters

// Serial RX buffer for host → CAN parsing
static uint8_t serial_rx_buf[SERIAL_BUF_SIZE];
static uint8_t serial_rx_idx = 0;
static bool serial_rx_in_frame = false;

// Statistics
static uint32_t can_rx_count = 0;
static uint32_t can_tx_count = 0;
static uint32_t serial_rx_count = 0;
static uint32_t crc_error_count = 0;
static uint32_t relay_trip_count = 0;

// ============================================================
// CRC-8 SAE J1850 (poly=0x1D)
// ============================================================

static void build_crc8_table() {
    for (int i = 0; i < 256; i++) {
        uint8_t crc = (uint8_t)i;
        for (int j = 0; j < 8; j++) {
            if (crc & 0x80) {
                crc = (uint8_t)((crc << 1) ^ CRC8_POLY);
            } else {
                crc = (uint8_t)(crc << 1);
            }
        }
        crc8_table[i] = crc;
    }
    crc8_table_ready = true;
}

static uint8_t compute_crc8(const uint8_t* data, uint8_t len) {
    uint8_t crc = CRC8_INIT;
    for (uint8_t i = 0; i < len; i++) {
        crc = crc8_table[crc ^ data[i]];
    }
    return crc ^ CRC8_FINAL_XOR;
}

// Verify CRC in byte 6 of an 8-byte CAN frame
static bool verify_frame_crc(const uint8_t* frame) {
    uint8_t expected = compute_crc8(frame, 6);
    return expected == frame[6];
}

// Inject CRC into byte 6 of an 8-byte CAN frame
static void inject_frame_crc(uint8_t* frame) {
    frame[6] = compute_crc8(frame, 6);
}

// ============================================================
// COBS Encoding/Decoding
// ============================================================
// COBS: Consistent Overhead Byte Stuffing
// Replaces 0x00 bytes in data with length-to-next-zero pointers.
// Frame format:
//   [0x00] [COBS_BLOCK...] [0x00]
// Each COBS block: [overhead_byte][data_without_zeros...]

// Encode src_len bytes from src into dst.
// Returns encoded length (includes overhead byte but not terminators).
static int cobs_encode(const uint8_t* src, int src_len, uint8_t* dst) {
    int write_idx = 0;
    int overhead_idx = write_idx++;  // Reserve space for overhead byte
    uint8_t code = 0x01;

    for (int i = 0; i < src_len; i++) {
        if (src[i] == 0x00) {
            dst[overhead_idx] = code;
            overhead_idx = write_idx++;
            code = 0x01;
        } else {
            dst[write_idx++] = src[i];
            code++;
            if (code == 0xFF) {
                dst[overhead_idx] = code;
                overhead_idx = write_idx++;
                code = 0x01;
            }
        }
    }
    dst[overhead_idx] = code;
    return write_idx;
}

// Decode COBS block (between two 0x00 delimiters).
// Returns decoded length, or -1 on error.
static int cobs_decode(const uint8_t* src, int src_len, uint8_t* dst) {
    int read_idx = 0;
    int write_idx = 0;

    while (read_idx < src_len) {
        uint8_t code = src[read_idx++];
        if (code == 0x00) return -1;  // Invalid: 0x00 inside COBS block

        for (uint8_t i = 1; i < code; i++) {
            if (read_idx >= src_len) return -1;
            dst[write_idx++] = src[read_idx++];
        }
        if (code < 0xFF && read_idx < src_len) {
            dst[write_idx++] = 0x00;
        }
    }
    return write_idx;
}

// ============================================================
// Serial Packet Builder / Parser
// ============================================================

/*
 * Outgoing packet format (CAN → Serial):
 *   Payload (before COBS): [CAN_ID_H][CAN_ID_L][DLC][DATA[0..7]][CRC8][COUNTER]
 *   Total payload before COBS: 13 bytes
 *   After COBS encode + 0x00 delimiters: ~16 bytes
 *
 * Incoming packet format (Serial → CAN): same structure.
 */

static void send_can_frame_to_serial(uint32_t can_id, uint8_t dlc, uint8_t* data) {
    uint8_t raw[13];  // [ID_H][ID_L][DLC][D0..D7][CRC8][CTR]
    raw[0] = (uint8_t)((can_id >> 8) & 0xFF);
    raw[1] = (uint8_t)(can_id & 0xFF);
    raw[2] = dlc;
    for (uint8_t i = 0; i < 8; i++) {
        raw[3 + i] = (i < dlc) ? data[i] : 0x00;
    }
    // CRC over the raw packet header + data (not including CRC/CTR fields)
    raw[11] = compute_crc8(raw, 11);
    raw[12] = (tx_counter[(uint8_t)(can_id & 0xFF)]++) & 0x0F;

    uint8_t encoded[20];
    int enc_len = cobs_encode(raw, 13, encoded);

    Serial.write(COBS_FRAME_START);
    Serial.write(encoded, enc_len);
    Serial.write(COBS_FRAME_END);
}

// Parse a complete COBS-decoded serial payload → inject as CAN frame
static bool parse_serial_frame_to_can(const uint8_t* decoded, int dec_len) {
    if (dec_len < 13) return false;

    uint32_t can_id = ((uint32_t)decoded[0] << 8) | decoded[1];
    uint8_t  dlc    = decoded[2];
    if (dlc > MAX_DLC) return false;

    uint8_t data[8];
    for (uint8_t i = 0; i < 8; i++) {
        data[i] = decoded[3 + i];
    }

    // Verify packet-level CRC
    uint8_t expected_crc = compute_crc8(decoded, 11);
    if (expected_crc != decoded[11]) {
        crc_error_count++;
        return false;
    }

    // Verify inner CAN frame CRC (byte 6 of data)
    if (dlc == 8 && !verify_frame_crc(data)) {
        crc_error_count++;
        return false;
    }

    // Transmit on CAN bus
    uint8_t ret = CAN.sendMsgBuf(can_id, 0, dlc, data);
    if (ret == CAN_OK) {
        can_tx_count++;

        // Update steering watchdog on valid steering frame
        if (can_id == STEERING_CAN_ID) {
            last_steer_rx_ms = millis();
        }
        return true;
    }
    return false;
}

// ============================================================
// Safety Relay Control
// ============================================================

static void open_relay() {
    if (relay_active) {
        relay_active = false;
        digitalWrite(RELAY_PIN, LOW);
        relay_trip_count++;
        Serial.println("RELAY:OPEN:FAILSAFE");
    }
}

static void close_relay() {
    if (!relay_active) {
        relay_active = true;
        digitalWrite(RELAY_PIN, HIGH);
        Serial.println("RELAY:CLOSED:SAFE");
    }
}

// Check steering watchdog — open relay if no steering command within STEER_TIMEOUT_MS
static void check_steering_watchdog() {
    unsigned long now = millis();
    if (now - last_steer_rx_ms > STEER_TIMEOUT_MS) {
        open_relay();
    } else {
        close_relay();
    }
}

// ============================================================
// CAN RX Interrupt Handler
// ============================================================

static volatile bool can_frame_ready = false;

static void can_isr() {
    can_frame_ready = true;
}

// ============================================================
// Status Heartbeat Builder
// ============================================================
// Periodically send a local status frame on CAN bus for diagnostic purposes.

static void send_local_status_heartbeat() {
    uint8_t data[8] = {0};
    // Byte 0: Node ID (0x02 = Arduino Primary)
    data[0] = 0x02;
    // Byte 1: Relay state (0=open, 1=closed)
    data[1] = relay_active ? 1 : 0;
    // Bytes 2-3: CAN RX count (low 16 bits)
    data[2] = (uint8_t)(can_rx_count & 0xFF);
    data[3] = (uint8_t)((can_rx_count >> 8) & 0xFF);
    // Bytes 4-5: CRC error count
    data[4] = (uint8_t)(crc_error_count & 0xFF);
    data[5] = (uint8_t)((crc_error_count >> 8) & 0xFF);
    // Byte 6: CRC (injected below)
    // Byte 7: Counter (lower nibble)
    static uint8_t hb_ctr = 0;
    data[7] = (hb_ctr++) & 0x0F;
    inject_frame_crc(data);

    CAN.sendMsgBuf(0x050, 0, 8, data);
}

// ============================================================
// Arduino setup()
// ============================================================

void setup() {
    // Initialize hardware watchdog first (500 ms)
    wdt_enable(WATCHDOG_TIMEOUT);

    // Pin modes
    pinMode(RELAY_PIN, OUTPUT);
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW);   // Start with relay OPEN (safe state)
    digitalWrite(STATUS_LED_PIN, LOW);

    // Serial bridge
    Serial.begin(SERIAL_BAUD_RATE);
    Serial.println("APEX-LKA v4.0 Arduino CAN Bridge starting...");

    // CRC-8 table initialization
    build_crc8_table();
    Serial.println("CRC-8 SAE J1850 table built.");

    // MCP2515 initialization
    int retries = 0;
    while (CAN.begin(MCP_ANY, CAN_BIT_RATE, MCP2515_CLOCK) != CAN_OK) {
        Serial.print("MCP2515 init failed, retry ");
        Serial.println(retries++);
        delay(200);
        wdt_reset();
        if (retries > 10) {
            Serial.println("FATAL: MCP2515 init failed after 10 retries.");
            // Remain in failsafe loop with watchdog reset
            while (true) {
                wdt_reset();
                delay(100);
            }
        }
    }

    CAN.setMode(MCP_NORMAL);

    // Configure CAN RX interrupt
    attachInterrupt(digitalPinToInterrupt(CAN_INT_PIN), can_isr, FALLING);

    // Initialize steering watchdog timestamp (give 2 s grace period on startup)
    last_steer_rx_ms = millis() + 2000UL;

    Serial.println("MCP2515 initialized: 500 kbps, INT on pin 2.");
    Serial.println("APEX-LKA CAN Bridge READY.");
    digitalWrite(STATUS_LED_PIN, HIGH);

    wdt_reset();
}

// ============================================================
// Arduino loop()
// ============================================================

static unsigned long last_hb_ms = 0;
static unsigned long last_status_ms = 0;
static const unsigned long HB_INTERVAL_MS = 100;     // Send local HB at 10 Hz
static const unsigned long STATUS_INTERVAL_MS = 5000; // Print status every 5 s

void loop() {
    // Always pet the watchdog first
    wdt_reset();

    unsigned long now = millis();

    // --- 1. Process incoming CAN frames ---
    if (can_frame_ready || !digitalRead(CAN_INT_PIN)) {
        can_frame_ready = false;

        while (CAN.checkReceive() == CAN_MSGAVAIL) {
            uint32_t rx_id = 0;
            uint8_t  rx_dlc = 0;
            uint8_t  rx_data[8] = {0};

            if (CAN.readMsgBuf(&rx_id, &rx_dlc, rx_data) == CAN_OK) {
                can_rx_count++;
                wdt_reset();

                // Validate inner CAN frame CRC if DLC=8
                bool crc_valid = true;
                if (rx_dlc == 8) {
                    crc_valid = verify_frame_crc(rx_data);
                    if (!crc_valid) {
                        crc_error_count++;
                    }
                }

                // Forward to host serial regardless of CRC result
                // (host can decide to drop bad frames)
                send_can_frame_to_serial(rx_id, rx_dlc, rx_data);

                // Process safety-critical frames directly
                if (rx_id == STEERING_CAN_ID && crc_valid) {
                    last_steer_rx_ms = now;
                }
            }
        }
    }

    // --- 2. Process incoming serial bytes (host → CAN) ---
    while (Serial.available()) {
        uint8_t b = Serial.read();
        wdt_reset();

        if (b == COBS_FRAME_START && !serial_rx_in_frame) {
            // Start of new frame
            serial_rx_in_frame = true;
            serial_rx_idx = 0;
        } else if (b == COBS_FRAME_END && serial_rx_in_frame) {
            // End of frame — decode and process
            serial_rx_in_frame = false;
            if (serial_rx_idx > 0) {
                uint8_t decoded[SERIAL_BUF_SIZE];
                int dec_len = cobs_decode(serial_rx_buf, serial_rx_idx, decoded);
                if (dec_len > 0) {
                    bool ok = parse_serial_frame_to_can(decoded, dec_len);
                    if (ok) {
                        serial_rx_count++;
                        digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));  // Toggle on TX
                    }
                }
            }
            serial_rx_idx = 0;
        } else if (serial_rx_in_frame) {
            // Accumulate frame bytes
            if (serial_rx_idx < SERIAL_BUF_SIZE - 1) {
                serial_rx_buf[serial_rx_idx++] = b;
            } else {
                // Buffer overflow — abort frame
                serial_rx_in_frame = false;
                serial_rx_idx = 0;
            }
        }
    }

    // --- 3. Check steering watchdog ---
    check_steering_watchdog();

    // --- 4. Periodic local heartbeat on CAN ---
    if (now - last_hb_ms >= HB_INTERVAL_MS) {
        last_hb_ms = now;
        send_local_status_heartbeat();
        wdt_reset();
    }

    // --- 5. Periodic status print to serial ---
    if (now - last_status_ms >= STATUS_INTERVAL_MS) {
        last_status_ms = now;
        Serial.print("STATUS: can_rx=");
        Serial.print(can_rx_count);
        Serial.print(" can_tx=");
        Serial.print(can_tx_count);
        Serial.print(" serial_rx=");
        Serial.print(serial_rx_count);
        Serial.print(" crc_err=");
        Serial.print(crc_error_count);
        Serial.print(" relay_trips=");
        Serial.print(relay_trip_count);
        Serial.print(" relay_state=");
        Serial.println(relay_active ? "CLOSED" : "OPEN");
        wdt_reset();
    }

    // --- 6. Emergency check: if relay tripped more than 5 times, blink LED ---
    if (relay_trip_count > 5) {
        // Fast blink as visual alert
        static unsigned long last_blink = 0;
        if (now - last_blink > 100) {
            digitalWrite(STATUS_LED_PIN, !digitalRead(STATUS_LED_PIN));
            last_blink = now;
        }
    }
}

// ============================================================
// End of arduino_can_bridge.ino
// ============================================================
