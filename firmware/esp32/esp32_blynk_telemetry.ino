// firmware/esp32/esp32_poll_telemetry.ino
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
const char* BACKEND_BASE = "http://YOUR_BACKEND_IP:8000"; // e.g. http://192.168.1.100:8000

const int SOIL_PIN = 34;
const int TEMP_PIN = 35;
const int FLOW_PIN = 25;
const int HUM_PIN = 32;
const int RELAY_PIN = 26;

volatile unsigned long flow_pulse_count = 0;
void IRAM_ATTR flow_pulse() { flow_pulse_count++; }

unsigned long last_sent = 0;
unsigned long last_polled = 0;
const unsigned long TELEMETRY_INTERVAL = 30000; // 30s
const unsigned long POLL_INTERVAL = 5000; // 5s (device polls every 5s for quick reaction)

String device_id = "esp32-01";

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  pinMode(FLOW_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), flow_pulse, RISING);

  WiFi.begin(ssid, password);
  Serial.print("Connecting WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println(" connected");
}

void sendTelemetry() {
  int raw_soil = analogRead(SOIL_PIN);
  float soil_percent = map(raw_soil, 0, 4095, 100, 0);
  int raw_temp = analogRead(TEMP_PIN);
  float temp_c = (raw_temp / 4095.0) * 100.0;
  int raw_h = analogRead(HUM_PIN);
  float air_humidity = map(raw_h, 0, 4095, 0, 100);

  noInterrupts();
  unsigned long pulses = flow_pulse_count;
  flow_pulse_count = 0;
  interrupts();
  float flow_rate = pulses * 0.1; // customize for your flow sensor

  StaticJsonDocument<256> doc;
  doc["device_id"] = device_id;
  doc["ts"] = millis();
  doc["soil_moisture"] = soil_percent;
  doc["air_humidity"] = air_humidity;
  doc["temp_c"] = temp_c;
  doc["flow_rate"] = flow_rate;

  String body; serializeJson(doc, body);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    String url = String(BACKEND_BASE) + "/telemetry";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(body);
    if (code > 0) {
      String payload = http.getString();
      Serial.printf("Telemetry posted, code=%d payload=%s\n", code, payload.c_str());
    } else {
      Serial.printf("POST failed, error=%s\n", http.errorToString(code).c_str());
    }
    http.end();
  }
}

void pollForActions() {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(BACKEND_BASE) + "/poll/" + device_id;
  http.begin(url);
  int code = http.GET();
  if (code == 200) {
    String payload = http.getString();
    StaticJsonDocument<256> doc;
    DeserializationError err = deserializeJson(doc, payload);
    if (!err) {
      if (doc.containsKey("action") && !doc["action"].isNull()) {
        JsonObject action = doc["action"].as<JsonObject>();
        const char* pin = action["pin"] | "";
        int value = action["value"] | 0;
        // interpret pin: if "relay" or "V1" -> toggle RELAY_PIN
        if (strcmp(pin, "relay") == 0 || strcmp(pin, "V1") == 0 || strcmp(pin, "relay_1") == 0) {
          if (value == 1) digitalWrite(RELAY_PIN, HIGH);
          else digitalWrite(RELAY_PIN, LOW);
          Serial.printf("Executed action pin=%s value=%d\n", pin, value);
          // ack back to server
          ackAction(true, String("executed pin=") + String(pin));
        } else {
          // unknown pin - send ack failure
          ackAction(false, String("unknown pin:") + String(pin));
        }
      }
    }
  }
  http.end();
}

void ackAction(bool success, String note) {
  if (WiFi.status() != WL_CONNECTED) return;
  StaticJsonDocument<128> doc;
  doc["device_id"] = device_id;
  doc["success"] = success;
  doc["note"] = note;
  String body; serializeJson(doc, body);
  HTTPClient http;
  String url = String(BACKEND_BASE) + "/ack_action";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(body);
  if (code > 0) {
    Serial.printf("Ack posted, code=%d\n", code);
  } else {
    Serial.printf("Ack failed: %s\n", http.errorToString(code).c_str());
  }
  http.end();
}

void loop() {
  unsigned long now = millis();
  if (now - last_sent > TELEMETRY_INTERVAL) {
    sendTelemetry();
    last_sent = now;
  }
  if (now - last_polled > POLL_INTERVAL) {
    pollForActions();
    last_polled = now;
  }
  delay(50);
}
