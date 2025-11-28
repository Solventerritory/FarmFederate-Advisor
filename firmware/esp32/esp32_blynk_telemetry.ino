// firmware/esp32/esp32_blynk_telemetry.ino
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <BlynkSimpleEsp32.h>

const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";
char BLYNK_AUTH[] = "YOUR_BLYNK_TOKEN";
const char* BACKEND_URL = "http://YOUR_BACKEND_IP:8000/telemetry";

const int SOIL_PIN = 34;
const int TEMP_PIN = 35;
const int FLOW_PIN = 25;
const int HUM_PIN = 32;
const int RELAY_PIN = 26;

volatile unsigned long flow_pulse_count = 0;

void IRAM_ATTR flow_pulse() {
  flow_pulse_count++;
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  pinMode(FLOW_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), flow_pulse, RISING);
  WiFi.begin(ssid, password);
  Serial.print("WiFi connecting");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println(" connected");
  Blynk.begin(BLYNK_AUTH, ssid, password);
}

BLYNK_WRITE(V1) {
  int v = param.asInt();
  if (v==1) digitalWrite(RELAY_PIN, HIGH);
  else digitalWrite(RELAY_PIN, LOW);
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
  float flow_rate = pulses * 0.1;

  StaticJsonDocument<256> doc;
  doc["device_id"] = "esp32-01";
  doc["ts"] = String(millis());
  doc["soil_moisture"] = soil_percent;
  doc["air_humidity"] = air_humidity;
  doc["temp_c"] = temp_c;
  doc["flow_rate"] = flow_rate;
  String body; serializeJson(doc, body);

  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(BACKEND_URL);
    http.addHeader("Content-Type", "application/json");
    int code = http.POST(body);
    if (code > 0) {
      String payload = http.getString();
      Serial.printf("Telemetry posted, code=%d payload=%s\n", code, payload.c_str());
    } else Serial.printf("POST failed, error=%s\n", http.errorToString(code).c_str());
    http.end();
  }
}

unsigned long last_sent = 0;
void loop() {
  Blynk.run();
  if (millis() - last_sent > 30000) { sendTelemetry(); last_sent = millis(); }
}
