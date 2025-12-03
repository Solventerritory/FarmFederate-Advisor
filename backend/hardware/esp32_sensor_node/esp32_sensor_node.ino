// esp32_sensor_node.ino
// Reads: DHT22 (temp/hum), Soil moisture (analog), Flow sensor (pulse)
// Publishes: MQTT JSON to topic farm/<DEVICE_ID>/sensors
// HTTP POST: BACKEND_URL + "/api/sensor_upload" (JSON)

#include <WiFi.h>
#include <HTTPClient.h>
#include <PubSubClient.h>
#include "DHT.h"
#include <OneWire.h>
#include <DallasTemperature.h>

/////////////////////////////////////
// === CONFIG - FILL THESE ===
const char* WIFI_SSID     = "YOUR_SSID";
const char* WIFI_PASS     = "YOUR_PASS";

const char* MQTT_SERVER   = "192.168.1.10"; // or broker hostname
const uint16_t MQTT_PORT  = 1883;

const char* BACKEND_URL   = "http://192.168.1.10:8000"; // backend host (no trailing slash)
const char* DEVICE_ID     = "esp32-node-01"; // unique id used in MQTT topics
const char* MQTT_USER     = ""; // optional
const char* MQTT_PASSWD   = ""; // optional
const bool   USE_HTTP_POST = true; // also POST to backend

const uint32_t REPORT_INTERVAL_MS = 30*1000; // 30s
/////////////////////////////////////

// Pins - change to match your wiring
#define DHTPIN 21
#define DHTTYPE DHT22
#define SOIL_PIN 34        // ADC1_6
#define FLOW_PIN 26        // pulse input (attachInterrupt)
#define RELAY_PIN 16       // optional relay control output
#define ONEWIRE_PIN 4      // optional DS18B20 data pin

DHT dht(DHTPIN, DHTTYPE);
OneWire oneWire(ONEWIRE_PIN);
DallasTemperature sensors_ds(&oneWire);

WiFiClient espClient;
PubSubClient mqtt(espClient);

volatile unsigned long flow_pulses = 0;
unsigned long last_report = 0;
unsigned long last_flow_calc = 0;
unsigned long flow_pulses_snapshot = 0;
float flow_lpm = 0.0; // liters per minute (approx)

void IRAM_ATTR flow_isr() {
  flow_pulses++;
}

void connectWiFi() {
  if (WiFi.status() == WL_CONNECTED) return;
  Serial.print("Connecting to WiFi...");
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  uint32_t start = millis();
  while (WiFi.status() != WL_CONNECTED && millis()-start < 20000) {
    delay(500); Serial.print(".");
  }
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println(" connected.");
  } else {
    Serial.println(" failed.");
  }
}

void reconnectMQTT() {
  if (mqtt.connected()) return;
  while (!mqtt.connected()) {
    Serial.print("Connecting to MQTT...");
    if (mqtt.connect(DEVICE_ID, MQTT_USER, MQTT_PASSWD)) {
      Serial.println(" connected.");
      // subscribe for commands
      String topic = String("farm/") + DEVICE_ID + "/commands";
      mqtt.subscribe(topic.c_str());
    } else {
      Serial.print(" fail, rc=");
      Serial.print(mqtt.state());
      Serial.println(" retry in 3s");
      delay(3000);
    }
  }
}

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  // Very simple handler: payload expected as JSON or simple string e.g. "OPEN" "CLOSE"
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.print("MQTT msg on ");
  Serial.print(topic);
  Serial.print(": ");
  Serial.println(msg);
  // Example command: {"relay":1} or "relay_on"
  if (msg.indexOf("relay") >= 0 || msg.indexOf("ON") >= 0) {
    digitalWrite(RELAY_PIN, HIGH);
  } else if (msg.indexOf("OFF") >= 0) {
    digitalWrite(RELAY_PIN, LOW);
  }
}

void setup() {
  Serial.begin(115200);
  delay(100);

  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);

  dht.begin();
  sensors_ds.begin();

  pinMode(FLOW_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_PIN), flow_isr, RISING);

  mqtt.setCallback(mqttCallback);
  mqtt.setServer(MQTT_SERVER, MQTT_PORT);

  connectWiFi();
  reconnectMQTT();
}

String build_json_payload(float temp, float hum, float soil, float lpm, float ds_temp) {
  // Minimal JSON. Add timestamp if you want server time sync.
  char buf[512];
  snprintf(buf, sizeof(buf),
           "{\"device_id\":\"%s\",\"temp\":%.2f,\"hum\":%.2f,\"soil\":%.2f,\"flow_lpm\":%.3f,\"ds_temp\":%.2f}",
           DEVICE_ID, temp, hum, soil, lpm, ds_temp);
  return String(buf);
}

void http_post_sensor(const String& json) {
  if (WiFi.status() != WL_CONNECTED) return;
  HTTPClient http;
  String url = String(BACKEND_URL) + "/api/sensor_upload";
  http.begin(url);
  http.addHeader("Content-Type", "application/json");
  int code = http.POST(json);
  if (code > 0) {
    Serial.printf("HTTP POST %d\n", code);
    String resp = http.getString();
    Serial.println(resp);
  } else {
    Serial.printf("HTTP POST failed: %d\n", code);
  }
  http.end();
}

void loop() {
  connectWiFi();
  mqtt.loop();
  reconnectMQTT();

  unsigned long now = millis();

  // flow calculation every 5s window
  if (now - last_flow_calc >= 5000) {
    noInterrupts();
    unsigned long pulses = flow_pulses;
    interrupts();
    // pulses per second approx
    // typical sensor (e.g., YF-S201) gives 7.5 pulses per liter; adjust as required
    const float PULSES_PER_LITER = 7.5;
    float liters = pulses / PULSES_PER_LITER;
    // But pulses variable is cumulative; compute delta
    static unsigned long prev_pulses = 0;
    unsigned long delta = pulses - prev_pulses;
    prev_pulses = pulses;
    float liters_window = delta / PULSES_PER_LITER;
    flow_lpm = (liters_window) * (60.0 / 5.0); // liters per minute estimate
    last_flow_calc = now;
  }

  if (now - last_report >= REPORT_INTERVAL_MS) {
    float soil_raw = analogRead(SOIL_PIN); // 0-4095
    // convert to percent (calibrate for your sensor)
    float soil_pct = 100.0 - ((soil_raw / 4095.0) * 100.0);

    float hum = dht.readHumidity();
    float temp = dht.readTemperature();

    sensors_ds.requestTemperatures();
    float ds_temp = sensors_ds.getTempCByIndex(0);

    String json = build_json_payload(temp, hum, soil_pct, flow_lpm, ds_temp);

    // Publish MQTT
    String topic = String("farm/") + DEVICE_ID + "/sensors";
    mqtt.publish(topic.c_str(), json.c_str());

    // Optionally also POST to backend
    if (USE_HTTP_POST) {
      http_post_sensor(json);
    }

    Serial.println("Published sensor payload:");
    Serial.println(json);

    last_report = now;
  }

  delay(200);
}
