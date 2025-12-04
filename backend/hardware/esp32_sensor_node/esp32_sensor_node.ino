// esp32_sensor_node.ino
// publish sensor JSON to MQTT broker (topic: farmfederate/sensors/<client_id>)

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>

const char* ssid = "YOUR_SSID";
const char* pass = "YOUR_PASS";

const char* mqtt_server = "192.168.1.100"; // your broker IP
int mqtt_port = 1883;
const char* client_id = "esp32_sensor_01";

WiFiClient espClient;
PubSubClient client(espClient);

#define SOIL_PIN 34 // ADC pin
#define DHT_PIN 4   // if using DHT22 (example)
#include <DHT.h>
#define DHTTYPE DHT22
DHT dht(DHT_PIN, DHTTYPE);

void setup_wifi() {
  delay(10);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect(client_id)) {
      // connected
    } else {
      delay(2000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  setup_wifi();
  client.setServer(mqtt_server, mqtt_port);
  dht.begin();
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();

  float soil_raw = analogRead(SOIL_PIN); // 0-4095
  float soil_pct = map(soil_raw, 4095, 0, 0, 100); // calibrate as needed

  float temp = dht.readTemperature();
  float hum = dht.readHumidity();

  StaticJsonDocument<256> doc;
  doc["client_id"] = client_id;
  doc["soil_moisture"] = soil_pct;
  doc["temp"] = temp;
  doc["humidity"] = hum;

  char buf[256];
  size_t n = serializeJson(doc, buf);
  client.publish("farmfederate/sensors/esp32", buf, n);

  delay(30 * 1000); // every 30s
}
