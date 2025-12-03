// esp32_mqtt_only.ino
#include <WiFi.h>
#include <PubSubClient.h>

const char* WIFI_SSID   = "YOUR_SSID";
const char* WIFI_PASS   = "YOUR_PASS";
const char* MQTT_SERVER = "192.168.1.10";
const uint16_t MQTT_PORT = 1883;
const char* DEVICE_ID = "esp32-relay-01";
const int RELAY_PIN = 16;

WiFiClient espClient;
PubSubClient client(espClient);

void callback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i = 0; i < length; i++) msg += (char)payload[i];
  Serial.print("Got command: "); Serial.println(msg);
  if (msg.indexOf("ON") >= 0 || msg.indexOf("open") >= 0 || msg.indexOf("1") >= 0) {
    digitalWrite(RELAY_PIN, HIGH);
  } else {
    digitalWrite(RELAY_PIN, LOW);
  }
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("WiFi");
  while (WiFi.status() != WL_CONNECTED) { delay(500); Serial.print("."); }
  Serial.println(" connected");
}

void reconnectMQTT() {
  while (!client.connected()) {
    if (client.connect(DEVICE_ID)) {
      String topic = String("farm/") + DEVICE_ID + "/commands";
      client.subscribe(topic.c_str());
    } else {
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  connectWiFi();
  client.setServer(MQTT_SERVER, MQTT_PORT);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) reconnectMQTT();
  client.loop();
}
