// esp32_mqtt_relay.ino
#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "YOUR_SSID";
const char* pass = "YOUR_PASS";
const char* mqtt_server = "192.168.1.100";
int mqtt_port = 1883;

WiFiClient espClient;
PubSubClient client(espClient);
const int RELAY_PIN = 26;

void callback(char* topic, byte* payload, unsigned int length) {
  String msg;
  for (unsigned int i=0;i<length;i++) msg += (char)payload[i];
  if (String(topic).endsWith("/relay")) {
    if (msg == "ON") digitalWrite(RELAY_PIN, HIGH);
    else digitalWrite(RELAY_PIN, LOW);
  }
}

void reconnect() {
  while (!client.connected()) {
    if (client.connect("esp32_relay")) {
      client.subscribe("farmfederate/control/#");
    } else {
      delay(2000);
    }
  }
}

void setup() {
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED) delay(500);
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
}

void loop() {
  if (!client.connected()) reconnect();
  client.loop();
}
