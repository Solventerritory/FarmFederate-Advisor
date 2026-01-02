/*
 * ESP32 Complete Sensor Node with DHT22, Flow Meter, Relay, and Solenoid Valve
 * For FarmFederate-Advisor Project
 * 
 * Hardware:
 * - DHT22 Temperature/Humidity sensor
 * - Water flow meter (YF-S201 or similar)
 * - Relay module (for solenoid valve control)
 * - Solenoid valve (connected to relay)
 * - Optional: Soil moisture sensor
 * 
 * Installation:
 * 1. Install libraries: WiFi, PubSubClient, ArduinoJson, DHT sensor library
 * 2. Update WiFi credentials below
 * 3. Update MQTT broker IP (your computer's IP address)
 * 4. Upload to ESP32
 * 5. Open Serial Monitor at 115200 baud to see connection status
 */

#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// ============================================
// CONFIGURATION - CHANGE THESE VALUES!
// ============================================

// WiFi Settings
const char* WIFI_SSID = "YOUR_WIFI_SSID";        // ← Change to your WiFi name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"; // ← Change to your WiFi password

// MQTT Broker Settings
const char* MQTT_SERVER = "192.168.1.100";  // ← Change to YOUR computer's IP address!
const int MQTT_PORT = 1883;
const char* CLIENT_ID = "esp32_field_01";   // ← Unique ID for this device

// Pin Definitions
#define DHT_PIN 4              // DHT22 data pin → GPIO 4
#define RELAY_PIN 5            // Relay control → GPIO 5
#define FLOW_SENSOR_PIN 18     // Flow meter signal → GPIO 18
#define SOIL_MOISTURE_PIN 34   // Soil moisture analog → GPIO 34 (optional)

// Sensor Settings
#define DHTTYPE DHT22
#define FLOW_CALIBRATION 7.5   // Flow sensor calibration factor (pulses per liter)
#define READ_INTERVAL 10000    // Read sensors every 10 seconds

// ============================================
// GLOBAL VARIABLES
// ============================================

DHT dht(DHT_PIN, DHTTYPE);
WiFiClient espClient;
PubSubClient mqttClient(espClient);

// Flow meter variables
volatile int flowPulseCount = 0;
float flowRateLPM = 0.0;
unsigned long flowOldTime = 0;

// ============================================
// INTERRUPT HANDLER FOR FLOW SENSOR
// ============================================

void IRAM_ATTR flowPulseCounter() {
  flowPulseCount++;
}

// ============================================
// WIFI CONNECTION
// ============================================

void setupWiFi() {
  delay(10);
  Serial.println();
  Serial.println("========================================");
  Serial.println("Starting WiFi connection...");
  Serial.print("SSID: ");
  Serial.println(WIFI_SSID);
  
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 30) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  Serial.println();
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("✓ WiFi connected successfully!");
    Serial.print("IP Address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal Strength: ");
    Serial.print(WiFi.RSSI());
    Serial.println(" dBm");
  } else {
    Serial.println("✗ WiFi connection FAILED!");
    Serial.println("Check:");
    Serial.println("  - SSID spelling (case-sensitive)");
    Serial.println("  - Password");
    Serial.println("  - Router distance");
    Serial.println("  - 2.4GHz WiFi (ESP32 doesn't support 5GHz)");
  }
  Serial.println("========================================");
}

// ============================================
// MQTT CONNECTION
// ============================================

void mqttCallback(char* topic, byte* payload, unsigned int length) {
  Serial.print("MQTT Message [");
  Serial.print(topic);
  Serial.print("]: ");
  
  String message = "";
  for (unsigned int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);
  
  // Control relay/solenoid valve
  if (String(topic) == "farmfederate/control/relay") {
    if (message == "ON" || message == "1") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("→ Relay/Solenoid ACTIVATED (Water ON)");
    } else if (message == "OFF" || message == "0") {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("→ Relay/Solenoid DEACTIVATED (Water OFF)");
    }
  }
}

void reconnectMQTT() {
  while (!mqttClient.connected()) {
    Serial.println("----------------------------------------");
    Serial.println("Connecting to MQTT broker...");
    Serial.print("Server: ");
    Serial.print(MQTT_SERVER);
    Serial.print(":");
    Serial.println(MQTT_PORT);
    Serial.print("Client ID: ");
    Serial.println(CLIENT_ID);
    
    if (mqttClient.connect(CLIENT_ID)) {
      Serial.println("✓ MQTT connected successfully!");
      
      // Subscribe to control topic
      String controlTopic = "farmfederate/control/relay";
      if (mqttClient.subscribe(controlTopic.c_str())) {
        Serial.print("✓ Subscribed to: ");
        Serial.println(controlTopic);
      }
      
      Serial.println("Ready to publish sensor data!");
    } else {
      Serial.print("✗ MQTT connection failed, rc=");
      Serial.println(mqttClient.state());
      Serial.println("Error codes:");
      Serial.println("  -4 = Connection timeout");
      Serial.println("  -3 = Connection lost");
      Serial.println("  -2 = Connect failed (wrong IP?)");
      Serial.println("  -1 = Disconnected");
      Serial.println("Retrying in 5 seconds...");
      delay(5000);
    }
  }
}

// ============================================
// SENSOR READING
// ============================================

void readAndPublishSensors() {
  Serial.println();
  Serial.println("========================================");
  Serial.println("Reading Sensors...");
  
  // Read DHT22 Temperature & Humidity
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  
  if (isnan(temperature) || isnan(humidity)) {
    Serial.println("✗ DHT22 read failed!");
    Serial.println("  Check: Wire connections, power, pin number");
    temperature = 0;
    humidity = 0;
  } else {
    Serial.print("✓ Temperature: ");
    Serial.print(temperature);
    Serial.println(" °C");
    Serial.print("✓ Humidity: ");
    Serial.print(humidity);
    Serial.println(" %");
  }
  
  // Read Soil Moisture (if connected)
  int soilRaw = analogRead(SOIL_MOISTURE_PIN);
  float soilMoisture = map(soilRaw, 0, 4095, 0, 100);
  Serial.print("✓ Soil Moisture: ");
  Serial.print(soilMoisture);
  Serial.print(" % (raw: ");
  Serial.print(soilRaw);
  Serial.println(")");
  
  // Calculate Flow Rate
  if (millis() - flowOldTime > 1000) {
    detachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN));
    
    float pulsesPerSecond = flowPulseCount;
    flowRateLPM = (pulsesPerSecond * 60.0) / FLOW_CALIBRATION;
    
    Serial.print("✓ Flow Rate: ");
    Serial.print(flowRateLPM);
    Serial.print(" L/min (pulses: ");
    Serial.print(flowPulseCount);
    Serial.println(")");
    
    flowOldTime = millis();
    flowPulseCount = 0;
    attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  }
  
  // Read Relay State
  bool relayState = digitalRead(RELAY_PIN);
  Serial.print("✓ Relay/Valve: ");
  Serial.println(relayState ? "ON (Water flowing)" : "OFF (Water stopped)");
  
  // Create JSON payload
  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  doc["temperature"] = round(temperature * 10) / 10.0;  // 1 decimal place
  doc["humidity"] = round(humidity * 10) / 10.0;
  doc["soil_moisture"] = round(soilMoisture * 10) / 10.0;
  doc["flow_rate"] = round(flowRateLPM * 10) / 10.0;
  doc["relay_state"] = relayState ? "ON" : "OFF";
  doc["timestamp"] = millis() / 1000;
  doc["wifi_rssi"] = WiFi.RSSI();
  
  char jsonBuffer[512];
  serializeJson(doc, jsonBuffer);
  
  // Publish to MQTT
  String topic = String("farmfederate/sensors/") + CLIENT_ID;
  Serial.println("----------------------------------------");
  Serial.print("Publishing to: ");
  Serial.println(topic);
  Serial.print("Payload: ");
  Serial.println(jsonBuffer);
  
  bool published = mqttClient.publish(topic.c_str(), jsonBuffer, false);
  
  if (published) {
    Serial.println("✓ Data published successfully!");
  } else {
    Serial.println("✗ Publish failed!");
    Serial.println("  Check: MQTT connection, broker running, firewall");
  }
  Serial.println("========================================");
}

// ============================================
// SETUP
// ============================================

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  delay(1000);
  
  Serial.println();
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║  FarmFederate ESP32 Sensor Node      ║");
  Serial.println("║  Complete Hardware Integration        ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println();
  
  // Initialize pins
  Serial.println("Initializing hardware...");
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Start with relay OFF
  Serial.println("✓ Relay initialized (OFF)");
  
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  Serial.println("✓ Flow sensor initialized");
  
  // Initialize DHT sensor
  dht.begin();
  delay(2000);  // DHT needs time to stabilize
  Serial.println("✓ DHT22 sensor initialized");
  
  Serial.println("✓ Soil moisture sensor ready");
  Serial.println();
  
  // Connect to WiFi
  setupWiFi();
  
  // Setup MQTT
  mqttClient.setServer(MQTT_SERVER, MQTT_PORT);
  mqttClient.setCallback(mqttCallback);
  mqttClient.setKeepAlive(60);
  
  Serial.println();
  Serial.println("Setup complete! Starting main loop...");
  Serial.println();
}

// ============================================
// MAIN LOOP
// ============================================

void loop() {
  // Maintain WiFi connection
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected! Reconnecting...");
    setupWiFi();
  }
  
  // Maintain MQTT connection
  if (!mqttClient.connected()) {
    reconnectMQTT();
  }
  mqttClient.loop();
  
  // Read and publish sensors periodically
  static unsigned long lastRead = 0;
  if (millis() - lastRead >= READ_INTERVAL) {
    lastRead = millis();
    readAndPublishSensors();
  }
  
  // Small delay to prevent watchdog issues
  delay(10);
}
