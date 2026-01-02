/*
 * ESP32 Sensor Node - USB Serial Version
 * For FarmFederate-Advisor Project
 * 
 * This version sends data via USB Serial (no WiFi needed!)
 * 
 * Hardware:
 * - DHT22 Temperature/Humidity sensor
 * - Water flow meter (YF-S201 or similar)
 * - Relay module (for solenoid valve control)
 * - Soil moisture sensor
 * 
 * Data sent as JSON every 5 seconds via USB Serial
 */

#include <ArduinoJson.h>
#include <DHT.h>

// Pin Definitions
#define DHT_PIN 4              // DHT22 data pin → GPIO 4
#define RELAY_PIN 5            // Relay control → GPIO 5
#define FLOW_SENSOR_PIN 18     // Flow meter signal → GPIO 18
#define SOIL_MOISTURE_PIN 34   // Soil moisture analog → GPIO 34

// Sensor Settings
#define DHTTYPE DHT22
#define FLOW_CALIBRATION 7.5   // Flow sensor calibration factor (pulses per liter)
#define READ_INTERVAL 5000     // Read sensors every 5 seconds
const char* CLIENT_ID = "esp32_usb_01";

// Global Variables
DHT dht(DHT_PIN, DHTTYPE);

// Flow meter variables
volatile int flowPulseCount = 0;
float flowRateLPM = 0.0;
float totalLiters = 0.0;
unsigned long flowOldTime = 0;

// Interrupt Handler for Flow Sensor
void IRAM_ATTR flowPulseCounter() {
  flowPulseCount++;
}

void setup() {
  // Initialize Serial at 115200 baud
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("╔════════════════════════════════════════╗");
  Serial.println("║  ESP32 Sensor Node - USB Serial Mode  ║");
  Serial.println("║  FarmFederate-Advisor                  ║");
  Serial.println("╚════════════════════════════════════════╝");
  Serial.println();
  
  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);  // Start with relay OFF
  Serial.println("✓ Relay initialized (OFF)");
  
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  Serial.println("✓ Flow sensor initialized");
  
  // Initialize DHT sensor
  dht.begin();
  delay(2000);
  Serial.println("✓ DHT22 sensor initialized");
  Serial.println("✓ Soil moisture sensor ready");
  Serial.println();
  Serial.println("Ready! Sending sensor data via USB Serial...");
  Serial.println("JSON data will be prefixed with 'DATA:' for easy parsing");
  Serial.println();
}

void loop() {
  static unsigned long lastRead = 0;
  
  // Read and send sensor data periodically
  if (millis() - lastRead >= READ_INTERVAL) {
    lastRead = millis();
    readAndPublishSensors();
  }
  
  // Check for commands from serial port
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    command.trim();
    
    if (command == "RELAY_ON" || command == "ON" || command == "1") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("CMD_ACK: Relay ON");
    } else if (command == "RELAY_OFF" || command == "OFF" || command == "0") {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("CMD_ACK: Relay OFF");
    } else if (command == "STATUS") {
      Serial.println("STATUS: Running");
    } else if (command == "RESET_FLOW") {
      totalLiters = 0;
      Serial.println("CMD_ACK: Flow meter reset");
    }
  }
  
  delay(10);
}

void readAndPublishSensors() {
  // Read DHT22 Temperature & Humidity
  float temperature = dht.readTemperature();
  float humidity = dht.readHumidity();
  
  if (isnan(temperature) || isnan(humidity)) {
    temperature = 0;
    humidity = 0;
  }
  
  // Read Soil Moisture
  int soilRaw = analogRead(SOIL_MOISTURE_PIN);
  float soilMoisture = map(soilRaw, 0, 4095, 0, 100);
  
  // Calculate Flow Rate
  if (millis() - flowOldTime > 1000) {
    detachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN));
    
    float pulsesPerSecond = flowPulseCount;
    flowRateLPM = (pulsesPerSecond * 60.0) / FLOW_CALIBRATION;
    
    // Calculate total liters
    totalLiters += (flowPulseCount / FLOW_CALIBRATION);
    
    flowOldTime = millis();
    flowPulseCount = 0;
    attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  }
  
  // Read Relay State
  bool relayState = digitalRead(RELAY_PIN);
  
  // Create JSON payload
  StaticJsonDocument<512> doc;
  doc["client_id"] = CLIENT_ID;
  doc["temperature"] = round(temperature * 10) / 10.0;
  doc["humidity"] = round(humidity * 10) / 10.0;
  doc["soil_moisture"] = round(soilMoisture * 10) / 10.0;
  doc["flow_rate"] = round(flowRateLPM * 100) / 100.0;
  doc["total_liters"] = round(totalLiters * 100) / 100.0;
  doc["relay_state"] = relayState ? "ON" : "OFF";
  doc["uptime_sec"] = millis() / 1000;
  
  // Send JSON with prefix for easy parsing
  Serial.print("DATA:");
  serializeJson(doc, Serial);
  Serial.println();
}
