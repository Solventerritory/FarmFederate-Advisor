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

#include <Arduino.h>
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

// Objects
DHT dht(DHT_PIN, DHTTYPE);

// Flow meter variables
volatile int flowPulseCount = 0;
float flowRate = 0.0;
float totalLiters = 0.0;
unsigned long oldFlowTime = 0;

// Relay state
bool relayState = false;

// Flow meter interrupt handler
void IRAM_ATTR flowPulseCounter() {
  flowPulseCount++;
}

void setup() {
  // Initialize Serial
  Serial.begin(115200);
  delay(2000);
  
  Serial.println("\n\n═══════════════════════════════════════");
  Serial.println("   ESP32 USB SENSOR NODE");
  Serial.println("   FarmFederate-Advisor");
  Serial.println("═══════════════════════════════════════");
  
  // Initialize sensors
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  
  dht.begin();
  
  Serial.println("\n✓ DHT22 sensor initialized");
  Serial.println("✓ Flow meter initialized");
  Serial.println("✓ Soil moisture sensor ready");
  Serial.println("✓ Relay initialized");
  Serial.println("\n📡 Starting data transmission...\n");
  
  oldFlowTime = millis();
}

void loop() {
  static unsigned long lastReadTime = 0;
  unsigned long currentTime = millis();
  
  // Calculate flow rate every second
  if (currentTime - oldFlowTime > 1000) {
    // Disable interrupts while reading
    detachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN));
    
    // Calculate flow rate (L/min)
    flowRate = ((1000.0 / (currentTime - oldFlowTime)) * flowPulseCount) / FLOW_CALIBRATION;
    
    // Update total liters
    totalLiters += (flowPulseCount / FLOW_CALIBRATION);
    
    // Reset for next calculation
    flowPulseCount = 0;
    oldFlowTime = currentTime;
    
    // Re-enable interrupts
    attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  }
  
  // Read all sensors every READ_INTERVAL
  if (currentTime - lastReadTime >= READ_INTERVAL) {
    lastReadTime = currentTime;
    
    // Read DHT22
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    
    // Read soil moisture (convert to percentage)
    int soilRaw = analogRead(SOIL_MOISTURE_PIN);
    float soilMoisture = map(soilRaw, 4095, 0, 0, 100); // Inverted for capacitive sensors
    
    // Toggle relay occasionally (simulating irrigation control)
    static int readCount = 0;
    readCount++;
    if (readCount % 6 == 0) { // Every 30 seconds
      relayState = !relayState;
      digitalWrite(RELAY_PIN, relayState ? HIGH : LOW);
    }
    
    // Check for sensor errors
    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("ERROR: Failed to read from DHT sensor!");
      temperature = 0.0;
      humidity = 0.0;
    }
    
    // Create JSON document
    StaticJsonDocument<256> doc;
    doc["client_id"] = CLIENT_ID;
    doc["temperature"] = round(temperature * 10) / 10.0;
    doc["humidity"] = round(humidity * 10) / 10.0;
    doc["soil_moisture"] = round(soilMoisture * 10) / 10.0;
    doc["flow_rate"] = round(flowRate * 100) / 100.0;
    doc["total_liters"] = round(totalLiters * 100) / 100.0;
    doc["relay_state"] = relayState;
    doc["timestamp"] = currentTime / 1000;
    
    // Send data with prefix
    Serial.print("DATA:");
    serializeJson(doc, Serial);
    Serial.println();
    
    // Also print human-readable format
    Serial.print("📊 Temp: ");
    Serial.print(temperature, 1);
    Serial.print("°C | Humidity: ");
    Serial.print(humidity, 1);
    Serial.print("% | Soil: ");
    Serial.print(soilMoisture, 1);
    Serial.print("% | Flow: ");
    Serial.print(flowRate, 2);
    Serial.print(" L/min | Total: ");
    Serial.print(totalLiters, 2);
    Serial.print(" L | Relay: ");
    Serial.println(relayState ? "ON" : "OFF");
    Serial.println();
  }
  
  delay(100);
}
