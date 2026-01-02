# ESP32 Hardware Troubleshooting Guide

## Your Hardware Setup
- ESP32 board
- Temperature & Humidity sensor (DHT22)
- Relay module
- Solenoid valve
- Water flow meter
- Soil moisture sensor (optional)

## Problem: No Data Being Received

### Step 1: Check ESP32 Serial Monitor

1. **Connect ESP32 to computer via USB**
2. **Open Arduino IDE → Tools → Serial Monitor** (115200 baud)
3. **Look for these messages:**
   ```
   Connecting to WiFi...
   WiFi connected
   IP address: 192.168.x.x
   MQTT connecting...
   MQTT connected
   Publishing sensor data...
   ```

### Step 2: Verify WiFi Connection

**Check if ESP32 has:**
- ✓ WiFi SSID configured correctly
- ✓ WiFi password correct
- ✓ ESP32 gets an IP address
- ✓ Same network as your computer

**In the ESP32 code (`esp32_sensor_node.ino`):**
```cpp
const char* ssid = "YOUR_ACTUAL_WIFI_NAME";    // ← Change this!
const char* pass = "YOUR_ACTUAL_WIFI_PASSWORD"; // ← Change this!
```

### Step 3: Verify MQTT Broker IP Address

**Find your computer's IP address:**
- Open Command Prompt: `ipconfig`
- Look for "IPv4 Address" under your WiFi/Ethernet adapter
- Example: `192.168.1.105`

**Update ESP32 code:**
```cpp
const char* mqtt_server = "192.168.1.105"; // ← Your computer's IP!
```

⚠️ **DO NOT use "localhost" or "127.0.0.1" on ESP32** - it won't work!

### Step 4: Check MQTT Broker on Computer

**Verify Mosquitto is running:**
```powershell
Get-Service mosquitto
# Should show Status: Running
```

**If not running:**
```powershell
Start-Service mosquitto
```

**Test if broker accepts connections:**
```powershell
mosquitto_sub -h localhost -t "test" -v
```
Leave this running, then in another window:
```powershell
mosquitto_pub -h localhost -t "test" -m "hello"
```
You should see "hello" appear.

### Step 5: Monitor MQTT Messages

**Open a PowerShell window and run:**
```powershell
mosquitto_sub -h localhost -t "farmfederate/sensors/#" -v
```

This will show ALL messages from ESP32. If nothing appears:
- ESP32 is not connected to MQTT broker
- Wrong IP address
- Firewall blocking port 1883

### Step 6: Check Firewall

**Allow MQTT port 1883:**
```powershell
New-NetFirewallRule -DisplayName "MQTT Broker" -Direction Inbound -Protocol TCP -LocalPort 1883 -Action Allow
```

### Step 7: Verify ESP32 Pin Connections

**DHT22 Temperature/Humidity Sensor:**
```
DHT22 VCC  → ESP32 3.3V
DHT22 DATA → ESP32 GPIO 4
DHT22 GND  → ESP32 GND
```

**Water Flow Meter:**
```
Flow Sensor VCC    → ESP32 5V
Flow Sensor SIGNAL → ESP32 GPIO (define in code)
Flow Sensor GND    → ESP32 GND
```

**Relay Module:**
```
Relay VCC → ESP32 5V
Relay IN  → ESP32 GPIO (define in code)
Relay GND → ESP32 GND
```

**Solenoid Valve:**
```
Connect to relay output (NO/NC terminals)
Valve needs external power supply (12V/24V)
```

### Step 8: Complete ESP32 Code with All Sensors

Create a new file: `esp32_full_sensors.ino`

```cpp
#include <WiFi.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include <DHT.h>

// WiFi Configuration - CHANGE THESE!
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// MQTT Configuration - CHANGE THIS TO YOUR PC'S IP!
const char* mqtt_server = "192.168.1.XXX";  // Your computer's IP
const int mqtt_port = 1883;
const char* client_id = "esp32_field_01";

// Pin Definitions
#define DHT_PIN 4           // DHT22 data pin
#define RELAY_PIN 5         // Relay control
#define FLOW_SENSOR_PIN 18  // Water flow meter
#define SOIL_MOISTURE_PIN 34 // Analog soil moisture (optional)

#define DHTTYPE DHT22
DHT dht(DHT_PIN, DHTTYPE);

WiFiClient espClient;
PubSubClient client(espClient);

// Flow meter variables
volatile int flowPulseCount = 0;
float flowRate = 0.0;
unsigned long oldTime = 0;

void IRAM_ATTR flowPulseCounter() {
  flowPulseCount++;
}

void setup_wifi() {
  delay(10);
  Serial.println();
  Serial.print("Connecting to WiFi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("");
    Serial.println("WiFi connected!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
  } else {
    Serial.println("");
    Serial.println("WiFi connection FAILED!");
  }
}

void reconnect_mqtt() {
  while (!client.connected()) {
    Serial.print("Attempting MQTT connection to ");
    Serial.print(mqtt_server);
    Serial.print(":");
    Serial.println(mqtt_port);
    
    if (client.connect(client_id)) {
      Serial.println("MQTT connected!");
      // Subscribe to control topic
      client.subscribe("farmfederate/control/relay");
      Serial.println("Subscribed to control topic");
    } else {
      Serial.print("MQTT connection failed, rc=");
      Serial.print(client.state());
      Serial.println(" retrying in 5 seconds");
      delay(5000);
    }
  }
}

void callback(char* topic, byte* payload, unsigned int length) {
  Serial.print("Message arrived [");
  Serial.print(topic);
  Serial.print("]: ");
  
  String message = "";
  for (int i = 0; i < length; i++) {
    message += (char)payload[i];
  }
  Serial.println(message);
  
  // Control relay based on message
  if (String(topic) == "farmfederate/control/relay") {
    if (message == "ON") {
      digitalWrite(RELAY_PIN, HIGH);
      Serial.println("Relay turned ON");
    } else if (message == "OFF") {
      digitalWrite(RELAY_PIN, LOW);
      Serial.println("Relay turned OFF");
    }
  }
}

void setup() {
  Serial.begin(115200);
  Serial.println("ESP32 Starting...");
  
  // Initialize pins
  pinMode(RELAY_PIN, OUTPUT);
  digitalWrite(RELAY_PIN, LOW);
  pinMode(FLOW_SENSOR_PIN, INPUT_PULLUP);
  
  // Attach interrupt for flow sensor
  attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
  
  // Initialize sensors
  dht.begin();
  
  // Connect to WiFi
  setup_wifi();
  
  // Setup MQTT
  client.setServer(mqtt_server, mqtt_port);
  client.setCallback(callback);
  
  Serial.println("Setup complete!");
}

void loop() {
  if (!client.connected()) {
    reconnect_mqtt();
  }
  client.loop();
  
  // Read sensors every 10 seconds
  static unsigned long lastRead = 0;
  if (millis() - lastRead > 10000) {
    lastRead = millis();
    
    // Read DHT22
    float temperature = dht.readTemperature();
    float humidity = dht.readHumidity();
    
    // Read soil moisture (if connected)
    int soilRaw = analogRead(SOIL_MOISTURE_PIN);
    float soilMoisture = map(soilRaw, 0, 4095, 0, 100);
    
    // Calculate flow rate
    if (millis() - oldTime > 1000) {
      detachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN));
      flowRate = ((1000.0 / (millis() - oldTime)) * flowPulseCount) / 7.5; // L/min
      oldTime = millis();
      flowPulseCount = 0;
      attachInterrupt(digitalPinToInterrupt(FLOW_SENSOR_PIN), flowPulseCounter, FALLING);
    }
    
    // Check for valid readings
    if (isnan(temperature) || isnan(humidity)) {
      Serial.println("Failed to read from DHT sensor!");
      temperature = 0;
      humidity = 0;
    }
    
    // Create JSON
    StaticJsonDocument<512> doc;
    doc["client_id"] = client_id;
    doc["temperature"] = temperature;
    doc["humidity"] = humidity;
    doc["soil_moisture"] = soilMoisture;
    doc["flow_rate"] = flowRate;
    doc["relay_state"] = digitalRead(RELAY_PIN) ? "ON" : "OFF";
    doc["timestamp"] = millis() / 1000;
    
    char jsonBuffer[512];
    serializeJson(doc, jsonBuffer);
    
    // Publish to MQTT
    String topic = String("farmfederate/sensors/") + client_id;
    bool published = client.publish(topic.c_str(), jsonBuffer);
    
    if (published) {
      Serial.println("✓ Published sensor data:");
      Serial.println(jsonBuffer);
    } else {
      Serial.println("✗ Failed to publish!");
    }
  }
}
```

### Step 9: Test Connection from Computer

**Test if ESP32 can be reached:**
```powershell
# Find your ESP32's IP from serial monitor, then:
ping 192.168.1.XXX  # Replace with ESP32's IP
```

### Step 10: Check Received Data

**View saved sensor files:**
```powershell
cd backend
Get-ChildItem checkpoints_paper -Recurse -Filter "*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 5
```

**Read latest sensor file:**
```powershell
Get-Content (Get-ChildItem checkpoints_paper -Recurse -Filter "*.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
```

## Common Issues & Solutions

### Issue 1: "WiFi connection FAILED"
- ✓ Check SSID spelling (case-sensitive!)
- ✓ Check password
- ✓ ESP32 too far from router
- ✓ WiFi is 2.4GHz (ESP32 doesn't support 5GHz)

### Issue 2: "MQTT connection failed, rc=-2"
- ✓ Wrong broker IP address
- ✓ Mosquitto not running
- ✓ Firewall blocking port 1883

### Issue 3: "Failed to read from DHT sensor"
- ✓ DHT22 not connected properly
- ✓ Wrong pin number
- ✓ Sensor damaged
- ✓ Missing pull-up resistor (10kΩ on data line)

### Issue 4: "ESP32 connects but no data appears"
- ✓ MQTT listener not running
- ✓ Wrong topic in listener
- ✓ Check listener console for errors

### Issue 5: Flow meter reads 0
- ✓ Water not flowing
- ✓ Wrong pin assignment
- ✓ Sensor not connected
- ✓ Check calibration factor

## Quick Test Commands

**On your computer:**
```powershell
# 1. Check MQTT broker
Get-Service mosquitto

# 2. Monitor MQTT messages
mosquitto_sub -h localhost -t "farmfederate/sensors/#" -v

# 3. Test publish from computer
mosquitto_pub -h localhost -t "farmfederate/sensors/test" -m '{"test":true}'

# 4. Check sensor data files
Get-ChildItem backend\checkpoints_paper -Recurse -Filter "*.json"

# 5. Test backend API
curl http://localhost:8000/sensors/latest
```

## Need More Help?

1. **Copy your ESP32 serial monitor output** and share it
2. **Show MQTT monitor output** (from mosquitto_sub)
3. **Check Windows Firewall** - may need to allow mosquitto.exe
4. **Verify network** - ESP32 and computer on same WiFi network
5. **Test with simple code first** - just WiFi + MQTT, no sensors
