# FarmFederate Hardware Setup Guide

## Overview
This guide provides step-by-step instructions for setting up the ESP32 hardware components according to the research paper implementation.

## Hardware Components

### 1. ESP32-CAM Module (Enhanced v2.0-federated)
**Purpose**: Capture crop images with multi-shot quality assessment and federated learning support

**Hardware Requirements**:
- ESP32-CAM (AI-Thinker or compatible)
- USB-to-Serial programmer (FTDI or CP2102)
- MicroSD card (optional, for local storage)
- LED flash (built-in)

**Wiring for Programming**:
```
ESP32-CAM          USB-to-Serial
---------------------------------
5V         <->     5V (or 3.3V)
GND        <->     GND
U0R (RX)   <->     TX
U0T (TX)   <->     RX
IO0        <->     GND (for upload mode)
```

**Configuration Steps**:

1. **Edit Configuration** in `backend/hardware/esp32cam_uploader/src/main.cpp`:
   ```cpp
   // Lines 39-40: WiFi Credentials
   const char* WIFI_SSID = "YourWiFiNetwork";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   
   // Lines 43-44: Backend Server URLs
   const char* SERVER_URL = "http://YOUR_BACKEND_IP:8000/predict";
   const char* TELEMETRY_URL = "http://YOUR_BACKEND_IP:8000/telemetry";
   
   // Line 45: Device ID (unique for federated learning)
   const char* DEVICE_ID = "esp32cam_01";  // Change for each camera
   ```

2. **Enhanced Features Configuration**:
   ```cpp
   // Multi-shot capture (Line 49)
   #define MULTI_SHOT_COUNT      3        // Capture 3 images per session
   
   // Quality assessment (Line 50)
   #define QUALITY_THRESHOLD     0.7      // Only upload high-quality images
   
   // Adaptive intervals (Line 53)
   #define ADAPTIVE_INTERVAL     true     // Adjust frequency based on detections
   #define MIN_CAPTURE_INTERVAL  30000    // 30 seconds minimum
   #define MAX_CAPTURE_INTERVAL  300000   // 5 minutes maximum
   ```

3. **Upload Firmware**:
   ```bash
   cd backend/hardware/esp32cam_uploader
   
   # Using PlatformIO (recommended)
   platformio run --target upload
   
   # Monitor serial output
   platformio device monitor --baud 115200
   ```

4. **Expected Serial Output**:
   ```
   ========================================
   ESP32-CAM BOOT
   ========================================
   LED init OK
   Camera OK (JPEG 800x600 Q=12)
   WiFi connecting... OK
   IP: 192.168.1.112
   RSSI: -65 dBm
   === READY ===
   
   [Capture] Multi-shot session started (3 images)
   [Quality] Shot 1: 0.85 (GOOD)
   [Quality] Shot 2: 0.78 (GOOD)
   [Quality] Shot 3: 0.92 (BEST - Selected)
   [Upload] Sending to http://192.168.1.100:8000/predict
   [Response] Status: healthy, Confidence: 0.91
   [Telemetry] Sent: uptime=125s, captures=5, success=4
   ```

---

### 2. ESP32 Sensor Node
**Purpose**: Monitor environmental conditions (soil moisture, temperature, humidity) for federated data collection

**Hardware Requirements**:
- ESP32 DevKit or similar
- DHT22 Temperature/Humidity sensor
- Soil moisture sensor (analog or capacitive)
- Jumper wires

**Wiring Diagram**:
```
ESP32          Sensor
---------------------------------
GPIO 4   <->   DHT22 Data Pin
GPIO 34  <->   Soil Moisture Analog Out
3.3V     <->   Sensor VCC (both sensors)
GND      <->   Sensor GND (both sensors)
```

**Configuration Steps**:

1. **Edit Configuration** in `backend/hardware/esp32_sensor_node/esp32_sensor_node.ino`:
   ```cpp
   // Lines 7-8: WiFi Credentials
   const char* ssid = "YourWiFiNetwork";
   const char* pass = "YourWiFiPassword";
   
   // Line 10: MQTT Broker IP
   const char* mqtt_server = "192.168.1.100";  // Your backend server IP
   
   // Line 12: Unique Client ID
   const char* client_id = "esp32_sensor_01";  // Change for each device
   ```

2. **Install Required Libraries** (Arduino IDE):
   - WiFi (built-in)
   - PubSubClient (by Nick O'Leary)
   - ArduinoJson (by Benoit Blanchon)
   - DHT sensor library (by Adafruit)

3. **Upload Firmware**:
   - Open Arduino IDE
   - Select Board: "ESP32 Dev Module"
   - Select Port: COM port of your ESP32
   - Click Upload

4. **Sensor Calibration**:
   ```cpp
   // Adjust Line 54 for your soil sensor:
   float soil_pct = map(soil_raw, 4095, 0, 0, 100);
   
   // Calibration:
   // - In dry air: note the value (e.g., 3500)
   // - In water: note the value (e.g., 1500)
   // - Update: map(soil_raw, 3500, 1500, 0, 100)
   ```

5. **Verify Data Flow**:
   - Open Serial Monitor (115200 baud)
   - Should see MQTT connection messages
   - Check backend terminal for: "Saved sensor: checkpoints_paper\sensors\esp32_sensor_01.json"

---

## MQTT Topics and Data Format

### Sensor Data (Published by ESP32)
**Topic**: `farmfederate/sensors/esp32`

**JSON Format**:
```json
{
  "client_id": "esp32_sensor_01",
  "soil_moisture": 45.2,
  "temp": 28.4,
  "humidity": 62.1
}
```

### Control Commands (Subscribed by ESP32)
**Topic**: `farm/cmd/<device_id>`

**JSON Format**:
```json
{
  "action": "activate_relay",
  "device": "relay1",
  "state": true
}
```

---

## Federated Learning Hardware Deployment

### Multi-Device Setup
For federated learning, deploy multiple ESP32-CAM and sensor nodes:

1. **Device Naming Convention**:
   - ESP32-CAM: `esp32cam_01`, `esp32cam_02`, etc.
   - Sensors: `esp32_sensor_01`, `esp32_sensor_02`, etc.

2. **Geographic Distribution**:
   - Deploy devices in different field locations
   - Each device becomes a federated learning client
   - Local data stays on device, only model updates shared

3. **Data Collection Strategy**:
   - **Uniform**: All devices capture at same interval
   - **Adaptive**: Devices adjust based on local conditions
   - **On-demand**: Trigger captures via MQTT commands

---

## Research Features Implementation

### 1. Multi-Shot Capture
- Captures 3 images per session
- Assesses quality based on file size and compression
- Selects best image automatically
- Reduces noisy training data

### 2. Quality Assessment
- Scores: 0.0 (poor) to 1.0 (excellent)
- Threshold: 0.7 (configurable)
- Metrics: file size, compression ratio, brightness

### 3. Adaptive Intervals
- **Disease detected**: Increase frequency (30s minimum)
- **Healthy crops**: Decrease frequency (5min maximum)
- Saves power and bandwidth

### 4. Telemetry System
- Reports every 10 captures or 5 minutes
- Includes: uptime, capture counts, quality scores
- Enables fleet monitoring and debugging

### 5. Exponential Backoff
- Initial retry delay: 5 seconds
- Multiplier: 2x per failure
- Maximum retries: 3
- Prevents server overload

---

## Troubleshooting

### ESP32-CAM Issues

**Problem**: Camera fails to initialize
```
Solution:
1. Check power supply (needs stable 5V, 500mA+)
2. Try different camera model in code
3. Reset ESP32-CAM (press reset button)
```

**Problem**: WiFi connection fails
```
Solution:
1. Verify SSID and password
2. Check WiFi signal strength (RSSI > -70 dBm)
3. Ensure 2.4GHz network (ESP32 doesn't support 5GHz)
```

**Problem**: Upload fails with "Connection refused"
```
Solution:
1. Verify backend server is running (check port 8000)
2. Confirm SERVER_URL has correct IP address
3. Test with: curl http://YOUR_IP:8000/health
```

### ESP32 Sensor Node Issues

**Problem**: MQTT connection fails
```
Solution:
1. Verify Mosquitto service is running
2. Check mqtt_server IP address
3. Test with: mosquitto_sub -t "farmfederate/sensors/#"
```

**Problem**: Sensor readings are incorrect
```
Solution:
1. Check wiring connections
2. Calibrate sensors (see Sensor Calibration section)
3. Test sensors individually with simple sketches
```

**Problem**: No data in backend
```
Solution:
1. Verify MQTT listener is running
2. Check topic name matches ("farmfederate/sensors/esp32")
3. Look for sensor JSON files in: backend/checkpoints_paper/sensors/
```

---

## Performance Monitoring

### ESP32-CAM Metrics
- **Capture Success Rate**: Should be > 90%
- **Average Quality Score**: Should be > 0.75
- **Upload Latency**: Should be < 5 seconds
- **WiFi RSSI**: Should be > -70 dBm

### ESP32 Sensor Metrics
- **MQTT Publish Rate**: Every 30 seconds
- **Data Loss**: Should be < 5%
- **Sensor Accuracy**: ±2% (after calibration)

---

## Next Steps

1. **Deploy Hardware**: Follow configuration steps above
2. **Verify Data Flow**: Check backend logs and sensor files
3. **Run Federated Training**: Use collected data with `train_fed_multimodal.py`
4. **Monitor Performance**: Check telemetry data and system health

## Additional Resources

- ESP32-CAM Pinout: [link]
- PlatformIO Documentation: https://docs.platformio.org/
- Arduino ESP32 Core: https://github.com/espressif/arduino-esp32
- Research Paper Implementation: See RESEARCH_PAPER_IMPLEMENTATION.md
