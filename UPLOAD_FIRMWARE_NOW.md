# ESP32 FIRMWARE UPLOAD INSTRUCTIONS

## ⚠️ IMPORTANT: Update WiFi Credentials First!

The firmware file is ready with the correct MQTT IP address (192.168.0.195).

**YOU MUST UPDATE** these lines in the file:
```cpp
const char* WIFI_SSID = "YOUR_WIFI_SSID";        // ← Your WiFi name
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"; // ← Your WiFi password
```

## Step-by-Step Upload Process

### 1. Open Arduino IDE
   - If not installed: https://www.arduino.cc/en/software

### 2. Install ESP32 Board Support
   - File → Preferences
   - Add to "Additional Board Manager URLs":
     ```
     https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
     ```
   - Tools → Board → Boards Manager
   - Search "ESP32" and install "esp32 by Espressif Systems"

### 3. Install Required Libraries
   - Tools → Manage Libraries
   - Install these libraries:
     * **DHT sensor library** by Adafruit
     * **Adafruit Unified Sensor**
     * **PubSubClient** by Nick O'Leary
     * **ArduinoJson** by Benoit Blanchon

### 4. Open Firmware File
   ```
   File → Open →
   backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino
   ```

### 5. Update WiFi Credentials
   - Change lines 30-31 with YOUR WiFi name and password
   - IMPORTANT: WiFi name is case-sensitive!
   - ESP32 only supports 2.4GHz WiFi (not 5GHz)

### 6. Configure Arduino IDE
   - **Tools → Board** → ESP32 Dev Module
   - **Tools → Port** → COM7 (or your ESP32 port)
   - **Tools → Upload Speed** → 115200

### 7. Upload Firmware
   - Click the **Upload** button (→ icon)
   - Wait for "Done uploading" message
   - Should take about 30 seconds

### 8. Verify Upload
   - Click **Serial Monitor** button (magnifying glass icon)
   - Set baud rate to **115200**
   - Press **Reset** button on ESP32
   - You should see:
     ```
     ✓ WiFi connected successfully!
     IP Address: 192.168.X.X
     ✓ MQTT connected successfully!
     ✓ Temperature: XX.X °C
     ✓ Humidity: XX.X %
     ✓ Data published successfully!
     ```

## After Upload: Test Sensors

Run this command to verify all sensors are working:
```bash
python backend/check_sensors_usb.py
```

Expected output:
```
✅ Temperature (DHT22 on GPIO 4)
✅ Humidity (DHT22 on GPIO 4)
✅ Soil Moisture (GPIO 34)
✅ Flow Meter (GPIO 18)
✅ Relay/Valve (GPIO 5)
```

Or check via MQTT:
```bash
python backend/diagnose_sensors.py
```

## Troubleshooting

### Upload Failed?
- Close Serial Monitor if open
- Press and hold **BOOT** button on ESP32, then click Upload
- Try a different USB cable (must support data, not power-only)
- Try a different USB port

### WiFi Not Connecting?
- Check SSID spelling (case-sensitive)
- Check password
- Make sure it's 2.4GHz WiFi (ESP32 doesn't support 5GHz)
- Check router distance

### MQTT Not Connecting?
- Verify IP address: 192.168.0.195
- Make sure MQTT broker is running: `net start mosquitto`
- Check firewall isn't blocking port 1883

### No Sensor Data?
- Check wiring connections
- Make sure sensors have power
- Verify GPIO pin numbers in firmware match your wiring

## Quick Commands

After successful upload:

```bash
# Check sensors via USB
python backend/check_sensors_usb.py

# Check via MQTT
python backend/diagnose_sensors.py

# Open valve
python backend/valve_controller.py open

# Close valve
python backend/valve_controller.py close
```

---

**File Location:** `backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino`

**✅ MQTT IP Already Updated:** 192.168.0.195  
**⚠️ YOU MUST UPDATE:** WiFi SSID and Password (lines 30-31)
