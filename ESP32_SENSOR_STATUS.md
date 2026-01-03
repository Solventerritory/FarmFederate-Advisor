# ESP32 Sensor Connection Status Report

**Generated:** January 3, 2026, 13:44  
**Status:** ❌ ESP32 NOT SENDING SENSOR DATA

---

## Current Situation

### ✅ What's Working:
- MQTT Broker running on localhost:1883
- ESP32 detected on **COM7**
- Valve control commands being sent successfully (retained in broker)
- Computer IP: **192.168.0.195**

### ❌ What's Not Working:
- ESP32 not publishing sensor data to MQTT
- No serial output from ESP32
- Sensors not detected

---

## Expected Sensors

The ESP32 should have these sensors connected:

| Sensor | GPIO Pin | Description |
|--------|----------|-------------|
| **DHT22** | GPIO 4 | Temperature & Humidity sensor |
| **Soil Moisture** | GPIO 34 | Analog soil moisture sensor |
| **Flow Meter** | GPIO 18 | YF-S201 water flow sensor |
| **Relay** | GPIO 5 or GPIO 26 | Water valve relay control |
| **Solenoid Valve** | GPIO 27 | Connected via relay |

---

## Root Cause Analysis

### Most Likely Issues:

1. **Firmware Not Uploaded or Wrong Firmware**
   - ESP32 may have old/different firmware
   - Needs esp32_complete_sensors.ino uploaded

2. **WiFi Configuration**
   - SSID or password incorrect in firmware
   - ESP32 supports only 2.4GHz WiFi (not 5GHz)

3. **MQTT Server IP**
   - Firmware has wrong IP address
   - Should be: **192.168.0.195**

4. **Sensors Not Physically Connected**
   - Wires may be loose or disconnected
   - Sensors may need power

---

## Step-by-Step Fix

### Step 1: Upload Correct Firmware

1. Open Arduino IDE
2. Open file: `backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino`
3. Update these lines:
   ```cpp
   const char* WIFI_SSID = "YOUR_WIFI_NAME";     // ← Your WiFi name
   const char* WIFI_PASSWORD = "YOUR_PASSWORD";  // ← Your WiFi password
   const char* MQTT_SERVER = "192.168.0.195";    // ← This computer's IP
   ```
4. Select: **Tools → Board → ESP32 Dev Module**
5. Select: **Tools → Port → COM7**
6. Click **Upload** button
7. Wait for "Done uploading"

### Step 2: Verify Connection

1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. Press **Reset** button on ESP32
4. Look for these messages:
   ```
   ✓ WiFi connected successfully!
   ✓ MQTT connected successfully!
   ✓ Temperature: XX.X °C
   ✓ Humidity: XX.X %
   ```

### Step 3: Check Sensor Wiring

**DHT22 (Temperature/Humidity):**
- VCC → 3.3V or 5V
- DATA → GPIO 4
- GND → GND

**Soil Moisture:**
- VCC → 3.3V or 5V
- SIGNAL → GPIO 34
- GND → GND

**Flow Meter:**
- RED → 5V
- BLACK → GND
- YELLOW → GPIO 18

**Relay Module:**
- VCC → 5V
- IN → GPIO 5
- GND → GND

**Solenoid Valve:**
- Connect to relay NO (Normally Open) terminals

### Step 4: Test Sensors

Run this command after uploading firmware:
```bash
python backend/check_sensors_usb.py
```

Expected output:
```
✅ Temperature (DHT22 on GPIO 4)
    Value: 25.3°C
✅ Humidity (DHT22 on GPIO 4)
    Value: 65.2%
✅ Soil Moisture (GPIO 34)
    Value: 45.0%
✅ Flow Meter (GPIO 18)
    Value: 0.0 L/min
✅ Relay/Valve (GPIO 5)
```

---

## Quick Diagnostic Commands

```bash
# 1. Check if ESP32 is connected via USB
python backend/check_sensors_usb.py

# 2. Check MQTT sensor data (if WiFi connected)
python backend/diagnose_sensors.py

# 3. Troubleshoot connection issues
python backend/troubleshoot_esp32.py

# 4. Test valve control
python backend/valve_controller.py open

# 5. Monitor MQTT traffic
python backend/mqtt_monitor.py
```

---

## Firewall Configuration (Optional)

If sensors still don't work after firmware upload, add firewall rule:

```powershell
New-NetFirewallRule -DisplayName 'MQTT Broker' -Direction Inbound -LocalPort 1883 -Protocol TCP -Action Allow
```

---

## Alternative: Test Without WiFi

If you want to test sensors without WiFi/MQTT:

1. Open: `backend/hardware/esp32_sensor_node/esp32_usb_serial.ino`
2. Upload this simpler version
3. It will output sensor data to USB serial only
4. Run: `python backend/usb_serial_reader.py`

---

## What Happens After Fix

Once ESP32 is properly configured and sensors connected:

1. **ESP32 boots up** → Connects to WiFi
2. **Connects to MQTT broker** at your computer's IP
3. **Reads sensors** every 10 seconds:
   - Temperature & Humidity from DHT22
   - Soil moisture from analog sensor
   - Flow rate from water meter
   - Relay state
4. **Publishes data** to `farmfederate/sensors/esp32_field_01`
5. **Receives valve commands** from `farmfederate/control/relay`
6. **Controls relay** to open/close solenoid valve

---

## Success Indicators

### Arduino Serial Monitor:
```
✓ WiFi connected successfully!
IP Address: 192.168.X.X
✓ MQTT connected successfully!
✓ Subscribed to: farmfederate/control/relay
✓ Temperature: 24.5 °C
✓ Humidity: 68.2 %
✓ Data published successfully!
```

### Python Diagnostic:
```
✅ ESP32 CONNECTED
✅ ALL SENSORS WORKING CORRECTLY
Sensors Connected: 5/5
```

---

## Need Help?

**Files to check:**
- Firmware: `backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino`
- Wiring diagram: `backend/hardware/README.md`
- Troubleshooting: `ESP32_TROUBLESHOOTING.md`

**Tools created:**
- `backend/check_sensors_usb.py` - Check via USB
- `backend/diagnose_sensors.py` - Check via MQTT
- `backend/troubleshoot_esp32.py` - Connection troubleshooter
- `backend/valve_controller.py` - Control solenoid valve

---

**Next Action:** Upload correct firmware to ESP32 with updated WiFi credentials and MQTT server IP (192.168.0.195)
