# ESP32 Firmware Upload Using VS Code

## Method 1: Using PlatformIO (Recommended ⭐)

PlatformIO is better than Arduino IDE - it's faster, has better debugging, and integrates with VS Code.

### Step 1: Install PlatformIO Extension

1. Open VS Code
2. Click Extensions icon (Ctrl+Shift+X)
3. Search: **"PlatformIO IDE"**
4. Click **Install**
5. Wait for installation (takes 2-3 minutes)
6. Click **Reload** when prompted

### Step 2: Update WiFi Credentials

Open the firmware file (already open in VS Code):
```
backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino
```

**Change lines 30-31:**
```cpp
const char* WIFI_SSID = "YourActualWiFiName";      // ← Your WiFi
const char* WIFI_PASSWORD = "YourActualPassword";  // ← Your password
```

**Already set for you:**
```cpp
const char* MQTT_SERVER = "192.168.0.195";  // ✓ Correct IP
```

### Step 3: Open Project in PlatformIO

1. Click **PlatformIO icon** in sidebar (alien head icon)
2. Click **"Open Project"**
3. Navigate to: `backend/hardware/esp32_sensor_node/`
4. Click **"Open"**

### Step 4: Upload Firmware

**Method A - Using PlatformIO Toolbar:**
1. Look at bottom status bar in VS Code
2. Click the **→** (Upload) button
3. Wait for compilation and upload

**Method B - Using Command Palette:**
1. Press **Ctrl+Shift+P**
2. Type: **"PlatformIO: Upload"**
3. Press Enter

**Method C - Using Terminal:**
```bash
cd backend/hardware/esp32_sensor_node
pio run --target upload
```

### Step 5: Monitor Serial Output

After upload, monitor the ESP32:

**Method A - PlatformIO Monitor:**
1. Click **🔌** (Serial Monitor) in bottom status bar

**Method B - Command Palette:**
1. Press **Ctrl+Shift+P**
2. Type: **"PlatformIO: Serial Monitor"**

**Method C - Terminal:**
```bash
pio device monitor --port COM7 --baud 115200
```

You should see:
```
✓ WiFi connected successfully!
✓ MQTT connected successfully!
✓ Temperature: 24.5 °C
✓ Data published successfully!
```

---

## Method 2: Using Arduino Extension

If you prefer Arduino over PlatformIO:

### Step 1: Install Arduino Extension

1. Open VS Code
2. Extensions (Ctrl+Shift+X)
3. Search: **"Arduino"** by Microsoft
4. Click **Install**

### Step 2: Configure Arduino

1. Press **Ctrl+Shift+P**
2. Type: **"Arduino: Board Config"**
3. Select: **ESP32 Dev Module**

4. Press **Ctrl+Shift+P**
5. Type: **"Arduino: Select Serial Port"**
6. Select: **COM7**

### Step 3: Update WiFi Credentials

Edit lines 30-31 in the opened file.

### Step 4: Upload

1. Press **Ctrl+Shift+P**
2. Type: **"Arduino: Upload"**
3. Or click **→** in top right corner

### Step 5: Open Serial Monitor

1. Press **Ctrl+Shift+P**
2. Type: **"Arduino: Open Serial Monitor"**
3. Set baud rate to **115200**

---

## Quick Upload Commands (After Setup)

### PlatformIO (Recommended):
```bash
cd backend/hardware/esp32_sensor_node
pio run --target upload && pio device monitor
```

### Or use VS Code tasks:
Press **Ctrl+Shift+B** → Select **"PlatformIO: Upload"**

---

## After Successful Upload

Test sensors:
```bash
python backend/check_sensors_usb.py
python backend/diagnose_sensors.py
```

Control valve:
```bash
python backend/valve_controller.py open
python backend/valve_controller.py close
```

---

## Troubleshooting

### Upload Failed?
- **Hold BOOT button** on ESP32 while clicking Upload
- Try different USB port
- Check cable supports data (not power-only)

### PlatformIO Not Found?
- Restart VS Code after installation
- Check: View → Output → PlatformIO

### Serial Monitor Not Working?
- Close other serial monitors
- Make sure port is COM7
- Check baud rate is 115200

---

## Files Created

- ✅ `platformio.ini` - PlatformIO configuration (auto-installs libraries)
- ✅ `esp32_complete_sensors.ino` - Firmware with correct MQTT IP
- ✅ All diagnostic tools ready in backend/

---

## What's Already Done For You

✅ MQTT IP configured (192.168.0.195)
✅ PlatformIO config with all libraries
✅ Upload port set to COM7
✅ Monitor speed set to 115200
✅ All required libraries listed

**YOU ONLY NEED TO:**
1. Install PlatformIO extension in VS Code
2. Update WiFi credentials (lines 30-31)
3. Click Upload button

**Estimated time: 5 minutes** ⚡
