# ESP32-CAM Upload Instructions

## ⚠️ IMPORTANT: Put ESP32-CAM in Programming Mode

The ESP32-CAM needs to be in **programming mode** before uploading firmware.

### Step-by-Step Upload Process

#### 1. **Prepare Hardware**
   - Connect ESP32-CAM to PC via FTDI programmer
   - **CRITICAL**: Connect **GPIO 0** to **GND** (ground pin)
   - This puts the ESP32 in flash/programming mode

#### 2. **Power Cycle the Device**
   - Disconnect USB cable
   - Wait 2 seconds
   - Reconnect USB cable
   - ESP32-CAM will boot in programming mode

#### 3. **Upload Firmware**
   Run this command in PowerShell:
   ```powershell
   cd "c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend\hardware\esp32cam_uploader"
   & "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run --target upload
   ```

#### 4. **After Upload Completes**
   - **Disconnect GPIO 0 from GND**
   - Press the **RESET** button on ESP32-CAM
   - Or power cycle (unplug and replug USB)

#### 5. **Monitor Serial Output**
   ```powershell
   & "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" device monitor
   ```
   Press `Ctrl+C` to exit monitor

## 📌 Hardware Setup Diagram

```
┌──────────────────┐
│   FTDI Adapter   │
│                  │
│  VCC → 5V        │ ──┐
│  GND → GND       │ ──┤ ESP32-CAM Power
│  TX  → U0R       │ ──┤
│  RX  → U0T       │ ──┘
└──────────────────┘
         ↓
    Programming Mode:
    GPIO 0 ──→ GND (short with jumper wire)
```

### Visual Connection Guide

**ESP32-CAM Pins (Back view):**
```
      ┌─────────────┐
      │  ESP32-CAM  │
      │             │
 GND -│●           │
 5V  -│●           │
 U0R -│●   [ANT]   │
 U0T -│●           │
GPIO0-│●           │- Short this to GND
      │   [CHIP]   │
      └─────────────┘
```

## 🔧 Troubleshooting

### Error: "Could not open COM7, the port is busy"
**Cause**: Another program is using the COM port

**Solution**:
1. Close Arduino IDE Serial Monitor
2. Close PlatformIO Serial Monitor
3. Stop any Python scripts using serial (like usb_serial_reader.py)
4. Try upload again

**Quick Fix Command**:
```powershell
# Check what's using COM7
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
```

### Error: "A fatal error occurred: Failed to connect"
**Cause**: ESP32-CAM not in programming mode

**Solution**:
1. ✅ **GPIO 0 MUST be connected to GND** before powering on
2. Power cycle the device (unplug/replug USB)
3. Try upload again
4. If still fails, hold GPIO 0 to GND and press RESET button

### Error: "Timed out waiting for packet header"
**Cause**: Wrong baud rate or hardware issue

**Solution**:
1. Check connections (especially TX/RX crossover)
2. Ensure FTDI is supplying sufficient power (500mA+)
3. Try slower upload speed in platformio.ini:
   ```ini
   upload_speed = 115200  ; Change from 921600
   ```

### Upload Successful but Nothing Happens
**Cause**: Still in programming mode

**Solution**:
1. ✅ **DISCONNECT GPIO 0 from GND**
2. Press RESET button
3. Device should now run the firmware

## 📝 Configuration Checklist

Before uploading, verify these settings in `src/main.cpp`:

```cpp
// 1. WiFi Credentials
const char* WIFI_SSID = "YOUR_SSID";          // ← Change this
const char* WIFI_PASSWORD = "YOUR_PASSWORD";  // ← Change this

// 2. Backend Server URL
const char* SERVER_URL = "http://192.168.1.100:8000/predict";  // ← Change IP

// 3. Device ID (unique for each camera)
const char* DEVICE_ID = "esp32cam_01";  // ← Change if using multiple cameras
```

### Finding Your PC's IP Address:
```powershell
# Windows
ipconfig | Select-String "IPv4"

# Look for something like: 192.168.1.XXX
```

## 🚀 Quick Upload Script

Save this as `upload_esp32cam.ps1`:

```powershell
# ESP32-CAM Upload Script
$projectPath = "c:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend\hardware\esp32cam_uploader"
$pio = "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "ESP32-CAM Firmware Upload Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "INSTRUCTIONS:" -ForegroundColor Yellow
Write-Host "1. Connect GPIO 0 to GND on ESP32-CAM" -ForegroundColor Yellow
Write-Host "2. Power cycle the device (unplug/replug USB)" -ForegroundColor Yellow
Write-Host "3. Press ENTER to start upload" -ForegroundColor Yellow
Write-Host ""
Pause

Write-Host "Stopping any processes using COM7..." -ForegroundColor Green
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force
Start-Sleep -Seconds 1

Write-Host "Starting upload..." -ForegroundColor Green
& $pio run --target upload --project-dir $projectPath

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "✓ Upload Successful!" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "NEXT STEPS:" -ForegroundColor Yellow
    Write-Host "1. Disconnect GPIO 0 from GND" -ForegroundColor Yellow
    Write-Host "2. Press RESET button on ESP32-CAM" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Press ENTER to open Serial Monitor..." -ForegroundColor Cyan
    Pause
    & $pio device monitor --baud 115200
} else {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Red
    Write-Host "✗ Upload Failed" -ForegroundColor Red
    Write-Host "========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Troubleshooting:" -ForegroundColor Yellow
    Write-Host "- Is GPIO 0 connected to GND?" -ForegroundColor Yellow
    Write-Host "- Did you power cycle after connecting GPIO 0?" -ForegroundColor Yellow
    Write-Host "- Check FTDI connections (TX→U0R, RX→U0T)" -ForegroundColor Yellow
}
```

## 🎯 Expected Serial Monitor Output

After successful upload and RESET, you should see:

```
========================================
ESP32-CAM Leaf Disease Detection System
========================================

[INIT] Configuring camera...
[OK] Camera initialized successfully
[INFO] Frame size: 1600x1200, Quality: 10
[INIT] Connecting to WiFi...
[WiFi] Connecting....... Connected!
[WiFi] IP Address: 192.168.1.XXX
[WiFi] Signal Strength: -45 dBm
[READY] System initialized successfully!
Waiting for image capture trigger...
```

## 🆘 Still Having Issues?

1. **Verify Hardware**:
   - Test FTDI adapter with another device
   - Check USB cable quality
   - Ensure 5V power supply (not 3.3V)

2. **Check Drivers**:
   - Silicon Labs CP210x driver installed?
   - Device Manager shows COM7 without errors?

3. **Test Backend**:
   ```powershell
   curl http://192.168.1.XXX:8000/health
   ```
   Should return: `{"status":"ok"}`

4. **Alternative Upload Methods**:
   - Use Arduino IDE instead of PlatformIO
   - Try ESP32 Flash Tool from Espressif
   - Use ESP-IDF (advanced)

## 📚 Additional Resources

- [ESP32-CAM Pinout](https://randomnerdtutorials.com/esp32-cam-ai-thinker-pinout/)
- [PlatformIO ESP32 Guide](https://docs.platformio.org/en/latest/boards/espressif32/esp32cam.html)
- [Troubleshooting ESP32-CAM](https://randomnerdtutorials.com/esp32-cam-troubleshooting-guide/)
