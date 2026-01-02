# ESP32-CAM Camera Troubleshooting Guide

## Issue Diagnosed: Camera Hardware Failure

### Test Results Summary

✅ **WORKING COMPONENTS:**
- ESP32 chip (ESP32-D0WD-V3) - Fully functional
- Serial communication (115200 baud)
- WiFi connectivity (connects to "Ayush" network)
- LED flash control
- Program logic and loop execution
- USB connection via COM8

❌ **FAILED COMPONENT:**
- **Camera Module** - Initialization fails, causes boot loop

### Symptoms

1. **With Camera Init Enabled:**
   - Continuous boot loop (restarts every 2-3 seconds)
   - Only prints startup banner "========================================"
   - Never reaches WiFi connection phase
   - Error: `esp_camera_init()` returns failure

2. **With Camera Init Disabled (Test Mode):**
   - System runs perfectly stable
   - WiFi connects successfully
   - LED flashes every 5 seconds
   - No crashes or restarts

### Root Cause

The camera module (OV2640 sensor) is either:
1. **Not properly connected** via ribbon cable
2. **Faulty/defective** sensor hardware
3. **Incompatible** with the board

### Solutions

#### 1. Check Physical Connection (Try This First!)

**Steps:**
1. Power off the ESP32-CAM completely
2. Locate the camera ribbon cable connector
3. Gently pull out the camera ribbon cable
4. Inspect the cable for damage
5. Reinsert the cable firmly into the connector
   - Blue side should face the board
   - Ensure it's fully seated
6. Lock the connector (if applicable)
7. Power on and test

#### 2. Verify Camera Module

- Camera module should have "OV2640" marking
- Check for physical damage on the lens
- Ensure no bent pins on the connector

#### 3. Power Supply Check

- Use a quality USB cable (data + power)
- Try a powered USB hub (5V 2A recommended)
- Avoid USB extension cables

#### 4. Replace Camera Module

If above steps don't work, the camera module is likely defective.

**Where to buy:**
- Search for "OV2640 ESP32-CAM camera module"
- Cost: $3-10 USD
- Ensure it's compatible with AI-Thinker ESP32-CAM

### Firmware Configuration

**Current Settings (Optimized for stability):**
```cpp
// Camera resolution: SVGA (800x600)
config.frame_size = FRAMESIZE_SVGA;
config.jpeg_quality = 12;
```

**After camera is fixed, you can try higher resolution:**
```cpp
// For better quality (requires more power)
config.frame_size = FRAMESIZE_XGA;  // 1024x768
config.jpeg_quality = 10;
```

### Testing After Repair

**1. Check Serial Output:**
Run the monitor script:
```powershell
cd C:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend\hardware\esp32cam_uploader
python monitor_and_wait.py
```

**Expected Output:**
```
========================================
ESP32-CAM Leaf Disease Detection System
========================================

[INFO] Chip: ESP32-D0WD-V3
[INFO] MAC: 28:05:a5:66:17:dc
[INFO] Free Heap: XXXXX bytes
[INFO] PSRAM: Found

[INIT] Configuring camera...
[INIT] Camera initialized successfully!    <-- This line is critical
[INIT] Connecting to WiFi...
[WiFi] Connected! IP: 192.168.X.X
[READY] System initialized successfully!
```

**2. Verify Backend Connection:**
The backend server should receive POST requests with images:
```
INFO:     192.168.X.X:XXXX - "POST /predict HTTP/1.1" 200 OK
```

### Backend Server

Backend is running at: `http://0.0.0.0:8000`

To restart backend:
```powershell
cd C:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend
python server.py
```

### Network Configuration

- WiFi SSID: `Ayush`
- WiFi Password: `123093211`
- Network: 2.4GHz (ESP32 doesn't support 5GHz)
- Server URL: `http://192.168.208.1:8000/predict`

### Upload Firmware

To upload modified firmware:
```powershell
cd C:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend\hardware\esp32cam_uploader
& "$env:USERPROFILE\.platformio\penv\Scripts\platformio.exe" run --target upload
```

### Additional Notes

- The ESP32 chip itself is confirmed working
- WiFi module is functional
- Original firmware has been restored with camera code
- Once camera hardware is fixed, system should work end-to-end

---

**Status as of January 2, 2026:**
- ESP32 chip: ✅ Working
- WiFi: ✅ Working
- Backend: ✅ Running
- Camera: ❌ Hardware issue - needs physical repair/replacement
