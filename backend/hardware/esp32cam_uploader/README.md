# ESP32-CAM Leaf Disease Detection System

This firmware turns your ESP32-CAM into an intelligent plant health monitoring device that captures leaf images and sends them to the FarmFederate backend for AI-powered disease detection.

## 📋 Features

- ✅ **Automatic Image Capture**: Captures leaf images every 60 seconds (configurable)
- ✅ **Manual Trigger**: Press button to capture on demand
- ✅ **Flash Control**: Automatic flash for optimal lighting
- ✅ **High Quality Images**: UXGA resolution (1600x1200) for detailed analysis
- ✅ **Multimodal AI Analysis**: Uses RoBERTa + ViT model for disease detection
- ✅ **Retry Logic**: Automatic retry on upload failure
- ✅ **Real-time Results**: Displays disease predictions via serial monitor
- ✅ **WiFi Auto-reconnect**: Maintains connection automatically

## 🛠️ Hardware Requirements

### Required Components
1. **ESP32-CAM Module** (AI-Thinker or compatible)
   - Built-in camera (OV2640)
   - PSRAM support for high-resolution images
   
2. **FTDI Programmer** or **ESP32-CAM-MB Programmer Board**
   - For uploading firmware via USB
   
3. **Power Supply**
   - 5V via USB or external power
   - Minimum 500mA current capacity

### Optional Components
- **Push Button**: Connect between GPIO 13 and GND for manual trigger
- **External Antenna**: For better WiFi range (if supported by your module)
- **Enclosure**: Weather-resistant case for outdoor deployment

## 📦 Hardware Connections

### ESP32-CAM Pinout (AI-Thinker)
```
┌─────────────────────────┐
│  ESP32-CAM (Top View)   │
│                         │
│  [CAMERA]  [LED Flash]  │
│                         │
│  GND  │  5V             │
│  U0R  │  U0T            │
│  GPIO │  GPIO           │
│  ...  │  ...            │
└─────────────────────────┘
```

### Programming Connections (FTDI)
```
FTDI → ESP32-CAM
──────────────────
VCC  → 5V
GND  → GND
TX   → U0R (GPIO 3)
RX   → U0T (GPIO 1)

For Upload Mode:
GPIO 0 → GND (connect before power on)
```

### Optional Button Connection
```
GPIO 13 ──┬── Button ──┬── GND
          └── 10kΩ ────┘
          (Pull-up)
```

## 💻 Software Setup

### Prerequisites
1. **PlatformIO IDE** (VS Code extension) or **Arduino IDE**
2. **Python 3.8+** (for backend server)
3. **Backend Server Running** (FastAPI server on port 8000)

### Installation Steps

#### Option 1: Using PlatformIO (Recommended)

1. **Install PlatformIO**
   ```bash
   # In VS Code: Install "PlatformIO IDE" extension
   ```

2. **Open Project**
   ```bash
   cd backend/hardware/esp32cam_uploader
   code .
   ```

3. **Configure Settings**
   Edit `esp32cam_upload.ino`:
   ```cpp
   const char* WIFI_SSID = "YourWiFiName";
   const char* WIFI_PASSWORD = "YourWiFiPassword";
   const char* SERVER_URL = "http://192.168.1.100:8000/predict";
   ```

4. **Update COM Port**
   Edit `platformio.ini`:
   ```ini
   upload_port = COM7  ; Change to your port
   ```

5. **Upload Firmware**
   - Connect ESP32-CAM via programmer
   - Connect GPIO 0 to GND (for programming mode)
   - Press upload button in PlatformIO
   - Disconnect GPIO 0 after upload
   - Press RESET button on ESP32-CAM

#### Option 2: Using Arduino IDE

1. **Install ESP32 Board Support**
   - Open Arduino IDE
   - File → Preferences
   - Add URL: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Tools → Board → Boards Manager
   - Install "esp32" by Espressif Systems

2. **Install Required Libraries**
   - Sketch → Include Library → Manage Libraries
   - Install: **ArduinoJson** by Benoit Blanchon (v6.21.3 or newer)

3. **Configure Board**
   - Tools → Board → ESP32 Arduino → AI Thinker ESP32-CAM
   - Tools → Port → Select your COM port
   - Tools → Partition Scheme → Huge APP (3MB)

4. **Upload Firmware**
   - Open `esp32cam_upload.ino`
   - Update WiFi and server settings
   - Connect GPIO 0 to GND
   - Click Upload
   - Disconnect GPIO 0 and press RESET

## 🔧 Configuration Options

### Timing Settings
```cpp
#define CAPTURE_INTERVAL  60000    // Capture every 60 seconds
#define RETRY_DELAY       5000     // Wait 5 sec before retry
#define MAX_RETRIES       3        // Max upload attempts
```

### Camera Quality Settings
```cpp
config.frame_size = FRAMESIZE_UXGA;  // Resolution
config.jpeg_quality = 10;             // Quality (0-63, lower=better)
```

Available frame sizes:
- `FRAMESIZE_UXGA` - 1600x1200 (best quality)
- `FRAMESIZE_SXGA` - 1280x1024
- `FRAMESIZE_XGA` - 1024x768
- `FRAMESIZE_SVGA` - 800x600
- `FRAMESIZE_VGA` - 640x480

### Network Settings
```cpp
const char* WIFI_SSID = "YourNetwork";
const char* WIFI_PASSWORD = "YourPassword";
const char* SERVER_URL = "http://192.168.1.100:8000/predict";
const char* DEVICE_ID = "esp32cam_01";  // Unique ID
```

## 🚀 Usage

### First Boot
1. Power on ESP32-CAM
2. Open Serial Monitor (115200 baud)
3. Watch for successful WiFi connection
4. Wait for "System initialized successfully!" message

### Automatic Mode
- Camera automatically captures images every 60 seconds
- Images uploaded to backend for analysis
- Results displayed in serial monitor

### Manual Mode
- Press button connected to GPIO 13
- Immediate image capture and upload
- Useful for testing or on-demand analysis

## 📊 Reading Results

### Serial Monitor Output
```
========================================
[CAPTURE #1] Starting image capture...
[OK] Image captured: 89234 bytes

[UPLOAD] Attempt 1/3
[POST] Uploading 89567 bytes to http://192.168.1.100:8000/predict
[RESPONSE] HTTP Code: 200
[SUCCESS] Upload successful!

========== ANALYSIS RESULTS ==========
[DETECTED] 2 disease(s) detected:
  • Leaf Blight (87.3% confidence)
  • Fungal Infection (72.1% confidence)

[RECOMMENDATIONS]:
  • Apply fungicide treatment
  • Increase air circulation
  • Remove affected leaves

[ALL SCORES]:
  ✓ Leaf Blight: 87.3% (threshold: 50.0%)
  ✓ Fungal Infection: 72.1% (threshold: 50.0%)
    Bacterial Spot: 23.4% (threshold: 50.0%)
    Healthy: 12.1% (threshold: 50.0%)
======================================
```

## 🐛 Troubleshooting

### Camera Init Failed
**Error**: `Camera init failed with error 0x20001`
**Solution**: 
- Check camera module connection
- Verify power supply is adequate (500mA minimum)
- Try different USB cable/power source

### WiFi Connection Failed
**Error**: WiFi doesn't connect
**Solution**:
- Verify SSID and password are correct
- Check 2.4GHz WiFi (ESP32 doesn't support 5GHz)
- Move closer to router
- Check router settings (some block IoT devices)

### Upload Failed
**Error**: HTTP POST failed or timeout
**Solution**:
- Verify backend server is running (`netstat -ano | findstr :8000`)
- Check SERVER_URL matches backend IP
- Ensure firewall allows port 8000
- Test with `curl http://YOUR_IP:8000/health`

### Image Quality Issues
**Problem**: Dark or blurry images
**Solution**:
- Adjust `s->set_ae_level()` for exposure
- Increase flash duration (delay after flashControl)
- Change `jpeg_quality` to lower number
- Ensure adequate lighting

### Memory Issues
**Error**: Failed to allocate memory
**Solution**:
- Reduce `FRAMESIZE_UXGA` to `FRAMESIZE_SVGA`
- Increase `jpeg_quality` number (more compression)
- Ensure PSRAM is enabled in platformio.ini

## 🔌 Backend Integration

The ESP32-CAM sends images to the backend `/predict` endpoint using multipart/form-data:

### Request Format
```http
POST /predict HTTP/1.1
Content-Type: multipart/form-data; boundary=----WebKitFormBoundary

------WebKitFormBoundary
Content-Disposition: form-data; name="image"; filename="leaf.jpg"
Content-Type: image/jpeg

[JPEG IMAGE DATA]
------WebKitFormBoundary
Content-Disposition: form-data; name="text"

Leaf image captured by esp32cam_01
------WebKitFormBoundary
Content-Disposition: form-data; name="client_id"

esp32cam_01
------WebKitFormBoundary--
```

### Response Format
```json
{
  "active_labels": [
    {
      "label": "Leaf Blight",
      "prob": 0.873,
      "threshold": 0.5
    }
  ],
  "all_scores": [...],
  "advice": [
    "Apply fungicide treatment",
    "Increase air circulation"
  ]
}
```

## 📁 Project Structure
```
esp32cam_uploader/
├── esp32cam_upload/
│   └── esp32cam_upload.ino    # Main firmware
├── platformio.ini              # PlatformIO config
└── README.md                   # This file
```

## 🔐 Security Considerations

### Production Deployment
- ⚠️ **Don't hardcode WiFi passwords** - Use WPS or provisioning
- ⚠️ **Use HTTPS** - Enable SSL/TLS for production
- ⚠️ **Secure backend** - Add authentication/API keys
- ⚠️ **Network isolation** - Use dedicated IoT VLAN

### Example with API Key
```cpp
http.addHeader("X-API-Key", "your-secret-key");
```

## 📈 Performance Metrics

### Typical Performance
- **Image Capture**: ~500ms
- **WiFi Upload**: 2-5 seconds (depends on network)
- **Total Cycle**: ~6-8 seconds per capture
- **Power Consumption**: 
  - Idle: ~180mA
  - Capture + Flash: ~350mA
  - WiFi Upload: ~250mA

### Storage Requirements
- **Image Size**: 50-150KB (UXGA, quality 10)
- **Flash Memory**: ~1.5MB used (Arduino core + libraries)
- **Available**: ~1.5MB for OTA updates

## 🆘 Support

### Common Questions

**Q: Can I use multiple ESP32-CAMs?**  
A: Yes! Change `DEVICE_ID` to unique values (esp32cam_01, esp32cam_02, etc.)

**Q: How do I update firmware OTA?**  
A: Requires OTA library implementation (not included yet)

**Q: Can it work offline?**  
A: No, requires WiFi connection to backend for AI analysis

**Q: Battery powered operation?**  
A: Possible with deep sleep mode (requires code modification)

### Getting Help
1. Check serial monitor for error messages
2. Test backend with curl: `curl http://YOUR_IP:8000/health`
3. Verify camera with test capture
4. Check WiFi signal strength in serial output

## 📝 License
Part of the FarmFederate-Advisor project

## 🙏 Credits
- ESP32-CAM Arduino Core: Espressif Systems
- Camera Library: ESP32 Arduino Community
- ArduinoJson: Benoit Blanchon
