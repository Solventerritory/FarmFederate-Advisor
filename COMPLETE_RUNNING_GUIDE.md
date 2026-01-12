# FarmFederate System - Complete Running Guide

## 🎯 System Overview

The FarmFederate system is now operational with the following components:

### ✅ Currently Running
1. **Backend Server** (FastAPI) - Port 8000
2. **MQTT Broker** (Mosquitto) - Port 1883  
3. **MQTT Listener** - Capturing sensor data
4. **Flutter Frontend** - Starting on Windows

### 📝 Ready for Configuration
5. **ESP32-CAM** - Enhanced firmware ready
6. **ESP32 Sensors** - Firmware ready

---

## 🚀 Quick Access

### Backend API
- **Health**: http://localhost:8000/health
- **API Docs**: http://localhost:8000/docs
- **Predict**: http://localhost:8000/predict

### PowerShell Windows
You should see 3 PowerShell windows running:
1. **Backend Server** - Shows model loading and API requests
2. **MQTT Listener** - Shows sensor data being saved
3. **Flutter App** - Shows app building and launching

---

## 📱 Using the Frontend

The Flutter app provides:

1. **Dashboard Screen**
   - Real-time sensor data visualization
   - Current crop health status
   - Device telemetry

2. **Prediction Screen**
   - Upload crop images
   - Get AI-powered crop advisory
   - View disease detection results

3. **Settings Screen**
   - Configure backend URL
   - MQTT connection settings
   - Device management

---

## 🔧 Hardware Deployment

### ESP32-CAM Setup (5 minutes)

1. **Connect Hardware**:
   ```
   ESP32-CAM → USB-to-Serial Programmer
   Connect IO0 to GND for upload mode
   ```

2. **Configure WiFi** in `backend/hardware/esp32cam_uploader/src/main.cpp`:
   ```cpp
   const char* WIFI_SSID = "YourNetwork";
   const char* WIFI_PASSWORD = "YourPassword";
   const char* SERVER_URL = "http://YOUR_IP:8000/predict";
   ```

3. **Upload Firmware**:
   ```bash
   cd backend/hardware/esp32cam_uploader
   platformio run --target upload
   platformio device monitor --baud 115200
   ```

4. **Expected Output**:
   ```
   WiFi connected! IP: 192.168.1.x
   Camera initialized
   === READY ===
   [Capture] Multi-shot session started
   ```

### ESP32 Sensor Setup (5 minutes)

1. **Wire Sensors**:
   ```
   DHT22 Data → GPIO 4
   Soil Sensor → GPIO 34
   3.3V and GND to both sensors
   ```

2. **Configure** in `backend/hardware/esp32_sensor_node/esp32_sensor_node.ino`:
   ```cpp
   const char* ssid = "YourNetwork";
   const char* pass = "YourPassword";
   const char* mqtt_server = "YOUR_IP";
   const char* client_id = "esp32_sensor_01";
   ```

3. **Upload** via Arduino IDE:
   - Select Board: "ESP32 Dev Module"
   - Select Port: Your COM port
   - Click Upload

4. **Verify Data Flow**:
   ```bash
   # In backend terminal, you should see:
   Saved sensor: checkpoints_paper\sensors\esp32_sensor_01.json
   ```

---

## 🧪 Testing the System

### Test 1: Backend Health Check
```powershell
Invoke-WebRequest http://localhost:8000/health
# Expected: {"status":"ok","device":"cpu","model_loaded":true}
```

### Test 2: MQTT Connection
```powershell
mosquitto_sub -t "farmfederate/sensors/#" -v
# Should show sensor data when ESP32 publishes
```

### Test 3: End-to-End Prediction
```powershell
# Upload a crop image via Flutter app
# Or use API directly:
curl http://localhost:8000/predict `
  -F "image=@test_crop.jpg" `
  -F "text=Check for diseases"
```

---

## 📊 Monitoring System Performance

### Check Running Services
```powershell
# Backend server
Get-Process python | Select-Object Id, ProcessName

# MQTT broker
Get-Service mosquitto

# Flutter app
Get-Process | Where-Object {$_.MainWindowTitle -like "*FarmFederate*"}
```

### View Logs
- **Backend**: Check PowerShell window #1
- **MQTT Listener**: Check PowerShell window #2
- **Flutter**: Check PowerShell window #3

### View Sensor Data
```powershell
# List recent sensor files
Get-ChildItem backend\checkpoints_paper\sensors\*.json | 
  Sort-Object LastWriteTime -Descending

# View latest sensor data
Get-Content backend\checkpoints_paper\sensors\esp32_sensor_01.json | ConvertFrom-Json | Format-List
```

---

## 🔬 Research Features (Federated Learning)

### Run Federated Training

Once you have data from multiple ESP32 devices:

```bash
cd backend

# Basic federated learning
python train_fed_multimodal.py --rounds 10 --clients 5

# With differential privacy (research feature)
python train_fed_multimodal.py --rounds 10 --clients 5 --use-dp --noise-scale 0.01

# With Byzantine-robust aggregation (research feature)
python train_fed_multimodal.py --rounds 10 --clients 5 --aggregation krum --num-byzantine 1

# With gradient compression (research feature)
python train_fed_multimodal.py --rounds 10 --clients 5 --compression-ratio 0.1

# Full enhanced training (all research features)
python train_fed_multimodal.py \
  --rounds 10 \
  --clients 5 \
  --strategy importance \
  --use-dp \
  --noise-scale 0.01 \
  --compression-ratio 0.1 \
  --use-cross-attention
```

### Monitor Training
- Metrics saved to: `metrics/federated_metrics.json`
- Checkpoints saved to: `checkpoints_multimodal/`
- View progress in terminal

---

## 🛠️ Troubleshooting

### Backend Not Responding
```powershell
# Restart backend
# Close PowerShell window #1, then:
cd backend
python server.py
```

### MQTT Not Receiving Data
```powershell
# Check Mosquitto service
Get-Service mosquitto

# Restart if needed
Restart-Service mosquitto

# Restart listener
# Close PowerShell window #2, then:
cd backend
python mqtt_listener.py
```

### Flutter App Not Starting
```powershell
# Check Flutter installation
flutter doctor

# Rebuild and run
cd frontend
flutter clean
flutter pub get
flutter run -d windows
```

### ESP32 Connection Issues
```
1. Check WiFi credentials in firmware
2. Verify network is 2.4GHz (ESP32 doesn't support 5GHz)
3. Check backend IP address is correct
4. Test with: curl http://YOUR_IP:8000/health
5. Monitor serial output at 115200 baud
```

---

## 📖 Documentation Index

1. **[README.md](README.md)** - Project overview and features
2. **[QUICK_START.md](QUICK_START.md)** - Fast setup guide
3. **[RESEARCH_PAPER_IMPLEMENTATION.md](RESEARCH_PAPER_IMPLEMENTATION.md)** - Research features detail
4. **[HARDWARE_SETUP_GUIDE.md](HARDWARE_SETUP_GUIDE.md)** - Complete hardware instructions
5. **[SYSTEM_STATUS_NOW.md](SYSTEM_STATUS_NOW.md)** - Current system status
6. **[ESP32_TROUBLESHOOTING.md](ESP32_TROUBLESHOOTING.md)** - Hardware debugging

---

## 🎓 Research Paper Alignment

This implementation includes all features from the research paper:

### Federated Learning Core
- ✅ FedAvg aggregation with weighted averaging
- ✅ Differential privacy (ε, δ)-DP with Gaussian noise
- ✅ Byzantine-robust aggregation (Krum algorithm)
- ✅ Adaptive client sampling (importance, loss-weighted, staleness)
- ✅ Gradient compression (Top-K sparsification)
- ✅ Comprehensive metrics tracking

### Multimodal Model Enhancements
- ✅ Cross-modal attention (text-to-image, image-to-text)
- ✅ Uncertainty estimation (Monte Carlo dropout)
- ✅ Attention visualization for explainability
- ✅ Deeper architecture (512-d projections, 4x fusion)

### Hardware Intelligence
- ✅ Multi-shot capture with quality assessment
- ✅ Adaptive capture intervals based on detections
- ✅ Exponential backoff retry logic
- ✅ Comprehensive telemetry system
- ✅ Device health monitoring (RSSI, heap, quality)

---

## 🎉 Success Indicators

Your system is working correctly if:

1. ✅ Backend responds to http://localhost:8000/health
2. ✅ MQTT listener shows "mqtt connected 0"
3. ✅ Flutter app window appears
4. ✅ ESP32-CAM serial shows "=== READY ==="
5. ✅ ESP32 sensor data appears in `checkpoints_paper/sensors/`
6. ✅ Predictions return JSON with crop advisory

---

## 📞 Next Steps

1. **Deploy Hardware**: Flash ESP32 devices with configuration
2. **Collect Data**: Let system gather sensor and image data
3. **Run Training**: Execute federated learning with collected data
4. **Monitor Results**: Track metrics and model performance
5. **Iterate**: Adjust parameters and retrain as needed

---

**System is operational and ready for research experiments! 🚀**

For detailed instructions, refer to specific documentation files listed above.
