# FarmFederate-Advisor Running Setup

## Current Status (Real Sensor Data Mode)

### ✅ Backend Server
- **Status**: Running in separate PowerShell window
- **URL**: http://0.0.0.0:8000
- **Model**: Multimodal classifier loaded from `checkpoints/global_central.pt`
- **Endpoints**:
  - `/docs` - API documentation
  - `/predict` - Multimodal prediction endpoint
  - `/api/sensor_upload` - Sensor data upload
  - `/api/image_upload` - Image upload

### ✅ MQTT Broker
- **Status**: Running (Mosquitto service)
- **Host**: localhost
- **Port**: 1883
- **Topics**: 
  - `farmfederate/sensors/#` - Real sensor data from devices

### ✅ MQTT Listener
- **Status**: Running in separate PowerShell window
- **Function**: Captures real sensor data from MQTT broker
- **Storage**: Saves to `backend/checkpoints_paper/sensors/<client_id>.json`
- **Script**: `backend/mqtt_listener.py`





### ❌ Frontend (Flutter)
- **Status**: Not running (Flutter not installed/configured)
- **Location**: `frontend/`
- **Requirements**: 
  - Install Flutter SDK
  - Run `flutter pub get`
  - Configure backend URL in services
  - Run `flutter run` for mobile or web

### 🚫 Test/Dummy Data Scripts (DISABLED for Real Sensor Mode)

The following scripts should **NOT** be used when collecting real sensor data:
- ❌ `backend/backend_integration/send_test_sensor.py` - Sends fake sensor data
- ❌ `backend/backend_integration/upload_test_image.py` - Uploads test images

### 📊 Real Data Flow

```
Sensors → WiFi → MQTT Broker (localhost:1883) 
                            ↓
                    MQTT Listener (mqtt_listener.py)
                            ↓
                Saves to: checkpoints_paper/sensors/<device_id>.json
                            ↓
                    Backend Server reads sensor data
                            ↓
                    Combined with images for prediction
```

### 🔄 How to Verify Real Data Collection

1. **Check MQTT Listener Window**:
   - Should show: `mqtt connected 0`
   - Should show: `Saved sensor: checkpoints_paper\sensors\<device_id>.json`

2. **Check Sensor Data Files**:
   ```powershell
   Get-ChildItem backend\checkpoints_paper\ingest\sensors\*.json | Sort-Object LastWriteTime -Descending | Select-Object -First 5
   ```

3. **Test MQTT Publishing Manually** (for debugging):
   ```powershell
   # Use your preferred publisher or test script to publish messages to the MQTT broker
   ```

### 🛠️ Troubleshooting

**Backend not responding?**
- Check the PowerShell window for errors
- Verify model checkpoint exists: `backend/checkpoints/global_central.pt`
- Test with: `curl http://localhost:8000/docs -UseBasicParsing`

**MQTT not receiving data?**
- Verify Mosquitto service: `Get-Service mosquitto`
- Check device serial output or logs for connection errors
- Test with: `mosquitto_sub -t "farmfederate/sensors/#"`

**No sensor data files?**
- Ensure MQTT listener is running
- Check devices are publishing (view serial monitor)
- Verify device network credentials and configuration
- Check MQTT broker IP address matches

### 🚀 Next Steps

1. **Deploy Real Sensors**:
   - Flash firmware to devices (if applicable)
   - Deploy in field with sensors attached
   - Monitor MQTT listener for incoming data

2. **Configure Frontend** (if needed):
   - Install Flutter SDK
   - Update backend URL configuration
   - Run on mobile device or web

3. **Monitor and Validate**:
   - Check real-time sensor data collection
   - Verify predictions with actual field conditions
   - Adjust thresholds as needed

### 📝 Notes

- Backend runs on CPU by default (can enable CUDA if GPU available)
- Model updates require restarting backend server
- Sensor data stored with timestamp in filename for historical tracking
- MQTT listener creates directories automatically if they don't exist
