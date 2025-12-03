# Hardware Integration for FarmFederate-Advisor

This folder contains sketches and helper scripts to integrate devices (ESP32, ESP32-CAM, Raspberry Pi camera) with the backend.

## Quick start
1. Edit credentials in the ESP32 sketches:
   - `WIFI_SSID`, `WIFI_PASS`, `MQTT_SERVER`, `BACKEND_URL`, `DEVICE_ID`.
2. Flash sketches using Arduino IDE / PlatformIO.
3. Start Mosquitto (optional) for local MQTT:
   - `docker-compose up -d` in `hardware/mqtt/`
4. Start backend server (leave your model & backend running).
5. Use the Python helpers in `hardware/backend_integration/` to test MQTT and HTTP endpoints.

## Endpoints expected on backend
- `POST /api/sensor_upload` : accepts JSON `device_id,temp,hum,soil,flow_lpm,ds_temp`
- `POST /api/image_upload`  : accepts multipart form with `device_id` and `image` field
- MQTT topic `farm/<device_id>/commands` : device listens for JSON commands like `{"cmd":"RELAY_ON"}`

Adapt backend `server.py` routes to these endpoints if necessary.
