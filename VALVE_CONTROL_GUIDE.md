# Solenoid Valve Control Guide

## Quick Commands

### Open Valve (Water ON)
```bash
python backend/valve_controller.py open
```

### Close Valve (Water OFF)
```bash
python backend/valve_controller.py close
```

### Check Status
```bash
python backend/valve_controller.py status
```

## How It Works

1. **MQTT Communication**: Commands are sent via MQTT broker to topic `farmfederate/control/relay`
2. **ESP32 Receives**: ESP32 subscribed to this topic receives ON/OFF commands
3. **Relay Activation**: ESP32 triggers relay on GPIO 5 or GPIO 26
4. **Solenoid Opens**: Relay controls the solenoid valve (230V AC irrigation valve on GPIO 27)

## Current Status

✅ **VALVE IS NOW OPEN** - Water flow enabled

### MQTT Broker Status
- Running: ✓ localhost:1883
- Topic: farmfederate/control/relay
- Last command: ON (OPEN)

### Hardware Configuration
- **Relay Pin**: GPIO 5 or GPIO 26 (depends on ESP32 firmware)
- **Solenoid Valve Pin**: GPIO 27 (230V AC valve)
- **Control Method**: MQTT → ESP32 → Relay → Solenoid

## Troubleshooting

### If Valve Doesn't Respond:

1. **Check ESP32 Connection**
   ```bash
   python backend/check_valve_status.py
   ```

2. **Verify MQTT Broker**
   ```bash
   net start mosquitto
   ```

3. **Check ESP32 Serial Monitor**
   - Open Arduino IDE
   - Select correct COM port
   - Set baud rate to 115200
   - Look for connection messages

4. **Test Manual MQTT**
   ```bash
   python backend/hardware/backend_integration/publish_cmd.py
   ```

### ESP32 Not Connected?

The command is still sent and logged. The valve will activate when:
- ESP32 powers on
- ESP32 connects to WiFi
- ESP32 connects to MQTT broker
- ESP32 receives the queued command

## Alternative Control Methods

### Direct Python Command
```python
import paho.mqtt.client as mqtt
c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
c.connect('localhost', 1883, 60)
c.publish('farmfederate/control/relay', 'ON')  # Open valve
c.disconnect()
```

### Via Backend API (if server.py running)
```bash
curl -X POST http://localhost:8000/control/solenoid_valve -H "Content-Type: application/json" -d '{"state": true}'
```

### Via Flutter App
- Open FarmFederate app
- Go to Dashboard
- Toggle "Solenoid Valve" switch

## Command Log

All valve commands are logged to: `backend/valve_control_log.txt`

View recent commands:
```bash
python backend/valve_controller.py status
```

## Safety Notes

⚠️ **Important:**
- Solenoid valve is 230V AC - Handle with care
- Don't leave valve open unattended
- Check water pressure before opening
- Ensure proper drainage
- Monitor flow sensor readings

## Files

- `backend/valve_controller.py` - Main control script (recommended)
- `backend/control_solenoid.py` - Simple control script
- `backend/check_valve_status.py` - Status checker
- `backend/hardware/backend_integration/publish_cmd.py` - Direct MQTT test

---

**Last Updated**: January 3, 2026
**Valve State**: OPEN (Water ON)
