#!/usr/bin/env python3
"""
ESP32 Sensor Connection Diagnostic Tool
Checks all sensors connected to ESP32 and validates their data
"""
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime
from collections import defaultdict

# Expected sensors from esp32_complete_sensors.ino
EXPECTED_SENSORS = {
    'temperature': {'unit': '°C', 'range': (-10, 50), 'pin': 'GPIO 4 (DHT22)'},
    'humidity': {'unit': '%', 'range': (0, 100), 'pin': 'GPIO 4 (DHT22)'},
    'soil_moisture': {'unit': '%', 'range': (0, 100), 'pin': 'GPIO 34 (Analog)'},
    'flow_rate': {'unit': 'L/min', 'range': (0, 100), 'pin': 'GPIO 18 (YF-S201)'},
    'relay_state': {'unit': 'ON/OFF', 'range': None, 'pin': 'GPIO 5'},
}

sensor_data = defaultdict(list)
message_count = 0
esp32_connected = False
last_message_time = None

def validate_sensor_value(sensor, value):
    """Check if sensor value is within expected range"""
    if sensor not in EXPECTED_SENSORS:
        return True, "Unknown sensor"
    
    config = EXPECTED_SENSORS[sensor]
    if config['range'] is None:
        return True, "OK"
    
    min_val, max_val = config['range']
    
    try:
        num_val = float(value) if value != "ON" and value != "OFF" else 0
        if min_val <= num_val <= max_val:
            return True, "OK"
        else:
            return False, f"Out of range ({min_val}-{max_val})"
    except:
        return True, "Non-numeric"

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"\n✓ Connected to MQTT broker")
    client.subscribe("farmfederate/sensors/#")
    print("✓ Subscribed to sensor topics")
    print("\nListening for ESP32 sensor data...\n")

def on_message(client, userdata, msg):
    global message_count, esp32_connected, last_message_time
    
    try:
        payload = msg.payload.decode()
        data = json.loads(payload)
        
        message_count += 1
        esp32_connected = True
        last_message_time = datetime.now()
        
        # Store sensor values
        for sensor, value in data.items():
            if sensor in EXPECTED_SENSORS:
                sensor_data[sensor].append(value)
        
        # Print incoming data
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"[{timestamp}] Message #{message_count} from {data.get('client_id', 'unknown')}")
        
    except json.JSONDecodeError:
        print(f"⚠ Received non-JSON message: {msg.payload.decode()}")
    except Exception as e:
        print(f"✗ Error processing message: {e}")

def print_diagnostic_report():
    """Print comprehensive sensor diagnostic report"""
    print("\n" + "=" * 80)
    print("ESP32 SENSOR CONNECTION DIAGNOSTIC REPORT")
    print("=" * 80)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Duration: 10 seconds")
    print(f"Messages Received: {message_count}")
    print("=" * 80)
    
    if not esp32_connected:
        print("\n❌ ESP32 NOT CONNECTED")
        print("\nPossible Issues:")
        print("  1. ESP32 not powered on")
        print("  2. WiFi credentials incorrect in firmware")
        print("  3. MQTT broker IP address incorrect")
        print("  4. ESP32 not on same network as this computer")
        print("  5. Firewall blocking MQTT port 1883")
        print("\nTroubleshooting Steps:")
        print("  • Open Arduino Serial Monitor (115200 baud)")
        print("  • Check for WiFi connection messages")
        print("  • Verify MQTT_SERVER IP in esp32_complete_sensors.ino")
        print("  • Ensure MQTT broker is running: net start mosquitto")
        print("=" * 80)
        return
    
    print(f"\n✅ ESP32 CONNECTED")
    print(f"Last Message: {last_message_time.strftime('%H:%M:%S')}")
    print(f"Message Rate: {message_count / 10:.1f} messages/second")
    
    print("\n" + "-" * 80)
    print("SENSOR STATUS")
    print("-" * 80)
    print(f"{'Sensor':<20} {'Pin':<20} {'Status':<15} {'Value':<15} {'Unit':<10}")
    print("-" * 80)
    
    all_sensors_ok = True
    
    for sensor, config in EXPECTED_SENSORS.items():
        if sensor in sensor_data and len(sensor_data[sensor]) > 0:
            # Get latest value
            latest_value = sensor_data[sensor][-1]
            is_valid, validation_msg = validate_sensor_value(sensor, latest_value)
            
            status_icon = "✅" if is_valid else "⚠️"
            status_text = "CONNECTED" if is_valid else "WARNING"
            
            # Format value
            if isinstance(latest_value, (int, float)):
                value_str = f"{latest_value:.1f}"
            else:
                value_str = str(latest_value)
            
            print(f"{status_icon} {sensor:<18} {config['pin']:<20} {status_text:<15} {value_str:<15} {config['unit']:<10}")
            
            if not is_valid:
                print(f"   └─ {validation_msg}")
                all_sensors_ok = False
        else:
            print(f"❌ {sensor:<18} {config['pin']:<20} {'NOT DETECTED':<15} {'N/A':<15} {config['unit']:<10}")
            all_sensors_ok = False
    
    print("-" * 80)
    
    # Detailed sensor readings
    if sensor_data:
        print("\nDETAILED SENSOR READINGS (Last 3 values)")
        print("-" * 80)
        for sensor, values in sensor_data.items():
            if sensor in EXPECTED_SENSORS:
                recent_values = values[-3:] if len(values) >= 3 else values
                print(f"{sensor}: {recent_values}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    connected_count = len([s for s in EXPECTED_SENSORS if s in sensor_data and len(sensor_data[s]) > 0])
    total_count = len(EXPECTED_SENSORS)
    
    print(f"Sensors Connected: {connected_count}/{total_count}")
    
    if all_sensors_ok and connected_count == total_count:
        print("✅ ALL SENSORS WORKING CORRECTLY")
    elif connected_count > 0:
        print("⚠️  SOME SENSORS CONNECTED")
        missing = [s for s in EXPECTED_SENSORS if s not in sensor_data or len(sensor_data[s]) == 0]
        if missing:
            print(f"\nMissing Sensors: {', '.join(missing)}")
            print("\nCheck:")
            print("  • Wire connections to ESP32")
            print("  • Sensor power supply")
            print("  • Correct GPIO pins in firmware")
    else:
        print("❌ NO SENSORS DETECTED")
    
    print("=" * 80)

def main():
    print("=" * 80)
    print("ESP32 SENSOR CONNECTION DIAGNOSTIC")
    print("=" * 80)
    print("\nExpected Sensors:")
    for sensor, config in EXPECTED_SENSORS.items():
        print(f"  • {sensor:<20} on {config['pin']}")
    
    print("\nStarting diagnostic test (10 seconds)...")
    print("-" * 80)
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect("localhost", 1883, 60)
        client.loop_start()
        
        # Wait 10 seconds for data
        time.sleep(10)
        
        client.loop_stop()
        client.disconnect()
        
        # Generate report
        print_diagnostic_report()
        
    except ConnectionRefusedError:
        print("\n✗ Error: Could not connect to MQTT broker at localhost:1883")
        print("  Start broker with: net start mosquitto")
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
