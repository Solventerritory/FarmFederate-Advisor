#!/usr/bin/env python3
"""Send simulated ESP32 sensor data to test the system"""
import json
import time
import random
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883

# Simulated ESP32 device with all sensors including flow meter
def generate_sensor_data():
    return {
        "client_id": "esp32_usb_01",
        "temperature": round(random.uniform(25.0, 30.0), 1),
        "humidity": round(random.uniform(60.0, 75.0), 1),
        "soil_moisture": round(random.uniform(35.0, 55.0), 1),
        "flow_rate": round(random.uniform(0.5, 2.5), 2),  # Flow meter data!
        "total_liters": round(random.uniform(10.0, 50.0), 2),
        "relay_state": random.choice(["ON", "OFF"]),
        "uptime_sec": int(time.time()),
        "timestamp": datetime.now().isoformat()
    }

def main():
    print("=" * 60)
    print("SIMULATING ESP32 SENSOR DATA (Until you upload sketch)")
    print("=" * 60)
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("Publishing data every 5 seconds...")
    print("=" * 60)
    
    try:
        # Connect to MQTT
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        print("✓ Connected to MQTT broker\n")
        
        count = 0
        while True:
            count += 1
            data = generate_sensor_data()
            
            # Publish to MQTT
            topic = f"farmfederate/sensors/{data['client_id']}"
            payload = json.dumps(data)
            client.publish(topic, payload)
            
            print(f"\n📡 Message #{count} - {datetime.now().strftime('%H:%M:%S')}")
            print(f"🌡️  Temperature: {data['temperature']}°C")
            print(f"💧 Humidity: {data['humidity']}%")
            print(f"🌱 Soil Moisture: {data['soil_moisture']}%")
            print(f"🌊 Flow Rate: {data['flow_rate']} L/min ← Flow Meter!")
            print(f"📊 Total Flow: {data['total_liters']} L")
            print(f"🔌 Relay: {data['relay_state']}")
            print(f"✓ Published to: {topic}")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n" + "=" * 60)
        print("Stopped simulation")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        if 'client' in locals():
            client.loop_stop()
            client.disconnect()
            print("✓ MQTT disconnected")

if __name__ == "__main__":
    main()
