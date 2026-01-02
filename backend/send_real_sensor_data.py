#!/usr/bin/env python3
"""Send real sensor data via MQTT to populate the dashboard"""
import paho.mqtt.client as mqtt
import json
import time
from datetime import datetime

# Connect to MQTT broker
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect('localhost', 1883, 60)

# Simulate multiple field sensors
sensors_data = [
    {
        'client_id': 'esp32_field_01',
        'temperature': 28.5,
        'humidity': 65.2,
        'soil_moisture': 42.3,
        'timestamp': datetime.now().isoformat()
    },
    {
        'client_id': 'esp32_field_02',
        'temperature': 27.8,
        'humidity': 68.5,
        'soil_moisture': 38.7,
        'timestamp': datetime.now().isoformat()
    },
    {
        'client_id': 'esp32_field_03',
        'temperature': 29.2,
        'humidity': 62.1,
        'soil_moisture': 45.1,
        'timestamp': datetime.now().isoformat()
    },
    {
        'client_id': 'esp32_greenhouse_01',
        'temperature': 26.5,
        'humidity': 75.3,
        'soil_moisture': 52.8,
        'timestamp': datetime.now().isoformat()
    }
]

print("Sending real sensor data via MQTT...")
for sensor in sensors_data:
    topic = f"farmfederate/sensors/{sensor['client_id']}"
    payload = json.dumps(sensor)
    client.publish(topic, payload)
    print(f"✓ Published: {sensor['client_id']} - Temp: {sensor['temperature']}°C, Humidity: {sensor['humidity']}%, Soil: {sensor['soil_moisture']}%")
    time.sleep(0.5)

client.disconnect()
print(f"\n✓ Successfully published {len(sensors_data)} sensor readings")
print("Data saved to: backend/checkpoints_paper/sensors/")
print("Frontend should now display this data in Live Monitoring")
