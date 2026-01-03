#!/usr/bin/env python3
"""
Real-time MQTT Monitor
Shows live MQTT traffic and ESP32 activity
"""
import paho.mqtt.client as mqtt
import time
from datetime import datetime
import json

def on_connect(client, userdata, flags, reason_code, properties):
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n[{timestamp}] ✓ Connected to MQTT broker")
    print("Subscribing to all farmfederate topics...")
    client.subscribe("farmfederate/#")
    print("Listening for messages (Ctrl+C to stop)...\n")
    print("-" * 70)

def on_message(client, userdata, msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    topic = msg.topic
    
    try:
        # Try to parse as JSON for prettier output
        payload = msg.payload.decode()
        try:
            data = json.loads(payload)
            print(f"[{timestamp}] 📡 {topic}")
            for key, value in data.items():
                print(f"    {key}: {value}")
        except:
            print(f"[{timestamp}] 📡 {topic}: {payload}")
    except:
        print(f"[{timestamp}] 📡 {topic}: <binary data>")
    
    print("-" * 70)

def main():
    print("=" * 70)
    print("FarmFederate - Real-time MQTT Monitor")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Broker: localhost:1883")
    print("Topics: farmfederate/#")
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect("localhost", 1883, 60)
        client.loop_forever()
        
    except KeyboardInterrupt:
        print("\n\n✓ Monitor stopped by user")
    except ConnectionRefusedError:
        print("\n✗ Could not connect to MQTT broker at localhost:1883")
        print("  Start broker: net start mosquitto")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
