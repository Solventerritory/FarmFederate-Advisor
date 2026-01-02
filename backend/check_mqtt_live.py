#!/usr/bin/env python3
"""Check for live MQTT messages from ESP32 sensors"""
import paho.mqtt.client as mqtt
import time
import json

messages_received = []

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode('utf-8')
        data = json.loads(payload)
        messages_received.append({
            'topic': msg.topic,
            'data': data,
            'timestamp': time.strftime('%H:%M:%S')
        })
        print(f"\n✓ MQTT Message Received at {time.strftime('%H:%M:%S')}")
        print(f"  Topic: {msg.topic}")
        print(f"  Data: {json.dumps(data, indent=2)}")
    except Exception as e:
        print(f"  Raw: {msg.payload.decode('utf-8')}")

def on_connect(client, userdata, flags, rc, properties=None):
    print(f"✓ Connected to MQTT broker (code: {rc})")
    client.subscribe('farmfederate/sensors/#')
    print("✓ Subscribed to: farmfederate/sensors/#")

if __name__ == "__main__":
    print("=" * 60)
    print("CHECKING FOR LIVE ESP32 SENSOR DATA")
    print("=" * 60)
    print("Connecting to MQTT broker at localhost:1883...")
    
    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    
    try:
        client.connect('localhost', 1883, 60)
        print("Listening for messages for 15 seconds...")
        print("(If your ESP32 is connected, you should see data appear)\n")
        
        client.loop_start()
        time.sleep(15)
        client.loop_stop()
        
        print("\n" + "=" * 60)
        print(f"RESULTS: {len(messages_received)} messages received")
        print("=" * 60)
        
        if messages_received:
            print("\n✓ ESP32 sensors are ACTIVELY publishing!")
            print("\nSensor Summary:")
            for msg in messages_received:
                data = msg['data']
                print(f"\n  Device: {data.get('client_id', 'unknown')}")
                if 'temperature' in data:
                    print(f"    Temperature: {data['temperature']}°C")
                if 'humidity' in data:
                    print(f"    Humidity: {data['humidity']}%")
                if 'soil_moisture' in data:
                    print(f"    Soil Moisture: {data['soil_moisture']}%")
                if 'flow_rate' in data:
                    print(f"    Flow Rate: {data['flow_rate']} L/min")
                if 'total_liters' in data:
                    print(f"    Total Flow: {data['total_liters']} L")
        else:
            print("\n⚠ No messages received from ESP32 devices!")
            print("\nTroubleshooting:")
            print("  1. Check ESP32 is powered on")
            print("  2. Verify WiFi credentials in ESP32 sketch")
            print("  3. Confirm MQTT_SERVER IP in sketch matches your PC IP")
            print("  4. Check ESP32 Serial Monitor for connection status")
            print("  5. Run: ipconfig (find your PC's IP address)")
            
    except Exception as e:
        print(f"\n✗ Error: {e}")
