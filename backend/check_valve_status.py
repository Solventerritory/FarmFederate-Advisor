#!/usr/bin/env python3
"""
Check valve and MQTT status
Subscribes to MQTT topics to verify if ESP32 is connected and responding
"""
import paho.mqtt.client as mqtt
import time
import sys

received_messages = []

def on_connect(client, userdata, flags, reason_code, properties):
    print(f"✓ Connected to MQTT broker (code: {reason_code})")
    # Subscribe to all farmfederate topics
    client.subscribe("farmfederate/#")
    print("✓ Subscribed to farmfederate/# (all topics)")
    print("\nListening for messages... (will wait 5 seconds)")
    print("If ESP32 is connected, you should see sensor data.\n")

def on_message(client, userdata, msg):
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg.topic}: {msg.payload.decode()}")
    received_messages.append((msg.topic, msg.payload.decode()))

def main():
    try:
        print("=" * 60)
        print("FarmFederate - Valve & MQTT Status Check")
        print("=" * 60)
        
        # Create MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.on_connect = on_connect
        client.on_message = on_message
        
        # Connect
        print("Connecting to MQTT broker at localhost:1883...")
        client.connect("localhost", 1883, 60)
        
        # Start loop in background
        client.loop_start()
        
        # Wait and collect messages
        time.sleep(5)
        
        # Stop loop
        client.loop_stop()
        client.disconnect()
        
        print("\n" + "=" * 60)
        print("Status Summary:")
        print("=" * 60)
        
        if len(received_messages) > 0:
            print(f"✓ Received {len(received_messages)} messages")
            print("✓ ESP32 appears to be connected and publishing")
            
            # Check for sensor data
            topics = [msg[0] for msg in received_messages]
            if any('sensors' in t for t in topics):
                print("✓ Sensor data is flowing")
            
            # Send test command
            print("\nSending test valve OPEN command...")
            test_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            test_client.connect("localhost", 1883, 60)
            test_client.publish("farmfederate/control/relay", "ON")
            time.sleep(0.5)
            test_client.disconnect()
            print("✓ Valve OPEN command sent!")
            
        else:
            print("✗ No messages received")
            print("  Possible issues:")
            print("  - ESP32 not connected to WiFi")
            print("  - ESP32 not connected to MQTT broker")
            print("  - Wrong MQTT broker IP in ESP32 code")
            print("  - ESP32 not powered on")
            print("\n  To fix:")
            print("  1. Check ESP32 Serial Monitor for connection status")
            print("  2. Verify MQTT broker IP matches your computer's IP")
            print("  3. Make sure ESP32 is on the same WiFi network")
        
        print("=" * 60)
        
    except ConnectionRefusedError:
        print("\n✗ Error: Could not connect to MQTT broker")
        print("  Start the broker: net start mosquitto")
    except Exception as e:
        print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
