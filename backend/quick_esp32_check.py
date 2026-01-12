#!/usr/bin/env python3
"""Quick ESP32 connection check"""
import paho.mqtt.client as mqtt
import time

messages = []

def on_message(client, userdata, msg):
    messages.append((msg.topic, msg.payload.decode()))
    print(f"✓ Received: {msg.topic} = {msg.payload.decode()}")

print("Checking for ESP32 messages (3 seconds)...")
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_message = on_message
client.connect("localhost", 1883, 60)
client.subscribe("farmfederate/#")
client.loop_start()
time.sleep(3)
client.loop_stop()
client.disconnect()

print(f"\nTotal messages: {len(messages)}")
if len(messages) > 0:
    print("✓ ESP32 IS CONNECTED AND PUBLISHING!")
else:
    print("✗ No messages detected - ESP32 may not be connected yet")
