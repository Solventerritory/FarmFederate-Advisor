#!/usr/bin/env python3
"""
Force Valve Open - Multiple Methods
Sends valve open command repeatedly to ensure ESP32 receives it
"""
import paho.mqtt.client as mqtt
import time

def send_valve_command():
    print("=" * 70)
    print("FORCING VALVE OPEN - Multiple Attempts")
    print("=" * 70)
    
    commands_sent = 0
    
    try:
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect("localhost", 1883, 60)
        
        # Send multiple commands with different variations
        topics_and_commands = [
            ("farmfederate/control/relay", "ON"),
            ("farmfederate/control/relay", "1"),
            ("farmfederate/control/solenoid_valve", "ON"),
            ("farmfederate/control/solenoid_valve", "1"),
            ("farmfederate/control/valve", "ON"),
        ]
        
        for topic, cmd in topics_and_commands:
            result = client.publish(topic, cmd, qos=1, retain=True)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"✓ Sent: {topic} = {cmd} (retained)")
                commands_sent += 1
            time.sleep(0.2)
        
        # Wait for message delivery
        time.sleep(1)
        client.disconnect()
        
        print("\n" + "=" * 70)
        print(f"✓ Sent {commands_sent} commands to MQTT broker")
        print("=" * 70)
        print("\nCommands are RETAINED - ESP32 will receive them when it connects!")
        print("\nWhat this means:")
        print("  • Commands are stored in MQTT broker")
        print("  • ESP32 will get them immediately upon connection")
        print("  • Valve will open as soon as ESP32 is ready")
        print("\nESP32 Status Check:")
        print("  1. Open Arduino Serial Monitor (115200 baud)")
        print("  2. Look for 'MQTT connected' message")
        print("  3. Should see 'Relay/Solenoid ACTIVATED (Water ON)'")
        print("\nHardware:")
        print("  • Relay LED should light up on ESP32")
        print("  • Solenoid valve should click and open")
        print("  • Water should start flowing")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    send_valve_command()
