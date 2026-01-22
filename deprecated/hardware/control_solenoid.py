#!/usr/bin/env python3
"""
Control script for solenoid valve
Opens the solenoid valve by publishing MQTT command
"""
import paho.mqtt.client as mqtt
import time
import sys

def control_solenoid(action="open"):
    """
    Control the solenoid valve via MQTT
    action: "open" or "close"
    """
    try:
        # Create MQTT client
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        
        # Connect to MQTT broker
        print(f"Connecting to MQTT broker at localhost:1883...")
        client.connect("localhost", 1883, 60)
        
        # Publish command based on action
        # Note: The ESP32 firmware listens to "farmfederate/control/relay" topic
        if action.lower() == "open":
            client.publish("farmfederate/control/relay", "ON")
            print("✓ Solenoid valve OPENED - Water flow ON")
            print("  Topic: farmfederate/control/relay")
            print("  Command: ON")
        elif action.lower() == "close":
            client.publish("farmfederate/control/relay", "OFF")
            print("✓ Solenoid valve CLOSED - Water flow OFF")
            print("  Topic: farmfederate/control/relay")
            print("  Command: OFF")
        else:
            print(f"Invalid action: {action}. Use 'open' or 'close'")
            return False
        
        # Give time for message to send
        time.sleep(0.5)
        
        # Disconnect
        client.disconnect()
        return True
        
    except ConnectionRefusedError:
        print("✗ Error: Could not connect to MQTT broker at localhost:1883")
        print("  Make sure the MQTT broker (Mosquitto) is running.")
        print("  Start it with: net start mosquitto")
        return False
    except Exception as e:
        print(f"✗ Error controlling solenoid valve: {e}")
        return False

if __name__ == "__main__":
    # Check command line argument
    if len(sys.argv) > 1:
        action = sys.argv[1]
    else:
        action = "open"  # Default to open
    
    print("=" * 50)
    print("FarmFederate Solenoid Valve Control")
    print("=" * 50)
    
    success = control_solenoid(action)
    
    if success:
        print(f"\nCommand sent successfully!")
        print(f"The ESP32 should have received the command.")
        print(f"Check GPIO 27 (solenoid valve pin) for relay activation.")
    else:
        print("\nFailed to send command. Check MQTT broker status.")
    
    print("=" * 50)
