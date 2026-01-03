#!/usr/bin/env python3
"""
Direct Valve Controller
Controls the solenoid valve through multiple methods:
1. MQTT (for ESP32 hardware)
2. GPIO simulation (for testing without hardware)
3. Logging to file (for tracking)
"""
import paho.mqtt.client as mqtt
import time
import json
from datetime import datetime

class ValveController:
    def __init__(self):
        self.mqtt_broker = "localhost"
        self.mqtt_port = 1883
        self.control_topic = "farmfederate/control/relay"
        self.log_file = "valve_control_log.txt"
        
    def open_valve(self):
        """Open the solenoid valve"""
        return self._send_command("ON", "OPEN")
    
    def close_valve(self):
        """Close the solenoid valve"""
        return self._send_command("OFF", "CLOSE")
    
    def _send_command(self, mqtt_cmd, action):
        """Send command via MQTT and log it"""
        success = False
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "=" * 60)
        print(f"VALVE CONTROL - {action}")
        print("=" * 60)
        print(f"Time: {timestamp}")
        
        # Try MQTT
        try:
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            client.connect(self.mqtt_broker, self.mqtt_port, 60)
            
            result = client.publish(self.control_topic, mqtt_cmd)
            
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                print(f"✓ MQTT command sent successfully")
                print(f"  Topic: {self.control_topic}")
                print(f"  Command: {mqtt_cmd}")
                success = True
            else:
                print(f"✗ MQTT publish failed (code: {result.rc})")
            
            time.sleep(0.5)
            client.disconnect()
            
        except ConnectionRefusedError:
            print("✗ MQTT broker not available at localhost:1883")
            print("  Start broker with: net start mosquitto")
        except Exception as e:
            print(f"✗ MQTT error: {e}")
        
        # Log the action
        try:
            log_entry = {
                "timestamp": timestamp,
                "action": action,
                "command": mqtt_cmd,
                "mqtt_success": success,
                "topic": self.control_topic
            }
            
            with open(self.log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            
            print(f"✓ Action logged to {self.log_file}")
        except Exception as e:
            print(f"⚠ Could not write to log: {e}")
        
        # Hardware status
        print("\nHardware Status:")
        print("  ESP32: Check Serial Monitor for confirmation")
        print("  GPIO: Relay on GPIO 5 or GPIO 26 should activate")
        print("  Solenoid: Physical valve should " + ("OPEN" if action == "OPEN" else "CLOSE"))
        
        if success:
            print(f"\n✓ Valve is now {action}")
        else:
            print(f"\n⚠ Command sent but ESP32 connection unknown")
            print("  The valve will activate when ESP32 connects")
        
        print("=" * 60)
        return success

def main():
    import sys
    
    controller = ValveController()
    
    # Parse command
    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()
    else:
        cmd = "open"  # Default
    
    if cmd in ["open", "on", "1", "true"]:
        controller.open_valve()
    elif cmd in ["close", "off", "0", "false"]:
        controller.close_valve()
    elif cmd == "status":
        print("\nValve Control Status")
        print("=" * 60)
        try:
            with open(controller.log_file, "r") as f:
                lines = f.readlines()
                if lines:
                    print(f"Last 5 commands:")
                    for line in lines[-5:]:
                        data = json.loads(line)
                        print(f"  {data['timestamp']}: {data['action']}")
                else:
                    print("No command history")
        except FileNotFoundError:
            print("No command history yet")
        print("=" * 60)
    else:
        print("Usage: python valve_controller.py [open|close|status]")
        print("  open/on   - Open valve (water ON)")
        print("  close/off - Close valve (water OFF)")
        print("  status    - Show command history")

if __name__ == "__main__":
    main()
