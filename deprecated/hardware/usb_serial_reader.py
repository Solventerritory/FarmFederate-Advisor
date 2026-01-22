#!/usr/bin/env python3
"""
USB Serial Reader for ESP32 Sensors
Reads sensor data from ESP32 connected via USB and publishes to MQTT
"""
import serial
import serial.tools.list_ports
import json
import time
from datetime import datetime
import paho.mqtt.client as mqtt

# MQTT Configuration
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_BASE = "farmfederate/sensors"

def find_esp32_port():
    """Automatically detect ESP32 COM port"""
    ports = serial.tools.list_ports.comports()
    
    print("\n" + "="*60)
    print("AVAILABLE SERIAL PORTS:")
    print("="*60)
    
    for i, port in enumerate(ports, 1):
        print(f"{i}. {port.device}")
        print(f"   Description: {port.description}")
        print(f"   Manufacturer: {port.manufacturer}")
        
        # Auto-detect ESP32
        if "CP210" in port.description or "CH340" in port.description or "USB-SERIAL" in port.description or "USB Serial" in port.description:
            print(f"   ⭐ Likely ESP32 device!")
            return port.device
        print()
    
    if ports:
        print(f"\nFound {len(ports)} port(s)")
        choice = input("\nEnter port number to use (or press Enter for port 1): ").strip()
        if not choice:
            choice = "1"
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ports):
                return ports[idx].device
        except:
            pass
    
    print("\n⚠ No serial ports found or invalid selection")
    return None

def read_esp32_serial(port_name, baud_rate=115200):
    """Read sensor data from ESP32 via USB Serial"""
    
    print("\n" + "="*60)
    print("ESP32 USB SERIAL READER")
    print("="*60)
    print(f"Port: {port_name}")
    print(f"Baud Rate: {baud_rate}")
    print(f"MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
    print("="*60)
    print("\nConnecting to ESP32...")
    
    try:
        # Open serial port
        ser = serial.Serial(port_name, baud_rate, timeout=1)
        time.sleep(2)  # Wait for connection to stabilize
        print("✓ Serial port opened successfully")
        
        # Connect to MQTT
        mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print("✓ Connected to MQTT broker")
        print("\n" + "="*60)
        print("RECEIVING SENSOR DATA (Press Ctrl+C to stop)")
        print("="*60 + "\n")
        
        while True:
            if ser.in_waiting > 0:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    # Look for JSON data with DATA: prefix
                    if line.startswith('DATA:'):
                        json_str = line[5:]  # Remove 'DATA:' prefix
                        data = json.loads(json_str)
                        
                        # Add timestamp
                        data['timestamp'] = datetime.now().isoformat()
                        
                        # Print to console
                        print(f"\n⏰ {datetime.now().strftime('%H:%M:%S')}")
                        print(f"📍 Device: {data.get('client_id', 'unknown')}")
                        print(f"🌡️  Temperature: {data.get('temperature', 0)}°C")
                        print(f"💧 Humidity: {data.get('humidity', 0)}%")
                        print(f"🌱 Soil Moisture: {data.get('soil_moisture', 0)}%")
                        print(f"🌊 Flow Rate: {data.get('flow_rate', 0)} L/min")
                        print(f"📊 Total Flow: {data.get('total_liters', 0)} L")
                        print(f"🔌 Relay: {data.get('relay_state', 'OFF')}")
                        
                        # Publish to MQTT
                        client_id = data.get('client_id', 'esp32_usb')
                        topic = f"{MQTT_TOPIC_BASE}/{client_id}"
                        payload = json.dumps(data)
                        mqtt_client.publish(topic, payload)
                        print(f"✓ Published to MQTT: {topic}")
                        
                    else:
                        # Print other messages (debug info)
                        if line:
                            print(f"[ESP32] {line}")
                            
                except json.JSONDecodeError:
                    # Not JSON, just print it
                    if line and not line.startswith('DATA:'):
                        print(f"[ESP32] {line}")
                except Exception as e:
                    print(f"Error processing line: {e}")
            
            time.sleep(0.01)
            
    except serial.SerialException as e:
        print(f"\n✗ Serial port error: {e}")
        print("\nTroubleshooting:")
        print("  1. Check ESP32 is connected via USB")
        print("  2. Make sure no other program is using the port (close Arduino Serial Monitor)")
        print("  3. Try unplugging and reconnecting ESP32")
        print("  4. Check USB cable (use data cable, not charge-only)")
    except KeyboardInterrupt:
        print("\n\n" + "="*60)
        print("Stopped by user")
        print("="*60)
    except Exception as e:
        print(f"\n✗ Error: {e}")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()
            print("✓ Serial port closed")
        if 'mqtt_client' in locals():
            mqtt_client.loop_stop()
            mqtt_client.disconnect()
            print("✓ MQTT disconnected")

if __name__ == "__main__":
    print("\n╔════════════════════════════════════════════════════════════╗")
    print("║  FarmFederate-Advisor - ESP32 USB Serial Reader          ║")
    print("╚════════════════════════════════════════════════════════════╝")
    
    # Auto-detect or select port
    port = find_esp32_port()
    
    if port:
        print(f"\n✓ Selected port: {port}")
        print("\nMake sure:")
        print("  1. ESP32 is connected via USB")
        print("  2. ESP32 sketch 'esp32_usb_serial.ino' is uploaded")
        print("  3. Arduino Serial Monitor is CLOSED")
        print("  4. Sensors are connected to ESP32")
        
        input("\nPress Enter to start reading...")
        read_esp32_serial(port)
    else:
        print("\n✗ No suitable port found. Please:")
        print("  1. Connect ESP32 via USB")
        print("  2. Install USB drivers (CP210x or CH340)")
        print("  3. Run this script again")
