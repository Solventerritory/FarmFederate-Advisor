#!/usr/bin/env python3
"""
Quick ESP32 Sensor Check via USB
Reads sensor data directly from ESP32 serial output
"""
import serial
import serial.tools.list_ports
import time
import re

def find_esp32():
    """Find ESP32 COM port"""
    ports = serial.tools.list_ports.comports()
    
    print("Scanning for ESP32...")
    for port in ports:
        desc = port.description.upper()
        if any(x in desc for x in ["CP210", "CH340", "USB-SERIAL", "USB SERIAL", "UART", "ESP"]):
            print(f"✓ Found ESP32 on {port.device}")
            return port.device
    
    if ports:
        print(f"\nAvailable ports:")
        for port in ports:
            print(f"  • {port.device}: {port.description}")
        return ports[0].device
    
    return None

def check_sensors_via_usb(port, duration=10):
    """Read and analyze sensor data from USB"""
    
    sensors_detected = {
        'temperature': False,
        'humidity': False,
        'soil_moisture': False,
        'flow_rate': False,
        'relay': False,
        'wifi': False,
        'mqtt': False
    }
    
    sensor_values = {}
    
    print("\n" + "=" * 70)
    print("ESP32 SENSOR CHECK VIA USB")
    print("=" * 70)
    print(f"Port: {port}")
    print(f"Reading for {duration} seconds...")
    print("-" * 70 + "\n")
    
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        time.sleep(2)
        
        # Clear buffer
        ser.reset_input_buffer()
        
        start_time = time.time()
        line_count = 0
        
        while (time.time() - start_time) < duration:
            if ser.in_waiting:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        line_count += 1
                        print(line)
                        
                        # Check for sensor readings
                        if 'Temperature' in line or 'temperature' in line:
                            sensors_detected['temperature'] = True
                            match = re.search(r'(\d+\.?\d*)\s*°?C', line)
                            if match:
                                sensor_values['temperature'] = float(match.group(1))
                        
                        if 'Humidity' in line or 'humidity' in line:
                            sensors_detected['humidity'] = True
                            match = re.search(r'(\d+\.?\d*)\s*%', line)
                            if match:
                                sensor_values['humidity'] = float(match.group(1))
                        
                        if 'Soil' in line or 'soil' in line:
                            sensors_detected['soil_moisture'] = True
                            match = re.search(r'(\d+\.?\d*)\s*%', line)
                            if match:
                                sensor_values['soil_moisture'] = float(match.group(1))
                        
                        if 'Flow' in line or 'flow' in line:
                            sensors_detected['flow_rate'] = True
                            match = re.search(r'(\d+\.?\d*)\s*L/min', line)
                            if match:
                                sensor_values['flow_rate'] = float(match.group(1))
                        
                        if 'Relay' in line or 'relay' in line:
                            sensors_detected['relay'] = True
                        
                        if 'WiFi connected' in line or 'IP Address' in line:
                            sensors_detected['wifi'] = True
                        
                        if 'MQTT connected' in line or 'MQTT Message' in line:
                            sensors_detected['mqtt'] = True
                
                except Exception as e:
                    pass
        
        ser.close()
        
        # Print summary
        print("\n" + "=" * 70)
        print("SENSOR DETECTION SUMMARY")
        print("=" * 70)
        print(f"Lines read: {line_count}")
        print()
        
        print("Connection Status:")
        print(f"  {'✅' if sensors_detected['wifi'] else '❌'} WiFi Connected")
        print(f"  {'✅' if sensors_detected['mqtt'] else '❌'} MQTT Connected")
        print()
        
        print("Sensors Detected:")
        print(f"  {'✅' if sensors_detected['temperature'] else '❌'} Temperature (DHT22 on GPIO 4)")
        if 'temperature' in sensor_values:
            print(f"      Value: {sensor_values['temperature']:.1f}°C")
        
        print(f"  {'✅' if sensors_detected['humidity'] else '❌'} Humidity (DHT22 on GPIO 4)")
        if 'humidity' in sensor_values:
            print(f"      Value: {sensor_values['humidity']:.1f}%")
        
        print(f"  {'✅' if sensors_detected['soil_moisture'] else '❌'} Soil Moisture (GPIO 34)")
        if 'soil_moisture' in sensor_values:
            print(f"      Value: {sensor_values['soil_moisture']:.1f}%")
        
        print(f"  {'✅' if sensors_detected['flow_rate'] else '❌'} Flow Meter (GPIO 18)")
        if 'flow_rate' in sensor_values:
            print(f"      Value: {sensor_values['flow_rate']:.1f} L/min")
        
        print(f"  {'✅' if sensors_detected['relay'] else '❌'} Relay/Valve (GPIO 5)")
        
        print("\n" + "=" * 70)
        
        # Overall status
        detected_count = sum(sensors_detected.values())
        if detected_count >= 5:
            print("✅ ESP32 IS WORKING - Most systems operational")
        elif detected_count >= 2:
            print("⚠️  ESP32 PARTIALLY WORKING - Some sensors detected")
        else:
            print("❌ ESP32 NOT RESPONDING - Check connections")
        
        print("=" * 70)
        
    except serial.SerialException as e:
        print(f"\n❌ Error: Could not open port {port}")
        print(f"   {e}")
        print("\nTroubleshooting:")
        print("  • Make sure ESP32 is plugged in via USB")
        print("  • Close Arduino Serial Monitor if open")
        print("  • Try a different USB port")
        print("  • Check if driver is installed")
    except KeyboardInterrupt:
        print("\n\nTest interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")

def main():
    print("=" * 70)
    print("ESP32 SENSOR DIAGNOSTIC - USB SERIAL METHOD")
    print("=" * 70)
    
    port = find_esp32()
    
    if not port:
        print("\n❌ No ESP32 found on USB")
        print("\nMake sure:")
        print("  • ESP32 is connected via USB")
        print("  • USB drivers are installed")
        print("  • Cable supports data (not power-only)")
        return
    
    check_sensors_via_usb(port, duration=10)
    
    print("\nFor continuous monitoring, run:")
    print("  python backend/usb_serial_reader.py")

if __name__ == "__main__":
    main()
