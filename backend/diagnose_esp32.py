#!/usr/bin/env python3
"""
ESP32 Connection Diagnostics
Helps identify why ESP32 sensor data is not being received
"""
import socket
import subprocess
import sys

def check_mqtt_broker():
    """Check if MQTT broker is running"""
    print("\n[1] Checking MQTT Broker...")
    try:
        result = subprocess.run(['sc', 'query', 'mosquitto'], 
                              capture_output=True, text=True)
        if 'RUNNING' in result.stdout:
            print("    ✓ Mosquitto MQTT broker is RUNNING")
            return True
        else:
            print("    ✗ Mosquitto is NOT running!")
            print("    → Start it with: Start-Service mosquitto")
            return False
    except Exception as e:
        print(f"    ✗ Error checking service: {e}")
        return False

def check_mqtt_port():
    """Check if MQTT port 1883 is listening"""
    print("\n[2] Checking MQTT Port 1883...")
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(2)
        result = sock.connect_ex(('localhost', 1883))
        sock.close()
        
        if result == 0:
            print("    ✓ Port 1883 is OPEN and listening")
            return True
        else:
            print("    ✗ Port 1883 is NOT accessible!")
            print("    → Check if Mosquitto is running")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        return False

def get_local_ip():
    """Get local IP address"""
    print("\n[3] Getting Your Computer's IP Address...")
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"    ✓ Your IP Address: {local_ip}")
        print(f"    → Use this IP in ESP32 code: const char* MQTT_SERVER = \"{local_ip}\";")
        return local_ip
    except Exception as e:
        print(f"    ✗ Error getting IP: {e}")
        return None

def test_mqtt_publish():
    """Test MQTT publish"""
    print("\n[4] Testing MQTT Publish...")
    try:
        import paho.mqtt.client as mqtt
        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        client.connect('localhost', 1883, 60)
        result = client.publish('farmfederate/test', '{"test": true}')
        client.disconnect()
        
        if result.rc == 0:
            print("    ✓ Successfully published test message")
            return True
        else:
            print(f"    ✗ Publish failed with code: {result.rc}")
            return False
    except Exception as e:
        print(f"    ✗ Error: {e}")
        print("    → Install paho-mqtt: pip install paho-mqtt")
        return False

def check_firewall():
    """Check Windows Firewall"""
    print("\n[5] Checking Firewall Rules...")
    try:
        result = subprocess.run(
            ['netsh', 'advfirewall', 'firewall', 'show', 'rule', 'name=all'],
            capture_output=True, text=True
        )
        if '1883' in result.stdout or 'mosquitto' in result.stdout.lower():
            print("    ✓ Firewall rule found for MQTT")
            return True
        else:
            print("    ⚠ No firewall rule found for MQTT port 1883")
            print("    → Add rule with:")
            print('      New-NetFirewallRule -DisplayName "MQTT" -Direction Inbound -Protocol TCP -LocalPort 1883 -Action Allow')
            return False
    except Exception as e:
        print(f"    ⚠ Could not check firewall: {e}")
        return False

def check_sensor_files():
    """Check if sensor data files exist"""
    print("\n[6] Checking Sensor Data Files...")
    import os
    import glob
    
    paths = [
        'checkpoints_paper/sensors/*.json',
        'checkpoints_paper/ingest/sensors/*.json'
    ]
    
    found_files = []
    for pattern in paths:
        files = glob.glob(pattern)
        found_files.extend(files)
    
    if found_files:
        print(f"    ✓ Found {len(found_files)} sensor data file(s)")
        for f in sorted(found_files, key=os.path.getmtime, reverse=True)[:3]:
            mtime = os.path.getmtime(f)
            from datetime import datetime
            time_str = datetime.fromtimestamp(mtime).strftime('%H:%M:%S')
            print(f"      - {os.path.basename(f)} (last modified: {time_str})")
        return True
    else:
        print("    ✗ No sensor data files found")
        print("    → ESP32 may not be publishing data")
        return False

def main():
    print("="*60)
    print("   ESP32 SENSOR DATA TROUBLESHOOTING")
    print("="*60)
    
    results = {
        'mqtt_broker': check_mqtt_broker(),
        'mqtt_port': check_mqtt_port(),
        'local_ip': get_local_ip() is not None,
        'mqtt_publish': test_mqtt_publish(),
        'firewall': check_firewall(),
        'sensor_files': check_sensor_files()
    }
    
    print("\n" + "="*60)
    print("   DIAGNOSTIC SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for check, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} - {check.replace('_', ' ').title()}")
    
    print(f"\nScore: {passed}/{total} checks passed")
    
    if passed == total:
        print("\n✓ All checks passed! Your system is ready.")
        print("\nIf ESP32 still not working:")
        print("  1. Check ESP32 Serial Monitor for connection messages")
        print("  2. Verify WiFi SSID/password in ESP32 code")
        print("  3. Verify MQTT server IP in ESP32 code")
        print("  4. Ensure ESP32 and computer on same WiFi network")
    else:
        print("\n✗ Some checks failed. Fix the issues above.")
    
    print("\n" + "="*60)
    print("\nNEXT STEPS:")
    print("1. Fix any failed checks above")
    print("2. Upload esp32_complete_sensors.ino to your ESP32")
    print("3. Open Serial Monitor (115200 baud) to see connection status")
    print("4. Watch for MQTT messages:")
    print("   mosquitto_sub -h localhost -t 'farmfederate/sensors/#' -v")
    print("="*60)

if __name__ == '__main__':
    main()
