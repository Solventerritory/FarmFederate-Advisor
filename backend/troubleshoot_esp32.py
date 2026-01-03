#!/usr/bin/env python3
"""
ESP32 Connection Troubleshooter
Helps diagnose why ESP32 isn't connecting
"""
import socket
import subprocess
import os

def get_computer_ip():
    """Get this computer's IP address"""
    try:
        # Connect to external address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except:
        return "Unable to determine"

def check_mqtt_broker():
    """Check if MQTT broker is running"""
    try:
        result = subprocess.run(
            ["powershell", "-Command", "Get-Service mosquitto -ErrorAction SilentlyContinue"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "Running" in result.stdout:
            return True, "Running"
        elif "Stopped" in result.stdout:
            return False, "Stopped"
        else:
            return False, "Not installed"
    except:
        return False, "Unknown"

def check_firewall():
    """Check if firewall might be blocking MQTT"""
    try:
        result = subprocess.run(
            ["powershell", "-Command", 
             "Get-NetFirewallRule | Where-Object {$_.LocalPort -eq 1883} | Select-Object DisplayName, Enabled"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.stdout.strip():
            return True, "Firewall rules found for port 1883"
        else:
            return False, "No firewall rules for port 1883"
    except:
        return False, "Unable to check"

def main():
    print("=" * 80)
    print("ESP32 CONNECTION TROUBLESHOOTER")
    print("=" * 80)
    
    print("\n[1] NETWORK CONFIGURATION")
    print("-" * 80)
    ip = get_computer_ip()
    print(f"Your Computer's IP Address: {ip}")
    print(f"\n⚠ IMPORTANT: Update ESP32 firmware with this IP!")
    print(f"   In esp32_complete_sensors.ino, change:")
    print(f"   const char* MQTT_SERVER = \"{ip}\";")
    
    print("\n[2] MQTT BROKER STATUS")
    print("-" * 80)
    is_running, status = check_mqtt_broker()
    if is_running:
        print(f"✅ Mosquitto MQTT Broker: {status}")
    else:
        print(f"❌ Mosquitto MQTT Broker: {status}")
        if status == "Not installed":
            print("   Install: https://mosquitto.org/download/")
        elif status == "Stopped":
            print("   Start with: net start mosquitto")
    
    print("\n[3] FIREWALL STATUS")
    print("-" * 80)
    has_rules, firewall_status = check_firewall()
    print(f"Port 1883 Firewall: {firewall_status}")
    if not has_rules:
        print("⚠  Consider adding firewall rule:")
        print("   New-NetFirewallRule -DisplayName 'MQTT' -Direction Inbound -LocalPort 1883 -Protocol TCP -Action Allow")
    
    print("\n[4] ESP32 FIRMWARE CHECKLIST")
    print("-" * 80)
    print("Open: backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino")
    print("\nVerify these settings:")
    print("  ☐ WiFi SSID is correct (case-sensitive)")
    print("  ☐ WiFi Password is correct")
    print(f"  ☐ MQTT_SERVER = \"{ip}\"")
    print("  ☐ MQTT_PORT = 1883")
    print("  ☐ Uploaded to ESP32 successfully")
    
    print("\n[5] HARDWARE CHECKLIST")
    print("-" * 80)
    print("Sensors that should be connected:")
    print("  ☐ DHT22 Temperature/Humidity → GPIO 4")
    print("  ☐ Soil Moisture Sensor → GPIO 34 (Analog)")
    print("  ☐ Flow Meter (YF-S201) → GPIO 18")
    print("  ☐ Relay Module → GPIO 5")
    print("  ☐ Solenoid Valve → Connected to Relay")
    
    print("\n[6] QUICK TESTS")
    print("-" * 80)
    print("Run these commands to test:")
    print(f"  1. Test MQTT broker:")
    print(f"     python backend/quick_esp32_check.py")
    print(f"\n  2. Monitor ESP32 (Arduino Serial Monitor):")
    print(f"     • Baud rate: 115200")
    print(f"     • Look for: 'WiFi connected' and 'MQTT connected'")
    print(f"\n  3. Test valve control:")
    print(f"     python backend/valve_controller.py open")
    
    print("\n[7] COMMON SOLUTIONS")
    print("-" * 80)
    print("If ESP32 won't connect:")
    print("  1. Check ESP32 has power (LED should be on)")
    print("  2. ESP32 only supports 2.4GHz WiFi (not 5GHz)")
    print("  3. Reset ESP32 (press reset button)")
    print("  4. Re-upload firmware with correct IP address")
    print("  5. Check USB cable is data cable (not power-only)")
    print("  6. Try different USB port")
    
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Update MQTT_SERVER IP in ESP32 firmware")
    print("2. Upload firmware to ESP32")
    print("3. Open Serial Monitor to see connection status")
    print("4. Run: python backend/diagnose_sensors.py")
    print("=" * 80)

if __name__ == "__main__":
    main()
