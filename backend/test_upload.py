#!/usr/bin/env python3
"""
Post-Upload Test - Verify ESP32 connection and sensors
Run this after uploading firmware to ESP32
"""
import time
import sys

def main():
    print("\n" + "=" * 70)
    print("ESP32 POST-UPLOAD VERIFICATION")
    print("=" * 70)
    print("\n✅ Firmware uploaded with:")
    print("   WiFi: Ayush")
    print("   MQTT Server: 192.168.0.195")
    print("   Port: COM7")
    
    print("\n" + "=" * 70)
    input("\nPress Enter after ESP32 boots up (wait 10 seconds after upload)...")
    
    print("\n[TEST 1] Checking USB Serial Connection...")
    print("-" * 70)
    
    import subprocess
    result = subprocess.run(
        ["python", "backend/check_sensors_usb.py"],
        capture_output=False
    )
    
    print("\n[TEST 2] Checking MQTT Connection...")
    print("-" * 70)
    
    subprocess.run(
        ["python", "backend/diagnose_sensors.py"],
        capture_output=False
    )
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    
    print("\nIf sensors detected:")
    print("  ✅ Upload successful!")
    print("  ✅ All systems operational!")
    print("\nYou can now:")
    print("  • Open valve: python backend/valve_controller.py open")
    print("  • Close valve: python backend/valve_controller.py close")
    print("  • Monitor: python backend/mqtt_monitor.py")
    
    print("\nIf no sensors detected:")
    print("  • Check Serial Monitor for errors")
    print("  • Verify WiFi 'Ayush' is 2.4GHz (not 5GHz)")
    print("  • Check MQTT broker: net start mosquitto")
    print("  • Run troubleshooter: python backend/troubleshoot_esp32.py")
    
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
