#!/usr/bin/env python3
"""
Post-Upload Verification Script
Automatically checks if ESP32 firmware was uploaded successfully
"""
import time
import sys

def main():
    print("=" * 70)
    print("ESP32 FIRMWARE UPLOAD VERIFICATION")
    print("=" * 70)
    print("\n✋ WAIT! Before running this script:\n")
    print("1. Open Arduino IDE")
    print("2. Open: backend/hardware/esp32_sensor_node/esp32_complete_sensors.ino")
    print("3. Update WiFi SSID and Password (lines 30-31)")
    print("4. Select: Tools → Board → ESP32 Dev Module")
    print("5. Select: Tools → Port → COM7")
    print("6. Click Upload button (→)")
    print("7. Wait for 'Done uploading' message")
    print("\n" + "=" * 70)
    
    response = input("\nHave you uploaded the firmware? (yes/no): ").lower()
    
    if response not in ['yes', 'y']:
        print("\n❌ Please upload firmware first, then run this script again.")
        print("\nQuick command to run after upload:")
        print("  python backend/verify_upload.py")
        return
    
    print("\n" + "=" * 70)
    print("TESTING ESP32 CONNECTION...")
    print("=" * 70)
    
    # Test via USB
    print("\n[Test 1] Checking USB Serial Connection...")
    print("Running: python backend/check_sensors_usb.py")
    print("-" * 70)
    
    import subprocess
    result = subprocess.run(
        ["python", "backend/check_sensors_usb.py"],
        capture_output=False
    )
    
    print("\n" + "=" * 70)
    
    if result.returncode == 0:
        print("\n[Test 2] Checking MQTT Connection...")
        print("Running: python backend/quick_esp32_check.py")
        print("-" * 70)
        
        subprocess.run(
            ["python", "backend/quick_esp32_check.py"],
            capture_output=False
        )
    
    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)
    print("\nIf sensors are detected, you can now:")
    print("  • Control valve: python backend/valve_controller.py open")
    print("  • Monitor sensors: python backend/diagnose_sensors.py")
    print("  • Check status: python backend/troubleshoot_esp32.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
