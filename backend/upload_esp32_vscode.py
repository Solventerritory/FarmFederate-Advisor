#!/usr/bin/env python3
"""
Quick ESP32 upload using VS Code + PlatformIO
"""
import subprocess
import os
import sys

def main():
    print("=" * 70)
    print("ESP32 UPLOAD - VS CODE + PLATFORMIO")
    print("=" * 70)
    
    project_dir = r"C:\Users\USER_HP\Desktop\FarmFederate\FarmFederate-Advisor\backend\hardware\esp32_sensor_node"
    
    print("\n📋 Pre-flight Checklist:")
    print("  ☐ PlatformIO extension installed in VS Code")
    print("  ☐ WiFi credentials updated in esp32_complete_sensors.ino")
    print("  ☐ ESP32 connected to COM7")
    print()
    
    response = input("Ready to upload? (y/n): ").lower()
    
    if response != 'y':
        print("\n⚠️  Please complete the checklist first.")
        print("\nTo install PlatformIO:")
        print("  1. Open VS Code")
        print("  2. Extensions (Ctrl+Shift+X)")
        print("  3. Search 'PlatformIO IDE'")
        print("  4. Install")
        print("\nTo update WiFi (already open in VS Code):")
        print("  Lines 30-31 in esp32_complete_sensors.ino")
        return
    
    print("\n" + "=" * 70)
    print("UPLOADING FIRMWARE...")
    print("=" * 70)
    
    os.chdir(project_dir)
    
    # Check if pio is available
    try:
        result = subprocess.run(["pio", "--version"], capture_output=True, text=True)
        print(f"✓ PlatformIO Core: {result.stdout.strip()}")
    except FileNotFoundError:
        print("✗ PlatformIO CLI not found")
        print("\nAlternative: Use VS Code GUI")
        print("  1. Open PlatformIO sidebar (alien icon)")
        print("  2. Click 'Upload' under esp32dev")
        print("  Or press Ctrl+Shift+P → 'PlatformIO: Upload'")
        return
    
    # Upload
    print("\nCompiling and uploading...")
    print("-" * 70)
    
    result = subprocess.run(
        ["pio", "run", "--target", "upload"],
        cwd=project_dir
    )
    
    if result.returncode == 0:
        print("\n" + "=" * 70)
        print("✅ UPLOAD SUCCESSFUL!")
        print("=" * 70)
        
        print("\nStarting serial monitor...")
        print("(Press Ctrl+C to stop)\n")
        print("-" * 70)
        
        try:
            subprocess.run(
                ["pio", "device", "monitor", "--port", "COM7", "--baud", "115200"],
                cwd=project_dir
            )
        except KeyboardInterrupt:
            print("\n\nMonitor stopped.")
        
        print("\n" + "=" * 70)
        print("NEXT STEPS")
        print("=" * 70)
        print("\nTest sensors:")
        print("  python backend/check_sensors_usb.py")
        print("  python backend/diagnose_sensors.py")
        print("\nControl valve:")
        print("  python backend/valve_controller.py open")
        print("=" * 70)
    else:
        print("\n" + "=" * 70)
        print("❌ UPLOAD FAILED")
        print("=" * 70)
        print("\nTroubleshooting:")
        print("  • Hold BOOT button on ESP32 during upload")
        print("  • Try different USB port")
        print("  • Check COM7 is correct port")
        print("  • Make sure Serial Monitor is closed")
        print("\nOr upload via VS Code GUI:")
        print("  Ctrl+Shift+P → 'PlatformIO: Upload'")

if __name__ == "__main__":
    main()
