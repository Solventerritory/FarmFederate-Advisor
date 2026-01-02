import serial
import time

print("="*70)
print("ESP32-CAM MONITOR - Waiting for data...")
print("="*70)
print("[INFO] If nothing appears:")
print("       1. Press the RESET button on ESP32-CAM")
print("       2. Check if the power LED is ON")
print("       3. Replug the USB cable")
print("-"*70)

try:
    ser = serial.Serial('COM8', 115200, timeout=1)
    print("[OK] Connected to COM8 at 115200 baud\n")
    
    start_time = time.time()
    data_found = False
    
    # Monitor for 30 seconds
    while (time.time() - start_time) < 30:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    if not data_found:
                        print("[DATA DETECTED]\n")
                        data_found = True
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {line}")
            except:
                pass
        time.sleep(0.05)
    
    if not data_found:
        print("\n[NO DATA] No output detected in 30 seconds.")
        print("[ACTION REQUIRED] Please:")
        print("  1. Press the RESET button on your ESP32-CAM")
        print("  2. Check if red power LED is lit")
        print("  3. Try disconnecting and reconnecting USB")
        print("  4. Run this script again after reset")
    
    ser.close()
    print("\n" + "="*70)
    print("Monitor completed")
    print("="*70)

except serial.SerialException as e:
    print(f"\n[ERROR] Could not open COM8: {e}")
    print("       ESP32-CAM might be disconnected or port is in use")
except Exception as e:
    print(f"\n[ERROR] {e}")
