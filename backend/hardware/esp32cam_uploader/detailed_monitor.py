import serial
import time

print("="*70)
print("ESP32-CAM DETAILED MONITOR")
print("="*70)

try:
    ser = serial.Serial('COM8', 115200, timeout=2)
    print("[OK] Connected to COM8 at 115200 baud\n")
    print("-"*70)
    
    # Read for 20 seconds to see full cycle
    start_time = time.time()
    while (time.time() - start_time) < 20:
        if ser.in_waiting > 0:
            try:
                # Read full lines without truncation
                line = ser.readline().decode('utf-8', errors='ignore').rstrip()
                if line:
                    print(line)
            except Exception as e:
                pass
        time.sleep(0.01)
    
    ser.close()
    print("\n" + "="*70)
    print("Monitor completed")
    print("="*70)

except serial.SerialException as e:
    print(f"\n[ERROR] Could not open COM8: {e}")
except Exception as e:
    print(f"\n[ERROR] {e}")
