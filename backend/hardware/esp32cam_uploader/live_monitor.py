import serial
import time
import sys

print("=" * 60)
print("ESP32-CAM LIVE MONITOR")
print("=" * 60)
print("[INFO] Connecting to COM8 at 115200 baud...")
print("[INFO] Press RESET button on ESP32-CAM to see boot messages")
print("[INFO] Press Ctrl+C to stop monitoring\n")

try:
    ser = serial.Serial('COM8', 115200, timeout=0.5)
    print("[OK] Connected to COM8\n")
    print("-" * 60)
    
    while True:
        if ser.in_waiting > 0:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    timestamp = time.strftime("%H:%M:%S")
                    print(f"[{timestamp}] {line}")
                    sys.stdout.flush()
            except Exception as e:
                pass
        time.sleep(0.05)
        
except serial.SerialException as e:
    print(f"\n[ERROR] Could not open COM8: {e}")
    print("       Make sure ESP32-CAM is connected and no other program is using COM8")
except KeyboardInterrupt:
    print("\n" + "-" * 60)
    print("[STOPPED] Monitoring stopped by user")
    print("=" * 60)
finally:
    try:
        ser.close()
    except:
        pass
