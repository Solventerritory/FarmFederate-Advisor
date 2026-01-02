import serial
import time

print("[MONITOR] Starting ESP32-CAM serial monitor on COM8...")
try:
    ser = serial.Serial('COM8', 115200, timeout=1)
    print("[MONITOR] Connected successfully. Reading output...\n")
    time.sleep(2)  # Give it a moment to stabilize
    
    # Read for 30 seconds
    start_time = time.time()
    while (time.time() - start_time) < 30:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
        time.sleep(0.1)
    
    print("\n[MONITOR] 30 second capture complete.")
    ser.close()
except serial.SerialException as e:
    print(f"[ERROR] Could not open COM8: {e}")
except KeyboardInterrupt:
    print("\n[MONITOR] Stopped by user.")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
