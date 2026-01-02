import serial
import time

print("[CHECK] Checking ESP32-CAM on COM8 at boot baud rate (74880)...")
try:
    ser = serial.Serial('COM8', 74880, timeout=1)
    print("[CHECK] Connected. Waiting for boot messages...")
    print("[CHECK] If nothing appears, try pressing the RESET button on ESP32-CAM\n")
    
    time.sleep(1)
    
    # Read for 10 seconds
    start_time = time.time()
    found_output = False
    while (time.time() - start_time) < 10:
        if ser.in_waiting > 0:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(line)
                found_output = True
        time.sleep(0.1)
    
    if not found_output:
        print("\n[CHECK] No output detected. The ESP32-CAM might be:")
        print("  1. Not powered properly (needs 5V with sufficient current)")
        print("  2. In deep sleep or not running")
        print("  3. Press the RESET button to restart")
    else:
        print("\n[CHECK] Boot messages detected!")
    
    ser.close()
except serial.SerialException as e:
    print(f"[ERROR] Could not open COM8: {e}")
    print("       The ESP32-CAM might be disconnected or another program is using the port.")
except Exception as e:
    print(f"[ERROR] Unexpected error: {e}")
