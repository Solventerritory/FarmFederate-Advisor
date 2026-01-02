import serial
import time

# Connect to ESP32-CAM
ser = serial.Serial('COM8', 115200, timeout=5)
time.sleep(2)

print("[INFO] Connected to ESP32-CAM on COM8")
print("[INFO] Waiting for automatic capture (occurs every 60 seconds)...")
print("[INFO] Monitoring serial output for 90 seconds...\n")

start_time = time.time()
while (time.time() - start_time) < 90:
    if ser.in_waiting:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(line)
            # Check for key events
            if "TRIGGER" in line or "Camera" in line or "UPLOAD" in line or "HTTP" in line:
                print(f">>> {line}")

ser.close()
print("\n[INFO] Monitoring complete")
