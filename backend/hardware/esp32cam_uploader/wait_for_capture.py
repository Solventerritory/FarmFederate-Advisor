import serial
import time
from datetime import datetime

# Connect to ESP32-CAM
ser = serial.Serial('COM8', 115200, timeout=1)
time.sleep(1)

print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected to ESP32-CAM")
print(f"[{datetime.now().strftime('%H:%M:%S')}] Waiting for next automatic capture...")
print(f"[{datetime.now().strftime('%H:%M:%S')}] (ESP32-CAM captures every 60 seconds)")
print("=" * 70)

capture_detected = False
upload_detected = False

try:
    while True:
        if ser.in_waiting:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                timestamp = datetime.now().strftime('%H:%M:%S')
                
                # Highlight important events
                if "TRIGGER" in line:
                    print(f"\n[{timestamp}] 📸 {line}")
                    capture_detected = True
                elif "Camera" in line or "UPLOAD" in line or "Sending" in line:
                    print(f"[{timestamp}] 📤 {line}")
                    upload_detected = True
                elif "HTTP" in line or "Response" in line or "200" in line:
                    print(f"[{timestamp}] ✅ {line}")
                    if capture_detected and upload_detected:
                        print("\n" + "=" * 70)
                        print("SUCCESS! Image captured and uploaded to backend!")
                        print("=" * 70)
                        break
                elif "WiFi" in line or "Connected" in line or "IP" in line:
                    print(f"[{timestamp}] 🌐 {line}")
                elif "ERROR" in line or "WARN" in line or "Failed" in line:
                    print(f"[{timestamp}] ⚠️  {line}")
                else:
                    print(f"[{timestamp}] {line}")
                    
except KeyboardInterrupt:
    print("\n\nMonitoring stopped by user")
finally:
    ser.close()
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Serial connection closed")
