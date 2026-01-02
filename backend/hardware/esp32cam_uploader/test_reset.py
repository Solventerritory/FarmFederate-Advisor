import serial
import time

print("ESP32-CAM Reset Detection Test")
print("=" * 50)
print("Press the RESET button on ESP32-CAM NOW...")
print("Listening for 10 seconds...\n")

port = "COM7"
baud = 115200

try:
    ser = serial.Serial(port, baud, timeout=1)
    
    start_time = time.time()
    while time.time() - start_time < 10:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print(f"Received: {data}")
        time.sleep(0.1)
    
    ser.close()
    print("\n" + "=" * 50)
    print("If you saw data above: ✓ Serial RX is working!")
    print("If no data: ✗ Check TX/RX wiring")
    
except Exception as e:
    print(f"Error: {e}")
