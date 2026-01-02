import serial
import time

port = 'COM8'
baud = 115200

try:
    ser = serial.Serial(port, baud, timeout=1)
    print(f"Connected to {port}")
    print("=" * 70)
    
    start_time = time.time()
    line_count = 0
    
    while time.time() - start_time < 10 and line_count < 50:
        line = ser.readline().decode('utf-8', errors='ignore').strip()
        if line:
            print(f"[{line_count:03d}] {line}")
            line_count += 1
    
    ser.close()
    print("=" * 70)
    print(f"Captured {line_count} lines in 10 seconds")
    
except Exception as e:
    print(f"Error: {e}")
