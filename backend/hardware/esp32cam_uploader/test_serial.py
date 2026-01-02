import serial
import time
import sys

print("ESP32-CAM Serial Communication Test")
print("=" * 50)

port = "COM7"
baud_rates = [115200, 9600, 74880]  # ESP32 boot messages at 74880

for baud in baud_rates:
    print(f"\nTesting {port} at {baud} baud...")
    try:
        ser = serial.Serial(port, baud, timeout=2)
        print(f"✓ Port opened successfully")
        
        # Try to read any data
        print("Listening for 3 seconds...")
        time.sleep(0.5)
        
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            print(f"✓ Received {len(data)} bytes: {data[:100]}")
        else:
            print("✗ No data received")
        
        ser.close()
    except serial.SerialException as e:
        print(f"✗ Error: {e}")
    except Exception as e:
        print(f"✗ Unexpected error: {e}")

print("\n" + "=" * 50)
print("Test complete")
print("\nIf you see NO data at any baud rate, check:")
print("1. TX/RX wires - Try swapping them")
print("2. All 4 wires are firmly connected")
print("3. FTDI is set to 5V (if there's a jumper)")
