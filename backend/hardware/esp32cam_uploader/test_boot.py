import serial
import time

print("ESP32-CAM Boot Message Test")
print("=" * 60)
print("This will test at ESP32 boot baud rate (74880)")
print("\n** PLEASE PRESS THE RESET BUTTON ON ESP32-CAM NOW **\n")
print("Listening for 15 seconds...\n")

port = "COM8"
baud = 74880  # ESP32 boot ROM baud rate

try:
    ser = serial.Serial(port, baud, timeout=0.5)
    
    start_time = time.time()
    received_any = False
    
    while time.time() - start_time < 15:
        if ser.in_waiting > 0:
            data = ser.read(ser.in_waiting)
            received_any = True
            try:
                print(data.decode('utf-8', errors='ignore'), end='')
            except:
                print(f"[HEX: {data.hex()}]", end='')
        time.sleep(0.1)
    
    ser.close()
    
    print("\n" + "=" * 60)
    if received_any:
        print("✓ SUCCESS! ESP32 is transmitting data")
        print("  This means TX connection is working")
    else:
        print("✗ NO DATA RECEIVED")
        print("\nPossible issues:")
        print("1. ESP32-CAM is not powered on (check 5V and GND)")
        print("2. FTDI TX→ESP32 U0R connection issue")
        print("3. ESP32-CAM might be damaged")
        print("4. FTDI adapter might be faulty")
        print("\nTry:")
        print("- Check if ESP32-CAM has a power LED that's lit")
        print("- Try powering ESP32-CAM with separate 5V supply")
        print("- Check if FTDI VCC jumper is set to 5V (not 3.3V)")
    
except Exception as e:
    print(f"Error: {e}")
