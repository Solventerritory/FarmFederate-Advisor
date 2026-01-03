#!/usr/bin/env python3
"""
USB Serial Valve Controller
Controls solenoid valve via direct USB serial commands to ESP32
"""
import serial
import time
import json
import sys

def find_esp32_port():
    """Find ESP32 COM port"""
    return 'COM7'  # We know it's COM7

def send_valve_command(command):
    """Send valve control command via USB serial"""
    port = find_esp32_port()
    
    print("\n" + "="*70)
    print("🔌 USB SERIAL VALVE CONTROLLER")
    print("="*70)
    print(f"\nPort: {port}")
    print(f"Command: {command.upper()}\n")
    
    try:
        # Open serial connection
        ser = serial.Serial(port, 115200, timeout=2)
        time.sleep(0.5)  # Wait for connection to stabilize
        
        # Send command
        if command.lower() == 'on' or command.lower() == 'open':
            cmd = "RELAY:ON\n"
            print("📤 Sending: RELAY:ON")
        elif command.lower() == 'off' or command.lower() == 'close':
            cmd = "RELAY:OFF\n"
            print("📤 Sending: RELAY:OFF")
        elif command.lower() == 'status':
            cmd = "STATUS\n"
            print("📤 Sending: STATUS")
        else:
            print(f"❌ Unknown command: {command}")
            ser.close()
            return False
        
        ser.write(cmd.encode())
        ser.flush()
        
        # Wait for response and read sensor data
        print("\n⏳ Waiting for response...\n")
        time.sleep(2)
        
        # Read multiple lines to get updated status
        valve_status = None
        for _ in range(10):
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if line:
                print(f"[ESP32] {line}")
                
                # Parse JSON data if present
                if 'DATA:' in line:
                    try:
                        data = json.loads(line.split('DATA:')[1])
                        valve_status = data.get('relay_state')
                    except:
                        pass
        
        ser.close()
        
        # Display final status
        print("\n" + "="*70)
        print("VALVE STATUS AFTER COMMAND")
        print("="*70)
        
        if valve_status is not None:
            status_icon = "🟢" if valve_status else "🔴"
            status_text = "OPEN (ON)" if valve_status else "CLOSED (OFF)"
            print(f"\n{status_icon} Solenoid Valve: {status_text}\n")
            
            if command.lower() in ['on', 'open'] and valve_status:
                print("✅ SUCCESS: Valve opened!")
                print("💦 Water is now flowing!")
            elif command.lower() in ['off', 'close'] and not valve_status:
                print("✅ SUCCESS: Valve closed!")
                print("🛑 Water flow stopped!")
            elif command.lower() == 'status':
                print(f"📊 Current status: {status_text}")
            else:
                print("⚠️  Command sent but status didn't change as expected")
                print("   This may be normal if:")
                print("   - Valve was already in requested state")
                print("   - Firmware doesn't support serial commands")
        else:
            print("\n⚠️  Could not verify valve status")
            print("   Command was sent but no response received")
        
        print("\n" + "="*70 + "\n")
        return True
        
    except serial.SerialException as e:
        print(f"\n❌ Error connecting to {port}: {e}")
        print("\nTroubleshooting:")
        print("  • Check ESP32 is connected via USB")
        print("  • Close other programs using COM7")
        print("  • Try unplugging and replugging ESP32\n")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}\n")
        return False

def main():
    if len(sys.argv) < 2:
        print("\n" + "="*70)
        print("USB SERIAL VALVE CONTROLLER")
        print("="*70)
        print("\nUsage:")
        print("  python usb_valve_control.py on      - Turn valve ON (open)")
        print("  python usb_valve_control.py off     - Turn valve OFF (close)")
        print("  python usb_valve_control.py open    - Turn valve ON (open)")
        print("  python usb_valve_control.py close   - Turn valve OFF (close)")
        print("  python usb_valve_control.py status  - Check valve status")
        print("\nExamples:")
        print("  python usb_valve_control.py open")
        print("  python usb_valve_control.py close")
        print("\n" + "="*70 + "\n")
        sys.exit(1)
    
    command = sys.argv[1]
    send_valve_command(command)

if __name__ == "__main__":
    main()
