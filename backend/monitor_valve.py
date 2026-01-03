import serial
import json
import time
from datetime import datetime

print("\n" + "="*70)
print("🔌 VALVE STATUS MONITOR")
print("="*70)
print("\nMonitoring ESP32 for valve state changes...")
print("Will watch for 30 seconds or until valve opens\n")

ser = serial.Serial('COM7', 115200, timeout=1)
time.sleep(1)

start_time = time.time()
last_relay_state = None
valve_opened = False

while time.time() - start_time < 30:
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    
    if 'DATA:' in line:
        try:
            data = json.loads(line.split('DATA:')[1])
            relay_state = data.get('relay_state')
            
            if relay_state != last_relay_state:
                timestamp = datetime.now().strftime("%H:%M:%S")
                
                if relay_state:
                    print(f"🟢 {timestamp} - VALVE OPENED!")
                    print(f"   💦 Water is now flowing!")
                    valve_opened = True
                    break
                else:
                    print(f"🔴 {timestamp} - Valve is CLOSED")
                
                last_relay_state = relay_state
            
        except:
            pass
    
    elapsed = int(time.time() - start_time)
    if elapsed % 5 == 0 and elapsed > 0:
        status = "OPEN" if last_relay_state else "CLOSED" if last_relay_state is not None else "UNKNOWN"
        print(f"   ⏱️  {elapsed}s - Status: {status}")
    
    time.sleep(0.5)

ser.close()

print("\n" + "="*70)
print("FINAL STATUS")
print("="*70)

if valve_opened:
    print("\n✅ SUCCESS! Valve opened during monitoring!")
elif last_relay_state:
    print("\n✅ Valve was already OPEN")
elif last_relay_state == False:
    print("\n❌ Valve remains CLOSED")
    print("\nPossible reasons:")
    print("  • ESP32 not receiving MQTT commands (WiFi not connected)")
    print("  • USB-only firmware doesn't support MQTT control")
    print("  • Need to use WiFi-enabled firmware for remote control")
else:
    print("\n⚠️  No data received from ESP32")

print("\n" + "="*70 + "\n")
