import serial
import json
import time

ser = serial.Serial('COM7', 115200, timeout=3)
time.sleep(1)

print("\n" + "="*60)
print("VALVE STATUS CHECK")
print("="*60)

valve_data = None
for _ in range(12):
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if 'DATA:' in line:
        try:
            valve_data = json.loads(line.split('DATA:')[1])
            break
        except:
            pass

ser.close()

if valve_data:
    relay = valve_data.get('relay_state')
    if relay:
        print("\n🟢 VALVE STATUS: OPEN (ON)")
        print("💦 WATER IS FLOWING!")
    else:
        print("\n🔴 VALVE STATUS: CLOSED (OFF)")
        print("🛑 Water flow stopped")
    
    print(f"\n📊 Sensors:")
    print(f"   🌡️  Temperature: {valve_data.get('temperature')}°C")
    print(f"   💧 Humidity: {valve_data.get('humidity')}%")
    print(f"   🌱 Soil Moisture: {valve_data.get('soil_moisture')}%")
    print(f"   💦 Flow Rate: {valve_data.get('flow_rate')} L/min")
else:
    print("\n⚠️  No data received from ESP32")
    print("   Check USB connection")

print("="*60 + "\n")
