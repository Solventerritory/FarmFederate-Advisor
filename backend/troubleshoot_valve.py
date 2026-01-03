import serial
import json
import time

print("\n" + "="*70)
print("🔍 VALVE TROUBLESHOOTING - CHECKING HARDWARE")
print("="*70 + "\n")

ser = serial.Serial('COM7', 115200, timeout=2)
time.sleep(1)

data = None
for _ in range(12):
    line = ser.readline().decode('utf-8', errors='ignore').strip()
    if 'DATA:' in line:
        try:
            data = json.loads(line.split('DATA:')[1])
            break
        except:
            pass

ser.close()

if data:
    relay = data.get('relay_state')
    flow = data.get('flow_rate', 0)
    
    print("="*70)
    print("ELECTRICAL STATUS")
    print("="*70)
    print(f"\n🔌 Relay GPIO Pin: {'✅ ON (Energized)' if relay else '❌ OFF (De-energized)'}")
    print(f"   Expected: GPIO 5 should be HIGH (3.3V)")
    print(f"   Status: {'Signal sent to valve' if relay else 'No signal to valve'}")
    
    print("\n" + "="*70)
    print("PHYSICAL WATER FLOW")
    print("="*70)
    print(f"\n💦 Flow Sensor: {flow} L/min")
    
    if relay and flow == 0:
        print("   ⚠️  PROBLEM: Relay is ON but no water flowing!")
        print("\n" + "="*70)
        print("TROUBLESHOOTING STEPS")
        print("="*70)
        print("\n1. CHECK WATER SUPPLY:")
        print("   • Is water source turned on?")
        print("   • Is there water pressure available?")
        print("   • Check inlet hose is connected")
        
        print("\n2. CHECK SOLENOID VALVE:")
        print("   • Listen for 'click' sound when opening")
        print("   • Valve may be stuck or malfunctioning")
        print("   • Check valve is correctly wired to relay")
        
        print("\n3. CHECK RELAY MODULE:")
        print("   • LED on relay should be ON when activated")
        print("   • Relay may be defective")
        print("   • Check wiring: GPIO 5 → Relay IN")
        
        print("\n4. CHECK CONNECTIONS:")
        print("   • Relay GND → ESP32 GND")
        print("   • Relay VCC → ESP32 5V or 3.3V")
        print("   • Relay NO/NC → Solenoid valve wires")
        
        print("\n5. TEST RELAY MANUALLY:")
        print("   • Use multimeter to test relay output")
        print("   • Measure voltage across relay terminals")
        print("   • Should show ~12V when relay ON")
        
        print("\n6. VALVE SPECIFICATIONS:")
        print("   • Check valve voltage rating (12V DC typical)")
        print("   • Verify relay can handle valve current")
        print("   • Some valves need 24V DC")
        
    elif relay and flow > 0:
        print("   ✅ Water is flowing normally!")
        print(f"   Flow rate: {flow} L/min")
    elif not relay:
        print("   ℹ️  Relay is OFF - valve should be closed")
    
    print("\n" + "="*70)
    print("SENSOR READINGS")
    print("="*70)
    print(f"\n🌡️  Temperature: {data.get('temperature')}°C")
    print(f"💧 Humidity: {data.get('humidity')}%")
    print(f"🌱 Soil Moisture: {data.get('soil_moisture')}%")
    print(f"💦 Flow Rate: {flow} L/min")
    
    print("\n" + "="*70)
    print("WIRING DIAGRAM REFERENCE")
    print("="*70)
    print("""
ESP32 → Relay Module:
  GPIO 5 → Relay IN
  GND → Relay GND
  5V → Relay VCC

Relay Module → Solenoid Valve:
  Relay COM → Valve power (+12V)
  Relay NO → Valve terminal
  Valve GND → Power supply GND

Note: NO = Normally Open (valve opens when relay energized)
      NC = Normally Closed (valve closes when relay energized)
""")
    
else:
    print("❌ No data received from ESP32")
    print("   Check USB connection and try again")

print("="*70 + "\n")
