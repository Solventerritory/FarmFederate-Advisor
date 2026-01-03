#!/usr/bin/env python3
"""
Real-time Valve Monitor and Controller
Shows live valve status and allows manual control
"""
import serial
import json
import time
import sys
from datetime import datetime

def monitor_and_control():
    print("\n" + "="*70)
    print("🔌 REAL-TIME VALVE MONITOR & CONTROLLER")
    print("="*70)
    print("\nConnecting to ESP32 on COM7...")
    
    try:
        ser = serial.Serial('COM7', 115200, timeout=1)
        time.sleep(1)
        print("✓ Connected!\n")
        
        print("="*70)
        print("COMMANDS:")
        print("  Press 'O' = Open valve (turn ON)")
        print("  Press 'C' = Close valve (turn OFF)")
        print("  Press 'Q' = Quit")
        print("="*70)
        print("\nMonitoring valve status (updates every 5 seconds)...\n")
        
        last_relay_state = None
        
        while True:
            # Read serial data
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                
                if 'DATA:' in line:
                    try:
                        data = json.loads(line.split('DATA:')[1])
                        relay_state = data.get('relay_state', None)
                        temp = data.get('temperature')
                        humidity = data.get('humidity')
                        soil = data.get('soil_moisture')
                        flow = data.get('flow_rate')
                        
                        # Only print when relay state changes
                        if relay_state != last_relay_state and relay_state is not None:
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            status_icon = "🟢" if relay_state else "🔴"
                            status_text = "OPEN (ON)" if relay_state else "CLOSED (OFF)"
                            
                            print("\n" + "─"*70)
                            print(f"⏰ {timestamp}")
                            print(f"{status_icon} VALVE STATUS: {status_text}")
                            print(f"🌡️  Temp: {temp}°C | 💧 Humidity: {humidity}% | 🌱 Soil: {soil}%")
                            if relay_state:
                                print("💦 WATER IS FLOWING!")
                            else:
                                print("🛑 Water flow stopped")
                            print("─"*70)
                            
                            last_relay_state = relay_state
                    
                    except json.JSONDecodeError:
                        pass
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n\n✓ Monitoring stopped")
        ser.close()
    except serial.SerialException as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("  • ESP32 is connected to COM7")
        print("  • No other programs are using COM7")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    monitor_and_control()
