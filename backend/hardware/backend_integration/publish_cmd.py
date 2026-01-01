# hardware/backend_integration/publish_cmd.py
import json
import paho.mqtt.client as mqtt
b = mqtt.Client()
b.connect("localhost",1883,60)
b.publish("farmfederate/control/relay", "ON")
print("Relay ON")
input("Press Enter to turn OFF")
b.publish("farmfederate/control/relay", "OFF")
print("Relay OFF")
b.disconnect()
