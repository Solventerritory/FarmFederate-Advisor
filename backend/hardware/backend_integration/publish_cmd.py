# hardware/backend_integration/publish_cmd.py
import json
import paho.mqtt.client as mqtt
b = mqtt.Client()
b.connect("localhost",1883,60)
payload = {"client_id":"esp_test","soil_moisture":22.5,"temp":29.8,"humidity":68}
b.publish("farmfederate/sensors/esp_test", json.dumps(payload))
b.disconnect()
