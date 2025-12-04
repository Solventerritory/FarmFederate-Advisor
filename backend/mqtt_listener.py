# backend/mqtt_listener.py
import json, os
from paho.mqtt import client as mqtt
BROKER = os.environ.get("MQTT_HOST","localhost")
PORT = int(os.environ.get("MQTT_PORT",1883))
TOPIC = "farmfederate/sensors/#"

def on_connect(client, userdata, flags, rc):
    print("mqtt connected", rc)
    client.subscribe(TOPIC)

def on_message(client, userdata, msg):
    try:
        payload = msg.payload.decode("utf-8")
        data = json.loads(payload)
        cid = data.get("client_id","esp")
        p = os.path.join("checkpoints_paper","sensors", f"{cid}.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p,"w") as f:
            json.dump(data, f, indent=2)
        print("Saved sensor:", p)
    except Exception as e:
        print("mqtt message error", e)

if __name__=="__main__":
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(BROKER, PORT, 60)
    client.loop_forever()
