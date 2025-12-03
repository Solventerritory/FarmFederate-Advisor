# publish_cmd.py
# pip install paho-mqtt
import json
import argparse
import paho.mqtt.client as mqtt

def publish(broker, port, topic, payload, user=None, pwd=None):
    client = mqtt.Client()
    if user:
        client.username_pw_set(user, pwd)
    client.connect(broker, port, 60)
    client.loop_start()
    client.publish(topic, payload)
    client.loop_stop()
    client.disconnect()
    print("Published:", topic, payload)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--device", required=True)
    parser.add_argument("--cmd", default="RELAY_ON")
    args = parser.parse_args()
    topic = f"farm/{args.device}/commands"
    payload = json.dumps({"cmd": args.cmd})
    publish(args.broker, args.port, topic, payload)
