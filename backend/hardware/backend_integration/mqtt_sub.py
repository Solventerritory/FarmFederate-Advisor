import paho.mqtt.client as mqtt

def on_message(client, userdata, msg):
    import os
    import json
    from datetime import datetime
    print(f"{msg.topic} {msg.payload.decode()}")
    # Try to parse payload as JSON
    try:
        data = json.loads(msg.payload.decode())
        # Ensure the directory exists
        sensors_dir = os.path.join(os.path.dirname(__file__), '../../checkpoints_paper/ingest/sensors')
        sensors_dir = os.path.abspath(sensors_dir)
        os.makedirs(sensors_dir, exist_ok=True)
        # Save with timestamp in filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'sensor_{timestamp}.json'
        filepath = os.path.join(sensors_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved sensor data to {filepath}")
    except Exception as e:
        print(f"Failed to save sensor data: {e}")

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect("localhost", 1883, 60)
client.subscribe("farmfederate/sensors/#")
client.on_message = on_message
print("Subscribed to farmfederate/sensors/#. Waiting for messages...")
client.loop_forever()
