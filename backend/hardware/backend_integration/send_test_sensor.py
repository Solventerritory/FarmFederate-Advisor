# send_test_sensor.py
# pip install requests
import requests, json, argparse, time

def send(url, payload):
    headers = {'Content-Type': 'application/json'}
    r = requests.post(url, data=json.dumps(payload), headers=headers, timeout=10)
    print("Status:", r.status_code, r.text)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/api/sensor_upload')
    p.add_argument('--device', default='test-node-1')
    args = p.parse_args()

    payload = {
        "device_id": args.device,
        "temp": 28.4,
        "hum": 62.1,
        "soil": 36.2,
        "flow_lpm": 0.0,
        "ds_temp": 28.0,
    }
    send(args.url, payload)
