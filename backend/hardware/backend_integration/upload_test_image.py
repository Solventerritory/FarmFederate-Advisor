# upload_test_image.py
import requests, argparse

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--url', default='http://127.0.0.1:8000/api/image_upload')
    p.add_argument('--device', default='test-cam')
    p.add_argument('--file', default='test.jpg')
    args = p.parse_args()

    files = {'image': (args.file, open(args.file, 'rb'), 'image/jpeg')}
    data = {'device_id': args.device}
    r = requests.post(args.url, files=files, data=data, timeout=30)
    print(r.status_code, r.text)
