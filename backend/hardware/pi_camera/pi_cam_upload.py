# pi_cam_upload.py
# Usage: python3 pi_cam_upload.py --url http://host:8000/api/image_upload --device_id pi-cam-01
import argparse, time, requests, io
import cv2

def capture_image():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Camera not found")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError("Failed capture")
    # encode as jpeg
    ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ret:
        raise RuntimeError("Encode failed")
    return buf.tobytes()

def upload_image(url, device_id, img_bytes):
    files = {
        'image': ('img.jpg', img_bytes, 'image/jpeg'),
    }
    data = {'device_id': device_id}
    r = requests.post(url, files=files, data=data, timeout=30)
    return r

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--url', required=True)
    p.add_argument('--device_id', default='pi-cam-01')
    p.add_argument('--interval', type=int, default=60)
    args = p.parse_args()

    while True:
        try:
            img = capture_image()
            resp = upload_image(args.url, args.device_id, img)
            print("Upload status", resp.status_code, resp.text)
        except Exception as e:
            print("Error:", e)
        time.sleep(args.interval)
