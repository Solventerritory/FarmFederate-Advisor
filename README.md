# FarmFederate-Advisor

FarmFederate integrates an ESP32-based field node, a FastAPI backend, a Flutter mobile app and a federated multimodal training pipeline (LoRA on CLIP) to detect crop stress and automate irrigation.

## Quick start

1. Backend
   - Create a Python venv in `/backend` and install requirements:
     ```
     cd backend
     python -m venv venv
     venv\Scripts\activate     # Windows
     source venv/bin/activate  # macOS/Linux
     pip install -r requirements.txt
     ```
   - Add model URLs to `backend/model_store/multimodal_demo/manifest.json` (from GitHub Releases or S3).
   - Set env: `BLYNK_TOKEN`, `BLYNK_HOST` (if using Blynk).
   - Run:
     ```bash
     uvicorn backend.server:app --host 0.0.0.0 --port 8000
     ```

2. ESP32 firmware
   - Edit `firmware/esp32/esp32_blynk_telemetry.ino` with WiFi, Blynk token, backend IP.
   - Upload via Arduino IDE / PlatformIO.

3. Flutter app
   - Open `frontend/farm_federate_flutter` and replace `YOUR_BACKEND_IP` in services.
   - Run `flutter pub get` and `flutter run`.

4. Training & models
   - Use `backend/agri_fed_multimodal.py` (train in Colab).
   - Upload model artifacts to GitHub Releases.
   - Update `manifest.json` with release asset URLs.
   - Backend will download them on startup.

## Notes
- Keep binary models out of git; use GitHub Releases or cloud storage.
- Add auth for /telemetry and /control in production.
