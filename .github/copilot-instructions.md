# FarmFederate-Advisor AI Coding Instructions

## Project Overview
FarmFederate-Advisor is a **federated multimodal learning system** for crop stress detection that combines:
- **Backend**: FastAPI server with PyTorch multimodal model (RoBERTa + ViT) for detecting 5 crop stress types
- **Frontend**: Flutter cross-platform app (iOS, Android, Web) for field diagnostics
"""Copilot instructions — FarmFederate-Advisor (concise)

This file gives AI coding agents the minimal, high-value facts to get productive quickly.

**Architecture (big picture)**
- **Backend**: FastAPI service in `backend/server.py` serving `POST /predict` using a PyTorch multimodal model defined in `backend/multimodal_model.py`.
- **Training / Federated**: scripts in `backend/` (`federated_core.py`, `train_fed_multimodal.py`) produce checkpoints under `checkpoints/` (`global_round{N}.pt`).
- **Frontend**: Flutter app in `frontend/` (main entry `frontend/lib/main.dart`) uses Firebase auth and uploads to `/predict` via `lib/services/api_service.dart`.
- **Hardware**: ESP32 sensors publish MQTT to `farmfederate/sensors/<client_id>`; listener saves JSON to `checkpoints_paper/sensors/`.

**Key conventions & gotchas**
- **Labels order**: `ISSUE_LABELS = ["water_stress","nutrient_def","pest_risk","disease_risk","heat_stress"]` (canonical order used across code).
- **Checkpoint formats**: `safe_load_checkpoint()` accepts `{model_state_dict}`, `{state_dict}`, or raw state dict — always confirm model keys match `multimodal_model`.
- **Sensor fusion**: sensor JSON is appended to text before tokenization (see `build_text_with_sensors()` / server code); there is no separate sensor encoder.
- **Env vars for server**: `CHECKPOINT_PATH`, `TEXT_MODEL_NAME`, `IMAGE_MODEL_NAME`, `MAX_LEN`, `IMG_SIZE`, `DEVICE`.
- **Flutter backend URL**: sometimes set in `frontend/lib/main.dart` and `lib/constants.dart` — prefer `lib/constants.dart` for edits.

**Essential commands (copyable)**
- Run backend (Windows PowerShell):
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
$env:CHECKPOINT_PATH="checkpoints/global_central.pt"
uvicorn server:app --host 0.0.0.0 --port 8000
```
- Train (local / quick):
```powershell
cd backend
python train_fed_multimodal.py
```
- Run Flutter web (dev):
```powershell
cd frontend
flutter pub get
flutter run -d chrome
```

**Files to inspect first when debugging**
- `backend/server.py` — request handling, checkpoint loading, sensor fusion, `ADVICE` dict
- `backend/multimodal_model.py` — model architecture, expected state_dict keys
- `backend/datasets_loader.py` — label list, dataset functions, synthetic data helpers
- `frontend/lib/services/api_service.dart` — how `/predict` is called
- `backend/hardware/*` — ESP32 sketches and MQTT config

**Integration examples to reuse in patches**
- To add a new label: update `ISSUE_LABELS` in `backend/datasets_loader.py`, regenerate data, retrain — server and Flutter UI expect lengths to match.
- To change encoder: set `TEXT_MODEL_NAME` / `IMAGE_MODEL_NAME` env var, but retrain or convert checkpoints (architectures must match).

**Testing snippets**
- Backend predict (PowerShell):
```powershell
$body = @{ text = "leaves yellowing"; sensor_data = '{"temp":28,"hum":65}' }
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body $body
```
- MQTT test: run `backend/hardware/backend_integration/send_test_sensor.py` (publishes to `farmfederate/sensors/test_client`).

If anything here is unclear or you want added examples (e.g., a small unit-test harness or a minimal reproducible training run), tell me which area to expand.

---
Updated: condensed core facts, commands, and pointers for quick agent onboarding.
"""
```powershell
