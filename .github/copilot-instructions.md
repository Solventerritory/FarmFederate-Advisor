# FarmFederate-Advisor AI Coding Instructions

## Project Overview
FarmFederate-Advisor is a **federated multimodal learning system** for crop stress detection that combines:
- **Backend**: FastAPI server with PyTorch multimodal model (RoBERTa + ViT) for detecting 5 crop stress types
- **Frontend**: Flutter cross-platform app (iOS, Android, Web) for field diagnostics
- **Hardware**: ESP32/ESP32-CAM sensor nodes + MQTT broker for IoT telemetry
- **ML Pipeline**: Federated learning with LoRA adapters, class imbalance handling (FocalLoss), and Dirichlet data splits

## Architecture & Data Flows

### Core Components
1. **Multimodal Classifier** (`backend/multimodal_model.py`)
   - Text encoder: `roberta-base` → 256-d projection
   - Image encoder: `google/vit-base-patch16-224-in21k` → 256-d projection
   - Fusion: concat(text, image) → 512-d → MLP → 5 labels
   - Model aliases: `MultiModalModel` = `MultimodalClassifier` (both used interchangeably)
   - Attributes: `.text_encoder`, `.vision_encoder`, `.classifier` (required by PEFT and server)

2. **Federated Training** (`backend/federated_core.py`, `backend/train_fed_multimodal.py`)
   - Clients split via Dirichlet distribution (alpha=0.5 for heterogeneity)
   - FocalLoss for multi-label class imbalance (gamma=2.0, dynamic alpha per class)
   - Checkpoints saved as: `global_round{N}.pt` → contains `model_state_dict`, `thresholds`, `round_num`
   - Training expects: text DataFrame with `["text", "labels"]` + optional HF image dataset with `"image"` column

3. **FastAPI Server** (`backend/server.py`)
   - Endpoint: `POST /predict` (multipart: `text`, `sensor_data`, `image`)
   - Checkpoint loading: flexible parser handles `model_state_dict`, `state_dict`, or raw dicts
   - Sensor fusion: concatenates text + sensor JSON → single input string
   - Response: `{advice, scores: [{label, prob}], active_labels, logits}`
   - Env vars: `CHECKPOINT_PATH`, `TEXT_MODEL_NAME`, `IMAGE_MODEL_NAME`, `MAX_LEN`, `IMG_SIZE`

4. **Hardware Integration** (`backend/hardware/`)
   - ESP32 nodes publish to MQTT topic: `farmfederate/sensors/<client_id>`
   - JSON format: `{client_id, temp, hum, soil, flow_lpm, ds_temp}`
   - MQTT listener (`mqtt_listener.py`) saves to `checkpoints_paper/sensors/<client_id>.json`
   - ESP32-CAM uploads images via `POST /api/image_upload` (multipart form)

5. **Flutter App** (`frontend/`)
   - Main tabs: Chat (text+image prediction) + Hardware Dashboard (sensor polling)
   - **Authentication**: Firebase Auth with email/password (login/register screens)
   - API service: `lib/services/api_service.dart` (multipart form upload to `/predict`)
   - Auth service: `lib/services/auth_service.dart` (Firebase sign in/up/out, password reset)
   - MQTT service: `lib/services/mqtt_service.dart` (subscribes to `farm/sensors/#`)
   - Constants: `lib/constants.dart` (configure `DEFAULT_BACKEND`, `MQTT_BROKER`)
   - Platform handling: `main.dart` uses `dart:html` for web file/camera picker
   - Routes: Login → Register → Home (authenticated)

## Critical Workflows

### Running the Backend
```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate  # Windows
pip install -r requirements.txt
$env:CHECKPOINT_PATH="checkpoints/global_central.pt"
uvicorn server:app --host 0.0.0.0 --port 8000
```

### Training Federated Model
```powershell
cd backend
python train_fed_multimodal.py  # Outputs checkpoints/global_round{N}.pt
# For Colab: use farm_advisor_multimodal_full.ipynb
```
Key training params: `n_clients=3`, `rounds=2`, `epochs_per_round=2`, `lr=2e-5`, `batch_size=16`

### Running Flutter App
```powershell
cd frontend
flutter pub get

# IMPORTANT: Configure Firebase first! See frontend/FIREBASE_SETUP.md
# Replace YOUR_API_KEY, YOUR_APP_ID, etc. in lib/main.dart

flutter run -d chrome  # For web
flutter run -d <device_id>  # For iOS/Android
```
**IMPORTANT**: 
1. Update Firebase credentials in `lib/main.dart` (see `FIREBASE_SETUP.md`)
2. Update `lib/constants.dart` with actual backend IP before deploying to devices

### Firebase Authentication Setup
1. Create Firebase project at console.firebase.google.com
2. Enable Email/Password authentication
3. Add Web app and copy config to `main.dart`
4. See detailed guide: `frontend/FIREBASE_SETUP.md`

**Login Flow**: App starts at LoginScreen → user authenticates → navigates to HomePage (tabs)
- Login: `/` or `/login`
- Register: `/register`  
- Home (after auth): `/home` (contains Chat and Hardware tabs)
- Logout: Menu button in AppBar

### Hardware Setup
1. Flash ESP32: Edit WiFi/MQTT credentials in `backend/hardware/esp32_sensor_node/esp32_sensor_node.ino`
2. Start MQTT broker: `docker-compose up -d` in `backend/hardware/mqtt/`
3. Start MQTT listener: `python backend/mqtt_listener.py` (separate process from server)

## Project-Specific Conventions

### Issue Labels (Fixed Taxonomy)
All code references these 5 labels in order:
```python
ISSUE_LABELS = ["water_stress", "nutrient_def", "pest_risk", "disease_risk", "heat_stress"]
```
- Multi-label binary classification (not mutually exclusive)
- Thresholds: Default 0.5, but checkpoints store optimized per-label thresholds

### Checkpoint Format Flexibility
Server (`safe_load_checkpoint`) handles multiple formats:
- `{model_state_dict: {...}}` (federated training format)
- `{state_dict: {...}}` (common PyTorch format)
- Raw `OrderedDict` of tensors (direct `model.state_dict()` save)

**Always test checkpoint loading**: Model architecture must match (same encoder names, projection dims)

### Sensor Data Fusion Pattern
Text is augmented with sensor readings before tokenization:
```python
full_text = f"{user_text} Sensor: temp={temp}C, hum={hum}%, soil={soil}"
```
This is the ONLY way sensors influence predictions (no separate sensor encoder).

### Dataset Naming & Structure
- Text datasets loaded from HuggingFace: `CGIAR/gardian-ai-ready-docs`, `argilla/farming`, `ag_news` (filtered)
- Image datasets: `BrandonFors/Plant-Diseases-PlantVillage-Dataset` (PlantVillage), `Saon110/bd-crop-vegetable-plant-disease-dataset`
- Fallback: If all image datasets fail (timeout/permission), uses 224x224 gray dummy image
- Dataset loader: `datasets_loader.py` functions: `build_text_corpus_mix()`, `load_stress_image_datasets_hf()`

### Flutter Platform Handling
- Web: Uses `dart:html` for file picker (`html.InputElement` with `capture` attribute for camera)
- Mobile: Standard `image_picker` package (not shown in main.dart but expected pattern)
- Backend URL: Hardcoded in `main.dart` (`apiBase`) AND `constants.dart` (choose one location!)
- **Authentication**: Firebase SDK handles cross-platform auth (web, iOS, Android)
- **Session persistence**: Uses `shared_preferences` to store login state locally

## Common Pitfalls & Fixes

### Backend Issues
- **Import errors**: `MultimodalClassifier` vs `MultiModalModel` — both names valid, check imports
- **Checkpoint loading fails**: Check if dict has `model_state_dict` key; use `safe_load_checkpoint()`
- **CUDA OOM**: Reduce `batch_size=8`, set `$env:IMG_SIZE=196`, or force CPU with `$env:DEVICE="cpu"`
- **Missing priors**: `farm_advisor.py` optional; server works without it (no-op priors)

### Frontend Issues
- **Firebase not initialized**: Ensure Firebase.initializeApp() completes before runApp()
- **Authentication errors**: Verify Firebase project has Email/Password provider enabled
- **CORS errors**: Backend `CORSMiddleware` allows all origins; check if server running on correct port
- **Android emulator**: Use `http://10.0.2.2:8000` not `localhost:8000`
- **Web camera access**: Requires HTTPS in production (or `localhost` exception)
- **MQTT not connecting**: Ensure `MQTT_BROKER` uses WebSocket port (e.g., `:9001`) not raw TCP (`:1883`)
- **"User not found" on login**: Create test users in Firebase Console → Authentication → Users

### Hardware Issues
- **ESP32 not publishing**: Check Serial monitor for WiFi/MQTT errors; verify broker IP reachable
- **Sensor values always 0**: ESP32 ADC pins are 3.3V max; soil sensors may need 5V → use level shifter
- **MQTT listener crashes**: Ensure `checkpoints_paper/sensors/` directory exists (auto-created if missing)

## Integration Points

### Adding New Stress Labels
1. Update `ISSUE_LABELS` in `datasets_loader.py`
2. Regenerate synthetic data in `datasets_loader.py` (search for `generate_synthetic_agri_mini`)
3. Retrain model (will auto-resize classifier head to `len(ISSUE_LABELS)`)
4. Update `ADVICE` dict in `server.py` with remediation text
5. Update Flutter UI in `chat_screen.dart` to display new labels

### Adding New Sensor Types
1. Modify ESP32 sketch JSON payload in `esp32_sensor_node.ino`
2. Update `build_text_with_sensors()` in `server.py` to include new fields
3. Update Flutter `hardware_dashboard.dart` to display new sensor readings

### Model Swaps
- Replace text encoder: Change `TEXT_MODEL_NAME` env var (e.g., `"distilroberta-base"`)
- Replace image encoder: Change `IMAGE_MODEL_NAME` (e.g., `"microsoft/resnet-50"`)
- **Must retrain**: Pretrained checkpoints are incompatible across architecture changes

## Key Files Reference
- **Model definition**: `backend/multimodal_model.py` (181 lines, defines forward pass)
- **Server entrypoint**: `backend/server.py` (387 lines, FastAPI routes)
- **Federated training**: `backend/train_fed_multimodal.py` (launches client training)
- **Dataset builder**: `backend/datasets_loader.py` (550 lines, HF dataset downloaders)
- **Flutter main**: `frontend/lib/main.dart` (499 lines, chat + hardware tabs)
- **ESP32 firmware**: `backend/hardware/esp32_sensor_node/esp32_sensor_node.ino` (71 lines, MQTT publish loop)

## Testing Commands
```powershell
# Test backend prediction (PowerShell)
$body = @{ text = "leaves yellowing"; sensor_data = '{"temp":28,"hum":65}' }
Invoke-RestMethod -Uri http://localhost:8000/predict -Method POST -Body $body

# Test MQTT publish (Python)
cd backend/hardware/backend_integration
python send_test_sensor.py  # Publishes to farmfederate/sensors/test_client

# Run Flutter web in dev mode
cd frontend
flutter run -d chrome --web-hostname localhost --web-port 8080
```

## Dependencies & Versions
- **Python**: 3.9+ (tested on 3.11)
- **Firebase**: firebase_core ^2.24.2, firebase_auth ^4.15.3
- **PyTorch**: 2.0+ (CUDA 11.8 if GPU)
- **Transformers**: 4.30+
- **Flutter**: SDK >=2.19.0 <4.0.0
- **ESP32**: Arduino core 2.0.9+ or PlatformIO
- **MQTT Broker**: Eclipse Mosquitto 2.0+ (Docker recommended)
