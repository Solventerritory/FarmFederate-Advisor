# backend/orchestrator.py
import os, subprocess, sys, shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT / "agri_fed_multimodal.py"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_STORE = ROOT / "model_store" / "multimodal_demo"
ADAPTER_DIR = ROOT / "backend_data" / "federated_llm_adapter"

def start_training(rounds=3, clients=4, image_dir=None):
    cmd = [sys.executable, str(TRAIN_SCRIPT), "--rounds", str(rounds), "--clients", str(clients), "--apply_lora_client"]
    if image_dir:
        cmd += ["--image_dir", str(image_dir)]
    print("Running training:", cmd)
    subprocess.check_call(cmd, cwd=str(ROOT))
    outs = sorted((OUTPUTS_DIR).glob("mmrun_*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not outs:
        print("No outputs found.")
        return
    latest = outs[0]
    models_dir = latest / "models"
    adapter = models_dir / "global_multimodal_adapter.bin"
    full = models_dir / "global_multimodal_full.pt"
    MODEL_STORE.mkdir(parents=True, exist_ok=True)
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    if adapter.exists():
        shutil.copy(adapter, ADAPTER_DIR / "global_adapter.pt")
    if full.exists():
        shutil.copy(full, MODEL_STORE / "multimodal.pt")
    print("Artifacts moved to model_store and backend_data.")
    return {"adapter": str(ADAPTER_DIR / "global_adapter.pt"), "full": str(MODEL_STORE / "multimodal.pt")}

if __name__ == "__main__":
    start_training()
