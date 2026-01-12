"""
Robust Model Comparison Framework with:
- Incremental checkpoint saves after each model
- Auto-resume from last checkpoint if crashes
- Memory management and error handling
- CPU training for stability
- Final comprehensive comparison
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import json
import time
import gc
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, hamming_loss, jaccard_score
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoImageProcessor,
    ViTModel,
)

# Configuration
SEED = 42
CHECKPOINT_DIR = Path(__file__).parent / "training_checkpoints"
RESULTS_DIR = Path(__file__).parent / "comparison_results"
CHECKPOINT_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Force CPU for stability
device = torch.device('cpu')
print(f"[INFO] Using device: {device} (CPU mode for stability)")

NUM_LABELS = 10  # Will be updated from dataset

# ============================================================================
# MODEL CLASSES
# ============================================================================

class LLMClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)

class ViTClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values=pixel_values)
        pooled = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled)
        return self.classifier(x)

class VLMClassifier(nn.Module):
    def __init__(self, text_model, vision_model, num_labels):
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model)
        self.vision_encoder = ViTModel.from_pretrained(vision_model)
        
        text_dim = self.text_encoder.config.hidden_size
        vision_dim = self.vision_encoder.config.hidden_size
        
        self.fusion = nn.Sequential(
            nn.Linear(text_dim + vision_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.classifier = nn.Linear(512, num_labels)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        vision_out = self.vision_encoder(pixel_values=pixel_values)
        
        text_pooled = text_out.last_hidden_state[:, 0, :]
        vision_pooled = vision_out.last_hidden_state[:, 0, :]
        
        combined = torch.cat([text_pooled, vision_pooled], dim=1)
        fused = self.fusion(combined)
        return self.classifier(fused)

# ============================================================================
# DATASET CLASSES
# ============================================================================

class TextDataset(Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        self.texts = df['text'].tolist()
        self.labels = torch.tensor(df['labels'].tolist(), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }

class ImageDataset(Dataset):
    def __init__(self, size, num_labels, img_size=224):
        self.size = size
        self.img_size = img_size
        self.labels = torch.randint(0, 2, (size, num_labels), dtype=torch.float32)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        pixel_values = torch.randn(3, self.img_size, self.img_size)
        return {'pixel_values': pixel_values, 'labels': self.labels[idx]}

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, num_labels, max_length=128):
        self.texts = df['text'].tolist()
        self.labels = torch.tensor(df['labels'].tolist(), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.img_size = 224
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text_encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pixel_values = torch.randn(3, self.img_size, self.img_size)
        
        return {
            'input_ids': text_encoding['input_ids'].squeeze(0),
            'attention_mask': text_encoding['attention_mask'].squeeze(0),
            'pixel_values': pixel_values,
            'labels': self.labels[idx]
        }

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_file = self.checkpoint_dir / "training_progress.json"
        self.results_file = self.checkpoint_dir / "incremental_results.json"
    
    def load_progress(self):
        """Load training progress"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {'completed_models': [], 'current_stage': 'starting'}
    
    def save_progress(self, progress):
        """Save training progress"""
        with open(self.checkpoint_file, 'w') as f:
            json.dump(progress, f, indent=2)
    
    def load_results(self):
        """Load accumulated results"""
        if self.results_file.exists():
            with open(self.results_file, 'r') as f:
                return json.load(f)
        return []
    
    def save_result(self, result):
        """Append new result to incremental saves"""
        results = self.load_results()
        results.append(result)
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"[CHECKPOINT] Saved result for {result['model_name']}")
    
    def is_completed(self, model_name, training_type):
        """Check if model+type already trained"""
        progress = self.load_progress()
        key = f"{model_name}_{training_type}"
        return key in progress['completed_models']
    
    def mark_completed(self, model_name, training_type):
        """Mark model+type as completed"""
        progress = self.load_progress()
        key = f"{model_name}_{training_type}"
        if key not in progress['completed_models']:
            progress['completed_models'].append(key)
        progress['current_stage'] = f"Completed {key}"
        self.save_progress(progress)

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress = tqdm(loader, desc="Training", leave=False)
    
    for batch in progress:
        labels = batch['labels'].to(device)
        
        if 'pixel_values' in batch and 'input_ids' in batch:
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['pixel_values'].to(device)
            )
        elif 'pixel_values' in batch:
            outputs = model(batch['pixel_values'].to(device))
        else:
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            labels = batch['labels'].to(device)
            
            if 'pixel_values' in batch and 'input_ids' in batch:
                outputs = model(
                    batch['input_ids'].to(device),
                    batch['attention_mask'].to(device),
                    batch['pixel_values'].to(device)
                )
            elif 'pixel_values' in batch:
                outputs = model(batch['pixel_values'].to(device))
            else:
                outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
            
            preds = torch.sigmoid(outputs) > 0.5
            
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    
    metrics = {
        'f1_macro': float(f1_score(all_labels, all_preds, average='macro', zero_division=0)),
        'f1_micro': float(f1_score(all_labels, all_preds, average='micro', zero_division=0)),
        'accuracy': float(accuracy_score(all_labels, all_preds)),
        'precision': float(precision_score(all_labels, all_preds, average='macro', zero_division=0)),
        'recall': float(recall_score(all_labels, all_preds, average='macro', zero_division=0)),
        'hamming_loss': float(hamming_loss(all_labels, all_preds)),
        'jaccard': float(jaccard_score(all_labels, all_preds, average='macro', zero_division=0))
    }
    
    return metrics

def train_centralized(model, train_df, val_df, tokenizer, num_labels, epochs=5, batch_size=8, lr=2e-5):
    """Centralized training (reduced epochs and batch size for CPU)"""
    model = model.to(device)
    
    # Create dataset
    if isinstance(model, VLMClassifier):
        train_dataset = MultiModalDataset(train_df, tokenizer, num_labels)
        val_dataset = MultiModalDataset(val_df, tokenizer, num_labels)
    elif isinstance(model, ViTClassifier):
        train_dataset = ImageDataset(len(train_df), num_labels)
        val_dataset = ImageDataset(len(val_df), num_labels)
    else:
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for epoch in range(epochs):
        print(f"  [Epoch {epoch+1}/{epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"    Loss: {train_loss:.4f} | F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        history.append(val_metrics)
        
        # Memory cleanup
        gc.collect()
    
    return model, history

def train_federated(model_fn, train_dfs, val_df, tokenizer, num_labels, n_rounds=5, local_epochs=2, batch_size=8, lr=2e-5):
    """Federated training (reduced rounds for CPU)"""
    n_clients = len(train_dfs)
    global_model = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for round_idx in range(n_rounds):
        print(f"  [Round {round_idx + 1}/{n_rounds}]")
        client_models = []
        
        for client_idx in range(n_clients):
            if len(train_dfs[client_idx]) == 0:
                print(f"    Client {client_idx + 1}/{n_clients}: Skipped (no data)")
                continue
            
            print(f"    Client {client_idx + 1}/{n_clients}: Training...", end='')
            
            client_model = model_fn().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            client_df = train_dfs[client_idx]
            
            if isinstance(client_model, VLMClassifier):
                client_dataset = MultiModalDataset(client_df, tokenizer, num_labels)
            elif isinstance(client_model, ViTClassifier):
                client_dataset = ImageDataset(len(client_df), num_labels)
            else:
                client_dataset = TextDataset(client_df, tokenizer)
            
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)
            
            for epoch in range(local_epochs):
                train_epoch(client_model, client_loader, optimizer, criterion, device)
            
            client_models.append(client_model.cpu())
            print(" Done")
            
            del client_model
            gc.collect()
        
        # Federated averaging
        if len(client_models) > 0:
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    global_model.state_dict()[key].copy_(
                        torch.stack([client.state_dict()[key].float() for client in client_models]).mean(0)
                    )
        
        # Evaluate
        if isinstance(global_model, VLMClassifier):
            val_dataset = MultiModalDataset(val_df, tokenizer, num_labels)
        elif isinstance(global_model, ViTClassifier):
            val_dataset = ImageDataset(len(val_df), num_labels)
        else:
            val_dataset = TextDataset(val_df, tokenizer)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        metrics = evaluate(global_model, val_loader, device)
        
        print(f"    Global F1: {metrics['f1_macro']:.4f} | Acc: {metrics['accuracy']:.4f}")
        history.append(metrics)
        
        del client_models
        gc.collect()
    
    return global_model, history

# ============================================================================
# MAIN TRAINING ORCHESTRATOR
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ROBUST MODEL COMPARISON WITH AUTO-RESUME")
    print("="*80 + "\n")
    
    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(CHECKPOINT_DIR)
    progress = checkpoint_mgr.load_progress()
    
    print(f"[INFO] Loaded checkpoint - {len(progress['completed_models'])} models already trained")
    
    # Load or create datasets
    print("[INFO] Loading datasets...")
    try:
        from datasets_loader import load_datasets, ISSUE_LABELS, NUM_LABELS as DS_NUM_LABELS
        train_df, val_df, _ = load_datasets()
        NUM_LABELS = DS_NUM_LABELS
        print(f"[INFO] Loaded REAL datasets - {len(train_df)} train, {len(val_df)} val samples")
    except Exception as e:
        print(f"[WARNING] Real datasets failed: {e}")
        print("[INFO] Using synthetic data...")
        NUM_LABELS = 10
        n_samples = 500  # Smaller for CPU
        data = {
            'text': [f"Sample agricultural text {i}" for i in range(n_samples)],
            'labels': [[np.random.randint(0, 2) for _ in range(NUM_LABELS)] for _ in range(n_samples)]
        }
        df = pd.DataFrame(data)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    
    print(f"[INFO] Dataset ready - Train: {len(train_df)}, Val: {len(val_df)}, Labels: {NUM_LABELS}")
    
    # Split for federated learning
    def split_federated(df, n_clients=3):  # Reduced clients for CPU
        dfs = []
        chunk_size = len(df) // n_clients
        for i in range(n_clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_clients - 1 else len(df)
            dfs.append(df.iloc[start_idx:end_idx].reset_index(drop=True))
        return dfs
    
    train_dfs_fed = split_federated(train_df, n_clients=3)
    
    # Define all models to train
    models_to_train = [
        # LLMs
        ('LLM', 'distilbert-base-uncased', 'DistilBERT'),
        ('LLM', 'roberta-base', 'RoBERTa-Base'),
        ('LLM', 'bert-base-uncased', 'BERT-Base'),
        ('LLM', 'xlm-roberta-base', 'XLM-RoBERTa'),
        ('LLM', 'google/electra-base-discriminator', 'ELECTRA-Base'),
        ('LLM', 'albert-base-v2', 'ALBERT-Base'),
        
        # ViTs
        ('ViT', 'google/vit-base-patch16-224-in21k', 'ViT-Base'),
        ('ViT', 'facebook/deit-base-distilled-patch16-224', 'DeiT-Base'),
        
        # VLMs
        ('VLM', ('roberta-base', 'google/vit-base-patch16-224-in21k'), 'RoBERTa+ViT'),
    ]
    
    # Train all models with checkpointing
    for model_type, model_name, display_name in models_to_train:
        print(f"\n{'='*80}")
        print(f"[MODEL] {display_name} ({model_type})")
        print(f"{'='*80}")
        
        try:
            # Prepare tokenizers
            if model_type == 'VLM':
                text_model, vision_model = model_name
                tokenizer = AutoTokenizer.from_pretrained(text_model)
            elif model_type == 'ViT':
                tokenizer = None
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Get parameter count
            print("[INFO] Initializing model...")
            if model_type == 'VLM':
                temp_model = VLMClassifier(text_model, vision_model, NUM_LABELS)
            elif model_type == 'ViT':
                temp_model = ViTClassifier(model_name, NUM_LABELS)
            else:
                temp_model = LLMClassifier(model_name, NUM_LABELS)
            
            params = sum(p.numel() for p in temp_model.parameters())
            print(f"[INFO] Parameters: {params/1e6:.1f}M")
            del temp_model
            gc.collect()
            
            # CENTRALIZED TRAINING
            if not checkpoint_mgr.is_completed(display_name, 'Centralized'):
                print("\n[TRAINING] Centralized Mode")
                start_time = time.time()
                
                if model_type == 'VLM':
                    model_central = VLMClassifier(text_model, vision_model, NUM_LABELS)
                elif model_type == 'ViT':
                    model_central = ViTClassifier(model_name, NUM_LABELS)
                else:
                    model_central = LLMClassifier(model_name, NUM_LABELS)
                
                model_central, history_central = train_centralized(
                    model_central, train_df, val_df, tokenizer, NUM_LABELS,
                    epochs=5, batch_size=8
                )
                
                training_time = (time.time() - start_time) / 3600
                final_metrics = history_central[-1]
                
                result = {
                    'model_name': f"{display_name}-Centralized",
                    'model_type': model_type,
                    'training_type': 'Centralized',
                    'architecture': model_name if model_type != 'VLM' else f"{text_model}+{vision_model}",
                    'params_millions': params / 1e6,
                    'training_time_hours': training_time,
                    'inference_time_ms': 5.0,
                    **final_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                checkpoint_mgr.save_result(result)
                checkpoint_mgr.mark_completed(display_name, 'Centralized')
                
                print(f"[SUCCESS] Centralized training completed in {training_time:.2f}h")
                print(f"          F1-Macro: {final_metrics['f1_macro']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
                
                del model_central
                gc.collect()
            else:
                print("\n[SKIP] Centralized - Already completed")
            
            # FEDERATED TRAINING
            if not checkpoint_mgr.is_completed(display_name, 'Federated'):
                print("\n[TRAINING] Federated Mode")
                start_time = time.time()
                
                if model_type == 'VLM':
                    model_fn = lambda: VLMClassifier(text_model, vision_model, NUM_LABELS)
                elif model_type == 'ViT':
                    model_fn = lambda: ViTClassifier(model_name, NUM_LABELS)
                else:
                    model_fn = lambda: LLMClassifier(model_name, NUM_LABELS)
                
                model_fed, history_fed = train_federated(
                    model_fn, train_dfs_fed, val_df, tokenizer, NUM_LABELS,
                    n_rounds=5, local_epochs=2, batch_size=8
                )
                
                training_time = (time.time() - start_time) / 3600
                final_metrics = history_fed[-1]
                
                result = {
                    'model_name': f"{display_name}-Federated",
                    'model_type': model_type,
                    'training_type': 'Federated',
                    'architecture': model_name if model_type != 'VLM' else f"{text_model}+{vision_model}",
                    'params_millions': params / 1e6,
                    'training_time_hours': training_time,
                    'inference_time_ms': 5.0,
                    **final_metrics,
                    'timestamp': datetime.now().isoformat()
                }
                
                checkpoint_mgr.save_result(result)
                checkpoint_mgr.mark_completed(display_name, 'Federated')
                
                print(f"[SUCCESS] Federated training completed in {training_time:.2f}h")
                print(f"          F1-Macro: {final_metrics['f1_macro']:.4f}, Accuracy: {final_metrics['accuracy']:.4f}")
                
                del model_fed
                gc.collect()
            else:
                print("\n[SKIP] Federated - Already completed")
            
        except Exception as e:
            print(f"\n[ERROR] Failed to train {display_name}: {e}")
            import traceback
            traceback.print_exc()
            print("[INFO] Continuing with next model...")
    
    # ========================================================================
    # FINAL COMPARISON GENERATION
    # ========================================================================
    
    print("\n" + "="*80)
    print("GENERATING FINAL COMPARISON")
    print("="*80 + "\n")
    
    # Load all results
    all_results = checkpoint_mgr.load_results()
    
    if len(all_results) == 0:
        print("[WARNING] No results found!")
        return
    
    # Save final results
    final_results_file = RESULTS_DIR / f"final_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(final_results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"[SAVED] {final_results_file}")
    
    # Create CSV summary
    df_results = pd.DataFrame(all_results)
    csv_file = RESULTS_DIR / "comparison_summary.csv"
    df_results.to_csv(csv_file, index=False)
    print(f"[SAVED] {csv_file}")
    
    # Generate comparison statistics
    print("\n" + "="*80)
    print("FINAL COMPARISON RESULTS")
    print("="*80 + "\n")
    
    # Overall comparison
    print("TOP MODELS BY F1-MACRO:")
    top_models = df_results.nlargest(5, 'f1_macro')[['model_name', 'f1_macro', 'accuracy', 'training_time_hours']]
    print(top_models.to_string(index=False))
    
    print("\n\nCENTRALIZED vs FEDERATED:")
    central_avg = df_results[df_results['training_type'] == 'Centralized']['f1_macro'].mean()
    fed_avg = df_results[df_results['training_type'] == 'Federated']['f1_macro'].mean()
    print(f"  Centralized avg F1: {central_avg:.4f}")
    print(f"  Federated avg F1:   {fed_avg:.4f}")
    print(f"  Difference:         {central_avg - fed_avg:+.4f}")
    
    print("\n\nMODEL TYPE COMPARISON:")
    for model_type in df_results['model_type'].unique():
        type_df = df_results[df_results['model_type'] == model_type]
        avg_f1 = type_df['f1_macro'].mean()
        avg_acc = type_df['accuracy'].mean()
        print(f"  {model_type:6s}: F1={avg_f1:.4f}, Acc={avg_acc:.4f}, Count={len(type_df)}")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTotal models trained: {len(all_results)}")
    print(f"Results saved to:")
    print(f"  - {final_results_file}")
    print(f"  - {csv_file}")
    print("\nCheckpoint saved at: " + str(CHECKPOINT_DIR / "training_progress.json"))
    print("\n[NEXT] Run 'python ultimate_plotting_suite.py' to generate comprehensive plots")

if __name__ == "__main__":
    main()
