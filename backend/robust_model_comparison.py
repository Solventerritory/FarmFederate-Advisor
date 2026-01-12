"""
Robust Model Comparison Framework with Checkpoint Support
- Incremental saves after each model
- Automatic resume from checkpoint
- CPU-optimized for stability
- Better memory management
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
    ViTModel
)

from datasets_loader import ISSUE_LABELS, NUM_LABELS

# Configuration
SEED = 42
RESULTS_DIR = Path(__file__).parent / "comparison_results"
CHECKPOINT_FILE = RESULTS_DIR / "training_checkpoint.json"
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# Force CPU to avoid memory issues
device = torch.device('cpu')
print(f"[INFO] Using device: {device} (CPU mode for stability)")

# ============================================================================
# MODEL CLASSES
# ============================================================================

class LLMClassifier(nn.Module):
    def __init__(self, model_name, num_labels=NUM_LABELS):
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
    def __init__(self, model_name, num_labels=NUM_LABELS):
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
    def __init__(self, text_model, vision_model, num_labels=NUM_LABELS):
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
    def __init__(self, size=1000, img_size=224):
        self.size = size
        self.img_size = img_size
        self.labels = torch.randint(0, 2, (size, NUM_LABELS), dtype=torch.float32)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        pixel_values = torch.randn(3, self.img_size, self.img_size)
        return {'pixel_values': pixel_values, 'labels': self.labels[idx]}

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, max_length=128):
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
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
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
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
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

def train_centralized(model, train_df, val_df, tokenizer, image_processor, epochs=5, batch_size=8):
    """Reduced epochs and batch size for CPU training"""
    model = model.to(device)
    
    if isinstance(model, VLMClassifier):
        train_dataset = MultiModalDataset(train_df, tokenizer, image_processor)
        val_dataset = MultiModalDataset(val_df, tokenizer, image_processor)
    elif isinstance(model, ViTClassifier):
        train_dataset = ImageDataset(size=len(train_df))
        val_dataset = ImageDataset(size=len(val_df))
    else:
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"  Loss: {train_loss:.4f} | Val F1: {val_metrics['f1_macro']:.4f} | Acc: {val_metrics['accuracy']:.4f}")
        
        history.append(val_metrics)
    
    return model, history

def train_federated(model_fn, train_dfs, val_df, tokenizer, image_processor, n_rounds=5, local_epochs=2, batch_size=8):
    """Reduced rounds for faster completion"""
    n_clients = len(train_dfs)
    global_model = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for round_idx in range(n_rounds):
        print(f"\n[Round {round_idx + 1}/{n_rounds}]")
        client_models = []
        client_weights = []
        
        for client_idx in range(n_clients):
            if len(train_dfs[client_idx]) == 0:
                continue
            
            print(f"  Client {client_idx + 1}/{n_clients}...", end=" ")
            
            client_model = model_fn().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            client_df = train_dfs[client_idx]
            
            if isinstance(client_model, VLMClassifier):
                client_dataset = MultiModalDataset(client_df, tokenizer, image_processor)
            elif isinstance(client_model, ViTClassifier):
                client_dataset = ImageDataset(size=len(client_df))
            else:
                client_dataset = TextDataset(client_df, tokenizer)
            
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=2e-5)
            
            for epoch in range(local_epochs):
                train_epoch(client_model, client_loader, optimizer, criterion, device)
            
            client_models.append(client_model.cpu())
            client_weights.append(len(client_df))
            print("Done")
            
            del client_model
            gc.collect()
        
        # Federated averaging
        if len(client_models) > 0:
            total_weight = sum(client_weights)
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    weighted_sum = torch.zeros_like(global_model.state_dict()[key])
                    for client, weight in zip(client_models, client_weights):
                        weighted_sum += client.state_dict()[key].float() * (weight / total_weight)
                    global_model.state_dict()[key].copy_(weighted_sum)
        
        # Evaluate
        if isinstance(global_model, VLMClassifier):
            val_dataset = MultiModalDataset(val_df, tokenizer, image_processor)
        elif isinstance(global_model, ViTClassifier):
            val_dataset = ImageDataset(size=len(val_df))
        else:
            val_dataset = TextDataset(val_df, tokenizer)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
        metrics = evaluate(global_model, val_loader, device)
        
        print(f"  Global F1: {metrics['f1_macro']:.4f}, Acc: {metrics['accuracy']:.4f}")
        history.append(metrics)
        
        del client_models
        gc.collect()
    
    return global_model, history

# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def load_checkpoint():
    """Load training checkpoint"""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'results': []}

def save_checkpoint(checkpoint):
    """Save training checkpoint"""
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump(checkpoint, f, indent=2)

def save_incremental_results(results):
    """Save results incrementally"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save JSON
    results_file = RESULTS_DIR / f"results_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save CSV
    df = pd.DataFrame(results)
    csv_file = RESULTS_DIR / f"results_{timestamp}.csv"
    df.to_csv(csv_file, index=False)
    
    print(f"[SAVED] Results saved to {csv_file.name}")

# ============================================================================
# MAIN TRAINING
# ============================================================================

def main():
    print("\n" + "="*80)
    print("ROBUST MODEL COMPARISON WITH CHECKPOINT SUPPORT")
    print("="*80 + "\n")
    
    # Load checkpoint
    checkpoint = load_checkpoint()
    completed_models = set(checkpoint['completed'])
    results = checkpoint['results']
    
    if completed_models:
        print(f"[RESUME] Found {len(completed_models)} completed models: {completed_models}")
    
    # Create synthetic dataset
    print("[INFO] Creating synthetic dataset...")
    n_samples = 1000
    data = {
        'text': [f"Agricultural sample text number {i} about crop health and farming" for i in range(n_samples)],
        'labels': [[np.random.randint(0, 2) for _ in range(NUM_LABELS)] for _ in range(n_samples)]
    }
    df = pd.DataFrame(data)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Split for federated (3 clients for faster training)
    def split_federated(df, n_clients=3):
        dfs = []
        chunk_size = len(df) // n_clients
        for i in range(n_clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_clients - 1 else len(df)
            dfs.append(df.iloc[start_idx:end_idx].reset_index(drop=True))
        return dfs
    
    train_dfs_fed = split_federated(train_df, n_clients=3)
    
    # Models to train (comprehensive list)
    models_to_train = [
        ('LLM', 'roberta-base', 'RoBERTa-Base'),
        ('LLM', 'bert-base-uncased', 'BERT-Base'),
        ('LLM', 'distilbert-base-uncased', 'DistilBERT'),
        ('LLM', 'google/electra-base-discriminator', 'ELECTRA-Base'),
        ('LLM', 'albert-base-v2', 'ALBERT-Base'),
        ('ViT', 'google/vit-base-patch16-224-in21k', 'ViT-Base'),
        ('ViT', 'facebook/deit-base-distilled-patch16-224', 'DeiT-Base'),
        ('VLM', ('roberta-base', 'google/vit-base-patch16-224-in21k'), 'RoBERTa+ViT'),
    ]
    
    # Train each model
    for model_type, model_name, display_name in models_to_train:
        model_key = f"{model_type}_{display_name}"
        
        if model_key in completed_models:
            print(f"\n[SKIP] {display_name} already completed")
            continue
        
        print(f"\n{'='*80}")
        print(f"[TRAINING] {display_name}")
        print(f"{'='*80}")
        
        try:
            # Prepare tokenizers/processors
            if model_type == 'VLM':
                text_model, vision_model = model_name
                tokenizer = AutoTokenizer.from_pretrained(text_model)
                image_processor = AutoImageProcessor.from_pretrained(vision_model)
            elif model_type == 'ViT':
                tokenizer = None
                image_processor = AutoImageProcessor.from_pretrained(model_name)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                image_processor = None
            
            # Get parameters
            if model_type == 'VLM':
                temp_model = VLMClassifier(text_model, vision_model)
            elif model_type == 'ViT':
                temp_model = ViTClassifier(model_name)
            else:
                temp_model = LLMClassifier(model_name)
            
            params = sum(p.numel() for p in temp_model.parameters())
            del temp_model
            gc.collect()
            
            # Centralized training
            print("\n[CENTRALIZED]")
            start_time = time.time()
            
            if model_type == 'VLM':
                model = VLMClassifier(text_model, vision_model)
            elif model_type == 'ViT':
                model = ViTClassifier(model_name)
            else:
                model = LLMClassifier(model_name)
            
            model, history = train_centralized(
                model, train_df, val_df, tokenizer, image_processor,
                epochs=5, batch_size=8
            )
            
            training_time = (time.time() - start_time) / 3600
            final_metrics = history[-1]
            
            result_central = {
                'model_name': f"{display_name}-Centralized",
                'model_type': model_type,
                'training_type': 'Centralized',
                'params_millions': params / 1e6,
                'training_time_hours': training_time,
                **final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result_central)
            
            print(f"[DONE] Centralized: F1={final_metrics['f1_macro']:.4f}, Time={training_time:.2f}h")
            
            del model
            gc.collect()
            
            # Federated training
            print("\n[FEDERATED]")
            start_time = time.time()
            
            if model_type == 'VLM':
                model_fn = lambda: VLMClassifier(text_model, vision_model)
            elif model_type == 'ViT':
                model_fn = lambda: ViTClassifier(model_name)
            else:
                model_fn = lambda: LLMClassifier(model_name)
            
            model, history = train_federated(
                model_fn, train_dfs_fed, val_df, tokenizer, image_processor,
                n_rounds=5, local_epochs=2, batch_size=8
            )
            
            training_time = (time.time() - start_time) / 3600
            final_metrics = history[-1]
            
            result_fed = {
                'model_name': f"{display_name}-Federated",
                'model_type': model_type,
                'training_type': 'Federated',
                'params_millions': params / 1e6,
                'training_time_hours': training_time,
                **final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            results.append(result_fed)
            
            print(f"[DONE] Federated: F1={final_metrics['f1_macro']:.4f}, Time={training_time:.2f}h")
            
            del model
            gc.collect()
            
            # Mark as completed and save checkpoint
            completed_models.add(model_key)
            checkpoint['completed'] = list(completed_models)
            checkpoint['results'] = results
            save_checkpoint(checkpoint)
            
            # Save incremental results
            save_incremental_results(results)
            
        except Exception as e:
            print(f"\n[ERROR] Failed to train {display_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final save
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    save_incremental_results(results)
    
    # Print summary
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    if len(df_results) > 0:
        summary = df_results[['model_name', 'model_type', 'training_type', 'f1_macro', 'accuracy', 'training_time_hours']]
        print(summary.to_string(index=False))
        
        print("\n[STATISTICS]")
        print(f"Total models trained: {len(df_results)}")
        print(f"Total training time: {df_results['training_time_hours'].sum():.2f} hours")
        print(f"Average F1-Macro: {df_results['f1_macro'].mean():.4f}")
        print(f"Best model: {df_results.loc[df_results['f1_macro'].idxmax(), 'model_name']} (F1={df_results['f1_macro'].max():.4f})")

if __name__ == "__main__":
    main()
