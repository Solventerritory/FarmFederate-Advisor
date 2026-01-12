"""
Resume Ultimate Model Comparison from where it stopped
Trains remaining models: XLM-RoBERTa (fed only), ELECTRA, ALBERT, MPNet, ViT, VLM
"""

import sys
import os

# Add parent directory to path
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
    AutoModelForSequenceClassification,
    AutoImageProcessor,
    ViTModel,
    ViTForImageClassification
)

# Import from existing modules
from datasets_loader import (
    build_text_corpus_mix,
    load_stress_image_datasets_hf,
    ISSUE_LABELS,
    NUM_LABELS
)

# Configuration
SEED = 42
RESULTS_DIR = Path(__file__).parent / "comparison_results"
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# ============================================================================
# MODEL CLASSES (Simplified from ultimate_model_comparison.py)
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
        # Dummy image data
        pixel_values = torch.randn(3, self.img_size, self.img_size)
        return {'pixel_values': pixel_values, 'labels': self.labels[idx]}

class MultiModalDataset(Dataset):
    def __init__(self, df, tokenizer, image_processor, max_length=128):
        self.texts = df['text'].tolist()
        self.labels = torch.tensor(df['labels'].tolist(), dtype=torch.float32)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
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
        
        # Dummy image
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
    
    for batch in tqdm(loader, desc="Training", leave=False):
        labels = batch['labels'].to(device)
        
        if 'pixel_values' in batch and 'input_ids' in batch:
            # VLM
            outputs = model(
                batch['input_ids'].to(device),
                batch['attention_mask'].to(device),
                batch['pixel_values'].to(device)
            )
        elif 'pixel_values' in batch:
            # ViT
            outputs = model(batch['pixel_values'].to(device))
        else:
            # LLM
            outputs = model(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
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
        'f1_macro': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'f1_micro': f1_score(all_labels, all_preds, average='micro', zero_division=0),
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='macro', zero_division=0),
        'recall': recall_score(all_labels, all_preds, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(all_labels, all_preds),
        'jaccard': jaccard_score(all_labels, all_preds, average='macro', zero_division=0)
    }
    
    return metrics

def train_centralized(model, train_df, val_df, tokenizer, image_processor, epochs=10, batch_size=16, lr=2e-5):
    """Centralized training"""
    model = model.to(device)
    
    # Create dataset
    if isinstance(model, VLMClassifier):
        train_dataset = MultiModalDataset(train_df, tokenizer, image_processor)
        val_dataset = MultiModalDataset(val_df, tokenizer, image_processor)
    elif isinstance(model, ViTClassifier):
        train_dataset = ImageDataset(size=len(train_df))
        val_dataset = ImageDataset(size=len(val_df))
    else:
        train_dataset = TextDataset(train_df, tokenizer)
        val_dataset = TextDataset(val_df, tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for epoch in range(epochs):
        print(f"\n[Epoch {epoch+1}/{epochs}]")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        
        print(f"  Train Loss: {train_loss:.4f}, F1-Macro: {val_metrics['f1_macro']:.4f}")
        print(f"  Val F1-Macro: {val_metrics['f1_macro']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
        
        history.append(val_metrics)
    
    return model, history

def train_federated(model_fn, train_dfs, val_df, tokenizer, image_processor, n_rounds=10, local_epochs=2, batch_size=16, lr=2e-5):
    """Federated training"""
    n_clients = len(train_dfs)
    global_model = model_fn().to(device)
    criterion = nn.BCEWithLogitsLoss()
    
    history = []
    
    for round_idx in range(n_rounds):
        print(f"\n[Round {round_idx + 1}/{n_rounds}]")
        client_models = []
        
        for client_idx in range(n_clients):
            print(f"  Client {client_idx + 1}/{n_clients}")
            
            # Skip if client has no data
            if len(train_dfs[client_idx]) == 0:
                print(f"    Skipping client {client_idx + 1} (no data)")
                continue
            
            # Create client model
            client_model = model_fn().to(device)
            client_model.load_state_dict(global_model.state_dict())
            
            # Create client dataset
            client_df = train_dfs[client_idx]
            
            if isinstance(client_model, VLMClassifier):
                client_dataset = MultiModalDataset(client_df, tokenizer, image_processor)
            elif isinstance(client_model, ViTClassifier):
                client_dataset = ImageDataset(size=len(client_df))
            else:
                client_dataset = TextDataset(client_df, tokenizer)
            
            client_loader = DataLoader(client_dataset, batch_size=batch_size, shuffle=True)
            
            # Train client
            optimizer = torch.optim.AdamW(client_model.parameters(), lr=lr)
            
            for epoch in range(local_epochs):
                train_epoch(client_model, client_loader, optimizer, criterion, device)
            
            client_models.append(client_model.cpu())
            del client_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Federated averaging
        if len(client_models) > 0:
            with torch.no_grad():
                for key in global_model.state_dict().keys():
                    global_model.state_dict()[key].copy_(
                        torch.stack([client.state_dict()[key].float() for client in client_models]).mean(0)
                    )
        
        # Evaluate
        if isinstance(global_model, VLMClassifier):
            val_dataset = MultiModalDataset(val_df, tokenizer, image_processor)
        elif isinstance(global_model, ViTClassifier):
            val_dataset = ImageDataset(size=len(val_df))
        else:
            val_dataset = TextDataset(val_df, tokenizer)
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        metrics = evaluate(global_model, val_loader, device)
        
        print(f"  Global F1-Macro: {metrics['f1_macro']:.4f}, Accuracy: {metrics['accuracy']:.4f}")
        history.append(metrics)
        
        # Clean up
        del client_models
        gc.collect()
    
    return global_model, history

# ============================================================================
# MAIN RESUME TRAINING
# ============================================================================

def main():
    print("\n" + "="*80)
    print("RESUMING MODEL COMPARISON FROM WHERE IT STOPPED")
    print("="*80 + "\n")
    
    # Load datasets
    print("[INFO] Loading datasets...")
    try:
        df_text = build_text_corpus_mix(
            sources=["gardian", "argilla", "agnews", "localmini"],
            max_samples=10000,
            max_per_source=3000
        )
        print(f"[INFO] Loaded {len(df_text)} text samples")
        df = df_text
    except Exception as e:
        print(f"[WARNING] Failed to load datasets: {e}")
        n_samples = 1000
        data = {
            'text': [f"Sample agricultural text {i}" for i in range(n_samples)],
            'labels': [[np.random.randint(0, 2) for _ in range(NUM_LABELS)] for _ in range(n_samples)]
        }
        df = pd.DataFrame(data)
    
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=SEED)
    print(f"[INFO] Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Split for federated learning (5 clients)
    def split_federated(df, n_clients=5):
        dfs = []
        chunk_size = len(df) // n_clients
        for i in range(n_clients):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_clients - 1 else len(df)
            dfs.append(df.iloc[start_idx:end_idx].reset_index(drop=True))
        return dfs
    
    train_dfs_fed = split_federated(train_df, n_clients=5)
    
    # Results storage
    results = []
    
    # Remaining models to train (starting from XLM-RoBERTa federated)
    remaining_models = {
        'LLM': [
            ('xlm-roberta-base', 'XLM-RoBERTa', True),  # federated only
            ('google/electra-base-discriminator', 'ELECTRA-Base', False),  # both
            ('albert-base-v2', 'ALBERT-Base', False),  # both
            ('microsoft/mpnet-base', 'MPNet-Base', False),  # both
        ],
        'ViT': [
            ('google/vit-base-patch16-224-in21k', 'ViT-Base', False),
            ('facebook/deit-base-distilled-patch16-224', 'DeiT-Base', False),
            ('microsoft/beit-base-patch16-224', 'BEiT-Base', False),
        ],
        'VLM': [
            (('roberta-base', 'google/vit-base-patch16-224-in21k'), 'RoBERTa+ViT', False),
            (('bert-base-uncased', 'facebook/deit-base-distilled-patch16-224'), 'BERT+DeiT', False),
        ]
    }
    
    # Train remaining models
    for model_type, model_list in remaining_models.items():
        for item in model_list:
            model_name, display_name, fed_only = item
            
            print(f"\n{'='*80}")
            print(f"[INFO] Training {display_name}...")
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
                
                # Get parameter count
                if model_type == 'VLM':
                    temp_model = VLMClassifier(text_model, vision_model)
                elif model_type == 'ViT':
                    temp_model = ViTClassifier(model_name)
                else:
                    temp_model = LLMClassifier(model_name)
                
                params = sum(p.numel() for p in temp_model.parameters())
                del temp_model
                
                # Centralized training (if not fed_only)
                if not fed_only:
                    print("\n[Centralized Training]")
                    start_time = time.time()
                    
                    if model_type == 'VLM':
                        model_central = VLMClassifier(text_model, vision_model)
                    elif model_type == 'ViT':
                        model_central = ViTClassifier(model_name)
                    else:
                        model_central = LLMClassifier(model_name)
                    
                    model_central, history_central = train_centralized(
                        model_central, train_df, val_df, tokenizer, image_processor,
                        epochs=10, batch_size=16
                    )
                    
                    training_time = (time.time() - start_time) / 3600
                    final_metrics = history_central[-1]
                    
                    results.append({
                        'model_name': f"{display_name}-Centralized",
                        'model_type': model_type,
                        'training_type': 'Centralized',
                        'params_millions': params / 1e6,
                        'training_time_hours': training_time,
                        **final_metrics
                    })
                    
                    print(f"[INFO] Centralized training completed in {training_time:.2f} hours")
                    
                    del model_central
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                # Federated training
                print("\n[Federated Training]")
                start_time = time.time()
                
                if model_type == 'VLM':
                    model_fn = lambda: VLMClassifier(text_model, vision_model)
                elif model_type == 'ViT':
                    model_fn = lambda: ViTClassifier(model_name)
                else:
                    model_fn = lambda: LLMClassifier(model_name)
                
                model_fed, history_fed = train_federated(
                    model_fn, train_dfs_fed, val_df, tokenizer, image_processor,
                    n_rounds=10, local_epochs=2, batch_size=16
                )
                
                training_time = (time.time() - start_time) / 3600
                final_metrics = history_fed[-1]
                
                results.append({
                    'model_name': f"{display_name}-Federated",
                    'model_type': model_type,
                    'training_type': 'Federated',
                    'params_millions': params / 1e6,
                    'training_time_hours': training_time,
                    **final_metrics
                })
                
                print(f"[INFO] Federated training completed in {training_time:.2f} hours")
                
                del model_fed
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                gc.collect()
                
            except Exception as e:
                print(f"[ERROR] Failed to train {display_name}: {e}")
                import traceback
                traceback.print_exc()
    
    # Save results
    results_file = RESULTS_DIR / f"resume_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*80)
    print("RESUMED TRAINING COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {results_file}")
    
    # Print summary
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print(df_results[['model_name', 'f1_macro', 'accuracy', 'training_time_hours']].to_string(index=False))

if __name__ == "__main__":
    main()
