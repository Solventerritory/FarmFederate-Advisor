#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE MODEL COMPARISON - Federated Learning
====================================================
Train and compare multiple architectures for:
- LLM: T5, DistilBERT, RoBERTa, BERT, GPT-2
- ViT: ViT-Base, DeiT, Swin Transformer
- VLM: CLIP, BLIP, ALBEF

Determines best model for each modality.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModel,
    T5ForConditionalGeneration, T5Tokenizer,
    DistilBertModel, DistilBertTokenizer,
    RobertaModel, RobertaTokenizer,
    BertModel, BertTokenizer,
    GPT2LMHeadModel, GPT2Tokenizer,
    ViTModel, ViTImageProcessor,
    DeiTModel, DeiTImageProcessor,
    CLIPModel, CLIPProcessor,
)
import numpy as np
import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Import from main system
from federated_complete_system import (
    load_datasets, split_data_federated, fedavg_aggregate,
    TextDataset, ImageDataset, MultimodalDataset,
    NUM_LABELS, DEVICE, SEED, OUTPUT_DIR, PLOTS_DIR, RESULTS_DIR
)

LABEL_NAMES = ["water_stress", "nutrient_deficiency", "pest_risk", "disease_risk", "heat_stress"]

# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

class LLMModel(nn.Module):
    """Flexible LLM architecture"""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        if "t5" in model_name.lower():
            self.base = T5ForConditionalGeneration.from_pretrained(model_name).encoder
            self.hidden_size = self.base.config.d_model
        elif "distilbert" in model_name.lower():
            self.base = DistilBertModel.from_pretrained(model_name)
            self.hidden_size = self.base.config.hidden_size
        elif "roberta" in model_name.lower():
            self.base = RobertaModel.from_pretrained(model_name)
            self.hidden_size = self.base.config.hidden_size
        elif "bert" in model_name.lower():
            self.base = BertModel.from_pretrained(model_name)
            self.hidden_size = self.base.config.hidden_size
        elif "gpt" in model_name.lower():
            self.base = GPT2LMHeadModel.from_pretrained(model_name).transformer
            self.hidden_size = self.base.config.n_embd
        else:
            self.base = AutoModel.from_pretrained(model_name)
            self.hidden_size = self.base.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, NUM_LABELS)
        )
    
    def forward(self, input_ids, attention_mask):
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            hidden = outputs.pooler_output
        else:
            hidden = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(hidden)


class ViTModelWrapper(nn.Module):
    """Flexible ViT architecture"""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        if "deit" in model_name.lower():
            self.base = DeiTModel.from_pretrained(model_name)
        elif "swin" in model_name.lower():
            from transformers import SwinModel
            self.base = SwinModel.from_pretrained(model_name)
        else:
            self.base = ViTModel.from_pretrained(model_name)
        
        self.hidden_size = self.base.config.hidden_size
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.hidden_size),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, NUM_LABELS)
        )
    
    def forward(self, pixel_values):
        outputs = self.base(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state[:, 0, :]
        return self.classifier(hidden)


class VLMModelWrapper(nn.Module):
    """Flexible VLM architecture"""
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        if "clip" in model_name.lower():
            self.base = CLIPModel.from_pretrained(model_name)
            self.hidden_size = self.base.config.projection_dim
        elif "blip" in model_name.lower():
            from transformers import BlipModel
            self.base = BlipModel.from_pretrained(model_name)
            self.hidden_size = 512
        else:
            raise ValueError(f"Unknown VLM: {model_name}")
        
        fusion_dim = self.hidden_size * 2
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Dropout(0.1),
            nn.Linear(fusion_dim, 512),
            nn.GELU()
        )
        self.classifier = nn.Linear(512, NUM_LABELS)
    
    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values
        )
        
        vision_embeds = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.vision_model_output.pooler_output
        text_embeds = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs.text_model_output.pooler_output
        
        fused = torch.cat([vision_embeds, text_embeds], dim=1)
        fused = self.fusion(fused)
        return self.classifier(fused)


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_epoch_simple(model, dataloader, optimizer, criterion, device):
    """Train one epoch"""
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        if isinstance(model, VLMModelWrapper):
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["pixel_values"].to(device)
            )
        elif isinstance(model, ViTModelWrapper):
            logits = model(batch["pixel_values"].to(device))
        else:  # LLM
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device)
            )
        
        loss = criterion(logits, batch["labels"].to(device))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate_model_simple(model, dataloader, device):
    """Evaluate model"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(model, VLMModelWrapper):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                    batch["pixel_values"].to(device)
                )
            elif isinstance(model, ViTModelWrapper):
                logits = model(batch["pixel_values"].to(device))
            else:
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device)
                )
            
            preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()
            labels = batch["labels"].cpu().numpy()
            
            all_preds.append(preds)
            all_labels.append(labels)
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'accuracy': accuracy
    }


def train_federated_model(model, clients_data, test_loader, num_rounds=5, local_epochs=2, lr=2e-5):
    """Train with federated averaging"""
    criterion = nn.BCEWithLogitsLoss()
    history = []
    
    print(f"   Training {model.model_name}...")
    print(f"   Rounds: {num_rounds}, Clients: {len(clients_data)}, Local epochs: {local_epochs}")
    
    for round_idx in range(num_rounds):
        client_models = []
        client_weights = []
        
        # Train each client
        for client_id, client_df in enumerate(clients_data):
            # Prepare data
            if isinstance(model, VLMModelWrapper):
                tokenizer = CLIPProcessor.from_pretrained(model.model_name.replace("clip", "openai/clip") if "clip" in model.model_name else model.model_name)
                dataset = MultimodalDataset(client_df, tokenizer.tokenizer, tokenizer.image_processor)
            elif isinstance(model, ViTModelWrapper):
                processor = ViTImageProcessor.from_pretrained(model.model_name)
                dataset = ImageDataset(client_df, processor)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model.model_name)
                dataset = TextDataset(client_df, tokenizer)
            
            loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
            
            # Local training
            local_model = type(model)(model.model_name).to(DEVICE)
            local_model.load_state_dict(model.state_dict())
            optimizer = torch.optim.AdamW(local_model.parameters(), lr=lr)
            
            for _ in range(local_epochs):
                train_epoch_simple(local_model, loader, optimizer, criterion, DEVICE)
            
            client_models.append(local_model.state_dict())
            client_weights.append(len(client_df))
        
        # Aggregate
        global_state = fedavg_aggregate(client_models, client_weights)
        model.load_state_dict(global_state)
        
        # Evaluate
        metrics = evaluate_model_simple(model, test_loader, DEVICE)
        history.append(metrics)
        
        print(f"      Round {round_idx+1}/{num_rounds}: F1-Macro={metrics['f1_macro']:.4f}, Acc={metrics['accuracy']:.4f}")
    
    return history


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    print("\n" + "="*70)
    print("🏆 COMPREHENSIVE MODEL COMPARISON - FEDERATED LEARNING")
    print("="*70)
    
    # Load datasets
    text_df, image_df, multimodal_df = load_datasets()
    
    # Split
    text_train, text_test = train_test_split(text_df, test_size=0.2, random_state=SEED)
    image_train, image_test = train_test_split(image_df, test_size=0.2, random_state=SEED)
    multi_train, multi_test = train_test_split(multimodal_df, test_size=0.2, random_state=SEED)
    
    # Federated split
    num_clients = 5
    text_clients = split_data_federated(text_train, num_clients)
    image_clients = split_data_federated(image_train, num_clients)
    multi_clients = split_data_federated(multi_train, num_clients)
    
    # Model configurations
    llm_models = [
        "t5-small",
        "distilbert-base-uncased",
        "roberta-base",
        "bert-base-uncased",
        "gpt2"
    ]
    
    vit_models = [
        "google/vit-base-patch16-224",
        "facebook/deit-base-patch16-224",
        "microsoft/swin-tiny-patch4-window7-224"
    ]
    
    vlm_models = [
        "openai/clip-vit-base-patch32",
        "Salesforce/blip-image-captioning-base"
    ]
    
    all_results = {}
    
    # ========================================================================
    # TRAIN LLMs
    # ========================================================================
    print("\n" + "="*70)
    print("1️⃣ TRAINING FEDERATED LLMs")
    print("="*70)
    
    llm_results = {}
    for model_name in llm_models:
        print(f"\n📝 Training {model_name}...")
        try:
            model = LLMModel(model_name).to(DEVICE)
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            test_dataset = TextDataset(text_test, tokenizer)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
            
            history = train_federated_model(model, text_clients, test_loader, num_rounds=5)
            llm_results[model_name] = {
                'history': history,
                'final_metrics': history[-1]
            }
        except Exception as e:
            print(f"   ⚠️ Failed to train {model_name}: {e}")
            llm_results[model_name] = {'error': str(e)}
    
    all_results['llm'] = llm_results
    
    # ========================================================================
    # TRAIN ViTs
    # ========================================================================
    print("\n" + "="*70)
    print("2️⃣ TRAINING FEDERATED ViTs")
    print("="*70)
    
    vit_results = {}
    for model_name in vit_models:
        print(f"\n🖼️  Training {model_name}...")
        try:
            model = ViTModelWrapper(model_name).to(DEVICE)
            processor = ViTImageProcessor.from_pretrained(model_name)
            test_dataset = ImageDataset(image_test, processor)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
            
            history = train_federated_model(model, image_clients, test_loader, num_rounds=5)
            vit_results[model_name] = {
                'history': history,
                'final_metrics': history[-1]
            }
        except Exception as e:
            print(f"   ⚠️ Failed to train {model_name}: {e}")
            vit_results[model_name] = {'error': str(e)}
    
    all_results['vit'] = vit_results
    
    # ========================================================================
    # TRAIN VLMs
    # ========================================================================
    print("\n" + "="*70)
    print("3️⃣ TRAINING FEDERATED VLMs")
    print("="*70)
    
    vlm_results = {}
    for model_name in vlm_models:
        print(f"\n🔀 Training {model_name}...")
        try:
            model = VLMModelWrapper(model_name).to(DEVICE)
            processor = CLIPProcessor.from_pretrained(model_name)
            test_dataset = MultimodalDataset(multi_test, processor.tokenizer, processor.image_processor)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)
            
            history = train_federated_model(model, multi_clients, test_loader, num_rounds=5)
            vlm_results[model_name] = {
                'history': history,
                'final_metrics': history[-1]
            }
        except Exception as e:
            print(f"   ⚠️ Failed to train {model_name}: {e}")
            vlm_results[model_name] = {'error': str(e)}
    
    all_results['vlm'] = vlm_results
    
    # ========================================================================
    # DETERMINE BEST MODELS
    # ========================================================================
    print("\n" + "="*70)
    print("🏆 DETERMINING BEST MODELS")
    print("="*70)
    
    def get_best_model(results_dict):
        best_name = None
        best_f1 = -1
        for name, data in results_dict.items():
            if 'error' not in data:
                f1 = data['final_metrics']['f1_macro']
                if f1 > best_f1:
                    best_f1 = f1
                    best_name = name
        return best_name, best_f1
    
    best_llm, best_llm_f1 = get_best_model(llm_results)
    best_vit, best_vit_f1 = get_best_model(vit_results)
    best_vlm, best_vlm_f1 = get_best_model(vlm_results)
    
    print(f"\n🥇 BEST LLM: {best_llm} (F1-Macro: {best_llm_f1:.4f})")
    print(f"🥇 BEST ViT: {best_vit} (F1-Macro: {best_vit_f1:.4f})")
    print(f"🥇 BEST VLM: {best_vlm} (F1-Macro: {best_vlm_f1:.4f})")
    
    # Save results
    results_file = RESULTS_DIR / "model_comparison_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'all_results': all_results,
            'best_models': {
                'llm': {'name': best_llm, 'f1_macro': best_llm_f1},
                'vit': {'name': best_vit, 'f1_macro': best_vit_f1},
                'vlm': {'name': best_vlm, 'f1_macro': best_vlm_f1}
            }
        }, f, indent=2, default=str)
    
    print(f"\n✅ Results saved to {results_file}")
    
    # ========================================================================
    # GENERATE COMPARISON PLOTS
    # ========================================================================
    print("\n" + "="*70)
    print("📊 GENERATING COMPARISON PLOTS")
    print("="*70)
    
    generate_comparison_plots(all_results)
    
    return all_results


def generate_comparison_plots(results):
    """Generate comprehensive comparison plots"""
    
    # Plot 1: LLM Comparison
    plt.figure(figsize=(12, 6))
    for model_name, data in results['llm'].items():
        if 'error' not in data:
            f1_scores = [h['f1_macro'] for h in data['history']]
            plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='o', label=model_name)
    plt.xlabel('Federated Round')
    plt.ylabel('F1-Macro Score')
    plt.title('Federated LLM Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "llm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: ViT Comparison
    plt.figure(figsize=(12, 6))
    for model_name, data in results['vit'].items():
        if 'error' not in data:
            f1_scores = [h['f1_macro'] for h in data['history']]
            plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='s', label=model_name)
    plt.xlabel('Federated Round')
    plt.ylabel('F1-Macro Score')
    plt.title('Federated ViT Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "vit_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: VLM Comparison
    plt.figure(figsize=(12, 6))
    for model_name, data in results['vlm'].items():
        if 'error' not in data:
            f1_scores = [h['f1_macro'] for h in data['history']]
            plt.plot(range(1, len(f1_scores)+1), f1_scores, marker='^', label=model_name)
    plt.xlabel('Federated Round')
    plt.ylabel('F1-Macro Score')
    plt.title('Federated VLM Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(PLOTS_DIR / "vlm_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 4: Final Performance Comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for idx, (category, ax) in enumerate(zip(['llm', 'vit', 'vlm'], axes)):
        model_names = []
        f1_scores = []
        for name, data in results[category].items():
            if 'error' not in data:
                model_names.append(name.split('/')[-1])
                f1_scores.append(data['final_metrics']['f1_macro'])
        
        ax.barh(model_names, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(model_names)])
        ax.set_xlabel('F1-Macro Score')
        ax.set_title(f'{category.upper()} Models')
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "final_comparison_all_models.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Plots saved!")


if __name__ == "__main__":
    main()
