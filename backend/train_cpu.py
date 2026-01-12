"""
FarmFederate CPU Training Script
Trains Federated LLM, ViT, and VLM models on CPU in VS Code
Optimized for CPU with smaller models and checkpointing
"""

import os
import sys
import time
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Force CPU mode
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cpu')

print(f"""
╔════════════════════════════════════════════════════════════╗
║         FarmFederate CPU Training (VS Code)                ║
║         Optimized for local CPU training                   ║
╚════════════════════════════════════════════════════════════╝

Device: {device}
PyTorch: {torch.__version__}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

""")

# Configuration optimized for CPU
CONFIG = {
    'device': 'cpu',
    'num_clients': 4,  # Reduced from 8 for CPU
    'num_rounds': 5,   # Reduced from 10 for faster training
    'local_epochs': 2, # Reduced from 3
    'batch_size': 4,   # Small batch for CPU
    'learning_rate': 2e-5,
    'max_length': 128, # Reduced from 512
    'image_size': 224,
    'lora_r': 8,       # Reduced from 16
    'lora_alpha': 16,  # Reduced from 32
    'checkpoint_dir': 'checkpoints_cpu',
    'save_every': 2,   # Save every 2 rounds
}

# Create checkpoint directory
os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)

# Training time estimates (CPU)
TIME_ESTIMATES = {
    'llm_small': '2-3 hours',
    'llm_base': '4-6 hours',
    'vit_small': '3-4 hours',
    'vit_base': '5-7 hours',
    'vlm_clip': '4-5 hours',
    'vlm_blip': '6-8 hours',
}

print("⏱️  CPU Training Time Estimates:")
for model, time_est in TIME_ESTIMATES.items():
    print(f"   {model:15s}: {time_est}")
print()

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_synthetic_text_data(num_samples=1000):
    """Generate synthetic text data for plant stress detection"""
    print("📊 Loading synthetic text dataset...")
    
    stress_types = ['drought', 'nutrient deficiency', 'pest attack', 'disease', 'heat stress']
    symptoms = [
        'yellowing leaves', 'wilting', 'brown spots', 'stunted growth',
        'leaf curl', 'discoloration', 'necrosis', 'chlorosis'
    ]
    
    texts = []
    labels = []
    for i in range(num_samples):
        stress = np.random.choice(stress_types)
        symptom = np.random.choice(symptoms)
        text = f"Plant shows {symptom} indicating possible {stress}"
        texts.append(text)
        labels.append(stress_types.index(stress))
    
    print(f"   ✓ Generated {num_samples} text samples")
    return texts, labels

def load_synthetic_image_data(num_samples=1000):
    """Generate synthetic image data for plant disease detection"""
    print("📊 Loading synthetic image dataset...")
    
    # Generate random images (3, 224, 224)
    images = torch.randn(num_samples, 3, 224, 224)
    labels = torch.randint(0, 10, (num_samples,))  # 10 disease classes
    
    print(f"   ✓ Generated {num_samples} image samples")
    return images, labels

def load_synthetic_multimodal_data(num_samples=1000):
    """Generate synthetic multimodal (text + image) data"""
    print("📊 Loading synthetic multimodal dataset...")
    
    texts, text_labels = load_synthetic_text_data(num_samples)
    images, img_labels = load_synthetic_image_data(num_samples)
    
    print(f"   ✓ Generated {num_samples} multimodal pairs")
    return texts, images, text_labels

# ============================================================================
# FEDERATED LEARNING UTILITIES
# ============================================================================

def split_data_non_iid(data, labels, num_clients, alpha=0.3):
    """Split data into non-IID partitions for federated learning"""
    print(f"🔀 Splitting data for {num_clients} clients (Non-IID α={alpha})...")
    
    n_samples = len(data)
    client_data = [[] for _ in range(num_clients)]
    client_labels = [[] for _ in range(num_clients)]
    
    # Simple partition: assign samples to clients
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    
    for i, split in enumerate(splits):
        if isinstance(data, list):
            client_data[i] = [data[idx] for idx in split]
        else:
            client_data[i] = data[split]
        
        if isinstance(labels, list):
            client_labels[i] = [labels[idx] for idx in split]
        else:
            client_labels[i] = labels[split]
    
    print(f"   ✓ Split into {num_clients} partitions")
    return client_data, client_labels

def federated_averaging(global_model, client_models, client_weights=None):
    """FedAvg: Average model parameters from clients"""
    if client_weights is None:
        client_weights = [1.0 / len(client_models)] * len(client_models)
    
    global_dict = global_model.state_dict()
    
    for key in global_dict.keys():
        global_dict[key] = torch.zeros_like(global_dict[key], dtype=torch.float32)
        
        for client_model, weight in zip(client_models, client_weights):
            client_dict = client_model.state_dict()
            global_dict[key] += weight * client_dict[key].float()
    
    global_model.load_state_dict(global_dict)
    return global_model

# ============================================================================
# MODEL TRAINING FUNCTIONS
# ============================================================================

def train_federated_llm(model_name='distilbert-base-uncased', save_name='llm_distilbert'):
    """Train Federated LLM for text-based plant stress detection"""
    print(f"\n{'='*60}")
    print(f"🤖 Training Federated LLM: {model_name}")
    print(f"{'='*60}\n")
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from peft import LoraConfig, get_peft_model
    
    # Load model and tokenizer
    print("📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=5,  # 5 stress types
        ignore_mismatched_sizes=True
    )
    
    # Apply LoRA for parameter efficiency
    print("🔧 Applying LoRA...")
    # For DistilBERT, target attention layers
    if 'distilbert' in model_name.lower():
        target_modules = ["q_lin", "v_lin"]
    elif 'bert' in model_name.lower():
        target_modules = ["query", "value"]
    else:
        target_modules = ["q_proj", "v_proj"]
    
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="SEQ_CLS"
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    print(f"   ✓ Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    
    # Load data
    texts, labels = load_synthetic_text_data(num_samples=800)
    client_texts, client_labels = split_data_non_iid(texts, labels, CONFIG['num_clients'])
    
    # Federated training
    print(f"\n🔄 Starting Federated Training ({CONFIG['num_rounds']} rounds)...\n")
    training_start = time.time()
    
    for round_num in range(CONFIG['num_rounds']):
        round_start = time.time()
        print(f"Round {round_num + 1}/{CONFIG['num_rounds']}")
        
        client_models = []
        round_losses = []
        
        # Train each client
        for client_id in range(CONFIG['num_clients']):
            print(f"  Client {client_id + 1}/{CONFIG['num_clients']}...", end=' ')
            
            # Tokenize client data
            client_encodings = tokenizer(
                client_texts[client_id],
                truncation=True,
                padding=True,
                max_length=CONFIG['max_length'],
                return_tensors='pt'
            )
            
            # Create simple dataset
            client_dataset = torch.utils.data.TensorDataset(
                client_encodings['input_ids'],
                client_encodings['attention_mask'],
                torch.tensor(client_labels[client_id])
            )
            
            client_loader = torch.utils.data.DataLoader(
                client_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True
            )
            
            # Local training
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
            model.train()
            
            client_loss = 0
            for epoch in range(CONFIG['local_epochs']):
                for batch in client_loader:
                    input_ids, attention_mask, batch_labels = [b.to(device) for b in batch]
                    
                    outputs = model(input_ids, attention_mask=attention_mask, labels=batch_labels)
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    client_loss += loss.item()
            
            avg_loss = client_loss / (CONFIG['local_epochs'] * len(client_loader))
            round_losses.append(avg_loss)
            print(f"Loss: {avg_loss:.4f}")
            
            # Save client model state
            client_models.append(model.state_dict().copy())
        
        # Federated averaging
        print("  Aggregating client models...", end=' ')
        model = federated_averaging(model, [model] * len(client_models))
        print("✓")
        
        round_time = time.time() - round_start
        avg_round_loss = np.mean(round_losses)
        print(f"  Round Loss: {avg_round_loss:.4f} | Time: {round_time/60:.1f}min\n")
        
        # Save checkpoint
        if (round_num + 1) % CONFIG['save_every'] == 0 or round_num == CONFIG['num_rounds'] - 1:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{save_name}_round{round_num + 1}.pt"
            )
            torch.save({
                'round': round_num + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_round_loss,
                'config': CONFIG,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(CONFIG['checkpoint_dir'], f"{save_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    
    training_time = time.time() - training_start
    print(f"\n✅ LLM Training Complete!")
    print(f"   Total Time: {training_time/3600:.2f} hours")
    print(f"   Final Loss: {avg_round_loss:.4f}")
    print(f"   Saved: {final_path}\n")
    
    return model, avg_round_loss

def train_federated_vit(model_name='google/vit-base-patch16-224', save_name='vit_base'):
    """Train Federated ViT for image-based plant disease detection"""
    print(f"\n{'='*60}")
    print(f"🖼️  Training Federated ViT: {model_name}")
    print(f"{'='*60}\n")
    
    from transformers import ViTForImageClassification, ViTImageProcessor
    from peft import LoraConfig, get_peft_model
    
    # Load model
    print("📥 Loading ViT model...")
    processor = ViTImageProcessor.from_pretrained(model_name)
    base_model = ViTForImageClassification.from_pretrained(
        model_name,
        num_labels=10,  # 10 disease classes
        ignore_mismatched_sizes=True
    )
    
    # Apply LoRA
    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=["query", "value"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"  # Use FEATURE_EXTRACTION for ViT
    )
    model = get_peft_model(base_model, lora_config)
    model.to(device)
    
    print(f"   ✓ Trainable parameters: {model.num_parameters(only_trainable=True):,}")
    
    # Load data
    images, labels = load_synthetic_image_data(num_samples=800)
    client_images, client_labels = split_data_non_iid(images, labels, CONFIG['num_clients'])
    
    # Federated training
    print(f"\n🔄 Starting Federated Training ({CONFIG['num_rounds']} rounds)...\n")
    training_start = time.time()
    
    for round_num in range(CONFIG['num_rounds']):
        round_start = time.time()
        print(f"Round {round_num + 1}/{CONFIG['num_rounds']}")
        
        client_models = []
        round_losses = []
        
        # Train each client
        for client_id in range(CONFIG['num_clients']):
            print(f"  Client {client_id + 1}/{CONFIG['num_clients']}...", end=' ')
            
            client_dataset = torch.utils.data.TensorDataset(
                client_images[client_id],
                client_labels[client_id]
            )
            
            client_loader = torch.utils.data.DataLoader(
                client_dataset,
                batch_size=CONFIG['batch_size'],
                shuffle=True
            )
            
            # Local training
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
            model.train()
            
            client_loss = 0
            for epoch in range(CONFIG['local_epochs']):
                for batch_images, batch_labels in client_loader:
                    batch_images = batch_images.to(device)
                    batch_labels = batch_labels.to(device)
                    
                    outputs = model(pixel_values=batch_images, labels=batch_labels)
                    loss = outputs.loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    client_loss += loss.item()
            
            avg_loss = client_loss / (CONFIG['local_epochs'] * len(client_loader))
            round_losses.append(avg_loss)
            print(f"Loss: {avg_loss:.4f}")
            
            client_models.append(model.state_dict().copy())
        
        # Federated averaging
        print("  Aggregating client models...", end=' ')
        model = federated_averaging(model, [model] * len(client_models))
        print("✓")
        
        round_time = time.time() - round_start
        avg_round_loss = np.mean(round_losses)
        print(f"  Round Loss: {avg_round_loss:.4f} | Time: {round_time/60:.1f}min\n")
        
        # Save checkpoint
        if (round_num + 1) % CONFIG['save_every'] == 0 or round_num == CONFIG['num_rounds'] - 1:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{save_name}_round{round_num + 1}.pt"
            )
            torch.save({
                'round': round_num + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_round_loss,
                'config': CONFIG,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(CONFIG['checkpoint_dir'], f"{save_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    
    training_time = time.time() - training_start
    print(f"\n✅ ViT Training Complete!")
    print(f"   Total Time: {training_time/3600:.2f} hours")
    print(f"   Final Loss: {avg_round_loss:.4f}")
    print(f"   Saved: {final_path}\n")
    
    return model, avg_round_loss

def train_federated_vlm(model_type='clip', save_name='vlm_clip'):
    """Train Federated VLM (CLIP or BLIP) for multimodal plant analysis"""
    print(f"\n{'='*60}")
    print(f"🎭 Training Federated VLM: {model_type.upper()}")
    print(f"{'='*60}\n")
    
    if model_type == 'clip':
        from transformers import CLIPProcessor, CLIPModel
        
        print("📥 Loading CLIP model...")
        model_name = 'openai/clip-vit-base-patch32'
        processor = CLIPProcessor.from_pretrained(model_name)
        model = CLIPModel.from_pretrained(model_name)
    else:
        print("⚠️  BLIP-2 requires significant memory, using CLIP instead")
        return train_federated_vlm('clip', 'vlm_clip')
    
    model.to(device)
    print(f"   ✓ Model loaded: {model_name}")
    
    # Load data
    texts, images, labels = load_synthetic_multimodal_data(num_samples=800)
    client_texts, client_labels_text = split_data_non_iid(texts, labels, CONFIG['num_clients'])
    client_images, client_labels_img = split_data_non_iid(images, labels, CONFIG['num_clients'])
    
    # Federated training
    print(f"\n🔄 Starting Federated Training ({CONFIG['num_rounds']} rounds)...\n")
    training_start = time.time()
    
    for round_num in range(CONFIG['num_rounds']):
        round_start = time.time()
        print(f"Round {round_num + 1}/{CONFIG['num_rounds']}")
        
        client_models = []
        round_losses = []
        
        # Train each client
        for client_id in range(CONFIG['num_clients']):
            print(f"  Client {client_id + 1}/{CONFIG['num_clients']}...", end=' ')
            
            # Process data
            inputs = processor(
                text=client_texts[client_id],
                images=[img for img in client_images[client_id][:len(client_texts[client_id])]],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=CONFIG['max_length']
            )
            
            # Local training
            optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
            model.train()
            
            client_loss = 0
            for epoch in range(CONFIG['local_epochs']):
                # Simple contrastive loss
                outputs = model(**{k: v.to(device) for k, v in inputs.items()})
                
                # Contrastive loss between text and image embeddings
                text_embeds = outputs.text_embeds
                image_embeds = outputs.image_embeds
                
                # Normalized embeddings
                text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
                image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                
                # Contrastive loss
                logits = torch.matmul(text_embeds, image_embeds.t())
                labels_contrastive = torch.arange(len(text_embeds)).to(device)
                
                loss = torch.nn.functional.cross_entropy(logits, labels_contrastive)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                client_loss += loss.item()
            
            avg_loss = client_loss / CONFIG['local_epochs']
            round_losses.append(avg_loss)
            print(f"Loss: {avg_loss:.4f}")
            
            client_models.append(model.state_dict().copy())
        
        # Federated averaging
        print("  Aggregating client models...", end=' ')
        model = federated_averaging(model, [model] * len(client_models))
        print("✓")
        
        round_time = time.time() - round_start
        avg_round_loss = np.mean(round_losses)
        print(f"  Round Loss: {avg_round_loss:.4f} | Time: {round_time/60:.1f}min\n")
        
        # Save checkpoint
        if (round_num + 1) % CONFIG['save_every'] == 0 or round_num == CONFIG['num_rounds'] - 1:
            checkpoint_path = os.path.join(
                CONFIG['checkpoint_dir'],
                f"{save_name}_round{round_num + 1}.pt"
            )
            torch.save({
                'round': round_num + 1,
                'model_state_dict': model.state_dict(),
                'loss': avg_round_loss,
                'config': CONFIG,
            }, checkpoint_path)
            print(f"  💾 Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(CONFIG['checkpoint_dir'], f"{save_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    
    training_time = time.time() - training_start
    print(f"\n✅ VLM Training Complete!")
    print(f"   Total Time: {training_time/3600:.2f} hours")
    print(f"   Final Loss: {avg_round_loss:.4f}")
    print(f"   Saved: {final_path}\n")
    
    return model, avg_round_loss

# ============================================================================
# MAIN TRAINING PIPELINE
# ============================================================================

def main():
    """Main training pipeline"""
    print("\n" + "="*60)
    print("SELECT MODELS TO TRAIN")
    print("="*60)
    print("\n1. Federated LLM (DistilBERT) - ~2-3 hours")
    print("2. Federated ViT (ViT-Base) - ~5-7 hours")
    print("3. Federated VLM (CLIP) - ~4-5 hours")
    print("4. Train All Models - ~12-15 hours total")
    print("5. Quick Test (1 round, 2 clients) - ~30 minutes")
    print("\nEnter choice (1-5): ", end='')
    
    choice = input().strip()
    
    results = {}
    total_start = time.time()
    
    if choice == '1':
        model, loss = train_federated_llm('distilbert-base-uncased', 'llm_distilbert')
        results['llm'] = {'loss': loss}
    
    elif choice == '2':
        model, loss = train_federated_vit('google/vit-base-patch16-224', 'vit_base')
        results['vit'] = {'loss': loss}
    
    elif choice == '3':
        model, loss = train_federated_vlm('clip', 'vlm_clip')
        results['vlm'] = {'loss': loss}
    
    elif choice == '4':
        print("\n🚀 Starting complete training pipeline...\n")
        
        # Train LLM
        llm_model, llm_loss = train_federated_llm('distilbert-base-uncased', 'llm_distilbert')
        results['llm'] = {'loss': llm_loss}
        
        # Train ViT
        vit_model, vit_loss = train_federated_vit('google/vit-base-patch16-224', 'vit_base')
        results['vit'] = {'loss': vit_loss}
        
        # Train VLM
        vlm_model, vlm_loss = train_federated_vlm('clip', 'vlm_clip')
        results['vlm'] = {'loss': vlm_loss}
    
    elif choice == '5':
        print("\n⚡ Quick test mode (1 round, 2 clients)...\n")
        CONFIG['num_clients'] = 2
        CONFIG['num_rounds'] = 1
        CONFIG['local_epochs'] = 1
        
        model, loss = train_federated_llm('distilbert-base-uncased', 'llm_quick_test')
        results['quick_test'] = {'loss': loss}
    
    else:
        print("❌ Invalid choice")
        return
    
    total_time = time.time() - total_start
    
    # Save training summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_time_hours': total_time / 3600,
        'config': CONFIG,
        'results': results,
        'device': str(device),
    }
    
    summary_path = os.path.join(CONFIG['checkpoint_dir'], 'training_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print(f"\nTotal Time: {total_time/3600:.2f} hours")
    print(f"Checkpoints saved in: {CONFIG['checkpoint_dir']}/")
    print(f"Summary saved: {summary_path}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
