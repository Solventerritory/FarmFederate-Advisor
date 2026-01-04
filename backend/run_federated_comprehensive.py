#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_federated_comprehensive.py
==============================
Main execution script for comprehensive federated learning comparison

This script:
1. Loads all datasets (text + image)
2. Trains Federated LLM models (Flan-T5, GPT-2, etc.)
3. Trains Federated ViT models
4. Trains Federated VLM models (CLIP, BLIP)
5. Generates 15-20 comparison plots
6. Compares with baseline papers
7. Produces comprehensive evaluation report

Usage:
    python run_federated_comprehensive.py --quick_test  # Fast test with small models
    python run_federated_comprehensive.py --full        # Full comparison
    python run_federated_comprehensive.py --models flan-t5-small vit-base clip-base
"""

import os
import sys
import argparse
import time
import json
import gc
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# Import our modules
from federated_llm_vit_vlm_complete import (
    SEED, DEVICE, ISSUE_LABELS, NUM_LABELS,
    ModelConfig, MODEL_CONFIGS,
    TextDataset, ImageDataset, MultiModalDataset,
    FederatedLLM, FederatedViT, FederatedVLM,
    split_data_federated, train_federated_model, count_parameters
)

from federated_plotting_comparison import (
    ModelResults, ComparisonFramework
)

from research_paper_comparison import (
    RESEARCH_PAPERS, ResearchPaperComparator
)

# Try to import transformers tokenizers
from transformers import AutoTokenizer, ViTImageProcessor, CLIPProcessor

# ============================================================================
# DATASET LOADING
# ============================================================================

def load_real_crop_stress_dataset(data_dir="data/real_datasets", max_samples=10000):
    """Load real crop stress detection dataset"""
    print("[Data] Loading real crop stress detection dataset...")
    
    from pathlib import Path
    
    data_path = Path(data_dir)
    
    # Check if datasets exist
    text_csv = data_path / "text" / "crop_stress_descriptions.csv"
    image_csv = data_path / "plantvillage" / "metadata.csv"
    multimodal_csv = data_path / "multimodal" / "multimodal_pairs.csv"
    
    if not text_csv.exists():
        print(f"[Data] Real dataset not found at: {text_csv}")
        print("[Data] Please run: python download_real_datasets.py")
        print("[Data] Generating sample data for now...")
        return generate_sample_data(max_samples)
    
    try:
        # Load text dataset
        print(f"  Loading text dataset: {text_csv}")
        df = pd.read_csv(text_csv)
        
        # Convert stress_labels from string to list
        if 'stress_labels' in df.columns:
            df['stress_labels'] = df['stress_labels'].apply(eval)
        
        # Convert stress labels to multi-hot encoding
        label_lists = []
        for _, row in df.iterrows():
            stress_labels = row['stress_labels']
            label_indices = [ISSUE_LABELS.index(label) for label in stress_labels 
                           if label in ISSUE_LABELS]
            label_lists.append(label_indices if label_indices else [0])
        
        df['labels'] = label_lists
        
        # Limit samples
        if len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=SEED)
        
        print(f"  ✓ Loaded {len(df)} real crop stress samples")
        print(f"  ✓ Stress categories: {df['stress_labels'].explode().nunique()}")
        print(f"  ✓ Crops: {df['crop'].nunique()}")
        
        return df
    
    except Exception as e:
        print(f"[Data] Error loading real dataset: {e}")
        print("[Data] Generating sample data...")
        return generate_sample_data(max_samples)


def generate_sample_data(num_samples=5000):
    """Generate sample crop stress data"""
    print(f"[Data] Generating {num_samples} sample crop stress descriptions...")
    
    crops = ["tomato", "potato", "corn", "wheat", "rice", "soybean", "pepper"]
    
    templates = [
        "The {crop} crop shows visible signs of {stress} with {symptom}.",
        "Field inspection revealed {symptom} in {crop}, indicating {stress}.",
        "Observed {symptom} in {crop} plantation, suggesting {stress}.",
        "{crop} plants exhibit {symptom}, consistent with {stress}.",
        "Agricultural assessment: {crop} displays {symptom} indicating {stress}."
    ]
    
    stress_symptoms = {
        "water_stress": ["wilting leaves", "drooping stems", "dry soil", "leaf curling"],
        "nutrient_def": ["yellowing leaves", "stunted growth", "chlorosis", "pale coloration"],
        "pest_risk": ["holes in leaves", "insect damage", "leaf defoliation", "chewed edges"],
        "disease_risk": ["leaf spots", "mold growth", "lesions", "tissue discoloration"],
        "heat_stress": ["scorched leaves", "burnt edges", "blistering", "sun damage"],
    }
    
    texts = []
    labels_list = []
    
    for i in range(num_samples):
        # Random number of stress types (1-2)
        num_stresses = np.random.randint(1, 3)
        stress_types = np.random.choice(NUM_LABELS, size=num_stresses, replace=False).tolist()
        
        # Generate text
        crop = np.random.choice(crops)
        primary_stress = ISSUE_LABELS[stress_types[0]]
        symptom = np.random.choice(stress_symptoms[primary_stress])
        
        template = np.random.choice(templates)
        text = template.format(
            crop=crop,
            stress=primary_stress.replace('_', ' '),
            symptom=symptom
        )
        
        texts.append(text)
        labels_list.append(stress_types)
    
    df = pd.DataFrame({
        'text': texts,
        'labels': labels_list
    })
    
    print(f"  ✓ Generated {len(df)} sample texts")
    return df


# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def prepare_data_loaders(df, config: ModelConfig, num_clients=5, batch_size=16):
    """Prepare federated data loaders"""
    print(f"\n[DataLoaders] Preparing for {config.name}...")
    
    # Split into train/test
    train_df = df.sample(frac=0.8, random_state=SEED)
    test_df = df.drop(train_df.index).reset_index(drop=True)
    train_df = train_df.reset_index(drop=True)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    
    # Split train data among clients (non-IID)
    client_dfs = split_data_federated(train_df, num_clients, alpha=0.5)
    
    # Create datasets based on model type
    if config.model_type == "llm":
        # Text-only datasets
        tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)
        
        client_train_loaders = []
        client_val_loaders = []
        
        for client_df in client_dfs:
            # Split client data into train/val
            val_size = max(10, int(0.15 * len(client_df)))
            client_val_df = client_df.iloc[:val_size]
            client_train_df = client_df.iloc[val_size:]
            
            train_ds = TextDataset(client_train_df, tokenizer, config.max_length)
            val_ds = TextDataset(client_val_df, tokenizer, config.max_length)
            
            client_train_loaders.append(DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0
            ))
            client_val_loaders.append(DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            ))
        
        test_ds = TextDataset(test_df, tokenizer, config.max_length)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return client_train_loaders, client_val_loaders, test_loader
    
    elif config.model_type == "vit":
        # Image-only datasets
        image_processor = ViTImageProcessor.from_pretrained(config.pretrained_name)
        
        client_train_loaders = []
        client_val_loaders = []
        
        for client_df in client_dfs:
            val_size = max(10, int(0.15 * len(client_df)))
            client_val_df = client_df.iloc[:val_size]
            client_train_df = client_df.iloc[val_size:]
            
            train_ds = ImageDataset(client_train_df, image_processor, config.image_size)
            val_ds = ImageDataset(client_val_df, image_processor, config.image_size)
            
            client_train_loaders.append(DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0
            ))
            client_val_loaders.append(DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            ))
        
        test_ds = ImageDataset(test_df, image_processor, config.image_size)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return client_train_loaders, client_val_loaders, test_loader
    
    elif config.model_type == "vlm":
        # Multimodal datasets
        if "clip" in config.pretrained_name.lower():
            processor = CLIPProcessor.from_pretrained(config.pretrained_name)
            tokenizer = processor.tokenizer
            image_processor = processor.image_processor
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.pretrained_name)
            image_processor = ViTImageProcessor.from_pretrained(config.pretrained_name)
        
        client_train_loaders = []
        client_val_loaders = []
        
        for client_df in client_dfs:
            val_size = max(10, int(0.15 * len(client_df)))
            client_val_df = client_df.iloc[:val_size]
            client_train_df = client_df.iloc[val_size:]
            
            train_ds = MultiModalDataset(
                client_train_df, tokenizer, image_processor,
                config.max_length, config.image_size
            )
            val_ds = MultiModalDataset(
                client_val_df, tokenizer, image_processor,
                config.max_length, config.image_size
            )
            
            client_train_loaders.append(DataLoader(
                train_ds, batch_size=batch_size, shuffle=True, num_workers=0
            ))
            client_val_loaders.append(DataLoader(
                val_ds, batch_size=batch_size, shuffle=False, num_workers=0
            ))
        
        test_ds = MultiModalDataset(
            test_df, tokenizer, image_processor,
            config.max_length, config.image_size
        )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
        return client_train_loaders, client_val_loaders, test_loader
    
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def train_model(config: ModelConfig, df, num_rounds=5, num_clients=5, 
                batch_size=16, save_dir="results"):
    """Train a single model"""
    print(f"\n{'='*80}")
    print(f"TRAINING: {config.name}")
    print(f"{'='*80}\n")
    
    # Prepare data loaders
    client_train_loaders, client_val_loaders, test_loader = prepare_data_loaders(
        df, config, num_clients, batch_size
    )
    
    # Initialize model
    if config.model_type == "llm":
        model = FederatedLLM(config)
    elif config.model_type == "vit":
        model = FederatedViT(config)
    elif config.model_type == "vlm":
        model = FederatedVLM(config)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    params_count = count_parameters(model)
    print(f"[Model] Trainable parameters: {params_count:,}")
    
    # Train
    start_time = time.time()
    
    try:
        metrics_history, final_metrics, training_time = train_federated_model(
            model, config, client_train_loaders, client_val_loaders,
            test_loader, num_rounds=num_rounds, device=DEVICE, save_dir=save_dir
        )
    except Exception as e:
        print(f"[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0.0
    
    # Create result object
    result = ModelResults(config.name, config.model_type, config)
    result.metrics_history = metrics_history
    result.final_metrics = final_metrics
    result.training_time = training_time
    result.params_count = params_count
    result.memory_mb = memory_mb
    
    # Clean up
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return result


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Comprehensive Comparison")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Models to train (default: all)")
    parser.add_argument("--quick_test", action="store_true",
                       help="Quick test with small models and few rounds")
    parser.add_argument("--full", action="store_true",
                       help="Full comparison with all models")
    parser.add_argument("--rounds", type=int, default=5,
                       help="Number of federated rounds (default: 5)")
    parser.add_argument("--clients", type=int, default=5,
                       help="Number of federated clients (default: 5)")
    parser.add_argument("--batch_size", type=int, default=16,
                       help="Batch size (default: 16)")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Number of samples to use (default: 5000)")
    parser.add_argument("--use_real_data", action="store_true",
                       help="Use real crop stress datasets (run download_real_datasets.py first)")
    parser.add_argument("--data_dir", type=str, default="data/real_datasets",
                       help="Directory containing real datasets")
    parser.add_argument("--save_dir", type=str, default="results",
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("FEDERATED LEARNING COMPREHENSIVE COMPARISON")
    print("LLM (Text) + ViT (Image) + VLM (Multimodal)")
    print("="*80 + "\n")
    
    # Determine models to train
    if args.quick_test:
        print("[Mode] Quick test mode")
        models_to_train = ["flan-t5-small", "vit-base", "clip-base"]
        args.rounds = 3
        args.samples = 1000
    elif args.full:
        print("[Mode] Full comparison mode")
        models_to_train = list(MODEL_CONFIGS.keys())
        args.rounds = 10
    elif args.models:
        models_to_train = args.models
    else:
        # Default: representative subset
        models_to_train = [
            "flan-t5-small", "flan-t5-base", "gpt2",  # LLMs
            "vit-base", "deit-base",  # ViTs
            "clip-base", "blip"  # VLMs
        ]
    
    print(f"[Config] Models to train: {len(models_to_train)}")
    print(f"[Config] Rounds: {args.rounds}")
    print(f"[Config] Clients: {args.clients}")
    print(f"[Config] Batch size: {args.batch_size}")
    print(f"[Config] Samples: {args.samples}")
    print(f"[Config] Use real data: {args.use_real_data}")
    print(f"[Config] Device: {DEVICE}")
    print(f"[Config] Save directory: {args.save_dir}\n")
    
    # Load data
    if args.use_real_data:
        print(f"[Data] Loading real crop stress dataset from: {args.data_dir}")
        df = load_real_crop_stress_dataset(data_dir=args.data_dir, max_samples=args.samples)
    else:
        print("[Data] Generating sample crop stress data...")
        df = generate_sample_data(num_samples=args.samples)
    
    print(f"\n[Dataset] Total samples: {len(df)}")
    print(f"[Dataset] Label distribution:")
    label_counts = [0] * NUM_LABELS
    for labels in df['labels']:
        for label_idx in labels:
            label_counts[label_idx] += 1
    for label_name, count in zip(ISSUE_LABELS, label_counts):
        print(f"  {label_name}: {count}")
    
    # Initialize comparison framework
    comparison = ComparisonFramework(save_dir=os.path.join(args.save_dir, "comparisons"))
    
    # Train all models
    results = []
    total_start = time.time()
    
    for i, model_key in enumerate(models_to_train, 1):
        print(f"\n{'#'*80}")
        print(f"MODEL {i}/{len(models_to_train)}: {model_key}")
        print(f"{'#'*80}")
        
        if model_key not in MODEL_CONFIGS:
            print(f"[ERROR] Unknown model: {model_key}")
            continue
        
        config = MODEL_CONFIGS[model_key]
        
        try:
            result = train_model(
                config, df,
                num_rounds=args.rounds,
                num_clients=args.clients,
                batch_size=args.batch_size,
                save_dir=args.save_dir
            )
            
            if result is not None:
                results.append(result)
                comparison.add_result(result)
                
                print(f"\n[✓] {config.name} completed successfully")
                print(f"    Final Micro-F1: {result.final_metrics.get('micro_f1', 0):.4f}")
                print(f"    Final Macro-F1: {result.final_metrics.get('macro_f1', 0):.4f}")
                print(f"    Training time: {result.training_time:.1f}s")
            else:
                print(f"\n[✗] {config.name} training failed")
        
        except Exception as e:
            print(f"\n[ERROR] Exception during training: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    total_time = time.time() - total_start
    
    print(f"\n{'='*80}")
    print(f"ALL MODELS TRAINED")
    print(f"Total time: {total_time:.1f}s")
    print(f"Successful models: {len(results)}/{len(models_to_train)}")
    print(f"{'='*80}\n")
    
    # Generate all comparison plots
    if len(results) > 0:
        print("\n[Plots] Generating comprehensive comparison plots...")
        comparison.generate_all_plots()
        
        # Generate research paper comparisons
        print("\n[Research Papers] Comparing with state-of-the-art papers...")
        results_dict = {r.model_name: r for r in results}
        paper_comparator = ResearchPaperComparator(
            our_results=results_dict,
            save_dir=os.path.join(args.save_dir, "paper_comparison")
        )
        paper_comparator.generate_all_comparisons()
        
        print(f"\n{'='*80}")
        print("COMPARISON COMPLETE")
        print(f"{'='*80}")
        print(f"Results saved to: {args.save_dir}")
        print(f"Plots saved to: {comparison.save_dir}")
        print(f"Paper comparison plots saved to: {paper_comparator.save_dir}")
        print(f"Total plots generated: 30+ (20 internal + 10 paper comparisons)")
        print(f"{'='*80}\n")
    else:
        print("\n[ERROR] No successful model training, cannot generate plots")
    
    # Save summary JSON
    summary_path = os.path.join(args.save_dir, "training_summary.json")
    summary = {
        "total_models": len(models_to_train),
        "successful_models": len(results),
        "total_time_seconds": total_time,
        "config": {
            "rounds": args.rounds,
            "clients": args.clients,
            "batch_size": args.batch_size,
            "samples": args.samples,
        },
        "results": [r.to_dict() for r in results]
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"[Summary] Training summary saved to: {summary_path}")
    
    print("\n[✓] ALL DONE!")


if __name__ == "__main__":
    main()
