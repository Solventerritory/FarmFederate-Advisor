"""
Quick Start Script for Enhanced Farm Advisor System
Run different configurations easily
"""

import os
import sys

# Preset configurations
CONFIGS = {
    "baseline": {
        "name": "Baseline RoBERTa",
        "model_type": "roberta",
        "use_images": False,
        "rounds": 2,
        "clients": 4,
        "compare_all": False,
    },
    "multimodal": {
        "name": "Multimodal (Text + Images)",
        "model_type": "roberta",
        "use_images": True,
        "image_dir": "images_hf",
        "rounds": 2,
        "clients": 4,
    },
    "federated_llm": {
        "name": "Federated LLM (Flan-T5)",
        "model_type": "flan-t5-small",
        "use_federated_llm": True,
        "rounds": 3,
        "clients": 5,
        "use_images": False,
    },
    "vlm": {
        "name": "Vision-Language Model (CLIP)",
        "model_type": "clip",
        "use_vlm": True,
        "use_images": True,
        "image_dir": "images_hf",
        "rounds": 2,
        "clients": 4,
    },
    "compare_all": {
        "name": "Full Model Comparison",
        "compare_all": True,
        "load_all_datasets": True,
        "use_images": True,
        "rounds": 2,
        "clients": 4,
        "save_comparisons": True,
    },
    "quick_test": {
        "name": "Quick Test (Fast)",
        "model_type": "distilbert",
        "max_per_source": 100,
        "max_samples": 500,
        "rounds": 1,
        "clients": 2,
        "batch_size": 4,
        "lowmem": True,
    },
}

def display_menu():
    print("\n" + "="*80)
    print("🌾 ENHANCED FARM ADVISOR - QUICK START")
    print("="*80)
    print("\nAvailable Configurations:")
    for i, (key, config) in enumerate(CONFIGS.items(), 1):
        print(f"{i}. {config['name']} ({key})")
    print(f"{len(CONFIGS)+1}. Custom configuration")
    print(f"{len(CONFIGS)+2}. Exit")
    print("="*80)

def update_args_override(config):
    """Update ArgsOverride in the main script"""
    script_path = "farm_advisor_complete.py"
    
    if not os.path.exists(script_path):
        print(f"❌ Error: {script_path} not found!")
        return False
    
    # Read the script
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find and replace ArgsOverride section
    import_start = content.find("class ArgsOverride:")
    if import_start == -1:
        print("❌ Error: Could not find ArgsOverride class!")
        return False
    
    # Find the end of the class (next non-indented line or next class/def)
    import_end = content.find("\n\n# Apply overrides", import_start)
    if import_end == -1:
        import_end = content.find("\nclass ", import_start + 1)
    if import_end == -1:
        import_end = content.find("\ndef ", import_start + 1)
    
    # Build new ArgsOverride
    new_override = "class ArgsOverride:\n"
    for key, value in config.items():
        if key == "name":
            continue
        if isinstance(value, str):
            new_override += f"    {key} = \"{value}\"\n"
        else:
            new_override += f"    {key} = {value}\n"
    
    # Replace
    new_content = content[:import_start] + new_override + content[import_end:]
    
    # Write back
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print("✅ Configuration updated successfully!")
    return True

def main():
    while True:
        display_menu()
        
        try:
            choice = int(input("\nSelect configuration (number): "))
        except ValueError:
            print("❌ Invalid input! Please enter a number.")
            continue
        
        if choice == len(CONFIGS) + 2:
            print("👋 Goodbye!")
            sys.exit(0)
        
        if choice == len(CONFIGS) + 1:
            print("\n📝 Custom Configuration Mode")
            print("Edit farm_advisor_complete.py manually to set custom parameters.")
            input("Press Enter to continue...")
            continue
        
        if choice < 1 or choice > len(CONFIGS):
            print("❌ Invalid choice!")
            continue
        
        # Get selected config
        config_key = list(CONFIGS.keys())[choice - 1]
        config = CONFIGS[config_key]
        
        print(f"\n✓ Selected: {config['name']}")
        print("\nConfiguration:")
        for key, value in config.items():
            if key != "name":
                print(f"  {key}: {value}")
        
        confirm = input("\nProceed with this configuration? (y/n): ").lower()
        if confirm != 'y':
            continue
        
        # Update script
        if update_args_override(config):
            print("\n🚀 Starting training...")
            print("="*80)
            
            # Run the script
            os.system("python farm_advisor_complete.py")
            
            print("\n" + "="*80)
            print("✅ Training complete!")
            print("="*80)
        
        again = input("\nRun another configuration? (y/n): ").lower()
        if again != 'y':
            break
    
    print("\n👋 Thank you for using Farm Advisor!")

if __name__ == "__main__":
    main()
