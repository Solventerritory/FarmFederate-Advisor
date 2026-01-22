from FarmFederate_Colab import check_imports, generate_synthetic_text_data, TextDataset, LightweightTextClassifier, train_model, Config
import torch
from torch.utils.data import DataLoader

# Setup
check_imports()
config = Config()
config.epochs = 2
config.batch_size = 8
config.max_samples_per_class = 10  # small dataset

# Generate data
text_df = generate_synthetic_text_data(config.max_samples_per_class * 5)
# Ensure labels column
if 'labels' not in text_df.columns and 'label' in text_df.columns:
    text_df['labels'] = text_df['label'].apply(lambda x: [int(x)])

# Split
train_size = int(0.8 * len(text_df))
text_train = text_df.iloc[:train_size]
text_val = text_df.iloc[train_size:]

# Datasets and loaders
train_ds = TextDataset(text_train, None, config.max_seq_length)
val_ds = TextDataset(text_val, None, config.max_seq_length)
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LightweightTextClassifier(num_labels=config.num_labels).to(device)

# Train
best_f1, history, final_metrics = train_model(model, train_loader, val_loader, config, device, 'text')

print('\n=== SMOKE TRAIN RESULT ===')
print('Best F1:', best_f1)
print('History:', history)
print('Final metrics:', final_metrics)
