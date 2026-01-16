# ============================================================================
# ADVANCED FEATURES: Fusion Model, FocalLoss, Sensor Fusion, Weak Labeling
# ============================================================================
print("="*70)
print("LOADING ADVANCED FEATURES")
print("="*70)

# ============================================================================
# 1. SENSOR FUSION - Simulate IoT sensor data and fuse with text
# ============================================================================
def simulate_sensor_summary():
    """Generate realistic IoT sensor readings for agricultural monitoring."""
    soil_m = round(np.clip(np.random.normal(30, 8), 10, 50), 1)
    soil_ph = round(np.clip(np.random.normal(6.5, 0.5), 5.0, 8.0), 1)
    temp = round(np.clip(np.random.normal(28, 5), 15, 42), 1)
    humidity = round(np.clip(np.random.normal(60, 15), 25, 95), 0)
    vpd = round(np.clip(np.random.normal(1.4, 0.5), 0.4, 2.8), 2)
    rainfall = round(np.clip(np.random.exponential(2.0), 0.0, 15.0), 1)
    light = round(np.clip(np.random.normal(45000, 15000), 5000, 80000), 0)

    trend = np.random.choice(["rising", "falling", "stable"], p=[0.3, 0.3, 0.4])

    return {
        'soil_moisture': soil_m,
        'soil_ph': soil_ph,
        'temperature': temp,
        'humidity': humidity,
        'vpd': vpd,
        'rainfall_24h': rainfall,
        'light_intensity': light,
        'trend': trend
    }

def format_sensor_text(sensors: dict) -> str:
    """Format sensor readings as text for model input."""
    return (f"SENSORS: soil_moisture={sensors['soil_moisture']}%, "
            f"soil_pH={sensors['soil_ph']}, temp={sensors['temperature']}°C, "
            f"humidity={sensors['humidity']}%, VPD={sensors['vpd']} kPa, "
            f"rainfall_24h={sensors['rainfall_24h']}mm, "
            f"light={sensors['light_intensity']} lux (trend: {sensors['trend']})")

def fuse_sensor_with_text(text: str, sensors: dict = None) -> str:
    """Fuse sensor data with text description."""
    if sensors is None:
        sensors = simulate_sensor_summary()
    sensor_str = format_sensor_text(sensors)
    return f"{sensor_str}\nLOG: {text.strip()}"

def sensor_to_priors(sensors: dict) -> np.ndarray:
    """Convert sensor readings to label prior probabilities."""
    priors = np.zeros(NUM_LABELS, dtype=np.float32)

    # Water stress indicators
    if sensors['soil_moisture'] < 20 or sensors['vpd'] > 2.0:
        priors[0] += 0.3  # water_stress
    if sensors['soil_moisture'] > 45:
        priors[0] -= 0.2

    # Nutrient deficiency indicators
    if sensors['soil_ph'] < 5.5 or sensors['soil_ph'] > 7.5:
        priors[1] += 0.2  # nutrient_def

    # Pest risk indicators
    if 20 < sensors['temperature'] < 32 and 50 < sensors['humidity'] < 75:
        priors[2] += 0.15  # pest_risk (favorable conditions)

    # Disease risk indicators
    if sensors['humidity'] > 80 or sensors['rainfall_24h'] > 5:
        priors[3] += 0.25  # disease_risk
    if sensors['humidity'] < 40:
        priors[3] -= 0.15

    # Heat stress indicators
    if sensors['temperature'] > 35 or sensors['vpd'] > 2.2:
        priors[4] += 0.3  # heat_stress
    if sensors['temperature'] < 25:
        priors[4] -= 0.2

    return np.clip(priors, -0.5, 0.5)

# ============================================================================
# 2. WEAK LABELING - Keyword-based automatic labeling
# ============================================================================
STRESS_KEYWORDS = {
    'water_stress': [
        'dry', 'wilting', 'wilt', 'drought', 'parched', 'moisture', 'irrigation',
        'drooping', 'cracking soil', 'water stress', 'dehydration', 'thirsty'
    ],
    'nutrient_def': [
        'nitrogen', 'phosphorus', 'potassium', 'npk', 'fertilizer', 'chlorosis',
        'yellowing', 'pale leaves', 'stunted', 'deficiency', 'nutrient', 'spad',
        'interveinal', 'necrotic margin', 'micronutrient'
    ],
    'pest_risk': [
        'pest', 'aphid', 'whitefly', 'borer', 'caterpillar', 'larvae', 'insect',
        'thrips', 'mites', 'weevil', 'hopper', 'chewed', 'holes', 'webbing',
        'honeydew', 'frass', 'infestation'
    ],
    'disease_risk': [
        'blight', 'rust', 'mildew', 'rot', 'fungal', 'bacterial', 'viral',
        'lesion', 'spot', 'necrosis', 'pathogen', 'infection', 'disease',
        'powdery', 'downy', 'canker', 'wilt disease', 'mosaic'
    ],
    'heat_stress': [
        'heat', 'hot', 'scorch', 'sunburn', 'thermal', 'high temperature',
        'heatwave', 'burning', 'desiccation', 'heat stress', 'blistering'
    ]
}

def weak_label_text(text: str) -> List[int]:
    """Generate weak labels from text using keyword matching."""
    text_lower = text.lower()
    labels = []

    for label_name, keywords in STRESS_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            labels.append(ISSUE_LABELS.index(label_name))

    return sorted(set(labels))

def enhance_labels_with_sensors(text_labels: List[int], sensors: dict, threshold: float = 0.2) -> List[int]:
    """Enhance text-based labels with sensor-based priors."""
    priors = sensor_to_priors(sensors)
    enhanced = set(text_labels)

    for i, prior in enumerate(priors):
        if prior > threshold and i not in enhanced:
            enhanced.add(i)

    return sorted(enhanced)

# ============================================================================
# 3. FOCAL LOSS - Handle class imbalance
# ============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in multi-label classification."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 label_smoothing: float = 0.02, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing

        # Compute BCE
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Compute focal weight
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha weighting
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # Compute focal loss
        focal_loss = alpha_weight * focal_weight * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

# ============================================================================
# 4. MULTIMODAL FUSION MODEL - Combines Text + Image in single architecture
# ============================================================================
class MultiModalFusionModel(nn.Module):
    """
    Unified multimodal model that fuses text (LLM) and image (ViT) features.
    This is different from CLIP - it's a custom fusion architecture.
    """

    def __init__(self, text_model_name: str, vit_model_name: str, num_labels: int,
                 fusion_type: str = 'concat', use_lora: bool = True,
                 freeze_text: bool = True, freeze_vision: bool = True):
        super().__init__()

        self.fusion_type = fusion_type
        self.num_labels = num_labels

        # Text Encoder (LLM)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.text_hidden_size = self.text_encoder.config.hidden_size

        if freeze_text:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA to text encoder
        if use_lora and HAS_PEFT:
            lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=get_lora_target_modules(text_model_name),
                lora_dropout=0.1, bias="none"
            )
            self.text_encoder = get_peft_model(self.text_encoder, lora_config)

        # Vision Encoder (ViT)
        self.vision_encoder = ViTModel.from_pretrained(vit_model_name)
        self.vision_hidden_size = self.vision_encoder.config.hidden_size

        if freeze_vision:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        # Apply LoRA to vision encoder
        if use_lora and HAS_PEFT:
            vit_lora_config = LoraConfig(
                r=8, lora_alpha=16,
                target_modules=["query", "value"],
                lora_dropout=0.1, bias="none"
            )
            self.vision_encoder = get_peft_model(self.vision_encoder, vit_lora_config)

        # Fusion layers
        if fusion_type == 'concat':
            fusion_dim = self.text_hidden_size + self.vision_hidden_size
        elif fusion_type == 'attention':
            fusion_dim = self.text_hidden_size
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=self.text_hidden_size,
                num_heads=8,
                dropout=0.1,
                batch_first=True
            )
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
        elif fusion_type == 'gated':
            fusion_dim = self.text_hidden_size
            self.gate = nn.Sequential(
                nn.Linear(self.text_hidden_size + self.vision_hidden_size, self.text_hidden_size),
                nn.Sigmoid()
            )
            self.vision_proj = nn.Linear(self.vision_hidden_size, self.text_hidden_size)
        else:
            fusion_dim = self.text_hidden_size + self.vision_hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_labels)
        )

        # Sensor prior integration
        self.sensor_proj = nn.Linear(num_labels, num_labels)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None,
                sensor_priors=None, return_features=False):

        # Text encoding
        if input_ids is not None:
            text_outputs = self.text_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
                text_features = text_outputs.pooler_output
            else:
                text_features = text_outputs.last_hidden_state[:, 0]
        else:
            text_features = None

        # Vision encoding
        if pixel_values is not None:
            vision_outputs = self.vision_encoder(
                pixel_values=pixel_values,
                return_dict=True
            )
            if hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                vision_features = vision_outputs.pooler_output
            else:
                vision_features = vision_outputs.last_hidden_state[:, 0]
        else:
            vision_features = None

        # Fusion
        if text_features is not None and vision_features is not None:
            if self.fusion_type == 'concat':
                fused = torch.cat([text_features, vision_features], dim=-1)
            elif self.fusion_type == 'attention':
                vision_proj = self.vision_proj(vision_features).unsqueeze(1)
                text_seq = text_features.unsqueeze(1)
                attn_out, _ = self.cross_attention(text_seq, vision_proj, vision_proj)
                fused = (text_features + attn_out.squeeze(1)) / 2
            elif self.fusion_type == 'gated':
                vision_proj = self.vision_proj(vision_features)
                gate = self.gate(torch.cat([text_features, vision_features], dim=-1))
                fused = text_features + gate * vision_proj
            else:
                fused = torch.cat([text_features, vision_features], dim=-1)
        elif text_features is not None:
            # Text only - pad with zeros for vision
            if self.fusion_type == 'concat':
                fused = torch.cat([text_features, torch.zeros(text_features.size(0), self.vision_hidden_size, device=text_features.device)], dim=-1)
            else:
                fused = text_features
        elif vision_features is not None:
            # Vision only - pad with zeros for text
            if self.fusion_type == 'concat':
                fused = torch.cat([torch.zeros(vision_features.size(0), self.text_hidden_size, device=vision_features.device), vision_features], dim=-1)
            else:
                fused = self.vision_proj(vision_features) if hasattr(self, 'vision_proj') else vision_features
        else:
            raise ValueError("At least one of text or image must be provided")

        # Classification
        logits = self.classifier(fused)

        # Apply sensor priors if available
        if sensor_priors is not None:
            prior_adjustment = self.sensor_proj(sensor_priors)
            logits = logits + 0.3 * prior_adjustment

        if return_features:
            return logits, fused
        return logits

# ============================================================================
# 5. ADVANCED TRAINING UTILITIES
# ============================================================================
class EMAModel:
    """Exponential Moving Average for model weights."""

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )

    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def client_dropout(client_models: List, dropout_prob: float = 0.1) -> List:
    """Simulate client dropout during federated learning."""
    if dropout_prob <= 0:
        return client_models

    kept = []
    for model in client_models:
        if np.random.random() > dropout_prob:
            kept.append(model)

    # Ensure at least one client remains
    if len(kept) == 0:
        kept = [client_models[np.random.randint(len(client_models))]]

    return kept

# ============================================================================
# 6. ENHANCED DATASET CLASS
# ============================================================================
class EnhancedMultiModalDataset(Dataset):
    """Enhanced dataset with sensor fusion and weak labeling."""

    def __init__(self, texts, images, labels, sources=None,
                 tokenizer=None, image_transform=None,
                 max_length=128, use_sensor_fusion=True,
                 use_weak_labels=True):

        self.texts = texts
        self.images = images
        self.labels = labels
        self.sources = sources
        self.tokenizer = tokenizer
        self.image_transform = image_transform
        self.max_length = max_length
        self.use_sensor_fusion = use_sensor_fusion
        self.use_weak_labels = use_weak_labels

        # Pre-generate sensors for consistency
        self.sensors = [simulate_sensor_summary() for _ in range(len(texts))]

    def __len__(self):
        return len(self.texts) if self.texts is not None else len(self.images)

    def __getitem__(self, idx):
        item = {}
        sensors = self.sensors[idx]

        # Process text
        if self.texts is not None and self.tokenizer is not None:
            text = str(self.texts[idx])

            # Add sensor fusion
            if self.use_sensor_fusion:
                text = fuse_sensor_with_text(text, sensors)

            encoded = self.tokenizer(
                text, max_length=self.max_length,
                padding='max_length', truncation=True,
                return_tensors='pt'
            )
            item['input_ids'] = encoded['input_ids'].squeeze(0)
            item['attention_mask'] = encoded['attention_mask'].squeeze(0)
            item['raw_text'] = text

        # Process image
        if self.images is not None:
            img = self.images[idx]
            if isinstance(img, str):
                try:
                    img = Image.open(img).convert('RGB')
                except:
                    img = Image.new('RGB', (224, 224), color='gray')
            elif isinstance(img, np.ndarray):
                img = Image.fromarray(img)

            if self.image_transform is not None:
                item['pixel_values'] = self.image_transform(img)
            else:
                item['pixel_values'] = T.ToTensor()(img)

        # Process labels
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, list):
                label_tensor = torch.zeros(NUM_LABELS, dtype=torch.float32)
                for l in label:
                    if isinstance(l, int) and 0 <= l < NUM_LABELS:
                        label_tensor[l] = 1.0
            else:
                label_tensor = torch.tensor(label, dtype=torch.float32)

            # Enhance with weak labels if enabled
            if self.use_weak_labels and self.texts is not None:
                weak = weak_label_text(str(self.texts[idx]))
                for l in weak:
                    label_tensor[l] = max(label_tensor[l], 0.8)  # Soft label

            item['labels'] = label_tensor

        # Add sensor priors
        item['sensor_priors'] = torch.tensor(sensor_to_priors(sensors), dtype=torch.float32)

        return item

print("[OK] Advanced features loaded:")
print("  - Sensor fusion (IoT data simulation)")
print("  - Weak labeling (keyword-based)")
print("  - FocalLoss (class imbalance handling)")
print("  - MultiModalFusionModel (text+image fusion)")
print("  - EMA (exponential moving average)")
print("  - Client dropout simulation")
print("  - Enhanced dataset class")
