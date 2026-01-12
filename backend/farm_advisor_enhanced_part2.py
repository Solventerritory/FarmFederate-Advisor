# Part 2: Model Architectures, Federated LLM, VLM, and Comparison Framework
# This file continues from farm_advisor_enhanced_full.py

# --------------------- Label helpers --------------------- 
def oversample_by_class(df: pd.DataFrame, target_each_map: Dict[int, int] = None) -> pd.DataFrame:
    if target_each_map is None:
        target_each_map = {0: 1500, 1: 2400, 2: 1700, 3: 1700, 4: 1500}
    
    idxs = {i: [] for i in range(NUM_LABELS)}
    for idx, labs in enumerate(df["labels"]):
        for k in labs:
            idxs[k].append(idx)
    
    keep = []
    for k, tgt in target_each_map.items():
        pool = idxs[k]
        if not pool:
            continue
        if len(pool) >= tgt:
            keep.extend(random.sample(pool, tgt))
        else:
            need = tgt - len(pool)
            keep.extend(pool + random.choices(pool, k=need))
    
    keep = sorted(set(keep))
    out = df.iloc[keep].copy()
    return out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)

def summarize_labels(df: pd.DataFrame, tag="set"):
    counts = np.zeros(NUM_LABELS, int)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    print(f"[{tag}] label counts:", {ISSUE_LABELS[i]: int(c) for i, c in enumerate(counts)})

def apply_label_noise(df: pd.DataFrame, p: float) -> pd.DataFrame:
    if p <= 0:
        return df
    rng = np.random.default_rng(SEED)
    rows = []
    for t, labs in df[["text", "labels"]].itertuples(index=False):
        labs = list(labs)
        if rng.random() < p:
            if labs and rng.random() < 0.5:
                del labs[rng.integers(0, len(labs))]
            else:
                k = rng.integers(0, NUM_LABELS)
                if k not in labs:
                    labs.append(k)
        labs = sorted(set(labs))
        rows.append((t, labs))
    return pd.DataFrame(rows, columns=["text", "labels"])

# --------------------- *** Enhanced Corpus Builder *** ---------- 
def build_corpus_with_images(image_csv: str, image_root: str = "", max_samples: int = 0) -> pd.DataFrame:
    if not os.path.exists(image_csv):
        raise RuntimeError(f"image_csv not found: {image_csv}")
    
    df_raw = pd.read_csv(image_csv)
    rows = []
    for _, r in df_raw.iterrows():
        text = str(r.get("text", "")).strip()
        fname = str(r.get("filename", "") or r.get("image_path", "")).strip()
        labs_raw = r.get("labels", "")
        if pd.isna(labs_raw) or str(labs_raw).strip() == "":
            labs = []
        elif isinstance(labs_raw, str):
            parts = [x.strip() for x in labs_raw.split(",") if x.strip()]
            labs = []
            for p in parts:
                if p.isdigit():
                    labs.append(int(p))
                elif p in LABEL_TO_ID:
                    labs.append(LABEL_TO_ID[p])
            labs = sorted(set(labs))
        elif isinstance(labs_raw, (list, tuple)):
            labs = list(labs_raw)
        else:
            labs = []
        
        if not labs:
            continue
        
        txt = fuse_text(simulate_sensor_summary(), text)
        rows.append((txt, labs, fname))
    
    out = pd.DataFrame(rows, columns=["text", "labels", "image_path"])
    if max_samples and len(out) > max_samples:
        out = out.sample(max_samples, random_state=SEED).reset_index(drop=True)
    return out

def build_corpus() -> pd.DataFrame:
    """
    Enhanced corpus builder supporting:
    - All available HF agricultural datasets
    - Multimodal (text + images)
    - Configurable sources
    """
    # Load all datasets if requested
    if ARGS.load_all_datasets:
        all_datasets = load_all_agricultural_datasets(ARGS.max_per_source)
        text_dfs = []
        img_dfs = []
        
        for name, df in all_datasets.items():
            if "image_path" in df.columns:
                img_dfs.append(df)
            else:
                text_dfs.append(df)
        
        if text_dfs:
            text_df = pd.concat(text_dfs, ignore_index=True)
        else:
            text_df = build_localmini(ARGS.max_samples, ARGS.mqtt_csv, ARGS.extra_csv)
        
        if img_dfs:
            img_df = pd.concat(img_dfs, ignore_index=True)
        else:
            img_df = None
    
    # Standard mix mode
    elif ARGS.dataset == "mix":
        print("[Dataset] MIX:", ARGS.mix_sources)
        text_df = build_mix(ARGS.max_per_source, ARGS.mqtt_csv, ARGS.extra_csv)
        img_df = None
    
    # Single dataset modes
    elif ARGS.dataset == "localmini":
        text_df = build_localmini(ARGS.max_samples or 0, ARGS.mqtt_csv, ARGS.extra_csv)
        img_df = None
    else:
        if ARGS.dataset == "gardian" and HAS_DATASETS:
            raws = build_gardian_stream(ARGS.max_per_source)
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        elif ARGS.dataset == "argilla" and HAS_DATASETS:
            raws = build_argilla_stream(ARGS.max_per_source)
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        else:
            raws = build_agnews_agri(ARGS.max_per_source) if HAS_DATASETS else []
            rows = [(fuse_text(simulate_sensor_summary(), r), weak_labels(r)) for r in raws]
            text_df = pd.DataFrame([(t, l) for (t, l) in rows if l], columns=["text", "labels"])
        img_df = None
    
    # Process text data
    summarize_labels(text_df, "pre-oversample")
    text_df = apply_label_noise(text_df, ARGS.label_noise)
    
    # Add OOD negatives
    ood = [
        "City council discussed budget allocations for public transport.",
        "The software team published patch notes for the new release.",
        "The arts festival announced its opening night lineup.",
    ]
    for t in ood:
        text_df.loc[len(text_df)] = [fuse_text(simulate_sensor_summary(), t), []]
    
    if ARGS.max_samples and len(text_df) > ARGS.max_samples:
        text_df = text_df.sample(ARGS.max_samples, random_state=SEED)
    text_df = text_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[Build] text size: {len(text_df)}")
    
    if len(text_df) == 0:
        raise RuntimeError("Empty dataset after filtering")
    
    # ----- IMAGE PART -----
    if ARGS.use_images and ARGS.image_csv and os.path.exists(ARGS.image_csv):
        print("[Build] Using provided image_csv")
        img_df = build_corpus_with_images(ARGS.image_csv, ARGS.image_dir, ARGS.max_samples)
    elif ARGS.use_images and img_df is None:
        print("[Build] Harvesting HF image datasets")
        parts = []
        for src in AGRICULTURAL_IMAGE_DATASETS:
            try:
                dfp = prepare_images_from_hf(src, ARGS.max_per_source, ARGS.image_dir)
                if len(dfp) > 0:
                    parts.append(dfp)
            except Exception as e:
                print(f"[Images] error preparing {src}: {e}")
        
        if parts:
            img_df = pd.concat(parts, ignore_index=True)
            if ARGS.max_samples and len(img_df) > ARGS.max_samples:
                img_df = img_df.sample(ARGS.max_samples, random_state=SEED).reset_index(drop=True)
            print(f"[Build] image size: {len(img_df)}")
    
    # ----- MERGE TEXT + IMAGES -----
    if "image_path" not in text_df.columns:
        text_df = text_df.copy()
        text_df["image_path"] = ""
    
    if img_df is None:
        final_df = text_df
    else:
        needed_cols = ["text", "labels", "image_path"]
        for c in needed_cols:
            if c not in img_df.columns:
                if c == "image_path":
                    img_df[c] = ""
        final_df = pd.concat([text_df[needed_cols], img_df[needed_cols]], ignore_index=True, sort=False)
    
    final_df = final_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"[Build] final multimodal size: {len(final_df)}")
    return final_df

# --------------------- Dataset classes --------------------- 
class MultiLabelDS(Dataset):
    def __init__(self, df, tok, max_len):
        self.df = df
        self.tok = tok
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            y[k] = 1.0
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": y,
            "raw_text": text,
        }

class MultiModalDS(Dataset):
    def __init__(self, df, tok, max_len, img_size=224, image_root=""):
        self.df = df.reset_index(drop=True)
        self.tok = tok
        self.max_len = max_len
        self.img_size = img_size
        self.image_root = image_root or ""
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.df)
    
    def _load_image(self, path: str):
        if not path or (isinstance(path, float) and pd.isna(path)):
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
        p = path if os.path.isabs(path) else os.path.join(self.image_root, path)
        try:
            im = Image.open(p).convert("RGB")
            return self.transform(im)
        except Exception:
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)
    
    def __getitem__(self, i):
        row = self.df.iloc[i]
        text = row["text"]
        enc = self.tok(
            text,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt",
        )
        y = torch.zeros(NUM_LABELS, dtype=torch.float32)
        for k in row["labels"]:
            y[k] = 1.0
        
        img_path = row.get("image_path", "")
        img = self._load_image(str(img_path) if not pd.isna(img_path) else "")
        
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "image": img,
            "labels": y,
            "raw_text": text,
            "image_path": img_path,
        }

def make_weights_for_balanced_classes(df: pd.DataFrame):
    counts = np.zeros(NUM_LABELS)
    for labs in df["labels"]:
        for k in labs:
            counts[k] += 1
    inv = 1.0 / np.maximum(counts, 1)
    inst_w = []
    for labs in df["labels"]:
        w = np.mean([inv[k] for k in labs]) if labs else np.mean(inv)
        inst_w.append(w)
    inst_w = np.array(inst_w, dtype=np.float32)
    inst_w = inst_w / (inst_w.mean() + 1e-12)
    return inst_w, counts

# --------------------- Loss --------------------- 
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor = None, gamma=2.5, label_smoothing=0.02):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.smooth = label_smoothing
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, logits, targets):
        if self.smooth > 0:
            targets = targets * (1 - self.smooth) + 0.5 * self.smooth
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha.view(1, -1)
        return loss.mean()

# --------------------- Enhanced Model Architectures --------------------- 
def build_tokenizer(model_name: str = None):
    """Build tokenizer with better error handling"""
    model_name = model_name or ARGS.model_name
    try:
        return AutoTokenizer.from_pretrained(model_name, local_files_only=ARGS.offline)
    except OSError as e:
        print(f"[Warn] failed to load tokenizer for {model_name}: {e}")
        print("[Warn] Retrying with local_files_only=True")
        try:
            return AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        except Exception as e2:
            raise RuntimeError(f"Tokenizer for {model_name} not found: {e2}")

def infer_lora_targets_from_model(model) -> List[str]:
    """Infer LoRA target modules from model architecture"""
    names = [n for n, _ in model.named_modules()]
    cand_sets = [
        ["q_lin", "k_lin", "v_lin", "out_lin"],  # DistilBERT
        ["query", "key", "value", "dense"],  # BERT/RoBERTa
        ["query_proj", "key_proj", "value_proj", "o_proj"],  # DeBERTa
        ["q_proj", "k_proj", "v_proj", "o_proj"],  # T5/GPT
        ["c_attn", "c_proj"],  # GPT-2
    ]
    for cands in cand_sets:
        found = [c for c in cands if any(("." + c) in n or n.endswith(c) for n in names)]
        if len(found) >= 2:
            return found
    return ["classifier"]

class FederatedLLMModel(nn.Module):
    """
    Federated LLM wrapper supporting:
    - Seq2Seq models (Flan-T5, T5)
    - Decoder-only models (GPT-2)
    - LoRA adaptation
    """
    def __init__(self, model_config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = model_config
        self.num_labels = num_labels
        
        print(f"[FederatedLLM] Loading {model_config.model_type}: {model_config.model_name}")
        
        # Load base model based on type
        if model_config.model_type == "seq2seq":
            try:
                self.base_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] Failed to load pretrained, using random init: {e}")
                cfg = AutoConfig.from_pretrained(model_config.model_name, local_files_only=True)
                self.base_model = AutoModelForSeq2SeqLM.from_config(cfg)
            hidden_size = self.base_model.config.d_model
        
        elif model_config.model_type == "decoder":
            try:
                self.base_model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] Failed to load pretrained, using random init: {e}")
                cfg = AutoConfig.from_pretrained(model_config.model_name, local_files_only=True)
                self.base_model = AutoModelForCausalLM.from_config(cfg)
            hidden_size = self.base_model.config.n_embd if hasattr(self.base_model.config, 'n_embd') else 768
        
        else:
            raise ValueError(f"Unsupported model type: {model_config.model_type}")
        
        # Freeze base if requested
        if model_config.freeze_base:
            for p in self.base_model.parameters():
                p.requires_grad = False
        
        # Apply LoRA
        if model_config.use_lora and HAS_PEFT:
            targets = infer_lora_targets_from_model(self.base_model)
            lora_config = LoraConfig(
                r=model_config.lora_r,
                lora_alpha=model_config.lora_alpha,
                lora_dropout=model_config.lora_dropout,
                bias="none",
                task_type="SEQ_2_SEQ_LM" if model_config.model_type == "seq2seq" else "CAUSAL_LM",
                target_modules=targets,
            )
            self.base_model = get_peft_model(self.base_model, lora_config)
            print(f"[LoRA] Applied to {targets}")
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_labels),
        )
        
        self.hidden_size = hidden_size
    
    def forward(self, input_ids=None, attention_mask=None):
        # Get encoder/decoder outputs
        if self.config.model_type == "seq2seq":
            outputs = self.base_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Pool encoder hidden states
            hidden = outputs.last_hidden_state.mean(dim=1)
        else:  # decoder
            outputs = self.base_model.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            # Pool using attention mask
            hidden = outputs.last_hidden_state * attention_mask.unsqueeze(-1)
            hidden = hidden.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        
        logits = self.classifier(hidden)
        return type("O", (), {"logits": logits})

class VisionLanguageModel(nn.Module):
    """
    Vision-Language Model wrapper supporting:
    - CLIP (contrastive learning)
    - BLIP (image-text matching)
    """
    def __init__(self, model_config: ModelConfig, num_labels: int):
        super().__init__()
        self.config = model_config
        self.num_labels = num_labels
        
        print(f"[VLM] Loading {model_config.name}: {model_config.model_name}")
        
        if "clip" in model_config.name.lower():
            try:
                self.vlm_model = CLIPModel.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
                self.processor = CLIPProcessor.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] CLIP load failed: {e}")
                raise
            hidden_size = self.vlm_model.config.projection_dim
        
        elif "blip" in model_config.name.lower():
            try:
                self.vlm_model = BlipForImageTextRetrieval.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
                self.processor = BlipProcessor.from_pretrained(
                    model_config.model_name,
                    local_files_only=ARGS.offline
                )
            except Exception as e:
                print(f"[Warn] BLIP load failed: {e}")
                raise
            hidden_size = self.vlm_model.config.hidden_size
        
        else:
            raise ValueError(f"Unsupported VLM: {model_config.name}")
        
        # Freeze VLM backbone
        for p in self.vlm_model.parameters():
            p.requires_grad = False
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(512, num_labels),
        )
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        if "clip" in self.config.name.lower():
            # CLIP forward
            outputs = self.vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=image,
                return_dict=True
            )
            # Use text-image similarity features
            text_embeds = outputs.text_embeds
            image_embeds = outputs.image_embeds
            # Concatenate or average
            combined = (text_embeds + image_embeds) / 2
        
        else:  # BLIP
            outputs = self.vlm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=image,
                return_dict=True
            )
            # Use pooled output
            combined = outputs.image_embeds.mean(dim=1) if hasattr(outputs, 'image_embeds') else outputs.last_hidden_state.mean(dim=1)
        
        logits = self.classifier(combined)
        return type("O", (), {"logits": logits})

class MultiModalModel(nn.Module):
    """
    Standard multimodal model with text encoder + ViT vision encoder
    """
    def __init__(
        self,
        text_model_name,
        vit_name,
        num_labels,
        freeze_text=True,
        freeze_vision=False,
        lora_r=8,
        lora_alpha=32,
        lora_dropout=0.05,
    ):
        super().__init__()
        
        # Text encoder
        try:
            text_base = AutoModel.from_pretrained(text_model_name, local_files_only=ARGS.offline)
        except OSError as e:
            print(f"[Warn] failed to load pretrained text model {text_model_name}: {e}")
            print("[Warn] Falling back to random-init text encoder")
            tcfg = AutoConfig.from_pretrained(text_model_name, local_files_only=True)
            text_base = AutoModel.from_config(tcfg)
        
        if freeze_text:
            for p in text_base.parameters():
                p.requires_grad = False
        
        if HAS_PEFT:
            targets = infer_lora_targets_from_model(text_base)
            lcfg = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                target_modules=targets,
            )
            text_peft = get_peft_model(text_base, lcfg)
            self.text_encoder = text_peft
            print(f"[LoRA Text] targets={targets}")
        else:
            self.text_encoder = text_base
        
        text_dim = getattr(self.text_encoder.config, "hidden_size", 768)
        
        # Vision encoder (ViT with robust fallback)
        try:
            self.vision = ViTModel.from_pretrained(vit_name, local_files_only=ARGS.offline)
        except Exception as e:
            print(f"[Warn] failed to load pretrained ViT {vit_name}: {e}")
            print("[Warn] Falling back to randomly initialized ViT")
            try:
                vcfg = AutoConfig.from_pretrained(vit_name, local_files_only=True)
            except Exception as e2:
                print(f"[Warn] ViT AutoConfig load failed: {e2}")
                vcfg = ViTConfig(
                    image_size=ARGS.img_size,
                    num_channels=3,
                    patch_size=16,
                    hidden_size=256,
                    num_hidden_layers=6,
                    num_attention_heads=8,
                    intermediate_size=512,
                )
            self.vision = ViTModel(vcfg)
        
        if freeze_vision:
            for p in self.vision.parameters():
                p.requires_grad = False
        
        vision_dim = getattr(self.vision.config, "hidden_size", 768)
        fusion_dim = text_dim + vision_dim
        
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, max(512, fusion_dim // 2)),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(max(512, fusion_dim // 2), num_labels),
        )
        
        print(f"[Model] text_dim={text_dim} vision_dim={vision_dim}")
    
    def forward(self, input_ids=None, attention_mask=None, image=None):
        txt_out = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        if hasattr(txt_out, "pooler_output") and txt_out.pooler_output is not None:
            tfeat = txt_out.pooler_output
        else:
            tfeat = txt_out.last_hidden_state.mean(dim=1)
        
        if image is None:
            vfeat = torch.zeros(tfeat.size(0), self.vision.config.hidden_size, device=tfeat.device)
        else:
            vit_out = self.vision(pixel_values=image, return_dict=True)
            if hasattr(vit_out, "pooler_output") and vit_out.pooler_output is not None:
                vfeat = vit_out.pooler_output
            else:
                vfeat = vit_out.last_hidden_state.mean(dim=1)
        
        feat = torch.cat([tfeat, vfeat], dim=1)
        logits = self.classifier(feat)
        return type("O", (), {"logits": logits})

def build_text_model(num_labels: int, freeze_base: bool = True, model_name: str = None):
    """Build standard encoder-based text model"""
    if not HAS_PEFT:
        raise RuntimeError("peft not available: install pip install peft")
    
    model_name = model_name or ARGS.model_name
    kwargs = dict(num_labels=num_labels, problem_type="multi_label_classification")
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, **kwargs, local_files_only=ARGS.offline
        )
    except OSError as e:
        print(f"[Warn] failed to load pretrained weights for {model_name}: {e}")
        print("[Warn] Falling back to random-init from local config")
        cfg = AutoConfig.from_pretrained(
            model_name,
            local_files_only=True,
            num_labels=num_labels,
            problem_type="multi_label_classification",
        )
        model = AutoModelForSequenceClassification.from_config(cfg)
    
    if freeze_base and hasattr(model, "base_model"):
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif freeze_base:
        for n, p in model.named_parameters():
            if "classifier" not in n:
                p.requires_grad = False
    
    targets = infer_lora_targets_from_model(model)
    lcfg = LoraConfig(
        r=ARGS.lora_r,
        lora_alpha=ARGS.lora_alpha,
        lora_dropout=ARGS.lora_dropout,
        bias="none",
        task_type="SEQ_CLS",
        target_modules=targets,
    )
    model = get_peft_model(model, lcfg)
    print(f"[LoRA] target_modules: {targets}")
    return model

def build_model_from_config(model_config: ModelConfig, num_labels: int):
    """Factory function to build models from config"""
    if model_config.model_type in ["seq2seq", "decoder"]:
        return FederatedLLMModel(model_config, num_labels)
    elif model_config.model_type == "vlm":
        return VisionLanguageModel(model_config, num_labels)
    elif model_config.model_type == "vit":
        # Pure vision model
        return MultiModalModel(
            "roberta-base",
            model_config.model_name,
            num_labels,
            freeze_text=True,
            freeze_vision=model_config.freeze_base,
        )
    else:  # encoder
        return build_text_model(num_labels, model_config.freeze_base, model_config.model_name)

def amp_enabled():
    return torch.cuda.is_available()

print("[✓] Model architectures loaded")
print("[✓] Federated LLM support: enabled")
print("[✓] VLM support: enabled")
