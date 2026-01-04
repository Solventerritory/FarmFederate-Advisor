# Part 3: Training, Evaluation, Comparison Framework, and Execution

# --------------------- Calibration & Metrics + Plots --------------------- 
def calibrate_thresholds(model, loader, precision_target=0.90) -> np.ndarray:
    model.eval()
    model.to(DEVICE)
    probs_all = []
    y_all = []
    
    with torch.no_grad():
        for b in loader:
            bt = {k: v.to(DEVICE) for k, v in b.items() if k not in ("labels", "raw_text", "image_path")}
            out = model(**bt)
            logits = out.logits
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            y_all.append(b["labels"].numpy())
    
    P = np.vstack(probs_all)
    T = np.vstack(y_all)
    C = P.shape[1]
    thr = np.zeros(C, dtype=np.float32)
    
    for j in range(C):
        col, y = P[:, j], T[:, j].astype(int)
        best_t_f1, best_f1 = 0.5, -1.0
        best_t_prec = None
        for t in np.linspace(0.05, 0.9, 35):
            pred = (col >= t).astype(int)
            prec = precision_score(y, pred, zero_division=0)
            f1v = f1_score(y, pred, zero_division=0)
            if prec >= precision_target:
                if best_t_prec is None or f1v > best_f1:
                    best_t_prec, best_f1 = t, f1v
            if f1v > best_f1:
                best_t_f1, best_f1 = t, f1v
        thr[j] = best_t_prec if best_t_prec is not None else best_t_f1
    
    thr = np.clip(thr, 0.20, 0.80)
    return thr

def evaluate_with_thr(model, loader, thr) -> Dict[str, float]:
    def _cap(x):
        return min(0.999, float(x))
    
    model.eval()
    model.to(DEVICE)
    P_all, T_all, R_all = [], [], []
    
    with torch.no_grad():
        for b in loader:
            bt = {k: v.to(DEVICE) for k, v in b.items() if k not in ("labels", "raw_text", "image_path")}
            out = model(**bt)
            logits = out.logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs >= thr).astype(int)
            P_all.append(preds)
            T_all.append(b["labels"].numpy())
            R_all.append(probs)
    
    P = np.vstack(P_all)
    T = np.vstack(T_all)
    R = np.vstack(R_all)
    
    micro = f1_score(T, P, average="micro", zero_division=0)
    macro = f1_score(T, P, average="macro", zero_division=0)
    prec = precision_score(T, P, average=None, zero_division=0)
    rec = recall_score(T, P, average=None, zero_division=0)
    f1s = [f1_score(T[:, i], P[:, i], zero_division=0) for i in range(NUM_LABELS)]
    supports = T.sum(axis=0)
    
    if not ARGS.quiet_eval:
        print("\nPer-class metrics:")
        for i, lab in enumerate(ISSUE_LABELS):
            if supports[i] < 20:
                print(f"  - {lab:14s} | insufficient support (n={int(supports[i])})")
                continue
            print(f"  - {lab:14s} | P={_cap(prec[i]):.3f} R={_cap(rec[i]):.3f} F1={_cap(f1s[i]):.3f} thr={thr[i]:.2f}")
        print(f"\nOverall: micro-F1={_cap(micro):.3f} macro-F1={_cap(macro):.3f}")
        
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score
            pr_micro = average_precision_score(T, R, average="micro")
            pr_macro = average_precision_score(T, R, average="macro")
            roc_micro = roc_auc_score(T, R, average="micro")
            roc_macro = roc_auc_score(T, R, average="macro")
            print(f"AUPRC micro={_cap(pr_micro):.3f} macro={_cap(pr_macro):.3f} | AUROC micro={_cap(roc_micro):.3f} macro={_cap(roc_macro):.3f}")
        except Exception:
            pass
    
    return {
        "micro_f1": micro,
        "macro_f1": macro,
        "per_class": {"precision": prec, "recall": rec, "f1": np.array(f1s)},
        "supports": supports,
    }

def save_tables_and_plots(metrics: Dict[str, float], save_dir: str, model_name: str = ""):
    os.makedirs(save_dir, exist_ok=True)
    prec = metrics["per_class"]["precision"]
    rec = metrics["per_class"]["recall"]
    f1s = metrics["per_class"]["f1"]
    
    df = pd.DataFrame({"Label": ISSUE_LABELS, "Precision": prec, "Recall": rec, "F1": f1s})
    csv_path = os.path.join(save_dir, f"results_table_{model_name}.csv" if model_name else "results_table.csv")
    df.to_csv(csv_path, index=False)
    
    plt.figure(figsize=(10, 6))
    x = np.arange(len(ISSUE_LABELS))
    width = 0.25
    plt.bar(x - width, prec, width, label='Precision', alpha=0.8)
    plt.bar(x, rec, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1s, width, label='F1', alpha=0.8)
    plt.xlabel('Issue Type')
    plt.ylabel('Score')
    plt.title(f'Per-Class Metrics{" - " + model_name if model_name else ""}')
    plt.xticks(x, ISSUE_LABELS, rotation=45, ha='right')
    plt.legend()
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    png_path = os.path.join(save_dir, f"metrics_bar_{model_name}.png" if model_name else "metrics_bar.png")
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    
    print(f"[Saved] {csv_path}")
    print(f"[Saved] {png_path}")

# --------------------- Comparison Framework --------------------- 
@dataclass
class ComparisonResult:
    model_name: str
    model_type: str
    micro_f1: float
    macro_f1: float
    per_class_f1: np.ndarray
    training_time: float
    inference_time: float
    params_count: int
    memory_usage: float

class ModelComparator:
    """Framework for comparing multiple model architectures"""
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.results: List[ComparisonResult] = []
        os.makedirs(save_dir, exist_ok=True)
    
    def add_result(self, result: ComparisonResult):
        self.results.append(result)
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.results:
            print("[Comparator] No results to compare")
            return
        
        print("\n" + "="*80)
        print("MODEL COMPARISON REPORT")
        print("="*80)
        
        # Create comparison dataframe
        comp_data = []
        for r in self.results:
            comp_data.append({
                "Model": r.model_name,
                "Type": r.model_type,
                "Micro-F1": f"{r.micro_f1:.4f}",
                "Macro-F1": f"{r.macro_f1:.4f}",
                "Avg F1": f"{r.per_class_f1.mean():.4f}",
                "Train Time (s)": f"{r.training_time:.1f}",
                "Infer Time (s)": f"{r.inference_time:.3f}",
                "Params (M)": f"{r.params_count/1e6:.2f}",
                "Memory (MB)": f"{r.memory_usage:.1f}",
            })
        
        df = pd.DataFrame(comp_data)
        print(df.to_string(index=False))
        
        # Save to CSV
        csv_path = os.path.join(self.save_dir, "model_comparison.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n[Saved] {csv_path}")
        
        # Generate comparison plots
        self._plot_f1_comparison()
        self._plot_efficiency_comparison()
        self._plot_per_class_heatmap()
    
    def _plot_f1_comparison(self):
        """Plot F1 score comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r.model_name for r in self.results]
        micro_f1s = [r.micro_f1 for r in self.results]
        macro_f1s = [r.macro_f1 for r in self.results]
        
        x = np.arange(len(names))
        width = 0.35
        
        ax1.bar(x - width/2, micro_f1s, width, label='Micro-F1', alpha=0.8)
        ax1.bar(x + width/2, macro_f1s, width, label='Macro-F1', alpha=0.8)
        ax1.set_xlabel('Model')
        ax1.set_ylabel('F1 Score')
        ax1.set_title('F1 Score Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(names, rotation=45, ha='right')
        ax1.legend()
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        
        # Per-class average
        avg_f1s = [r.per_class_f1.mean() for r in self.results]
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        ax2.barh(names, avg_f1s, color=colors, alpha=0.8)
        ax2.set_xlabel('Average Per-Class F1')
        ax2.set_title('Average Performance Across All Classes')
        ax2.set_xlim(0, 1.0)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "f1_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_comparison(self):
        """Plot efficiency metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        names = [r.model_name for r in self.results]
        train_times = [r.training_time for r in self.results]
        params = [r.params_count / 1e6 for r in self.results]
        
        colors = plt.cm.plasma(np.linspace(0, 1, len(names)))
        
        ax1.barh(names, train_times, color=colors, alpha=0.8)
        ax1.set_xlabel('Training Time (seconds)')
        ax1.set_title('Training Efficiency')
        ax1.grid(axis='x', alpha=0.3)
        
        ax2.barh(names, params, color=colors, alpha=0.8)
        ax2.set_xlabel('Parameters (Millions)')
        ax2.set_title('Model Size')
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "efficiency_comparison.png"), dpi=200, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_heatmap(self):
        """Plot per-class F1 heatmap across models"""
        matrix = np.array([r.per_class_f1 for r in self.results])
        names = [r.model_name for r in self.results]
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            matrix,
            xticklabels=ISSUE_LABELS,
            yticklabels=names,
            annot=True,
            fmt='.3f',
            cmap='YlGnBu',
            vmin=0,
            vmax=1,
            cbar_kws={'label': 'F1 Score'}
        )
        plt.title('Per-Class F1 Scores Across Models')
        plt.xlabel('Issue Type')
        plt.ylabel('Model')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "per_class_heatmap.png"), dpi=200, bbox_inches='tight')
        plt.close()

# --------------------- Federated utils --------------------- 
def split_clients(df: pd.DataFrame, n: int, alpha: float) -> List[pd.DataFrame]:
    """Split data non-IID using Dirichlet distribution"""
    prim = []
    rng = np.random.default_rng(SEED)
    for labs in df["labels"]:
        if labs:
            prim.append(int(rng.choice(labs)))
        else:
            prim.append(int(rng.integers(0, NUM_LABELS)))
    
    df2 = df.copy()
    df2["_y"] = prim
    class_client_probs = rng.dirichlet([alpha] * n, size=NUM_LABELS)
    client_bins = [[] for _ in range(n)]
    
    for idx, y in enumerate(df2["_y"].tolist()):
        k = int(rng.choice(n, p=class_client_probs[y]))
        client_bins[k].append(idx)
    
    out = []
    for k in range(n):
        part = df2.iloc[client_bins[k]].drop(columns=["_y"]).reset_index(drop=True)
        out.append(part)
    return out

def ema_update(ema_params, model_params, decay):
    for ep, mp in zip(ema_params, model_params):
        ep.data.mul_(decay).add_(mp.data, alpha=1.0 - decay)

def train_local(
    model,
    tok,
    tr_df,
    va_df,
    class_alpha: torch.Tensor
) -> Tuple[float, float, Dict, np.ndarray, int]:
    """Local training for one client"""
    if ARGS.use_images and "image_path" in tr_df.columns:
        tr_ds = MultiModalDS(tr_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
        va_ds = MultiModalDS(va_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir)
    else:
        tr_ds = MultiLabelDS(tr_df, tok, ARGS.max_len)
        va_ds = MultiLabelDS(va_df, tok, ARGS.max_len)
    
    weights, _ = make_weights_for_balanced_classes(tr_df)
    sampler = WeightedRandomSampler(
        weights=torch.tensor(weights, dtype=torch.double),
        num_samples=max(len(tr_df), ARGS.batch_size),
        replacement=True,
    )
    
    tr_loader = DataLoader(tr_ds, batch_size=ARGS.batch_size, sampler=sampler, num_workers=0, drop_last=False)
    va_loader = DataLoader(va_ds, batch_size=max(ARGS.batch_size, 16), shuffle=False, num_workers=0, drop_last=False)
    
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=ARGS.lr,
        weight_decay=0.05,
    )
    
    steps_per_epoch = max(1, math.ceil(len(tr_loader) / max(1, ARGS.grad_accum)))
    total_steps = ARGS.local_epochs * steps_per_epoch
    sch = get_linear_schedule_with_warmup(opt, max(1, int(0.1 * total_steps)), total_steps)
    
    loss_fn = FocalLoss(alpha=class_alpha.to(DEVICE), gamma=2.5, label_smoothing=0.02)
    model.train()
    model.to(DEVICE)
    
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    ema_params = [p.detach().clone() for p in trainable_params]
    opt.zero_grad(set_to_none=True)
    
    for _ in range(ARGS.local_epochs):
        for it, batch in enumerate(tr_loader, start=1):
            text_inputs = {
                "input_ids": batch["input_ids"].to(DEVICE),
                "attention_mask": batch["attention_mask"].to(DEVICE),
            }
            img = batch.get("image", None)
            if img is not None:
                img = img.to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            with torch.amp.autocast("cuda", enabled=amp_enabled()):
                if img is not None:
                    out = model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"], image=img)
                else:
                    out = model(input_ids=text_inputs["input_ids"], attention_mask=text_inputs["attention_mask"])
                loss = loss_fn(out.logits, labels) / max(1, ARGS.grad_accum)
            
            scaler.scale(loss).backward()
            
            if it % max(1, ARGS.grad_accum) == 0:
                nn.utils.clip_grad_norm_(trainable_params, 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sch.step()
            
            ema_update(ema_params, trainable_params, ARGS.ema_decay)
    
    # Eval with EMA weights
    backup = [p.detach().clone() for p in trainable_params]
    for p, ep in zip(trainable_params, ema_params):
        p.data.copy_(ep.data)
    
    thr = calibrate_thresholds(model, va_loader, precision_target=ARGS.precision_target)
    was_quiet = ARGS.quiet_eval
    ARGS.quiet_eval = True
    mets = evaluate_with_thr(model, va_loader, thr)
    ARGS.quiet_eval = was_quiet
    
    micro_f1, macro_f1 = mets["micro_f1"], mets["macro_f1"]
    
    for p, bp in zip(trainable_params, backup):
        p.data.copy_(bp.data)
    
    try:
        # Try getting PEFT state dict
        if hasattr(model, 'text_encoder'):
            lora_sd = get_peft_model_state_dict(model.text_encoder)
        elif hasattr(model, 'base_model'):
            lora_sd = get_peft_model_state_dict(model.base_model)
        else:
            lora_sd = get_peft_model_state_dict(model)
    except Exception:
        lora_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    
    lora_sd = {k: v.detach().cpu() for k, v in lora_sd.items()}
    
    del tr_loader, va_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return micro_f1, macro_f1, lora_sd, thr, len(tr_df)

def fedavg_weighted(states: List[Dict[str, torch.Tensor]], sizes: List[int]) -> Dict[str, torch.Tensor]:
    total = float(sum(sizes))
    ws = [s / total for s in sizes]
    keys = list(states[0].keys())
    out = {}
    for k in keys:
        out[k] = torch.stack([st[k].float() * w for st, w in zip(states, ws)], dim=0).sum(0)
    return out

# --------------------- Main Training Function --------------------- 
def run_training(model_config: ModelConfig = None, comparator: ModelComparator = None):
    """
    Main training loop supporting:
    - Standard encoder models
    - Federated LLMs
    - Multimodal models
    - VLMs
    """
    if model_config is None:
        model_config = MODEL_CONFIGS[ARGS.model_type]
    
    model_name = model_config.name
    print(f"\n{'='*80}")
    print(f"Training: {model_name} ({model_config.model_type})")
    print(f"{'='*80}")
    
    start_time = time.time()
    
    print(f"Device: {DEVICE} (AMP={'on' if amp_enabled() else 'off'})")
    
    # Build tokenizer
    tok = build_tokenizer(model_config.model_name)
    
    # Build corpus
    df = build_corpus()
    multimodal = ARGS.use_images and ("image_path" in df.columns)
    
    # Split data
    clients_all = split_clients(df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    val_k = max(1, int(0.15 * len(clients_all)))
    val_df = pd.concat(clients_all[:val_k], ignore_index=True)
    train_clients = clients_all[val_k:]
    train_df = pd.concat(train_clients, ignore_index=True)
    train_df, test_df = train_test_split(train_df, test_size=0.15, random_state=SEED, shuffle=True)
    
    _, counts = make_weights_for_balanced_classes(train_df)
    inv = 1.0 / np.maximum(counts, 1)
    alpha = (inv / inv.mean()).astype(np.float32)
    alpha[1] *= 1.2  # boost nutrient_def
    alpha = torch.tensor(alpha)
    
    clients = split_clients(train_df, max(1, ARGS.clients), ARGS.dirichlet_alpha)
    
    # Build model
    if multimodal and model_config.model_type not in ["vlm"]:
        global_model = MultiModalModel(
            model_config.model_name,
            ARGS.vit_name,
            NUM_LABELS,
            freeze_text=model_config.freeze_base,
            freeze_vision=ARGS.freeze_vision,
            lora_r=model_config.lora_r,
            lora_alpha=model_config.lora_alpha,
            lora_dropout=model_config.lora_dropout,
        ).to(DEVICE)
    else:
        global_model = build_model_from_config(model_config, NUM_LABELS).to(DEVICE)
    
    # Count parameters
    params_count = sum(p.numel() for p in global_model.parameters())
    trainable_params = sum(p.numel() for p in global_model.parameters() if p.requires_grad)
    print(f"[Model] Total params: {params_count/1e6:.2f}M | Trainable: {trainable_params/1e6:.2f}M")
    
    # Validation loader
    if "image_path" in val_df.columns and ARGS.use_images:
        val_loader = DataLoader(
            MultiModalDS(val_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
            batch_size=32,
            shuffle=False,
        )
    else:
        val_loader = DataLoader(MultiLabelDS(val_df, tok, ARGS.max_len), batch_size=32, shuffle=False)
    
    metrics_dir = os.path.join(ARGS.save_dir, "metrics", model_name)
    os.makedirs(metrics_dir, exist_ok=True)
    
    def evaluate_global(thr):
        if "image_path" in test_df.columns and ARGS.use_images:
            test_loader = DataLoader(
                MultiModalDS(test_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
                batch_size=64,
                shuffle=False,
            )
        else:
            test_loader = DataLoader(MultiLabelDS(test_df, tok, ARGS.max_len), batch_size=64, shuffle=False)
        return evaluate_with_thr(global_model, test_loader, thr)
    
    thr_history = []
    
    # Federated training rounds
    for r in range(1, max(1, ARGS.rounds) + 1):
        print(f"\n==== Round {r}/{ARGS.rounds} ====")
        rng = np.random.default_rng(SEED + r)
        k_all = list(range(len(clients)))
        rng.shuffle(k_all)
        m = max(1, int(ARGS.participation * len(k_all)))
        chosen = k_all[:m]
        
        states, sizes = [], []
        for i in chosen:
            if rng.random() < ARGS.client_dropout:
                print(f"[Client {i+1}] dropped this round")
                continue
            
            cdf = clients[i]
            if len(cdf) < 80:
                print(f"[Client {i+1}] skipped (too small: n={len(cdf)})")
                continue
            
            n = len(cdf)
            val_n = max(1, int(ARGS.val_frac * n))
            va_df, tr_df = cdf.iloc[:val_n], cdf.iloc[val_n:]
            
            # Create local model
            if multimodal and model_config.model_type not in ["vlm"]:
                local = MultiModalModel(
                    model_config.model_name,
                    ARGS.vit_name,
                    NUM_LABELS,
                    freeze_text=model_config.freeze_base,
                    freeze_vision=ARGS.freeze_vision,
                    lora_r=model_config.lora_r,
                    lora_alpha=model_config.lora_alpha,
                    lora_dropout=model_config.lora_dropout,
                ).to(DEVICE)
            else:
                local = build_model_from_config(model_config, NUM_LABELS).to(DEVICE)
            
            # Load global state
            try:
                if hasattr(global_model, 'text_encoder') and hasattr(local, 'text_encoder'):
                    set_peft_model_state_dict(local.text_encoder, get_peft_model_state_dict(global_model.text_encoder))
                elif hasattr(global_model, 'base_model') and hasattr(local, 'base_model'):
                    set_peft_model_state_dict(local.base_model, get_peft_model_state_dict(global_model.base_model))
                else:
                    set_peft_model_state_dict(local, get_peft_model_state_dict(global_model))
            except Exception:
                pass
            
            rng_local = np.random.default_rng(SEED + r + i)
            local_epochs = int(rng_local.choice([2, 3], p=[0.6, 0.4]))
            orig_local = ARGS.local_epochs
            ARGS.local_epochs = local_epochs
            
            micro, macro, lora_sd, thr_local, used_n = train_local(local, tok, tr_df, va_df, class_alpha=alpha)
            
            ARGS.local_epochs = orig_local
            print(f"[Client {i+1}] micro_f1={_fmt_str(micro)} macro_f1={_fmt_str(macro)} (n={len(cdf)}) thr={np.round(thr_local,2)}")
            
            states.append(lora_sd)
            sizes.append(used_n)
            
            del local
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Aggregate
        if states:
            avg_sd = fedavg_weighted(states, sizes)
            try:
                if hasattr(global_model, 'text_encoder'):
                    set_peft_model_state_dict(global_model.text_encoder, avg_sd)
                elif hasattr(global_model, 'base_model'):
                    set_peft_model_state_dict(global_model.base_model, avg_sd)
                else:
                    set_peft_model_state_dict(global_model, avg_sd)
            except Exception:
                try:
                    global_model.load_state_dict(avg_sd, strict=False)
                except Exception:
                    print("[Warn] couldn't set averaged state")
            
            final_thr = calibrate_thresholds(global_model, val_loader, precision_target=ARGS.precision_target)
            final_thr = np.clip(final_thr + np.array([+0.03, 0.00, 0.00, +0.02, 0.00]), 0.05, 0.90)
        else:
            print("No client updates this round")
            final_thr = thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS)
        
        thr_history.append(final_thr)
        test_mets = evaluate_global(final_thr)
        
        # Save round metrics
        round_tag = f"round_{r:02d}"
        np.save(os.path.join(metrics_dir, f"{round_tag}_thr.npy"), final_thr)
        with open(os.path.join(metrics_dir, f"{round_tag}_summary.json"), "w") as f:
            json.dump({
                "round": r,
                "micro_f1": float(test_mets["micro_f1"]),
                "macro_f1": float(test_mets["macro_f1"]),
            }, f, indent=2)
    
    # Save final model
    model_save_dir = os.path.join(ARGS.save_dir, model_name)
    os.makedirs(model_save_dir, exist_ok=True)
    
    ap = os.path.join(model_save_dir, "model.pt")
    thp = os.path.join(model_save_dir, "thresholds.npy")
    
    try:
        if hasattr(global_model, 'text_encoder'):
            torch.save(get_peft_model_state_dict(global_model.text_encoder), ap)
        elif hasattr(global_model, 'base_model'):
            torch.save(get_peft_model_state_dict(global_model.base_model), ap)
        else:
            torch.save(get_peft_model_state_dict(global_model), ap)
    except Exception:
        torch.save(global_model.state_dict(), ap)
    
    np.save(thp, thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS))
    print(f"[Save] model → {ap}")
    print(f"[Save] thresholds → {thp}")
    
    # Final evaluation
    if "image_path" in test_df.columns and ARGS.use_images:
        cal_df = test_df.sample(min(400, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(
            MultiModalDS(cal_df, tok, ARGS.max_len, img_size=ARGS.img_size, image_root=ARGS.image_dir),
            batch_size=32,
            shuffle=False,
        )
    else:
        cal_df = test_df.sample(min(400, len(test_df)), random_state=SEED)
        cal_loader = DataLoader(MultiLabelDS(cal_df, tok, ARGS.max_len), batch_size=32, shuffle=False)
    
    final_thr = thr_history[-1] if thr_history else np.array([0.5] * NUM_LABELS)
    mets = evaluate_with_thr(global_model, cal_loader, final_thr)
    
    figs_dir = os.path.join(ARGS.save_dir, "figs")
    save_tables_and_plots(mets, save_dir=figs_dir, model_name=model_name)
    
    training_time = time.time() - start_time
    
    # Measure inference time
    infer_start = time.time()
    _ = evaluate_with_thr(global_model, cal_loader, final_thr)
    inference_time = time.time() - infer_start
    
    # Get memory usage
    if torch.cuda.is_available():
        memory_usage = torch.cuda.max_memory_allocated() / 1024**2  # MB
    else:
        memory_usage = 0.0
    
    # Add to comparator
    if comparator:
        result = ComparisonResult(
            model_name=model_name,
            model_type=model_config.model_type,
            micro_f1=mets["micro_f1"],
            macro_f1=mets["macro_f1"],
            per_class_f1=mets["per_class"]["f1"],
            training_time=training_time,
            inference_time=inference_time,
            params_count=params_count,
            memory_usage=memory_usage,
        )
        comparator.add_result(result)
    
    print(f"\n[{model_name}] Training complete in {training_time:.1f}s")
    print(f"[{model_name}] Final Micro-F1: {mets['micro_f1']:.4f} | Macro-F1: {mets['macro_f1']:.4f}")
    
    return mets

# --------------------- Main Execution --------------------- 
def main():
    if ARGS.inference:
        print("[Mode] Inference mode not yet implemented in comparison framework")
        return
    
    if ARGS.compare_all:
        print("\n" + "="*80)
        print("COMPREHENSIVE MODEL COMPARISON MODE")
        print("="*80)
        
        comparator = ModelComparator(os.path.join(ARGS.save_dir, "comparisons"))
        
        # Models to compare
        models_to_compare = [
            "roberta",  # Baseline encoder
            "distilbert",  # Efficient encoder
            "flan-t5-small",  # Federated LLM (seq2seq)
            "gpt2",  # Federated LLM (decoder)
        ]
        
        if ARGS.use_images:
            models_to_compare.append("vit")  # Pure vision
        
        if ARGS.use_vlm:
            models_to_compare.append("clip")  # VLM
        
        for model_key in models_to_compare:
            if model_key not in MODEL_CONFIGS:
                print(f"[Warn] Skipping unknown model: {model_key}")
                continue
            
            try:
                config = MODEL_CONFIGS[model_key]
                run_training(config, comparator)
            except Exception as e:
                print(f"[Error] Failed to train {model_key}: {e}")
                continue
        
        # Generate comparison report
        comparator.generate_comparison_report()
        
        print("\n" + "="*80)
        print("COMPARISON COMPLETE")
        print(f"Results saved to: {comparator.save_dir}")
        print("="*80)
    
    else:
        # Single model training
        config = MODEL_CONFIGS.get(ARGS.model_type)
        if config is None:
            print(f"[Error] Unknown model type: {ARGS.model_type}")
            return
        
        run_training(config)

if __name__ == "__main__":
    main()

print("\n[✓] Enhanced Farm Advisor System Loaded")
print("[✓] Features:")
print("  - Federated LLM support (Flan-T5, GPT-2)")
print("  - ViT encoder for crop stress detection")
print("  - VLM support (CLIP, BLIP)")
print("  - Comprehensive dataset loading")
print("  - Model comparison framework")
print("  - Full evaluation and benchmarking")
print("\n[✓] Ready to run training")
print(f"[✓] Use --compare_all to compare all models")
print(f"[✓] Use --use_federated_llm for LLM training")
print(f"[✓] Use --use_vlm for Vision-Language Models")
