# train_fed_multimodal.py
"""
Federated training driver for multimodal model.

Usage:
  python train_fed_multimodal.py --num_clients 4 --rounds 2 --local_epochs 1 --max_text 1000 --max_images 1000

Auto-detects GPU if available. Saves adapters and fusion head.
"""

import os, argparse, time, math, random
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

from multimodal_model import MultimodalClassifier, get_text_adapter_state_dict, set_text_adapter_state_dict, get_image_head_state_dict, set_image_head_state_dict, NUM_LABELS, ISSUE_LABELS
from datasets_loader import build_unified_dataset
from federated_core import split_clients_list, fedavg_state_dicts

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleMultiDataset(Dataset):
    def __init__(self, items, tokenizer, image_processor, max_len=160):
        self.items = items
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        text = it.get("text","") or ""
        labs = it.get("labels", []) or []
        label_vec = np.zeros(len(ISSUE_LABELS), dtype=np.float32)
        for l in labs: label_vec[l]=1.0
        sample = {"labels": torch.tensor(label_vec, dtype=torch.float32)}
        # tokenize if text exists
        if text.strip():
            enc = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_len, return_tensors="pt")
            sample["input_ids"] = enc["input_ids"].squeeze(0)
            sample["attention_mask"] = enc["attention_mask"].squeeze(0)
        else:
            sample["input_ids"] = torch.zeros((self.max_len,), dtype=torch.long)
            sample["attention_mask"] = torch.zeros((self.max_len,), dtype=torch.long)
        # image
        pv = None
        if it.get("image"):
            from PIL import Image
            img = Image.open(it["image"]).convert("RGB")
            proc = self.image_processor(images=img, return_tensors="pt")
            pv = proc["pixel_values"].squeeze(0)
        else:
            pv = torch.zeros((3,224,224), dtype=torch.float32)
        sample["pixel_values"] = pv
        return sample

def train_local_model(local_model, dataset_items, device, epochs=1, batch_size=8, lr=1e-4):
    tokenizer = local_model.tokenizer
    image_processor = local_model.image_processor
    ds = SimpleMultiDataset(dataset_items, tokenizer, image_processor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    # Only train adapter params + image_proj + fusion head (others frozen)
    trainable = [p for p in local_model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    local_model.train(); local_model.to(device)
    for ep in range(epochs):
        for b in loader:
            # move to device
            inputs = {}
            for k,v in b.items():
                if k in ("labels",):
                    inputs[k] = v.to(device)
                else:
                    inputs[k] = v.to(device)
            opt.zero_grad()
            logits = local_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"])
            loss = loss_fn(logits, inputs["labels"])
            loss.backward()
            opt.step()
    # return adapter states and image head states and fusion head
    text_state = get_text_adapter_state_dict(local_model)
    image_head_state = get_image_head_state_dict(local_model)
    fusion_state = local_model.fusion.state_dict()
    return text_state, image_head_state, fusion_state, len(dataset_items)

def evaluate_model(model, items, device, max_eval=256):
    # quick evaluation: compute micro f1 over small set
    from sklearn.metrics import f1_score
    tokenizer = model.tokenizer
    image_processor = model.image_processor
    ds = SimpleMultiDataset(items[:max_eval], tokenizer, image_processor)
    loader = DataLoader(ds, batch_size=16, shuffle=False)
    model.eval(); model.to(device)
    preds, trues = [], []
    with torch.no_grad():
        for b in loader:
            logits = model(input_ids=b["input_ids"].to(device),
                           attention_mask=b["attention_mask"].to(device),
                           pixel_values=b["pixel_values"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs >= 0.5).astype(int))
            trues.append(b["labels"].numpy().astype(int))
    if len(preds)==0: return 0.0
    import numpy as np
    P = np.vstack(preds); T = np.vstack(trues)
    try:
        return float(f1_score(T, P, average="micro", zero_division=0))
    except Exception:
        return 0.0

def main(args):
    device = get_device()
    print(f"[train] device: {device}")
    # build dataset
    unified = build_unified_dataset(max_per_text=args.max_text, max_images=args.max_images, offline=False)
    print(f"[train] unified items: {len(unified)}")
    # split to clients
    clients = split_clients_list(unified, num_clients=args.num_clients, alpha=args.alpha, seed=SEED)
    print(f"[train] simulated {len(clients)} clients with sizes {[len(c) for c in clients]}")
    # initialize global model
    global_model = MultimodalClassifier(use_lora=True, freeze_backbones=True)
    # ensure adapters present: text backbone is peft-wrapped
    # save initial global adapters (may be random-initialized)
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    # federated rounds
    global_text_state = get_text_adapter_state_dict(global_model)
    global_image_state = get_image_head_state_dict(global_model)
    global_fusion_state = global_model.fusion.state_dict()
    for r in range(1, args.rounds+1):
        print(f"\n=== Federated round {r}/{args.rounds} ===")
        states_text, states_image, states_fusion = [], [], []
        sizes = []
        # sample participating clients
        idxs = list(range(len(clients)))
        random.shuffle(idxs)
        m = max(1, int(args.participation * len(idxs)))
        chosen = idxs[:m]
        for ci in chosen:
            shard = clients[ci]
            if len(shard) < 1:
                print(f"[client {ci}] empty, skipping")
                continue
            # create local model and load global adapters/heads
            local_model = MultimodalClassifier(use_lora=True, freeze_backbones=True)
            # set global adapters
            set_text_adapter_state_dict(local_model, global_text_state)
            set_image_head_state_dict(local_model, global_image_state)
            local_model.fusion.load_state_dict(global_fusion_state)
            # unfreeze adapters and image_proj/fusion for training
            for n, p in local_model.named_parameters():
                # unfreeze peft adapter params (they are typically under 'lora' or 'peft' names)
                if "lora" in n or "image_proj" in n or "fusion" in n or "fusion" in n:
                    p.requires_grad = True
                else:
                    p.requires_grad = False
            # local epochs jitter
            local_epochs = args.local_epochs
            # train local
            text_state, image_state, fusion_state, used_n = train_local_model(local_model, shard, device,
                                                                             epochs=local_epochs,
                                                                             batch_size=args.batch_size,
                                                                             lr=args.lr)
            states_text.append(text_state)
            states_image.append(image_state)
            states_fusion.append(fusion_state)
            sizes.append(used_n)
            print(f"[client {ci}] trained on {used_n} samples")
        # aggregate
        if len(states_text) > 0:
            # fedavg images and fusion: we convert dicts to tensors
            # convert image/fusion states into torch tensors
            # for text (peft) some keys may be absent from some states; assume same keys
            # Fedavg for text adapters
            # Build stacked state for each key
            # NOTE: states_text is list of dict(str->tensor)
            # we implement simple weighted average
            # text
            agg_text = {}
            keys = states_text[0].keys()
            import torch
            total = float(sum(sizes))
            for k in keys:
                agg = None
                for sd, sz in zip(states_text, sizes):
                    v = sd[k].float() * (sz / total)
                    if agg is None: agg = v.clone()
                    else: agg += v
                agg_text[k] = agg
            global_text_state = agg_text
            # images
            agg_img = {}
            keys_img = states_image[0].keys()
            for k in keys_img:
                agg = None
                for sd, sz in zip(states_image, sizes):
                    v = sd[k].float() * (sz / total)
                    if agg is None: agg = v.clone()
                    else: agg += v
                agg_img[k] = agg
            global_image_state = agg_img
            # fusion
            agg_fus = {}
            for k in states_fusion[0].keys():
                agg = None
                for sd, sz in zip(states_fusion, sizes):
                    v = sd[k].float() * (sz / total)
                    if agg is None: agg = v.clone()
                    else: agg += v
                agg_fus[k] = agg
            global_fusion_state = agg_fus
            # set back to global model
            set_text_adapter_state_dict(global_model, global_text_state)
            set_image_head_state_dict(global_model, global_image_state)
            global_model.fusion.load_state_dict(global_fusion_state)
            # quick eval on a random subset
            f1 = evaluate_model(global_model, unified, device, max_eval=256)
            print(f"[round {r}] eval micro-F1 (quick) = {f1:.4f}")
            # save intermediate
            torch.save({"text_adapter": {k: v.cpu() for k,v in global_text_state.items()},
                        "image_head": {k: v.cpu() for k,v in global_image_state.items()},
                        "fusion": {k: v.cpu() for k,v in global_fusion_state.items()}},
                       os.path.join(save_dir, f"round_{r}_checkpoint.pt"))
        else:
            print("[round] no client updates")
    # final save
    torch.save({"text_adapter": {k:v.cpu() for k,v in global_text_state.items()},
                "image_head": {k:v.cpu() for k,v in global_image_state.items()},
                "fusion": {k:v.cpu() for k,v in global_fusion_state.items()}},
               os.path.join(save_dir, "global_multimodal.pt"))
    print("[train] done. saved to", os.path.join(save_dir, "global_multimodal.pt"))

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_clients", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--max_text", type=int, default=1000)
    ap.add_argument("--max_images", type=int, default=1000)
    ap.add_argument("--save_dir", type=str, default="checkpoints")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--participation", type=float, default=0.8)
    args = ap.parse_args()
    main(args)
