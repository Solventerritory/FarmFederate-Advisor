# federated_core.py
"""
Federated utilities:
 - split_clients(df_list, num_clients, alpha)
 - local training (text+image adapters/heads)
 - fedavg for adapter dicts
"""

import math, random
import torch
import numpy as np
from typing import List, Dict

def split_clients_list(items: List[Dict], num_clients:int=4, alpha: float = 0.5, seed=123):
    """
    items: list of dicts with keys: 'text','image','labels'
    returns list of client shards (list of lists)
    Dirichlet split using primary label if exists; else random.
    """
    random.seed(seed)
    # assign primary label
    prim = []
    for it in items:
        labs = it.get("labels", [])
        if labs:
            prim.append(random.choice(labs))
        else:
            prim.append(random.randrange(0,5))
    C = 5
    # class->indices
    cls_idxs = {i:[] for i in range(C)}
    for idx, p in enumerate(prim):
        cls_idxs[p].append(idx)
    # draw client proportions per class
    client_bins = [[] for _ in range(num_clients)]
    for c in range(C):
        probs = np.random.dirichlet([alpha]*num_clients)
        idxs = cls_idxs[c]
        if len(idxs)==0: continue
        # distribute indices proportionally
        counts = (probs * len(idxs)).astype(int)
        # fix counts to sum
        while counts.sum() < len(idxs):
            counts[np.argmin(counts)] += 1
        start = 0
        for j in range(num_clients):
            take = counts[j]
            if take > 0:
                sel = idxs[start:start+take]
                client_bins[j].extend(sel)
                start += take
    # create client lists
    clients = []
    for b in client_bins:
        if not b:
            clients.append([])
        else:
            clients.append([items[i] for i in b])
    return clients

def fedavg_state_dicts(state_dicts: List[Dict[str, torch.Tensor]], sizes: List[int]):
    # state_dicts share same keys
    total = float(sum(sizes))
    weights = [s/total for s in sizes]
    out = {}
    keys = state_dicts[0].keys()
    for k in keys:
        stacked = torch.stack([sd[k].float() * w for sd,w in zip(state_dicts, weights)], dim=0)
        out[k] = stacked.sum(dim=0)
    return out
