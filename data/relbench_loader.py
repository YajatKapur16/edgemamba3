import os
import time
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# data/relbench_loader.py

# PyTorch 2.6+ defaults weights_only=True, which breaks cached dataset loading.
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Tuple, List

from relbench.datasets import get_dataset
from relbench.tasks import get_task
from relbench.modeling.graph import make_pkey_fkey_graph
from relbench.modeling.utils import get_stype_proposal
import torch_frame


# ─── Dataset registry ────────────────────────────────────────────────────────

RELBENCH_CONFIGS = {
    "rel-hm/user-churn": {
        "dataset": "rel-hm",
        "task": "user-churn",
        "entity_table": "customers",
        "entity_col": "customer_id",
        "event_table": "transactions",
        "event_entity_col": "customer_id",
        "event_time_col": "t_dat",
        "task_type": "binary_classification",
        "metric": "auroc",
        "num_outputs": 1,
    },
    "rel-amazon/user-ltv": {
        "dataset": "rel-amazon",
        "task": "user-ltv",
        "entity_table": "customer",
        "entity_col": "customer_id",
        "event_table": "review",
        "event_entity_col": "customer_id",
        "event_time_col": "review_time",
        "task_type": "regression",
        "metric": "mae",
        "num_outputs": 1,
    },
}


# ─── Dataset class ───────────────────────────────────────────────────────────

class RelBenchEventDataset(Dataset):
    """
    Per-entity event sequence dataset for RelBench.

    For each entity (customer), extracts the temporal sequence of
    associated event nodes (transactions, reviews) up to seed_time,
    sorted by timestamp.

    Key insight: In RelBench, transactions are NODES not edges.
    They have rich features (price, channel, product type, date).
    We sequence these nodes temporally for Mamba-3 input.
    """

    def __init__(
        self,
        entity_ids: torch.Tensor,       # [N_entities]
        labels: torch.Tensor,           # [N_entities]
        event_features: torch.Tensor,   # [N_events, event_feat_dim]
        event_entity_ids: torch.Tensor, # [N_events] — which entity each event belongs to
        event_timestamps: torch.Tensor, # [N_events]
        seed_time: float,
        max_seq_len: int = 256,
        min_seq_len: int = 1,
    ):
        self.entity_ids       = entity_ids
        self.labels           = labels
        self.event_features   = event_features
        self.event_entity_ids = event_entity_ids
        self.event_timestamps = event_timestamps
        self.seed_time        = seed_time
        self.max_seq_len      = max_seq_len
        self.min_seq_len      = min_seq_len
        self.event_feat_dim   = event_features.shape[1]

        # Pre-build index: entity_id → list of event indices (before seed_time)
        self._build_index()

    def _build_index(self):
        """Pre-compute which events belong to each entity, before seed_time."""
        import time as _time
        t0 = _time.time()
        n = len(self.entity_ids)
        print(f"    Building entity-event index for {n} entities (vectorized)...", flush=True)
        causal_mask = self.event_timestamps < self.seed_time
        valid_indices = causal_mask.nonzero(as_tuple=True)[0]
        
        # Filter to only valid events
        valid_entity_ids = self.event_entity_ids[valid_indices]
        valid_timestamps = self.event_timestamps[valid_indices]
        
        # Create mapping from global entity ID to the index in self.entity_ids (0 to n-1)
        # Using a tensor array for fast lookup. Assumes entity IDs are bounded reasonably.
        # Alternatively, use a dict or torch searchsorted if entity_ids are large.
        max_id = max(self.entity_ids.max().item(), self.event_entity_ids.max().item())
        global_to_local = torch.full((max_id + 1,), -1, dtype=torch.long)
        global_to_local[self.entity_ids] = torch.arange(n)
        
        # Map valid events to their local entity index
        local_entity_ids = global_to_local[valid_entity_ids]
        
        # Keep only events belonging to entities in this split
        split_mask = local_entity_ids != -1
        valid_indices = valid_indices[split_mask]
        local_entity_ids = local_entity_ids[split_mask]
        valid_timestamps = valid_timestamps[split_mask]
        
        # Sort by timestamp (oldest first)
        sort_order = valid_timestamps.argsort()
        valid_indices = valid_indices[sort_order]
        local_entity_ids = local_entity_ids[sort_order]
        
        # Stable sort by local_entity_ids to group all events for each entity together
        # while preserving the timestamp ordering we just created
        group_order = local_entity_ids.argsort(stable=True)
        valid_indices = valid_indices[group_order]
        local_entity_ids = local_entity_ids[group_order]
        
        # Compute lengths and start indices for each local entity
        self.counts = torch.bincount(local_entity_ids, minlength=n)
        self.starts = torch.cat([torch.tensor([0]), self.counts.cumsum(0)[:-1]])
        self.valid_indices = valid_indices
        
        print(f"      done in {_time.time()-t0:.2f}s", flush=True)

    def __len__(self):
        return len(self.entity_ids)

    def __getitem__(self, idx):
        label  = self.labels[idx]
        count  = self.counts[idx].item()

        if count == 0:
            # No events before seed_time — return zero sequence
            seq   = torch.zeros(self.min_seq_len, self.event_feat_dim)
            dt    = torch.zeros(self.min_seq_len)
            mask  = torch.zeros(self.min_seq_len, dtype=torch.bool)
            return seq, dt, mask, label
        
        start = self.starts[idx].item()
        ev_idx = self.valid_indices[start : start + count]

        # Truncate to max_seq_len (keep most recent events)
        if len(ev_idx) > self.max_seq_len:
            ev_idx = ev_idx[-self.max_seq_len:]

        seq = self.event_features[ev_idx]  # [L, D]
        ts  = self.event_timestamps[ev_idx]

        # Delta-t: time gaps between consecutive events
        dt = torch.zeros(len(ts))
        if len(ts) > 1:
            dt[1:] = (ts[1:] - ts[:-1]).float().clamp(min=0)

        # Padding mask (True = valid, False = padding)
        mask = torch.ones(len(seq), dtype=torch.bool)

        return seq, dt, mask, label


def collate_relbench(batch):
    """
    Pads sequences to the nearest length multiple of 64 within a batch.
    """
    seqs, dts, masks, labels = zip(*batch)
    max_len = max(s.shape[0] for s in seqs)
    
    # Bucket lengths to nearest multiple of 64 to avoid Triton recompilation storms
    bucketed_len = ((max_len + 63) // 64) * 64

    padded_seqs  = torch.zeros(len(seqs), bucketed_len, seqs[0].shape[-1])
    padded_dts   = torch.zeros(len(seqs), bucketed_len)
    padded_masks = torch.zeros(len(seqs), bucketed_len, dtype=torch.bool)

    for i, (seq, dt, mask) in enumerate(zip(seqs, dts, masks)):
        L = seq.shape[0]
        padded_seqs[i, :L]  = seq
        padded_dts[i, :L]   = dt
        padded_masks[i, :L] = mask

    return padded_seqs, padded_dts, padded_masks, torch.stack(labels)


def load_relbench(
    config_key: str,
    batch_size: int = 64,
    max_seq_len: int = 256,
    num_workers: int = 4,
    **kwargs
):
    """
    Load RelBench dataset & task, and construct train/val/test splits.
    
    Returns:
        train_loader, val_loader, test_loader, meta_dict
    """
    
    if config_key not in RELBENCH_CONFIGS:
        raise ValueError(f"Unknown config: {config_key}. Choose from {list(RELBENCH_CONFIGS.keys())}")
        
    cfg = RELBENCH_CONFIGS[config_key]
    
    # ── Check for Cached Processed Data ───────────────────────────────────────
    base_cache_dir = os.environ.get("EDGEMAMBA_DATA_CACHE", "./data/cache")
    processed_cache_dir = os.path.join(base_cache_dir, "relbench_processed")
    os.makedirs(processed_cache_dir, exist_ok=True)
    safe_name = config_key.replace("/", "_")
    cache_path = os.path.join(processed_cache_dir, f"{safe_name}.pt")
    
    if os.path.exists(cache_path):
        print(f"[RelBench Loader] Loading precomputed dataset from {cache_path}...", flush=True)
        t0 = time.time()
        cached_data = torch.load(cache_path, weights_only=False)
        splits = cached_data["splits"]
        meta = cached_data["meta"]
        print(f"  Loaded in {time.time()-t0:.2f}s", flush=True)
        
        return _build_loaders_from_splits(splits, meta, batch_size, num_workers, True)

    print(f"[RelBench Loader] No cache found. Processing {config_key} from scratch...", flush=True)
    max_seq_len = kwargs.get("max_seq_len", 256) # Extract max_seq_len from kwargs

    # ── Load RelBench dataset ─────────────────────────────────────────────────
    print(f"Loading RelBench: {config_key}...")
    dataset = get_dataset(cfg["dataset"], download=True) # Original cache_dir parameter removed, assuming get_dataset handles its own cache
    task    = get_task(cfg["dataset"], cfg["task"], download=True)

    db = dataset.get_db()
    print(f"  Database loaded. Tables: {list(db.table_dict.keys())}", flush=True)

    # ── Extract event table (transactions / reviews) ──────────────────────────
    print(f"  Extracting event table '{cfg['event_table']}'...", flush=True)
    event_df = db.table_dict[cfg["event_table"]].df
    print(f"  Event table: {len(event_df)} rows, columns: {event_df.columns.tolist()}", flush=True)

    # Encode event features using torch_frame
    # In RelBench, event table rows are nodes — each row is an event
    col_to_stype = get_stype_proposal(db)
    event_cols = [c for c in event_df.columns
                  if c not in [cfg["event_entity_col"], cfg["event_time_col"]]]
    print(f"  Encoding {len(event_cols)} feature columns: {event_cols[:5]}{'...' if len(event_cols) > 5 else ''}", flush=True)

    # Simple numeric encoding for now (torch_frame handles categoricals)
    event_features   = _encode_features(event_df, event_cols)
    event_entity_ids = torch.tensor(event_df[cfg["event_entity_col"]].values,
                                    dtype=torch.long)
    event_timestamps = torch.tensor(
        pd.to_datetime(event_df[cfg["event_time_col"]]).astype(int).values / 1e9,
        dtype=torch.float
    )
    event_feat_dim = event_features.shape[1]
    print(f"  Features encoded: shape={event_features.shape}, feat_dim={event_feat_dim}", flush=True)

    # ── Build train/val/test splits ───────────────────────────────────────────
    splits = {}
    for split_name in ["train", "val", "test"]:
        print(f"  Building {split_name} split...", flush=True)
        split_table = task.get_table(split_name)
        entity_ids  = torch.tensor(split_table.df[cfg["entity_col"]].values,
                                   dtype=torch.long)
        if task.target_col in split_table.df.columns:
            labels = torch.tensor(split_table.df[task.target_col].values, dtype=torch.float)
        else:
            labels = torch.full((len(entity_ids),), float("nan"), dtype=torch.float)
        seed_time   = float(pd.Timestamp(split_table.max_timestamp).value / 1e9)
        print(f"    {split_name}: {len(entity_ids)} entities, seed_time={seed_time:.0f}", flush=True)

        dataset_obj = RelBenchEventDataset(
            entity_ids=entity_ids,
            labels=labels,
            event_features=event_features,
            event_entity_ids=event_entity_ids,
            event_timestamps=event_timestamps,
            seed_time=seed_time,
            max_seq_len=max_seq_len,
        )
        splits[split_name] = dataset_obj

    meta = {
        "event_feat_dim": event_feat_dim,
        "task_type": task.task_type.value,
        "metric": task.eval_metric.value if hasattr(task, "eval_metric") else "auroc",
        "num_outputs": 1  # Standard for many RelBench node classification / regression tasks
    }

    # ── Save Processed Data to Cache ──────────────────────────────────────────
    print(f"  Saving processed dataset to cache: {cache_path}...", flush=True)
    t0 = time.time()
    torch.save({"splits": splits, "meta": meta}, cache_path)
    print(f"  Saved in {time.time()-t0:.2f}s", flush=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    actual_workers = 0 if os.name == 'nt' else num_workers
    persistent = True if actual_workers > 0 else False
    
    train_loader, val_loader, test_loader, metadata = _build_loaders_from_splits(
        splits, meta, batch_size, actual_workers, persistent
    )

    print(f"  Train: {metadata['train_size']} | Val: {metadata['val_size']} "
          f"| Test: {metadata['test_size']}")
    print(f"  event_feat_dim={event_feat_dim}, max_seq_len={max_seq_len}")

    return train_loader, val_loader, test_loader, metadata


def _build_loaders_from_splits(splits, meta, batch_size, actual_workers, persistent):
    """Helper to build loaders with DDP samplers if initialized."""
    
    is_dist = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(splits["train"], shuffle=True) if is_dist else None
    val_sampler   = DistributedSampler(splits["val"], shuffle=False) if is_dist else None
    test_sampler  = DistributedSampler(splits["test"], shuffle=False) if is_dist else None

    # DataLoader wrapper
    def create_loader(split_data, bs, shuffle, drop_last, sampler):
        return DataLoader(
            split_data, 
            batch_size=bs,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            collate_fn=collate_relbench,
            num_workers=actual_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=persistent
        )

    train_loader = create_loader(splits["train"], batch_size, True, True, train_sampler)
    val_loader   = create_loader(splits["val"], batch_size, False, False, val_sampler)
    test_loader  = create_loader(splits["test"], batch_size, False, False, test_sampler)

    metadata = {
        "event_feat_dim": meta["event_feat_dim"],
        "num_outputs": meta["num_outputs"],
        "task_type": meta["task_type"],
        "metric": meta["metric"],
        "train_size": len(splits["train"]),
        "val_size": len(splits["val"]),
        "test_size": len(splits["test"]),
    }
    
    return train_loader, val_loader, test_loader, metadata


def _encode_features(df: pd.DataFrame, cols: list) -> torch.Tensor:
    """
    Simple feature encoding: numeric cols pass through,
    categorical cols are label-encoded.
    Fills NaN with 0.
    """
    encoded = []
    for col in cols:
        series = df[col]
        if series.dtype in [np.float64, np.float32, np.int64, np.int32]:
            encoded.append(series.fillna(0).values.astype(np.float32))
        else:
            # Label encode
            codes, _ = pd.factorize(series.fillna("__missing__"))
            encoded.append(codes.astype(np.float32))
    if not encoded:
        return torch.zeros(len(df), 1)
    return torch.tensor(np.stack(encoded, axis=1), dtype=torch.float)
