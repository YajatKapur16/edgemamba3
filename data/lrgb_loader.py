# data/lrgb_loader.py

import os
import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# PyTorch 2.6+ defaults weights_only=True, which breaks PyG dataset loading.
# Patch to restore previous behavior.
_original_torch_load = torch.load
def _safe_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)
torch.load = _safe_torch_load
from torch_geometric.datasets import LRGBDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch_geometric.transforms as T


LRGB_CONFIGS = {
    "Peptides-func": {
        "task_type": "classification",
        "metric": "ap",
        "num_outputs": 10,
    },
    "Peptides-struct": {
        "task_type": "regression",
        "metric": "mae",
        "num_outputs": 11,
    },
}


def _fix_missing_edge_attr(data: Data) -> Data:
    """
    Some LRGB graphs have None edge_attr.
    Line graph transform crashes without it.
    Fill with ones as a safe default.
    """
    if data.edge_attr is None:
        data.edge_attr = torch.ones(data.num_edges, 1)
    return data


def load_lrgb(
    dataset_name: str,
    root: str = None,
    batch_size: int = 32,
    num_workers: int = 4,
):
    if root is None:
        root = os.environ.get("EDGEMAMBA_DATA_CACHE", "./data/cache/lrgb")
    """
    Load LRGB dataset and return DataLoaders.

    Args:
        dataset_name: "Peptides-func" or "Peptides-struct"
        root: cache directory for downloaded data
        batch_size: training batch size
        num_workers: DataLoader workers

    Returns:
        train_loader, val_loader, test_loader, metadata dict
    """
    assert dataset_name in LRGB_CONFIGS, \
        f"Unknown dataset: {dataset_name}. Choose from {list(LRGB_CONFIGS.keys())}"

    cfg = LRGB_CONFIGS[dataset_name]

    # Load splits with transform to lazily fix missing edge attributes during __getitem__
    # This avoids copying the entire dataset into memory as a list, which causes OOM stalls
    train_ds = LRGBDataset(root=root, name=dataset_name, split="train", transform=_fix_missing_edge_attr)
    val_ds   = LRGBDataset(root=root, name=dataset_name, split="val", transform=_fix_missing_edge_attr)
    test_ds  = LRGBDataset(root=root, name=dataset_name, split="test", transform=_fix_missing_edge_attr)

    # Infer dims from first sample
    sample = train_ds[0]
    node_dim = sample.x.shape[1]           # atom features
    edge_dim = sample.edge_attr.shape[1]   # bond features

    actual_workers = 0 if os.name == 'nt' else num_workers
    persistent = True if actual_workers > 0 else False

    # DDP DistributedSampler support
    is_dist = dist.is_available() and dist.is_initialized()
    train_sampler = DistributedSampler(train_ds, shuffle=True) if is_dist else None
    val_sampler   = DistributedSampler(val_ds, shuffle=False) if is_dist else None
    test_sampler  = DistributedSampler(test_ds, shuffle=False) if is_dist else None

    # DataLoader wrapper
    def create_loader(ds, bs, shuffle, drop_last, sampler):
        return DataLoader(
            ds,
            batch_size=bs,
            shuffle=(shuffle if sampler is None else False),
            sampler=sampler,
            num_workers=actual_workers,
            pin_memory=True,
            drop_last=drop_last,
            persistent_workers=persistent
        )

    train_loader = create_loader(train_ds, batch_size, True, True, train_sampler)
    val_loader   = create_loader(val_ds, batch_size, False, False, val_sampler)
    test_loader  = create_loader(test_ds, batch_size, False, False, test_sampler)

    metadata = {
        "node_dim": node_dim,
        "edge_dim": edge_dim,
        "num_outputs": cfg["num_outputs"],
        "task_type": cfg["task_type"],
        "metric": cfg["metric"],
        "train_size": len(train_ds),
        "val_size": len(val_ds),
        "test_size": len(test_ds),
    }

    print(f"Loaded {dataset_name}:")
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    print(f"  node_dim={node_dim}, edge_dim={edge_dim}")
    print(f"  Sample avg nodes={sample.num_nodes}, avg edges={sample.num_edges}")

    return train_loader, val_loader, test_loader, metadata


def compute_pos_weight(dataset) -> torch.Tensor:
    """
    Compute pos_weight for BCEWithLogitsLoss from label frequencies.
    Returns: [num_classes] tensor of (num_neg / num_pos) per class.
    """
    labels = torch.stack([d.y for d in dataset], dim=0)  # [N, C]
    pos_count = labels.sum(dim=0).float().clamp(min=1)
    neg_count = (labels.shape[0] - pos_count).float().clamp(min=1)
    pw = neg_count / pos_count
    # Clamp to avoid extreme weights
    pw = pw.clamp(max=10.0)
    return pw
