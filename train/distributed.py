# train/distributed.py
"""
Multi-GPU DDP launcher for EdgeMamba-3.

Provides `run_experiment()` which auto-detects GPUs and launches
DDP training via mp.spawn when multiple GPUs are available.
Falls back to single-GPU training otherwise.

Usage (notebook or script):
    from train.distributed import run_experiment
    results = run_experiment(config=cfg, domain="relbench", seeds=range(5))
"""

import os
import json
import time
import random
import socket
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np


def _find_free_port():
    """Find a free TCP port for DDP master."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_model(config, domain, meta):
    from models.edgemamba3 import EdgeMamba3

    if domain == "lrgb":
        return EdgeMamba3(
            domain="lrgb",
            node_in_dim=config["node_in_dim"],
            edge_in_dim=config["edge_in_dim"],
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            d_state=config["d_state"],
            mimo_rank=config["mimo_rank"],
            num_outputs=config["num_outputs"],
            task_type=config["task_type"],
            dropout=config["dropout"],
            drop_path=config.get("drop_path", 0.0),
            use_virtual_node=config.get("use_virtual_node", False),
            gradient_checkpointing=config.get("gradient_checkpointing", False),
            label_smoothing=config.get("label_smoothing", 0.0),
        )
    else:
        return EdgeMamba3(
            domain="relbench",
            event_feat_dim=meta["event_feat_dim"],
            d_model=config["d_model"],
            n_layers=config["n_layers"],
            d_state=config["d_state"],
            mimo_rank=config["mimo_rank"],
            num_outputs=config["num_outputs"],
            task_type=config["task_type"],
            dropout=config["dropout"],
            gradient_checkpointing=config.get("gradient_checkpointing", False),
        )


def _load_data(config, domain):
    if domain == "lrgb":
        from data.lrgb_loader import load_lrgb
        return load_lrgb(
            config["dataset"], batch_size=config["batch_size"]
        )
    else:
        from data.relbench_loader import load_relbench
        return load_relbench(
            config["config_key"],
            batch_size=config["batch_size"],
            max_seq_len=config.get("max_seq_len", 256),
        )


def _train_worker(rank, world_size, port, config, domain, seed,
                   save_path, report_path, result_path,
                   warmup_done_flag, pos_weight_list):
    """
    DDP worker launched by mp.spawn. Each worker trains on one GPU.

    Args:
        rank: GPU index (0 to world_size-1)
        world_size: total number of GPUs
        port: TCP port for DDP rendezvous
        config: training config dict
        domain: "lrgb" or "relbench"
        seed: random seed for this run
        save_path: where to save best checkpoint (rank 0 only)
        report_path: where to save test report (rank 0 only)
        result_path: where to write result JSON (rank 0 only)
        warmup_done_flag: shared Value to track cache warmup
        pos_weight_list: shared list for pos_weight (LRGB only)
    """
    # Initialize process group
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    _set_seed(seed)

    # Load data (each worker loads independently; DistributedSampler splits it)
    train_loader, val_loader, test_loader, meta = _load_data(config, domain)

    # LRGB: Pre-cache line graphs — ALL ranks must warm up their own process-local cache
    if domain == "lrgb":
        from models.line_graph import warmup_cache
        if rank == 0:
            warmup_cache(train_loader.dataset, desc="Pre-caching train line graphs")
            warmup_cache(val_loader.dataset, desc="Pre-caching val line graphs")
            warmup_cache(test_loader.dataset, desc="Pre-caching test line graphs")
        dist.barrier()
        # Rank 1+ caches after rank 0 finishes (avoids duplicate tqdm output)
        if rank != 0:
            warmup_cache(train_loader.dataset, desc=f"[rank{rank}] Caching train")
            warmup_cache(val_loader.dataset, desc=f"[rank{rank}] Caching val")
            warmup_cache(test_loader.dataset, desc=f"[rank{rank}] Caching test")
        dist.barrier()

    # Build model
    model = _build_model(config, domain, meta)

    # LRGB: apply pos_weight if configured
    if domain == "lrgb" and config.get("use_pos_weight", False):
        if rank == 0 and len(pos_weight_list) == 0:
            from data.lrgb_loader import compute_pos_weight
            pw = compute_pos_weight(train_loader.dataset)
            pos_weight_list.extend(pw.tolist())
        dist.barrier()
        if len(pos_weight_list) > 0:
            model.head.set_pos_weight(torch.tensor(pos_weight_list))

    param_count = sum(p.numel() for p in model.parameters())
    if rank == 0:
        print(f"   Parameters: {param_count:,} | GPUs: {world_size}")

    from train.trainer import Trainer
    trainer = Trainer(model, config, device=device,
                      run_name=f"{domain}_seed{seed}")

    t0 = time.time()
    best_val = trainer.fit(
        train_loader, val_loader,
        domain=domain, metric=config["metric"],
        save_path=save_path,
    )
    train_time = time.time() - t0

    # All ranks must call test() because evaluate uses dist.all_gather (collective op).
    # Only rank 0 writes the result file.
    test_score = trainer.test(
        test_loader, domain=domain,
        metric=config["metric"], checkpoint_path=save_path,
        report_path=report_path if rank == 0 else None,
        task_type=config.get("task_type"),
    )

    if rank == 0:
        result = {
            "seed": seed, "val": best_val, "test": test_score,
            "time_min": train_time / 60, "params": param_count,
        }
        with open(result_path, "w") as f:
            json.dump(result, f)

    dist.destroy_process_group()


def _train_single(config, domain, seed, save_path, report_path, result_path, device):
    """Single-GPU training for one seed."""
    _set_seed(seed)

    train_loader, val_loader, test_loader, meta = _load_data(config, domain)

    # LRGB: pre-cache
    if domain == "lrgb":
        from models.line_graph import warmup_cache
        warmup_cache(train_loader.dataset, desc="Pre-caching train line graphs")
        warmup_cache(val_loader.dataset, desc="Pre-caching val line graphs")
        warmup_cache(test_loader.dataset, desc="Pre-caching test line graphs")

    model = _build_model(config, domain, meta)

    if domain == "lrgb" and config.get("use_pos_weight", False):
        from data.lrgb_loader import compute_pos_weight
        pw = compute_pos_weight(train_loader.dataset)
        model.head.set_pos_weight(pw)
        print(f"   pos_weight: {pw.tolist()}")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,} | GPU: 1")

    from train.trainer import Trainer
    trainer = Trainer(model, config, device=device,
                      run_name=f"{domain}_seed{seed}")

    t0 = time.time()
    best_val = trainer.fit(
        train_loader, val_loader,
        domain=domain, metric=config["metric"],
        save_path=save_path,
    )
    train_time = time.time() - t0

    test_score = trainer.test(
        test_loader, domain=domain,
        metric=config["metric"], checkpoint_path=save_path,
        report_path=report_path, task_type=config.get("task_type"),
    )

    result = {
        "seed": seed, "val": best_val, "test": test_score,
        "time_min": train_time / 60, "params": param_count,
    }
    with open(result_path, "w") as f:
        json.dump(result, f)

    return result


def run_experiment(
    config: dict,
    domain: str,
    seeds=range(5),
    checkpoint_dir: str = "checkpoints",
    results_dir: str = "results",
    force_single_gpu: bool = False,
):
    """
    Launch training across all available GPUs for multiple seeds.

    Uses DDP (mp.spawn) when >1 GPU is available, else single-GPU.
    Returns list of result dicts: [{"seed", "val", "test", "time_min", "params"}, ...]

    Args:
        config: training config dict (loaded from YAML)
        domain: "lrgb" or "relbench"
        seeds: iterable of seed values
        checkpoint_dir: directory for model checkpoints
        results_dir: directory for metric reports
        force_single_gpu: disable multi-GPU even when available
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    world_size = torch.cuda.device_count() if not force_single_gpu else 1
    use_ddp = world_size > 1

    # Determine run name prefix
    if domain == "lrgb":
        run_prefix = config["dataset"].lower().replace("-", "_")
    else:
        run_prefix = config["config_key"].replace("/", "_").replace("-", "_")

    if use_ddp:
        print(f"\n🚀 Launching DDP training on {world_size} GPUs")
    else:
        print(f"\n🚀 Training on single GPU")

    all_results = []
    port = _find_free_port()

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Seed {seed}/{max(seeds)}")
        print(f"{'='*60}")

        save_path = os.path.join(checkpoint_dir, f"edgemamba3_{run_prefix}_seed{seed}.pt")
        report_path = os.path.join(results_dir, f"{run_prefix}_seed{seed}_report.txt")
        result_path = os.path.join(results_dir, f"_result_{run_prefix}_seed{seed}.json")

        if use_ddp:
            # Shared state for cross-process coordination
            ctx = mp.get_context("spawn")
            warmup_done = ctx.Value("i", 0)
            pos_weight_list = ctx.Manager().list()

            mp.spawn(
                _train_worker,
                args=(world_size, port, config, domain, seed,
                      save_path, report_path, result_path,
                      warmup_done, pos_weight_list),
                nprocs=world_size,
                join=True,
            )
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _train_single(config, domain, seed, save_path, report_path, result_path, device)

        # Read result written by rank 0
        with open(result_path) as f:
            result = json.load(f)
        all_results.append(result)

        print(f"   Seed {seed}: Val={result['val']:.4f}, Test={result['test']:.4f}, "
              f"Time={result['time_min']:.1f}min")

        # Clean up temp result file
        os.remove(result_path)

    return all_results
