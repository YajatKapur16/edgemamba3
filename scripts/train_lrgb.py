# scripts/train_lrgb.py

import argparse
import yaml
import torch
import torch.distributed as dist
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lrgb_loader import load_lrgb
from models.edgemamba3 import EdgeMamba3
from train.trainer import Trainer


def main(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1:
        dist.init_process_group("nccl")
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        if dist.get_rank() == 0:
            os.makedirs("checkpoints", exist_ok=True)
        dist.barrier()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs("checkpoints", exist_ok=True)

    # Run 5 seeds for statistical significance
    all_test_scores = []
    for seed in range(5):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        train_loader, val_loader, test_loader, meta = load_lrgb(
            cfg["dataset"], batch_size=cfg["batch_size"]
        )

        model = EdgeMamba3(
            domain="lrgb",
            node_in_dim=cfg["node_in_dim"],
            edge_in_dim=cfg["edge_in_dim"],
            d_model=cfg["d_model"],
            n_layers=cfg["n_layers"],
            d_state=cfg["d_state"],
            mimo_rank=cfg["mimo_rank"],
            num_outputs=cfg["num_outputs"],
            task_type=cfg["task_type"],
            dropout=cfg["dropout"],
            use_virtual_node=cfg.get("use_virtual_node", False),
            gradient_checkpointing=cfg.get("gradient_checkpointing", False),
        )

        save_path = cfg["save_path"].replace(".pt", f"_seed{seed}.pt")
        trainer   = Trainer(model, cfg, device=device,
                            run_name=f"{cfg['dataset']}_seed{seed}")

        best_val = trainer.fit(
            train_loader, val_loader,
            domain="lrgb", metric=cfg["metric"],
            save_path=save_path,
        )

        test_score = trainer.test(
            test_loader, domain="lrgb",
            metric=cfg["metric"], checkpoint_path=save_path,
        )
        all_test_scores.append(test_score)
        if local_rank in [-1, 0]:
            print(f"Seed {seed}: Val={best_val:.4f}, Test={test_score:.4f}")

    if local_rank in [-1, 0]:
        mean = sum(all_test_scores) / len(all_test_scores)
        std  = (sum((s-mean)**2 for s in all_test_scores) / len(all_test_scores)) ** 0.5

        print(f"\n{'='*50}")
        print(f"FINAL: {cfg['metric'].upper()} = {mean:.4f} ± {std:.4f}")
        print(f"Individual: {[round(s,4) for s in all_test_scores]}")
        print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
