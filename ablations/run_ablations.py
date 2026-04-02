# ablations/run_ablations.py

"""
Runs all 6 ablation studies systematically.
Each ablation is a single-variable change from the full EdgeMamba-3 model.
Results written to ablation_results.csv.
"""

import torch
import yaml
import csv
import os
from datetime import datetime

from data.lrgb_loader import load_lrgb
from models.edgemamba3 import EdgeMamba3
from baselines.node_mamba3 import NodeMamba3
from baselines.attn_ranking import EdgeMamba3_AttnRank
from baselines.static_serial import EdgeMamba3_Static
from baselines.edge_mamba2 import build_edge_mamba2
from train.trainer import Trainer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_single(model, config, dataset_name="Peptides-func",
               n_seeds=5, save_prefix="abl"):
    """Run model for n_seeds and return mean ± std."""
    scores = []
    for seed in range(n_seeds):
        torch.manual_seed(seed)

        train_loader, val_loader, test_loader, meta = load_lrgb(
            dataset_name, batch_size=config["batch_size"]
        )
        trainer = Trainer(model, config, device=DEVICE,
                          run_name=f"{save_prefix}_seed{seed}")

        trainer.fit(train_loader, val_loader,
                    domain="lrgb", metric=meta["metric"],
                    save_path=f"checkpoints/{save_prefix}_seed{seed}.pt")

        test_score = trainer.test(test_loader, domain="lrgb",
                                  metric=meta["metric"],
                                  checkpoint_path=f"checkpoints/{save_prefix}_seed{seed}.pt")
        scores.append(test_score)

    mean = sum(scores) / len(scores)
    std  = (sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5
    return mean, std, scores


BASE_CONFIG = {
    "d_model": 128, "n_layers": 6, "d_state": 32, "mimo_rank": 4,
    "dropout": 0.1, "lr": 0.0005, "weight_decay": 1e-5, "batch_size": 32,
    "epochs": 200, "patience": 30, "use_wandb": False,
}

BASE_KWARGS = {
    "node_in_dim": 9, "edge_in_dim": 3, "d_model": 128, "n_layers": 6,
    "d_state": 32, "mimo_rank": 4, "num_outputs": 10,
    "task_type": "classification", "dropout": 0.1,
}


# ─── Ablation definitions ────────────────────────────────────────────────────

ABLATIONS = {

    # RQ1: Edge vs Node centricity
    "abl1_edge_mamba3": {
        "description": "Full EdgeMamba-3 (proposed)",
        "model_fn": lambda: EdgeMamba3(domain="lrgb", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl1_node_mamba3": {
        "description": "NodeMamba-3 (no line graph)",
        "model_fn": lambda: NodeMamba3(
            node_in_dim=9,
            **{k: v for k, v in BASE_KWARGS.items()
               if k not in ("edge_in_dim", "node_in_dim")}
        ),
        "dataset": "Peptides-func",
    },

    # RQ2: Serialization comparison
    "abl2_ltas": {
        "description": "LTAS serialization (proposed, O(L log L))",
        "model_fn": lambda: EdgeMamba3(domain="lrgb", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl2_attn_rank": {
        "description": "Attention ranking (quadratic O(L^2))",
        "model_fn": lambda: EdgeMamba3_AttnRank(**BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl2_bfs": {
        "description": "BFS ordering (static)",
        "model_fn": lambda: EdgeMamba3_Static("bfs", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl2_dfs": {
        "description": "DFS ordering (static)",
        "model_fn": lambda: EdgeMamba3_Static("dfs", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl2_random": {
        "description": "Random ordering (lower bound)",
        "model_fn": lambda: EdgeMamba3_Static("random", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },

    # RQ3: Mamba version
    "abl3_mamba3_mimo": {
        "description": "Mamba-3 MIMO rank-4 (proposed)",
        "model_fn": lambda: EdgeMamba3(domain="lrgb", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    "abl3_mamba3_siso": {
        "description": "Mamba-3 SISO (no MIMO)",
        "model_fn": lambda: EdgeMamba3(domain="lrgb",
                                        **{**BASE_KWARGS, "mimo_rank": 1}),
        "dataset": "Peptides-func",
    },
    "abl3_mamba2": {
        "description": "Mamba-2 equivalent (d_state=64, mimo_rank=1)",
        "model_fn": lambda: build_edge_mamba2(domain="lrgb", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },

    # Ablation 4: Bidirectional vs Unidirectional
    "abl4_bidir": {
        "description": "Bidirectional Mamba-3 (proposed)",
        "model_fn": lambda: EdgeMamba3(domain="lrgb", **BASE_KWARGS),
        "dataset": "Peptides-func",
    },
    # (unidirectional requires a config flag in BidirectionalMamba3)

    # Ablation 5: Distance encoding
    "abl5_with_dist": {
        "description": "With graph distance encoding",
        "model_fn": lambda: EdgeMamba3(domain="lrgb",
                                        **{**BASE_KWARGS, "use_dist_enc": True}),
        "dataset": "Peptides-func",
    },
    "abl5_no_dist": {
        "description": "Without graph distance encoding",
        "model_fn": lambda: EdgeMamba3(domain="lrgb",
                                        **{**BASE_KWARGS, "use_dist_enc": False}),
        "dataset": "Peptides-func",
    },
}


def run_all_ablations(n_seeds: int = 5, output_csv: str = "ablation_results.csv"):
    os.makedirs("checkpoints", exist_ok=True)
    results = []

    for abl_name, abl_cfg in ABLATIONS.items():
        print(f"\n{'='*60}")
        print(f"Running: {abl_name} — {abl_cfg['description']}")
        print(f"{'='*60}")

        model = abl_cfg["model_fn"]()
        mean, std, seeds = run_single(
            model, BASE_CONFIG,
            dataset_name=abl_cfg["dataset"],
            n_seeds=n_seeds,
            save_prefix=abl_name,
        )

        print(f"Result: {mean:.4f} ± {std:.4f}")
        results.append({
            "ablation":    abl_name,
            "description": abl_cfg["description"],
            "dataset":     abl_cfg["dataset"],
            "mean":        round(mean, 4),
            "std":         round(std, 4),
            "seeds":       seeds,
            "timestamp":   datetime.now().isoformat(),
        })

    # Write CSV
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ablation", "description", "dataset",
                           "mean", "std", "seeds", "timestamp"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"\nAll ablations complete. Results saved to {output_csv}")
    return results


if __name__ == "__main__":
    run_all_ablations(n_seeds=5)
