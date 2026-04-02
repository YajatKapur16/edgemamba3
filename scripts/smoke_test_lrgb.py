# scripts/smoke_test_lrgb.py
"""
Smoke test for the LRGB pipeline.
Runs 3 epochs with a tiny model to verify end-to-end functionality.
"""
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.lrgb_loader import load_lrgb
from models.edgemamba3 import EdgeMamba3
from train.trainer import Trainer

torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[SMOKE-LRGB] Device: {device}")

# Use small batch for smoke test
train_loader, val_loader, _, meta = load_lrgb("Peptides-func", batch_size=4, num_workers=4)

print("[SMOKE-LRGB] Instantiating EdgeMamba3 model...")
model = EdgeMamba3(
    domain="lrgb",
    node_in_dim=meta["node_dim"],
    edge_in_dim=meta["edge_dim"],
    d_model=64,
    n_layers=1,
    d_state=16,
    mimo_rank=1,
    num_outputs=meta["num_outputs"],
    task_type=meta["task_type"],
)
print("[SMOKE-LRGB] Model instantiated successfully.")

config = {"lr": 1e-3, "epochs": 3, "patience": 100, "batch_size": 4, "limit_batches": 5}
trainer = Trainer(model, config, device=device)
print("[SMOKE-LRGB] Trainer initialized. Starting fit loop...")

best = trainer.fit(
    train_loader, val_loader,
    domain="lrgb",
    metric=meta["metric"],
    save_path="/tmp/smoke_lrgb.pt",
)
print("[SMOKE-LRGB] Fit loop completed.")

print("[SMOKE-LRGB] Testing best model and generating report...")
test_score = trainer.test(
    val_loader,
    domain="lrgb",
    metric=meta["metric"],
    checkpoint_path="/tmp/smoke_lrgb.pt",
    report_path="/workspace/results/smoke_lrgb_report.txt",
    task_type=meta["task_type"]
)

# Smoke test: just verify the pipeline runs without errors
print(f"[SMOKE-LRGB] Best validation {meta['metric'].upper()}: {best:.4f}")
print(f"[SMOKE-LRGB] Test {meta['metric'].upper()}: {test_score:.4f}")

if best > 0.0:
    print("[SMOKE-LRGB] PASSED — pipeline produces valid AP scores and generated a report.")
    sys.exit(0)
else:
    print("[SMOKE-LRGB] FAILED — AP is zero or negative")
    sys.exit(1)
