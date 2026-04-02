# scripts/smoke_test_relbench.py
"""
Smoke test for the RelBench pipeline.
Runs 3 epochs with a tiny model to verify end-to-end functionality.
"""
import torch
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def log(msg):
    print(f"[SMOKE-RB] {msg}", flush=True)

log(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(42)

# ── Step 1: Load RelBench data ────────────────────────────────────────────────
log("Loading RelBench dataset (rel-hm/user-churn)...")
t0 = time.time()

from data.relbench_loader import load_relbench
train_loader, val_loader, _, meta = load_relbench(
    "rel-hm/user-churn", batch_size=16, num_workers=0
)

log(f"Data loaded in {time.time()-t0:.1f}s")
log(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")
log(f"  event_feat_dim={meta['event_feat_dim']}, task_type={meta['task_type']}")

# ── Step 2: Create model ─────────────────────────────────────────────────────
log("Instantiating EdgeMamba3 model (relbench domain)...")
from models.edgemamba3 import EdgeMamba3

model = EdgeMamba3(
    domain="relbench",
    event_feat_dim=meta["event_feat_dim"],
    d_model=32,
    n_layers=1,
    d_state=16,
    mimo_rank=1,
    num_outputs=meta["num_outputs"],
    task_type=meta["task_type"],
)
log(f"Model created. Parameters: {sum(p.numel() for p in model.parameters()):,}")

# ── Step 3: Train ────────────────────────────────────────────────────────────
log("Trainer initialized. Starting fit loop (3 epochs, limit_batches=5)...")
from train.trainer import Trainer

config = {"lr": 1e-3, "epochs": 3, "patience": 100, "batch_size": 16, "limit_batches": 5}
trainer = Trainer(model, config, device=device)

t0 = time.time()
best = trainer.fit(
    train_loader, val_loader,
    domain="relbench",
    metric=meta["metric"],
    save_path="/tmp/smoke_relbench.pt",
)
log(f"Fit loop completed in {time.time()-t0:.1f}s")

# ── Step 4: Test & Report ────────────────────────────────────────────────────
log("Testing best model and generating report...")
test_score = trainer.test(
    val_loader,
    domain="relbench",
    metric=meta["metric"],
    checkpoint_path="/tmp/smoke_relbench.pt",
    report_path="/workspace/results/smoke_relbench_report.txt",
    task_type=meta["task_type"]
)

# ── Results ──────────────────────────────────────────────────────────────────
log(f"Best validation {meta['metric'].upper()}: {best:.4f}")
log(f"Test {meta['metric'].upper()}: {test_score:.4f}")

if best > 0.0:
    log("PASSED — pipeline produces valid AUROC scores and generated a report.")
    sys.exit(0)
else:
    log("FAILED — AUROC is zero or negative")
    sys.exit(1)
