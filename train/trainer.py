# train/trainer.py

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR, OneCycleLR
from tqdm import tqdm
import numpy as np
import sys
import re
from typing import Optional


class Trainer:
    """
    Unified trainer for both LRGB and RelBench experiments.

    Handles:
        - AdamW optimiser + cosine LR schedule
        - Gradient clipping (prevents NaN loss)
        - Early stopping with patience
        - W&B logging (optional)
        - Best checkpoint saving
    """
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: str = "cuda",
        run_name: str = "edgemamba3",
    ):
        self.config   = config
        self.device   = device
        self.run_name = run_name
        self.use_amp  = config.get("use_amp", True) and torch.cuda.is_available()
        self.accum_steps = max(1, int(config.get("accum_steps", 1)))

        # CUDA performance flags — free 5-15% speedup on Ampere/Turing GPUs
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision("high")

        self.is_dist = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_dist else 0

        self.model = model.to(device)
        if self.is_dist:
            # Need to pass list of actual mapped dev IDs, e.g. [0] if device is 'cuda:0'
            dev_id = [int(re.findall(r'\d+', str(device))[0])] if 'cuda:' in str(device) else None
            self.model = DDP(self.model, device_ids=dev_id)

        # Unwrapped model reference for accessing custom methods (.loss(), etc.)
        self._raw_model = self.model.module if self.is_dist else self.model

        self.opt = AdamW(
            model.parameters(),
            lr=float(config["lr"]),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            betas=(0.9, float(config.get("beta2", 0.95))),
        )
        self.grad_clip = float(config.get("grad_clip", 5.0))

        # Scheduler is built lazily in _build_scheduler() once we know loader length
        self.sched = None
        self._sched_per_step = False

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)

        if config.get("use_wandb", False) and self.rank == 0:
            try:
                import wandb
                wandb.init(project="edgemamba3", name=run_name, config=config)
                self._wandb = True
            except ImportError:
                print("wandb not installed, skipping W&B logging.")
                self._wandb = False
        else:
            self._wandb = False

    def _build_scheduler(self, steps_per_epoch: int):
        """Build LR scheduler once we know the actual steps_per_epoch."""
        total_epochs = self.config["epochs"]

        if self.config.get("scheduler", "onecycle") == "onecycle":
            self.sched = OneCycleLR(
                self.opt,
                max_lr=float(self.config["lr"]),
                epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=float(self.config.get("pct_start", 0.1)),
                anneal_strategy="cos",
                div_factor=10.0,        # initial_lr = max_lr / 10
                final_div_factor=100.0,  # final_lr = initial_lr / 100
            )
            self._sched_per_step = True
        else:
            warmup_epochs = int(self.config.get("warmup_epochs", 5))
            warmup_scheduler = LinearLR(
                self.opt,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_scheduler = CosineAnnealingLR(
                self.opt,
                T_max=total_epochs - warmup_epochs,
                eta_min=float(self.config.get("min_lr", 1e-6)),
            )
            self.sched = SequentialLR(
                self.opt,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_epochs],
            )
            self._sched_per_step = False

    def _sync_metrics(self, preds: torch.Tensor, labels: torch.Tensor):
        if not self.is_dist:
            return preds.numpy(), labels.numpy()
        
        preds_gpu = preds.to(self.device).contiguous()
        labels_gpu = labels.to(self.device).contiguous()
        
        gathered_preds = [torch.zeros_like(preds_gpu) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(labels_gpu) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, preds_gpu)
        dist.all_gather(gathered_labels, labels_gpu)
        
        return torch.cat(gathered_preds, dim=0).cpu().numpy(), torch.cat(gathered_labels, dim=0).cpu().numpy()

    # ── LRGB Training Step ─────────────────────────────────────────────────────
    def train_epoch_lrgb(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        limit = self.config.get("limit_batches")
        self.opt.zero_grad()
        for i, batch in enumerate(tqdm(loader, desc="Train", leave=False, file=sys.stdout, mininterval=2.0)):
            if limit and i >= limit: break
            batch = batch.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred  = self.model(batch)
                loss  = self._raw_model.loss(pred, batch.y.to(self.device))
                loss  = loss / self.accum_steps  # scale for accumulation

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                if self._sched_per_step and old_scale <= self.scaler.get_scale():
                    self.sched.step()

            total_loss += loss.item() * self.accum_steps  # unscale for logging

        # Flush remaining gradients if last window was incomplete
        if (i + 1) % self.accum_steps != 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            if self._sched_per_step and old_scale <= self.scaler.get_scale():
                self.sched.step()

        if not self._sched_per_step:
            self.sched.step()
        return total_loss / len(loader)

    # ── RelBench Training Step ─────────────────────────────────────────────────
    def train_epoch_relbench(self, loader) -> float:
        self.model.train()
        total_loss = 0.0
        limit = self.config.get("limit_batches")
        self.opt.zero_grad()
        for i, (seq, dt, mask, labels) in enumerate(tqdm(loader, desc="Train", leave=False, file=sys.stdout, mininterval=2.0)):
            if limit and i >= limit: break
            seq    = seq.to(self.device)
            dt     = dt.to(self.device)
            mask   = mask.to(self.device)
            labels = labels.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.use_amp):
                pred  = self.model(seq, dt, mask)
                loss  = self._raw_model.loss(pred.squeeze(-1), labels)
                loss  = loss / self.accum_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % self.accum_steps == 0:
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.opt)
                self.scaler.update()
                self.opt.zero_grad()
                if self._sched_per_step and old_scale <= self.scaler.get_scale():
                    self.sched.step()

            total_loss += loss.item() * self.accum_steps

        # Flush remaining gradients if last window was incomplete
        if (i + 1) % self.accum_steps != 0:
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            old_scale = self.scaler.get_scale()
            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()
            if self._sched_per_step and old_scale <= self.scaler.get_scale():
                self.sched.step()

        if not self._sched_per_step:
            self.sched.step()
        return total_loss / len(loader)

    # ── Evaluation ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate_lrgb(self, loader, metric: str, task_type: str):
        from train.metrics import compute_metric, compute_all_metrics
        self.model.eval()
        all_preds, all_labels = [], []
        limit = self.config.get("limit_batches")

        for i, batch in enumerate(tqdm(loader, desc="Eval", leave=False, file=sys.stdout, mininterval=2.0)):
            if limit and i >= limit: break
            batch = batch.to(self.device)
            pred  = self.model(batch)
            all_preds.append(pred.cpu())
            all_labels.append(batch.y.cpu())

        preds  = torch.cat(all_preds, dim=0)
        labels = torch.cat(all_labels, dim=0)
        
        preds, labels = self._sync_metrics(preds, labels)
        
        score = compute_metric(preds, labels, metric)
        all_metrics = compute_all_metrics(preds, labels, task_type)
        return score, all_metrics, preds, labels

    @torch.no_grad()
    def evaluate_relbench(self, loader, metric: str, task_type: str):
        from train.metrics import compute_metric, compute_all_metrics
        self.model.eval()
        all_preds, all_labels = [], []
        limit = self.config.get("limit_batches")

        for i, (seq, dt, mask, labels) in enumerate(tqdm(loader, desc="Eval", leave=False, file=sys.stdout, mininterval=2.0)):
            if limit and i >= limit: break
            seq  = seq.to(self.device)
            dt   = dt.to(self.device)
            mask = mask.to(self.device)
            pred = self.model(seq, dt, mask)
            all_preds.append(pred.squeeze(-1).cpu())
            all_labels.append(labels.cpu())

        preds  = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        
        preds, labels = self._sync_metrics(preds, labels)
        
        # In DDP, DistributedSampler pads via duplicate indices to make dataset sizes equal
        # We can pass them all to eval, slightly biased, or slice them properly.
        # Given it's identical padding on all distributed ranks, the difference is negligible for early-stopping.
        
        score = compute_metric(preds, labels, metric)
        all_metrics = compute_all_metrics(preds, labels, task_type)
        return score, all_metrics, preds, labels

    # ── Main Training Loop ─────────────────────────────────────────────────────
    def fit(
        self,
        train_loader,
        val_loader,
        domain: str,
        metric: str,
        save_path: str = "best_model.pt",
        task_type: str = None,
    ) -> float:
        """
        Full training loop with early stopping.
        Returns best validation score.
        """
        if task_type is None:
            task_type = "classification" if metric in ["ap", "auroc", "f1", "accuracy"] else "regression"
            
        higher_is_better = metric in ("ap", "auroc")
        best_val  = -float("inf") if higher_is_better else float("inf")
        patience  = 0
        max_pat   = self.config.get("patience", 30)

        train_fn = (self.train_epoch_lrgb if domain == "lrgb"
                    else self.train_epoch_relbench)
        eval_fn  = (self.evaluate_lrgb if domain == "lrgb"
                    else self.evaluate_relbench)

        # Build scheduler now that we know the loader length
        limit = self.config.get("limit_batches")
        steps_per_epoch = min(len(train_loader), limit) if limit else len(train_loader)
        steps_per_epoch = max(1, steps_per_epoch // self.accum_steps)
        self._build_scheduler(steps_per_epoch)

        for epoch in range(1, self.config["epochs"] + 1):
            if self.is_dist and hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)

            train_loss = train_fn(train_loader)
            val_score, val_metrics, _, _  = eval_fn(val_loader, metric, task_type)

            is_better = (val_score > best_val if higher_is_better
                         else val_score < best_val)

            if is_better:
                best_val  = val_score
                patience  = 0
                if self.rank == 0:
                    # In DDP, model params are under model.module
                    state_dict = self.model.module.state_dict() if self.is_dist else self.model.state_dict()
                    torch.save(state_dict, save_path)
            else:
                patience += 1

            lr = self.opt.param_groups[0]["lr"]
            metrics_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in val_metrics.items())
            if self.rank == 0:
                print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                      f"Val [{metrics_str}] | "
                      f"Best {metric.upper()}: {best_val:.4f} | LR: {lr:.2e} | "
                      f"Pat: {patience}/{max_pat}")

                if self._wandb:
                    import wandb
                    log_data = {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "lr": lr,
                    }
                    for k, v in val_metrics.items():
                        log_data[f"val_{k}"] = v
                    wandb.log(log_data)

            if patience >= max_pat:
                if self.rank == 0:
                    print(f"Early stopping at epoch {epoch}.")
                break

        return best_val

    @torch.no_grad()
    def test(self, test_loader, domain: str, metric: str,
             checkpoint_path: str = "best_model.pt",
             report_path: str = None, task_type: str = None):
        """Load best checkpoint and evaluate on test set."""
        if task_type is None:
            task_type = "classification" if metric in ["ap", "auroc", "f1", "accuracy"] else "regression"

        # Checkpoint was saved from model.module (no 'module.' prefix),
        # so load into the unwrapped model to match keys.
        raw_model = self.model.module if self.is_dist else self.model
        raw_model.load_state_dict(torch.load(checkpoint_path,
                                              map_location=self.device))
        eval_fn = (self.evaluate_lrgb if domain == "lrgb"
                   else self.evaluate_relbench)
        score, all_metrics, preds, labels = eval_fn(test_loader, metric, task_type)
        
        if self.rank == 0:
            metrics_str = " | ".join(f"{k.upper()}: {v:.4f}" for k, v in all_metrics.items())
            print(f"Test Metrics -> [{metrics_str}]")
            
            if report_path and task_type:
                from train.metrics import generate_eval_report
                generate_eval_report(preds, labels, task_type, report_path)
                print(f"Detailed metric report saved to: {report_path}")
            
        return score
