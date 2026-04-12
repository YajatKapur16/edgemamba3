# models/readout.py

import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool


class AttentionPool(nn.Module):
    """
    Attention-weighted graph-level pooling.

    Better than mean/sum because it learns which sequence positions
    are most informative for prediction.

    For LRGB: pools over L(G) nodes (bonds) → graph embedding
    For RelBench: pools over event sequence → entity embedding
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(
        self,
        h: torch.Tensor,              # [B, L, D] or [N, D]
        batch: torch.Tensor = None,   # [N] graph index (LRGB batched mode)
        mask: torch.Tensor  = None,   # [B, L] True=valid (RelBench padded mode)
    ) -> torch.Tensor:
        """Returns [B, D] pooled embedding."""

        if h.dim() == 2:
            # LRGB batched: h is [N, D], batch is [N] graph index
            weights = self.gate(h).squeeze(-1)                    # [N]
            weights = _masked_softmax_scatter(weights, batch)     # [N]
            weighted = h * weights.unsqueeze(-1)                  # [N, D]
            return global_add_pool(weighted, batch)               # [B, D]

        elif h.dim() == 3:
            # RelBench / LRGB single-graph: h is [B, L, D]
            weights = self.gate(h).squeeze(-1)                    # [B, L]
            if mask is not None:
                weights = weights.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(weights, dim=-1)              # [B, L]
            if mask is not None:
                weights = weights.masked_fill(~mask, 0.0)
            return (h * weights.unsqueeze(-1)).sum(dim=1)         # [B, D]

        else:
            raise ValueError(f"Unexpected h shape: {h.shape}")


def _masked_softmax_scatter(scores, batch):
    """Softmax within each graph in a batch."""
    from torch_scatter import scatter_softmax
    return scatter_softmax(scores, batch, dim=0)


class TaskHead(nn.Module):
    """
    Task-specific prediction head.

    classification: BCEWithLogitsLoss (handles multi-label)
    regression:     L1Loss (MAE)
    """
    def __init__(
        self,
        d_model: int,
        num_outputs: int,
        task_type: str,
        dropout: float = 0.1,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        assert task_type in ("classification", "regression",
                             "binary_classification")
        self.task_type = task_type
        self.label_smoothing = label_smoothing
        self._pos_weight = None

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_outputs),
        )

    def set_pos_weight(self, pos_weight: torch.Tensor):
        """Set class-level positive weights for imbalanced multi-label tasks."""
        self._pos_weight = pos_weight

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.mlp(h)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.task_type in ("classification", "binary_classification"):
            pw = self._pos_weight
            if pw is not None:
                pw = pw.to(pred.device)
            t = target.float()
            if self.label_smoothing > 0.0:
                t = t * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
            return nn.functional.binary_cross_entropy_with_logits(
                pred, t, pos_weight=pw
            )
        else:
            return nn.functional.l1_loss(pred, target.float())
