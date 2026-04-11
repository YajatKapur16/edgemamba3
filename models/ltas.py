# models/ltas.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class LTAS(nn.Module):
    """
    Linear Topology-Aware Serialization.

    Replaces Gumbel-Sinkhorn O(L² × K) and attention ranking O(L²)
    with a single GATConv layer + argsort.

    Complexity: O(|E_L|) for scoring + O(L log L) for sorting
                = O(L log L) total for sparse line graphs

    Architecture:
        scores = GATConv(d_model → 1)(h, edge_index_L)        # [L]
        h      = h + score_proj(scores)                        # [L, D]
        perm   = argsort(scores, descending=True)              # [L]
        h_seq  = h[perm]                                       # [L, D]

    GATConv receives gradients through the score_proj additive path:
        loss → Mamba encoder → h[perm] → (h + score_proj(scores)) → GATConv

    The score_proj (Linear 1→D, no bias) maps scalar scores into the
    feature space with learned, bounded weights. This replaces the
    broken STE approach that added raw unbounded scores to embeddings.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # Single GAT layer: d_model → scalar score per node
        self.score_gat = GATConv(
            in_channels=d_model,
            out_channels=1,
            heads=1,
            add_self_loops=True,
        )
        # Project scalar score → d_model for differentiable gradient path.
        # bias=False so zero scores contribute nothing at init.
        self.score_proj = nn.Linear(1, d_model, bias=False)

    def forward(
        self,
        h: torch.Tensor,           # [L, D]
        edge_index: torch.Tensor,  # [2, |E_L|]
    ) -> tuple:
        """
        Returns:
            h_ordered: [L, D] — topology-ordered sequence
            perm:      [L]    — permutation indices (for distance encoding)
            scores:    [L]    — raw scores (for visualisation / ablation)
        """
        scores = self.score_gat(h, edge_index).squeeze(-1)  # [L]

        # Inject score signal into features BEFORE sorting.
        # This is differentiable: GATConv gets gradients through score_proj.
        score_enc = self.score_proj(scores.unsqueeze(-1))    # [L, D]
        h = h + score_enc

        # Non-differentiable argsort for ordering
        perm_idx = scores.argsort(descending=True)
        h_ordered = h[perm_idx]

        return h_ordered, perm_idx, scores
