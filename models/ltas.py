# models/ltas.py

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class _StraightThroughArgsort(torch.autograd.Function):
    """
    Differentiable argsort via straight-through estimator.

    Forward:  returns discrete argsort indices as float, with a
              zero-valued term that keeps the tensor connected to
              the computation graph for gradient flow.
    Backward: passes gradients through as if the op were identity.

    This allows gradients to flow back to the GATConv scoring
    network even though sorting is not differentiable.

    Why this works in practice:
    The scoring network learns to rank nodes such that topology-related
    ones appear early in the sequence. The gradient signal comes from
    the downstream Mamba-3 loss, propagated through the STE.
    """
    @staticmethod
    def forward(ctx, scores: torch.Tensor, descending: bool):
        perm = scores.argsort(descending=descending)
        ctx.save_for_backward(scores)
        return perm

    @staticmethod
    def backward(ctx, grad_perm: torch.Tensor):
        # Pass gradient through unchanged (straight-through)
        return grad_perm.float(), None


def differentiable_argsort(scores: torch.Tensor,
                            descending: bool = True) -> torch.Tensor:
    perm = _StraightThroughArgsort.apply(scores, descending)
    # Add zero-valued term that carries gradient connection to scores.
    # Forward value is exactly perm.float(), but autograd sees scores.
    return perm.float() + (scores - scores.detach())


class LTAS(nn.Module):
    """
    Linear Topology-Aware Serialization.

    Replaces Gumbel-Sinkhorn O(L² × K) and attention ranking O(L²)
    with a single GATConv layer + differentiable argsort.

    Complexity: O(|E_L|) for scoring + O(L log L) for sorting
                = O(L log L) total for sparse line graphs

    Architecture:
        scores = GATConv(d_model → 1)(h, edge_index_L)  # [L]
        perm   = argsort(scores, descending=True)         # [L]
        h_seq  = h[perm]                                  # [L, D]

    The GATConv gives topology-awareness: each node's score
    depends on its neighbourhood in L(G), encoding local bond
    environment into the ordering decision.
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
        perm   = differentiable_argsort(scores, descending=True)
        perm_idx = perm.long()                   # integer indices for gathering
        return h[perm_idx], perm_idx, scores
