# baselines/attn_ranking.py
# Tests: does quadratic attention ranking beat LTAS at O(L²) cost?

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv


class AttentionRankingSerializer(nn.Module):
    """
    Quadratic attention-based serialization (your original Method B).

    Kept at full O(L²) cost deliberately — this is the ablation that
    answers: "does enforcing linearity in LTAS cost accuracy?"

    Architecture:
        K = W_K · (h + GATConv(h, E))    [L, D]  ← topology-aware keys
        Q = W_Q · h                       [L, D]
        attn = softmax(QKᵀ / √D)         [L, L]  ← O(L²) step
        scores = mean(attn, dim=0)        [L]     ← node importance
        perm = argsort(scores)            [L]

    The GATConv on keys gives the same topology-awareness as LTAS
    but uses full pairwise attention. Fair single-variable comparison.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.W_Q       = nn.Linear(d_model, d_model, bias=False)
        self.W_K       = nn.Linear(d_model, d_model, bias=False)
        self.struct_gat = GATConv(d_model, d_model, heads=1,
                                   concat=False, add_self_loops=True)
        self.scale = d_model ** 0.5

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """
        h:          [L, D]
        edge_index: [2, |E_L|]
        Returns: (h_ordered [L, D], perm [L], scores [L])
        """
        h_struct = h + self.struct_gat(h, edge_index)   # [L, D]
        Q = self.W_Q(h)                                  # [L, D]
        K = self.W_K(h_struct)                           # [L, D]

        # Full O(L²) attention matrix
        attn   = torch.softmax((Q @ K.T) / self.scale, dim=-1)  # [L, L]
        scores = attn.mean(dim=0)                                # [L]
        perm   = scores.argsort(descending=True)                 # [L]

        return h[perm], perm, scores


class EdgeMamba3_AttnRank(nn.Module):
    """
    EdgeMamba3 with attention ranking replacing LTAS.
    Inherits all other modules unchanged for clean ablation.
    """
    def __init__(self, **kwargs):
        super().__init__()
        from models.edgemamba3 import EdgeMamba3
        base = EdgeMamba3(domain="lrgb", **kwargs)

        self.embed      = base.embed
        self.serializer = AttentionRankingSerializer(kwargs["d_model"])
        self.encoder    = base.encoder
        self.pool       = base.pool
        self.head       = base.head

    def forward(self, data):
        from models.line_graph import build_line_graph
        line_data, orig_ei, x_node, x_edge = build_line_graph(data)
        h     = self.embed(x_node, x_edge, orig_ei)
        h_seq, perm, _ = self.serializer(h, line_data.edge_index)
        h_enc = self.encoder(h_seq.unsqueeze(0)).squeeze(0)
        h_g   = self.pool(h_enc.unsqueeze(0))
        return self.head(h_g)

    def loss(self, pred, target):
        return self.head.loss(pred, target)
