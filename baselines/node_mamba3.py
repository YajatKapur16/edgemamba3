# baselines/node_mamba3.py
# Tests RQ1: does line graph edge-centricity improve over node-centric SSM?

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from models.mamba3_encoder import BidirectionalMamba3
from models.readout import AttentionPool, TaskHead


class NodeMamba3(nn.Module):
    """
    Ablation: EdgeMamba-3 without line graph transformation.
    Sequences NODES instead of EDGES for Mamba-3.
    Everything else (encoder, readout, hyperparams) is IDENTICAL.
    """
    def __init__(
        self,
        node_in_dim: int,
        d_model: int     = 128,
        n_layers: int    = 4,
        d_state: int     = 32,
        mimo_rank: int   = 4,
        num_outputs: int = 1,
        task_type: str   = "classification",
        dropout: float   = 0.1,
    ):
        super().__init__()
        self.embed   = nn.Linear(node_in_dim, d_model)
        self.encoder = BidirectionalMamba3(
            d_model=d_model, d_state=d_state, mimo_rank=mimo_rank,
            n_layers=n_layers, dropout=dropout
        )
        self.pool = AttentionPool(d_model)
        self.head = TaskHead(d_model, num_outputs, task_type, dropout)

    def forward(self, data: Data) -> torch.Tensor:
        # Serialize nodes by degree (fair baseline — captures topology)
        deg = torch.zeros(data.num_nodes, device=data.x.device)
        src, dst = data.edge_index
        deg.scatter_add_(0, src, torch.ones_like(src, dtype=torch.float))
        deg.scatter_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        perm = deg.argsort(descending=True)

        h     = self.embed(data.x[perm]).unsqueeze(0)   # [1, |V|, D]
        h_enc = self.encoder(h).squeeze(0)              # [|V|, D]
        h_g   = self.pool(h_enc.unsqueeze(0))           # [1, D]
        return self.head(h_g)

    def loss(self, pred, target):
        return self.head.loss(pred, target)
