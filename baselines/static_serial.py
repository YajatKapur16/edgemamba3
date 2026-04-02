# baselines/static_serial.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from collections import deque
from models.mamba3_encoder import BidirectionalMamba3
from models.readout import AttentionPool, TaskHead
from models.line_graph import build_line_graph, DualEmbedding


def bfs_ordering(edge_index: torch.Tensor, num_nodes: int,
                 start: int = 0) -> torch.Tensor:
    """BFS traversal of line graph → node ordering."""
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)
        adj[d].append(s)

    visited = []
    seen    = set()
    queue   = deque([start])
    while queue:
        node = queue.popleft()
        if node not in seen:
            seen.add(node)
            visited.append(node)
            for nbr in adj[node]:
                if nbr not in seen:
                    queue.append(nbr)

    # Handle disconnected components
    for i in range(num_nodes):
        if i not in seen:
            visited.append(i)

    return torch.tensor(visited, dtype=torch.long)


def dfs_ordering(edge_index: torch.Tensor, num_nodes: int,
                 start: int = 0) -> torch.Tensor:
    """DFS traversal of line graph → node ordering."""
    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)
        adj[d].append(s)

    visited = []
    seen    = set()

    def _dfs(node):
        seen.add(node)
        visited.append(node)
        for nbr in adj[node]:
            if nbr not in seen:
                _dfs(nbr)

    import sys
    sys.setrecursionlimit(10000)
    _dfs(start)
    for i in range(num_nodes):
        if i not in seen:
            _dfs(i)

    return torch.tensor(visited, dtype=torch.long)


def random_ordering(num_nodes: int) -> torch.Tensor:
    """Random permutation — lower bound for serialization quality."""
    return torch.randperm(num_nodes)


STATIC_ORDERINGS = {
    "bfs":    bfs_ordering,
    "dfs":    dfs_ordering,
    "random": random_ordering,
}


class EdgeMamba3_Static(nn.Module):
    """
    EdgeMamba-3 with a static (non-learned) serializer.
    Used for BFS / DFS / Random ablations.
    """
    def __init__(self, ordering: str, node_in_dim: int,
                 edge_in_dim: int, d_model: int = 128,
                 n_layers: int = 4, d_state: int = 32,
                 mimo_rank: int = 4, num_outputs: int = 1,
                 task_type: str = "classification", dropout: float = 0.1):
        super().__init__()
        assert ordering in STATIC_ORDERINGS, f"Unknown ordering: {ordering}"
        self.ordering = ordering
        self.embed    = DualEmbedding(node_in_dim, edge_in_dim, d_model)
        self.encoder  = BidirectionalMamba3(
            d_model=d_model, d_state=d_state, mimo_rank=mimo_rank,
            n_layers=n_layers, dropout=dropout
        )
        self.pool = AttentionPool(d_model)
        self.head = TaskHead(d_model, num_outputs, task_type, dropout)

    def forward(self, data: Data) -> torch.Tensor:
        line_data, orig_ei, x_node, x_edge = build_line_graph(data)
        h = self.embed(x_node, x_edge, orig_ei)  # [|E|, D]

        L = h.shape[0]
        if self.ordering == "random":
            perm = random_ordering(L).to(h.device)
        else:
            perm = STATIC_ORDERINGS[self.ordering](
                line_data.edge_index, L
            ).to(h.device)

        h_seq = h[perm].unsqueeze(0)               # [1, L, D]
        h_enc = self.encoder(h_seq).squeeze(0)     # [L, D]
        h_g   = self.pool(h_enc.unsqueeze(0))      # [1, D]
        return self.head(h_g)

    def loss(self, pred, target):
        return self.head.loss(pred, target)
