# models/line_graph.py

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph as PyGLineGraph
from torch_scatter import scatter_mean


class DualEmbedding(nn.Module):
    """
    Module 1: Projects line graph nodes (original edges) into model dimension.

    Each L(G) node represents a bond/edge e=(u,v) in the primal graph.
    Its embedding combines:
        h_e = W_edge · x_e  +  W_node · (x_u + x_v) / 2

    The endpoint term gives the model access to atom context without
    losing the bond's own feature identity.

    Complexity: O(|E| · D)
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, d_model: int):
        super().__init__()
        self.edge_proj = nn.Linear(edge_in_dim, d_model)
        self.node_proj = nn.Linear(node_in_dim, d_model)
        self.norm      = nn.LayerNorm(d_model)

    def forward(
        self,
        x_node: torch.Tensor,       # [|V|, node_in_dim] — atom features
        x_edge: torch.Tensor,       # [|E|, edge_in_dim] — bond features
        edge_index: torch.Tensor,   # [2, |E|]
    ) -> torch.Tensor:
        """Returns [|E|, d_model] — line graph node embeddings."""
        src, dst = edge_index  # [|E|]

        # Edge/node features may be integer-encoded; linear layers need float
        x_edge = x_edge.float()
        x_node = x_node.float()

        h_edge      = self.edge_proj(x_edge)                           # [|E|, D]
        h_endpoints = (self.node_proj(x_node[src]) +
                       self.node_proj(x_node[dst])) / 2.0              # [|E|, D]

        return self.norm(h_edge + h_endpoints)


class LineGraphBuilder:
    """
    Builds L(G) from a PyG Data object.

    Important: PyG's LineGraph transform returns a new Data where:
        - new x = old edge_attr  (bond features → L(G) node features)
        - new edge_index = shared-endpoint connectivity
        - new edge_attr may be None

    We use the transform, then attach the original graph info
    needed for dual embedding.
    """
    def __init__(self):
        self._transform = PyGLineGraph(force_directed=False)

    def __call__(self, data: Data) -> tuple:
        """
        Returns:
            line_data: PyG Data representing L(G)
            orig_edge_index: original graph edge_index (for dual embedding)
            orig_node_feat: original graph node features
            orig_edge_feat: original graph edge features
        """
        # Store originals before transform
        orig_edge_index = data.edge_index.clone()
        orig_node_feat  = data.x.clone()
        orig_edge_feat  = data.edge_attr.clone()

        # Apply PyG line graph transform
        line_data = self._transform(data.clone())

        # line_data.edge_index = L(G) connectivity
        # line_data.x = None (edge features become L(G) node features
        #                      via DualEmbedding, not via this transform)

        return line_data, orig_edge_index, orig_node_feat, orig_edge_feat


_line_graph_builder = LineGraphBuilder()  # singleton


def build_line_graph(data: Data):
    """Convenience function."""
    return _line_graph_builder(data)


def compute_graph_distances(edge_index: torch.Tensor, num_nodes: int,
                             max_dist: int = 10) -> torch.Tensor:
    """
    Compute shortest-path distances between all pairs of nodes.
    Used for the positional distance encoding in Module 3.

    BFS from each node — O(|V| · (|V| + |E|)).
    Only feasible for LRGB graphs (avg |V|=307 bonds after L(G)).

    Returns: [num_nodes, num_nodes] distance matrix, clamped at max_dist
    """
    dist = torch.full((num_nodes, num_nodes), max_dist, dtype=torch.uint8)
    dist.fill_diagonal_(0)

    adj = [[] for _ in range(num_nodes)]
    src, dst = edge_index
    for s, d in zip(src.tolist(), dst.tolist()):
        adj[s].append(d)
        adj[d].append(s)

    for start in range(num_nodes):
        visited = {start: 0}
        queue   = [start]
        while queue:
            node  = queue.pop(0)
            for nbr in adj[node]:
                if nbr not in visited:
                    visited[nbr] = visited[node] + 1
                    dist[start][nbr] = min(visited[nbr], max_dist)
                    queue.append(nbr)

    return dist


_GLOBAL_CACHE = {}

def get_cached_line_graph_and_dist(data: Data):
    """
    Computes and caches the line graph connectivity and shortest-path BFS distances
    for a given PyG Data object. Drastically speeds up multi-epoch training by avoiding
    repetitive O(|V|^3) Python overhead on the main CPU thread.
    """
    # Simple fast hash signature for the graph
    edge_sum = data.edge_index.sum().item() if data.edge_index is not None else 0
    node_sum = data.x.sum().item() if data.x is not None else 0
    h = f"{data.num_nodes}_{data.num_edges}_{edge_sum}_{node_sum}"
    
    if h not in _GLOBAL_CACHE:
        line_data, orig_edge_index, _, _ = build_line_graph(data)
        dist_matrix = None
        if data.num_edges <= 500:
            dist_matrix = compute_graph_distances(line_data.edge_index, data.num_edges)
        
        # Cache only the required tensors (CPU)
        _GLOBAL_CACHE[h] = (line_data.edge_index, orig_edge_index, dist_matrix)
        
    l_edge_index, o_edge_index, dist_matrix = _GLOBAL_CACHE[h]
    
    # Return matched with dynamic node/edge features
    return l_edge_index, o_edge_index, dist_matrix, data.x, data.edge_attr
