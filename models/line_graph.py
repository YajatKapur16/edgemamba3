# models/line_graph.py

import hashlib

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.transforms import LineGraph as PyGLineGraph
from torch_scatter import scatter_mean
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class DualEmbedding(nn.Module):
    """
    Module 1: Projects line graph nodes (original edges) into model dimension.

    Each L(G) node represents a bond/edge e=(u,v) in the primal graph.
    Its embedding combines:
        h_e = W_edge · x_e  +  W_node · (x_u + x_v) / 2

    The endpoint term gives the model access to atom context without
    losing the bond's own feature identity.

    Uses OGB's AtomEncoder and BondEncoder for categorical molecular
    features (chirality, hybridisation, bond type, etc.) instead of
    nn.Linear on integer-encoded features.

    Complexity: O(|E| · D)
    """
    def __init__(self, node_in_dim: int, edge_in_dim: int, d_model: int):
        super().__init__()
        try:
            from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
            self.edge_proj = BondEncoder(d_model)
            self.node_proj = AtomEncoder(d_model)
            self._use_ogb = True
        except ImportError:
            self.edge_proj = nn.Linear(edge_in_dim, d_model)
            self.node_proj = nn.Linear(node_in_dim, d_model)
            self._use_ogb = False
        self.norm      = nn.LayerNorm(d_model)

    def forward(
        self,
        x_node: torch.Tensor,       # [|V|, node_in_dim] — atom features
        x_edge: torch.Tensor,       # [|E|, edge_in_dim] — bond features
        edge_index: torch.Tensor,   # [2, |E|]
    ) -> torch.Tensor:
        """Returns [|E|, d_model] — line graph node embeddings."""
        src, dst = edge_index  # [|E|]

        if self._use_ogb:
            # OGB encoders expect long-typed integer features
            x_edge = x_edge.long()
            x_node = x_node.long()
        else:
            # Linear layers need float
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
    Uses SciPy's optimized shortest_path on sparse CSR matrix for massive speedup.

    Returns: [num_nodes, num_nodes] distance matrix, clamped at max_dist
    """
    if num_nodes == 0:
        return torch.empty((0, 0), dtype=torch.uint8)
        
    src, dst = edge_index.cpu().numpy()
    weights = np.ones_like(src)
    adj = csr_matrix((weights, (src, dst)), shape=(num_nodes, num_nodes))
    
    dist_mat = shortest_path(csgraph=adj, directed=False, unweighted=True)
    dist_mat = np.clip(dist_mat, 0, max_dist)
    
    return torch.from_numpy(dist_mat).to(torch.uint8)


_GLOBAL_CACHE = {}


def _structural_hash(data: Data) -> str:
    """Compute a collision-resistant hash from actual tensor content."""
    h = hashlib.sha1()
    h.update(data.edge_index.cpu().numpy().tobytes())
    h.update(data.x.cpu().numpy().tobytes())
    return h.hexdigest()


def get_cached_line_graph_and_dist(data: Data):
    """
    Computes and caches the line graph connectivity and shortest-path BFS distances
    for a given PyG Data object. Drastically speeds up multi-epoch training by avoiding
    repetitive O(|V|^3) Python overhead on the main CPU thread.
    """
    h = _structural_hash(data)
    
    if h not in _GLOBAL_CACHE:
        line_data, orig_edge_index, _, _ = build_line_graph(data)
        dist_matrix = compute_graph_distances(
            line_data.edge_index, data.num_edges, max_dist=20
        )

        # Cache only the required tensors (CPU)
        _GLOBAL_CACHE[h] = (line_data.edge_index, orig_edge_index, dist_matrix)
        
    l_edge_index, o_edge_index, dist_matrix = _GLOBAL_CACHE[h]
    
    # Return matched with dynamic node/edge features
    return l_edge_index, o_edge_index, dist_matrix, data.x, data.edge_attr


def warmup_cache(dataset, desc="Pre-caching line graphs"):
    """
    Pre-compute and cache all line graphs + distance matrices before training.
    Eliminates the per-batch CPU stall during the first epoch.
    """
    from tqdm import tqdm
    for i in tqdm(range(len(dataset)), desc=desc, mininterval=2.0):
        data = dataset[i]
        get_cached_line_graph_and_dist(data)
    print(f"  Cached {len(_GLOBAL_CACHE)} unique graphs.")
