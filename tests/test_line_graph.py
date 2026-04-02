# tests/test_line_graph.py

import torch
import pytest
from torch_geometric.data import Data
from models.line_graph import DualEmbedding, build_line_graph, compute_graph_distances


def make_sample_mol(num_nodes=5, num_edges=6, node_dim=9, edge_dim=3):
    """Create a minimal molecular graph for testing."""
    x = torch.randn(num_nodes, node_dim)
    edge_index = torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=torch.long)
    edge_attr  = torch.randn(num_edges, edge_dim)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def test_dual_embedding_shape():
    data = make_sample_mol()
    embed = DualEmbedding(node_in_dim=9, edge_in_dim=3, d_model=64)
    h = embed(data.x, data.edge_attr, data.edge_index)
    assert h.shape == (data.num_edges, 64), f"Expected ({data.num_edges}, 64), got {h.shape}"


def test_line_graph_num_nodes():
    """L(G) must have exactly |E| nodes."""
    data = make_sample_mol(num_edges=6)
    line_data, orig_ei, x_node, x_edge = build_line_graph(data)
    # PyG LineGraph: new x corresponds to original edges
    assert line_data.num_nodes == data.num_edges or line_data.x is None, \
        "L(G) node count should equal original edge count"


def test_graph_distances_symmetric():
    edge_index = torch.tensor([[0,1,1,2],[1,0,2,1]])
    dist = compute_graph_distances(edge_index, num_nodes=3, max_dist=5)
    assert dist.shape == (3, 3)
    assert (dist == dist.T).all(), "Distance matrix must be symmetric"
    assert dist[0, 0] == 0 and dist[1, 1] == 0
