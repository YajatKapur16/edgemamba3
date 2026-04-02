# tests/test_baselines.py

import torch
import pytest
from torch_geometric.data import Data

_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if _CUDA else "cpu")
skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required for Triton kernels")


def make_sample_graph(n=8, e=12, node_dim=9, edge_dim=3):
    """Create a minimal graph for baseline testing."""
    return Data(
        x=torch.randn(n, node_dim),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, edge_dim),
        y=torch.zeros(1, 10),
    ).to(device)


@skip_no_cuda
def test_node_mamba3_forward():
    from baselines.node_mamba3 import NodeMamba3
    model = NodeMamba3(
        node_in_dim=9, d_model=32, n_layers=1, d_state=16,
        mimo_rank=1, num_outputs=10, task_type="classification"
    ).to(device)
    data = make_sample_graph()
    out = model(data)
    assert out.shape == (1, 10)
    assert not torch.isnan(out).any()


@skip_no_cuda
def test_edge_mamba2_forward():
    from baselines.edge_mamba2 import build_edge_mamba2
    model = build_edge_mamba2(
        domain="lrgb", node_in_dim=9, edge_in_dim=3,
        d_model=64, n_layers=1, d_state=16, mimo_rank=1,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_sample_graph()
    out = model(data)
    assert out.shape == (1, 10)


@skip_no_cuda
def test_static_bfs_forward():
    from baselines.static_serial import EdgeMamba3_Static
    model = EdgeMamba3_Static(
        "bfs", node_in_dim=9, edge_in_dim=3,
        d_model=32, n_layers=1, d_state=16, mimo_rank=1,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_sample_graph()
    out = model(data)
    assert out.shape == (1, 10)


@skip_no_cuda
def test_static_random_forward():
    from baselines.static_serial import EdgeMamba3_Static
    model = EdgeMamba3_Static(
        "random", node_in_dim=9, edge_in_dim=3,
        d_model=32, n_layers=1, d_state=16, mimo_rank=1,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_sample_graph()
    out = model(data)
    assert out.shape == (1, 10)


@skip_no_cuda
def test_attn_ranking_forward():
    from baselines.attn_ranking import EdgeMamba3_AttnRank
    model = EdgeMamba3_AttnRank(
        node_in_dim=9, edge_in_dim=3,
        d_model=64, n_layers=1, d_state=16, mimo_rank=1,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_sample_graph()
    out = model(data)
    assert out.shape == (1, 10)
