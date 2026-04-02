# tests/test_model_lrgb.py

import torch
import pytest
from torch_geometric.data import Data
from models.edgemamba3 import EdgeMamba3

_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if _CUDA else "cpu")
skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required for Triton kernels")


def make_peptide(n=10, e=20):
    return Data(
        x=torch.randn(n, 9),
        edge_index=torch.randint(0, n, (2, e)),
        edge_attr=torch.randn(e, 3),
        y=torch.zeros(1, 10),   # Peptides-func labels
    ).to(device)


@skip_no_cuda
def test_edgemamba3_lrgb_forward():
    model = EdgeMamba3(
        domain="lrgb", node_in_dim=9, edge_in_dim=3,
        d_model=64, n_layers=2, d_state=16, mimo_rank=2,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_peptide()
    out  = model(data)
    assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"
    assert not torch.isnan(out).any()


@skip_no_cuda
def test_edgemamba3_lrgb_backward():
    """End-to-end gradient flow."""
    model = EdgeMamba3(
        domain="lrgb", node_in_dim=9, edge_in_dim=3,
        d_model=64, n_layers=2, d_state=16, mimo_rank=2,
        num_outputs=10, task_type="classification"
    ).to(device)
    data = make_peptide()
    out  = model(data)
    loss = out.sum()
    loss.backward()

    # Check that key parameters received gradients
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            print(f"WARNING: No gradient for {name}")
