# tests/test_model_relbench.py

import torch
import pytest
from models.edgemamba3 import EdgeMamba3

_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if _CUDA else "cpu")
skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required for Triton kernels")


@skip_no_cuda
def test_edgemamba3_relbench_forward():
    B, L, FEAT = 8, 32, 15
    model = EdgeMamba3(
        domain="relbench", event_feat_dim=FEAT,
        d_model=64, n_layers=2, d_state=16, mimo_rank=2,
        num_outputs=1, task_type="binary_classification"
    ).to(device)
    seq  = torch.randn(B, L, FEAT, device=device)
    dt   = torch.rand(B, L, device=device) * 86400   # up to 1 day in seconds
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    mask[:, 25:] = False              # 7 padding positions

    out = model(seq, dt, mask)
    assert out.shape == (B, 1), f"Expected ({B}, 1), got {out.shape}"
    assert not torch.isnan(out).any()


@skip_no_cuda
def test_relbench_empty_sequence():
    """Model handles entities with no events gracefully."""
    B, L, FEAT = 4, 1, 15
    model = EdgeMamba3(
        domain="relbench", event_feat_dim=FEAT,
        d_model=64, n_layers=1, d_state=16, mimo_rank=1,
        num_outputs=1, task_type="binary_classification"
    ).to(device)
    seq  = torch.zeros(B, L, FEAT, device=device)    # all-zero sequence
    dt   = torch.zeros(B, L, device=device)
    mask = torch.zeros(B, L, dtype=torch.bool, device=device)  # all padding

    out = model(seq, dt, mask)
    assert out.shape == (B, 1)
