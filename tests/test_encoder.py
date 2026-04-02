# tests/test_encoder.py

import torch
import pytest
from models.mamba3_encoder import BidirectionalMamba3

_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if _CUDA else "cpu")
skip_no_cuda = pytest.mark.skipif(not _CUDA, reason="CUDA required for Triton kernels")


@skip_no_cuda
def test_encoder_basic():
    """Basic shape test."""
    B, L, D = 4, 32, 64
    encoder = BidirectionalMamba3(d_model=D, d_state=16, mimo_rank=2, n_layers=2).to(device)
    x   = torch.randn(B, L, D, device=device)
    out = encoder(x)
    assert out.shape == (B, L, D), f"Expected ({B},{L},{D}), got {out.shape}"


@skip_no_cuda
def test_encoder_with_time_enc():
    B, L, D = 2, 24, 64
    encoder = BidirectionalMamba3(d_model=D, d_state=16, mimo_rank=1,
                                   n_layers=1, use_time_enc=True).to(device)
    x      = torch.randn(B, L, D, device=device)
    delta_t = (torch.rand(B, L, device=device) * 3600)   # seconds
    out    = encoder(x, delta_t=delta_t)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any(), "NaN in encoder output with time encoding"


@skip_no_cuda
def test_encoder_no_nan():
    """No NaN for typical input ranges."""
    B, L, D = 4, 50, 128
    encoder = BidirectionalMamba3(d_model=D, d_state=32, mimo_rank=4, n_layers=3).to(device)
    x   = torch.randn(B, L, D, device=device)
    out = encoder(x)
    assert not torch.isnan(out).any(), "NaN in encoder output"
    assert not torch.isinf(out).any(), "Inf in encoder output"


@skip_no_cuda
def test_encoder_padding_mask():
    B, L, D = 3, 20, 64
    encoder = BidirectionalMamba3(d_model=D, d_state=16, mimo_rank=1, n_layers=1).to(device)
    x    = torch.randn(B, L, D, device=device)
    mask = torch.ones(B, L, dtype=torch.bool, device=device)
    mask[:, 15:] = False   # last 5 positions are padding
    out  = encoder(x, padding_mask=mask)
    assert out.shape == (B, L, D)
