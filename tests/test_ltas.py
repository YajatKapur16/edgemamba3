# tests/test_ltas.py

import torch
import pytest
from models.ltas import LTAS, differentiable_argsort


def test_ltas_output_shape():
    L, D = 20, 64
    h          = torch.randn(L, D)
    edge_index = torch.randint(0, L, (2, 50))

    ltas = LTAS(d_model=D)
    h_ord, perm, scores = ltas(h, edge_index)

    assert h_ord.shape == (L, D), f"Expected ({L}, {D}), got {h_ord.shape}"
    assert perm.shape  == (L,),   f"Expected ({L},), got {perm.shape}"
    assert scores.shape == (L,)


def test_ltas_is_permutation():
    """Output perm must be a valid permutation."""
    L, D = 15, 32
    h = torch.randn(L, D)
    edge_index = torch.randint(0, L, (2, 30))

    ltas = LTAS(D)
    _, perm, _ = ltas(h, edge_index)
    assert len(set(perm.tolist())) == L, "Permutation must have unique indices"
    assert set(perm.tolist()) == set(range(L))


def test_straight_through_grad():
    """Gradients must flow through differentiable_argsort."""
    scores = torch.randn(10, requires_grad=True)
    perm   = differentiable_argsort(scores)
    loss   = perm.float().sum()
    loss.backward()
    assert scores.grad is not None, "Gradients must flow through STE argsort"
