# models/mamba3_encoder.py

import torch
import torch.nn as nn
import math

# ─── Mamba Imports ────────────────────────────────────────────────────────────
_HAS_MAMBA3 = False
try:
    from mamba_ssm.modules.mamba3 import Mamba3
    from mamba_ssm import Mamba2
    _HAS_MAMBA3 = True
except ImportError:
    try:
        from mamba_ssm import Mamba2
        Mamba3 = None  # Mamba-3 unavailable; Mamba-2 used as fallback
        print("[mamba3_encoder] Mamba-3 not found — falling back to Mamba-2.")
    except ImportError:
        raise RuntimeError(
            "Neither Mamba-3 nor Mamba-2 found.\n"
            "Install: pip install mamba-ssm causal-conv1d"
        )


# ─── Positional Encodings ─────────────────────────────────────────────────────

class RelativeTimeEncoding(nn.Module):
    """
    Encodes irregular delta_t between RelBench events.

    Uses log-scaled sinusoidal encoding because RelBench timestamps
    span orders of magnitude (seconds to months).

    Mamba-3's trapezoidal discretization handles variable Δt natively
    in the recurrence — this encoding is an additional input-level signal.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.proj    = nn.Linear(d_model, d_model)

    def forward(self, delta_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        delta_t: [B, L] — time gaps in seconds
        x:       [B, L, D]
        Returns: [B, L, D]
        """
        B, L, D = x.shape
        dt_log = torch.log1p(delta_t.float().abs()).unsqueeze(-1)  # [B, L, 1]

        # Sinusoidal frequencies
        div_term = torch.exp(
            torch.arange(0, D, 2, device=x.device, dtype=torch.float)
            * -(math.log(10000.0) / D)
        )

        pe = torch.zeros(B, L, D, device=x.device)
        pe[..., 0::2] = torch.sin(dt_log * div_term)
        pe[..., 1::2] = torch.cos(dt_log * div_term[:D//2])

        return self.proj(pe)


class GraphDistanceEncoding(nn.Module):
    """
    Encodes shortest-path graph distances between serialized positions.

    In the serialized bond sequence, adjacent positions may be
    graph-distant bonds. This encoding signals that gap to Mamba-3,
    allowing it to adjust state transitions based on actual graph distance.

    Used only for LRGB (RelBench uses temporal encoding instead).
    """
    def __init__(self, d_model: int, max_dist: int = 20):
        super().__init__()
        self.embed = nn.Embedding(max_dist + 2, d_model, padding_idx=0)
        self.max_dist = max_dist

    def forward(self, dist_matrix: torch.Tensor) -> torch.Tensor:
        """
        dist_matrix: [L, L] or [B, L, L] — pairwise graph distances
        Returns: [1, L, D] or [B, L, D] — per-position mean distance encoding
        """
        mean_dist = dist_matrix.float().mean(dim=-1).long()  # [L] or [B, L]
        mean_dist = mean_dist.clamp(0, self.max_dist)
        pe = self.embed(mean_dist)                           # [L, D] or [B, L, D]
        if pe.dim() == 2:
            return pe.unsqueeze(0)                           # [1, L, D]
        return pe                                            # [B, L, D]


# ─── Core Encoder ─────────────────────────────────────────────────────────────

class BidirectionalMamba3(nn.Module):
    """
    Bidirectional Mamba-3 encoder.

    Runs two Mamba-3 passes (forward + backward) and projects
    their concatenation. Residual connections + RMSNorm per layer.

    Mamba-3 advantages over Mamba-2 used here:
        1. Trapezoidal discretization — better irregular Δt handling
        2. Complex-valued states via data-dependent RoPE — detects
           cyclic/periodic patterns (ring structures, periodic user behaviour)
        3. MIMO rank-R — compute-bound GPU execution

    For RelBench: backward pass sees reversed causal sequence.
    This is acceptable because the backward pass still only sees
    events that occurred BEFORE seed_time (same data, reversed order).

    Args:
        d_model:       hidden dimension
        d_state:       SSM state size (32 = Mamba-3 default, same quality as Mamba-2 64)
        mimo_rank:     MIMO rank (1=SISO Mamba-2-like, 4=MIMO efficient)
        n_layers:      number of Mamba-3 layers
        dropout:       dropout rate
        use_time_enc:  add relative time encoding (RelBench)
        use_dist_enc:  add graph distance encoding (LRGB)
    """
    def __init__(
        self,
        d_model: int    = 128,
        d_state: int    = 32,
        mimo_rank: int  = 4,
        n_layers: int   = 4,
        dropout: float  = 0.1,
        use_time_enc: bool = False,
        use_dist_enc: bool = False,
        expand: int     = 2,
        headdim: int    = 64,
        use_mamba2: bool = False,
    ):
        super().__init__()
        self.d_model     = d_model
        self.n_layers    = n_layers
        self.use_time    = use_time_enc
        self.use_dist    = use_dist_enc
        self.use_mamba2  = use_mamba2

        # Build Mamba kwargs
        mamba_kwargs = {
            "d_model": d_model,
            "d_state": d_state,
            "expand":  expand,
            "headdim": headdim,
        }
        
        # Select encoder class
        if use_mamba2 or not _HAS_MAMBA3:
            encoder_cls = Mamba2
            if not use_mamba2 and not _HAS_MAMBA3:
                print("[BidirectionalMamba3] Mamba-3 unavailable, using Mamba-2 fallback.")
        else:
            encoder_cls = Mamba3
            mamba_kwargs["mimo_rank"] = mimo_rank

        # Stacked bidirectional layers
        self.fwd_layers = nn.ModuleList(
            [encoder_cls(**mamba_kwargs) for _ in range(n_layers)]
        )
        self.bwd_layers = nn.ModuleList(
            [encoder_cls(**mamba_kwargs) for _ in range(n_layers)]
        )
        self.norms = nn.ModuleList(
            [nn.RMSNorm(d_model) for _ in range(n_layers)]
        )

        # Projection after concatenating fwd + bwd
        self.out_proj = nn.Linear(2 * d_model, d_model, bias=False)
        self.dropout    = nn.Dropout(dropout)

        # Optional encodings
        if use_time_enc:
            self.time_enc = RelativeTimeEncoding(d_model)
        if use_dist_enc:
            self.dist_enc = GraphDistanceEncoding(d_model)

    def forward(
        self,
        x: torch.Tensor,                      # [B, L, D]
        delta_t: torch.Tensor   = None,       # [B, L] for RelBench
        dist_matrix: torch.Tensor = None,     # [L, L] for LRGB
        padding_mask: torch.Tensor = None,    # [B, L] True=valid False=pad
    ) -> torch.Tensor:
        """Returns [B, L, D] encoded sequence."""

        # Add positional signals before the SSM layers
        if self.use_time and delta_t is not None:
            x = x + self.time_enc(delta_t, x)

        if self.use_dist and dist_matrix is not None:
            dist_pe = self.dist_enc(dist_matrix)  # [1, L, D]
            x = x + dist_pe

        # Zero-out padding positions before processing
        if padding_mask is not None:
            x = x * padding_mask.unsqueeze(-1).float()
            
            # Prepare vectorized fast flip mechanism for dynamic lengths
            lengths = padding_mask.sum(dim=1).long()  # [B]
            L = x.size(1)
            idx = torch.arange(L, device=x.device).unsqueeze(0).expand(x.size(0), -1)  # [B, L]
            rev_idx = torch.where(idx < lengths.unsqueeze(1), lengths.unsqueeze(1) - 1 - idx, idx)
            rev_idx = rev_idx.unsqueeze(-1).expand(-1, -1, x.size(-1))
            
            def flip_seq(t):
                return torch.gather(t, 1, rev_idx)
        else:
            def flip_seq(t):
                return t.flip(1)

        # Layer-wise bidirectional Mamba-3 with residual connections
        h = x
        for fwd, bwd, norm in zip(self.fwd_layers, self.bwd_layers, self.norms):
            fwd_out = fwd(h)                               # [B, L, D]
            bwd_out = flip_seq(bwd(flip_seq(h)))           # [B, L, D]

            # Merge directions
            combined = self.out_proj(
                torch.cat([fwd_out, bwd_out], dim=-1)
            )                                              # [B, L, D]

            # Residual + norm
            h = norm(h + self.dropout(combined))
            
            # Prevent padded positions from accumulating noise in successive backward passes
            if padding_mask is not None:
                h = h * padding_mask.unsqueeze(-1).float()

        return h
