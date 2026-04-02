# models/temporal_order.py
# Stateless — no parameters, no learned weights.
# The "serialization" for RelBench is just chronological order.

import torch


def temporal_order(
    h: torch.Tensor,            # [L, D] — event features
    timestamps: torch.Tensor,  # [L] — event times
    seed_time: float,          # causal boundary
    max_len: int = 256,
) -> tuple:
    """
    Orders events chronologically, enforcing causal constraint.

    Unlike LTAS, this requires zero learned parameters.
    The timestamps in RelBench are the natural serialization signal.

    Returns:
        h_ordered: [L', D] — causally ordered (L' ≤ L)
        delta_t:   [L']    — time gaps between consecutive events
        valid_mask:[L']    — all True (already filtered)
    """
    # Causal filter
    mask  = timestamps < seed_time
    h_t   = h[mask]
    ts    = timestamps[mask]

    if h_t.shape[0] == 0:
        return (torch.zeros(1, h.shape[-1], device=h.device),
                torch.zeros(1, device=h.device),
                torch.zeros(1, dtype=torch.bool, device=h.device))

    # Sort ascending (oldest → most recent)
    order = ts.argsort()
    h_ord = h_t[order]
    ts_ord = ts[order]

    # Truncate to max_len (keep most recent)
    if h_ord.shape[0] > max_len:
        h_ord  = h_ord[-max_len:]
        ts_ord = ts_ord[-max_len:]

    # Delta-t (time gaps)
    delta_t = torch.zeros(len(ts_ord), device=h.device)
    if len(ts_ord) > 1:
        delta_t[1:] = (ts_ord[1:] - ts_ord[:-1]).clamp(min=0)

    valid_mask = torch.ones(len(h_ord), dtype=torch.bool, device=h.device)
    return h_ord, delta_t, valid_mask
