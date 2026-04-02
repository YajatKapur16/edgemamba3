# baselines/edge_mamba2.py
# Tests RQ3: does Mamba-3 outperform Mamba-2 within EdgeMamba?

from models.edgemamba3 import EdgeMamba3


def build_edge_mamba2(domain: str, **kwargs) -> EdgeMamba3:
    """
    Returns an EdgeMamba3 instance but explicitly using Mamba-2 as the encoder.
    """
    # Override Mamba-specific params for Mamba-2
    kwargs["use_mamba2"] = True
    kwargs["d_state"]    = 64    # Mamba-2 default

    model = EdgeMamba3(domain=domain, **kwargs)
    return model
