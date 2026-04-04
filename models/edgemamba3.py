# models/edgemamba3.py

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

from models.line_graph import DualEmbedding, build_line_graph, compute_graph_distances
from models.ltas import LTAS
from models.mamba3_encoder import BidirectionalMamba3
from models.readout import AttentionPool, TaskHead


class EdgeMamba3(nn.Module):
    """
    Unified EdgeMamba-3 model.

    Two modes controlled by `domain` flag:

    domain="lrgb":
        - Applies line graph L(G) transformation
        - Uses LTAS for topology-aware serialization
        - Graph-level attention pooling
        - Suitable for: Peptides-func, Peptides-struct

    domain="relbench":
        - Transaction nodes are already rich-feature nodes (no L(G) needed)
        - Temporal ordering is free from timestamps (no LTAS needed)
        - Entity-level attention pooling over event sequence
        - Suitable for: rel-hm user-churn, rel-amazon user-ltv

    Shared components:
        - BidirectionalMamba3 encoder (same module, different configs)
        - AttentionPool + TaskHead
    """

    def __init__(
        self,
        domain: str,
        # LRGB-specific dims
        node_in_dim: int    = None,    # primal graph node feature dim
        edge_in_dim: int    = None,    # primal graph edge feature dim
        # RelBench-specific dims
        event_feat_dim: int = None,    # transaction feature dim
        # Shared
        d_model: int        = 128,
        n_layers: int       = 4,
        d_state: int        = 32,
        mimo_rank: int      = 4,
        num_outputs: int    = 1,
        task_type: str      = "classification",
        dropout: float      = 0.1,
        use_dist_enc: bool  = True,   # graph distance encoding (LRGB only)
        use_virtual_node: bool = False,
        use_mamba2: bool    = False,
    ):
        super().__init__()
        assert domain in ("lrgb", "relbench"), f"Unknown domain: {domain}"
        self.domain    = domain
        self.d_model   = d_model
        
        self.use_virtual_node = use_virtual_node
        if self.use_virtual_node and domain == "lrgb":
            self.virtual_token = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.normal_(self.virtual_token, std=0.02)

        # ── Module 1: Input Embedding ──────────────────────────────────────
        if domain == "lrgb":
            assert node_in_dim and edge_in_dim, \
                "LRGB mode requires node_in_dim and edge_in_dim"
            self.embed = DualEmbedding(node_in_dim, edge_in_dim, d_model)
        else:
            assert event_feat_dim, \
                "RelBench mode requires event_feat_dim"
            self.embed = nn.Sequential(
                nn.Linear(event_feat_dim, d_model),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout),
            )

        # ── Module 2: Serialization ────────────────────────────────────────
        if domain == "lrgb":
            self.serializer = LTAS(d_model)
        # RelBench: temporal sort happens in data loader (no learned params)

        # ── Module 3: Mamba-3 Encoder ──────────────────────────────────────
        self.encoder = BidirectionalMamba3(
            d_model=d_model,
            d_state=d_state,
            mimo_rank=mimo_rank,
            n_layers=n_layers,
            dropout=dropout,
            use_time_enc=(domain == "relbench"),
            use_dist_enc=(domain == "lrgb" and use_dist_enc),
            use_mamba2=use_mamba2,
        )

        # ── Module 4: Readout + Task Head ──────────────────────────────────
        self.pool = AttentionPool(d_model)
        self.head = TaskHead(d_model, num_outputs, task_type, dropout)

    # ── LRGB Forward ──────────────────────────────────────────────────────────
    def forward_lrgb(self, data: Data) -> torch.Tensor:
        """
        data: PyG Data object (single molecular graph or batched)
        Returns: [B, num_outputs] predictions
        """
        # Handle batched graphs from DataLoader
        if hasattr(data, 'batch') and data.batch is not None:
            return self._forward_lrgb_batched(data)
        else:
            return self._forward_lrgb_single(data)

    def _forward_lrgb_single(self, data: Data) -> torch.Tensor:
        """Process a single graph (used in testing)."""
        # Build L(G) using cached results to avoid repetitive CPU bottleneck
        from models.line_graph import get_cached_line_graph_and_dist
        line_edge_index, orig_edge_index, dist_matrix, x_node, x_edge = get_cached_line_graph_and_dist(data)

        # Module 1: Dual embedding → [|E|, D]
        h = self.embed(x_node, x_edge, orig_edge_index.to(x_node.device))

        # Module 2: LTAS → [|E|, D] ordered
        h_seq, perm, _ = self.serializer(h, line_edge_index.to(x_node.device))

        if self.use_virtual_node:
            h_seq = torch.cat([self.virtual_token.squeeze(0), h_seq], dim=0)

        # Graph distances for distance encoding
        if self.encoder.use_dist and dist_matrix is not None:
            dist_matrix = dist_matrix.to(h_seq.device)
            if perm is not None:
                dist_matrix = dist_matrix[perm][:, perm]
                
            if self.use_virtual_node:
                l = dist_matrix.shape[0]
                new_dist = torch.zeros((l+1, l+1), device=dist_matrix.device, dtype=torch.float)
                new_dist[1:, 1:] = dist_matrix
                new_dist[0, 1:] = 1.0
                new_dist[1:, 0] = 1.0
                dist_matrix = new_dist
        else:
            dist_matrix = None

        # Module 3: Mamba-3 → [1, |E|, D]
        h_enc = self.encoder(
            h_seq.unsqueeze(0),
            dist_matrix=dist_matrix,
        ).squeeze(0)  # [|E|, D]

        # Module 4: Pool + predict → [1, num_outputs]
        if self.use_virtual_node:
            h_graph = h_enc[0:1, :] # [1, D]
        else:
            h_graph = self.pool(h_enc.unsqueeze(0))  # [1, D]
        return self.head(h_graph)

    def _forward_lrgb_batched(self, batch: Batch) -> torch.Tensor:
        """
        Process a batch of graphs from DataLoader.
        To avoid catastrophic Triton compilation overheads from varying sequence lengths,
        we serialize graphs individually but pad them into a single [B, max_L, D]
        tensor for the Mamba-3 encoder.
        """
        from models.line_graph import get_cached_line_graph_and_dist
        from torch.nn.utils.rnn import pad_sequence

        h_seqs = []
        dist_mats = []
        
        # Unbatch on CPU to avoid CUDA kernel arch mismatches in PyG extensions
        device = batch.x.device
        graphs = batch.cpu().to_data_list()
        
        for data in graphs:
            # Module 1 & 2: Embedding and Serialization per graph
            line_edge_index, orig_edge_index, dist_matrix, x_node, x_edge = get_cached_line_graph_and_dist(data)
            
            # Move tensors to device (graphs are on CPU after unbatching)
            h = self.embed(x_node.to(device), x_edge.to(device), orig_edge_index.to(device))
            h_seq, perm, _ = self.serializer(h, line_edge_index.to(device))
            
            h_seqs.append(h_seq)
            if self.encoder.use_dist and dist_matrix is not None:
                # Reorder distance matrix according to LTAS permutation
                dist_matrix = dist_matrix.to(device)
                if perm is not None:
                    dist_matrix = dist_matrix[perm][:, perm]
                dist_mats.append(dist_matrix)

        # Pad sequences to [B, max_L, D]
        h_padded = pad_sequence(h_seqs, batch_first=True)  # [B, max_L, D]
        B, max_L, _ = h_padded.shape
        
        lengths = torch.tensor([s.shape[0] for s in h_seqs], device=device)
        
        if self.use_virtual_node:
            v_token = self.virtual_token.expand(B, 1, -1)
            h_padded = torch.cat([v_token, h_padded], dim=1)
            max_L += 1
            lengths = lengths + 1
        
        # ── Bucket max_L to nearest multiple of 64 ────────────────────────
        # Triton compiles a new kernel for every unique sequence length.
        # With shuffle=True, each epoch produces different max_L values,
        # causing a recompilation storm (~30-60s per new length).
        # Bucketing limits Triton to ≤10 compiled kernels total.
        import torch.nn.functional as F
        bucketed_L = ((max_L + 63) // 64) * 64
        if bucketed_L > max_L:
            h_padded = F.pad(h_padded, (0, 0, 0, bucketed_L - max_L))
        
        # Create padding mask [B, bucketed_L] (True for valid tokens)
        mask = torch.arange(bucketed_L, device=device).unsqueeze(0) < lengths.unsqueeze(1)
        
        # Distance matrix padding (if used)
        dist_padded = None
        if self.encoder.use_dist and len(dist_mats) == len(graphs):
            dist_padded = torch.zeros((B, bucketed_L, bucketed_L), device=device, dtype=torch.float)
            for i, mat in enumerate(dist_mats):
                l = mat.shape[0]
                if self.use_virtual_node:
                    dist_padded[i, 1:l+1, 1:l+1] = mat
                    dist_padded[i, 0, 1:l+1] = 1.0
                    dist_padded[i, 1:l+1, 0] = 1.0
                else:
                    dist_padded[i, :l, :l] = mat
        
        # Module 3: Mamba-3 batched execution (fixed bucketed lengths for Triton cache)
        h_enc = self.encoder(
            h_padded,
            dist_matrix=dist_padded,
            padding_mask=mask
        )  # [B, bucketed_L, D]
        
        # Module 4: Masked Attention Pooling
        if self.use_virtual_node:
            h_graph = h_enc[:, 0, :]  # [B, D]
        else:
            h_graph = self.pool(h_enc, mask=mask)  # [B, D]
        
        return self.head(h_graph)  # [B, num_outputs]

    # ── RelBench Forward ──────────────────────────────────────────────────────
    def forward_relbench(
        self,
        seq: torch.Tensor,            # [B, L, event_feat_dim]
        delta_t: torch.Tensor,        # [B, L]
        mask: torch.Tensor,           # [B, L] True=valid
    ) -> torch.Tensor:
        """Returns [B, num_outputs] predictions."""
        # Module 1: Project event features → [B, L, D]
        h = self.embed(seq)

        # Module 3: Mamba-3 with temporal encoding
        h_enc = self.encoder(h, delta_t=delta_t, padding_mask=mask)  # [B, L, D]

        # Module 4: Pool + predict
        h_entity = self.pool(h_enc, mask=mask)  # [B, D]
        return self.head(h_entity)              # [B, num_outputs]

    # ── Unified Forward ───────────────────────────────────────────────────────
    def forward(self, *args, **kwargs):
        if self.domain == "lrgb":
            return self.forward_lrgb(*args, **kwargs)
        else:
            return self.forward_relbench(*args, **kwargs)

    def loss(self, pred, target):
        return self.head.loss(pred, target)
