"""
HGT (Heterogeneous Graph Transformer) for edge classification in OMR.

Architecture
------------
  node features  →  per-type linear projection
                 →  HGTConv × num_layers
                 →  edge MLP([h_u ‖ h_v ‖ edge_attr])  →  binary logit

Replacing the rule-based engine
--------------------------------
  After training, swap `RuleBasedGraphInferenceEngine` in pipeline.py with
  `TrainedHeteroGNNEngine` (see bottom of this file).

Optional dependency — install with:
    pip install "xunpu-graph-omr[gnn]"
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from xunpu_graph_omr.graph.muscima_dataset import (
    CANDIDATE_PAIRS,
    EDGE_FEAT_DIM,
    NODE_FEAT_DIM,
    NODE_TYPES,
)


# --------------------------------------------------------------------------- #
# Model                                                                        #
# --------------------------------------------------------------------------- #

class _EdgeHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        in_dim = hidden_dim * 2 + EDGE_FEAT_DIM
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([h_src, h_dst, edge_attr], dim=-1))


class OMRHeteroGNN(nn.Module):
    """
    Parameters
    ----------
    hidden_dim : int
        Embedding dimension for all node types (uniform for HGTConv).
    num_layers : int
        Number of stacked HGTConv layers.
    num_heads : int
        Attention heads per HGTConv layer (must divide hidden_dim).
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        self.input_proj = nn.ModuleDict(
            {ntype: nn.Linear(NODE_FEAT_DIM, hidden_dim) for ntype in NODE_TYPES}
        )

        relations = [(s, "to", d) for s, d in CANDIDATE_PAIRS]
        relations += [(d, "rev_to", s) for s, d in CANDIDATE_PAIRS]
        metadata = (NODE_TYPES, relations)
        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata, num_heads) for _ in range(num_layers)]
        )

        self.edge_heads = nn.ModuleDict(
            {f"{s}__{d}": _EdgeHead(hidden_dim) for s, d in CANDIDATE_PAIRS}
        )

    # ------------------------------------------------------------------ #

    def encode(self, data) -> dict[str, torch.Tensor]:
        x_dict = {
            ntype: F.relu(self.input_proj[ntype](data[ntype].x))
            for ntype in NODE_TYPES
            if data[ntype].x.size(0) > 0
        }
        edge_index_dict = {
            (s, "to", d): data[(s, "to", d)].edge_index
            for s, d in CANDIDATE_PAIRS
            if data[(s, "to", d)].edge_index.size(1) > 0
        }
        for src_type, dst_type in CANDIDATE_PAIRS:
            forward = (src_type, "to", dst_type)
            if forward in edge_index_dict:
                edge_index_dict[(dst_type, "rev_to", src_type)] = edge_index_dict[forward].flip(0)
        if not edge_index_dict:
            return x_dict
        for conv in self.convs:
            updates = conv(x_dict, edge_index_dict)
            x_dict = {
                key: F.relu(updates[key]) if updates.get(key) is not None else value
                for key, value in x_dict.items()
            }
        return x_dict

    def forward(self, data) -> dict[tuple[str, str], torch.Tensor]:
        """Return logits dict keyed by (src_type, dst_type)."""
        x_dict = self.encode(data)
        logits: dict[tuple[str, str], torch.Tensor] = {}
        for src_type, dst_type in CANDIDATE_PAIRS:
            rel = (src_type, "to", dst_type)
            edge_index = data[rel].edge_index
            if edge_index.size(1) == 0:
                continue
            h_src = x_dict.get(src_type)
            h_dst = x_dict.get(dst_type)
            if h_src is None or h_dst is None:
                continue
            logits[(src_type, dst_type)] = self.edge_heads[f"{src_type}__{dst_type}"](
                h_src[edge_index[0]],
                h_dst[edge_index[1]],
                data[rel].edge_attr,
            )
        return logits


# --------------------------------------------------------------------------- #
# Bridge back to the project's own HeteroGraph / pipeline                     #
# --------------------------------------------------------------------------- #

def load_model(checkpoint: str | Path, device: torch.device | None = None, **kwargs) -> OMRHeteroGNN:
    """Load a trained OMRHeteroGNN from a checkpoint saved by train_hetero.py."""
    device = device or torch.device("cpu")
    model = OMRHeteroGNN(**kwargs)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    return model.to(device)
