from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_pair_features(query_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
    """Build [sim, dot, abs_diff, hadamard] features for reranker."""
    if query_emb.ndim == 1:
        query_emb = query_emb.unsqueeze(0)
    if product_emb.ndim == 1:
        product_emb = product_emb.unsqueeze(0)

    q = F.normalize(query_emb, dim=-1)
    p = F.normalize(product_emb, dim=-1)

    dot = (q * p).sum(dim=-1, keepdim=True)
    sim = F.cosine_similarity(q, p, dim=-1).unsqueeze(-1)
    abs_diff = torch.abs(q - p)
    hadamard = q * p
    return torch.cat([sim, dot, abs_diff, hadamard], dim=-1)


class MLPReRanker(nn.Module):
    def __init__(self, embedding_dim: int, hidden_dim: int = 512, dropout: float = 0.2) -> None:
        super().__init__()
        in_dim = 2 + 2 * embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)
