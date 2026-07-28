"""Siamese decoder layer: cross-attend to past, then self-attend, then FFN."""
from __future__ import annotations

import torch
import torch.nn as nn


class SiameseDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, current: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        cross, _ = self.cross_attention(current, past, past, need_weights=False)
        current = self.norm1(current + self.dropout(cross))
        attn, _ = self.self_attention(current, current, current, need_weights=False)
        current = self.norm2(current + self.dropout(attn))
        ff_out = self.ff(current)
        return self.norm3(current + self.dropout(ff_out))


class SiameseDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SiameseDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, current: torch.Tensor, past: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            current = layer(current, past)
        return self.norm(current)
