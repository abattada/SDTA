"""Flatten forecast head with official-style 1x1 Conv projection."""
from __future__ import annotations

import torch
import torch.nn as nn


class ForecastHead(nn.Module):
    """[B, num_features, num_patches, in_dim] -> [B, pred_len, num_features]."""

    def __init__(
        self,
        num_patches: int,
        in_dim: int,
        pred_len: int,
        dec_layers: int,
        dropout: float,
    ):
        super().__init__()
        if dec_layers < 1:
            raise ValueError(f"dec_layers must be >= 1, got {dec_layers}")
        self.flatten = nn.Flatten(start_dim=-2)
        in_features = num_patches * in_dim
        if dec_layers == 1:
            self.forecast_head = nn.Conv1d(in_features, pred_len, kernel_size=1)
        else:
            layers: list[nn.Module] = []
            for _ in range(dec_layers - 1):
                layers.extend([
                    nn.Conv1d(in_features, in_features, kernel_size=1),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Conv1d(in_features, pred_len, kernel_size=1))
            self.forecast_head = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x).transpose(1, 2)
        x = self.forecast_head(x)
        x = self.dropout(x)
        return x
