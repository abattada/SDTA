"""Linear flatten head (paper: Method — Architecture).

Flattens the N x d_model token grid and maps it to the prediction horizon with
a single linear layer (no activation); the same structure is used as the
pretraining reconstruction head and by the baselines for fair comparison."""
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
            self.forecast_head = nn.Linear(in_features, pred_len)
        else:
            layers: list[nn.Module] = []
            for _ in range(dec_layers - 1):
                layers.extend([
                    nn.Linear(in_features, in_features),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])
            layers.append(nn.Linear(in_features, pred_len))
            self.forecast_head = nn.Sequential(*layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.forecast_head(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)
