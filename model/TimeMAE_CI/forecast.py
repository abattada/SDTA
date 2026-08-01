from __future__ import annotations

import torch
import torch.nn as nn

from model._baseline_forecast import ForecastConfig, denormalize, masked_normalize
from .encoder import ChannelIndependentPatchEmbedding, build_encoder_backbone, num_patches


class TimeMAECIForecaster(nn.Module):
    """TimeMAE_CI finetune forecaster (channel-independent).

    Channel-independent counterpart of official TimeMAE's forecaster. Each channel is a
    separate sequence (`(B,L,V) -> (B*V, num_patches, patch_len)`), patch-embedded with
    `Linear(patch_len->d_model)`, encoded, then a per-channel flatten head maps
    `(num_patches*d_model) -> pred_len` (channel-shared, independent of enc_in) — exactly
    SimMTM / PatchTST / TimeSiam's CI head. This removes the channel-mixing head's
    enc_in-scaling capacity confound (the channel-mixing TimeMAE head is
    `Linear(num_patches*d_model -> pred_len*enc_in)`). Only `patch_embedding` + `encoder`
    carry pretrained weights (shape-matched load); the head trains from scratch.

    Per-window mean/std normalization (`masked_normalize`) kept as the repo convention.
    """

    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        self.pred_len = config.pred_len
        self.patch_embedding = ChannelIndependentPatchEmbedding(
            patch_len=config.patch_len, stride=config.stride, d_model=config.d_model,
        )
        self.encoder = build_encoder_backbone(
            config.d_model, config.n_heads, config.e_layers, config.d_ff, config.dropout
        )
        self.num_patches = num_patches(config.input_len, config.patch_len, config.stride)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(config.head_dropout),
            nn.Linear(self.num_patches * config.d_model, config.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = x.shape
        x_norm, means, stdev = masked_normalize(x)
        raw, _ = self.patch_embedding(x_norm)            # (B*V, P, d_model)
        emb = raw + self.patch_embedding.position(raw)
        enc = self.encoder(emb).reshape(batch_size, num_features, self.num_patches, -1)
        out = self.head(enc).permute(0, 2, 1)            # (B, pred_len, V)
        return denormalize(out, means, stdev)


def build_model(config: ForecastConfig) -> nn.Module:
    return TimeMAECIForecaster(config)
