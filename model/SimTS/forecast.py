from __future__ import annotations

import torch
import torch.nn as nn

from model._baseline_forecast import ForecastConfig, denormalize, masked_normalize
from .encoder import EncoderConfig, build_encoder


class SimTSForecaster(nn.Module):
    """SimTS finetune model with repo shared patch encoder and flatten head.

    Identical downstream protocol to SimMTM / SDTA: load only the pretrained
    `patch_embedding` + `encoder` weights, attach a fresh flatten forecast head,
    gradient-finetune the whole stack. The SimTS predictor (pretrain-only) is
    discarded here, just as SDTA discards its diffusion decoder / lineage.
    """

    def __init__(self, config: ForecastConfig):
        super().__init__()
        self.config = config
        enc_cfg = EncoderConfig(
            input_len=config.input_len,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            e_layers=config.e_layers,
            dropout=config.dropout,
        )
        self.patch_embedding, self.encoder = build_encoder(enc_cfg)
        self.num_patches = enc_cfg.num_patches
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(config.head_dropout),
            nn.Linear(self.num_patches * config.d_model, config.pred_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = x.shape
        x_norm, means, stdev = masked_normalize(x)
        emb, _, _ = self.patch_embedding(x_norm, add_pos=True)
        enc = self.encoder(emb).reshape(batch_size, num_features, self.num_patches, -1)
        out = self.head(enc).permute(0, 2, 1)
        return denormalize(out, means, stdev)


def build_model(config: ForecastConfig) -> nn.Module:
    return SimTSForecaster(config)
