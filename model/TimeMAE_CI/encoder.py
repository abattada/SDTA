"""TimeMAE_CI encoder pieces.

Channel-INDEPENDENT port of the official TimeMAE encoder. The decoupled-MAE
pretext is unchanged; only the input projection differs: instead of the official
TimeMAE channel-mixing `Conv1d(enc_in->d_model)`, each channel is treated as its
own sequence (`(B,L,V) -> (B*V, num_patches, patch_len)`) and patch-embedded with a
shared `Linear(patch_len->d_model)` — exactly the channel-independent patch front-end
used by PatchTST / TimeSiam / SimMTM (`model/conv_encoder.PatchEmbedding`). This makes
TimeMAE_CI apples-to-apples with those shared-encoder baselines, so a TimeMAE_CI-vs-them
comparison isolates the SSL objective (channel structure + head capacity confounds
removed). The channel-mixing front-end this replaces is the official TimeMAE one
(see THIRD_PARTY.md for the upstream repository).

The transformer backbone reuses the repo's shared `conv_encoder.TransformerEncoder`.
TimeMAE-specific pretrain modules (tokenizer / cross-attention regressor / momentum
encoder) live in `pretrain.py` (pretrain-only) and are identical to TimeMAE.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model.conv_encoder import AbsolutePositionEncoding, TransformerEncoder


def num_patches(input_len: int, patch_len: int, stride: int) -> int:
    return int((input_len - patch_len) / stride) + 1


class ChannelIndependentPatchEmbedding(nn.Module):
    """Channel-independent patch projection: `Linear(patch_len -> d_model)` applied
    per channel (channel folded into the batch dim), no cross-channel mixing.

    Input  x: (B, L, V)
    Output  : ((B*V), num_patches, d_model)  — no position added (caller adds it),
              and the channel count V (so the forecaster can unfold back to (B, ., V)).
    """

    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=True)
        self.position = AbsolutePositionEncoding(d_model)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch_size, _, num_vars = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * num_vars, -1)
        patches = x.unfold(-1, self.patch_len, self.stride)  # (B*V, num_patches, patch_len)
        return self.value_embedding(patches), num_vars


def build_encoder_backbone(d_model: int, n_heads: int, e_layers: int, d_ff: int, dropout: float) -> TransformerEncoder:
    return TransformerEncoder(
        d_model=d_model, num_heads=n_heads, num_layers=e_layers,
        feedforward_dim=d_ff, dropout=dropout,
    )


__all__ = ["num_patches", "ChannelIndependentPatchEmbedding", "build_encoder_backbone"]
