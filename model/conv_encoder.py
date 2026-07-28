from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EncoderConfig:
    input_len: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    d_ff: int
    e_layers: int
    dropout: float

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


class AbsolutePositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
        activation: str = "gelu",
    ):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        new_x, _ = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model, bias=True)
        self.position_encoding = AbsolutePositionEncoding(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def patchify(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        batch_size, _, num_vars = x.shape
        x = x.permute(0, 2, 1)
        x = x.reshape(batch_size * num_vars, -1)
        patches = x.unfold(-1, self.patch_len, self.stride)
        return patches, num_vars

    def embed_patches(self, patches: torch.Tensor, add_pos: bool = True) -> torch.Tensor:
        embedded = self.value_embedding(patches)
        if add_pos:
            embedded = embedded + self.position_encoding(embedded)
        return self.dropout(embedded)

    def forward(
        self,
        x: torch.Tensor,
        add_pos: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        patches, num_vars = self.patchify(x)
        embedded = self.embed_patches(patches, add_pos=add_pos)
        return embedded, patches, num_vars


class TransformerEncoder(nn.Module):
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
                TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        attn_mask: torch.Tensor | None = None
        if is_causal:
            attn_mask = nn.Transformer.generate_square_subsequent_mask(
                x.size(1), device=x.device
            )
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        return self.norm(x)


def build_encoder(config: EncoderConfig) -> tuple[PatchEmbedding, TransformerEncoder]:
    patch_embedding = PatchEmbedding(
        patch_len=config.patch_len,
        stride=config.stride,
        d_model=config.d_model,
        dropout=config.dropout,
    )
    encoder = TransformerEncoder(
        d_model=config.d_model,
        num_heads=config.n_heads,
        num_layers=config.e_layers,
        feedforward_dim=config.d_ff,
        dropout=config.dropout,
    )
    return patch_embedding, encoder
