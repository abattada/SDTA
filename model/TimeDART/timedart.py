"""TimeDART architecture adapted to the local SDTA-style pipeline."""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn as nn


def generate_causal_mask(seq_len: int, device: torch.device | str | None = None) -> torch.Tensor:
    return torch.triu(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
    )


def generate_partial_mask(
    seq_len: int, mask_ratio: float, device: torch.device | str | None = None
) -> torch.Tensor:
    mask = generate_causal_mask(seq_len, device=device)
    lower_indices = torch.tril_indices(seq_len, seq_len, offset=-1, device=device)
    num_lower = lower_indices.shape[1]
    num_to_mask = int(num_lower * mask_ratio)
    if num_to_mask:
        mask_indices = torch.randperm(num_lower, device=device)[:num_to_mask]
        mask[lower_indices[0][mask_indices], lower_indices[1][mask_indices]] = True
    return mask


class ChannelIndependence(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, input_len, _ = x.shape
        x = x.permute(0, 2, 1)
        return x.reshape(-1, input_len, 1)


class Patch(nn.Module):
    def __init__(self, patch_len: int, stride: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(-1)
        return x.unfold(-1, self.patch_len, self.stride)


class PatchEmbedding(nn.Module):
    def __init__(self, patch_len: int, d_model: int):
        super().__init__()
        self.patch_embedding = nn.Linear(patch_len, d_model, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.patch_embedding(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(x + self.pe[:, : x.size(1)])


class AddSosTokenAndDropLast(nn.Module):
    def __init__(self, sos_token: torch.Tensor):
        super().__init__()
        self.sos_token = sos_token

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sos_token_expanded = self.sos_token.expand(x.size(0), -1, -1)
        return torch.cat([sos_token_expanded, x], dim=1)[:, :-1, :]


class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


class CausalTransformer(nn.Module):
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

    def forward(self, x: torch.Tensor, is_mask: bool = True) -> torch.Tensor:
        mask = generate_causal_mask(x.size(1), device=x.device) if is_mask else None
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Diffusion(nn.Module):
    def __init__(self, time_steps: int, device: torch.device, scheduler: str = "cosine"):
        super().__init__()
        self.device = device
        self.time_steps = time_steps
        if scheduler == "cosine":
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif scheduler == "linear":
            self.betas = self._linear_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")
        self.alpha = 1 - self.betas
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start: float = 1e-4, beta_end: float = 0.02) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def sample_time_steps(self, shape: torch.Size) -> torch.Tensor:
        return torch.randint(0, self.time_steps, shape, device=self.device)

    def noise(self, x: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1)
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = self.sample_time_steps(x.shape[:2])
        noisy_x, noise = self.noise(x, t)
        return noisy_x, noise, t


class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.encoder_attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
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

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        tgt_mask: torch.Tensor | None,
        src_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        attn_output, _ = self.self_attention(query, query, query, attn_mask=tgt_mask)
        query = self.norm1(query + self.dropout(attn_output))
        attn_output, _ = self.encoder_attention(query, key, value, attn_mask=src_mask)
        query = self.norm2(query + self.dropout(attn_output))
        ff_output = self.ff(query)
        return self.norm3(query + self.dropout(ff_output))


class DenoisingPatchDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        mask_ratio: float,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.mask_ratio = mask_ratio

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_tgt_mask: bool = True,
        is_src_mask: bool = True,
    ) -> torch.Tensor:
        seq_len = query.size(1)
        tgt_mask = (
            generate_partial_mask(seq_len, self.mask_ratio, device=query.device)
            if is_tgt_mask
            else None
        )
        src_mask = (
            generate_partial_mask(seq_len, self.mask_ratio, device=query.device)
            if is_src_mask
            else None
        )
        for layer in self.layers:
            query = layer(query, key, value, tgt_mask, src_mask)
        return self.norm(query)


class FlattenHead(nn.Module):
    def __init__(self, seq_len: int, d_model: int, pred_len: int, dropout: float):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.forecast_head = nn.Linear(seq_len * d_model, pred_len)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.forecast_head(x)
        x = self.dropout(x)
        return x.permute(0, 2, 1)


class TimeDARTModel(nn.Module):
    """Official TimeDART Transformer backbone, fitted to local data loaders."""

    def __init__(self, args: SimpleNamespace):
        super().__init__()
        self.input_len = args.input_len
        self.d_model = args.d_model
        self.device = args.device
        self.task_name = args.task_name
        self.pred_len = args.pred_len
        self.use_norm = bool(args.use_norm)
        self.patch_len = args.patch_len
        self.stride = args.stride
        self.seq_len = int((self.input_len - self.patch_len) / self.stride) + 1
        self.channel_independence = ChannelIndependence()
        self.patch = Patch(patch_len=self.patch_len, stride=self.stride)
        self.enc_embedding = PatchEmbedding(patch_len=self.patch_len, d_model=self.d_model)
        self.positional_encoding = PositionalEncoding(d_model=self.d_model, dropout=args.dropout)
        sos_token = torch.randn(1, 1, self.d_model, device=self.device)
        self.sos_token = nn.Parameter(sos_token, requires_grad=True)
        self.add_sos_token_and_drop_last = AddSosTokenAndDropLast(self.sos_token)
        self.encoder = CausalTransformer(
            d_model=args.d_model,
            num_heads=args.n_heads,
            feedforward_dim=args.d_ff,
            dropout=args.dropout,
            num_layers=args.e_layers,
        )
        self.diffusion = Diffusion(
            time_steps=args.time_steps,
            device=args.device,
            scheduler=args.scheduler,
        )
        if self.task_name == "pretrain":
            self.denoising_patch_decoder = DenoisingPatchDecoder(
                d_model=args.d_model,
                num_layers=args.d_layers,
                num_heads=args.n_heads,
                feedforward_dim=args.d_ff,
                dropout=args.dropout,
                mask_ratio=args.mask_ratio,
            )
            self.projection = FlattenHead(
                seq_len=self.seq_len,
                d_model=self.d_model,
                pred_len=args.input_len,
                dropout=args.head_dropout,
            )
        elif self.task_name == "finetune":
            self.head = FlattenHead(
                seq_len=self.seq_len,
                d_model=args.d_model,
                pred_len=args.pred_len,
                dropout=args.head_dropout,
            )
        else:
            raise ValueError("task_name should be 'pretrain' or 'finetune'")

    def _normalize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        means = torch.mean(x, dim=1, keepdim=True).detach()
        x = x - means
        stdevs = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return x / stdevs, means, stdevs

    def pretrain(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, input_len, num_features = x.size()
        if self.use_norm:
            x, means, stdevs = self._normalize(x)
        else:
            means = stdevs = None

        x = self.channel_independence(x)
        x_patch = self.patch(x)
        x_embedding = self.enc_embedding(x_patch)
        x_embedding_bias = self.add_sos_token_and_drop_last(x_embedding)
        x_embedding_bias = self.positional_encoding(x_embedding_bias)
        x_out = self.encoder(x_embedding_bias, is_mask=True)

        noise_x_patch, _, _ = self.diffusion(x_patch)
        noise_x_embedding = self.enc_embedding(noise_x_patch)
        noise_x_embedding = self.positional_encoding(noise_x_embedding)
        predict_x = self.denoising_patch_decoder(
            query=noise_x_embedding,
            key=x_out,
            value=x_out,
            is_tgt_mask=True,
            is_src_mask=True,
        )
        predict_x = predict_x.reshape(batch_size, num_features, -1, self.d_model)
        predict_x = self.projection(predict_x)

        if means is not None and stdevs is not None:
            predict_x = predict_x * stdevs[:, 0, :].unsqueeze(1).repeat(1, input_len, 1)
            predict_x = predict_x + means[:, 0, :].unsqueeze(1).repeat(1, input_len, 1)
        return predict_x

    def forecast(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, num_features = x.size()
        if self.use_norm:
            x, means, stdevs = self._normalize(x)
        else:
            means = stdevs = None

        x = self.channel_independence(x)
        x = self.patch(x)
        x = self.enc_embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, is_mask=False)
        x = x.reshape(batch_size, num_features, -1, self.d_model)
        x = self.head(x)

        if means is not None and stdevs is not None:
            x = x * stdevs[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            x = x + means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
        return x

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:
        if self.task_name == "pretrain":
            return self.pretrain(batch_x)
        return self.forecast(batch_x)[:, -self.pred_len :, :]
