"""Forward diffusion (paper: Method — Pretext: Shifted-Window Diffusion).

q(x_t | x_0) = N(sqrt(gamma_t) x_0, (1 - gamma_t) I) with the cosine schedule
(Nichol & Dhariwal 2021), T = 1000; the step t is drawn independently per
patch (sample_t_per_patch), and NoiseStepEmbedding conditions the decoder
query on t. The model predicts x_0 (not the noise)."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Diffusion(nn.Module):
    """Forward q(x_t | x_0) with cosine or linear beta schedule."""

    def __init__(self, time_steps: int, device: torch.device, scheduler: str = "cosine"):
        super().__init__()
        self.time_steps = time_steps
        if scheduler == "cosine":
            betas = self._cosine_beta_schedule()
        elif scheduler == "linear":
            betas = self._linear_beta_schedule()
        else:
            raise ValueError(f"Invalid scheduler: {scheduler}")
        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alpha", 1 - self.betas)
        self.register_buffer("gamma", torch.cumprod(self.alpha, dim=0))

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        # NOTE: this `s` is the cosine-schedule offset of Nichol & Dhariwal
        # (2021), unrelated to the shifted distance s of the pretext.
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alphas_cumprod = torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(
        self,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
    ) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def sample_t_per_sample(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.time_steps, (batch_size,), device=self.gamma.device)

    def sample_t_per_patch(self, batch_size: int, num_patches: int) -> torch.Tensor:
        return torch.randint(
            0, self.time_steps, (batch_size, num_patches), device=self.gamma.device
        )

    def add_noise(
        self, x: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Broadcast-noise x by gamma[t]. t.shape must be a prefix of x.shape.

        Examples: x=[B,patch_len], t=[B] OR x=[B,N,patch_len], t=[B,N] OR
        x=[B,N,d_model], t=[B,N].
        """
        gamma_t = self.gamma[t]
        while gamma_t.dim() < x.dim():
            gamma_t = gamma_t.unsqueeze(-1)
        noise = torch.randn_like(x)
        noisy = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy, noise


def sinusoidal_t_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """t: [N] integer timesteps. Returns [N, dim] sinusoidal positional features.

    `max_period` is the longest wavelength (lowest frequency). Default 10000 keeps
    the standard Transformer-style embedding; the distance code passes `sampling_max` so
    the low-frequency band spans the actual distance range instead of running
    far past it.
    """
    t = t.float().unsqueeze(-1)
    half_dim = dim // 2
    if half_dim == 0:
        return t.expand(-1, dim)
    div_term = torch.exp(
        torch.arange(half_dim, device=t.device, dtype=torch.float32)
        * (-math.log(max_period) / max(half_dim - 1, 1))
    )
    emb = torch.cat([torch.sin(t * div_term), torch.cos(t * div_term)], dim=-1)
    if emb.size(-1) < dim:
        emb = F.pad(emb, (0, dim - emb.size(-1)))
    return emb


class NoiseStepEmbedding(nn.Module):
    """Sinusoidal embedding of t followed by a 2-layer MLP."""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(sinusoidal_t_embedding(t, self.d_model))
