"""SDTA backbone (paper: Method — Architecture, Temporal-Distance Attention).

Channel-independent PatchEmbedding (patch 8, stride 8), causal
TransformerEncoder, and the causal-masked CrossAttentionDecoder (separate
K/V so the key-side distance code never contaminates values).
RelativeLineagePredictor is the paper's temporal-distance attention:
l(p, s) = MLP(concat[SE(p), SE(s)]) varies along the key-position axis, which
breaks the softmax shift-invariance that cancels any broadcast (per-window
constant) code — the paper's motivation for a position-dependent code.
generate_partial_mask implements the decoder mask (causal + random drop of
mask_ratio of the admissible past key positions)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .diffusion import sinusoidal_t_embedding


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
    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        return self.norm2(x + self.dropout(ff_output))


class PatchEmbedding(nn.Module):
    """[B, T, C] -> raw patches and per-patch d_model embeddings."""

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


def _generate_causal_mask(seq_len: int, device: torch.device | str) -> torch.Tensor:
    """Upper triangular bool mask (True = masked); position i cannot attend to j > i."""
    return torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)


def generate_partial_mask(
    q_len: int,
    k_len: int,
    mask_ratio: float,
    device: torch.device | str,
) -> torch.Tensor:
    """TimeDART-style attention mask: causal + extra random masks in lower triangle.

    `mask_ratio = 0` returns pure causal mask (lower triangle + diagonal visible).
    `mask_ratio = 1.0` masks the entire lower triangle, leaving only the diagonal
    (each position only sees itself). Values in between sample that fraction of
    lower-triangle links to mask, sampled fresh per call.

    Requires `q_len == k_len`: the decoder's cross-attention always has Q from
    the target's N patches and K from the anchor's N patches.
    """
    assert q_len == k_len, f"partial mask requires q_len==k_len, got {q_len}!={k_len}"
    mask = _generate_causal_mask(q_len, device)
    if mask_ratio > 0:
        lower_indices = torch.tril_indices(q_len, q_len, offset=-1, device=device)
        num_lower = lower_indices.shape[1]
        num_to_mask = int(num_lower * mask_ratio)
        if num_to_mask:
            perm = torch.randperm(num_lower, device=device)[:num_to_mask]
            mask[lower_indices[0][perm], lower_indices[1][perm]] = True
    return mask


class TransformerDecoderBlock(nn.Module):
    """self-attn(query) -> cross-attn(query, key, value) -> FFN, post-norm.

    Cross-attention supports K≠V so a per-position routing bias (e.g. lineage)
    can be added on the key side without contaminating the aggregated content.
    """

    def __init__(self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
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
        kv_key: torch.Tensor,
        kv_value: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        cross_attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sa, _ = self.self_attn(
            query, query, query, attn_mask=self_attn_mask, need_weights=False
        )
        query = self.norm1(query + self.dropout(sa))
        ca, _ = self.cross_attn(
            query, kv_key, kv_value, attn_mask=cross_attn_mask, need_weights=False
        )
        query = self.norm2(query + self.dropout(ca))
        ff = self.ff(query)
        return self.norm3(query + self.dropout(ff))


class CrossAttentionDecoder(nn.Module):
    """Stacked decoder blocks. query=[B, Nq, D], key/value=[B, Nk, D] -> [B, Nq, D].

    Cross-attention is K≠V to keep routing-side bias separate from content.
    `mask_ratio > 0` enables MAE-style random partial mask shared between
    self-attn and cross-attn within a forward call (re-sampled each call).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        mask_ratio: float = 0.0,
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.layers = nn.ModuleList(
            [
                TransformerDecoderBlock(d_model, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        query: torch.Tensor,
        kv_key: torch.Tensor,
        kv_value: torch.Tensor,
    ) -> torch.Tensor:
        q_len = query.size(1)
        k_len = kv_key.size(1)
        self_attn_mask = generate_partial_mask(q_len, q_len, self.mask_ratio, query.device)
        cross_attn_mask = generate_partial_mask(q_len, k_len, self.mask_ratio, query.device)
        for layer in self.layers:
            query = layer(
                query, kv_key, kv_value,
                self_attn_mask=self_attn_mask,
                cross_attn_mask=cross_attn_mask,
            )
        return self.norm(query)


class LineagePredictor(nn.Module):
    """Distance-only lineage encoder. Output broadcasts over all N positions.

    Pipeline: sinusoidal_t_embedding(distance, d_model, max_period) -> stacked
    GeLU MLP -> [B*C, d_model]. The caller is responsible for broadcasting /
    unsqueezing over N before adding to past_enc on the decoder key side.

    `max_period` caps the lowest frequency to the actual distance range
    (typically `sampling_max`), so the sinusoidal band is not wasted on
    distances far outside the sample distribution.

    No content-awareness (does not read past_enc), no learnable position basis.
    """

    def __init__(self, d_model: int, max_period: float, num_layers: int = 2, dropout: float = 0.0):
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        if max_period <= 1:
            raise ValueError(f"max_period must be > 1, got {max_period}")
        self.d_model = d_model
        self.max_period = float(max_period)
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(d_model, d_model)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """distance: [B*C] (or any flat shape) -> [..., d_model]. Broadcast at call site."""
        emb = sinusoidal_t_embedding(distance, self.d_model, max_period=self.max_period)
        return self.mlp(emb)


class LineageTokenEmbedding(nn.Module):
    """Broadcast-token ablation: discrete TimeSiam-style tokens, indexed by the
    integer patch-unit shift s. Counterpart to LineagePredictor; same output
    shape so the caller can swap the two without changing the cross-attention
    path.

    `num_buckets` should be `sampling_max`; the table size is `num_buckets + 1`
    so index 0 is reserved (s starts at 1).
    """

    def __init__(self, num_buckets: int, d_model: int):
        super().__init__()
        if num_buckets < 1:
            raise ValueError(f"num_buckets must be >= 1, got {num_buckets}")
        self.num_buckets = int(num_buckets)
        self.d_model = d_model
        self.emb = nn.Embedding(num_buckets + 1, d_model)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        return self.emb(distance.long())


class RelativeLineagePredictor(nn.Module):
    """Temporal-distance attention: the position-dependent distance code.

    Background. Building the code from the distance alone and broadcasting a
    single d-dim vector over all N past-key positions is provably inert:
        K[b, p, :] = past_enc[b, p, :] + lineage[b]            (same lineage for every p)
        Q · K[b, p, :] = Q · past_enc[b, p, :] + Q · lineage[b]
                                                 └── constant in p ──┘
        softmax_p(Q · K) is shift-invariant in p, so the lineage term is
        analytically cancelled and the residual has zero effect on the
        attention weights. This is why a broadcast code measures as no better
        than removing the code altogether: it is structurally inert, not
        capacity-limited.

    Fix. Make the lineage depend on BOTH the key position p and the shift s so
    it varies along the K axis. Concretely (the paper's temporal-distance code):
        lineage[b, p, :] = MLP(concat(SE(p), SE(s)))
        K[b, p, :] = past_enc[b, p, :] + lineage[b, p, :]
        Q · K[b, p, :] = Q · past_enc[b, p, :] + Q · lineage[b, p, :]
                                                 └── varies in p ──┘
    The lineage term is no longer constant over p → softmax weights actually
    respond → lineage can contribute to routing.

    Output shape is [B*W*C, N, d_model] (caller does NOT need to unsqueeze /
    broadcast — the predictor already emits a per-K-position vector).
    """

    def __init__(
        self,
        d_model: int,
        max_distance: float,
        max_num_patches: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if num_layers < 0:
            raise ValueError(f"num_layers must be >= 0, got {num_layers}")
        if max_distance <= 1:
            raise ValueError(f"max_distance must be > 1, got {max_distance}")
        if max_num_patches < 1:
            raise ValueError(f"max_num_patches must be >= 1, got {max_num_patches}")
        if d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even for RelativeLineagePredictor (half goes "
                f"to the position code, half to the shift code); got {d_model}"
            )
        self.d_model = d_model
        self.max_distance = float(max_distance)
        # Use max(max_num_patches, 2) as the sinusoidal period for p so the
        # smallest meaningful period covers ≥ 2 patches.
        self.max_num_patches = max(int(max_num_patches), 2)
        self.half = d_model // 2
        layers: list[nn.Module] = []
        for _ in range(num_layers):
            layers += [nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)]
        layers += [nn.Linear(d_model, d_model)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, distance: torch.Tensor, num_patches: int) -> torch.Tensor:
        """Temporal-distance code l(p, s) = MLP(concat[SE(p), SE(s)]).

        `distance` carries the shifted distance s (patch units), one per window;
        `num_patches` gives the key positions p = 0..N-1.
        distance: [B] (or any flat shape), num_patches: int → [B, N, d_model].
        """
        if num_patches < 1:
            raise ValueError(f"num_patches must be >= 1, got {num_patches}")
        flat = distance.reshape(-1)                                            # [B]
        B = flat.size(0)
        device = flat.device

        # Two sinusoidal embeddings (half d_model each), concatenated.
        p_idx = torch.arange(num_patches, device=device, dtype=torch.float32)  # [N]
        p_emb = sinusoidal_t_embedding(
            p_idx, self.half, max_period=float(self.max_num_patches)
        )                                                                      # [N, half]
        d_emb = sinusoidal_t_embedding(
            flat, self.half, max_period=self.max_distance
        )                                                                      # [B, half]

        p_bcast = p_emb.unsqueeze(0).expand(B, -1, -1)                         # [B, N, half]
        d_bcast = d_emb.unsqueeze(1).expand(-1, num_patches, -1)               # [B, N, half]
        combined = torch.cat([p_bcast, d_bcast], dim=-1)                       # [B, N, d_model]
        return self.mlp(combined)                                              # [B, N, d_model]


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


def build_decoder(
    d_model: int,
    n_heads: int,
    num_layers: int,
    d_ff: int,
    dropout: float,
    mask_ratio: float = 0.0,
) -> CrossAttentionDecoder:
    return CrossAttentionDecoder(
        d_model=d_model,
        num_heads=n_heads,
        num_layers=num_layers,
        feedforward_dim=d_ff,
        dropout=dropout,
        mask_ratio=mask_ratio,
    )
