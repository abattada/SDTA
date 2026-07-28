#!/usr/bin/env python3
"""Verify the parameter counts stated in the paper.

Instantiates the shipped SDTA modules at the paper's fixed configuration
(d_model 32, d_ff 64, 4 heads, patch 8, N=12 tokens, input/recon length 96)
and prints:

  * transferred parameters (patch embedding + encoder) at E=1 (Small) and
    E=2 (Medium/Large) — paper: 8,896 and 17,440;
  * the pretraining head at D=1 — paper total: 54,080, decomposed here as
    cross-attention decoder 12,896 + distance code (temporal-distance MLP
    2,112 + noise-step embedding 2,112 = 4,224) + flatten reconstruction
    head 36,960. (The paper's "distance MLP" line item bundles the
    noise-step embedding.)

Baselines transfer the same encoder (identical patch embedding + encoder
classes at the same sizes), so the transferred counts apply to every method.

Usage (from the repository root): python scripts/count_params.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model.SDTA.diffusion import NoiseStepEmbedding
from model.SDTA.encoder import (
    CrossAttentionDecoder,
    PatchEmbedding,
    RelativeLineagePredictor,
    TransformerEncoder,
)
from model.SDTA.forecast_head import ForecastHead

D_MODEL, N_HEADS, D_FF, DROPOUT = 32, 4, 64, 0.2
PATCH, STRIDE, NUM_PATCHES, INPUT_LEN = 8, 8, 12, 96
SAMPLING_MAX, LINEAGE_MLP_LAYERS = 12, 1


def count(module) -> int:
    return sum(p.numel() for p in module.parameters())


def main() -> int:
    patch_emb = PatchEmbedding(PATCH, STRIDE, D_MODEL, DROPOUT)
    print(f"patch embedding: {count(patch_emb):,}")
    for e, label in [(1, "Small (E=1)"), (2, "Medium/Large (E=2)")]:
        enc = TransformerEncoder(D_MODEL, N_HEADS, e, D_FF, DROPOUT)
        total = count(patch_emb) + count(enc)
        print(f"transferred (patch embedding + encoder), {label}: {total:,}")

    dec = CrossAttentionDecoder(D_MODEL, N_HEADS, 1, D_FF, DROPOUT)
    lineage = RelativeLineagePredictor(
        D_MODEL, max_distance=SAMPLING_MAX, max_num_patches=NUM_PATCHES,
        num_layers=LINEAGE_MLP_LAYERS,
    )
    noise_step = NoiseStepEmbedding(D_MODEL)
    recon = ForecastHead(NUM_PATCHES, D_MODEL, INPUT_LEN, dec_layers=1, dropout=0.0)
    head_total = count(dec) + count(lineage) + count(noise_step) + count(recon)
    print(f"pretraining head at D=1: decoder {count(dec):,} "
          f"+ temporal-distance MLP {count(lineage):,} "
          f"+ noise-step embedding {count(noise_step):,} "
          f"+ flatten recon head {count(recon):,} = {head_total:,}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
