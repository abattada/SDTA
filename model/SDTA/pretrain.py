"""SDTA pretraining stage (paper: Method — Pretext, Architecture, Training Objective).

PretrainModel.forward implements the paper's pretext end to end: the anchor
window is encoded once by the causal Transformer encoder; each of the W target
windows is corrupted by forward diffusion (per-patch step t) and denoised by
the cross-attention decoder conditioned on the shared anchor encoding, with the
temporal-distance code added to attention KEYS only (kv_key = encoding +
lineage, kv_value = encoding); the loss is a single MSE on the x0-space
reconstruction. Ablation arms of the paper map to CLI flags — see
docs/paper_code_map.md. lineage_type: "relative" is the paper's
temporal-distance attention; "learnable_token" is the discrete broadcast-token
ablation; "sinusoidal" is the distance-only broadcast code shown inert by the
softmax shift-invariance analysis."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from model._test_io import write_done
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Large current_views (up to 16) make cross-attention see
# batch dim B*W*C = 16*16*862 ≈ 220k on traffic, which overflows the SDPA
# efficient kernel's seed/offset tensor (limit 65535). Disable the efficient
# backend module-wide; flash + math both handle large batches.
torch.backends.cuda.enable_mem_efficient_sdp(False)

from .cli import apply_cuda_visible_devices
from .dataset import SiamesePretrainDataset
from .diffusion import Diffusion, NoiseStepEmbedding
from .encoder import (
    EncoderConfig,
    LineagePredictor,
    LineageTokenEmbedding,
    RelativeLineagePredictor,
    build_decoder,
    build_encoder,
)
from .forecast_head import ForecastHead
from .utils import PRETRAIN_CHECKPOINT_ROOT, resolve_device, seed_everything


@dataclass
class PretrainConfig:
    dataset_name: str
    data_fine_dir: str
    model_id: str
    run_name: str
    features: str
    input_len: int
    enc_in: int
    patch_len: int
    stride: int
    d_model: int
    n_heads: int
    e_layers: int
    d_layers: int
    d_ff: int
    dropout: float
    use_norm: bool
    time_steps: int
    scheduler: str
    sampling_min: int
    sampling_max: int
    # Diffusion target space. "patch" adds noise to raw cur_patches; "latent"
    # adds noise to cur_emb after running through the encoder (detached).
    diffusion_space: str
    learning_rate: float
    batch_size: int
    train_epochs: int
    num_workers: int
    checkpoint_every: int
    # ExponentialLR gamma for pretraining (aligned with TimeDART).
    lr_decay: float = 0.5
    # Optional MAE-style partial mask on decoder self/cross attention.
    # 0.0 disables; >0 randomly masks that fraction of attention links per call.
    mask_ratio: float = 0.0
    # Hidden GeLU blocks in the shared distance-MLP.
    # 0 = degenerate (input_proj + final linear only); 2 is the default.
    lineage_mlp_layers: int = 2
    # W: target windows paired with each anchor per forward pass (the paper's
    # supervision axis). 1 = single target; >1 amortizes the anchor encoding.
    current_views: int = 1
    # Dropout inside the flatten reconstruction head.
    head_dropout: float = 0.0
    pretrain_causal: bool = True
    # Default: lineage_type="relative" — the paper's temporal-distance code,
    # position-dependent so it survives the softmax (see RelativeLineagePredictor).
    # "sinusoidal" = distance-only broadcast code (provably inert), kept for ablation.
    # "learnable_token" = TimeSiam-style discrete broadcast token.
    lineage_type: str = "relative"               # "relative" / "sinusoidal" / "learnable_token"
    kv_share_lineage: bool = False               # K=V: 1 = code also added to V
    forced_first_shift_one: bool = True          # 0 = the first target is also random
    lineage_disabled: bool = False               # no-TDA: 1 = skip the code entirely
    diffusion_noise_disabled: bool = False       # no-diffusion: 1 = clean target, no noise
    past_disabled: bool = False                  # no-anchor: 1 = learnable null token
    # Memory knob (orthogonal to ablations). 0 = no chunking (forward all W
    # views at once, batch dim B*W*C). >0 = chunk W into groups
    # of this size; forward per chunk, backward per chunk (gradient accumulates
    # into .grad), single optimizer.step() per data batch. Each chunk loss is
    # scaled by (chunk / W) so the accumulated gradient is mathematically
    # identical to processing all W in one shot. Use to fit large B*W*C into
    # VRAM (e.g., traffic + W=16 stress).
    w_chunk_size: int = 0
    device: str = "auto"
    seed: int = 2021

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


def per_window_normalize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    means = x.mean(dim=1, keepdim=True).detach()
    centered = x - means
    stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
    return centered / stdev, means, stdev


def denormalize(x: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
    return x * stdev + means


class PretrainModel(nn.Module):
    """Shared encoder + temporal-distance code + cross-attention decoder."""

    def __init__(self, config: PretrainConfig):
        super().__init__()
        if config.d_layers < 1:
            raise ValueError(f"d_layers must be >= 1, got {config.d_layers}")
        if config.diffusion_space not in {"patch", "latent"}:
            raise ValueError(
                f"diffusion_space must be 'patch' or 'latent', got {config.diffusion_space!r}"
            )
        if config.sampling_min < 0:
            raise ValueError(f"sampling_min must be >= 0, got {config.sampling_min}")
        if config.sampling_max < config.sampling_min:
            raise ValueError(
                f"sampling_max must be >= sampling_min, got "
                f"{config.sampling_max} < {config.sampling_min}"
            )
        if config.lineage_mlp_layers < 0:
            raise ValueError(
                f"lineage_mlp_layers must be >= 0, got {config.lineage_mlp_layers}"
            )
        if config.current_views < 1:
            raise ValueError(
                f"current_views must be >= 1, got {config.current_views}"
            )
        if config.lineage_type not in {"relative", "sinusoidal", "learnable_token"}:
            raise ValueError(
                f"lineage_type must be one of 'relative' (default), "
                f"'sinusoidal' (distance-only broadcast), or 'learnable_token'; "
                f"got {config.lineage_type!r}"
            )
        resolved_device = resolve_device(config.device)
        config.device = str(resolved_device)
        self.device = resolved_device
        self.config = config

        encoder_config = EncoderConfig(
            input_len=config.input_len,
            patch_len=config.patch_len,
            stride=config.stride,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            e_layers=config.e_layers,
            dropout=config.dropout,
        )
        self.patch_embedding, self.encoder = build_encoder(encoder_config)
        self.decoder = build_decoder(
            d_model=config.d_model,
            n_heads=config.n_heads,
            num_layers=config.d_layers,
            d_ff=config.d_ff,
            dropout=config.dropout,
            mask_ratio=config.mask_ratio,
        )

        self.diffusion = Diffusion(
            time_steps=config.time_steps,
            device=self.device,
            scheduler=config.scheduler,
        )
        self.noise_embedding = NoiseStepEmbedding(d_model=config.d_model)

        # Default: position-dependent code (RelativeLineagePredictor), output
        # [B, N, d_model] — varies along the key-position axis so softmax
        # routing is NOT shift-invariant under it. The broadcast
        # variants ("sinusoidal", "learnable_token") are
        # retained behind lineage_type for the ablations; both emit
        # [B, d_model] and the forward path unsqueezes them to broadcast.
        if config.lineage_type == "relative":
            self.lineage_predictor = RelativeLineagePredictor(
                d_model=config.d_model,
                max_distance=max(2.0, float(config.sampling_max)),
                max_num_patches=int(config.num_patches),
                num_layers=config.lineage_mlp_layers,
                dropout=config.dropout,
            )
        elif config.lineage_type == "sinusoidal":
            self.lineage_predictor = LineagePredictor(
                d_model=config.d_model,
                max_period=max(2.0, float(config.sampling_max)),
                num_layers=config.lineage_mlp_layers,
                dropout=config.dropout,
            )
        else:  # learnable_token
            self.lineage_predictor = LineageTokenEmbedding(
                num_buckets=config.sampling_max,
                d_model=config.d_model,
            )

        # no-anchor ablation: replace past_enc with a learnable null token broadcast
        # over (B*W*C, N) positions when past_disabled=1.
        if config.past_disabled:
            self.past_null = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            self.register_parameter("past_null", None)

        # TimeDART-style full-window flatten reconstruction head.
        # decoder output [B*W*C, N, d] -> reshape [B*W, C, N, d] -> [B*W, input_len, C].
        self.recon_head = ForecastHead(
            num_patches=config.num_patches,
            in_dim=config.d_model,
            pred_len=config.input_len,
            dec_layers=1,
            dropout=config.head_dropout,
        )

        self.to(self.device)

    def forward(
        self,
        past: torch.Tensor,
        currents: torch.Tensor,
        distances: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        past = past.to(self.device)
        currents = currents.to(self.device)
        distances = distances.to(self.device)
        batch_size, input_len, num_features = past.shape
        if input_len != self.config.input_len:
            raise ValueError(f"Expected input_len={self.config.input_len}, got {input_len}")
        if num_features != self.config.enc_in:
            raise ValueError(f"Expected enc_in={self.config.enc_in}, got {num_features}")

        # Accept (B, L, C)/(B,) for back-compat single-current callers; canonical
        # shape is (B, W, L, C) / (B, W).
        if currents.dim() == 3:
            currents = currents.unsqueeze(1)
        if distances.dim() == 1:
            distances = distances.unsqueeze(1)
        if currents.shape[0] != batch_size or currents.shape[2:] != (input_len, num_features):
            raise ValueError(
                f"Expected currents shape [B, W, {input_len}, {num_features}], "
                f"got {tuple(currents.shape)}"
            )
        if distances.shape != (batch_size, currents.shape[1]):
            raise ValueError(
                f"Expected distances shape [B, W={currents.shape[1]}], got {tuple(distances.shape)}"
            )

        num_views = currents.shape[1]                                 # W
        # Flatten currents along the W axis: each (b, w) becomes its own per-window
        # normalized window so that B*W is the effective batch for the cur branch.
        currents_flat = currents.reshape(batch_size * num_views, input_len, num_features)
        if self.config.use_norm:
            past_norm, _, _ = per_window_normalize(past)
            cur_norm, means_cur, stdev_cur = per_window_normalize(currents_flat)
        else:
            past_norm, cur_norm = past, currents_flat
            means_cur = stdev_cur = None

        # Channel-independent patchify.
        past_patches, n_vars = self.patch_embedding.patchify(past_norm)       # [B*C, N, patch_len]
        cur_patches, _ = self.patch_embedding.patchify(cur_norm)              # [B*W*C, N, patch_len]
        num_patches = past_patches.size(1)
        is_causal = self.config.pretrain_causal

        # Encode past once, then expand along W (treat W as extra batch dim).
        past_emb = self.patch_embedding.embed_patches(past_patches, add_pos=True)
        past_enc = self.encoder(past_emb, is_causal=is_causal)        # [B*C, N, d_model]
        # Interleave so order matches cur_patches's [B, W, C] flattening.
        past_enc_bcw = past_enc.view(batch_size, n_vars, num_patches, self.config.d_model)
        past_enc_w = (
            past_enc_bcw.unsqueeze(1)
            .expand(batch_size, num_views, n_vars, num_patches, self.config.d_model)
            .reshape(batch_size * num_views * n_vars, num_patches, self.config.d_model)
        )                                                              # [B*W*C, N, d]

        # no-anchor ablation: replace past_enc with a broadcast null token (learnable).
        if self.config.past_disabled:
            past_enc_w = self.past_null.expand(
                batch_size * num_views * n_vars, num_patches, self.config.d_model
            )

        # Distances: [B, W] -> [B*W*C] in the same flattening order.
        dist_flat = (
            distances.unsqueeze(-1)
            .expand(batch_size, num_views, n_vars)
            .reshape(-1)
        )                                                              # [B*W*C]
        # no-TDA ablation: skip the distance code entirely.
        # Default: the "relative" predictor returns [B*W*C, N, d] (per key position).
        # Broadcast paths ("sinusoidal" / "learnable_token") return [B*W*C, d];
        # unsqueeze(1) broadcasts uniformly across N (provably softmax-inert).
        if self.config.lineage_disabled:
            kv_key = past_enc_w
            kv_value = past_enc_w
        else:
            if self.config.lineage_type == "relative":
                lineage = self.lineage_predictor(dist_flat, num_patches)  # [B*W*C, N, d]
            else:
                lineage = self.lineage_predictor(dist_flat).unsqueeze(1)  # [B*W*C, 1, d]
            kv_key = past_enc_w + lineage                                 # broadcast (sin/token) or per-position (relative)
            # K=V ablation: kv_share_lineage=1 -> V also gets the code.
            kv_value = past_enc_w + lineage if self.config.kv_share_lineage else past_enc_w

        # Sample t freshly per (b, w, c, n) reconstruction view.
        flat_batch = cur_patches.size(0)                               # B*W*C
        t = self.diffusion.sample_t_per_patch(flat_batch, num_patches) # [B*W*C, N]

        # no-diffusion ablation: clean target as decoder query, no
        # step embedding. Decoder must reconstruct cur from cur (sanity check;
        # loss should drop near 0).
        if self.config.diffusion_noise_disabled:
            if self.config.diffusion_space == "patch":
                query_emb = self.patch_embedding.embed_patches(cur_patches, add_pos=True)
            else:  # latent
                query_emb = self.patch_embedding.embed_patches(cur_patches, add_pos=True)
                query_emb = self.encoder(query_emb, is_causal=is_causal).detach()
            step_emb = torch.zeros_like(query_emb)
        else:
            if self.config.diffusion_space == "patch":
                noisy_patch, _ = self.diffusion.add_noise(cur_patches, t)
                query_emb = self.patch_embedding.embed_patches(noisy_patch, add_pos=True)
            else:  # latent
                cur_emb = self.patch_embedding.embed_patches(cur_patches, add_pos=True)
                cur_enc = self.encoder(cur_emb, is_causal=is_causal).detach()
                noisy_enc, _ = self.diffusion.add_noise(cur_enc, t)
                query_emb = noisy_enc
            step_emb = self.noise_embedding(t)                         # [B*W*C, N, d]

        query = query_emb + step_emb

        decoded = self.decoder(query, kv_key, kv_value)                # [B*W*C, N, d]

        # Reshape to [B*W, C, N, d] for the flatten head, output length input_len.
        decoded_bcnd = decoded.view(
            batch_size * num_views, n_vars, num_patches, self.config.d_model
        )
        cur_recon = self.recon_head(decoded_bcnd)                      # [B*W, L, C] (per-window-normalized space)

        # Loss in fit-on-train standardized space (= currents_flat's space).
        if means_cur is not None:
            cur_recon = denormalize(cur_recon, means_cur, stdev_cur)   # reverse per-window RevIN
        loss = F.mse_loss(cur_recon, currents_flat)                    # currents_flat: [B*W, L, C]
        return {"loss": loss}


def _make_loader(config: PretrainConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = SiamesePretrainDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
    )


def pretrain_setting_name(config: PretrainConfig) -> str:
    return config.run_name


def pretrain_run_dir(config: PretrainConfig) -> Path:
    return (
        PRETRAIN_CHECKPOINT_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / config.run_name
    )


def _run_epoch(
    model: PretrainModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    loss_sum = 0.0
    steps = 0
    chunk_w = int(getattr(model.config, "w_chunk_size", 0) or 0)
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for past, current, distance in loader:
            if max_steps is not None and steps >= max_steps:
                break
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            # Canonical W axis is dim 1 of `current` ([B, W, L, C]); the
            # dataset back-compat path returns [B, L, C] which forward
            # unsqueezes to W=1. Read it directly to decide on chunking.
            W_total = int(current.shape[1]) if current.dim() == 4 else 1
            if chunk_w > 0 and W_total > chunk_w:
                # W-chunked gradient accumulation. Each chunk loss is scaled
                # by (chunk / W_total) so accumulated .grad equals the
                # single-shot mean-over-all-W gradient (mathematical identity).
                batch_loss = 0.0
                for w_start in range(0, W_total, chunk_w):
                    w_end = min(w_start + chunk_w, W_total)
                    cur_slice = current[:, w_start:w_end]
                    dist_slice = distance[:, w_start:w_end]
                    output = model(past, cur_slice, dist_slice)
                    scale = float(w_end - w_start) / float(W_total)
                    scaled = output["loss"] * scale
                    if is_train:
                        scaled.backward()
                    batch_loss += float(scaled.detach().cpu())
                if is_train:
                    optimizer.step()
                loss_sum += batch_loss
            else:
                output = model(past, current, distance)
                if is_train:
                    output["loss"].backward()
                    optimizer.step()
                loss_sum += float(output["loss"].detach().cpu())
            steps += 1
    return {"loss": loss_sum / steps if steps else float("nan")}


def _backbone_state_dict(model: PretrainModel) -> dict[str, torch.Tensor]:
    backbone_keys = ("patch_embedding.", "encoder.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if any(key.startswith(prefix) for prefix in backbone_keys)
    }


def _save_checkpoint(
    model: PretrainModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: PretrainConfig,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
    checkpoint_dir: Path,
    filename: str,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    encoder_state = _backbone_state_dict(model)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "data": config.dataset_name,
        "model": config.model_id,
        "setting": pretrain_setting_name(config),
        "config": config.__dict__.copy(),
        "encoder_state_dict": encoder_state,
        "model_state_dict": encoder_state,
        "full_model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    torch.save(payload, checkpoint_dir / filename)


def _reset_pretrain_logs(checkpoint_dir: Path, config: PretrainConfig) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "config.json").write_text(json.dumps(config.__dict__, indent=2) + "\n")
    (checkpoint_dir / "epoch_losses.jsonl").write_text("")
    with (checkpoint_dir / "epoch_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "lr",
                "elapsed_sec",
                "train_loss",
                "val_loss",
                "best_val_loss",
                "checkpoint_saved",
            ],
        )
        writer.writeheader()


def _append_pretrain_epoch_log(
    checkpoint_dir: Path,
    epoch_record: dict[str, float | int | bool],
) -> None:
    with (checkpoint_dir / "epoch_losses.jsonl").open("a") as handle:
        handle.write(json.dumps(epoch_record) + "\n")
    with (checkpoint_dir / "epoch_losses.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_record.keys()))
        writer.writerow(epoch_record)


def train_pretrain_model(
    config: PretrainConfig,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> PretrainModel:
    seed_everything(config.seed)
    model = PretrainModel(config)
    train_loader = _make_loader(config, split="train", shuffle=True)
    val_loader = _make_loader(config, split="validation", shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer=optimizer, gamma=config.lr_decay
    )
    checkpoint_dir = pretrain_run_dir(config)
    _reset_pretrain_logs(checkpoint_dir, config)
    best_val_loss: float | None = None

    print(f"Config: {config}")
    print(f"Device: {model.device}")
    print(
        f"Train windows: {len(train_loader.dataset)}, "
        f"Validation windows: {len(val_loader.dataset)}"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")

    for epoch in range(1, config.train_epochs + 1):
        start_time = time.time()
        train_losses = _run_epoch(model, train_loader, optimizer=optimizer, max_steps=max_train_steps)
        val_losses = _run_epoch(model, val_loader, optimizer=None, max_steps=max_val_steps)
        scheduler.step()
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | {elapsed:.2f}s | lr {lr:.7f} | "
            f"train loss {train_losses['loss']:.6f} | val loss {val_losses['loss']:.6f}"
        )

        checkpoint_saved = False
        if best_val_loss is None or val_losses["loss"] <= best_val_loss:
            previous = best_val_loss
            best_val_loss = val_losses["loss"]
            checkpoint_saved = True
            print(
                "Validation loss decreased "
                f"({previous if previous is not None else float('inf'):.6f} -> "
                f"{best_val_loss:.6f}). Saving ckpt_best.pth"
            )
            _save_checkpoint(
                model, optimizer, epoch, config, train_losses, val_losses,
                checkpoint_dir, "ckpt_best.pth",
            )

        _append_pretrain_epoch_log(
            checkpoint_dir,
            {
                "epoch": epoch,
                "lr": lr,
                "elapsed_sec": elapsed,
                "train_loss": train_losses["loss"],
                "val_loss": val_losses["loss"],
                "best_val_loss": best_val_loss,
                "checkpoint_saved": checkpoint_saved,
            },
        )

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            _save_checkpoint(
                model, optimizer, epoch, config, train_losses, val_losses,
                checkpoint_dir, f"ckpt{epoch}.pth",
            )

    write_done(checkpoint_dir)  # last write: pretrain stage complete
    return model


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--run_name", required=True, help="User-named folder for this pretrain config.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--input_len", type=int, required=True)
    parser.add_argument("--enc_in", type=int, required=True)
    parser.add_argument("--patch_len", type=int, required=True)
    parser.add_argument("--stride", type=int, required=True)
    parser.add_argument("--e_layers", type=int, required=True)
    parser.add_argument("--d_layers", type=int, required=True,
                        help="Number of cross-attention decoder layers.")
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--use_norm", type=int, choices=[0, 1], required=True)
    parser.add_argument("--time_steps", type=int, required=True)
    parser.add_argument("--scheduler", choices=["cosine", "linear"], required=True)
    parser.add_argument("--sampling_min", type=int, required=True,
                        help="Minimum shifted distance s, in PATCH units (the target's raw "
                             "start is anchor_start + s * patch_len).")
    parser.add_argument("--sampling_max", type=int, required=True,
                        help="Maximum shifted distance s, in PATCH units; the paper uses "
                             "s_max = 12.")
    parser.add_argument(
        "--lineage_mlp_layers",
        type=int,
        default=2,
        help="Hidden GeLU blocks in the shared LineagePredictor MLP. "
             "0 = degenerate (input_proj + final linear only).",
    )
    parser.add_argument(
        "--current_views",
        type=int,
        default=1,
        help="Random current windows paired with each past per forward pass. "
             "1 = a single target; >1 folds W independent targets into the same step.",
    )
    parser.add_argument(
        "--head_dropout",
        type=float,
        default=0.0,
        help="Dropout inside the flatten reconstruction head.",
    )
    parser.add_argument(
        "--diffusion_space",
        choices=["patch", "latent"],
        required=True,
        help="Noise target space. 'patch' adds noise to raw cur_patches; "
             "'latent' adds noise to encoded (detached) cur_emb.",
    )
    parser.add_argument(
        "--pretrain_causal",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 = pretrain encoder uses causal mask; 0 = non-causal (matches forecast path).",
    )
    parser.add_argument(
        "--mask_ratio",
        type=float,
        default=0.0,
        help="MAE-style partial mask ratio on decoder self/cross attention. "
             "0.0 disables; >0 masks that fraction of attention links per call.",
    )
    parser.add_argument(
        "--lr_decay",
        type=float,
        default=0.5,
        help="Pretrain ExponentialLR gamma (aligned with TimeDART).",
    )
    # Ablation knobs; the defaults are the paper's configuration.
    parser.add_argument(
        "--lineage_type",
        choices=["relative", "sinusoidal", "learnable_token"],
        default="relative",
        help="Default 'relative' = position-dependent distance code (varies along "
             "K-axis; breaks the softmax shift-invariance that makes a broadcast code inert). "
             "'sinusoidal' = distance-only broadcast code (provably inert, kept for ablation). "
             "'learnable_token' = discrete TimeSiam-style nn.Embedding indexed by s.",
    )
    parser.add_argument(
        "--kv_share_lineage",
        type=int,
        choices=[0, 1],
        default=0,
        help="K=V ablation. 0 = default (V=anchor, K=anchor+code); "
             "1 = K=V=past+lineage (lineage routed into V as well).",
    )
    parser.add_argument(
        "--forced_first_shift_one",
        type=int,
        choices=[0, 1],
        default=1,
        help="1 = default (the first target is pinned to shift s=1, "
             "TimeDART-equivalent); 0 = the first target is sampled like the rest.",
    )
    parser.add_argument(
        "--lineage_disabled",
        type=int,
        choices=[0, 1],
        default=0,
        help="no-TDA ablation. 1 = skip the distance code entirely (K=V=past_enc); "
             "isolates the contribution of distance-aware routing.",
    )
    parser.add_argument(
        "--diffusion_noise_disabled",
        type=int,
        choices=[0, 1],
        default=0,
        help="no-diffusion ablation. 1 = clean target as decoder query, no noise added, "
             "step embedding zeroed. Loss should collapse near 0 (sanity).",
    )
    parser.add_argument(
        "--past_disabled",
        type=int,
        choices=[0, 1],
        default=0,
        help="no-anchor ablation. 1 = past_enc replaced by a learnable null token "
             "broadcast across (B*W*C, N); isolates the value of past conditioning.",
    )
    parser.add_argument(
        "--w_chunk_size",
        type=int,
        default=0,
        help="Memory knob. 0 = forward all W views at once. "
             ">0 = chunk W into groups of this size; backward per chunk "
             "(gradient accumulates), single optimizer.step() per data batch. "
             "Loss scaling preserves the all-W mean-reduction gradient "
             "(mathematically identical). Use when B*W*C OOMs decoder VRAM.",
    )
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--train_epochs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--checkpoint_every", type=int, required=True)
    parser.add_argument(
        "--cuda_visible_devices",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        default=None,
        help="Comma-separated CUDA_VISIBLE_DEVICES, e.g. 1 or 1,4,5.",
    )
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda:0")
    parser.add_argument("--seed", type=int, default=2021)


def _config_from_args(args: argparse.Namespace) -> PretrainConfig:
    return PretrainConfig(
        dataset_name=args.dataset,
        data_fine_dir=args.data_fine_dir,
        model_id=args.model_id,
        run_name=args.run_name,
        features=args.features,
        input_len=args.input_len,
        enc_in=args.enc_in,
        patch_len=args.patch_len,
        stride=args.stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_norm=bool(args.use_norm),
        time_steps=args.time_steps,
        scheduler=args.scheduler,
        sampling_min=args.sampling_min,
        sampling_max=args.sampling_max,
        diffusion_space=args.diffusion_space,
        lineage_mlp_layers=args.lineage_mlp_layers,
        current_views=args.current_views,
        head_dropout=args.head_dropout,
        pretrain_causal=bool(args.pretrain_causal),
        mask_ratio=args.mask_ratio,
        lr_decay=args.lr_decay,
        lineage_type=args.lineage_type,
        kv_share_lineage=bool(args.kv_share_lineage),
        forced_first_shift_one=bool(args.forced_first_shift_one),
        lineage_disabled=bool(args.lineage_disabled),
        diffusion_noise_disabled=bool(args.diffusion_noise_disabled),
        past_disabled=bool(args.past_disabled),
        w_chunk_size=int(args.w_chunk_size),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain SDTA on data/fine windows")
    parser.add_argument("--epochs", type=int, default=None, help="Override config train_epochs")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    _add_config_args(parser)
    args = parser.parse_args()
    apply_cuda_visible_devices(args.cuda_visible_devices)

    config = _config_from_args(args)
    if args.epochs is not None:
        config.train_epochs = args.epochs
    train_pretrain_model(
        config,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )


if __name__ == "__main__":
    main()
