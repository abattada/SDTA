"""TimeSiam pretraining: Siamese past/current reconstruction with lineage tokens."""
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

from .cli import apply_cuda_visible_devices
from .dataset import SiamesePretrainDataset
from .encoder import EncoderConfig, build_encoder
from .siamese import SiameseDecoder
from .utils import (
    PRETRAIN_CHECKPOINT_ROOT,
    format_param,
    resolve_device,
    seed_everything,
)


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
    mask_rate: float
    sampling_range: int
    lineage_tokens: int
    current_token: bool
    learning_rate: float
    batch_size: int
    train_epochs: int
    num_workers: int
    checkpoint_every: int
    # Masking strategy for the current window. Matches official TimeSiam
    # `masked_rule`: "channel_continuous" (geometric/contiguous spans, official
    # ETT default) or "channel_binomial" (i.i.d. per-point Bernoulli). `lm` is
    # the average masked-span length used by the geometric rule.
    masked_rule: str = "channel_continuous"
    lm: int = 3
    device: str = "auto"
    seed: int = 2021

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


def channel_binomial_mask(
    batch_size: int,
    seq_len: int,
    num_vars: int,
    mask_rate: float,
    device: torch.device,
) -> torch.Tensor:
    """True = keep (unmasked), False = masked (loss is computed here).

    Per-point i.i.d. Bernoulli per channel; matches official `channel_binomial`.
    """
    keep_prob = 1.0 - mask_rate
    return torch.bernoulli(
        torch.full((batch_size, seq_len, num_vars), keep_prob, device=device)
    ).bool()


def channel_continuous_mask(
    batch_size: int,
    seq_len: int,
    num_vars: int,
    mask_rate: float,
    lm: int,
    device: torch.device,
) -> torch.Tensor:
    """True = keep, False = masked. Independent geometric (contiguous-span) mask per
    (sample, channel); distribution matches official `channel_continuous`
    (`geom_noise_mask_single`): a 2-state Markov chain along time where masked/kept
    streak lengths are geometric (mean `lm` for masked streaks).

    Vectorised on `device`: the only loop is over the time axis (`seq_len`), with all
    `batch_size * num_vars` sequences advanced in parallel. This replaces the original
    per-element Python/numpy triple loop (`batch * num_vars * seq_len` scalar RNG calls
    per batch, which starved the GPU on large-channel datasets). Not bit-identical to
    the official scalar version (RNG draw order differs) but statistically equivalent.
    """
    if mask_rate <= 0.0:
        return torch.ones(batch_size, seq_len, num_vars, dtype=torch.bool, device=device)
    if mask_rate >= 1.0:
        return torch.zeros(batch_size, seq_len, num_vars, dtype=torch.bool, device=device)

    n = batch_size * num_vars
    p_m = 1.0 / lm  # prob. a masked streak stops
    p_u = p_m * mask_rate / (1.0 - mask_rate)  # prob. an unmasked streak stops
    p_m_t = torch.tensor(p_m, device=device)
    p_u_t = torch.tensor(p_u, device=device)

    state = torch.rand(n, device=device) > mask_rate  # True = keep, False = mask
    keep = torch.empty(n, seq_len, dtype=torch.bool, device=device)
    for i in range(seq_len):
        keep[:, i] = state
        stop_p = torch.where(state, p_u_t, p_m_t)  # current streak's stop probability
        state = state ^ (torch.rand(n, device=device) < stop_p)  # toggle where it stops
    # n is laid out as (batch, num_vars); restore official [batch, seq_len, num_vars].
    return keep.view(batch_size, num_vars, seq_len).permute(0, 2, 1).contiguous()


def build_mask(
    rule: str,
    batch_size: int,
    seq_len: int,
    num_vars: int,
    mask_rate: float,
    lm: int,
    device: torch.device,
) -> torch.Tensor:
    if rule == "channel_binomial":
        return channel_binomial_mask(batch_size, seq_len, num_vars, mask_rate, device)
    if rule == "channel_continuous":
        return channel_continuous_mask(batch_size, seq_len, num_vars, mask_rate, lm, device)
    raise ValueError(
        f"Unsupported masked_rule '{rule}'; expected 'channel_continuous' or 'channel_binomial'."
    )


def masked_normalize(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-window normalization; if mask is given, stats use only kept entries."""
    if mask is None:
        means = x.mean(dim=1, keepdim=True).detach()
        centered = x - means
        stdev = torch.sqrt(centered.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
        return centered / stdev, means, stdev
    mask_f = mask.float()
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    means = (x * mask_f).sum(dim=1, keepdim=True) / counts
    means = means.detach()
    centered = (x - means) * mask_f
    stdev = torch.sqrt((centered ** 2).sum(dim=1, keepdim=True) / counts + 1e-5).detach()
    return (x - means) / stdev, means, stdev


def denormalize(x: torch.Tensor, means: torch.Tensor, stdev: torch.Tensor) -> torch.Tensor:
    return x * stdev + means


class PretrainModel(nn.Module):
    """Past encoder + current encoder (shared) + Siamese decoder + patch projection."""

    def __init__(self, config: PretrainConfig):
        super().__init__()
        if config.lineage_tokens < 1:
            raise ValueError(f"lineage_tokens must be >= 1, got {config.lineage_tokens}")
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
        self.tokens_past = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, 1, config.d_model)) for _ in range(config.lineage_tokens)]
        )
        if config.current_token:
            self.token_current = nn.Parameter(torch.zeros(1, 1, config.d_model))
        else:
            self.register_parameter("token_current", None)
        for token in self.tokens_past:
            nn.init.normal_(token, std=0.02)
        if self.token_current is not None:
            nn.init.normal_(self.token_current, std=0.02)

        self.siamese_decoder = SiameseDecoder(
            d_model=config.d_model,
            num_heads=config.n_heads,
            num_layers=config.d_layers,
            feedforward_dim=config.d_ff,
            dropout=config.dropout,
        )
        self.projection = nn.Linear(config.d_model, config.patch_len)
        self.to(self.device)

    def _select_past_token(self, segments: torch.Tensor, num_vars: int, num_patches: int) -> torch.Tensor:
        all_tokens = torch.cat([token for token in self.tokens_past], dim=0).squeeze(1)
        selected = all_tokens[segments.to(all_tokens.device)]
        selected = selected.unsqueeze(1).repeat_interleave(num_vars, dim=0)
        return selected.expand(-1, num_patches, -1)

    def forward(
        self,
        past: torch.Tensor,
        current: torch.Tensor,
        segments: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        past = past.to(self.device)
        current = current.to(self.device)
        mask = mask.to(self.device)
        segments = segments.to(self.device)

        batch_size, input_len, num_features = past.shape
        if input_len != self.config.input_len:
            raise ValueError(f"Expected input_len={self.config.input_len}, got {input_len}")
        if num_features != self.config.enc_in:
            raise ValueError(f"Expected enc_in={self.config.enc_in}, got {num_features}")

        if self.config.use_norm:
            past_norm, _, _ = masked_normalize(past)
            cur_norm, cur_means, cur_stdev = masked_normalize(current, mask=mask)
        else:
            past_norm = past
            cur_norm = current
            cur_means = torch.zeros_like(current[:, :1, :])
            cur_stdev = torch.ones_like(current[:, :1, :])

        past_emb, _, n_vars = self.patch_embedding(past_norm, add_pos=True)
        past_emb = past_emb + self._select_past_token(segments, n_vars, past_emb.size(1))
        past_enc = self.encoder(past_emb)

        cur_masked = cur_norm * mask.float()
        cur_emb, _, _ = self.patch_embedding(cur_masked, add_pos=True)
        if self.token_current is not None:
            cur_emb = cur_emb + self.token_current
        cur_enc = self.encoder(cur_emb)

        dec_out = self.siamese_decoder(cur_enc, past_enc)
        patch_pred = self.projection(dec_out)

        num_patches = past_emb.size(1)
        pred = patch_pred.reshape(batch_size, num_features, num_patches * self.config.patch_len)
        pred = pred.permute(0, 2, 1)

        if self.config.use_norm:
            pred = denormalize(pred, cur_means, cur_stdev)

        loss_mask = (~mask).float()
        squared_error = (pred - current) ** 2
        denom = loss_mask.sum().clamp_min(1.0)
        loss = (squared_error * loss_mask).sum() / denom
        return {"loss": loss, "pred": pred}


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
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    steps = 0
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_idx, (past, current, segments) in enumerate(loader):
            if max_steps is not None and batch_idx >= max_steps:
                break
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            mask = build_mask(
                model.config.masked_rule,
                past.size(0),
                model.config.input_len,
                model.config.enc_in,
                model.config.mask_rate,
                model.config.lm,
                model.device,
            )
            output = model(past, current, segments, mask)
            if is_train:
                output["loss"].backward()
                optimizer.step()
            total_loss += float(output["loss"].detach().cpu())
            steps += 1
    return total_loss / max(steps, 1)


def _backbone_state_dict(model: PretrainModel) -> dict[str, torch.Tensor]:
    backbone_keys = ("patch_embedding.", "encoder.", "tokens_past.", "token_current")
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
    train_loss: float,
    val_loss: float,
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
        "train_loss": train_loss,
        "val_loss": val_loss,
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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=config.train_epochs
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
        train_loss = _run_epoch(model, train_loader, optimizer=optimizer, max_steps=max_train_steps)
        val_loss = _run_epoch(model, val_loader, optimizer=None, max_steps=max_val_steps)
        scheduler.step()
        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | {elapsed:.2f}s | "
            f"lr {lr:.7f} | train loss {train_loss:.6f} | val loss {val_loss:.6f}"
        )

        checkpoint_saved = False
        if best_val_loss is None or val_loss <= best_val_loss:
            previous = best_val_loss
            best_val_loss = val_loss
            checkpoint_saved = True
            print(
                "Validation loss decreased "
                f"({previous if previous is not None else float('inf'):.6f} -> "
                f"{best_val_loss:.6f}). Saving ckpt_best.pth"
            )
            _save_checkpoint(
                model, optimizer, epoch, config, train_loss, val_loss,
                checkpoint_dir, "ckpt_best.pth",
            )

        _append_pretrain_epoch_log(
            checkpoint_dir,
            {
                "epoch": epoch,
                "lr": lr,
                "elapsed_sec": elapsed,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "checkpoint_saved": checkpoint_saved,
            },
        )

        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            _save_checkpoint(
                model, optimizer, epoch, config, train_loss, val_loss,
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
    parser.add_argument("--d_layers", type=int, required=True)
    parser.add_argument("--n_heads", type=int, required=True)
    parser.add_argument("--d_model", type=int, required=True)
    parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--use_norm", type=int, choices=[0, 1], required=True)
    parser.add_argument("--mask_rate", type=float, required=True)
    parser.add_argument(
        "--masked_rule",
        choices=["channel_continuous", "channel_binomial"],
        default="channel_continuous",
        help="Current-window masking rule. 'channel_continuous' = official ETT "
             "default (geometric spans); 'channel_binomial' = i.i.d. per-point.",
    )
    parser.add_argument("--lm", type=int, default=3,
                        help="Average masked-span length for channel_continuous.")
    parser.add_argument("--sampling_range", type=int, required=True,
                        help="Sample current within [0, sampling_range*input_len] past the anchor.")
    parser.add_argument("--lineage_tokens", type=int, required=True,
                        help="Number of past-lineage embeddings; must be >= 1.")
    parser.add_argument("--current_token", type=int, choices=[0, 1], default=0)
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
        mask_rate=args.mask_rate,
        masked_rule=args.masked_rule,
        lm=args.lm,
        sampling_range=args.sampling_range,
        lineage_tokens=args.lineage_tokens,
        current_token=bool(args.current_token),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain TimeSiam on data/fine windows")
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
