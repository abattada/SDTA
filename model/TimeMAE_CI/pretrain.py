"""TimeMAE_CI pretraining (decoupled masked autoencoder, channel-independent).

Channel-INDEPENDENT variant of `model/TimeMAE/pretrain.py`. The decoupled-MAE
pretext is byte-for-byte the same objective (momentum representation regression +
codeword classification, `loss = alpha*align + beta*reconstruct`, per-step
momentum_update); the ONLY change is channel handling: the input projection is the
channel-independent `Linear(patch_len->d_model)` (each channel folded into the batch
dim) instead of the official TimeMAE channel-mixing `Conv1d(enc_in->d_model)`. This
makes TimeMAE_CI apples-to-apples with the shared-encoder CI baselines (PatchTST /
TimeSiam / SimMTM), so the SSL objective is the only remaining variable.

Masking keeps TimeMAE's semantics: ONE random patch permutation per forward, applied
across the whole (B*V) batch (faithful to TimeMAE's single-shuffle-per-forward; not
per-sequence). Only `patch_embedding` + `encoder` weights transfer to the forecaster.

Source ref: `official_source/TimeMAE` (objective) + `model/TimeMAE` (this repo's
channel-mixing port). Deviation from official = channel handling (B-arch, allowed; see
docs/check/model_audit.md §10 and Diff_check/TimeMAE_CI.md).
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model._baseline_forecast import (
    PROJECT_ROOT,
    PRETRAIN_ROOT,
    masked_normalize,
    resolve_device,
    seed_everything,
    select_features,
)
from model._test_io import write_done
from .encoder import ChannelIndependentPatchEmbedding, build_encoder_backbone, num_patches


@dataclass
class PretrainConfig:
    dataset_name: str
    model_id: str
    run_name: str
    features: str
    input_len: int
    enc_in: int
    patch_len: int
    stride: int
    d_model: int
    d_ff: int
    n_heads: int
    e_layers: int
    dropout: float
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int
    num_workers: int
    checkpoint_every: int
    mask_ratio: float
    vocab_size: int
    reg_layers: int
    momentum: float
    alpha: float
    beta: float
    data_fine_dir: str = "data/fine"
    device: str = "auto"
    seed: int = 2021

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


def _split_path(config: PretrainConfig, split: str) -> Path:
    return PROJECT_ROOT / config.data_fine_dir / config.dataset_name / split / "data.npy"


class PretrainWindowDataset(Dataset):
    def __init__(self, config: PretrainConfig, split: str):
        self.config = config
        split = "validation" if split == "val" else split
        path = _split_path(config, split)
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path}")
        data = np.load(path, mmap_mode="r")
        self.data = select_features(data, config.features, config.enc_in)
        self.window_count = len(self.data) - config.input_len + 1
        if self.window_count <= 0:
            raise ValueError(f"Not enough rows for TimeMAE_CI pretrain windows: {path}")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        x = np.array(
            self.data[index:index + self.config.input_len],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x)


class Tokenizer(nn.Module):
    """Linear -> gumbel-softmax discrete codeword (target) + logits (prediction)."""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.center = nn.Linear(d_model, vocab_size)

    def codewords(self, x: torch.Tensor) -> torch.Tensor:
        # discrete target indices; argmax is non-differentiable so no grad leaks
        # into the tokenizer from the targets (matches official Tokenizer.forward)
        probs = F.gumbel_softmax(self.center(x), dim=-1)
        return probs.argmax(dim=-1)


class CrossAttnBlock(nn.Module):
    """Mask-token queries attend to visible representations (regressor block)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        q = self.norm1(query)
        attn_out, _ = self.attn(q, context, context, need_weights=False)
        query = query + self.dropout(attn_out)
        query = query + self.dropout(self.ffn(self.norm2(query)))
        return query


class PretrainModel(nn.Module):
    def __init__(self, config: PretrainConfig):
        super().__init__()
        self.config = config
        # patch_embedding (channel-independent) + encoder are the only modules that
        # transfer to the forecaster.
        self.patch_embedding = ChannelIndependentPatchEmbedding(
            patch_len=config.patch_len, stride=config.stride, d_model=config.d_model,
        )
        self.encoder = build_encoder_backbone(
            config.d_model, config.n_heads, config.e_layers, config.d_ff, config.dropout
        )
        # momentum (EMA) encoder produces the regression targets; same shape as encoder
        self.momentum_encoder = build_encoder_backbone(
            config.d_model, config.n_heads, config.e_layers, config.d_ff, config.dropout
        )
        for p in self.momentum_encoder.parameters():
            p.requires_grad_(False)
        self.num_patches = num_patches(config.input_len, config.patch_len, config.stride)
        self.mask_len = max(1, int(config.mask_ratio * self.num_patches))
        self.mask_token = nn.Parameter(torch.randn(config.d_model))
        self.tokenizer = Tokenizer(config.d_model, config.vocab_size)
        self.regressor = nn.ModuleList(
            [
                CrossAttnBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.reg_layers)
            ]
        )
        self.momentum = config.momentum
        self.alpha = config.alpha
        self.beta = config.beta

    @torch.no_grad()
    def copy_weight(self) -> None:
        for pa, pb in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            pb.data.copy_(pa.data)

    @torch.no_grad()
    def momentum_update(self) -> None:
        for pa, pb in zip(self.encoder.parameters(), self.momentum_encoder.parameters()):
            pb.data = self.momentum * pb.data + (1.0 - self.momentum) * pa.data

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x_norm, _, _ = masked_normalize(x)
        # channel-independent patch embedding: ((B*V), P, d_model) — channel folded into batch
        raw, _num_vars = self.patch_embedding(x_norm)
        pos = self.patch_embedding.position(raw)
        emb = raw + pos
        tokens = self.tokenizer.codewords(raw)  # ((B*V), P)
        mask_tokens = self.mask_token.expand(emb.size(0), self.num_patches, -1) + pos

        # one permutation per forward, shared across the whole (B*V) batch (TimeMAE semantics)
        index = torch.randperm(self.num_patches, device=x.device)
        v_index = index[: self.num_patches - self.mask_len]
        m_index = index[self.num_patches - self.mask_len:]

        rep_visible = self.encoder(emb[:, v_index, :])
        with torch.no_grad():
            rep_mask = self.momentum_encoder(emb[:, m_index, :])

        query = mask_tokens[:, m_index, :]
        for block in self.regressor:
            query = block(query, rep_visible)
        rep_mask_prediction = query
        token_logits = self.tokenizer.center(rep_mask_prediction)  # ((B*V), mask_len, vocab)
        tokens_masked = tokens[:, m_index]

        loss_align = F.mse_loss(rep_mask_prediction, rep_mask)
        loss_recon = F.cross_entropy(
            token_logits.reshape(-1, token_logits.size(-1)),
            tokens_masked.reshape(-1),
            label_smoothing=0.2,
        )
        loss = self.alpha * loss_align + self.beta * loss_recon
        return {"loss": loss, "loss_align": loss_align, "loss_recon": loss_recon}


def pretrain_run_dir(config: PretrainConfig) -> Path:
    return PRETRAIN_ROOT / config.model_id / config.dataset_name / f"il{config.input_len}" / config.run_name


def _encoder_state(model: PretrainModel) -> dict[str, torch.Tensor]:
    state = model.state_dict()
    return {
        key: value.detach().cpu()
        for key, value in state.items()
        if key.startswith("patch_embedding.") or key.startswith("encoder.")
    }


def _save_checkpoint(
    path: Path,
    model: PretrainModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: PretrainConfig,
    train_losses: dict[str, float],
    val_losses: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": _encoder_state(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "train_losses": train_losses,
            "val_losses": val_losses,
        },
        path,
    )


def _run_epoch(
    model: PretrainModel,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "loss_align": 0.0, "loss_recon": 0.0}
    count = 0
    for step, batch_x in enumerate(loader, start=1):
        batch_x = batch_x.float().to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            out = model(batch_x)
            if is_train:
                out["loss"].backward()
                optimizer.step()
                model.momentum_update()
        batch_size = int(batch_x.size(0))
        for key in totals:
            totals[key] += float(out[key].detach()) * batch_size
        count += batch_size
        if max_steps is not None and step >= max_steps:
            break
    return {key: value / max(count, 1) for key, value in totals.items()}


def pretrain(config: PretrainConfig, max_train_steps: int | None = None, max_val_steps: int | None = None) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.device = str(device)
    train_loader = DataLoader(
        PretrainWindowDataset(config, "train"),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        PretrainWindowDataset(config, "validation"),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model = PretrainModel(config).to(device)
    model.copy_weight()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    out_dir = pretrain_run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (out_dir / "epoch_losses.jsonl").write_text("")
    fieldnames = [
        "epoch", "elapsed_sec", "train_loss", "train_align", "train_recon",
        "val_loss", "val_align", "val_recon", "best_val_loss", "checkpoint_saved",
    ]
    with (out_dir / "epoch_losses.csv").open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()

    best = float("inf")
    best_path = out_dir / "ckpt_best.pth"
    # Official TimeMAE pretrain runs the full epoch budget (no early stop); the checkpoint is
    # kept at best validation loss. Completion is marked by the `done` sentinel written after the
    # loop (model_audit §4). Full budget is kept to stay faithful to the TimeMAE objective.
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_losses = _run_epoch(model, train_loader, device, optimizer, max_steps=max_train_steps)
        with torch.no_grad():
            val_losses = _run_epoch(model, val_loader, device, max_steps=max_val_steps)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train {train_losses['loss']:.6f} (align {train_losses['loss_align']:.6f}, "
            f"recon {train_losses['loss_recon']:.6f}) | val {val_losses['loss']:.6f} | {elapsed:.1f}s"
        )
        saved = False
        if val_losses["loss"] < best:
            best = val_losses["loss"]
            saved = True
            _save_checkpoint(best_path, model, optimizer, epoch, config, train_losses, val_losses)
        if config.checkpoint_every > 0 and epoch % config.checkpoint_every == 0:
            _save_checkpoint(out_dir / f"ckpt{epoch}.pth", model, optimizer, epoch, config, train_losses, val_losses)
        row = {
            "epoch": epoch,
            "elapsed_sec": elapsed,
            "train_loss": train_losses["loss"],
            "train_align": train_losses["loss_align"],
            "train_recon": train_losses["loss_recon"],
            "val_loss": val_losses["loss"],
            "val_align": val_losses["loss_align"],
            "val_recon": val_losses["loss_recon"],
            "best_val_loss": best,
            "checkpoint_saved": saved,
        }
        with (out_dir / "epoch_losses.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        with (out_dir / "epoch_losses.csv").open("a", newline="") as handle:
            csv.DictWriter(handle, fieldnames=fieldnames).writerow(row)
    (out_dir / "training_summary.json").write_text(
        json.dumps({"best_checkpoint": str(best_path), "best_val_loss": best}, indent=2) + "\n"
    )
    write_done(out_dir)  # last write: pretrain stage complete
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--patch_len", type=int, default=8)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--vocab_size", type=int, default=192)
    parser.add_argument("--reg_layers", type=int, default=1)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--max_val_steps", type=int)
    args = parser.parse_args()
    stride = args.stride if args.stride is not None else args.patch_len
    config = PretrainConfig(
        dataset_name=args.dataset,
        model_id=args.model_id,
        run_name=args.run_name,
        features=args.features,
        input_len=args.input_len,
        enc_in=args.enc_in,
        patch_len=args.patch_len,
        stride=stride,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        mask_ratio=args.mask_ratio,
        vocab_size=args.vocab_size,
        reg_layers=args.reg_layers,
        momentum=args.momentum,
        alpha=args.alpha,
        beta=args.beta,
        data_fine_dir=args.data_fine_dir,
        device=args.device,
        seed=args.seed,
    )
    path = pretrain(config, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
