"""SimTS pretraining ported onto the repo shared patch encoder (channel-independent).

Method = official SimTS (Zheng et al., 2023; `official_source/SimTS`, commit 3e98448):
a **history** window and a separate **future** window (non-overlapping) are each encoded
by the shared encoder (Siamese, weights tied); the **last history token** is the history
summary; a `LinearPred` predictor maps that summary to a predicted future-latent
sequence; the future window is encoded and **stop-gradient detached** as the target;
the loss is the **negative cosine similarity** between predicted and target future
latents (SimSiam-style: no negatives, no momentum encoder). Augmentation / masking are
*off* — the released forecasting `fit` comments out `PretrainDataset` and passes
`mask=None`, so the only "views" are history vs. future.

Windowing (input 96 / target 96, non-overlapping): each pretrain item is a contiguous
`span_len = input_len + future_len` slice. history = the **full `input_len` window** — the
SAME context the downstream forecaster sees (so the encoder is pretrained on 96-step
contexts, not a truncated sub-window); future (target) = the next `future_len` rows.
This maps official SimTS's "history = context length `K`, future = horizon" onto the
repo-unified `input_len`: history `K` -> `input_len=96`; future horizon -> `future_len`
(default 96, a bounded choice vs official's very long future). The downstream input budget
stays 96 (the future window is a pretrain-only target, discarded after pretrain).

What differs from official (all deliberate, see docs/check/SimTS.md):
  - **B-arch (encoder)**: official multi-scale dilated causal CNN (`CausalCNNEncoder`,
    channel-mixing `input_fc`, `repr_dims=320`) → repo shared patch Transformer
    (`model.conv_encoder`, channel-independent, `d_model=32`). This is the one allowed
    backbone swap; it makes SimTS comparable to SDTA / SimMTM / TimeSiam on the same
    encoder. Capacity is the repo budget, not `repr_dims=320` (B-arch budget).
  - **Wiring (future horizon)**: official future ≈ 2799 -> bounded `future_len` (default 96
    = input_len). history length is NOT truncated — it equals `input_len`, matching the
    downstream context. The summary uses the **last** history token (the encoder's
    documented `trend_h_repr`); the released code's `random idx >= 127` is tied to `K=201`
    and does not transfer.
  - **A-class (fairness)**: each window (history, future) gets its own input-window instance
    norm (mean+std) — the same RevIN the downstream forecaster applies to its input window
    (official SimTS has no per-window norm, only the global scaler) — like every repo model.

What is kept = official (B-method): SimSiam stop-grad + asymmetric predictor, negative
cosine loss, no negatives, no augmentation/masking, `SGD` with the predictor param group
at `predictor_lr_mult * lr` (official `0.0001 * lr`), `LinearPred` predictor with dropout
`0.25`, optional hierarchical cosine loss (default off).
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

from model._test_io import write_done
from model._baseline_forecast import (
    PROJECT_ROOT,
    PRETRAIN_ROOT,
    masked_normalize,
    resolve_device,
    seed_everything,
    select_features,
)
from .encoder import EncoderConfig, build_encoder


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
    head_dropout: float
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int
    num_workers: int
    checkpoint_every: int
    future_len: int = 96
    predictor_lr_mult: float = 0.0001
    predictor_dropout: float = 0.25
    hierarchical: bool = False
    data_fine_dir: str = "data/fine"
    device: str = "auto"
    seed: int = 2021

    @property
    def num_patches(self) -> int:
        """History = the full `input_len` window — same context the downstream forecaster sees."""
        return int((self.input_len - self.patch_len) / self.stride) + 1

    @property
    def n_fut_patches(self) -> int:
        """Future (target) = a separate, non-overlapping `future_len` window."""
        return int((self.future_len - self.patch_len) / self.stride) + 1

    @property
    def span_len(self) -> int:
        """Total contiguous rows per pretrain item: history (input_len) + future (future_len)."""
        return self.input_len + self.future_len


def _split_path(config: PretrainConfig, split: str) -> Path:
    return PROJECT_ROOT / config.data_fine_dir / config.dataset_name / split / "data.npy"


class PretrainWindowDataset(Dataset):
    """`span_len` = input_len + future_len contiguous rows per item.

    history = first `input_len` rows (the full downstream context), future (target) = the
    next `future_len` rows — **non-overlapping**. The model splits the span; the downstream
    input budget stays `input_len` (the future is a pretrain-only target, discarded after).
    """

    def __init__(self, config: PretrainConfig, split: str):
        self.config = config
        split = "validation" if split == "val" else split
        path = _split_path(config, split)
        if not path.exists():
            raise FileNotFoundError(f"Split not found: {path}")
        data = np.load(path, mmap_mode="r")
        self.data = select_features(data, config.features, config.enc_in)
        self.window_count = len(self.data) - config.span_len + 1
        if self.window_count <= 0:
            raise ValueError(
                f"Not enough rows for SimTS pretrain span (input_len+future_len="
                f"{config.span_len}): {path}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        x = np.array(
            self.data[index:index + self.config.span_len],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x)


class LinearPred(nn.Module):
    """SimTS predictor (faithful port of official `simts.LinearPred`).

    Maps a single history-summary vector to a predicted future-latent *sequence*:
    `(B, d, 1) -> relu(Linear(1 -> out_len)) -> (B, d, out_len) -> transpose ->
    (B, out_len, d) -> Linear(d -> d) -> (B, out_len, d)`. Dropout `0.25` per official.
    """

    def __init__(self, d_model: int, out_len: int, dropout: float):
        super().__init__()
        self.time_proj = nn.Linear(1, out_len)
        self.feat_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        # summary: (B, d) -> (B, d, 1)
        x = summary.unsqueeze(-1)
        x = self.relu(self.time_proj(x)).transpose(1, 2)  # (B, out_len, d)
        return self.feat_proj(x)


def cosine_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """Negative mean cosine similarity over the feature dim (official `models.loss.cosine_loss`).

    z, z_hat: (B, seq_len, d). cos over dim=2, mean over batch then over seq.
    """
    cos_sim = F.cosine_similarity(z, z_hat, dim=2)  # (B, seq_len)
    return -torch.mean(cos_sim, dim=0).mean()


def hierarchical_cosine_loss(z: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
    """Optional TS2Vec-style multi-scale cosine loss (official `hierarchical_cosine_loss`)."""
    loss = torch.zeros((), device=z.device)
    d = 0
    while z.size(1) > 1:
        loss = loss + cosine_loss(z, z_hat)
        z = F.max_pool1d(z.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z_hat = F.max_pool1d(z_hat.transpose(1, 2), kernel_size=2).transpose(1, 2)
        d += 1
    return loss / max(d, 1)


class PretrainModel(nn.Module):
    """SimTS pretext on the shared patch encoder (channel-independent)."""

    def __init__(self, config: PretrainConfig):
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
        self.predictor = LinearPred(
            d_model=config.d_model,
            out_len=config.n_fut_patches,
            dropout=config.predictor_dropout,
        )
        self.loss_fn = hierarchical_cosine_loss if config.hierarchical else cosine_loss

    def predictor_parameters(self):
        return self.predictor.parameters()

    def encoder_parameters(self):
        return list(self.patch_embedding.parameters()) + list(self.encoder.parameters())

    def _encode(self, segment: torch.Tensor) -> torch.Tensor:
        # segment: (B, seg_len, V) -> encoder folds channels into batch -> (B*V, n_patches, d)
        emb, _, _ = self.patch_embedding(segment, add_pos=True)
        return self.encoder(emb)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # x: (B, input_len + future_len, V). Split into NON-OVERLAPPING history | future.
        # history = the full input_len window (= the downstream forecaster's context);
        # future (target) = the next future_len rows.
        hist = x[:, : self.config.input_len, :]
        fut = x[:, self.config.input_len : self.config.span_len, :]
        # Each window gets its own input-window instance norm (mean+std) — the same RevIN
        # the downstream forecaster applies to its input window (repo A-class convention).
        hist_norm, _, _ = masked_normalize(hist)
        fut_norm, _, _ = masked_normalize(fut)

        hist_tokens = self._encode(hist_norm)            # (B*V, num_patches, d), online branch
        summary = hist_tokens[:, -1, :]                  # last history token = history summary
        fut_tokens = self._encode(fut_norm).detach()     # (B*V, n_fut, d), stop-gradient target
        pred_tokens = self.predictor(summary)            # (B*V, n_fut, d)

        loss = self.loss_fn(fut_tokens, pred_tokens)
        return {"loss": loss}


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
    total = 0.0
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
        batch_size = int(batch_x.size(0))
        total += float(out["loss"].detach()) * batch_size
        count += batch_size
        if max_steps is not None and step >= max_steps:
            break
    return {"loss": total / max(count, 1)}


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
    # Official SimTS optimizer: SGD, two param groups — encoder at `lr`, predictor at
    # `predictor_lr_mult * lr` (official 0.0001 * lr, which nearly freezes the predictor;
    # kept faithful per docs/check/SimTS.md). No scheduler (official commented its out).
    optimizer = torch.optim.SGD(
        [
            {"params": model.encoder_parameters(), "lr": config.learning_rate},
            {"params": list(model.predictor_parameters()), "lr": config.predictor_lr_mult * config.learning_rate},
        ],
        lr=config.learning_rate,
    )
    out_dir = pretrain_run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (out_dir / "epoch_losses.jsonl").write_text("")
    with (out_dir / "epoch_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "elapsed_sec", "train_loss", "val_loss", "best_val_loss", "checkpoint_saved"],
        )
        writer.writeheader()

    best = float("inf")
    best_path = out_dir / "ckpt_best.pth"
    # Full epoch budget, no early stop (repo budget); best-val checkpoint. Completion is the
    # `done` sentinel written after the loop (model_audit §4). Official runs an early-stop on
    # train loss, but the repo runs the full budget for resume safety (same as SimMTM).
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_losses = _run_epoch(model, train_loader, device, optimizer, max_steps=max_train_steps)
        with torch.no_grad():
            val_losses = _run_epoch(model, val_loader, device, max_steps=max_val_steps)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train {train_losses['loss']:.6f} | val {val_losses['loss']:.6f} | {elapsed:.1f}s"
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
            "val_loss": val_losses["loss"],
            "best_val_loss": best,
            "checkpoint_saved": saved,
        }
        with (out_dir / "epoch_losses.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        with (out_dir / "epoch_losses.csv").open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writerow(row)
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
    parser.add_argument("--patch_len", type=int, default=12)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--checkpoint_every", type=int, default=10)
    parser.add_argument(
        "--future_len",
        type=int,
        default=96,
        help="length of the future (target) window, predicted in latent space from the "
        "history summary. history = the full input_len window; future is the next future_len "
        "rows (non-overlapping). Default 96 = input_len (input 96 / target 96).",
    )
    parser.add_argument(
        "--predictor_lr_mult",
        type=float,
        default=0.0001,
        help="predictor param-group lr = predictor_lr_mult * learning_rate (official 0.0001).",
    )
    parser.add_argument("--predictor_dropout", type=float, default=0.25, help="official LinearPred dropout")
    parser.add_argument(
        "--hierarchical",
        type=int,
        default=0,
        help="1 = TS2Vec-style multi-scale cosine loss; 0 = plain cosine (official default).",
    )
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
        head_dropout=args.head_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        future_len=args.future_len,
        predictor_lr_mult=args.predictor_lr_mult,
        predictor_dropout=args.predictor_dropout,
        hierarchical=bool(args.hierarchical),
        data_fine_dir=args.data_fine_dir,
        device=args.device,
        seed=args.seed,
    )
    path = pretrain(config, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
