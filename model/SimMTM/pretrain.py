from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from model._test_io import write_done
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from model._baseline_forecast import (
    PROJECT_ROOT,
    PRETRAIN_ROOT,
    denormalize,
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
    positive_nums: int
    mask_rate: float
    temperature: float
    lm: int = 3
    mask_distribution: str = "geometric"
    select_channels: float = 1.0
    data_fine_dir: str = "data/fine"
    device: str = "auto"
    seed: int = 2023

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


def _split_path(config: PretrainConfig, split: str) -> Path:
    return PROJECT_ROOT / config.data_fine_dir / config.dataset_name / split / "data.npy"


def geom_noise_mask_single(length: int, lm: int, masking_ratio: float) -> np.ndarray:
    """Markov-chain geometric mask (faithful copy of official SimMTM `geom_noise_mask_single`).

    Returns a boolean keep-mask of length `length` where True keeps the value and the
    masked / kept run-lengths follow geometric distributions of mean `lm`.
    """
    keep_mask = np.ones(length, dtype=bool)
    p_m = 1 / lm
    p_u = p_m * masking_ratio / (1 - masking_ratio)
    p = [p_m, p_u]
    state = int(np.random.rand() > masking_ratio)  # 0 = masking, 1 = keeping
    for i in range(length):
        keep_mask[i] = state
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask


def make_keep_mask(
    n_views: int,
    batch_size: int,
    input_len: int,
    num_features: int,
    config: PretrainConfig,
    device: torch.device,
) -> torch.Tensor:
    """Build a float keep-mask `(n_views * batch_size, input_len, num_features)` (1 = keep).

    Mirrors official `utils.augmentations.masked_data` masking semantics: a single
    Markov chain spans the flattened `(n_views*bs, num_features, input_len)` tensor, then is
    laid back to `(..., input_len, num_features)`.
    """
    rows = n_views * batch_size
    if config.mask_distribution == "geometric":
        flat = geom_noise_mask_single(rows * num_features * input_len, config.lm, config.mask_rate)
        keep = flat.reshape(rows, num_features, input_len).transpose(0, 2, 1)
    elif config.mask_distribution == "random":
        # point-wise i.i.d. Bernoulli — reproduces the pre-audit (2026-06-01) behaviour.
        keep = np.random.rand(rows, input_len, num_features) > config.mask_rate
    else:
        raise ValueError(f"Unknown mask_distribution: {config.mask_distribution}")
    return torch.from_numpy(np.ascontiguousarray(keep)).to(device=device, dtype=torch.float32)


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
            raise ValueError(f"Not enough rows for SimMTM pretrain windows: {path}")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        x = np.array(
            self.data[index:index + self.config.input_len],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x)


class PoolerHead(nn.Module):
    """Series-wise representation head (official `Pooler_Head`)."""

    def __init__(self, num_patches: int, d_model: int, head_dropout: float):
        super().__init__()
        hidden = max(16, num_patches * d_model // 2)
        self.pooler = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(num_patches * d_model, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 128),
            nn.Dropout(head_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pooler(x)


class ReconstructionHead(nn.Module):
    """Flatten reconstruction head (patch analog of official `Flatten_Head`)."""

    def __init__(self, num_patches: int, d_model: int, input_len: int, head_dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(num_patches * d_model, input_len),
            nn.Dropout(head_dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AutomaticWeightedLoss(nn.Module):
    """Uncertainty-based multi-task weighting (faithful copy of official `AutomaticWeightedLoss`)."""

    def __init__(self, num: int = 2):
        super().__init__()
        self.params = nn.Parameter(torch.ones(num))

    def forward(self, *losses: torch.Tensor) -> torch.Tensor:
        total = 0.0
        for i, loss in enumerate(losses):
            total = total + 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return total


class PretrainModel(nn.Module):
    """SimMTM pretrain: masked multi-view → series-wise contrastive → similarity-weighted
    aggregation → reconstruct the original (un-masked) series. Encoder is the repo shared
    patch encoder (`model.conv_encoder`); pretext semantics follow official SimMTM."""

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
        self.pooler = PoolerHead(config.num_patches, config.d_model, config.head_dropout)
        self.reconstruction = ReconstructionHead(
            config.num_patches, config.d_model, config.input_len, config.head_dropout
        )
        self.awl = AutomaticWeightedLoss(2)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, L, V) -> encoder folds channels into batch -> (B*V, num_patches, d_model)
        emb, _, _ = self.patch_embedding(x, add_pos=True)
        enc = self.encoder(emb)
        pooled = self.pooler(enc)
        return enc, pooled

    def _contrastive(
        self, pooled: torch.Tensor, batch_size: int, n_views: int, num_features: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # pooled rows are ordered (view, sample, var): index = (view*bs + sample)*V + var.
        # Positives = same (sample, var) across the n_views (original + masked) blocks.
        pooled = F.normalize(pooled, dim=1)
        similarity = pooled @ pooled.T  # cosine, (N, N)
        n = similarity.size(0)
        device = pooled.device
        sample = (torch.arange(n_views * batch_size, device=device) % batch_size).repeat_interleave(num_features)
        var = torch.arange(num_features, device=device).repeat(n_views * batch_size)
        ids = sample * num_features + var
        eye = torch.eye(n, dtype=torch.bool, device=device)
        positive_mask = ids[:, None].eq(ids[None, :]) & ~eye
        negative_mask = ~ids[:, None].eq(ids[None, :]) & ~eye
        n_pos = n_views - 1  # uniform: each instance has exactly one positive per other view
        positives = similarity[positive_mask].view(n, n_pos)
        negatives = similarity[negative_mask].view(n, -1)
        logits = torch.cat([positives, negatives], dim=-1)
        y_true = torch.cat(
            [torch.ones(n, n_pos, device=device), torch.zeros(n, negatives.size(1), device=device)],
            dim=-1,
        )
        log_prob = torch.log_softmax(logits / self.config.temperature, dim=-1)
        # Official uses KLDivLoss(batchmean) with 0/1 targets; for binary targets this equals
        # the negative mean log-prob mass on positives (0*log0 = 0 convention) — same numerics,
        # no -inf from log(0).
        loss_cl = -(y_true * log_prob).sum(dim=-1).mean()
        return loss_cl, similarity

    def _aggregate(self, similarity: torch.Tensor, enc: torch.Tensor) -> torch.Tensor:
        # Reconstruct each series from a softmax-similarity-weighted mixture of all OTHER series.
        n = similarity.size(0)
        weights = similarity / self.config.temperature
        weights = weights - torch.eye(n, device=similarity.device) * 1e12
        weights = torch.softmax(weights, dim=-1)
        agg = weights @ enc.reshape(n, -1)
        return agg.reshape_as(enc)

    def _reconstruct(self, agg: torch.Tensor, batch_om: int, num_features: int) -> torch.Tensor:
        # agg: (batch_om*V, num_patches, d_model) -> (batch_om, L, V)
        flat = self.reconstruction(agg)  # (batch_om*V, input_len)
        flat = flat.reshape(batch_om, num_features, self.config.input_len)
        return flat.permute(0, 2, 1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Official `select_channels`: randomly subsample channels per batch during pretrain.
        # SimMTM's series-wise contrastive builds an (N×N) matrix with N=(positive_nums+1)·bs·C,
        # so memory is O(C²); high-channel datasets (electricity/traffic/PEMS) must subsample to
        # avoid OOM. The encoder is channel-independent so a per-step channel subset still yields a
        # valid per-series encoder; downstream finetune uses all channels.
        if self.config.select_channels < 1.0 and x.shape[2] > 1:
            full_c = x.shape[2]
            keep_c = max(1, int(full_c * self.config.select_channels))
            idx = torch.randperm(full_c, device=x.device)[:keep_c]
            x = x.index_select(2, idx)

        batch_size, input_len, num_features = x.shape
        positive_nums = self.config.positive_nums
        n_views = positive_nums + 1

        keep = make_keep_mask(positive_nums, batch_size, input_len, num_features, self.config, x.device)
        masked = x.repeat(positive_nums, 1, 1) * keep
        # om-batch = [original (kept whole), positive_nums masked views]
        x_om = torch.cat([x, masked], dim=0)
        keep_om = torch.cat([torch.ones_like(x), keep], dim=0)

        x_norm, means, stdev = masked_normalize(x_om, mask=keep_om)
        enc, pooled = self._encode(x_norm)

        loss_cl, similarity = self._contrastive(pooled, batch_size, n_views, num_features)
        agg = self._aggregate(similarity, enc)
        pred = self._reconstruct(agg, n_views * batch_size, num_features)
        pred = denormalize(pred, means, stdev)

        # Reconstruction is supervised only on the original (un-masked) series (official).
        loss_rb = F.mse_loss(pred[:batch_size], x)
        loss = self.awl(loss_cl, loss_rb)
        return {"loss": loss, "loss_cl": loss_cl, "loss_rb": loss_rb}


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
    totals = {"loss": 0.0, "loss_cl": 0.0, "loss_rb": 0.0}
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
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    # Official SimMTM pretrain schedules lr with a parameter-less CosineAnnealingLR over the
    # full epoch budget (no exponential `lr_decay` knob like SDTA) — stepped once per epoch.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs)
    out_dir = pretrain_run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (out_dir / "epoch_losses.jsonl").write_text("")
    with (out_dir / "epoch_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "elapsed_sec", "train_loss", "train_cl", "train_rb", "val_loss", "val_cl", "val_rb", "best_val_loss", "checkpoint_saved"],
        )
        writer.writeheader()

    best = float("inf")
    best_path = out_dir / "ckpt_best.pth"
    # Official SimMTM pretrain runs the full epoch budget (no early stop), tracking best-val for
    # the checkpoint. Completion is marked by the `done` sentinel written after the loop
    # (model_audit §4), so early stopping would no longer break resume — but full budget is kept
    # to stay faithful to official and because it never yields a worse best checkpoint.
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_losses = _run_epoch(model, train_loader, device, optimizer, max_steps=max_train_steps)
        scheduler.step()
        with torch.no_grad():
            val_losses = _run_epoch(model, val_loader, device, max_steps=max_val_steps)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train {train_losses['loss']:.6f} (cl {train_losses['loss_cl']:.6f}, rb {train_losses['loss_rb']:.6f}) | "
            f"val {val_losses['loss']:.6f} | {elapsed:.1f}s"
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
            "train_cl": train_losses["loss_cl"],
            "train_rb": train_losses["loss_rb"],
            "val_loss": val_losses["loss"],
            "val_cl": val_losses["loss_cl"],
            "val_rb": val_losses["loss_rb"],
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
    parser.add_argument("--positive_nums", type=int, default=3)
    parser.add_argument("--mask_rate", type=float, default=0.5)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--lm", type=int, default=3, help="average geometric masking run length")
    parser.add_argument(
        "--mask_distribution",
        default="geometric",
        choices=["geometric", "random"],
        help="geometric = official Markov-chain mask; random = point-wise i.i.d. Bernoulli (pre-audit)",
    )
    parser.add_argument(
        "--select_channels",
        type=float,
        default=1.0,
        help="official: fraction of channels randomly kept per pretrain batch (1.0=all). "
        "Bounds the O(C^2) contrastive memory for high-channel datasets.",
    )
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2023)
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
        positive_nums=args.positive_nums,
        mask_rate=args.mask_rate,
        temperature=args.temperature,
        lm=args.lm,
        mask_distribution=args.mask_distribution,
        select_channels=args.select_channels,
        data_fine_dir=args.data_fine_dir,
        device=args.device,
        seed=args.seed,
    )
    path = pretrain(config, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
