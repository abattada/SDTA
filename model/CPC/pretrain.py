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
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int
    num_workers: int
    checkpoint_every: int
    # --- CPC-specific pretext knobs ---
    prediction_steps: int  # K: number of future patch-steps the InfoNCE predicts (CPC k=1..K)
    num_negatives: int  # M: InfoNCE negatives drawn per anchor from the in-batch latent pool
    # --- shared memory-bounding knob (same semantics as SimMTM) ---
    select_channels: float = 1.0
    data_fine_dir: str = "data/fine"
    device: str = "auto"
    seed: int = 2023

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
            raise ValueError(f"Not enough rows for CPC pretrain windows: {path}")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        x = np.array(
            self.data[index:index + self.config.input_len],
            dtype=np.float32,
            copy=True,
        )
        return torch.from_numpy(x)


class PretrainModel(nn.Module):
    """Contrastive Predictive Coding (van den Oord et al., 2018) on the repo shared patch encoder.

    Mapping onto the fixed-encoder protocol (every module below is the shared
    one; only the CPC-specific prediction heads are extra):
      - g_enc = `patch_embedding.value_embedding` — local per-patch latent z_t (NO position
        encoding, so the InfoNCE target carries no positional shortcut).
      - g_ar  = `encoder` (shared TransformerEncoder) run with a CAUSAL mask — autoregressive
        context c_t = g_ar(z_{<=t}); only sees the past.
      - W_k   = `predictors[k-1]` (bias-free Linear d->d) — the discarded prediction heads.
    InfoNCE (log-bilinear, faithful CPC: no L2-norm, no temperature): for each anchor position
    t and step k, score the true future latent z_{t+k} against `num_negatives` latents sampled
    from the in-batch pool. Both `patch_embedding` and `encoder` transfer to forecasting (run
    NON-causal there, TimeDART-style); `predictors` are dropped.
    """

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
        self.predictors = nn.ModuleList(
            [nn.Linear(config.d_model, config.d_model, bias=False) for _ in range(config.prediction_steps)]
        )

    def _encode(self, x_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x_norm: (B, L, V) -> patch encoder folds channels into batch -> (B*V, N, d_model).
        patches, _ = self.patch_embedding.patchify(x_norm)
        z = self.patch_embedding.value_embedding(patches)  # local target latents, no pos/dropout
        pos = self.patch_embedding.position_encoding(z)
        ctx_in = self.patch_embedding.dropout(z + pos)  # context path mirrors PatchEmbedding(add_pos=True)
        c = self.encoder(ctx_in, is_causal=True)  # autoregressive context (past-only)
        return z, c

    def _info_nce(self, z: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bc, n, d = z.shape
        device = z.device
        m = self.config.num_negatives
        max_k = min(self.config.prediction_steps, n - 1)
        # Negative pool = EVERY latent in the batch (all sequences × all positions). CPC draws its
        # N−1 negatives from the minibatch marginal p(z) ("mixed" negatives — the audio default in
        # van den Oord 2018 §3.1 / Table 2); sampling uniformly over this pool realises that marginal.
        # Latents are NOT detached (pool is a view of z) → gradient flows through the negatives, so
        # the encoder is trained to push random (context, latent) pairs apart — faithful to InfoNCE.
        pool = z.reshape(bc * n, d)
        pool_size = pool.shape[0]

        seq_base = torch.arange(bc, device=device).unsqueeze(1) * n  # (bc, 1) flat offset per sequence
        losses = []
        correct = 0
        total = 0
        for k in range(1, max_k + 1):
            n_anchor = n - k
            pred = self.predictors[k - 1](c[:, :n_anchor, :])  # (bc, n_anchor, d) = W_k c_t
            pos_z = z[:, k:, :]  # (bc, n_anchor, d) = z_{t+k}
            a = bc * n_anchor
            pred_flat = pred.reshape(a, d)
            pos_logit = (pred_flat * pos_z.reshape(a, d)).sum(-1, keepdim=True)  # (a, 1)

            # M i.i.d. uniform draws per anchor (with replacement = the i.i.d. InfoNCE proposal),
            # freshly resampled every forward from the global RNG (seeded → reproducible run-to-run,
            # random step-to-step). M is fixed (num_negatives), decoupled from batch size.
            neg_idx = torch.randint(pool_size, (a, m), device=device)  # (a, M)
            neg_z = pool[neg_idx]  # (a, M, d)
            neg_logit = torch.einsum("ad,amd->am", pred_flat, neg_z)  # (a, M)
            # Mask the rare draw that lands on the anchor's OWN positive latent z_{t+k} (→ -inf,
            # dropped from the softmax) so the positive can never appear as one of its own negatives.
            pos_flat = (seq_base + torch.arange(k, n, device=device).unsqueeze(0)).reshape(a, 1)
            neg_logit = neg_logit.masked_fill(neg_idx.eq(pos_flat), float("-inf"))

            logits = torch.cat([pos_logit, neg_logit], dim=1)  # (a, 1+M), positive at column 0
            target = torch.zeros(a, dtype=torch.long, device=device)
            losses.append(F.cross_entropy(logits, target))
            correct += int((logits.argmax(dim=1) == 0).sum())
            total += a

        if not losses:
            raise ValueError(
                f"CPC needs num_patches > 1 to predict a future step (got num_patches={n})."
            )
        loss = torch.stack(losses).mean()
        acc = torch.tensor(correct / max(total, 1), device=device)
        return loss, acc

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        # Optional channel subsample — CPC DEFAULTS TO 1.0 (all channels) and no CPC cohort sets it.
        # CPC's InfoNCE memory is O(anchors x M), tractable at full channels (~7GB/job on the worst
        # dataset PEMS07 C=883, bs=16, measured), unlike SimMTM's O(C^2) similarity matrix which is
        # forced to subsample. Kept only as an escape hatch (encoder is channel-independent so a
        # per-step channel subset is a valid per-series encoder; finetune always uses all channels).
        if self.config.select_channels < 1.0 and x.shape[2] > 1:
            full_c = x.shape[2]
            keep_c = max(1, int(full_c * self.config.select_channels))
            idx = torch.randperm(full_c, device=x.device)[:keep_c]
            x = x.index_select(2, idx)

        x_norm, _, _ = masked_normalize(x)  # input-window instance norm (mean+std), A-class fair
        z, c = self._encode(x_norm)
        loss, acc = self._info_nce(z, c)
        return {"loss": loss, "loss_cpc": loss, "acc": acc}


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
    totals = {"loss": 0.0, "loss_cpc": 0.0, "acc": 0.0}
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
    # CPC (van den Oord 2018) trains with Adam and no LR schedule; we keep the repo's
    # parameter-less CosineAnnealingLR over the full epoch budget for parity with the other
    # SSL baselines' pretrain recipe (SimMTM/TimeMAE) — there is no CPC-specific schedule to match.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train_epochs)
    out_dir = pretrain_run_dir(config)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (out_dir / "epoch_losses.jsonl").write_text("")
    with (out_dir / "epoch_losses.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "elapsed_sec", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_loss", "checkpoint_saved"],
        )
        writer.writeheader()

    best = float("inf")
    best_path = out_dir / "ckpt_best.pth"
    # Full epoch budget, no early stop (best-val checkpoint tracked) — keeps the `done` sentinel
    # resume contract (model_audit §4) intact, same as SimMTM.
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_losses = _run_epoch(model, train_loader, device, optimizer, max_steps=max_train_steps)
        scheduler.step()
        with torch.no_grad():
            val_losses = _run_epoch(model, val_loader, device, max_steps=max_val_steps)
        elapsed = time.time() - start
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train {train_losses['loss']:.6f} (acc {train_losses['acc']:.3f}) | "
            f"val {val_losses['loss']:.6f} (acc {val_losses['acc']:.3f}) | {elapsed:.1f}s"
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
            "train_acc": train_losses["acc"],
            "val_loss": val_losses["loss"],
            "val_acc": val_losses["acc"],
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
    parser.add_argument(
        "--prediction_steps",
        type=int,
        default=4,
        help="K: number of future patch-steps predicted by InfoNCE (CPC k=1..K). "
        "Aligns with the SDTA multi-window count W for the diffusion-vs-contrastive head-to-head.",
    )
    parser.add_argument(
        "--num_negatives",
        type=int,
        default=64,
        help="M: InfoNCE negatives sampled per anchor from the in-batch latent pool. "
        "Decoupled from batch_size on purpose (a fixed objective knob, like SimMTM positive_nums).",
    )
    parser.add_argument(
        "--select_channels",
        type=float,
        default=1.0,
        help="fraction of channels randomly kept per pretrain batch (1.0=all, the CPC default). "
        "Escape hatch only — CPC pretrains on ALL channels (full-channel ~7GB/job is tractable; no "
        "CPC cohort sets this, unlike SimMTM which needs it for its O(C^2) contrastive matrix).",
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
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        patience=args.patience,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        prediction_steps=args.prediction_steps,
        num_negatives=args.num_negatives,
        select_channels=args.select_channels,
        data_fine_dir=args.data_fine_dir,
        device=args.device,
        seed=args.seed,
    )
    path = pretrain(config, max_train_steps=args.max_train_steps, max_val_steps=args.max_val_steps)
    print(f"Best checkpoint: {path}")


if __name__ == "__main__":
    main()
