"""TimeDART pretraining with SDTA-compatible output layout."""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from model._test_io import write_done
import torch.nn as nn
from torch.utils.data import DataLoader

from .cli import apply_cuda_visible_devices
from .dataset import PretrainWindowDataset
from .timedart import TimeDARTModel
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
    head_dropout: float
    use_norm: bool
    time_steps: int
    scheduler: str
    mask_ratio: float
    learning_rate: float
    batch_size: int
    train_epochs: int
    num_workers: int
    checkpoint_every: int
    lr_decay: float = 0.5
    device: str = "auto"
    seed: int = 2021

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.stride) + 1


def pretrain_run_dir(config: PretrainConfig) -> Path:
    return (
        PRETRAIN_CHECKPOINT_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / config.run_name
    )


def _model_args(config: PretrainConfig) -> SimpleNamespace:
    return SimpleNamespace(
        task_name="pretrain",
        input_len=config.input_len,
        pred_len=config.input_len,
        enc_in=config.enc_in,
        d_model=config.d_model,
        n_heads=config.n_heads,
        e_layers=config.e_layers,
        d_layers=config.d_layers,
        d_ff=config.d_ff,
        dropout=config.dropout,
        head_dropout=config.head_dropout,
        use_norm=config.use_norm,
        patch_len=config.patch_len,
        stride=config.stride,
        time_steps=config.time_steps,
        scheduler=config.scheduler,
        mask_ratio=config.mask_ratio,
        device=torch.device(config.device),
    )


def _make_model(config: PretrainConfig) -> TimeDARTModel:
    resolved_device = resolve_device(config.device)
    config.device = str(resolved_device)
    model = TimeDARTModel(_model_args(config))
    return model.to(resolved_device)


def _make_loader(config: PretrainConfig, split: str, shuffle: bool) -> DataLoader:
    dataset = PretrainWindowDataset(config, split=split)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        drop_last=True,
    )


def _run_epoch(
    model: TimeDARTModel,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0
    device = next(model.parameters()).device
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for step, batch_x in enumerate(loader, start=1):
            batch_x = batch_x.float().to(device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            pred_x = model(batch_x)
            loss = criterion(pred_x, batch_x)
            if is_train:
                loss.backward()
                optimizer.step()
            batch_size = int(batch_x.size(0))
            total_loss += float(loss.detach()) * batch_size
            total_count += batch_size
            if max_steps is not None and step >= max_steps:
                break
    return {"loss": total_loss / max(total_count, 1)}


def _backbone_state_dict(model: TimeDARTModel) -> dict[str, torch.Tensor]:
    prefixes = ("enc_embedding.", "encoder.")
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def _save_checkpoint(
    model: TimeDARTModel,
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
        "setting": config.run_name,
        "config": asdict(config),
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
    (checkpoint_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
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
) -> TimeDARTModel:
    seed_everything(config.seed)
    model = _make_model(config)
    train_loader = _make_loader(config, split="train", shuffle=True)
    val_loader = _make_loader(config, split="validation", shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=config.lr_decay)
    criterion = nn.MSELoss()
    checkpoint_dir = pretrain_run_dir(config)
    _reset_pretrain_logs(checkpoint_dir, config)
    best_val_loss: float | None = None

    print(f"Config: {config}")
    print(f"Device: {next(model.parameters()).device}")
    print(
        f"Train windows: {len(train_loader.dataset)}, "
        f"Validation windows: {len(val_loader.dataset)}"
    )
    print(f"Checkpoint dir: {checkpoint_dir}")

    for epoch in range(1, config.train_epochs + 1):
        start_time = time.time()
        train_losses = _run_epoch(
            model, train_loader, criterion, optimizer=optimizer, max_steps=max_train_steps
        )
        val_losses = _run_epoch(model, val_loader, criterion, max_steps=max_val_steps)
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
    parser.add_argument("--run_name", required=True)
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
    parser.add_argument("--head_dropout", type=float, required=True)
    parser.add_argument("--use_norm", type=int, choices=[0, 1], required=True)
    parser.add_argument("--time_steps", type=int, required=True)
    parser.add_argument("--scheduler", choices=["cosine", "linear"], required=True)
    parser.add_argument("--mask_ratio", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--train_epochs", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--checkpoint_every", type=int, required=True)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument(
        "--cuda_visible_devices",
        "--cuda-visible-devices",
        dest="cuda_visible_devices",
        default=None,
    )
    parser.add_argument("--device", default="auto")
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
        head_dropout=args.head_dropout,
        use_norm=bool(args.use_norm),
        time_steps=args.time_steps,
        scheduler=args.scheduler,
        mask_ratio=args.mask_ratio,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        num_workers=args.num_workers,
        checkpoint_every=args.checkpoint_every,
        lr_decay=args.lr_decay,
        device=args.device,
        seed=args.seed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pretrain TimeDART on data/fine windows")
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
