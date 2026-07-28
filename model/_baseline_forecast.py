from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ._test_io import write_done, write_results_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "train"
TEST_ROOT = PROJECT_ROOT / "outputs" / "test"
PRETRAIN_ROOT = PROJECT_ROOT / "outputs" / "pretrain"


@dataclass
class ForecastConfig:
    dataset_name: str
    model_id: str
    features: str
    input_len: int
    pred_len: int
    enc_in: int
    d_model: int = 32
    d_ff: int = 64
    n_heads: int = 4
    e_layers: int = 1
    d_layers: int = 1
    dropout: float = 0.1
    head_dropout: float = 0.0
    patch_len: int = 16
    stride: int = 8
    moving_avg: int = 25
    seg_len: int = 25
    d_hidden: int = 128
    individual: bool = False
    learning_rate: float = 0.0001
    batch_size: int = 32
    train_epochs: int = 10
    patience: int = 3
    lradj: str = "type1"
    lr_decay: float = 0.9
    pct_start: float = 0.3
    num_workers: int = 0
    data_fine_dir: str = "data/fine"
    pretrain_run: str = ""
    train_run: str = "run"
    load_pretrain_weights: bool = False
    device: str = "auto"
    seed: int = 2021
    fc_dropout: float = 0.0
    decomposition: bool = False
    kernel_size: int = 25
    channel_independence: int = 1
    down_sampling_layers: int = 1
    down_sampling_window: int = 2
    down_sampling_method: str = "avg"
    decomp_method: str = "moving_avg"
    top_k: int = 5
    use_norm: bool = True
    use_future_temporal_feature: bool = False


def resolve_device(raw: str = "auto") -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(raw)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def masked_normalize(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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


def _split_dir(data_fine_dir: str, dataset_name: str, split: str) -> Path:
    return PROJECT_ROOT / data_fine_dir / dataset_name / split


def _read_valid_offset(data_fine_dir: str, dataset_name: str, split: str) -> int:
    meta_path = _split_dir(data_fine_dir, dataset_name, split) / "metadata.json"
    if not meta_path.exists():
        return 0
    return int(json.loads(meta_path.read_text()).get("valid_offset_in_file", 0))


def _load_split_array(data_fine_dir: str, dataset_name: str, split: str) -> np.ndarray:
    path = _split_dir(data_fine_dir, dataset_name, split) / "data.npy"
    if not path.exists():
        raise FileNotFoundError(f"Split not found: {path}")
    return np.load(path, mmap_mode="r")


def select_features(data: np.ndarray, features: str, enc_in: int) -> np.ndarray:
    if features in {"M", "MS"}:
        return data[:, :enc_in]
    if features == "S":
        return data[:, -1:].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported features mode: {features}")


class ForecastWindowDataset(Dataset):
    def __init__(self, config: ForecastConfig, split: str):
        self.config = config
        self.split = "validation" if split == "val" else split
        data = _load_split_array(config.data_fine_dir, config.dataset_name, self.split)
        self.data = select_features(data, config.features, config.enc_in)
        valid_offset = _read_valid_offset(config.data_fine_dir, config.dataset_name, self.split)
        self.start_offset = max(valid_offset - config.input_len, 0)
        usable_rows = len(self.data) - self.start_offset
        self.window_count = usable_rows - config.input_len - config.pred_len + 1
        if self.window_count <= 0:
            raise ValueError(
                "Not enough rows for forecasting windows: "
                f"split={self.split}, rows={len(self.data)}, start_offset={self.start_offset}, "
                f"input_len={config.input_len}, pred_len={config.pred_len}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_begin = self.start_offset + index
        x_end = x_begin + self.config.input_len
        y_end = x_end + self.config.pred_len
        x = np.array(self.data[x_begin:x_end], dtype=np.float32, copy=True)
        y = np.array(self.data[x_end:y_end], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)


def pretrain_tag(config: ForecastConfig, supports_pretrain: bool) -> str:
    if not supports_pretrain or not config.load_pretrain_weights or not config.pretrain_run:
        return "no_pretrain"
    return config.pretrain_run


def train_run_dir(config: ForecastConfig, supports_pretrain: bool) -> Path:
    return (
        TRAIN_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / f"pre_{pretrain_tag(config, supports_pretrain)}"
        / config.train_run
        / f"pl{config.pred_len}"
    )


def checkpoint_path(config: ForecastConfig, supports_pretrain: bool) -> Path:
    return train_run_dir(config, supports_pretrain) / "checkpoint.pth"


def result_dir(config: ForecastConfig, supports_pretrain: bool) -> Path:
    return (
        TEST_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / f"pre_{pretrain_tag(config, supports_pretrain)}"
        / config.train_run
    )


def pretrain_checkpoint_path(config: ForecastConfig) -> Path:
    return (
        PRETRAIN_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / config.pretrain_run
        / "ckpt_best.pth"
    )


def _make_loaders(config: ForecastConfig) -> tuple[DataLoader, DataLoader]:
    train = ForecastWindowDataset(config, "train")
    val = ForecastWindowDataset(config, "validation")
    return (
        DataLoader(
            train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            drop_last=True,
        ),
        DataLoader(
            val,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        ),
    )


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    step_scheduler_per_batch: bool = False,
    max_steps: int | None = None,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    for step, (batch_x, batch_y) in enumerate(loader, start=1):
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            mae = torch.mean(torch.abs(pred - batch_y))
            if is_train:
                loss.backward()
                optimizer.step()
                if scheduler is not None and step_scheduler_per_batch:
                    scheduler.step()
        batch_size = int(batch_x.size(0))
        total_loss += float(loss.detach()) * batch_size
        total_mae += float(mae.detach()) * batch_size
        total_count += batch_size
        if max_steps is not None and step >= max_steps:
            break
    denom = max(total_count, 1)
    return {"mse": total_loss / denom, "mae": total_mae / denom}


def _make_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ForecastConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.lradj in {"step", "TST"}:
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=config.pct_start,
            epochs=config.train_epochs,
            max_lr=config.learning_rate,
        )
    if config.lradj == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_decay)
    return None


def _adjust_lr_after_epoch(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: ForecastConfig,
) -> None:
    if config.lradj in {"step", "TST"}:
        return
    if config.lradj == "type1":
        lr = config.learning_rate * (0.5 ** (epoch - 1))
    elif config.lradj == "type3":
        lr = config.learning_rate if epoch < 3 else config.learning_rate * (0.9 ** (epoch - 3))
    elif config.lradj == "constant":
        lr = config.learning_rate
    elif config.lradj == "decay":
        lr = config.learning_rate * (config.lr_decay ** (epoch - 1))
    elif config.lradj == "exp" and scheduler is not None:
        scheduler.step()
        return
    else:
        return
    for group in optimizer.param_groups:
        group["lr"] = lr
    print(f"Updating learning rate to {lr}")


def _reset_logs(train_dir: Path, config: ForecastConfig) -> None:
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    (train_dir / "epoch_metrics.jsonl").write_text("")
    with (train_dir / "epoch_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "lr",
                "elapsed_sec",
                "train_mse",
                "train_mae",
                "val_mse",
                "val_mae",
                "best_val_mse",
                "checkpoint_saved",
            ],
        )
        writer.writeheader()


def _append_epoch_log(train_dir: Path, row: dict[str, float | int | bool]) -> None:
    with (train_dir / "epoch_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(row) + "\n")
    with (train_dir / "epoch_metrics.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writerow(row)


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ForecastConfig,
    path: Path,
    supports_pretrain: bool,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "supports_pretrain": supports_pretrain,
            "pretrain_checkpoint": (
                str(pretrain_checkpoint_path(config)) if pretrain_tag(config, supports_pretrain) != "no_pretrain" else ""
            ),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )


def load_matched_pretrain_weights(model: nn.Module, config: ForecastConfig, device: torch.device) -> None:
    if not config.load_pretrain_weights or not config.pretrain_run:
        return
    path = pretrain_checkpoint_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    source = payload.get("encoder_state_dict") or payload.get("model_state_dict") or payload
    target = model.state_dict()
    matched = {
        key: value
        for key, value in source.items()
        if key in target and tuple(value.shape) == tuple(target[key].shape)
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(
        f"Loaded pretrain weights from {path}: matched={len(matched)}, "
        f"missing_after_load={len(missing)}, unexpected={len(unexpected)}"
    )


ModelFactory = Callable[[ForecastConfig], nn.Module]


def train_forecaster(
    config: ForecastConfig,
    factory: ModelFactory,
    supports_pretrain: bool,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> Path:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    config.device = str(device)
    train_loader, val_loader = _make_loaders(config)
    model = factory(config).to(device)
    if supports_pretrain:
        load_matched_pretrain_weights(model, config, device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _make_scheduler(optimizer, config, len(train_loader))
    ckpt = checkpoint_path(config, supports_pretrain)
    train_dir = ckpt.parent
    _reset_logs(train_dir, config)

    best_val = float("inf")
    stale = 0
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_metrics = _run_epoch(
            model,
            train_loader,
            device,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            step_scheduler_per_batch=config.lradj in {"step", "TST"},
            max_steps=max_train_steps,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model, val_loader, device, criterion, max_steps=max_val_steps
            )
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train mse {train_metrics['mse']:.6f} mae {train_metrics['mae']:.6f} | "
            f"val mse {val_metrics['mse']:.6f} mae {val_metrics['mae']:.6f} | "
            f"lr {lr:.8f} | {elapsed:.1f}s"
        )
        saved = False
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            stale = 0
            saved = True
            _save_checkpoint(
                model, optimizer, epoch, config, ckpt, supports_pretrain, train_metrics, val_metrics
            )
        else:
            stale += 1
        _append_epoch_log(
            train_dir,
            {
                "epoch": epoch,
                "lr": lr,
                "elapsed_sec": elapsed,
                "train_mse": train_metrics["mse"],
                "train_mae": train_metrics["mae"],
                "val_mse": val_metrics["mse"],
                "val_mae": val_metrics["mae"],
                "best_val_mse": best_val,
                "checkpoint_saved": saved,
            },
        )
        _adjust_lr_after_epoch(optimizer, scheduler, epoch, config)
        if not saved and stale >= config.patience:
            print(f"Early stopping after {config.patience} stale validation epochs.")
            break
    (train_dir / "training_summary.json").write_text(
        json.dumps({"best_checkpoint": str(ckpt), "best_val_mse": best_val}, indent=2) + "\n"
    )
    write_done(train_dir)  # last write: train stage complete
    return ckpt


def _load_checkpoint_config(path: Path) -> ForecastConfig:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return ForecastConfig(**payload["config"])


def _resolve_checkpoint(
    model_id: str,
    dataset_name: str,
    input_len: int,
    pred_len: int,
    train_run: str,
    pretrain_name: str,
) -> Path:
    path = (
        TRAIN_ROOT
        / model_id
        / dataset_name
        / f"il{input_len}"
        / f"pre_{pretrain_name}"
        / train_run
        / f"pl{pred_len}"
        / "checkpoint.pth"
    )
    if path.exists():
        return path
    root = TRAIN_ROOT / model_id / dataset_name
    matches = sorted(root.glob(f"il*/pre_{pretrain_name}/{train_run}/pl{pred_len}/checkpoint.pth"))
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Train checkpoint not found: {path}")
    joined = "\n".join(str(item) for item in matches)
    raise RuntimeError(f"Multiple train checkpoints matched run {train_run!r}:\n{joined}")


def test_forecaster(
    template: ForecastConfig,
    factory: ModelFactory,
    supports_pretrain: bool,
    pred_lens: list[int],
    save_predictions: bool = False,
    max_test_steps: int | None = None,
) -> dict[int, dict[str, float]]:
    device = resolve_device(template.device)
    effective_pretrain = pretrain_tag(template, supports_pretrain)
    out_dir = result_dir(template, supports_pretrain)
    metrics_per_pl: dict[int, dict[str, float]] = {}
    train_config: dict = {}
    pretrain_config: dict | None = None
    for pred_len in pred_lens:
        ckpt = _resolve_checkpoint(
            template.model_id,
            template.dataset_name,
            template.input_len,
            pred_len,
            template.train_run,
            effective_pretrain,
        )
        config = _load_checkpoint_config(ckpt)
        config.pred_len = pred_len
        config.batch_size = template.batch_size
        config.num_workers = template.num_workers
        config.device = str(device)
        dataset = ForecastWindowDataset(config, "test")
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            drop_last=False,
        )
        model = factory(config).to(device)
        payload = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(payload["model_state_dict"], strict=True)
        model.eval()
        criterion = nn.MSELoss()
        with torch.no_grad():
            metrics = _run_epoch(model, loader, device, criterion, max_steps=max_test_steps)
        if save_predictions:
            preds, trues = [], []
            with torch.no_grad():
                for step, (batch_x, batch_y) in enumerate(loader, start=1):
                    pred = model(batch_x.float().to(device)).detach().cpu().numpy()
                    preds.append(pred)
                    trues.append(batch_y.numpy())
                    if max_test_steps is not None and step >= max_test_steps:
                        break
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"pred_pl{pred_len}.npy", np.concatenate(preds, axis=0))
            np.save(out_dir / f"true_pl{pred_len}.npy", np.concatenate(trues, axis=0))
        print(f"pl{pred_len}: MSE {metrics['mse']:.6f}, MAE {metrics['mae']:.6f}")
        metrics_per_pl[pred_len] = metrics
        train_config = asdict(config)
        if supports_pretrain and config.pretrain_run:
            try:
                pre_payload = torch.load(pretrain_checkpoint_path(config), map_location="cpu", weights_only=False)
                pretrain_config = dict(pre_payload.get("config", {}))
            except FileNotFoundError:
                pretrain_config = None

    write_results_file(
        out_dir=out_dir,
        setting=template.train_run,
        model_id=template.model_id,
        dataset_name=template.dataset_name,
        input_len=template.input_len,
        pretrain_run=template.pretrain_run if effective_pretrain != "no_pretrain" else "",
        train_run=template.train_run,
        pred_len_metrics=metrics_per_pl,
        pretrain_config=pretrain_config,
        train_config=train_config,
        test_config={
            "pred_lens": pred_lens,
            "batch_size": template.batch_size,
            "num_workers": template.num_workers,
            "save_predictions": save_predictions,
            "max_test_steps": max_test_steps,
            "seed": template.seed,
        },
    )
    return metrics_per_pl


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--features", default="M")
    parser.add_argument("--input_len", type=int, default=96)
    parser.add_argument("--pred_len", type=int, default=96)
    parser.add_argument("--pred_lens", default="96,192,336,720")
    parser.add_argument("--enc_in", type=int, default=7)
    parser.add_argument("--d_model", type=int, default=32)
    parser.add_argument("--d_ff", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--e_layers", type=int, default=1)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--head_dropout", type=float, default=0.0)
    parser.add_argument("--fc_dropout", type=float, default=0.0)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int)
    parser.add_argument("--moving_avg", type=int, default=25)
    parser.add_argument("--seg_len", type=int, default=25)
    parser.add_argument("--d_hidden", type=int, default=128)
    parser.add_argument("--individual", type=int, default=0)
    parser.add_argument("--learning_rate", "--lr", dest="learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--lradj", default="type1")
    parser.add_argument("--lr_decay", type=float, default=0.9)
    parser.add_argument("--pct_start", type=float, default=0.3)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--pretrain_run", default="")
    parser.add_argument("--run_name", default="run")
    parser.add_argument("--train_run", default="")
    parser.add_argument("--load_pretrain_weights", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--decomposition", type=int, default=0)
    parser.add_argument("--kernel_size", type=int, default=25)
    parser.add_argument("--channel_independence", type=int, default=1)
    parser.add_argument("--down_sampling_layers", type=int, default=1)
    parser.add_argument("--down_sampling_window", type=int, default=2)
    parser.add_argument("--down_sampling_method", default="avg")
    parser.add_argument("--decomp_method", default="moving_avg")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--use_norm", type=int, default=1)
    parser.add_argument("--use_future_temporal_feature", type=int, default=0)
    parser.add_argument("--max_train_steps", type=int)
    parser.add_argument("--max_val_steps", type=int)
    parser.add_argument("--max_test_steps", type=int)
    parser.add_argument("--save_predictions", type=int, default=0)


def config_from_args(args: argparse.Namespace, is_test: bool = False) -> ForecastConfig:
    stride = args.stride if args.stride is not None else args.patch_len
    return ForecastConfig(
        dataset_name=args.dataset,
        model_id=args.model_id,
        features=args.features,
        input_len=args.input_len,
        pred_len=args.pred_len,
        enc_in=args.enc_in,
        d_model=args.d_model,
        d_ff=args.d_ff,
        n_heads=args.n_heads,
        e_layers=args.e_layers,
        d_layers=args.d_layers,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        patch_len=args.patch_len,
        stride=stride,
        moving_avg=args.moving_avg,
        seg_len=args.seg_len,
        d_hidden=args.d_hidden,
        individual=bool(args.individual),
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        patience=args.patience,
        lradj=args.lradj,
        lr_decay=args.lr_decay,
        pct_start=args.pct_start,
        num_workers=args.num_workers,
        data_fine_dir=args.data_fine_dir,
        pretrain_run=args.pretrain_run,
        train_run=args.train_run or args.run_name,
        load_pretrain_weights=bool(args.load_pretrain_weights),
        device=args.device,
        seed=args.seed,
        fc_dropout=args.fc_dropout,
        decomposition=bool(args.decomposition),
        kernel_size=args.kernel_size,
        channel_independence=args.channel_independence,
        down_sampling_layers=args.down_sampling_layers,
        down_sampling_window=args.down_sampling_window,
        down_sampling_method=args.down_sampling_method,
        decomp_method=args.decomp_method,
        top_k=args.top_k,
        use_norm=bool(args.use_norm),
        use_future_temporal_feature=bool(args.use_future_temporal_feature),
    )


def parse_pred_lens(raw: str) -> list[int]:
    return [int(piece) for piece in str(raw).split(",") if piece]


# Architecture fields a pretrained model must take from its pretrain config so the
# finetune encoder shape matches the pretrained weights exactly (otherwise
# load_matched_pretrain_weights silently shape-skips them). Only applied for models
# that opt in via supports_pretrain=True (SimMTM / TimeMAE / ST_MTM among the shared
# _baseline_forecast users); supervised baselines never load a pretrain config.
#
# Core fields are universal and required (a missing one raises — loud desync).
# Optional fields are model-shape-specific and inherited only when the pretrain
# config actually carries them: patch models write patch_len/stride; ST-MTM writes
# kernel_size/seg_len/d_hidden/top_k and has no patches. This lets one inherit path
# serve both without forcing every model to declare fields it does not use.
_PRETRAIN_ARCH_FIELDS = (
    "input_len",
    "enc_in",
    "d_model",
    "d_ff",
    "n_heads",
    "e_layers",
)
_PRETRAIN_ARCH_FIELDS_OPTIONAL = (
    "patch_len",
    "stride",
    "kernel_size",
    "seg_len",
    "d_hidden",
    "top_k",
)


def inherit_pretrain_arch(config: ForecastConfig) -> ForecastConfig:
    """Override `config`'s arch fields from its pretrain run's config.json.

    Mirrors the SDTA / TimeDART / TimeSiam inherit contract: the pretrain phase is
    the single source of truth for encoder shape, so the finetune CLI only supplies
    train-time knobs (lr, dropout, epochs, pred_len). Train-time fields are left
    untouched. A missing *core* field raises so a desync is loud, not silent;
    optional shape-specific fields are inherited only when present.
    """
    from ._config_inherit import inherit_fields, load_pretrain_config

    pretrain_cfg = load_pretrain_config(
        config.model_id, config.dataset_name, config.pretrain_run
    )
    inherited = inherit_fields(pretrain_cfg, list(_PRETRAIN_ARCH_FIELDS))
    for field in _PRETRAIN_ARCH_FIELDS:
        setattr(config, field, int(inherited[field]))
    for field in _PRETRAIN_ARCH_FIELDS_OPTIONAL:
        if field in pretrain_cfg:
            setattr(config, field, int(pretrain_cfg[field]))
    return config


def train_main(factory: ModelFactory, supports_pretrain: bool = False) -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    if supports_pretrain and config.load_pretrain_weights and config.pretrain_run:
        config = inherit_pretrain_arch(config)
    path = train_forecaster(
        config,
        factory,
        supports_pretrain=supports_pretrain,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )
    print(f"Best checkpoint: {path}")


def test_main(factory: ModelFactory, supports_pretrain: bool = False) -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args, is_test=True)
    test_forecaster(
        config,
        factory,
        supports_pretrain=supports_pretrain,
        pred_lens=parse_pred_lens(args.pred_lens),
        save_predictions=bool(args.save_predictions),
        max_test_steps=args.max_test_steps,
    )


def moving_average(x: torch.Tensor, kernel_size: int) -> torch.Tensor:
    front = x[:, 0:1, :].repeat(1, (kernel_size - 1) // 2, 1)
    end = x[:, -1:, :].repeat(1, (kernel_size - 1) // 2, 1)
    x_pad = torch.cat([front, x, end], dim=1)
    return nn.functional.avg_pool1d(x_pad.permute(0, 2, 1), kernel_size, stride=1).permute(0, 2, 1)


def series_decomp(x: torch.Tensor, kernel_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    trend = moving_average(x, kernel_size)
    return x - trend, trend

