"""SDTA downstream forecasting (paper: Method — Training Objective and Transfer).

Only the patch embedding + encoder transfer from pretraining
(_load_pretrained_encoder); the diffusion module, decoder, and
temporal-distance code exist only during pretraining and are discarded. The
Forecaster fine-tunes encoder + linear flatten head with per-channel instance
normalization; metrics are MSE/MAE on the globally standardized scale.
Set --load_pretrain_weights 0 for the random-init control (architecture still
inherits from the named pretrain run's config.json, weights stay random;
outputs are tagged pre_no_pretrain); --freeze_encoder 1 is the paper's
frozen-encoder (linear probe) ablation. Architecture fields are inherited from the pretrain run's
config.json, never from the CLI (model/_config_inherit.py).
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
from torch.utils.data import DataLoader

from .dataset import ForecastWindowDataset
from .encoder import EncoderConfig, build_encoder
from .forecast_head import ForecastHead
from .pretrain import denormalize, per_window_normalize
from .utils import (
    PROJECT_ROOT,
    PRETRAIN_CHECKPOINT_ROOT,
    TEST_RESULTS_ROOT,
    TRAIN_CHECKPOINT_ROOT,
    format_param,
    resolve_device,
    seed_everything,
)
from .._test_io import write_done, write_results_file


NO_PRETRAIN_TAG = "no_pretrain"


@dataclass
class ForecastConfig:
    dataset_name: str
    model_id: str
    features: str
    input_len: int
    pred_len: int
    enc_in: int
    patch_len: int
    d_model: int
    n_heads: int
    enc_layers: int
    dec_layers: int
    d_ff: int
    dropout: float
    head_dropout: float
    learning_rate: float
    batch_size: int
    train_epochs: int
    patience: int
    pct_start: float
    lr_decay: float
    lradj: str
    num_workers: int
    pretrain_run: str
    train_run: str
    data_fine_dir: str = "data/fine"
    use_norm: bool = True
    device: str = "auto"
    seed: int = 2021
    # When False, treat the run as the random-init ablation (arch is still
    # read from the named pretrain config, weights are randomly initialized,
    # outputs go to pre_no_pretrain/ for direct comparison).
    load_pretrain_weights: bool = True
    # Linear-probe ablation. 1 = freeze patch_embedding + encoder so only
    # the forecast head trains. Tests representation strength independent of
    # fine-tuning.
    freeze_encoder: bool = False

    @property
    def num_patches(self) -> int:
        return int((self.input_len - self.patch_len) / self.patch_len) + 1


class Forecaster(nn.Module):
    """Pretrain encoder + flatten/MLP forecasting head (PatchTST-style)."""

    def __init__(self, config: ForecastConfig):
        super().__init__()
        resolved_device = resolve_device(config.device)
        config.device = str(resolved_device)
        self.config = config
        self.device = resolved_device

        encoder_config = EncoderConfig(
            input_len=config.input_len,
            patch_len=config.patch_len,
            stride=config.patch_len,
            d_model=config.d_model,
            n_heads=config.n_heads,
            d_ff=config.d_ff,
            e_layers=config.enc_layers,
            dropout=config.dropout,
        )
        self.patch_embedding, self.encoder = build_encoder(encoder_config)
        self.head = ForecastHead(
            num_patches=config.num_patches,
            in_dim=config.d_model,
            pred_len=config.pred_len,
            dec_layers=config.dec_layers,
            dropout=config.head_dropout,
        )
        self.to(self.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        batch_size, _, num_features = x.size()

        if self.config.use_norm:
            x_norm, means, stdev = per_window_normalize(x)
        else:
            x_norm = x
            means = stdev = None

        base_emb, _, _ = self.patch_embedding(x_norm, add_pos=True)
        encoded = self.encoder(base_emb)
        encoded = encoded.reshape(batch_size, num_features, encoded.size(-2), encoded.size(-1))
        forecast = self.head(encoded)
        if means is not None and stdev is not None:
            forecast = denormalize(forecast, means, stdev)
        return forecast[:, -self.config.pred_len :, :]


def train_setting_name(config: ForecastConfig) -> str:
    return config.train_run


def _pretrain_tag(config: ForecastConfig) -> str:
    if not config.pretrain_run or not config.load_pretrain_weights:
        return NO_PRETRAIN_TAG
    return config.pretrain_run


def train_run_dir(config: ForecastConfig) -> Path:
    return (
        TRAIN_CHECKPOINT_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / f"pre_{_pretrain_tag(config)}"
        / train_setting_name(config)
        / f"pl{config.pred_len}"
    )


def checkpoint_path(config: ForecastConfig) -> Path:
    return train_run_dir(config) / "checkpoint.pth"


def result_dir(config: ForecastConfig, result_setting: str | None = None) -> Path:
    """Test result dir (one per train_run, all pred_lens share it)."""
    return (
        TEST_RESULTS_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / f"pre_{_pretrain_tag(config)}"
        / (result_setting or train_setting_name(config))
    )


def pretrain_checkpoint_path(config: ForecastConfig) -> Path | None:
    if not config.pretrain_run:
        return None
    return (
        PRETRAIN_CHECKPOINT_ROOT
        / config.model_id
        / config.dataset_name
        / f"il{config.input_len}"
        / config.pretrain_run
        / "ckpt_best.pth"
    )


def _reset_train_logs(train_dir: Path, config: ForecastConfig) -> None:
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


def _append_train_epoch_log(
    train_dir: Path,
    epoch_record: dict[str, float | int | bool],
) -> None:
    with (train_dir / "epoch_metrics.jsonl").open("a") as handle:
        handle.write(json.dumps(epoch_record) + "\n")
    with (train_dir / "epoch_metrics.csv").open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(epoch_record.keys()))
        writer.writerow(epoch_record)


def _load_pretrained_encoder(model: Forecaster, config: ForecastConfig) -> None:
    if not config.load_pretrain_weights:
        print(
            f"--load_pretrain_weights=0: using arch from '{config.pretrain_run}' "
            "but skipping weight load (random init)."
        )
        return
    path = pretrain_checkpoint_path(config)
    if path is None:
        print("No pretrain checkpoint provided; training from random initialization.")
        return
    if not path.exists():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")
    payload = torch.load(path, map_location=model.device)
    source = payload.get("encoder_state_dict") or payload.get("model_state_dict") or payload
    target = model.state_dict()
    matched: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in source.items():
        if key in target and tuple(value.shape) == tuple(target[key].shape):
            matched[key] = value
        else:
            skipped += 1
    missing, unexpected = model.load_state_dict(matched, strict=False)
    print(
        f"Loaded pretrained weights from {path}: matched={len(matched)}, "
        f"source_skipped={skipped}, missing_after_load={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )


def _make_loaders(config: ForecastConfig) -> tuple[DataLoader, DataLoader]:
    train_dataset = ForecastWindowDataset(config, "train")
    val_dataset = ForecastWindowDataset(config, "validation")
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    return train_loader, val_loader


def _run_epoch(
    model: Forecaster,
    loader: DataLoader,
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
        batch_x = batch_x.float().to(model.device)
        batch_y = batch_y.float().to(model.device)
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

    return {
        "mse": total_loss / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
    }


def _make_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    config: ForecastConfig,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if config.lradj == "step":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=max(steps_per_epoch, 1),
            pct_start=config.pct_start,
            epochs=config.train_epochs,
            max_lr=config.learning_rate,
        )
    if config.lradj == "exp":
        return torch.optim.lr_scheduler.ExponentialLR(
            optimizer=optimizer,
            gamma=config.lr_decay,
        )
    return None


def _adjust_learning_rate_after_epoch(
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    epoch: int,
    config: ForecastConfig,
) -> None:
    if config.lradj == "step":
        return
    if config.lradj == "decay":
        lr = config.learning_rate * (config.lr_decay ** (epoch - 1))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print(f"Updating learning rate to {lr}")
        return
    if config.lradj == "constant":
        for param_group in optimizer.param_groups:
            param_group["lr"] = config.learning_rate
        print(f"Updating learning rate to {config.learning_rate}")
        return
    if config.lradj == "exp" and scheduler is not None:
        scheduler.step()


def _save_checkpoint(
    model: Forecaster,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: ForecastConfig,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float],
) -> Path:
    path = checkpoint_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    pre_path = pretrain_checkpoint_path(config)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "config": asdict(config),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "pretrain_checkpoint": str(pre_path) if pre_path is not None else "",
            "pretrain_run": config.pretrain_run,
            "train_run": config.train_run,
        },
        path,
    )
    return path


def train_forecaster(
    config: ForecastConfig,
    max_train_steps: int | None = None,
    max_val_steps: int | None = None,
) -> Path:
    seed_everything(config.seed)
    train_loader, val_loader = _make_loaders(config)
    model = Forecaster(config)
    _load_pretrained_encoder(model, config)

    criterion = nn.MSELoss()
    if config.freeze_encoder:
        # Linear probe: freeze patch_embedding + encoder; only the head trains.
        for name, p in model.named_parameters():
            if name.startswith("patch_embedding.") or name.startswith("encoder."):
                p.requires_grad = False
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=config.learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = _make_lr_scheduler(optimizer, config, steps_per_epoch=len(train_loader))

    best_val = float("inf")
    stale_epochs = 0
    best_path = checkpoint_path(config)
    train_dir = best_path.parent
    _reset_train_logs(train_dir, config)
    for epoch in range(1, config.train_epochs + 1):
        start = time.time()
        train_metrics = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            step_scheduler_per_batch=config.lradj == "step",
            max_steps=max_train_steps,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(model, val_loader, criterion, max_steps=max_val_steps)
        elapsed = time.time() - start
        lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:03d}/{config.train_epochs:03d} | "
            f"train mse {train_metrics['mse']:.6f} mae {train_metrics['mae']:.6f} | "
            f"val mse {val_metrics['mse']:.6f} mae {val_metrics['mae']:.6f} | "
            f"lr {lr:.8f} | {elapsed:.1f}s"
        )

        checkpoint_saved = False
        if val_metrics["mse"] < best_val:
            print(
                f"Validation MSE decreased ({best_val:.6f} -> {val_metrics['mse']:.6f}). "
                "Saving checkpoint."
            )
            best_val = val_metrics["mse"]
            stale_epochs = 0
            checkpoint_saved = True
            best_path = _save_checkpoint(model, optimizer, epoch, config, train_metrics, val_metrics)
        else:
            stale_epochs += 1
        _append_train_epoch_log(
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
                "checkpoint_saved": checkpoint_saved,
            },
        )
        _adjust_learning_rate_after_epoch(optimizer, scheduler, epoch, config)
        if not checkpoint_saved and stale_epochs >= config.patience:
            print(f"Early stopping after {config.patience} stale validation epochs.")
            break
    pre_path = pretrain_checkpoint_path(config)
    (train_dir / "training_summary.json").write_text(
        json.dumps(
            {
                "best_checkpoint": str(best_path),
                "best_val_mse": best_val,
                "pretrain_checkpoint": str(pre_path) if pre_path is not None else "",
            },
            indent=2,
        )
        + "\n"
    )
    write_done(train_dir)  # last write: train stage complete
    return best_path


def load_trained_model(config: ForecastConfig, checkpoint: str | None = None) -> Forecaster:
    model = Forecaster(config)
    path = PROJECT_ROOT / checkpoint if checkpoint else checkpoint_path(config)
    if not path.exists():
        raise FileNotFoundError(f"Train checkpoint not found: {path}")
    payload = torch.load(path, map_location=model.device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    print(f"Loaded train checkpoint: {path}")
    return model


def _run_test_one_pred_len(
    config: ForecastConfig,
    checkpoint: str | None,
    save_predictions: bool,
    max_test_steps: int | None,
    out_dir: Path,
) -> dict[str, float]:
    dataset = ForecastWindowDataset(config, "test")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        drop_last=False,
    )
    model = load_trained_model(config, checkpoint=checkpoint)
    criterion = nn.MSELoss()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        metrics = _run_epoch(model, loader, criterion, max_steps=max_test_steps)
        if save_predictions:
            for step, (batch_x, batch_y) in enumerate(loader, start=1):
                batch_x = batch_x.float().to(model.device)
                pred = model(batch_x).detach().cpu().numpy()
                predictions.append(pred)
                targets.append(batch_y.numpy())
                if max_test_steps is not None and step >= max_test_steps:
                    break

    if save_predictions:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / f"pred_pl{config.pred_len}.npy", np.concatenate(predictions, axis=0))
        np.save(out_dir / f"true_pl{config.pred_len}.npy", np.concatenate(targets, axis=0))
    print(f"pl{config.pred_len}: MSE {metrics['mse']:.6f}, MAE {metrics['mae']:.6f}")
    return metrics


def test_forecaster(
    template_config: ForecastConfig,
    pred_lens: list[int],
    save_predictions: bool = False,
    max_test_steps: int | None = None,
) -> dict[int, dict[str, float]]:
    """Run test across multiple pred_lens, write a single results file."""
    setting = template_config.train_run
    out_dir = result_dir(template_config, result_setting=setting)

    metrics_per_pl: dict[int, dict[str, float]] = {}
    pretrain_config: dict | None = None
    train_config: dict = {}
    # When load_pretrain_weights=0, the training output is tagged as no_pretrain
    # regardless of which pretrain_run provided the arch — mirror that here.
    effective_pretrain_run = (
        template_config.pretrain_run if template_config.load_pretrain_weights else ""
    )
    for pred_len in pred_lens:
        checkpoint = _resolve_test_checkpoint(
            dataset=template_config.dataset_name,
            model_id=template_config.model_id,
            input_len=template_config.input_len,
            pred_len=pred_len,
            pretrain_run=effective_pretrain_run,
            train_run=template_config.train_run,
        )
        config = _load_checkpoint_config(checkpoint)
        config.pred_len = pred_len
        config.batch_size = template_config.batch_size
        config.num_workers = template_config.num_workers
        if template_config.data_fine_dir:
            config.data_fine_dir = template_config.data_fine_dir
        config.device = template_config.device
        train_config = asdict(config)
        if config.pretrain_run:
            try:
                pre_payload = torch.load(
                    pretrain_checkpoint_path(config), map_location="cpu", weights_only=False
                )
                pretrain_config = dict(pre_payload.get("config", {}))
            except FileNotFoundError:
                pretrain_config = None
        metrics_per_pl[pred_len] = _run_test_one_pred_len(
            config=config,
            checkpoint=str(checkpoint),
            save_predictions=save_predictions,
            max_test_steps=max_test_steps,
            out_dir=out_dir,
        )

    test_config = {
        "pred_lens": pred_lens,
        "batch_size": template_config.batch_size,
        "num_workers": template_config.num_workers,
        "save_predictions": save_predictions,
        "max_test_steps": max_test_steps,
        "seed": template_config.seed,
    }
    write_results_file(
        out_dir=out_dir,
        setting=setting,
        model_id=template_config.model_id,
        dataset_name=template_config.dataset_name,
        input_len=template_config.input_len,
        pretrain_run=template_config.pretrain_run or "no_pretrain",
        train_run=template_config.train_run,
        pred_len_metrics=metrics_per_pl,
        pretrain_config=pretrain_config,
        train_config=train_config,
        test_config=test_config,
    )
    print(f"Saved test results to {out_dir}/results.json")
    return metrics_per_pl


_INHERITED_FROM_PRETRAIN: list[str] = [
    "features",
    "input_len",
    "enc_in",
    "patch_len",
    "d_model",
    "n_heads",
    "d_ff",
    "use_norm",
]
# SDTA pretrain config uses `e_layers`; forecast uses `enc_layers`. Map at load time.
_INHERITED_KEY_RENAMES: dict[str, str] = {"enc_layers": "e_layers"}


def add_forecast_args(parser: argparse.ArgumentParser) -> None:
    # Identity / paths
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--model_id", required=True)
    parser.add_argument(
        "--pretrain_run",
        required=True,
        help="Pretrain run_name (folder). Arch params are inherited from "
             "outputs/pretrain/{model_id}/{dataset}/il*/{pretrain_run}/config.json.",
    )
    parser.add_argument(
        "--load_pretrain_weights",
        type=int,
        choices=[0, 1],
        default=1,
        help="0 = ablation: use pretrain arch but random-init weights "
             "(output dir tagged as pre_no_pretrain).",
    )
    parser.add_argument(
        "--run_name",
        required=True,
        help="User-named folder for this train config.",
    )
    # Forecast-specific
    parser.add_argument("--pred_len", type=int, required=True)
    parser.add_argument("--dec_layers", type=int, required=True,
                        help="MLP depth in the flatten forecast head.")
    parser.add_argument("--head_dropout", type=float, required=True)
    # Train-time adjustable (intentionally NOT inherited from pretrain)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--learning_rate", type=float, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--train_epochs", type=int, required=True)
    parser.add_argument("--patience", type=int, required=True)
    parser.add_argument("--pct_start", type=float, required=True)
    parser.add_argument("--lr_decay", type=float, default=0.5)
    parser.add_argument(
        "--lradj",
        choices=["step", "decay", "exp", "constant"],
        default="step",
    )
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument(
        "--freeze_encoder",
        type=int,
        choices=[0, 1],
        default=0,
        help="Linear-probe ablation. 1 = freeze patch_embedding + encoder; "
             "optimizer only updates the forecast head.",
    )


def config_from_args(args: argparse.Namespace) -> ForecastConfig:
    from .._config_inherit import inherit_fields, load_pretrain_config

    pretrain_cfg = load_pretrain_config(args.model_id, args.dataset, args.pretrain_run)
    # Pull inherited fields (renaming where pretrain naming differs from forecast).
    source_keys = [_INHERITED_KEY_RENAMES.get(k, k) for k in _INHERITED_FROM_PRETRAIN]
    src = inherit_fields(pretrain_cfg, source_keys)
    inherited = {dst: src[_INHERITED_KEY_RENAMES.get(dst, dst)] for dst in _INHERITED_FROM_PRETRAIN}
    # Forecast field name for encoder depth is `enc_layers`; pretrain stores `e_layers`.
    enc_layers = int(pretrain_cfg["e_layers"])

    return ForecastConfig(
        dataset_name=args.dataset,
        data_fine_dir=args.data_fine_dir,
        model_id=args.model_id,
        # Inherited from pretrain config:
        features=str(inherited["features"]),
        input_len=int(inherited["input_len"]),
        enc_in=int(inherited["enc_in"]),
        patch_len=int(inherited["patch_len"]),
        d_model=int(inherited["d_model"]),
        n_heads=int(inherited["n_heads"]),
        d_ff=int(inherited["d_ff"]),
        use_norm=bool(inherited["use_norm"]),
        enc_layers=enc_layers,
        # From CLI:
        pred_len=args.pred_len,
        dec_layers=args.dec_layers,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_epochs=args.train_epochs,
        patience=args.patience,
        pct_start=args.pct_start,
        lr_decay=args.lr_decay,
        lradj=args.lradj,
        num_workers=args.num_workers,
        pretrain_run=args.pretrain_run,
        train_run=args.run_name,
        seed=args.seed,
        load_pretrain_weights=bool(args.load_pretrain_weights),
        freeze_encoder=bool(args.freeze_encoder),
    )


def train_main() -> None:
    parser = argparse.ArgumentParser(description="Train SDTA forecaster")
    parser.add_argument("--max_train_steps", type=int, default=None)
    parser.add_argument("--max_val_steps", type=int, default=None)
    add_forecast_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    best_path = train_forecaster(
        config,
        max_train_steps=args.max_train_steps,
        max_val_steps=args.max_val_steps,
    )
    print(f"Best checkpoint: {best_path}")


def _load_checkpoint_config(path: Path) -> ForecastConfig:
    payload = torch.load(path, map_location="cpu")
    cfg_dict = dict(payload["config"])
    cfg_dict.setdefault("data_fine_dir", "data/fine")
    cfg_dict.setdefault("seed", 2021)
    config = ForecastConfig(**cfg_dict)
    config.device = "auto"
    return config


def _resolve_test_checkpoint(
    dataset: str,
    model_id: str,
    input_len: int | None,
    pred_len: int,
    train_run: str,
    pretrain_run: str,
) -> Path:
    pre_name = pretrain_run if pretrain_run else NO_PRETRAIN_TAG
    if input_len is None:
        root = TRAIN_CHECKPOINT_ROOT / model_id / dataset
        candidates = sorted(root.glob(f"il*/pre_{pre_name}/{train_run}/pl{pred_len}/checkpoint.pth"))
        if not candidates:
            raise FileNotFoundError(
                f"Train checkpoint not found under {root}/il*/pre_{pre_name}/{train_run}/pl{pred_len}"
            )
        if len(candidates) > 1:
            listing = "\n  ".join(str(path) for path in candidates)
            raise FileNotFoundError(
                f"Multiple train checkpoints match; pass --input_len to disambiguate:\n  {listing}"
            )
        return candidates[0]
    path = (
        TRAIN_CHECKPOINT_ROOT
        / model_id
        / dataset
        / f"il{input_len}"
        / f"pre_{pre_name}"
        / train_run
        / f"pl{pred_len}"
        / "checkpoint.pth"
    )
    if not path.exists():
        raise FileNotFoundError(f"Train checkpoint not found: {path}")
    return path


def _parse_pred_lens(arg: str) -> list[int]:
    items = [item.strip() for item in arg.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--pred_lens must list at least one pred_len")
    return [int(item) for item in items]


def test_main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained SDTA forecaster across pred_lens")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model_id", required=True)
    parser.add_argument("--input_len", type=int, default=None, help="Optional; inferred from checkpoint path when omitted.")
    parser.add_argument(
        "--pred_lens",
        type=_parse_pred_lens,
        required=True,
        help="Comma-separated pred_lens, e.g. 96,192,336,720.",
    )
    parser.add_argument("--pretrain_run", default="")
    parser.add_argument("--train_run", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, required=True)
    parser.add_argument("--data_fine_dir", default="data/fine")
    parser.add_argument("--save_predictions", type=int, choices=[0, 1], default=0)
    parser.add_argument("--max_test_steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2021)
    args = parser.parse_args()
    seed_everything(args.seed)

    first_pl = args.pred_lens[0]
    first_ckpt = _resolve_test_checkpoint(
        dataset=args.dataset,
        model_id=args.model_id,
        input_len=args.input_len,
        pred_len=first_pl,
        pretrain_run=args.pretrain_run,
        train_run=args.train_run,
    )
    template = _load_checkpoint_config(first_ckpt)
    template.batch_size = args.batch_size
    template.num_workers = args.num_workers
    template.data_fine_dir = args.data_fine_dir
    template.seed = args.seed

    test_forecaster(
        template,
        pred_lens=args.pred_lens,
        save_predictions=bool(args.save_predictions),
        max_test_steps=args.max_test_steps,
    )
