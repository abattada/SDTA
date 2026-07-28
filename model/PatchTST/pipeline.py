from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from .datautils import get_dls
from .datautils import parse_dsets
from .src.callback.patch_mask import PatchCB, PatchMaskCB
from .src.callback.tracking import SaveModelCB
from .src.callback.transforms import RevInCB
from .src.learner import Learner, transfer_weights
from .src.metrics import mae, mse
from .src.models.patchTST import PatchTST
from .._config_inherit import ConfigInheritError, inherit_fields, load_pretrain_config
from .._test_io import write_done


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_ROOT = PROJECT_ROOT / "outputs" / "pretrain"
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "train"
TEST_ROOT = PROJECT_ROOT / "outputs" / "test"
DEFAULT_MODEL_ID = "PatchTST"
PATCHTST_ARCH_FIELDS = [
    "features",
    "context_points",
    "patch_len",
    "stride",
    "n_layers",
    "n_heads",
    "d_model",
    "d_ff",
    "revin",
    "scaler",
    "use_time_features",
]


def _format_param(value: float | int | str) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _safe_name(value: str) -> str:
    return (
        str(value)
        .replace("/", "-")
        .replace("\\", "-")
        .replace(" ", "")
        .replace(":", "")
    )


def _dataset_for_path(dataset: str) -> str:
    aliases = {
        "etth1": "ETTh1",
        "etth2": "ETTh2",
        "ettm1": "ETTm1",
        "ettm2": "ETTm2",
        "ett_all": "ETT_all",
        "ettall": "ETT_all",
        "ett-all": "ETT_all",
        "all_ett": "ETT_all",
    }
    return aliases.get(dataset.lower(), dataset)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _num_patch(context_points: int, patch_len: int, stride: int) -> int:
    return (max(context_points, patch_len) - patch_len) // stride + 1


def build_model(c_in: int, args: argparse.Namespace, head_type: str) -> PatchTST:
    num_patch = _num_patch(args.context_points, args.patch_len, args.stride)
    print("number of patches:", num_patch)
    model = PatchTST(
        c_in=c_in,
        target_dim=args.target_points,
        patch_len=args.patch_len,
        stride=args.stride,
        num_patch=num_patch,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        shared_embedding=True,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_dropout=args.head_dropout,
        act="relu",
        head_type=head_type,
        res_attention=False,
    )
    print("number of model params", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model


def _data_source_suffix(args: argparse.Namespace) -> str:
    data_root = getattr(args, "data_root", "data/fine")
    if data_root == "data/fine":
        return ""
    return f"_src{_safe_name(data_root)}"


def _mixed_suffix(args: argparse.Namespace) -> str:
    dataset = _dataset_for_path(getattr(args, "dataset", ""))
    datasets = getattr(args, "datasets", None)
    if dataset != "ETT_all" and not datasets:
        return ""
    dsets = "-".join(_dataset_for_path(dset) for dset in parse_dsets(datasets))
    sampling = getattr(args, "sampling", "balanced")
    return f"_mix{_safe_name(dsets)}_samp{sampling}"


def _pretrain_dataset(args: argparse.Namespace) -> str:
    return getattr(args, "pretrain_dataset", None) or args.dataset


def _train_dataset(args: argparse.Namespace) -> str:
    return getattr(args, "train_dataset", None) or args.dataset


def pretrain_setting_name(args: argparse.Namespace) -> str:
    return args.run_name


def train_setting_name(args: argparse.Namespace) -> str:
    return args.run_name


def pretrain_run_dir(args: argparse.Namespace) -> Path:
    return (
        PRETRAIN_ROOT
        / args.model_id
        / _dataset_for_path(args.dataset)
        / f"il{args.context_points}"
        / args.run_name
    )


def pretrain_checkpoint_path(args: argparse.Namespace) -> Path:
    return pretrain_run_dir(args) / "ckpt_best.pth"


def _effective_pretrain_tag(args: argparse.Namespace) -> str:
    """Returns 'no_pretrain' when --load_pretrain_weights=0 or pretrain_run is empty."""
    load = bool(getattr(args, "load_pretrain_weights", 1))
    pre_run = getattr(args, "pretrain_run", "") or ""
    if not pre_run or not load:
        return "no_pretrain"
    return pre_run


def train_run_dir(args: argparse.Namespace) -> Path:
    return (
        TRAIN_ROOT
        / args.model_id
        / _dataset_for_path(_train_dataset(args))
        / f"il{args.context_points}"
        / f"pre_{_effective_pretrain_tag(args)}"
        / args.run_name
        / f"pl{args.target_points}"
    )


def train_checkpoint_path(args: argparse.Namespace) -> Path:
    return train_run_dir(args) / "checkpoint.pth"


def test_result_dir(args: argparse.Namespace) -> Path:
    return (
        TEST_ROOT
        / args.model_id
        / _dataset_for_path(args.dataset)
        / f"il{args.context_points}"
        / f"pre_{_effective_pretrain_tag(args)}"
        / args.train_run
    )


def resolve_pretrained_checkpoint(args: argparse.Namespace) -> str | None:
    if not bool(getattr(args, "load_pretrain_weights", 1)):
        return None
    if args.pretrained_checkpoint:
        return args.pretrained_checkpoint
    if not args.pretrain_run:
        return None
    path = (
        PRETRAIN_ROOT
        / args.model_id
        / _dataset_for_path(_pretrain_dataset(args))
        / f"il{args.context_points}"
        / args.pretrain_run
        / "ckpt_best.pth"
    )
    if not path.exists():
        raise FileNotFoundError(f"Pretrain checkpoint not found: {path}")
    return str(path)


def inherit_patchtst_train_config(args: argparse.Namespace) -> None:
    """Fill PatchTST train architecture fields from the referenced pretrain config."""
    if args.mode not in {"finetune", "linear_probe"}:
        raise ConfigInheritError(
            "PatchTST train now inherits architecture from pretrain config. "
            "Use --mode finetune or --mode linear_probe with --pretrain_run."
        )
    if args.pretrained_checkpoint and not args.pretrain_run:
        cfg_path = Path(args.pretrained_checkpoint).resolve().parent / "config.json"
        try:
            cfg = json.loads(cfg_path.read_text())
        except OSError as exc:
            raise ConfigInheritError(
                f"Cannot inherit architecture: missing config next to checkpoint: {cfg_path}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise ConfigInheritError(f"Failed to parse {cfg_path}: {exc}") from exc
    else:
        if not args.pretrain_run:
            raise ConfigInheritError(
                "--mode finetune/linear_probe requires --pretrain_run or --pretrained_checkpoint"
            )
        cfg = load_pretrain_config(
            args.model_id,
            _dataset_for_path(_pretrain_dataset(args)),
            args.pretrain_run,
            dir_glob="il*",
        )
    inherited = inherit_fields(cfg, PATCHTST_ARCH_FIELDS)
    for key, value in inherited.items():
        setattr(args, key, value)


def _namespace_for_dls(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        dset=args.dataset,
        datasets=getattr(args, "datasets", None),
        context_points=args.context_points,
        target_points=args.target_points,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        scaler=args.scaler,
        features=args.features,
        use_time_features=bool(args.use_time_features),
        data_root=args.data_root,
        sampling=getattr(args, "sampling", "balanced"),
    )


def _write_config(run_dir: Path, args: argparse.Namespace) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = vars(args).copy()
    payload["run_dir"] = str(run_dir)
    (run_dir / "config.json").write_text(json.dumps(payload, indent=2) + "\n")


def _save_losses(run_dir: Path, learner: Learner) -> None:
    train_loss = learner.recorder.get("train_loss", [])
    valid_loss = learner.recorder.get("valid_loss", [])
    pd.DataFrame({"train_loss": train_loss, "valid_loss": valid_loss}).to_csv(
        run_dir / "losses.csv",
        float_format="%.6f",
        index=False,
    )
    write_done(run_dir)  # last write: pretrain/train stage complete


def maybe_find_lr(
    dls,
    model: PatchTST,
    loss_func: torch.nn.Module,
    cbs: list,
    lr: float,
    use_lr_finder: bool,
) -> float:
    if not use_lr_finder:
        return lr
    learn = Learner(dls, model, loss_func, lr=lr, cbs=cbs)
    suggested_lr = learn.lr_finder(show_plot=False)
    print("suggested_lr", suggested_lr)
    return suggested_lr


def run_pretrain(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    dls = get_dls(_namespace_for_dls(args))
    loss_func = torch.nn.MSELoss(reduction="mean")
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio)]
    if args.use_lr_finder:
        args.lr = maybe_find_lr(
            dls,
            build_model(dls.vars, args, head_type="pretrain"),
            loss_func,
            cbs,
            args.lr,
            True,
        )

    run_dir = pretrain_run_dir(args)
    _write_config(run_dir, args)
    print("Checkpoint dir:", run_dir)
    model = build_model(dls.vars, args, head_type="pretrain")
    cbs = [RevInCB(dls.vars, denorm=False)] if args.revin else []
    cbs += [
        PatchMaskCB(patch_len=args.patch_len, stride=args.stride, mask_ratio=args.mask_ratio),
        SaveModelCB(monitor="valid_loss", fname="ckpt_best", path=run_dir),
    ]
    if args.checkpoint_every > 0:
        print("checkpoint_every is ignored for PatchTST because the original callback keeps one active saver.")
    learner = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs)
    learner.fit_one_cycle(n_epochs=args.n_epochs_pretrain, lr_max=args.lr)
    _save_losses(run_dir, learner)
    print(f"Best checkpoint: {pretrain_checkpoint_path(args)}")
    return pretrain_checkpoint_path(args)


def run_train(args: argparse.Namespace) -> Path:
    set_seed(args.seed)
    pretrained_checkpoint = resolve_pretrained_checkpoint(args)
    if (
        args.mode in {"finetune", "linear_probe"}
        and pretrained_checkpoint is None
        and bool(getattr(args, "load_pretrain_weights", 1))
    ):
        raise ValueError("--mode finetune/linear_probe requires --pretrain_run or --pretrained_checkpoint")
    dls = get_dls(_namespace_for_dls(args))
    loss_func = torch.nn.MSELoss(reduction="mean")

    if args.use_lr_finder:
        finder_model = build_model(dls.vars, args, head_type="prediction")
        if pretrained_checkpoint is not None:
            finder_model = transfer_weights(pretrained_checkpoint, finder_model)
        finder_cbs = [RevInCB(dls.vars)] if args.revin else []
        finder_cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        args.lr = maybe_find_lr(dls, finder_model, loss_func, finder_cbs, args.lr, True)

    run_dir = train_run_dir(args)
    _write_config(run_dir, args)
    print("Checkpoint dir:", run_dir)
    model = build_model(dls.vars, args, head_type="prediction")
    if pretrained_checkpoint is not None:
        model = transfer_weights(pretrained_checkpoint, model)

    cbs = [RevInCB(dls.vars)] if args.revin else []
    cbs += [
        PatchCB(patch_len=args.patch_len, stride=args.stride),
        SaveModelCB(monitor="valid_loss", fname="checkpoint", path=run_dir),
    ]
    learner = Learner(dls, model, loss_func, lr=args.lr, cbs=cbs, metrics=[mse])
    if args.mode == "finetune" and pretrained_checkpoint is not None:
        learner.fine_tune(n_epochs=args.n_epochs, base_lr=args.lr, freeze_epochs=args.freeze_epochs)
    elif args.mode == "linear_probe":
        learner.linear_probe(n_epochs=args.n_epochs, base_lr=args.lr)
    else:
        learner.fit_one_cycle(n_epochs=args.n_epochs, lr_max=args.lr, pct_start=0.2)
    _save_losses(run_dir, learner)
    print(f"Best checkpoint: {train_checkpoint_path(args)}")
    return train_checkpoint_path(args)


def _run_test_one_target(
    args: argparse.Namespace,
    target_points: int,
    out_dir: Path,
) -> dict[str, float]:
    """Run a single target_points (pred_len). Mutates args.target_points temporarily."""
    saved_tp = args.target_points
    args.target_points = target_points
    try:
        set_seed(args.seed)
        args.run_name = args.train_run
        checkpoint = _resolve_train_checkpoint_for_test(args, target_points)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Train checkpoint not found: {checkpoint}")
        dls = get_dls(_namespace_for_dls(args))
        model = build_model(dls.vars, args, head_type="prediction")
        cbs = [RevInCB(dls.vars, denorm=True)] if args.revin else []
        cbs += [PatchCB(patch_len=args.patch_len, stride=args.stride)]
        learner = Learner(dls, model, cbs=cbs)
        pred, target, scores = learner.test(
            dls.test, weight_path=str(checkpoint), scores=[mse, mae]
        )
        metrics = {"mse": float(scores[0]), "mae": float(scores[1])}
        if args.save_predictions:
            out_dir.mkdir(parents=True, exist_ok=True)
            np.save(out_dir / f"pred_pl{target_points}.npy", pred)
            np.save(out_dir / f"true_pl{target_points}.npy", target)
        print(f"pl{target_points}: MSE {metrics['mse']:.6f}, MAE {metrics['mae']:.6f}")
        return metrics
    finally:
        args.target_points = saved_tp


def _parse_target_lens(arg: str) -> list[int]:
    items = [item.strip() for item in arg.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("--target_points must list at least one pred_len")
    return [int(item) for item in items]


def _load_train_config(checkpoint: Path) -> dict | None:
    cfg_path = checkpoint.parent / "config.json"
    try:
        return json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _resolve_train_checkpoint_for_test(args: argparse.Namespace, target_points: int) -> Path:
    if args.checkpoint:
        checkpoint = Path(args.checkpoint)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Train checkpoint not found: {checkpoint}")
        return checkpoint

    root = TRAIN_ROOT / args.model_id / _dataset_for_path(_train_dataset(args))
    candidates = sorted(
        root.glob(f"*/pre_{_effective_pretrain_tag(args)}/{args.train_run}/pl{target_points}/checkpoint.pth")
    )
    if not candidates:
        raise FileNotFoundError(
            f"Train checkpoint not found under {root}/*/pre_{_effective_pretrain_tag(args)}/{args.train_run}/pl{target_points}"
        )
    if len(candidates) > 1:
        listing = "\n  ".join(str(path) for path in candidates)
        raise FileNotFoundError(
            f"Multiple train checkpoints match; pass --checkpoint to disambiguate:\n  {listing}"
        )
    return candidates[0]


def inherit_patchtst_test_config(args: argparse.Namespace) -> None:
    checkpoint = _resolve_train_checkpoint_for_test(args, args.target_lens[0])
    train_cfg = _load_train_config(checkpoint)
    if not train_cfg:
        raise ConfigInheritError(f"Cannot inherit test architecture: missing config next to {checkpoint}")

    for key in PATCHTST_ARCH_FIELDS + ["dropout", "head_dropout", "mode", "freeze_epochs", "sampling"]:
        if key in train_cfg:
            setattr(args, key, train_cfg[key])
    args.target_points = args.target_lens[0]
    if not args.pretrain_run:
        args.pretrain_run = train_cfg.get("pretrain_run", "") or ""
    if not getattr(args, "pretrain_dataset", None):
        args.pretrain_dataset = train_cfg.get("pretrain_dataset", None)
    if "load_pretrain_weights" in train_cfg:
        args.load_pretrain_weights = bool(train_cfg["load_pretrain_weights"])



def _load_pretrain_config(args: argparse.Namespace) -> dict | None:
    if not bool(getattr(args, "load_pretrain_weights", 1)):
        return None
    if not args.pretrain_run:
        return None
    pre_dir = (
        PRETRAIN_ROOT
        / args.model_id
        / _dataset_for_path(_pretrain_dataset(args))
        / f"il{args.context_points}"
        / args.pretrain_run
    )
    try:
        return json.loads((pre_dir / "config.json").read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_test(args: argparse.Namespace) -> dict[int, dict[str, float]]:
    """Run test across all target_points listed in args.target_lens; write one results file."""
    from .._test_io import write_results_file

    out_dir = test_result_dir(args)
    metrics_per_pl: dict[int, dict[str, float]] = {}
    train_config: dict | None = None
    pretrain_config: dict | None = None
    for tp in args.target_lens:
        # Capture train_config + pretrain_config once on first iteration.
        args.target_points = tp
        args.run_name = args.train_run
        ckpt_path = _resolve_train_checkpoint_for_test(args, tp)
        if train_config is None:
            train_config = _load_train_config(ckpt_path) or {}
            pretrain_config = _load_pretrain_config(args)
        metrics_per_pl[tp] = _run_test_one_target(args, tp, out_dir)

    test_config = {
        "target_lens": args.target_lens,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "save_predictions": bool(args.save_predictions),
    }
    write_results_file(
        out_dir=out_dir,
        setting=args.train_run,
        model_id=args.model_id,
        dataset_name=_dataset_for_path(args.dataset),
        input_len=args.context_points,
        pretrain_run=args.pretrain_run or "no_pretrain",
        train_run=args.train_run,
        pred_len_metrics=metrics_per_pl,
        pretrain_config=pretrain_config,
        train_config=train_config or {},
        test_config=test_config,
    )
    print(f"Saved test results to {out_dir}/results.json")
    return metrics_per_pl


def run_test_all(args: argparse.Namespace) -> dict[str, dict[int, dict[str, float]]]:
    results = {}
    for dataset in parse_dsets(args.datasets):
        dataset_args = argparse.Namespace(**vars(args))
        dataset_args.dataset = _dataset_for_path(dataset)
        results[_dataset_for_path(dataset)] = run_test(dataset_args)
    return results


def add_common_args(parser: argparse.ArgumentParser, *, include_arch: bool = True) -> None:
    parser.add_argument("--dataset", "--dset", dest="dataset", default=None)
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated datasets for opt-in mixed runs, e.g. ETTh1,ETTh2,ETTm1,ETTm2.",
    )
    parser.add_argument("--data_root", default="data/fine")
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    if include_arch:
        parser.add_argument("--features", required=True)
        parser.add_argument("--context_points", "--input_len", dest="context_points", type=int, required=True)
        parser.add_argument("--scaler", required=True)
        parser.add_argument("--use_time_features", type=int, choices=[0, 1], required=True)
        parser.add_argument("--patch_len", type=int, required=True)
        parser.add_argument("--stride", type=int, required=True)
        parser.add_argument("--revin", type=int, choices=[0, 1], required=True)
        parser.add_argument("--n_layers", type=int, required=True)
        parser.add_argument("--n_heads", type=int, required=True)
        parser.add_argument("--d_model", type=int, required=True)
        parser.add_argument("--d_ff", type=int, required=True)
    parser.add_argument("--target_points", "--pred_len", dest="target_points", type=int, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--head_dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=2021)
    parser.add_argument("--use_lr_finder", type=int, choices=[0, 1], default=0)
    parser.add_argument("--sampling", choices=["balanced", "proportional"], default="balanced")
    parser.add_argument("--cuda_visible_devices", "--cuda-visible-devices", default=None)


def parse_pretrain_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain PatchTST self-supervised")
    add_common_args(parser)
    parser.add_argument("--run_name", required=True, help="User-named folder for this pretrain config.")
    parser.add_argument("--dset_pretrain", dest="dataset", default=None)
    parser.add_argument("--n_epochs_pretrain", "--train_epochs", dest="n_epochs_pretrain", type=int, required=True)
    parser.add_argument("--mask_ratio", type=float, default=0.4)
    parser.add_argument("--checkpoint_every", type=int, default=0)
    args = parser.parse_args(argv)
    if args.dataset is None:
        parser.error("--dataset is required")
    if args.target_points is None:
        args.target_points = args.context_points
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    return args


def parse_train_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train or finetune PatchTST")
    add_common_args(parser, include_arch=False)
    parser.set_defaults(head_dropout=0.2)
    parser.add_argument("--run_name", required=True, help="User-named folder for this train config.")
    parser.add_argument("--dset_finetune", dest="dataset", default=None)
    parser.add_argument("--mode", choices=["supervised", "finetune", "linear_probe"], default="finetune")
    parser.add_argument("--n_epochs", "--train_epochs", dest="n_epochs", type=int, required=True)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument(
        "--pretrain_run",
        default="",
        help="Pretrain run_name (folder); empty string = train from scratch.",
    )
    parser.add_argument("--pretrain_dataset", default=None)
    parser.add_argument("--pretrained_checkpoint", default=None)
    parser.add_argument("--load_pretrain_weights", type=int, choices=[0, 1], default=1, help="0 = inherit pretrain arch but train random-init weights under pre_no_pretrain.")
    args = parser.parse_args(argv)
    if args.dataset is None:
        parser.error("--dataset is required")
    if args.target_points is None:
        parser.error("--target_points is required for train")
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    try:
        inherit_patchtst_train_config(args)
    except ConfigInheritError as exc:
        parser.error(str(exc))
    return args


def parse_test_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test a trained PatchTST checkpoint across pred_lens")
    add_common_args(parser, include_arch=False)
    parser.add_argument(
        "--target_lens",
        "--pred_lens",
        dest="target_lens",
        type=_parse_target_lens,
        required=True,
        help="Comma-separated pred_lens, e.g. 96,192,336,720.",
    )
    parser.add_argument("--dset_finetune", dest="dataset", default=None)
    parser.add_argument("--mode", choices=["supervised", "finetune", "linear_probe"], default="finetune")
    parser.add_argument("--n_epochs", "--train_epochs", dest="n_epochs", type=int, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--pretrain_run", default="")
    parser.add_argument("--pretrain_dataset", default=None)
    parser.add_argument("--train_dataset", default=None)
    parser.add_argument("--train_run", required=True, help="train run_name folder.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_predictions", type=int, choices=[0, 1], default=0)
    args = parser.parse_args(argv)
    if args.dataset is None:
        parser.error("--dataset is required")
    if args.target_points is None:
        args.target_points = args.target_lens[0]
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.mode == "supervised":
        args.pretrain_run = ""
    args.save_predictions = bool(args.save_predictions)
    try:
        inherit_patchtst_test_config(args)
    except ConfigInheritError as exc:
        parser.error(str(exc))
    return args


def parse_test_all_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test one PatchTST checkpoint across multiple datasets")
    add_common_args(parser, include_arch=False)
    parser.set_defaults(dataset="ETT_all")
    parser.add_argument(
        "--target_lens",
        "--pred_lens",
        dest="target_lens",
        type=_parse_target_lens,
        required=True,
        help="Comma-separated pred_lens, e.g. 96,192,336,720.",
    )
    parser.add_argument("--dset_finetune", dest="dataset", default=None)
    parser.add_argument("--mode", choices=["supervised", "finetune", "linear_probe"], default="finetune")
    parser.add_argument("--n_epochs", "--train_epochs", dest="n_epochs", type=int, default=None)
    parser.add_argument("--freeze_epochs", type=int, default=10)
    parser.add_argument("--pretrain_run", default="")
    parser.add_argument("--pretrain_dataset", default=None)
    parser.add_argument("--train_dataset", default="ETT_all")
    parser.add_argument("--train_run", required=True, help="train run_name folder.")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--save_predictions", type=int, choices=[0, 1], default=0)
    args = parser.parse_args(argv)
    if args.dataset is None:
        args.dataset = "ETT_all"
    if args.datasets is None:
        args.datasets = "ETTh1,ETTh2,ETTm1,ETTm2"
    if args.target_points is None:
        args.target_points = args.target_lens[0]
    if args.cuda_visible_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    if args.mode == "supervised":
        args.pretrain_run = ""
    args.save_predictions = bool(args.save_predictions)
    try:
        inherit_patchtst_test_config(args)
    except ConfigInheritError as exc:
        parser.error(str(exc))
    return args
