"""Config-driven experiment batch runner.

Example:
    python entry/batch.py entry/batch_configs/SDTA/v5_lr_1e-4_default.json
    python entry/batch.py SDTA/v5_lr_1e-4_default --dry-run

Scheduling model:
    * ``cuda_visible_devices`` lists which physical GPUs are usable.
    * Every ready step queries live GPU free RAM (pynvml preferred, nvidia-smi
      fallback) and picks the card with the most free; admission is gated by
      ``min_free_mb`` only.
    * ``concurrent`` optionally caps total simultaneous step processes when
      CPU / driver is the bottleneck. No per-GPU cap.
    * Failed steps that look like CUDA / cuDNN / cuBLAS allocation errors get
      requeued and re-tried after another job finishes (event-driven).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Make ``batch_src`` importable when invoked as a script.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from batch_src.scheduler import (  # noqa: E402
    RunningJob,
    Scheduler,
    SchedulerConfig,
    query_gpu_free_mb,
)
from batch_src.resume import (  # noqa: E402
    StepCompletionCheck,
    is_step_done,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = _THIS_DIR / "batch_configs"
BATCH_SRC_ROOT = _THIS_DIR / "batch_src"
BATCH_RUNS_PATH = PROJECT_ROOT / "docs" / "batch" / "runs.md"
SPECIAL_KEYS = {
    "cuda_visible_devices",
    "env",
    "module",
    "pred_lens",
    "pretrain_run",
    "run_name",
    "skip",
    "train_run",
}


@dataclass
class Step:
    index: int
    total: int
    dataset: str
    stage: str
    run_name: str
    command: list[str]
    log_path: Path
    combo: dict[str, Any]
    label: str
    env: dict[str, str]
    completion_check: StepCompletionCheck | None = None
    skip_reason: str | None = None


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() == ".json":
        return json.loads(text)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            f"{path} looks like YAML, but PyYAML is not installed. "
            "Use a .json config or install PyYAML."
        ) from exc
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise SystemExit(f"Config must be an object: {path}")
    return data


def _resolve_config_path(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    for suffix in (".json", ".yaml", ".yml"):
        candidate = CONFIG_ROOT / f"{raw}{suffix}"
        if candidate.exists():
            return candidate
    raise SystemExit(f"Config not found: {raw}")


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _expand_grid(grid: Any) -> list[dict[str, Any]]:
    if not grid:
        return [{}]
    if isinstance(grid, list):
        combos = []
        for item in grid:
            if not isinstance(item, dict):
                raise SystemExit("List-style grid entries must be objects.")
            combos.append(dict(item))
        return combos or [{}]
    if not isinstance(grid, dict):
        raise SystemExit("grid must be an object or a list of objects.")
    keys = list(grid)
    values = [_as_list(grid[key]) for key in keys]
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _split_combo(combo: dict[str, Any], stage: str) -> tuple[dict[str, Any], dict[str, Any]]:
    common: dict[str, Any] = {}
    stage_specific: dict[str, Any] = {}
    prefix = f"{stage}."
    for key, value in combo.items():
        if "." not in key:
            common[key] = value
        elif key.startswith(prefix):
            stage_specific[key[len(prefix):]] = value
    return common, stage_specific


def _dataset_overrides(config: dict[str, Any], dataset: str, stage: str) -> dict[str, Any]:
    settings = config.get("dataset_settings", {}).get(dataset, {})
    if not isinstance(settings, dict):
        raise SystemExit(f"dataset_settings.{dataset} must be an object.")
    result = {k: v for k, v in settings.items() if k not in {"pretrain", "train", "test"}}
    stage_block = settings.get(stage, {})
    if stage_block:
        if not isinstance(stage_block, dict):
            raise SystemExit(f"dataset_settings.{dataset}.{stage} must be an object.")
        result.update(stage_block)
    return result


def _dataset_enc_in(dataset: str) -> int | None:
    meta = PROJECT_ROOT / "data" / "fine" / dataset / "train" / "metadata.json"
    if meta.exists():
        return json.loads(meta.read_text()).get("num_features")
    return None


def _stage_params(config: dict[str, Any], dataset: str, stage: str, combo: dict[str, Any]) -> dict[str, Any]:
    defaults = config.get("stage_defaults", {}).get(stage, {})
    if defaults and not isinstance(defaults, dict):
        raise SystemExit(f"stage_defaults.{stage} must be an object.")
    common, stage_specific = _split_combo(combo, stage)
    params = dict(defaults)
    params.update(_dataset_overrides(config, dataset, stage))
    params.update(common)
    params.update(stage_specific)
    if params.get("stride") is None and params.get("patch_len") is not None:
        params["stride"] = params["patch_len"]
    if stage == "pretrain" and params.get("enc_in") is None and config.get("auto_enc_in", False):
        enc_in = _dataset_enc_in(dataset)
        if enc_in is not None:
            params["enc_in"] = enc_in
    return params


def _stage_variants(config: dict[str, Any], stage: str) -> list[dict[str, Any]]:
    if stage not in {"train", "test"}:
        return [{}]
    variants = config.get("variants")
    if not variants:
        return [{}]
    result: list[dict[str, Any]] = []
    for variant in _as_list(variants):
        if not isinstance(variant, dict):
            raise SystemExit("variants entries must be objects.")
        result.append(dict(variant))
    return result or [{}]


def _variant_stage_overrides(variant: dict[str, Any], stage: str) -> dict[str, Any]:
    overrides = {
        key: value
        for key, value in variant.items()
        if key not in {"name", "suffix", "train", "test"}
    }
    stage_block = variant.get(stage, {})
    if stage_block:
        if not isinstance(stage_block, dict):
            raise SystemExit(f"variant {variant.get('name', '')}.{stage} must be an object.")
        overrides.update(stage_block)
    return overrides


def _variant_suffix(variant: dict[str, Any]) -> str:
    suffix = str(variant.get("suffix", "")).strip("_")
    return _sanitize_piece(suffix) if suffix else ""


def _append_suffix(name: str, suffix: str) -> str:
    if not suffix:
        return name
    return f"{name}_{suffix}"


def _apply_variant_params(
    params: dict[str, Any],
    stage: str,
    variant: dict[str, Any],
    run_name: str,
) -> dict[str, Any]:
    if not variant:
        return params
    params = dict(params)
    params.update(_variant_stage_overrides(variant, stage))
    suffix = _variant_suffix(variant)
    if suffix and stage == "train":
        train_run = str(params.get("train_run", params.get("run_name", run_name)))
        params["run_name"] = _append_suffix(train_run, suffix)
    elif suffix and stage == "test":
        train_run = str(params.get("train_run", params.get("run_name", run_name)))
        params["train_run"] = _append_suffix(train_run, suffix)
    return params


def _step_label_run_name(stage: str, run_name: str, params: dict[str, Any]) -> str:
    if stage == "pretrain":
        return str(params.get("run_name", run_name))
    if stage == "train":
        return str(params.get("train_run", params.get("run_name", run_name)))
    if stage == "test":
        return str(params.get("train_run", params.get("run_name", run_name)))
    return run_name


def _sanitize_piece(value: Any) -> str:
    text = str(value).strip()
    text = text.replace("-", "m").replace("+", "")
    text = text.replace(".", "p").replace("/", "_").replace(",", "-")
    clean = "".join(ch if ch.isalnum() else "_" for ch in text)
    return "_".join(part for part in clean.split("_") if part)


def _lookup_param(config: dict[str, Any], combo: dict[str, Any], key: str) -> Any:
    if key in combo:
        return combo[key]
    suffix = f".{key}"
    for combo_key, value in combo.items():
        if combo_key.endswith(suffix):
            return value
    for stage in ("pretrain", "train", "test"):
        stage_defaults = config.get("stage_defaults", {}).get(stage, {})
        if isinstance(stage_defaults, dict) and key in stage_defaults:
            return stage_defaults[key]
    return None


def _run_name(config: dict[str, Any], combo: dict[str, Any]) -> str:
    run_cfg = config.get("run_name", {})
    if isinstance(run_cfg, str):
        return _sanitize_piece(run_cfg)
    if not isinstance(run_cfg, dict):
        raise SystemExit("run_name must be a string or an object.")

    base = str(run_cfg.get("base", "")).strip("_")
    aliases = run_cfg.get("aliases", {})
    if not isinstance(aliases, dict):
        raise SystemExit("run_name.aliases must be an object.")

    include = run_cfg.get("include")
    if include is None:
        keys: list[str] = []
        if run_cfg.get("include_grid", True):
            keys.extend(key.split(".", 1)[-1] for key in combo)
        keys.extend(str(key) for key in run_cfg.get("include_fixed", []))
    else:
        keys = [str(key) for key in _as_list(include)]

    parts = [_sanitize_piece(base)] if base else []
    seen: set[str] = set()
    for key in keys:
        if key in seen:
            continue
        seen.add(key)
        value = _lookup_param(config, combo, key)
        if value is None:
            continue
        label = aliases.get(key, key)
        parts.append(_sanitize_piece(label))
        parts.append(_sanitize_piece(value))
    return "_".join(part for part in parts if part) or "run"


def _format_arg_value(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (list, tuple)):
        return ",".join(_format_arg_value(item) for item in value)
    return str(value)


def _append_params(command: list[str], params: dict[str, Any], skip: set[str] | None = None) -> None:
    skip = skip or set()
    for key, value in params.items():
        if key in SPECIAL_KEYS or key in skip or value is None:
            continue
        command.extend([f"--{key}", _format_arg_value(value)])


def _base_env(config: dict[str, Any], params: dict[str, Any]) -> dict[str, str]:
    """Build env without CUDA_VISIBLE_DEVICES — the scheduler sets it per launch."""
    env = os.environ.copy()
    env.pop("CUDA_VISIBLE_DEVICES", None)

    memory_fraction = config.get("gpu_memory_fraction")
    if memory_fraction is not None:
        env["SDTA_CUDA_MEMORY_FRACTION"] = str(memory_fraction)
        previous = env.get("PYTHONPATH", "")
        pieces = [str(BATCH_SRC_ROOT)]
        if previous:
            pieces.append(previous)
        env["PYTHONPATH"] = os.pathsep.join(pieces)

    alloc_conf = config.get("pytorch_cuda_alloc_conf")
    if alloc_conf:
        env["PYTORCH_CUDA_ALLOC_CONF"] = str(alloc_conf)

    extra_env = config.get("env", {})
    if extra_env:
        if not isinstance(extra_env, dict):
            raise SystemExit("env must be an object.")
        env.update({str(k): str(v) for k, v in extra_env.items()})
    param_env = params.get("env", {})
    if param_env:
        if not isinstance(param_env, dict):
            raise SystemExit("stage env must be an object.")
        env.update({str(k): str(v) for k, v in param_env.items()})
    return env


def _module_name(config: dict[str, Any], stage: str, params: dict[str, Any]) -> str:
    if params.get("module"):
        return str(params["module"])
    prefix = config.get("module_prefix")
    model_id = config.get("model_id")
    if prefix:
        return f"{prefix}.{stage}"
    return f"model.{model_id}.{stage}"


def _make_command(
    config: dict[str, Any],
    dataset: str,
    stage: str,
    run_name: str,
    params: dict[str, Any],
    pred_len: int | None = None,
) -> list[str]:
    model_id = str(config.get("model_id", params.get("model_id", "")))
    if not model_id:
        raise SystemExit("model_id is required.")
    command = [sys.executable, "-u", "-m", _module_name(config, stage, params)]
    command.extend(["--dataset", dataset, "--model_id", model_id])

    if stage == "pretrain":
        command.extend(["--run_name", str(params.get("run_name", run_name))])
    elif stage == "train":
        pretrain_run = str(params.get("pretrain_run", run_name))
        train_run = str(params.get("train_run", params.get("run_name", run_name)))
        command.extend(["--pretrain_run", pretrain_run, "--run_name", train_run])
        if pred_len is None:
            raise SystemExit("Internal error: train step missing pred_len.")
        command.extend(["--pred_len", str(pred_len)])
    elif stage == "test":
        load_pretrain = config.get("stage_defaults", {}).get("train", {}).get("load_pretrain_weights", 1)
        load_pretrain = params.get("load_pretrain_weights", load_pretrain)
        default_pretrain = "" if str(load_pretrain).lower() in {"0", "false", "no", "off"} else run_name
        pretrain_run = str(params.get("pretrain_run", default_pretrain))
        train_run = str(params.get("train_run", params.get("run_name", run_name)))
        command.extend(["--pretrain_run", pretrain_run, "--train_run", train_run])
        pred_lens = params.get("pred_lens")
        if pred_lens is None:
            pred_lens = config.get("stage_defaults", {}).get("train", {}).get("pred_lens")
        command.extend(["--pred_lens", _format_arg_value(pred_lens)])
    else:
        raise SystemExit(f"Unsupported stage: {stage}")

    test_only_skip = {"load_pretrain_weights"} if stage == "test" else set()
    _append_params(command, params, skip=test_only_skip)
    return command


def _grid_keys(grid: Any) -> set[str]:
    """Collect bare grid keys (strip 'pretrain.' / 'train.' / 'test.' prefix; drop 'dataset')."""
    if not grid:
        return set()
    raw: list[str] = []
    if isinstance(grid, dict):
        raw.extend(grid.keys())
    elif isinstance(grid, list):
        for item in grid:
            if isinstance(item, dict):
                raw.extend(item.keys())
    bare: set[str] = set()
    for key in raw:
        if key == "dataset":
            continue
        bare.add(key.split(".", 1)[-1])
    return bare


def _check_grid_include_invariant(config: dict[str, Any]) -> None:
    """Build-time guard against silent path collision.

    System invariant: every grid key (post stage-prefix strip) must distinguish
    its combos in the run_name. Otherwise two combos write to the same path and
    the later one silently overwrites the earlier.

    Enforces two things:
      1. If run_name.include is explicitly listed, every grid key is in it.
         (When include is omitted, _run_name auto-includes all grid keys, so
         this branch is irrelevant — uniqueness is structural.)
      2. Sanity bottom check: compute every (dataset, run_name) tuple from the
         expanded grid and reject duplicates. Catches edge cases like literal
         string run_name + multi-combo grid, or alias collisions.
    """
    grid = config.get("grid")
    grid_keys = _grid_keys(grid)
    run_cfg = config.get("run_name", {})

    if isinstance(run_cfg, dict):
        include_raw = run_cfg.get("include")
        if include_raw is not None:
            include_set = {str(k) for k in _as_list(include_raw)}
            missing = grid_keys - include_set
            if missing:
                raise SystemExit(
                    "Config invariant violated: grid keys "
                    f"{sorted(missing)} are not in run_name.include. "
                    "Different values of these keys would collide on the same path "
                    "and silently overwrite each other. Add them to run_name.include "
                    "(and an alias for readability)."
                )

    # Bottom check: explicit per-(dataset, run_name) uniqueness over all combos.
    datasets = [str(d) for d in _as_list(config.get("datasets"))]
    combos = _expand_grid(grid)
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in datasets:
        for combo in combos:
            combo_dataset = combo.get("dataset")
            if combo_dataset is not None and str(combo_dataset) != dataset:
                continue
            effective_combo = {k: v for k, v in combo.items() if k != "dataset"}
            name = _run_name(config, effective_combo)
            key = (dataset, name)
            if key in seen:
                raise SystemExit(
                    "Config invariant violated: two combos produce the same "
                    f"(dataset, run_name) = ({dataset!r}, {name!r}):\n"
                    f"  combo A: {seen[key]}\n"
                    f"  combo B: {combo}\n"
                    "These would write to identical paths. Likely cause: "
                    "run_name.include missing a distinguishing grid key, or "
                    "literal-string run_name used with multi-combo grid."
                )
            seen[key] = combo


def _completion_check_for(
    config: dict[str, Any],
    dataset: str,
    stage: str,
    run_name: str,
    params: dict[str, Any],
    combo: dict[str, Any],
    pred_len: int | None = None,
) -> StepCompletionCheck | None:
    """Build the StepCompletionCheck used to test "is this step already on disk?"."""
    model_id = str(config.get("model_id", params.get("model_id", "")))
    if not model_id:
        return None
    input_len = _lookup_param(config, combo, "input_len")
    if input_len is None:
        return None
    try:
        input_len_int = int(input_len)
    except (TypeError, ValueError):
        return None

    if stage == "pretrain":
        train_epochs_raw = params.get("train_epochs")
        try:
            train_epochs = int(train_epochs_raw) if train_epochs_raw is not None else None
        except (TypeError, ValueError):
            train_epochs = None
        return StepCompletionCheck(
            stage="pretrain",
            model_id=model_id,
            dataset=dataset,
            input_len=input_len_int,
            run_name=str(params.get("run_name", run_name)),
            train_epochs=train_epochs,
        )

    if stage == "train":
        if pred_len is None:
            return None
        return StepCompletionCheck(
            stage="train",
            model_id=model_id,
            dataset=dataset,
            input_len=input_len_int,
            run_name=str(params.get("train_run", params.get("run_name", run_name))),
            pred_len=int(pred_len),
            pretrain_run=str(params.get("pretrain_run", run_name)),
            load_pretrain_weights=params.get("load_pretrain_weights", 1),
        )

    if stage == "test":
        load_pre_default = config.get("stage_defaults", {}).get("train", {}).get("load_pretrain_weights", 1)
        load_pre = params.get("load_pretrain_weights", load_pre_default)
        default_pretrain = "" if str(load_pre).lower() in {"0", "false", "no", "off"} else run_name
        pred_lens_src = params.get("pred_lens")
        if pred_lens_src is None:
            pred_lens_src = config.get("stage_defaults", {}).get("train", {}).get("pred_lens")
        pred_lens_int: list[int] = []
        for pl in _as_list(pred_lens_src):
            try:
                pred_lens_int.append(int(pl))
            except (TypeError, ValueError):
                continue
        return StepCompletionCheck(
            stage="test",
            model_id=model_id,
            dataset=dataset,
            input_len=input_len_int,
            run_name=str(params.get("train_run", params.get("run_name", run_name))),
            pred_lens=pred_lens_int,
            pretrain_run=str(params.get("pretrain_run", default_pretrain)),
            load_pretrain_weights=load_pre,
        )

    return None


def _apply_resume_skip(chains: list[list[Step]], enabled: bool) -> dict[str, int]:
    """Mark steps whose canonical output already exists. Returns per-stage skip counts.

    Chain-internal invalidation: if pretrain in chain C will re-run, every train +
    test in C is un-skipped (encoder will change so downstream outputs are stale).
    If any train(pl=N) in C will re-run, the chain's test is un-skipped (test bundles
    all pred_lens so a single pl re-run makes the whole results.json stale).
    """
    tally = {"pretrain": 0, "train": 0, "test": 0}
    if not enabled:
        return tally
    for chain in chains:
        for step in chain:
            if step.completion_check is not None and is_step_done(step.completion_check):
                step.skip_reason = "completed"

        if any(s.stage == "pretrain" and s.skip_reason is None for s in chain):
            for s in chain:
                if s.stage in {"train", "test"}:
                    s.skip_reason = None
        if any(s.stage == "train" and s.skip_reason is None for s in chain):
            for s in chain:
                if s.stage == "test":
                    s.skip_reason = None

        for s in chain:
            if s.skip_reason == "completed":
                tally[s.stage] = tally.get(s.stage, 0) + 1
    return tally


def _build_chains(config: dict[str, Any], run_dir: Path) -> list[list[Step]]:
    datasets = [str(item) for item in _as_list(config.get("datasets"))]
    if not datasets:
        raise SystemExit("datasets must list at least one dataset.")
    stages = [str(item) for item in _as_list(config.get("stages", ["pretrain", "train", "test"]))]
    combos = _expand_grid(config.get("grid"))

    raw_chains: list[list[tuple[str, str, str, dict[str, Any], list[str], dict[str, Any], str, StepCompletionCheck | None]]] = []
    for dataset in datasets:
        for combo in combos:
            combo_dataset = combo.get("dataset")
            if combo_dataset is not None and str(combo_dataset) != dataset:
                continue
            effective_combo = {key: value for key, value in combo.items() if key != "dataset"}
            chain: list[tuple[str, str, str, dict[str, Any], list[str], dict[str, Any], str, StepCompletionCheck | None]] = []
            run_name = _run_name(config, effective_combo)
            for stage in stages:
                base_params = _stage_params(config, dataset, stage, effective_combo)
                if base_params.get("skip"):
                    continue
                for variant in _stage_variants(config, stage):
                    params = _apply_variant_params(base_params, stage, variant, run_name)
                    if params.get("skip"):
                        continue
                    label_run = _step_label_run_name(stage, run_name, params)
                    if stage == "train":
                        pred_lens = _as_list(params.get("pred_lens"))
                        if not pred_lens:
                            raise SystemExit("train stage requires pred_lens.")
                        for pred_len in pred_lens:
                            step_params = dict(params)
                            step_params.pop("pred_lens", None)
                            command = _make_command(config, dataset, stage, run_name, step_params, int(pred_len))
                            label = f"{dataset} {stage} {label_run} pl{pred_len}"
                            check = _completion_check_for(
                                config, dataset, stage, run_name, step_params, combo, pred_len=int(pred_len)
                            )
                            chain.append((dataset, stage, label_run, combo, command, step_params, label, check))
                    else:
                        command = _make_command(config, dataset, stage, run_name, params)
                        label = f"{dataset} {stage} {label_run}"
                        check = _completion_check_for(config, dataset, stage, run_name, params, combo)
                        chain.append((dataset, stage, label_run, combo, command, params, label, check))
            if chain:
                raw_chains.append(chain)

    total = sum(len(chain) for chain in raw_chains)
    chains: list[list[Step]] = []
    index = 1
    for raw_chain in raw_chains:
        steps: list[Step] = []
        for dataset, stage, run_name, combo, command, params, label, check in raw_chain:
            safe_label = _sanitize_piece(label)
            log_path = run_dir / "logs" / dataset / run_name / f"{index:04d}_{stage}_{safe_label}.log"
            steps.append(
                Step(
                    index=index,
                    total=total,
                    dataset=dataset,
                    stage=stage,
                    run_name=run_name,
                    command=command,
                    log_path=log_path,
                    combo=dict(combo),
                    label=label,
                    env=_base_env(config, params),
                    completion_check=check,
                )
            )
            index += 1
        chains.append(steps)
    return chains


def _flatten_chains(chains: list[list[Step]]) -> list[Step]:
    return [step for chain in chains for step in chain]


def _tail(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""
    data = path.read_text(errors="replace").splitlines()
    return "\n".join(data[-lines:])


def _run_sequential(steps: list[Step], on_skip) -> int:
    """Fallback runner used when no GPU pool is configured (CPU-only or single-device)."""
    failures = 0
    for step in steps:
        if step.skip_reason:
            on_skip(step)
            continue
        rel_log = step.log_path.relative_to(PROJECT_ROOT)
        step.log_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with step.log_path.open("w") as handle:
            handle.write(f"# label: {step.label}\n")
            handle.write(f"# command: {shlex.join(step.command)}\n")
            handle.write(f"# log: {step.log_path}\n\n")
            handle.flush()
            proc = subprocess.Popen(
                step.command,
                cwd=PROJECT_ROOT,
                env=step.env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
            proc.wait()
            rc = proc.returncode
        elapsed = time.time() - started
        status = "done" if rc == 0 else f"failed({rc})"
        print(f"[{status}] [{step.index}/{step.total}] {step.label} ({elapsed:,.0f}s) log: {rel_log}")
        if rc != 0:
            failures += 1
            tail = _tail(step.log_path)
            if tail:
                print("Last log lines:")
                print(tail)
            break
    return failures


def _dry_run(steps: list[Step]) -> None:
    for step in steps:
        rel_log = step.log_path.relative_to(PROJECT_ROOT)
        marker = "would-skip" if step.skip_reason else "dry-run"
        print(f"[{marker}] [{step.index}/{step.total}] {step.label}")
        print(f"          log: {rel_log}")
        if step.skip_reason:
            print(f"          skip: {step.skip_reason}")
        else:
            print(f"          cmd: {shlex.join(step.command)}")


def _parse_cuda_devices(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, list):
        items = value
    else:
        items = str(value).split(",")
    out: list[int] = []
    for item in items:
        s = str(item).strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except ValueError as exc:
            raise SystemExit(f"cuda_visible_devices entry {s!r} is not an integer.") from exc
    return out


def _start_step(
    step: Step,
    chain_index: int,
    step_index_in_chain: int,
    gpu_id: int,
    attempt: int,
) -> RunningJob:
    rel_log = step.log_path.relative_to(PROJECT_ROOT)
    env = step.env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    step.log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = step.log_path.open("w")
    handle.write(f"# label: {step.label}\n")
    handle.write(f"# command: {shlex.join(step.command)}\n")
    handle.write(f"# cuda_visible_devices: {gpu_id}\n")
    handle.write(f"# attempt: {attempt + 1}\n")
    handle.write(f"# log: {step.log_path}\n\n")
    handle.flush()
    process = subprocess.Popen(
        step.command,
        cwd=PROJECT_ROOT,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    attempt_note = f" attempt {attempt + 1}" if attempt > 0 else ""
    print(
        f"[start] [{step.index}/{step.total}] gpu={gpu_id}{attempt_note} "
        f"{step.label} log: {rel_log}"
    )
    return RunningJob(
        chain_index=chain_index,
        step_index_in_chain=step_index_in_chain,
        gpu_id=gpu_id,
        step=step,
        process=process,
        handle=handle,
        started_at=time.time(),
        attempt=attempt,
    )


_STAGE_ORDER = ("pretrain", "train", "test")


def _make_progress_tracker(steps: list[Step]):
    """Return (on_finish, on_skip) callbacks that print a per-event progress line.

    Tracks per-stage and total done counts. Failures are reported but not
    counted as "done" (so partial-completion runs show a stalled progress).
    """
    total_per_stage = {stage: sum(1 for s in steps if s.stage == stage) for stage in _STAGE_ORDER}
    total = len(steps)
    done_per_stage: dict[str, int] = {stage: 0 for stage in _STAGE_ORDER}

    def progress_line() -> str:
        done = sum(done_per_stage.values())
        pct = (done / total * 100.0) if total else 100.0
        per_stage = "  ".join(
            f"{stage} {done_per_stage[stage]}/{total_per_stage[stage]}"
            for stage in _STAGE_ORDER
            if total_per_stage[stage] > 0
        )
        return f"progress: {pct:5.1f}% ({done}/{total})  {per_stage}"

    def on_finish(job: RunningJob, return_code: int) -> None:
        elapsed = time.time() - job.started_at
        rel_log = job.step.log_path.relative_to(PROJECT_ROOT)
        if return_code == 0:
            status = "done"
        else:
            status = f"failed({return_code})"
        attempt_note = f" attempt {job.attempt + 1}" if job.attempt > 0 else ""
        print(
            f"[{status}] [{job.step.index}/{job.step.total}] gpu={job.gpu_id}{attempt_note} "
            f"{job.step.label} ({elapsed:,.0f}s) log: {rel_log}"
        )
        if return_code != 0:
            tail = _tail(job.step.log_path)
            if tail:
                print("Last log lines:")
                print(tail)
            return
        done_per_stage[job.step.stage] = done_per_stage.get(job.step.stage, 0) + 1
        print(progress_line())

    def on_skip(step: Step) -> None:
        print(
            f"[skip] [{step.index}/{step.total}] {step.label} "
            f"(already done — {step.skip_reason})"
        )
        done_per_stage[step.stage] = done_per_stage.get(step.stage, 0) + 1
        print(progress_line())

    return on_finish, on_skip


def _on_oom_requeue(job: RunningJob, new_attempt: int) -> None:
    elapsed = time.time() - job.started_at
    rel_log = job.step.log_path.relative_to(PROJECT_ROOT)
    print(
        f"[oom-requeue] [{job.step.index}/{job.step.total}] gpu={job.gpu_id} "
        f"{job.step.label} ({elapsed:,.0f}s) -> retry {new_attempt + 1} when a job finishes. "
        f"log: {rel_log}"
    )


def _write_manifest(run_dir: Path, config_path: Path, config: dict[str, Any], steps: list[Step]) -> None:
    payload = {
        "config_path": str(config_path),
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "config": config,
        "steps": [
            {
                "index": step.index,
                "dataset": step.dataset,
                "stage": step.stage,
                "run_name": step.run_name,
                "label": step.label,
                "command": step.command,
                "log_path": str(step.log_path),
                "combo": step.combo,
                "skip_reason": step.skip_reason,
            }
            for step in steps
        ],
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")



def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _md_cell(value: Any, limit: int = 120) -> str:
    text = "" if value is None else str(value)
    text = text.replace("|", "\\|").replace("\n", " ")
    if len(text) > limit:
        text = text[: limit - 3].rstrip() + "..."
    return text


def _summarize_grid(config: dict[str, Any]) -> str:
    grid = config.get("grid")
    if not grid:
        return "-"
    if isinstance(grid, list):
        return f"{len(grid)} explicit combo(s)"
    if not isinstance(grid, dict):
        return type(grid).__name__
    parts: list[str] = []
    for key, raw_values in grid.items():
        values = _as_list(raw_values)
        if len(values) <= 4:
            rendered = ",".join(str(v) for v in values)
        else:
            rendered = f"{values[0]}..{values[-1]} ({len(values)})"
        parts.append(f"{key}={rendered}")
    return "; ".join(parts)


def _append_batch_run_record(
    run_dir: Path,
    config_path: Path,
    config: dict[str, Any],
    steps: list[Step],
    skip_tally: dict[str, int],
    failures: int,
) -> None:
    """Append one batch-level ledger row after a real run finishes."""
    BATCH_RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if BATCH_RUNS_PATH.exists():
        text = BATCH_RUNS_PATH.read_text()
    else:
        text = (
            "# Batch Runs\n\n"
            "Managed by `entry/batch.py`. One row is appended when a non-dry-run batch exits.\n\n"
            "| finished_at | status | batch | model_id | config | datasets | stages | grid | steps | skipped | failed | manifest |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |\n"
        )
    if not text.endswith("\n"):
        text += "\n"
    status = "failed" if failures else "success"
    row = [
        datetime.now(UTC).isoformat(timespec="seconds"),
        status,
        config.get("name") or config_path.stem,
        config.get("model_id", ""),
        _rel(config_path),
        ", ".join(str(item) for item in _as_list(config.get("datasets"))),
        ", ".join(str(item) for item in _as_list(config.get("stages", ["pretrain", "train", "test"]))),
        _summarize_grid(config),
        len(steps),
        sum(skip_tally.values()),
        failures,
        _rel(run_dir / "manifest.json"),
    ]
    text += "| " + " | ".join(_md_cell(cell) for cell in row) + " |\n"
    BATCH_RUNS_PATH.write_text(text)


def _warn_deprecated(config: dict[str, Any]) -> None:
    legacy = [k for k in ("jobs", "worker_cuda_devices", "jobs_per_gpu",
                          "per_job_gpu_mb", "auto_concurrent_target") if k in config]
    if legacy:
        print(
            f"[warn] config keys {legacy} are deprecated and ignored. "
            "Scheduler now admits step launches by live GPU free RAM (see `min_free_mb`); "
            "use `concurrent` for a CPU-side hard cap.",
            file=sys.stderr,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run config-driven experiment batches.")
    parser.add_argument("config", help="Path or name under entry/batch_configs.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running them.")
    parser.add_argument("--keep-going", action="store_true", help="Continue after a failed step.")
    parser.add_argument(
        "--max-oom-retries",
        type=int,
        default=None,
        help="Override OOM retry cap (default 2 or config max_oom_retries).",
    )
    parser.add_argument(
        "--min-free-mb",
        type=int,
        default=None,
        help=(
            "GPU is admitted for a step launch only if its current free RAM "
            ">= this many MiB (live query, no caching). Override config "
            "`min_free_mb` (default 1000)."
        ),
    )
    parser.add_argument(
        "--concurrent",
        type=int,
        default=None,
        help=(
            "Global hard cap on simultaneous step subprocesses across all "
            "GPUs. Memory admission self-limits most of the time; use this "
            "when CPU / driver headroom is the bottleneck. Override config "
            "`concurrent`. Default: no cap (memory admission alone)."
        ),
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help=(
            "Seconds to sleep after each step launch before considering the next. "
            "Default 15 (default-on). Protects the NVIDIA driver from concurrent "
            "cudaInit / cuBLAS handle creation across step subprocesses in the "
            "same burst, and gives each launched process time to claim CUDA memory "
            "so the next pick sees a fresh nvidia-smi snapshot. Override config 'delay'. "
            "Set 0 only when burst is known-small or driver headroom is plentiful."
        ),
    )
    parser.add_argument(
        "--stages",
        default=None,
        help="Comma-separated stages to run, e.g. train,test. Overrides config stages.",
    )
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Ignore existing checkpoints / results.json and re-run every step.",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    config = _load_config(config_path)
    _warn_deprecated(config)
    if args.stages is not None:
        config["stages"] = [s.strip() for s in args.stages.split(",") if s.strip()]
    _check_grid_include_invariant(config)
    name = str(config.get("name") or config_path.stem)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = PROJECT_ROOT / "outputs" / "batch" / name / stamp
    chains = _build_chains(config, run_dir)
    steps = _flatten_chains(chains)
    resume_enabled = not args.rerun and bool(config.get("resume_skip", True))
    skip_tally = _apply_resume_skip(chains, resume_enabled)
    _write_manifest(run_dir, config_path, config, steps)

    gpu_ids = _parse_cuda_devices(config.get("cuda_visible_devices"))
    max_oom_retries = (
        args.max_oom_retries
        if args.max_oom_retries is not None
        else int(config.get("max_oom_retries", 2))
    )
    min_free_mb = (
        args.min_free_mb
        if args.min_free_mb is not None
        else int(config.get("min_free_mb", 1000))
    )
    if min_free_mb < 0:
        raise SystemExit("min_free_mb must be >= 0")
    max_concurrent: int | None
    if args.concurrent is not None:
        max_concurrent = args.concurrent
    elif "concurrent" in config:
        max_concurrent = int(config["concurrent"])
    else:
        max_concurrent = None
    if max_concurrent is not None and max_concurrent < 1:
        raise SystemExit("concurrent must be >= 1")
    launch_delay_s: float
    if args.delay is not None:
        launch_delay_s = float(args.delay)
    elif "delay" in config:
        launch_delay_s = float(config["delay"])
    else:
        launch_delay_s = 15.0
    if launch_delay_s < 0:
        raise SystemExit("delay must be >= 0")

    print(f"Batch: {name}")
    print(f"Config: {config_path}")
    print(f"Run dir: {run_dir.relative_to(PROJECT_ROOT)}")
    print(f"Total steps: {len(steps)}")
    print(f"Chains: {len(chains)}")
    skipped_total = sum(skip_tally.values())
    if resume_enabled:
        per_stage = ", ".join(f"{stage}:{n}" for stage, n in skip_tally.items() if n) or "none"
        print(
            f"Resume skip: {skipped_total} / {len(steps)} already complete ({per_stage}); "
            f"running {len(steps) - skipped_total}"
        )
    else:
        print("Resume skip: disabled (--rerun or resume_skip=false)")
    if gpu_ids:
        cap_note = f", global cap {max_concurrent}" if max_concurrent is not None else ""
        print(
            f"GPUs: {gpu_ids} (live-query admission, min_free_mb={min_free_mb}{cap_note})"
        )
    else:
        print("GPUs: none configured (cuda_visible_devices unset) — running sequentially on default device")
    if config.get("gpu_memory_fraction") is not None:
        print(f"GPU memory fraction: {config['gpu_memory_fraction']} per process")
    print(f"OOM retries: {max_oom_retries}")
    if min_free_mb > 0:
        print(f"GPU free-RAM threshold: {min_free_mb} MiB (lower → skip)")
    if launch_delay_s > 0:
        print(f"Launch delay: {launch_delay_s:g}s between step launches (driver-safety stagger)")

    if gpu_ids and (config.get("gpu_memory_fraction") is not None or min_free_mb > 0):
        try:
            snaps = query_gpu_free_mb(gpu_ids)
            for gid in gpu_ids:
                snap = snaps.get(gid)
                if snap and snap.total_mb:
                    used_pct = 1.0 - snap.free_mb / snap.total_mb
                    if used_pct > 0.5:
                        print(
                            f"[warn] GPU {gid} already {used_pct:.0%} used "
                            f"({snap.free_mb} MiB free / {snap.total_mb} MiB total)"
                        )
            if min_free_mb > 0:
                max_total = max((s.total_mb for s in snaps.values()), default=0)
                if max_total and min_free_mb > max_total:
                    raise SystemExit(
                        f"min_free_mb={min_free_mb} exceeds the largest GPU "
                        f"({max_total} MiB total) — no GPU could ever qualify."
                    )
                if all(
                    snaps.get(gid) and snaps[gid].free_mb < min_free_mb
                    for gid in gpu_ids
                ):
                    print(
                        f"[warn] every configured GPU is currently below the "
                        f"{min_free_mb} MiB threshold — batch will wait for one to free up."
                    )
        except SystemExit:
            raise
        except Exception as exc:
            print(f"[warn] could not query nvidia-smi: {exc}", file=sys.stderr)

    if args.dry_run:
        _dry_run(steps)
        return

    on_finish, on_skip = _make_progress_tracker(steps)

    if not gpu_ids:
        failures = _run_sequential(steps, on_skip)
        _append_batch_run_record(run_dir, config_path, config, steps, skip_tally, failures)
        print(
            f"Finished batch with {failures} failure(s). "
            f"Manifest: {(run_dir / 'manifest.json').relative_to(PROJECT_ROOT)}"
        )
        if failures:
            raise SystemExit(1)
        return

    sched = Scheduler(
        chains=chains,
        config=SchedulerConfig(
            gpu_ids=gpu_ids,
            max_oom_retries=max_oom_retries,
            keep_going=args.keep_going,
            min_free_mb=min_free_mb,
            max_concurrent=max_concurrent,
            launch_delay_s=launch_delay_s,
        ),
        start_step=_start_step,
        on_finish=on_finish,
        on_oom_requeue=_on_oom_requeue,
        on_skip=on_skip,
    )
    failures = sched.run()
    _append_batch_run_record(run_dir, config_path, config, steps, skip_tally, failures)

    print(f"Finished batch with {failures} failure(s). Manifest: {(run_dir / 'manifest.json').relative_to(PROJECT_ROOT)}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
