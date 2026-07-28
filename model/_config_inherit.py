"""Shared helpers for inheriting architectural fields from a pretrain config.json.

Used by train CLIs in SDTA / TimeSiam / PatchTST. The contract:
  - The pretrain phase writes `config.json` next to `ckpt_best.pth`.
  - The train phase strips arch params from its CLI and reads them from that JSON.
  - Mismatched/missing fields raise so the user knows immediately.

Folder conventions:
  SDTA/TimeSiam: outputs/pretrain/{model_id}/{dataset}/il{input_len}/{run_name}/config.json
  PatchTST:      outputs/pretrain/{model_id}/{dataset}/{context_points}/{run_name}/config.json
"""
from __future__ import annotations

import json
from pathlib import Path


PRETRAIN_ROOT = Path(__file__).resolve().parents[1] / "outputs" / "pretrain"


class ConfigInheritError(RuntimeError):
    """Raised when the pretrain config cannot be resolved or is missing fields."""


def find_pretrain_config_path(
    model_id: str,
    dataset: str,
    run_name: str,
    dir_glob: str = "il*",
) -> Path:
    """Glob `outputs/pretrain/{model_id}/{dataset}/{dir_glob}/{run_name}/config.json`.

    `dir_glob` is the wildcard for the input-length-like directory.
    SDTA / TimeSiam use the default `il*`; PatchTST passes `*` because its
    pretrain dir is named with the raw context_points integer.

    Returns the single matching path. Errors if 0 or >1 candidates exist —
    the user must disambiguate by giving a unique `run_name`.
    """
    pattern = f"{model_id}/{dataset}/{dir_glob}/{run_name}/config.json"
    candidates = sorted(PRETRAIN_ROOT.glob(pattern))
    if not candidates:
        raise ConfigInheritError(
            f"No pretrain config found matching: {PRETRAIN_ROOT}/{pattern}. "
            "Check --model_id / --dataset / --pretrain_run."
        )
    if len(candidates) > 1:
        listing = "\n  ".join(str(c) for c in candidates)
        raise ConfigInheritError(
            f"Multiple pretrain configs match run_name='{run_name}':\n  {listing}\n"
            "Rename one of the pretrain runs so the lookup is unique."
        )
    return candidates[0]


def load_pretrain_config(
    model_id: str,
    dataset: str,
    run_name: str,
    dir_glob: str = "il*",
) -> dict:
    """Read and JSON-parse the unique pretrain config for (model_id, dataset, run_name)."""
    path = find_pretrain_config_path(model_id, dataset, run_name, dir_glob=dir_glob)
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigInheritError(f"Failed to parse {path}: {e}") from e


def inherit_fields(pretrain_cfg: dict, fields: list[str]) -> dict:
    """Pull `fields` from `pretrain_cfg`. Errors if any field is missing."""
    missing = [f for f in fields if f not in pretrain_cfg]
    if missing:
        raise ConfigInheritError(
            f"Pretrain config missing inherited fields: {missing}. "
            f"Available keys: {sorted(pretrain_cfg.keys())}"
        )
    return {f: pretrain_cfg[f] for f in fields}
