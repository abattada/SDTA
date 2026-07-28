"""Skip-if-done detection for entry/batch.py.

A batch step counts as "done" only when an end-of-stage marker exists, not
merely the best-so-far checkpoint:

* pretrain (SDTA_v* / TimeSiam) — `epoch_losses.jsonl` line count >= train_epochs
* pretrain (PatchTST)           — `losses.csv` exists (written after fit completes)
* train    (SDTA_v* / TimeSiam) — `training_summary.json` exists (written after the epoch loop)
* train    (PatchTST)           — `losses.csv` exists
* test     (all families)       — `results.json` exists and `scores` covers every required pred_len

`ckpt_best.pth` / `checkpoint.pth` are written every time val_loss improves and
can exist as early as epoch 1, so they are NOT used as completion markers — a
Ctrl-C at epoch 5 of 50 would otherwise look "done".

tag = "no_pretrain" when pretrain_run is empty or load_pretrain_weights is falsy,
else tag = pretrain_run.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_ROOT = PROJECT_ROOT / "outputs" / "pretrain"
TRAIN_ROOT = PROJECT_ROOT / "outputs" / "train"
TEST_ROOT = PROJECT_ROOT / "outputs" / "test"


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() not in {"", "0", "false", "no", "off"}


def pretrain_tag(pretrain_run: str, load_pretrain_weights: Any) -> str:
    if not pretrain_run or not _truthy(load_pretrain_weights):
        return "no_pretrain"
    return pretrain_run


def pretrain_run_dir(model_id: str, dataset: str, input_len: int, run_name: str) -> Path:
    return PRETRAIN_ROOT / model_id / dataset / f"il{input_len}" / run_name


def train_run_dir(
    model_id: str,
    dataset: str,
    input_len: int,
    train_run: str,
    pred_len: int,
    pretrain_run: str,
    load_pretrain_weights: Any,
) -> Path:
    tag = pretrain_tag(pretrain_run, load_pretrain_weights)
    return (
        TRAIN_ROOT
        / model_id
        / dataset
        / f"il{input_len}"
        / f"pre_{tag}"
        / train_run
        / f"pl{pred_len}"
    )


def _is_patchtst(model_id: str) -> bool:
    return model_id.startswith("PatchTST")


def _count_nonblank_lines(path: Path) -> int:
    try:
        with path.open("r") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


# Canonical completion sentinel: every stage writes an empty `done` file as its
# last action (model/_test_io.py write_done). "done exists" == "stage finished".
# Legacy family-specific markers below stay as a backward-compat fallback for runs
# produced before this convention. See docs/check/model_audit.md §4.
DONE_MARKER = "done"


def _stage_done(stage_dir: Path) -> bool:
    return (stage_dir / DONE_MARKER).exists()


def pretrain_done(
    model_id: str,
    dataset: str,
    input_len: int,
    run_name: str,
    train_epochs: int | None,
) -> bool:
    """End-of-pretrain marker: `done` sentinel, else legacy family-specific."""
    run_dir = pretrain_run_dir(model_id, dataset, input_len, run_name)
    if _stage_done(run_dir):
        return True
    if _is_patchtst(model_id):
        return (run_dir / "losses.csv").exists()
    if train_epochs is None or train_epochs <= 0:
        return False
    jsonl = run_dir / "epoch_losses.jsonl"
    if not jsonl.exists():
        return False
    return _count_nonblank_lines(jsonl) >= train_epochs


def train_done(
    model_id: str,
    dataset: str,
    input_len: int,
    train_run: str,
    pred_len: int,
    pretrain_run: str,
    load_pretrain_weights: Any,
) -> bool:
    """End-of-train marker: `done` sentinel, else legacy family-specific."""
    run_dir = train_run_dir(
        model_id, dataset, input_len, train_run, pred_len, pretrain_run, load_pretrain_weights
    )
    if _stage_done(run_dir):
        return True
    if _is_patchtst(model_id):
        return (run_dir / "losses.csv").exists()
    return (run_dir / "training_summary.json").exists()


def test_results_path(
    model_id: str,
    dataset: str,
    input_len: int,
    train_run: str,
    pretrain_run: str,
    load_pretrain_weights: Any,
) -> Path:
    tag = pretrain_tag(pretrain_run, load_pretrain_weights)
    return (
        TEST_ROOT
        / model_id
        / dataset
        / f"il{input_len}"
        / f"pre_{tag}"
        / train_run
        / "results.json"
    )


def test_results_cover(path: Path, required_pred_lens: list[int]) -> bool:
    """True iff results.json exists and its `scores` dict covers every required pred_len."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    scores = data.get("scores")
    if not isinstance(scores, dict):
        return False
    have = set(str(k) for k in scores.keys())
    return all(str(pl) in have for pl in required_pred_lens)


@dataclass
class StepCompletionCheck:
    stage: str
    model_id: str
    dataset: str
    input_len: int
    run_name: str  # the stage's effective run name (pretrain run, train_run, etc.)
    pred_len: int | None = None  # train only
    pred_lens: list[int] = field(default_factory=list)  # test only
    pretrain_run: str = ""  # train/test only
    load_pretrain_weights: Any = 1  # train/test only
    train_epochs: int | None = None  # pretrain only (line count target for epoch_losses.jsonl)


def is_step_done(check: StepCompletionCheck) -> bool:
    if check.stage == "pretrain":
        return pretrain_done(
            check.model_id, check.dataset, check.input_len, check.run_name, check.train_epochs
        )
    if check.stage == "train":
        if check.pred_len is None:
            return False
        return train_done(
            check.model_id,
            check.dataset,
            check.input_len,
            check.run_name,
            check.pred_len,
            check.pretrain_run,
            check.load_pretrain_weights,
        )
    if check.stage == "test":
        if not check.pred_lens:
            return False
        results_path = test_results_path(
            check.model_id,
            check.dataset,
            check.input_len,
            check.run_name,
            check.pretrain_run,
            check.load_pretrain_weights,
        )
        # `done` sentinel (written after results.json) or legacy coverage check.
        return _stage_done(results_path.parent) or test_results_cover(results_path, check.pred_lens)
    return False


__all__ = [
    "PRETRAIN_ROOT",
    "TRAIN_ROOT",
    "TEST_ROOT",
    "StepCompletionCheck",
    "is_step_done",
    "pretrain_done",
    "pretrain_run_dir",
    "pretrain_tag",
    "test_results_cover",
    "test_results_path",
    "train_done",
    "train_run_dir",
]
