"""Shared helpers for writing the unified per-train_run test results file.

Every model package calls `write_results_file` after looping its own
pred_lens, so the file format stays consistent across models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# Canonical per-stage completion sentinel. Every stage (pretrain / train / test)
# writes an empty `done` file as its LAST action; resume detection (entry/batch_src/
# resume.py) treats "done exists" as "stage cleanly finished". Decoupled from loss
# logs so completion is a simple existence check.
DONE_MARKER = "done"


def write_done(out_dir: Path) -> None:
    """Mark a stage output dir as cleanly complete (empty `done` sentinel)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / DONE_MARKER).write_text("")


def write_results_file(
    out_dir: Path,
    setting: str,
    model_id: str,
    dataset_name: str,
    input_len: int,
    pretrain_run: str,
    train_run: str,
    pred_len_metrics: dict[int, dict[str, float]],
    pretrain_config: dict[str, Any] | None,
    train_config: dict[str, Any],
    test_config: dict[str, Any],
) -> None:
    """Write results.json (machine) and results.txt (human) once per train_run.

    Designed as a single write: callers should accumulate metrics across all
    pred_lens, then call this helper exactly once.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "setting": setting,
        "model": model_id,
        "dataset": dataset_name,
        "input_len": input_len,
        "pretrain_run": pretrain_run,
        "train_run": train_run,
        "scores": {str(pl): pred_len_metrics[pl] for pl in sorted(pred_len_metrics)},
        "pretrain_config": pretrain_config,
        "train_config": train_config,
        "test_config": test_config,
    }
    (out_dir / "results.json").write_text(json.dumps(payload, indent=2) + "\n")

    lines: list[str] = []
    lines.append(f"# {model_id} / {dataset_name} / il{input_len}")
    lines.append(f"pretrain_run: {pretrain_run}")
    lines.append(f"train_run:    {train_run}")
    lines.append("")
    lines.append("## Scores")
    lines.append(f"{'pred_len':>10s}  {'mse':>12s}  {'mae':>12s}")
    for pl in sorted(pred_len_metrics):
        m = pred_len_metrics[pl]
        lines.append(f"{pl:>10d}  {m['mse']:>12.6f}  {m['mae']:>12.6f}")
    lines.append("")
    lines.append("## Pretrain config")
    lines.append(json.dumps(pretrain_config, indent=2) if pretrain_config else "(no pretrain)")
    lines.append("")
    lines.append("## Train config")
    lines.append(json.dumps(train_config, indent=2))
    lines.append("")
    lines.append("## Test config")
    lines.append(json.dumps(test_config, indent=2))
    (out_dir / "results.txt").write_text("\n".join(lines) + "\n")

    write_done(out_dir)  # last write: test stage complete
