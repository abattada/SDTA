#!/usr/bin/env python3
"""Summarize per-epoch wall-clock time from training logs.

Implements the timing protocol behind the paper's cost analysis: run the
`entry/batch_configs/time/*.json` configs (6 epochs, single seed 97, one GPU,
serial execution, num_workers=1), then average the per-epoch `elapsed_sec`
of epochs 2..N — epoch 1 is dropped as CUDA/cuDNN warm-up.

Reads every `epoch_losses.jsonl` (pretrain stage) and `epoch_metrics.jsonl`
(train stage) under outputs/pretrain and outputs/train.

The PatchTST baseline never writes an `elapsed_sec` field: its trainer prints
an epoch table whose last column is an `(h:)mm:ss` wall-clock string
(`model/PatchTST/src/callback/tracking.py`, `format_time`), which the batch
runner captures under `outputs/batch/<config>/<timestamp>/logs/`. Those tables
are parsed here so the PatchTST-SSL row of the cost table is script-derived
like every other row. Two caveats:
  * `format_time` truncates with `int(t)`, so PatchTST epoch times carry
    1-second resolution; that granularity is PatchTST's own logging, not a
    choice of this script, and it is why its cost-table entries are whole
    seconds.
  * Re-measuring a timing config leaves several timestamped batch directories.
    Timestamps sort chronologically, so the newest log that actually contains
    an epoch table wins; failed attempts (no table) never displace a good one.
An `elapsed_sec` series always takes precedence over an `(h:)mm:ss` one for the
same run, since it is the higher-resolution source.

Usage (from the repository root):
    python scripts/epoch_time.py [--root PATH] [--stage pretrain|train|both]
    python scripts/epoch_time.py --contains time_  # restrict by run-name substring
"""
from __future__ import annotations

import argparse
import json
import re
import shlex
import sys
from pathlib import Path

# One captured epoch row: an integer epoch, one or more float metrics, then the
# (h:)mm:ss column. The batch runner's own header lines start with '#', and the
# trainer's chatter ("Better model found at epoch 0 ...") never starts with a
# bare integer, so neither can match.
EPOCH_ROW_RE = re.compile(r"^\s*\d+(?:\s+[-+\d.eEnaN]+)+\s+(?:\d+:)?\d{1,2}:\d{2}\s*$")
COMMAND_RE = re.compile(r"^# command:\s*(.*)$")
MODULE_RE = re.compile(r"^model\.([^.]+)\.(pretrain|train)$")

def hms_seconds(token: str) -> float:
    """'(h:)mm:ss' -> seconds (whole seconds; PatchTST truncates when logging)."""
    secs = 0
    for part in token.split(":"):
        secs = secs * 60 + int(part)
    return float(secs)

def epoch_seconds(jsonl: Path) -> list[float]:
    out = []
    for line in jsonl.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "elapsed_sec" in rec:
            out.append(float(rec["elapsed_sec"]))
    return out

def log_epoch_seconds(log: Path) -> list[float]:
    """Per-epoch seconds from a captured (h:)mm:ss epoch table."""
    return [hms_seconds(line.split()[-1])
            for line in log.read_text(errors="replace").splitlines()
            if EPOCH_ROW_RE.match(line)]

def log_run_id(log: Path, root: Path) -> tuple[str, str] | None:
    """(stage, run key) from a batch log header, keyed like the jsonl rows.

    Every captured log opens with a shlex-joined '# command:' line written by
    entry/batch.py, so the run identity is recovered exactly. A pretrain key is
    built straight from the flags; a train command inherits --input_len from
    the pretrain config, so its il* component is resolved against outputs/train.
    """
    with log.open(errors="replace") as handle:
        for _ in range(8):
            line = handle.readline()
            if not line:
                return None
            found = COMMAND_RE.match(line)
            if found:
                break
        else:
            return None
    argv = shlex.split(found.group(1))
    if "-m" not in argv or argv.index("-m") + 1 >= len(argv):
        return None
    module = MODULE_RE.match(argv[argv.index("-m") + 1])
    if not module:
        return None
    model, stage = module.groups()
    flags = {a: argv[i + 1] for i, a in enumerate(argv[:-1]) if a.startswith("--")}
    try:
        if stage == "pretrain":
            return stage, (f"{model}/{flags['--dataset']}"
                           f"/il{flags['--input_len']}/{flags['--run_name']}")
        base = root / "outputs" / "train" / model / flags["--dataset"]
        leaf = (f"il*/pre_{flags['--pretrain_run']}/{flags['--run_name']}"
                f"/pl{flags['--pred_len']}")
    except KeyError:
        return None
    hits = sorted(base.glob(leaf))
    if not hits:
        return None
    return stage, str(hits[0].relative_to(root / "outputs" / "train"))

def summarize_logs(root: Path, stage: str, contains: str) -> dict[str, tuple[int, float]]:
    """Runs whose epoch times exist only as (h:)mm:ss in outputs/batch logs."""
    found: dict[str, tuple[int, float]] = {}
    base = root / "outputs" / "batch"
    if not base.is_dir():
        return found
    for log in sorted(base.rglob("*.log")):     # chronological: newest wins
        ident = log_run_id(log, root)
        if not ident or ident[0] != stage:
            continue
        rel = ident[1]
        if contains and contains not in rel:
            continue
        secs = log_epoch_seconds(log)
        if len(secs) < 2:
            continue
        steady = secs[1:]  # drop epoch 1 (warm-up)
        found[rel] = (len(secs), sum(steady) / len(steady))
    return found

def summarize(root: Path, stage: str, contains: str) -> list[tuple[str, int, float, str]]:
    fname = {"pretrain": "epoch_losses.jsonl", "train": "epoch_metrics.jsonl"}[stage]
    rows = []
    base = root / "outputs" / stage
    if base.is_dir():
        for jsonl in sorted(base.rglob(fname)):
            rel = jsonl.parent.relative_to(base)
            if contains and contains not in str(rel):
                continue
            secs = epoch_seconds(jsonl)
            if len(secs) < 2:
                continue
            steady = secs[1:]  # drop epoch 1 (warm-up)
            rows.append((str(rel), len(secs), sum(steady) / len(steady), "elapsed_sec"))
    seen = {rel for rel, _, _, _ in rows}
    for rel, (n, mean) in summarize_logs(root, stage, contains).items():
        if rel not in seen:                     # elapsed_sec wins where it exists
            rows.append((rel, n, mean, "mm:ss"))
    return sorted(rows)

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--stage", choices=["pretrain", "train", "both"], default="both")
    ap.add_argument("--contains", default="", help="only runs whose path contains this substring")
    args = ap.parse_args()

    stages = ["pretrain", "train"] if args.stage == "both" else [args.stage]
    for stage in stages:
        rows = summarize(args.root, stage, args.contains)
        print(f"\n=== {stage}: mean epoch seconds (epoch 1 dropped) ===")
        if not rows:
            print("(no runs found)")
            continue
        print("run,epochs,mean_epoch_sec,source")
        for rel, n, mean, src in rows:
            print(f"{rel},{n},{mean:.1f},{src}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
