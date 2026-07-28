#!/usr/bin/env python3
"""Summarize per-epoch wall-clock time from training logs.

Implements the timing protocol behind the paper's cost analysis: run the
`entry/batch_configs/time/*.json` configs (6 epochs, single seed 97, one GPU,
serial execution, num_workers=1), then average the per-epoch `elapsed_sec`
of epochs 2..N — epoch 1 is dropped as CUDA/cuDNN warm-up.

Reads every `epoch_losses.jsonl` (pretrain stage) and `epoch_metrics.jsonl`
(train stage) under outputs/pretrain and outputs/train.

Known limitation: the PatchTST baseline logs epoch times only as mm:ss strings
inside its batch logs (outputs/batch/time_PatchTST/*/logs/), not as an
`elapsed_sec` field; parse those logs manually for its wall-clock entry.

Usage (from the repository root):
    python scripts/epoch_time.py [--root PATH] [--stage pretrain|train|both]
    python scripts/epoch_time.py --contains time_  # restrict by run-name substring
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

def summarize(root: Path, stage: str, contains: str) -> list[tuple[str, int, float]]:
    fname = {"pretrain": "epoch_losses.jsonl", "train": "epoch_metrics.jsonl"}[stage]
    rows = []
    base = root / "outputs" / stage
    if not base.is_dir():
        return rows
    for jsonl in sorted(base.rglob(fname)):
        rel = jsonl.parent.relative_to(base)
        if contains and contains not in str(rel):
            continue
        secs = epoch_seconds(jsonl)
        if len(secs) < 2:
            continue
        steady = secs[1:]  # drop epoch 1 (warm-up)
        rows.append((str(rel), len(secs), sum(steady) / len(steady)))
    return rows

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
        print("run,epochs,mean_epoch_sec")
        for rel, n, mean in rows:
            print(f"{rel},{n},{mean:.1f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
