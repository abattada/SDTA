"""One-off migration: back-fill the `done` sentinel into already-complete runs.

Before the `done` convention, completion was detected from family-specific
legacy markers (each model family wrote its own final artefact, and resume had
to know all of them). This script scans existing
`outputs/{pretrain,train,test}/**` and writes an empty `done` file into every run
dir that the legacy markers say is complete, so resume keeps skipping them after
the switch to `done`.

Usage:
    python entry/batch_src/backfill_done.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs"
DONE_MARKER = "done"  # keep in sync with model/_test_io.py and resume.py


def _count_nonblank_lines(path: Path) -> int:
    try:
        with path.open("r") as fh:
            return sum(1 for line in fh if line.strip())
    except OSError:
        return 0


def _pretrain_complete(d: Path) -> bool:
    if (d / "losses.csv").exists():  # PatchTST family
        return True
    if (d / "training_summary.json").exists():  # SimMTM / TimeMAE / ST_MTM / baseline
        return True
    jsonl = d / "epoch_losses.jsonl"  # SDTA / TimeSiam / TimeDART: line-count vs train_epochs
    cfg = d / "config.json"
    if jsonl.exists() and cfg.exists():
        try:
            epochs = int(json.loads(cfg.read_text()).get("train_epochs", 0))
        except (json.JSONDecodeError, OSError, TypeError, ValueError):
            return False
        return epochs > 0 and _count_nonblank_lines(jsonl) >= epochs
    return False


def _train_complete(d: Path) -> bool:
    return (d / "training_summary.json").exists() or (d / "losses.csv").exists()


def _test_complete(d: Path) -> bool:
    results = d / "results.json"
    if not results.exists():
        return False
    try:
        scores = json.loads(results.read_text()).get("scores")
    except (json.JSONDecodeError, OSError):
        return False
    return isinstance(scores, dict) and len(scores) > 0


def _scan() -> list[Path]:
    """Return run dirs that are legacy-complete but missing the `done` sentinel."""
    todo: list[Path] = []
    for d in OUTPUTS.glob("pretrain/*/*/il*/*"):
        if d.is_dir() and _pretrain_complete(d) and not (d / DONE_MARKER).exists():
            todo.append(d)
    for d in OUTPUTS.glob("train/*/*/il*/pre_*/*/pl*"):
        if d.is_dir() and _train_complete(d) and not (d / DONE_MARKER).exists():
            todo.append(d)
    for d in OUTPUTS.glob("test/*/*/il*/pre_*/*"):
        if d.is_dir() and _test_complete(d) and not (d / DONE_MARKER).exists():
            todo.append(d)
    return todo


def main() -> None:
    parser = argparse.ArgumentParser(description="Back-fill `done` sentinel into complete runs.")
    parser.add_argument("--dry-run", action="store_true", help="List dirs without writing `done`.")
    args = parser.parse_args()

    if not OUTPUTS.exists():
        print(f"No outputs dir at {OUTPUTS}; nothing to do.")
        return
    todo = _scan()
    for d in todo:
        rel = d.relative_to(PROJECT_ROOT)
        if args.dry_run:
            print(f"[would-write] {rel}/{DONE_MARKER}")
        else:
            (d / DONE_MARKER).write_text("")
            print(f"[done] {rel}/{DONE_MARKER}")
    verb = "would back-fill" if args.dry_run else "back-filled"
    print(f"{verb} {len(todo)} run dir(s).")


if __name__ == "__main__":
    main()
