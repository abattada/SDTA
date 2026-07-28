#!/usr/bin/env python3
"""Statistical significance tests for the paper's headline comparisons.

Protocol (Demsar 2006, "Statistical Comparisons of Classifiers over Multiple
Data Sets"): one paired observation per dataset, obtained by averaging the
canonical prediction lengths and then the 5 seeds -- exactly the cells the
paper's main table reports.

  * Omnibus over all 8 methods: Friedman test on the 12 (or 11) datasets,
    plus average ranks.
  * One-versus-all: two-sided Wilcoxon signed-rank of SDTA against each of the
    7 baselines (exact test at these sample sizes), with Holm correction over
    the 7 comparisons.
  * Supervision axis: two-sided Wilcoxon of W=1 against W=4 for SDTA, and of
    each baseline's own feasible axis at its two extreme values.

Nothing here re-runs an experiment; it reads the same
`outputs/test/**/results.json` files that `aggregate_results.py` reads, so the
per-dataset inputs are bit-identical to the numbers printed in the tables.

Usage (from the repository root, after the experiments have been run):
    python scripts/significance_test.py                 # all tests, MSE + MAE
    python scripts/significance_test.py --metric mse    # MSE only
    python scripts/significance_test.py --csv sig.csv   # also dump a CSV

Requires scipy (>=1.9) and numpy; both are in requirements.txt.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = [2021, 2022, 2023, 2024, 2025]
DATASETS = [
    "ETTh1", "ETTh2", "ETTm1", "ETTm2",
    "exchange_rate", "weather", "electricity", "traffic",
    "PEMS03", "PEMS04", "PEMS07", "PEMS08",
]
PEMS = {"PEMS03", "PEMS04", "PEMS07", "PEMS08"}
# exchange_rate is too short for the 1080-step horizon (paper: 11-ds aggregate)
EXTENDED_DATASETS = [d for d in DATASETS if d != "exchange_rate"]

# Model directory and run-name template per method; {s} is the seed.
# Identical to aggregate_results.py's registry (see docs/paper_code_map.md).
SDTA_TPL = "arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_W_{W}_S_{{s}}"
METHODS = {
    "SDTA":         ("SDTA", SDTA_TPL.format(W=4)),
    "TimeMAE-CI":   ("TimeMAE_CI", "arch_scan_Enc_2_S_{s}"),
    "TimeDART":     ("TimeDART", "arch_scan_Enc_2_Dec_1_S_{s}"),
    "TimeSiam":     ("TimeSiam", "arch_scan_Enc_2_Dec_1_S_{s}"),
    "SimMTM":       ("SimMTM", "arch_scan_Enc_2_S_{s}"),
    "PatchTST-SSL": ("PatchTST", "arch_scan_Enc_2_S_{s}"),
    "CPC":          ("CPC", "arch_scan_Enc_2_K_1_Neg_64_S_{s}"),
    "SimTS":        ("SimTS", "arch_scan_Enc_2_S_{s}"),
}
CONTROL = "SDTA"

# Supervision-axis pairs: (label, low-arm, high-arm) with arms as (dir, tpl).
AXIS_PAIRS = [
    ("SDTA  W=1 vs W=4",
     ("SDTA", SDTA_TPL.format(W=1)), ("SDTA", SDTA_TPL.format(W=4))),
    ("TimeSiam W=1 vs W=4",
     ("TimeSiam", "arch_scan_Enc_2_Dec_1_S_{s}"), ("TimeSiam_W", "w_scan_W_4_S_{s}")),
    ("CPC   K=1 vs K=4",
     ("CPC", "arch_scan_Enc_2_K_1_Neg_64_S_{s}"), ("CPC", "arch_scan_Enc_2_K_4_Neg_64_S_{s}")),
]


def canonical_pred_lens(dataset: str) -> list[str]:
    return ["12", "24", "48", "96"] if dataset in PEMS else ["96", "192", "336", "720"]


def extended_pred_len(dataset: str) -> str:
    return "144" if dataset in PEMS else "1080"


def dataset_value(root: Path, model_dir: str, run_tpl: str, dataset: str,
                  metric: str, extended: bool) -> float | None:
    """One paired observation: mean over seeds of the reported per-dataset cell.

    Standard: mean over the four canonical horizons, then over seeds.
    Extended: the single long horizon, then over seeds.
    Returns None when a seed is missing, so an incomplete arm is never silently
    compared against a complete one.
    """
    per_seed = []
    for s in SEEDS:
        run = run_tpl.format(s=s)
        base = root / "outputs" / "test" / model_dir / dataset / "il96" / f"pre_{run}"
        std_p, plx_p = base / run / "results.json", base / f"{run}_PLX" / "results.json"
        std = json.loads(std_p.read_text())["scores"] if std_p.is_file() else None
        plx = json.loads(plx_p.read_text())["scores"] if plx_p.is_file() else None
        if extended:
            key = extended_pred_len(dataset)
            src = std if std and key in std else plx
            if not (src and key in src):
                return None
            per_seed.append(src[key][metric])
        else:
            pls = canonical_pred_lens(dataset)
            if not (std and all(k in std for k in pls)):
                return None
            per_seed.append(sum(std[k][metric] for k in pls) / len(pls))
    if len(per_seed) != len(SEEDS):
        return None
    return sum(per_seed) / len(per_seed)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in the input order."""
    k = len(pvals)
    order = sorted(range(k), key=lambda i: pvals[i])
    adj, running = [0.0] * k, 0.0
    for rank, i in enumerate(order):
        running = min(1.0, max(running, (k - rank) * pvals[i]))
        adj[i] = running
    return adj


def wilcoxon_pair(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int, int]:
    """Two-sided exact Wilcoxon signed-rank of a vs b; also wins and effect size.

    Returns (statistic, p-value, #datasets where a < b, n).
    """
    res = stats.wilcoxon(a, b, alternative="two-sided", zero_method="wilcox")
    return float(res.statistic), float(res.pvalue), int((a < b).sum()), len(a)


def run_block(root: Path, metric: str, extended: bool, out_rows: list) -> None:
    datasets = EXTENDED_DATASETS if extended else DATASETS
    horizon = "extended (1080 / PEMS 144)" if extended else "standard (canonical 4)"
    title = f"{metric.upper()} -- {horizon}, {len(datasets)} datasets, 5-seed means"
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)

    table, missing = {}, []
    for name, (mdir, tpl) in METHODS.items():
        vals = [dataset_value(root, mdir, tpl, d, metric, extended) for d in datasets]
        if any(v is None for v in vals):
            missing += [f"{name}/{d}" for d, v in zip(datasets, vals) if v is None]
            continue
        table[name] = np.asarray(vals, dtype=float)
    if missing:
        print(f"  [skipped, incomplete runs] {', '.join(missing)}", file=sys.stderr)
    if CONTROL not in table or len(table) < 3:
        print("  not enough complete arms for a test", file=sys.stderr)
        return

    names = [n for n in METHODS if n in table]
    arr = np.stack([table[n] for n in names])                       # methods x datasets
    fr = stats.friedmanchisquare(*arr)
    ranks = np.apply_along_axis(stats.rankdata, 0, arr).mean(axis=1)
    print(f"\nFriedman omnibus over {len(names)} methods: "
          f"chi2 = {fr.statistic:.3f}, p = {fr.pvalue:.3g}")
    print("Average rank (1 = best):")
    for n, r in sorted(zip(names, ranks), key=lambda t: t[1]):
        print(f"   {n:<14s} {r:.2f}   mean = {table[n].mean():.4f}")

    ctrl = table[CONTROL]
    rows = []
    for n in names:
        if n == CONTROL:
            continue
        w, p, wins, nn = wilcoxon_pair(ctrl, table[n])
        rows.append([n, wins, nn, float(ctrl.mean() - table[n].mean()), w, p])
    for row, p_adj in zip(rows, holm([r[5] for r in rows])):
        row.append(p_adj)

    print(f"\nWilcoxon signed-rank, {CONTROL} vs each baseline "
          f"(two-sided exact, Holm-corrected over {len(rows)} comparisons):")
    print(f"   {'baseline':<14s} {'wins':>7s} {'Delta mean':>11s} {'W':>6s} "
          f"{'p':>9s} {'p_Holm':>9s}")
    for n, wins, nn, dmean, w, p, p_adj in rows:
        flag = "*" if p_adj < 0.05 else " "
        print(f"   {n:<14s} {wins:>3d}/{nn:<3d} {dmean:>+11.4f} {w:>6.1f} "
              f"{p:>9.4f} {p_adj:>9.4f} {flag}")
        out_rows.append({
            "metric": metric, "horizon": "extended" if extended else "standard",
            "comparison": f"{CONTROL} vs {n}", "n": nn, "wins": wins,
            "delta_mean": f"{dmean:.6f}", "W": f"{w:.1f}",
            "p_two_sided": f"{p:.6g}", "p_holm": f"{p_adj:.6g}",
        })


def run_axis_block(root: Path, metric: str, extended: bool, out_rows: list) -> None:
    datasets = EXTENDED_DATASETS if extended else DATASETS
    print(f"\nSupervision axis, {metric.upper()} "
          f"({'extended' if extended else 'standard'} horizons), "
          f"two-sided Wilcoxon over {len(datasets)} datasets:")
    print(f"   {'axis':<22s} {'improves':>9s} {'Delta mean':>11s} {'W':>6s} {'p':>9s}")
    for label, (ld, lt), (hd, ht) in AXIS_PAIRS:
        lo = [dataset_value(root, ld, lt, d, metric, extended) for d in datasets]
        hi = [dataset_value(root, hd, ht, d, metric, extended) for d in datasets]
        if any(v is None for v in lo) or any(v is None for v in hi):
            print(f"   {label:<22s} {'--':>9s}   (incomplete runs)")
            continue
        lo, hi = np.asarray(lo), np.asarray(hi)
        w, p, wins, nn = wilcoxon_pair(hi, lo)     # does the high arm beat the low arm?
        flag = "*" if p < 0.05 else " "
        print(f"   {label:<22s} {wins:>3d}/{nn:<3d} {hi.mean() - lo.mean():>+11.4f} "
              f"{w:>6.1f} {p:>9.4f} {flag}")
        out_rows.append({
            "metric": metric, "horizon": "extended" if extended else "standard",
            "comparison": label, "n": nn, "wins": wins,
            "delta_mean": f"{hi.mean() - lo.mean():.6f}", "W": f"{w:.1f}",
            "p_two_sided": f"{p:.6g}", "p_holm": "",
        })


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                    help="repository root containing outputs/ (default: this repo)")
    ap.add_argument("--metric", choices=["mse", "mae", "both"], default="both")
    ap.add_argument("--skip-extended", action="store_true",
                    help="only test the standard horizons")
    ap.add_argument("--csv", type=Path, default=None, help="write all results to CSV")
    ap.add_argument("--dir-alias", action="append", default=None, metavar="NAME=DIR",
                    help="rename a method's outputs/test directory, e.g. SDTA=SDTA_v10; "
                         "only needed when outputs/ came from a differently-named tree")
    args = ap.parse_args()

    for spec in args.dir_alias or []:
        name, _, newdir = spec.partition("=")
        if name in METHODS:
            METHODS[name] = (newdir, METHODS[name][1])
        for i, (label, lo, hi) in enumerate(AXIS_PAIRS):
            AXIS_PAIRS[i] = (label,
                             (newdir, lo[1]) if lo[0] == name else lo,
                             (newdir, hi[1]) if hi[0] == name else hi)

    metrics = ["mse", "mae"] if args.metric == "both" else [args.metric]
    horizons = [False] if args.skip_extended else [False, True]
    rows: list[dict] = []
    for extended in horizons:
        for metric in metrics:
            run_block(args.root, metric, extended, rows)
        for metric in metrics:
            run_axis_block(args.root, metric, extended, rows)

    print("\n* = significant at alpha = 0.05.\n"
          "Wilcoxon pairs one dataset per observation, so n is the dataset count, "
          "not the number of runs;\neach observation is itself a 5-seed mean.")
    if args.csv:
        with args.csv.open("w", newline="") as fh:
            wtr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            wtr.writeheader()
            wtr.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
