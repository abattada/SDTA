"""Build Informer-style scaled splits for downstream pretrain/train/test.

Pipeline per dataset:
    1. Read raw file (csv with 'date' column dropped, or npz for PEMS).
    2. Compute split borders by the dataset's convention:
       - ETT month-based 12:4:4 (hour or 15-min ticks)
       - electricity / traffic / weather / exchange_rate: 7:1:2 ratio
       - PEMS03/04/07/08: 6:2:2 ratio (TimeDART convention)
    3. Optionally ffill/bfill NaNs (PEMS).
    4. Fit StandardScaler on the train segment, transform the full series.
    5. For val/test, prepend the last `max_overlap` post-scaler rows of the previous
       split so any input window with input_len <= max_overlap can use prior context
       while still placing its target inside the current split.
    6. Save per-split data.npy plus a global scaler.npz and borders.json.

Window construction (RevIN, history stats) is left to runtime — this script does
not bake window_len, patch_len, stride, or any model hyperparameter into the data.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ─── Dataset registry ────────────────────────────────────────────────────────
# Each entry: (raw_relative_path, format, border_rule)
#   format in {"csv", "pems_npz"}
#   border_rule:
#       ("ett_hour",)
#       ("ett_minute",)
#       ("ratio", train_ratio, val_ratio, test_ratio)
#         (test_ratio is computed from 1 - train - val so val_ratio determines it)

DATASETS: dict[str, tuple[str, str, tuple]] = {
    # ETT (Informer 12:4:4 month)
    "ETTh1":          ("ETTh1.csv", "csv", ("ett_hour",)),
    "ETTh2":          ("ETTh2.csv", "csv", ("ett_hour",)),
    "ETTm1":          ("ETTm1.csv", "csv", ("ett_minute",)),
    "ETTm2":          ("ETTm2.csv", "csv", ("ett_minute",)),
    # tslib (PatchTST / Informer 7:1:2)
    "electricity":    ("timeseries/tslib/electricity/electricity.csv",   "csv", ("ratio", 0.7, 0.1)),
    "traffic":        ("timeseries/tslib/traffic/traffic.csv",           "csv", ("ratio", 0.7, 0.1)),
    "weather":        ("timeseries/tslib/weather/weather.csv",           "csv", ("ratio", 0.7, 0.1)),
    "exchange_rate":  ("timeseries/tslib/exchange_rate/exchange_rate.csv","csv", ("ratio", 0.7, 0.1)),
    # PEMS (TimeDART 6:2:2)
    "PEMS03": ("timeseries/pems/PEMS03/PEMS03.npz", "pems_npz", ("ratio", 0.6, 0.2)),
    "PEMS04": ("timeseries/pems/PEMS04/PEMS04.npz", "pems_npz", ("ratio", 0.6, 0.2)),
    "PEMS07": ("timeseries/pems/PEMS07/PEMS07.npz", "pems_npz", ("ratio", 0.6, 0.2)),
    "PEMS08": ("timeseries/pems/PEMS08/PEMS08.npz", "pems_npz", ("ratio", 0.6, 0.2)),
}


# ─── Loaders ────────────────────────────────────────────────────────────────

def _read_csv_drop_date(path: Path) -> tuple[list[str], np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Raw csv not found: {path}")
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        header = next(reader, None)
        if header is None:
            raise ValueError(f"Empty csv: {path}")
        date_indices = [i for i, name in enumerate(header) if name == "date"]
        keep_indices = [i for i, name in enumerate(header) if name != "date"]
        if not date_indices:
            raise ValueError(f"Expected a 'date' column in {path}")
        feature_names = [header[i] for i in keep_indices]
        for row in reader:
            if not row:
                continue
            rows.append([float(row[i]) for i in keep_indices])
    if not rows:
        raise ValueError(f"No numeric rows in csv: {path}")
    return feature_names, np.asarray(rows, dtype=np.float64)


def _read_pems_npz(path: Path) -> tuple[list[str], np.ndarray]:
    """Load PEMS-style npz. The convention follows TimeDART: data[:, :, 0] → (T, N)."""
    if not path.exists():
        raise FileNotFoundError(f"PEMS npz not found: {path}")
    payload = np.load(path, allow_pickle=True)
    if "data" not in payload.files:
        raise ValueError(f"PEMS npz {path} missing 'data' key")
    arr = payload["data"]
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    elif arr.ndim != 2:
        raise ValueError(f"PEMS npz {path} has unexpected ndim={arr.ndim}, shape={arr.shape}")
    # ffill then bfill on NaNs (TimeDART convention).
    arr = arr.astype(np.float64, copy=True)
    mask = np.isnan(arr)
    if mask.any():
        # forward fill column-wise
        for col in range(arr.shape[1]):
            last = None
            for row in range(arr.shape[0]):
                if np.isnan(arr[row, col]):
                    if last is not None:
                        arr[row, col] = last
                else:
                    last = arr[row, col]
            # backward fill remaining (head NaNs)
            last = None
            for row in range(arr.shape[0] - 1, -1, -1):
                if np.isnan(arr[row, col]):
                    if last is not None:
                        arr[row, col] = last
                else:
                    last = arr[row, col]
    n_nodes = arr.shape[1]
    feature_names = [f"node_{i}" for i in range(n_nodes)]
    return feature_names, arr


def _load_raw(dataset: str, raw_root: Path) -> tuple[list[str], np.ndarray]:
    relpath, fmt, _ = DATASETS[dataset]
    path = raw_root / relpath
    if fmt == "csv":
        return _read_csv_drop_date(path)
    if fmt == "pems_npz":
        return _read_pems_npz(path)
    raise ValueError(f"Unknown format '{fmt}' for {dataset}")


# ─── Borders ────────────────────────────────────────────────────────────────

_ETT_HOUR = {
    "train_end": 12 * 30 * 24,
    "val_end":   12 * 30 * 24 + 4 * 30 * 24,
    "test_end":  12 * 30 * 24 + 8 * 30 * 24,
}
_ETT_MINUTE = {
    "train_end": 12 * 30 * 24 * 4,
    "val_end":   (12 * 30 * 24 + 4 * 30 * 24) * 4,
    "test_end":  (12 * 30 * 24 + 8 * 30 * 24) * 4,
}


def _borders_for(dataset: str, raw_rows: int) -> tuple[dict[str, int], str]:
    """Return (borders_dict, convention_string)."""
    rule = DATASETS[dataset][2]
    kind = rule[0]
    if kind == "ett_hour":
        return _ETT_HOUR, "Informer ETT 12:4:4 month split, hourly (30*24 rows/month)."
    if kind == "ett_minute":
        return _ETT_MINUTE, "Informer ETT 12:4:4 month split, 15-min (30*24*4 rows/month)."
    if kind == "ratio":
        train_ratio, val_ratio = rule[1], rule[2]
        n_train = int(raw_rows * train_ratio)
        n_val = int(raw_rows * val_ratio)
        n_test = raw_rows - n_train - n_val
        borders = {
            "train_end": n_train,
            "val_end":   n_train + n_val,
            "test_end":  n_train + n_val + n_test,
        }
        conv = (
            f"Ratio split {train_ratio}:{val_ratio}:{round(1 - train_ratio - val_ratio, 4)} "
            f"(raw_rows={raw_rows})."
        )
        return borders, conv
    raise ValueError(f"Unknown border rule kind '{kind}' for {dataset}")


def _fit_standard_scaler(train_segment: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_segment.mean(axis=0).astype(np.float32)
    std = train_segment.std(axis=0, ddof=0).astype(np.float32)
    std = np.maximum(std, np.float32(1e-8))
    return mean, std


# ─── Main build ─────────────────────────────────────────────────────────────

def build_one(dataset: str, raw_root: Path, fine_root: Path, max_overlap: int) -> None:
    if max_overlap < 0:
        raise ValueError(f"max_overlap must be >= 0, got {max_overlap}")
    if dataset not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Supported: {sorted(DATASETS)}"
        )

    feature_names, raw = _load_raw(dataset, raw_root)
    borders, convention = _borders_for(dataset, raw_rows=raw.shape[0])
    train_end = borders["train_end"]
    val_end = borders["val_end"]
    test_end = borders["test_end"]

    if test_end > raw.shape[0]:
        raise ValueError(
            f"{dataset}: raw rows={raw.shape[0]} < required test_end={test_end}. "
            "Check the file or border convention."
        )

    train_segment = raw[:train_end]
    scaler_mean, scaler_std = _fit_standard_scaler(train_segment)
    scaled = ((raw - scaler_mean) / scaler_std).astype(np.float32)

    out_root = fine_root / dataset
    out_root.mkdir(parents=True, exist_ok=True)

    np.savez(
        out_root / "scaler.npz",
        mean=scaler_mean,
        std=scaler_std,
        feature_names=np.asarray(feature_names),
    )

    split_specs = {
        "train":      (0,                                  train_end, 0),
        "validation": (max(train_end - max_overlap, 0),    val_end,   train_end),
        "test":       (max(val_end - max_overlap, 0),      test_end,  val_end),
    }

    splits_meta: dict[str, dict[str, int]] = {}
    for split_name, (file_start, file_end, valid_start) in split_specs.items():
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        segment = scaled[file_start:file_end]
        np.save(split_dir / "data.npy", segment)

        valid_offset = valid_start - file_start
        meta = {
            "dataset": dataset,
            "split": split_name,
            "file_start": int(file_start),
            "file_end": int(file_end),
            "valid_start": int(valid_start),
            "valid_offset_in_file": int(valid_offset),
            "rows": int(segment.shape[0]),
            "num_features": int(segment.shape[1]),
            "max_overlap": int(max_overlap),
            "feature_names": feature_names,
            "split_convention": convention,
            "normalization": (
                "Post-scaler raw values. The split csv is fit-on-train transform-all "
                "with a global StandardScaler (see scaler.npz). For val/test, the first "
                "`valid_offset_in_file` rows are prior-split context appended to enable "
                "Informer-style overlap. RevIN (per-window mean/std) is computed at "
                "runtime by the dataloader."
            ),
            "windowing": (
                "Window length, patch length, and stride are NOT baked in. The dataset "
                "object should construct sliding windows of input_len (and pred_len for "
                "forecasting) at __getitem__ time."
            ),
            "first_valid_window_start": (
                "For forecasting, the first window whose target lies entirely inside the "
                "split is at index `valid_offset_in_file - input_len` (clamped to >= 0). "
                "Pretrain may use any window starting at index >= 0."
            ),
        }
        (split_dir / "metadata.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )
        splits_meta[split_name] = {
            "file_start": int(file_start),
            "file_end": int(file_end),
            "valid_start": int(valid_start),
            "valid_offset_in_file": int(valid_offset),
            "rows": int(segment.shape[0]),
        }
        print(
            f"{dataset}/{split_name}: "
            f"file=[{file_start}, {file_end}) rows={segment.shape[0]} "
            f"valid_offset={valid_offset}"
        )

    borders_meta = {
        "dataset": dataset,
        "convention": convention,
        "raw_rows": int(raw.shape[0]),
        "num_features": int(raw.shape[1]),
        "train_end": int(train_end),
        "val_end": int(val_end),
        "test_end": int(test_end),
        "discarded_tail_rows": int(raw.shape[0] - test_end),
        "max_overlap": int(max_overlap),
        "scaler": {
            "fit_segment": "raw[0:train_end]",
            "feature_names": feature_names,
            "mean_shape": list(scaler_mean.shape),
            "std_shape": list(scaler_std.shape),
        },
        "splits": splits_meta,
    }
    (out_root / "borders.json").write_text(
        json.dumps(borders_meta, indent=2), encoding="utf-8"
    )
    print(f"{dataset}: scaler + borders written to {out_root}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build Informer-style fit-train-transform-all splits in data/fine.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help=(
            "Dataset name (case-sensitive). Use 'all' to build every known dataset. "
            f"Supported: {sorted(DATASETS)}"
        ),
    )
    parser.add_argument("--raw_root", default="data/raw")
    parser.add_argument("--fine_root", default="data/fine")
    parser.add_argument(
        "--max_overlap",
        type=int,
        default=336,
        help=(
            "Number of post-scaler rows from the previous split prepended to "
            "validation and test files. Set this to the largest input_len you "
            "intend to run; smaller input_len at runtime is supported."
        ),
    )
    args = parser.parse_args()

    raw_root = PROJECT_ROOT / args.raw_root
    fine_root = PROJECT_ROOT / args.fine_root

    targets = sorted(DATASETS) if args.dataset == "all" else [args.dataset]
    for dataset in targets:
        print(f"\n=== {dataset} ===")
        build_one(dataset, raw_root=raw_root, fine_root=fine_root, max_overlap=args.max_overlap)


if __name__ == "__main__":
    main()
