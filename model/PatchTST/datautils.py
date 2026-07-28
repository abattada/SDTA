"""PatchTST data plumbing — now backed by data/fine, mirroring SDTA / TimeSiam.

The old CSV-based path (Dataset_ETT_hour / minute / Custom in pred_dataset.py)
is no longer used by `get_dls`. We keep the pred_dataset module importable so
external callers don't break, but every dataset goes through `DataFineDataset`.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from .src.data.datamodule import DataLoaders
from .datautils_fine import DataFineDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ─── Registry ────────────────────────────────────────────────────────────────
# CLI dataset alias  →  folder name under data/fine
DATASET_SPECS: dict[str, str] = {
    "ettm1": "ETTm1",
    "ettm2": "ETTm2",
    "etth1": "ETTh1",
    "etth2": "ETTh2",
    "electricity": "electricity",
    "traffic": "traffic",
    "weather": "weather",
    "exchange": "exchange_rate",
    "exchange_rate": "exchange_rate",
    "pems03": "PEMS03",
    "pems04": "PEMS04",
    "pems07": "PEMS07",
    "pems08": "PEMS08",
}

DSETS = list(DATASET_SPECS.keys()) + ["ett_all"]
ETT_ALL_DSETS = ["etth1", "etth2", "ettm1", "ettm2"]


def _canonical_dset(dset: str) -> str:
    value = str(dset).lower()
    aliases = {
        "ettall": "ett_all",
        "ett-all": "ett_all",
        "all_ett": "ett_all",
    }
    return aliases.get(value, value)


def parse_dsets(value) -> list[str]:
    if value is None:
        return ETT_ALL_DSETS.copy()
    if isinstance(value, str):
        raw = [part.strip() for part in value.split(",") if part.strip()]
    else:
        raw = list(value)
    dsets: list[str] = []
    for item in raw:
        dset = _canonical_dset(item)
        if dset == "ett_all":
            dsets.extend(ETT_ALL_DSETS)
        else:
            dsets.append(dset)
    unknown = [dset for dset in dsets if dset not in DATASET_SPECS]
    if unknown:
        raise ValueError(
            f"Unrecognized mixed dset(s) {unknown}. Options include: {sorted(DATASET_SPECS)}"
        )
    return dsets


# ─── Dataloader builders ─────────────────────────────────────────────────────

def _dataset_kwargs(params, dset: str) -> dict:
    data_root = PROJECT_ROOT / getattr(params, "data_root", "data/fine")
    folder = DATASET_SPECS[dset]
    return {
        "root_path": str(data_root),
        "data_path": folder,
        "features": params.features,
        "scale": True,                # ignored — data/fine is already scaled
        "size": [params.context_points, 0, params.target_points],
        "use_time_features": params.use_time_features,
    }


def _make_dataset(params, dset: str, split: str) -> DataFineDataset:
    return DataFineDataset(**_dataset_kwargs(params, dset), split=split)


def _finalize_dls(dls, params):
    dls.vars, dls.len = dls.train.dataset[0][0].shape[1], params.context_points
    dls.c = dls.train.dataset[0][1].shape[0]
    return dls


def _single_dls(params, dset: str):
    dls = DataLoaders(
        datasetCls=DataFineDataset,
        dataset_kwargs=_dataset_kwargs(params, dset),
        batch_size=params.batch_size,
        workers=params.num_workers,
    )
    return _finalize_dls(dls, params)


def _mixed_loader(datasets, batch_size, workers, split, sampling):
    concat = ConcatDataset(datasets)
    shuffle = split == "train"
    sampler = None
    if shuffle and sampling == "balanced":
        weights = []
        for dataset in datasets:
            weights.extend([1.0 / len(dataset)] * len(dataset))
        sampler = WeightedRandomSampler(
            torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(concat),
            replacement=True,
        )
        shuffle = False
    return DataLoader(
        concat,
        shuffle=shuffle,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=workers,
    )


def get_mixed_dls(params):
    if getattr(params, "use_time_features", False):
        raise ValueError(
            "Mixed ETT_all PatchTST currently requires --use_time_features 0 "
            "because the mixed sampler does not realign multi-source time features."
        )
    sampling = getattr(params, "sampling", "balanced")
    if sampling not in {"balanced", "proportional"}:
        raise ValueError("--sampling must be either 'balanced' or 'proportional'")
    dsets = parse_dsets(getattr(params, "datasets", None))
    train_sets = [_make_dataset(params, dset, "train") for dset in dsets]
    valid_sets = [_make_dataset(params, dset, "val") for dset in dsets]
    test_sets = [_make_dataset(params, dset, "test") for dset in dsets]
    dls = SimpleNamespace(
        train=_mixed_loader(train_sets, params.batch_size, params.num_workers, "train", sampling),
        valid=_mixed_loader(valid_sets, params.batch_size, params.num_workers, "val", "proportional"),
        test=_mixed_loader(test_sets, params.batch_size, params.num_workers, "test", "proportional"),
        mixed_dsets=dsets,
        sampling=sampling,
    )
    dls.vars = train_sets[0][0][0].shape[1]
    dls.len = params.context_points
    dls.c = train_sets[0][0][1].shape[0]
    return dls


def get_dls(params):
    params.dset = _canonical_dset(params.dset)
    if params.dset not in DSETS:
        raise ValueError(f"Unrecognized dset ('{params.dset}'). Options: {DSETS}")
    if not hasattr(params, "use_time_features"):
        params.use_time_features = False
    if params.dset == "ett_all":
        return get_mixed_dls(params)
    return _single_dls(params, params.dset)


if __name__ == "__main__":
    class Params:
        dset = "etth2"
        data_root = "data/fine"
        context_points = 96
        target_points = 96
        batch_size = 64
        num_workers = 0
        features = "M"
        use_time_features = False
    dls = get_dls(Params())
    for i, batch in enumerate(dls.valid):
        print(i, len(batch), batch[0].shape, batch[1].shape)
        if i >= 2:
            break
