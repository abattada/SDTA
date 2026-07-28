"""data/fine-backed dataset for PatchTST, replacing the CSV pipeline.

Reads pre-scaled splits written by `data_preprocess/build_informer_windows.py`:

    data/fine/{dataset}/{split}/data.npy
    data/fine/{dataset}/{split}/metadata.json   (for valid_offset_in_file)

Returns (x, y) tensors compatible with PatchTST's Learner / callbacks.

This dataset mirrors `model/SDTA/dataset.py:ForecastWindowDataset` so PatchTST,
SDTA, and TimeSiam share a single data pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


_SPLIT_ALIAS = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}


class DataFineDataset(Dataset):
    """Sliding (x, y) windows from a pre-scaled data/fine split.

    Args mirror PatchTST's existing Dataset_* signature so DataLoaders can call
    us with `split=...`. Many fields are accepted for API compatibility but
    ignored because the data is already scaled and time features are not
    materialised at preprocessing time.
    """

    def __init__(
        self,
        root_path: str,
        data_path: str,
        features: str = "M",
        target: str = "OT",
        scale: bool = True,
        size: list[int] | None = None,
        timeenc: int = 0,
        freq: str = "h",
        use_time_features: bool = False,
        seasonal_patterns=None,
        split: str = "train",
    ):
        if size is None:
            size = [336, 0, 96]
        self.seq_len = int(size[0])
        self.pred_len = int(size[2])
        self.features = features
        self.use_time_features = bool(use_time_features)

        split_name = _SPLIT_ALIAS.get(split, split)
        split_dir = Path(root_path) / data_path / split_name
        npy_path = split_dir / "data.npy"
        if not npy_path.exists():
            raise FileNotFoundError(
                f"data/fine split not found: {npy_path}. "
                "Run data_preprocess/build_informer_windows.py first."
            )
        data = np.load(npy_path, mmap_mode="r")  # (rows, num_features)

        # Read the overlap offset from metadata so the first window's target
        # lies inside the actual split (mirrors SDTA's ForecastWindowDataset).
        meta_path = split_dir / "metadata.json"
        valid_offset = 0
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            valid_offset = int(meta.get("valid_offset_in_file", 0))
        self.start_offset = max(valid_offset - self.seq_len, 0)

        # Feature selection. M / MS both keep all input channels and let the
        # model / metric handle the target column. S keeps only the last column
        # (target) — matches Informer / PatchTST convention.
        if self.features == "S":
            data = data[:, -1:]
        self.data = np.asarray(data, dtype=np.float32)

        usable_rows = len(self.data) - self.start_offset
        self.window_count = usable_rows - self.seq_len - self.pred_len + 1
        if self.window_count <= 0:
            raise ValueError(
                f"Not enough rows for windows: dataset={data_path} split={split_name} "
                f"rows={len(self.data)} start_offset={self.start_offset} "
                f"seq_len={self.seq_len} pred_len={self.pred_len}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int):
        x_begin = self.start_offset + index
        x_end = x_begin + self.seq_len
        y_end = x_end + self.pred_len
        x = np.array(self.data[x_begin:x_end], dtype=np.float32, copy=True)
        y = np.array(self.data[x_end:y_end], dtype=np.float32, copy=True)
        if self.use_time_features:
            # data/fine has no time features; return zero placeholders so the
            # downstream code path that expects four tensors still works.
            x_mark = np.zeros((self.seq_len, 1), dtype=np.float32)
            y_mark = np.zeros((self.pred_len, 1), dtype=np.float32)
            return (
                torch.from_numpy(x),
                torch.from_numpy(y),
                torch.from_numpy(x_mark),
                torch.from_numpy(y_mark),
            )
        return torch.from_numpy(x), torch.from_numpy(y)
