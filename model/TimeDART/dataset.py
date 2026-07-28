"""Sliding-window datasets for TimeDART pretraining and forecasting."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .utils import PROJECT_ROOT


def _split_dir(data_fine_dir: str, dataset_name: str, split: str) -> Path:
    return PROJECT_ROOT / data_fine_dir / dataset_name / split


def _load_split_array(data_fine_dir: str, dataset_name: str, split: str) -> np.ndarray:
    root = _split_dir(data_fine_dir, dataset_name, split)
    path = root / "data.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Split not found: {path}. Run data_preprocess/build_informer_windows.py first."
        )
    return np.load(path, mmap_mode="r")


def _select_features(data: np.ndarray, features: str, enc_in: int) -> np.ndarray:
    if features in {"M", "MS"}:
        return data[:, :enc_in]
    if features == "S":
        return data[:, -1:].astype(np.float32, copy=False)
    raise ValueError(f"Unsupported features mode: {features}")


def _read_valid_offset(data_fine_dir: str, dataset_name: str, split: str) -> int:
    meta_path = _split_dir(data_fine_dir, dataset_name, split) / "metadata.json"
    if not meta_path.exists():
        return 0
    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    return int(meta.get("valid_offset_in_file", 0))


class PretrainWindowDataset(Dataset):
    """Input windows for TimeDART's self-reconstruction pretraining."""

    def __init__(self, config, split: str):
        self.config = config
        self.split = "validation" if split == "val" else split
        data = _load_split_array(config.data_fine_dir, config.dataset_name, self.split)
        self.data = _select_features(data, config.features, config.enc_in)
        if self.data.ndim != 2:
            raise ValueError(f"Expected data.npy shape (rows, features), got {self.data.shape}")
        if self.data.shape[1] != config.enc_in:
            raise ValueError(f"Expected enc_in={config.enc_in}, got {self.data.shape[1]}")
        self.start_offset = max(_read_valid_offset(config.data_fine_dir, config.dataset_name, self.split), 0)
        usable_rows = len(self.data) - self.start_offset
        self.window_count = usable_rows - config.input_len + 1
        if self.window_count <= 0:
            raise ValueError(
                f"Not enough rows for pretrain windows: split={self.split}, "
                f"rows={len(self.data)}, input_len={config.input_len}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> torch.Tensor:
        begin = self.start_offset + index
        end = begin + self.config.input_len
        x = np.array(self.data[begin:end], dtype=np.float32, copy=True)
        return torch.from_numpy(x)


class ForecastWindowDataset(Dataset):
    """Sliding (input, target) windows aligned to the split boundary."""

    def __init__(self, config, split: str):
        self.config = config
        self.split = "validation" if split == "val" else split
        data = _load_split_array(config.data_fine_dir, config.dataset_name, self.split)
        self.data = _select_features(data, config.features, config.enc_in)
        valid_offset = _read_valid_offset(config.data_fine_dir, config.dataset_name, self.split)
        self.start_offset = max(valid_offset - config.input_len, 0)

        usable_rows = len(self.data) - self.start_offset
        self.window_count = usable_rows - config.input_len - config.pred_len + 1
        if self.window_count <= 0:
            raise ValueError(
                "Not enough rows for forecasting windows: "
                f"split={self.split}, rows={len(self.data)}, "
                f"start_offset={self.start_offset}, input_len={config.input_len}, "
                f"pred_len={config.pred_len}"
            )

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_begin = self.start_offset + index
        x_end = x_begin + self.config.input_len
        y_end = x_end + self.config.pred_len
        x = np.array(self.data[x_begin:x_end], dtype=np.float32, copy=True)
        y = np.array(self.data[x_end:y_end], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)
