"""Sliding-window datasets: Siamese pretrain pairs + standard forecast pairs."""
from __future__ import annotations

import json
import random
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
            f"Pretrain/forecast split not found: {path}. "
            "Run data_preprocess/build_informer_windows.py first."
        )
    return np.load(path, mmap_mode="r")


def lineage_segment(range_span: int, num_tokens: int, offset: int) -> int:
    """Bucket `offset` (within `range_span`) into one of `num_tokens` segments."""
    if num_tokens <= 1 or range_span <= 0:
        return 0
    segment_length = range_span / num_tokens
    seg = int(offset // segment_length)
    return min(max(seg, 0), num_tokens - 1)


class SiamesePretrainDataset(Dataset):
    """Multi-window TimeSiam: identical to official TimeSiam's sampler, just
    drawing W currents per anchor instead of 1.

    For each anchor (past), each of the `current_views` (W) currents is sampled
    EXACTLY as TimeSiam does — `r_begin = randint(s_begin, r_limit)` where
    `r_limit = min(s_begin + sampling_range*input_len, max_r_begin)` (continuous
    timesteps, NOT patch units), with the lineage segment bucketed over the
    actual span `[s_begin, r_limit]` via `lineage_segment`. With `current_views=1`
    this is bit-exact to official TimeSiam (one random draw per anchor); W>1 just
    draws W independent currents. Nothing else about TimeSiam's sampling changes."""

    def __init__(self, config, split: str):
        self.config = config
        self.split = "validation" if split == "val" else split
        self.data = _load_split_array(config.data_fine_dir, config.dataset_name, self.split)
        if self.data.ndim != 2:
            raise ValueError(
                f"Expected data.npy of shape (rows, num_features), got {self.data.shape}"
            )
        if self.data.shape[1] != config.enc_in:
            raise ValueError(
                f"Expected enc_in={config.enc_in} features, got {self.data.shape[1]}"
            )
        if config.current_views < 1:
            raise ValueError(f"current_views must be >= 1, got {config.current_views}")
        if self.data.shape[0] < config.input_len + 1:
            raise ValueError(
                f"Split {self.split} has {self.data.shape[0]} rows, "
                f"need at least input_len={config.input_len} + 1"
            )

    def __len__(self) -> int:
        return self.data.shape[0] - self.config.input_len + 1

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = self.config.input_len
        s_begin = index
        s_end = s_begin + seq_len

        max_r_begin = len(self.data) - seq_len
        r_limit = min(s_begin + self.config.sampling_range * seq_len, max_r_begin)
        r_limit = max(r_limit, s_begin)
        span = r_limit - s_begin

        past = np.array(self.data[s_begin:s_end], dtype=np.float32, copy=True)
        currents = []
        segments = []
        for _ in range(self.config.current_views):
            r_begin = random.randint(s_begin, r_limit)          # TimeSiam-identical draw
            currents.append(np.array(self.data[r_begin:r_begin + seq_len], dtype=np.float32, copy=True))
            segments.append(lineage_segment(span, self.config.lineage_tokens, r_begin - s_begin))
        currents = np.stack(currents, axis=0)                   # [W, L, C]
        segments = np.array(segments, dtype=np.int64)           # [W]
        return (
            torch.from_numpy(past),
            torch.from_numpy(currents),
            torch.from_numpy(segments),
        )


class ForecastWindowDataset(Dataset):
    """Sliding (input, target) windows aligned to the split boundary."""

    def __init__(self, config, split: str):
        self.config = config
        self.split = "validation" if split == "val" else split
        self.data = _load_split_array(config.data_fine_dir, config.dataset_name, self.split)
        self.data = self._select_features(self.data)
        valid_offset = self._read_valid_offset(self.split)
        self.start_offset = max(valid_offset - config.input_len, 0)

        usable_rows = len(self.data) - self.start_offset
        self.window_count = usable_rows - config.input_len - config.pred_len + 1
        if self.window_count <= 0:
            raise ValueError(
                "Not enough rows for forecasting windows: "
                f"split={self.split}, rows={len(self.data)}, "
                f"start_offset={self.start_offset}, "
                f"input_len={config.input_len}, pred_len={config.pred_len}"
            )

    def _read_valid_offset(self, split: str) -> int:
        meta_path = _split_dir(self.config.data_fine_dir, self.config.dataset_name, split) / "metadata.json"
        if not meta_path.exists():
            return 0
        with meta_path.open("r", encoding="utf-8") as handle:
            meta = json.load(handle)
        return int(meta.get("valid_offset_in_file", 0))

    def _select_features(self, data: np.ndarray) -> np.ndarray:
        if self.config.features in {"M", "MS"}:
            return data[:, : self.config.enc_in]
        if self.config.features == "S":
            return data[:, -1:].astype(np.float32, copy=False)
        raise ValueError(f"Unsupported features mode: {self.config.features}")

    def __len__(self) -> int:
        return self.window_count

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_begin = self.start_offset + index
        x_end = x_begin + self.config.input_len
        y_end = x_end + self.config.pred_len
        x = np.array(self.data[x_begin:x_end], dtype=np.float32, copy=True)
        y = np.array(self.data[x_end:y_end], dtype=np.float32, copy=True)
        return torch.from_numpy(x), torch.from_numpy(y)
