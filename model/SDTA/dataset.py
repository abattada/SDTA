"""SDTA datasets (paper: Method — Pretext: Shifted-Window Diffusion).

SiamesePretrainDataset yields one clean anchor window plus W target windows at
patch-aligned shifts s sampled with replacement from [1, s_max]
(sampling_min/sampling_max, in patch units; repeats are possible); with
forced_first_shift_one the first target is pinned to s = 1.
ForecastWindowDataset yields the standard stride-1 input/target windows for
fine-tuning and evaluation, aligned via metadata.json valid_offset_in_file so
every prediction target stays inside its split."""
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


class SiamesePretrainDataset(Dataset):
    """For each anchor, sample W target windows at patch-aligned shifts.

    Shift semantics (the paper's shifted distance s):
      * The shift s is counted in PATCHES, not raw timesteps.
        `sampling_min` and `sampling_max` are patch-unit bounds,
        s ∈ [sampling_min, sampling_max].
        The target's raw start is anchor_begin + s * patch_len, so target_patch_j
        corresponds cleanly to anchor_patch_(j+s) on the time axis.
      * The first of the W targets is forced to s=1 (TimeDART-equivalent
        shift-by-one-patch). The remaining W-1 targets draw s independently at
        random from [sampling_min, sampling_max] (repeats are possible), clipped
        per-anchor to whatever the data tail allows.
      * The model receives shifts in patch units (s), so the sinusoidal
        embedding sees `patch 1 vs patch 2` as adjacent (smooth), matching the
        actual semantic. `lineage_predictor.max_period` is set to sampling_max
        (also patch units).
      * Anchors whose tail cannot fit the forced s=1 target are dropped
        from __len__.
    """

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
        if config.sampling_min < 1:
            raise ValueError(
                f"sampling_min (patch units) must be >= 1, got {config.sampling_min}"
            )
        if config.sampling_max < config.sampling_min:
            raise ValueError(
                f"sampling_max must be >= sampling_min (both in patch units); got "
                f"sampling_max={config.sampling_max}, sampling_min={config.sampling_min}"
            )
        if config.current_views < 1:
            raise ValueError(
                f"current_views must be >= 1, got {config.current_views}"
            )
        self._min_s = int(config.sampling_min)
        self._max_s = int(config.sampling_max)
        min_rows = config.input_len + config.patch_len
        if self.data.shape[0] < min_rows:
            raise ValueError(
                f"Split {self.split} has {self.data.shape[0]} rows, "
                f"need at least input_len + patch_len = {min_rows}"
            )

    def __len__(self) -> int:
        # An anchor starting at index i requires the forced s=1 target at
        # i + patch_len to fit: i + patch_len + input_len <= len(data).
        return max(0, self.data.shape[0] - self.config.input_len - self.config.patch_len + 1)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        seq_len = self.config.input_len
        P = self.config.patch_len

        anchor_begin = index
        anchor_end = anchor_begin + seq_len
        max_target_begin = len(self.data) - seq_len

        # Per-anchor cap: the largest shift s whose target fits in the data tail.
        max_s_anchor = (max_target_begin - anchor_begin) // P
        max_s_eff = min(self._max_s, max_s_anchor)
        min_s_eff = min(self._min_s, max_s_eff) if max_s_eff >= 1 else 1

        # Default (forced_first_shift_one=1): the first target is pinned to
        # s=1 (TimeDART-equivalent). The ablation (=0) samples it like the rest.
        if getattr(self.config, "forced_first_shift_one", True):
            shifts = [1]
            remaining = self.config.current_views - 1
        else:
            shifts = []
            remaining = self.config.current_views

        # Independent random patch-aligned shifts s ∈ [min_s_eff, max_s_eff].
        for _ in range(remaining):
            if max_s_eff < 1:
                s = 1
            else:
                s = random.randint(min_s_eff, max_s_eff)
            shifts.append(s)

        past = np.array(self.data[anchor_begin:anchor_end], dtype=np.float32, copy=True)
        target_starts = [anchor_begin + s * P for s in shifts]
        currents = np.stack(
            [np.array(self.data[r:r + seq_len], dtype=np.float32, copy=True)
             for r in target_starts],
            axis=0,
        )                                                       # [W, L, C]
        # Shifts are in PATCH UNITS (s), not raw timesteps.
        distances = np.array(shifts, dtype=np.int64)
        return (
            torch.from_numpy(past),
            torch.from_numpy(currents),
            torch.from_numpy(distances),
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
