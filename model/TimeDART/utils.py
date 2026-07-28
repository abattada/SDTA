"""Paths, device selection, and small helpers used by TimeDART."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRETRAIN_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "pretrain"
TRAIN_CHECKPOINT_ROOT = PROJECT_ROOT / "outputs" / "train"
TEST_RESULTS_ROOT = PROJECT_ROOT / "outputs" / "test"


def select_gpu_with_most_free_memory() -> int | None:
    if not torch.cuda.is_available():
        return None
    best_gpu = None
    best_free = -1
    for gpu_id in range(torch.cuda.device_count()):
        try:
            free_bytes, _ = torch.cuda.mem_get_info(gpu_id)
        except RuntimeError:
            continue
        if free_bytes > best_free:
            best_free = free_bytes
            best_gpu = gpu_id
    return best_gpu


def resolve_device(device: str) -> torch.device:
    if str(device).lower() == "auto":
        gpu_id = select_gpu_with_most_free_memory()
        if gpu_id is None:
            return torch.device("cpu")
        return torch.device(f"cuda:{gpu_id}")
    resolved = torch.device(device)
    if resolved.type == "cuda":
        if not torch.cuda.is_available():
            return torch.device("cpu")
        if resolved.index is not None:
            torch.cuda.set_device(resolved.index)
    return resolved


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def format_param(value: float | int) -> str:
    return f"{value:g}" if isinstance(value, float) else str(value)

