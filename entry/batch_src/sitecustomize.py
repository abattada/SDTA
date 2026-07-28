"""Optional CUDA memory cap injected by entry/batch.py.

Python imports sitecustomize during interpreter startup when this directory is
on PYTHONPATH. That lets the batch runner set a per-process PyTorch CUDA memory
fraction without changing any model package code.
"""
from __future__ import annotations

import os
import sys


def _main() -> None:
    raw = os.environ.get("SDTA_CUDA_MEMORY_FRACTION", "").strip()
    if not raw:
        return
    try:
        fraction = float(raw)
    except ValueError as exc:
        raise SystemExit(f"Invalid SDTA_CUDA_MEMORY_FRACTION={raw!r}") from exc
    if not 0.0 < fraction <= 1.0:
        raise SystemExit("SDTA_CUDA_MEMORY_FRACTION must be in (0, 1].")

    import torch

    if not torch.cuda.is_available():
        return
    for device_idx in range(torch.cuda.device_count()):
        torch.cuda.set_per_process_memory_fraction(fraction, device=device_idx)
    print(
        "[batch cuda limit] set per-process CUDA memory fraction "
        f"to {fraction:g} on {torch.cuda.device_count()} visible device(s)",
        file=sys.stderr,
        flush=True,
    )


_main()
