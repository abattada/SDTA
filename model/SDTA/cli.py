"""Pre-import CUDA_VISIBLE_DEVICES handling so torch sees the right GPUs."""
from __future__ import annotations

import argparse
import os
import sys


CUDA_VISIBLE_ARG_NAMES = (
    "--cuda_visible_devices",
    "--cuda-visible-devices",
)


def apply_cuda_visible_devices(value: str | None) -> None:
    if value:
        os.environ["CUDA_VISIBLE_DEVICES"] = value


def pop_cuda_visible_devices(argv: list[str] | None = None) -> list[str]:
    source = list(sys.argv if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(*CUDA_VISIBLE_ARG_NAMES, dest="cuda_visible_devices", default=None)
    args, remaining = parser.parse_known_args(source[1:])
    apply_cuda_visible_devices(args.cuda_visible_devices)
    return [source[0], *remaining]
