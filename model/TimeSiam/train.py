"""CLI entry: `python -m model.TimeSiam.train`. Pre-parses --cuda_visible_devices."""
import sys

from .cli import pop_cuda_visible_devices


if __name__ == "__main__":
    sys.argv = pop_cuda_visible_devices()
    from .forecast import train_main

    train_main()
