"""CLI entry: `python -m model.TimeDART.test`. Pre-parses --cuda_visible_devices."""
import sys

from .cli import pop_cuda_visible_devices


if __name__ == "__main__":
    sys.argv = pop_cuda_visible_devices()
    from .forecast import test_main

    test_main()
