from __future__ import annotations

from model._baseline_forecast import test_main
from .forecast import build_model


if __name__ == "__main__":
    test_main(build_model, supports_pretrain=True)
