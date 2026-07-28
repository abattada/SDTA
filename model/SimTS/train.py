from __future__ import annotations

from model._baseline_forecast import train_main
from .forecast import build_model


if __name__ == "__main__":
    train_main(build_model, supports_pretrain=True)
