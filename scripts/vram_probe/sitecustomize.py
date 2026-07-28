"""Peak-VRAM probe used for the paper's memory table.

Reproduces the "non-invasive atexit probe" methodology: put this directory on
PYTHONPATH and Python imports it automatically at interpreter startup; on exit
it prints torch.cuda.max_memory_allocated / max_memory_reserved for every
visible device. The paper's memory table reports peak *allocated* MiB during
pretraining (2 epochs are enough for the peak to stabilize; peak allocated
VRAM is machine-independent for a fixed config).

Usage (from the repository root; a 2-epoch probe of the SDTA hero cell):
    PYTHONPATH=scripts/vram_probe python -u -m model.SDTA.pretrain \
        --dataset PEMS07 ... --train_epochs 2

Do not combine with entry/batch.py's `gpu_memory_fraction` option: batch.py
prepends its own sitecustomize directory to PYTHONPATH, which would shadow
this one. Run the single stage directly as above.
"""
import atexit


def _report() -> None:
    try:
        import torch
    except Exception:
        return
    if not torch.cuda.is_available():
        return
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
        reserv = torch.cuda.max_memory_reserved(i) / (1024 ** 2)
        print(f"[vram_probe] device {i}: peak allocated {alloc:.1f} MiB, "
              f"peak reserved {reserv:.1f} MiB", flush=True)


atexit.register(_report)
