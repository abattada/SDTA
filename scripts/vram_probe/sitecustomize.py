"""Peak-VRAM probe used for the paper's memory table.

Reproduces the "non-invasive atexit probe" methodology: put this directory on
PYTHONPATH and Python imports it automatically at interpreter startup; on exit
it prints torch.cuda.max_memory_allocated / max_memory_reserved for every
visible device. The paper's memory table reports peak *allocated* MiB during
pretraining (2 epochs are enough for the peak to stabilize; peak allocated
VRAM is machine-independent for a fixed config).

Usage (from the repository root; a complete 2-epoch probe of the SDTA hero
cell, capacity Medium with W=4, on the widest dataset PEMS07 (883 channels)):

    PYTHONPATH=scripts/vram_probe python -u -m model.SDTA.pretrain \
        --dataset PEMS07 --enc_in 883 --model_id SDTA --run_name vram_S_97 \
        --device auto --features M --input_len 96 --patch_len 8 --stride 8 \
        --batch_size 16 --train_epochs 2 --d_model 32 --n_heads 4 --d_ff 64 \
        --time_steps 1000 --diffusion_space patch --sampling_min 1 \
        --sampling_max 12 --use_norm 1 --scheduler cosine --pretrain_causal 1 \
        --num_workers 1 --w_chunk_size 0 --checkpoint_every 10 \
        --learning_rate 0.001 --dropout 0.2 --head_dropout 0.1 --lr_decay 0.9 \
        --e_layers 2 --d_layers 1 --mask_ratio 0.5 --lineage_mlp_layers 1 \
        --current_views 4 --lineage_type relative --kv_share_lineage 0 \
        --forced_first_shift_one 1 --lineage_disabled 0 \
        --diffusion_noise_disabled 0 --past_disabled 0 --seed 97

That flag list is entry/batch_configs/time/SDTA_E2_D1.json expanded: its
`stage_defaults.pretrain` block one for one, `grid` for the swept flags, and
`auto_enc_in` for `--enc_in` (ETTh1 7, weather 21, PEMS08 170, electricity 321,
PEMS07 883). To probe another method or dataset, print the command instead of
transcribing it -- `python entry/batch.py time/<method> --dry-run` emits a
`cmd:` line per stage for every entry/batch_configs/time/*.json -- then prefix
the pretrain line with PYTHONPATH=scripts/vram_probe, lower --train_epochs to
2, and give it its own --run_name so the probe does not land in the timing
run's output directory.

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
