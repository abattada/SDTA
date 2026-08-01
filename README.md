# SDTA: Shifted-Window Diffusion with Temporal-Distance Attention

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-ee4c2c)
![License](https://img.shields.io/badge/license-MIT-green)

Official code for **"Shifted-Window Diffusion with Temporal-Distance Attention
for Self-Supervised Time-Series Forecasting"** (SDTA).

SDTA is a self-supervised pretraining pretext that scales **supervision**
instead of model size: one encoded input window conditions the denoising of
$W$ diffusion-corrupted target windows, each shifted a sampled distance into
the future, with a continuous temporal-distance code injected on the attention
key side only. Raising $W$ multiplies supervision at linear activation memory,
fits one consumer GPU, and adds no deployment cost.

This repository reproduces **every table in the paper**: the 12-dataset
comparison against 7 self-supervised baselines, the supervision-axis
($W$) sweeps, the extended-horizon stress test, all 8 ablation arms, and the
wall-clock/VRAM cost analysis.

## Building the code appendix

**Do not zip this directory.** It is a live git repository: `.git/` carries the
commit history with its author identity and remote URL, and the gitignored
`data/` and `outputs/` trees add hundreds of megabytes. Build the archive with

```bash
./scripts/make_code_appendix.sh            # -> code_appendix.zip
```

The script copies exactly the files git tracks or would offer to track
(`git ls-files --cached --others --exclude-standard`), taken from the **working
tree**, so the archive always matches what is on disk — uncommitted fixes and
not-yet-added files included — while `.gitignore` keeps `data/`, `outputs/` and
`__pycache__` out and `.git` is never reached. The script also excludes itself:
it is packaging tooling, not paper implementation, and it has to spell the
deanonymizing strings out in order to scan for them.

It then unzips its own output and verifies it, failing closed on any of: a
`.git` directory, `__pycache__`, `.pyc`, a `data/` or `outputs/` tree, CJK text
in file contents, a deanonymizing or non-ASCII **path name**, or a deanonymizing
string anywhere in the contents (GitHub URLs outside the third-party attribution
list, author names, e-mail addresses, absolute home paths). Contents are scanned
as text even inside binary files. It exits non-zero and deletes the archive if
any check fails, so a failed run cannot leave a publishable-looking file
behind.

## Results at a glance

12-dataset average MSE, one fixed configuration, identical encoder capacity,
5-seed average (lower is better):

| **SDTA (W=4)** | TimeMAE-CI | CPC | TimeDART | TimeSiam | SimMTM | PatchTST-SSL | SimTS |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0.3445** | 0.3491 | 0.3610 | 0.3614 | 0.3621 | 0.3643 | 0.3790 | 0.3844 |

SDTA is best on 5/12 datasets, top-two on 10/12, and the only method that
gains accuracy from scaling its supervision axis (`W=1 → 4`: −4.1% at
standard horizons, −5.6% at 1.5× extended horizons). Peak pretraining VRAM at
the default configuration is 4874 MiB on the widest dataset (883 channels).

## Quick start

```bash
# 1. Environment (Python >= 3.11 required)
conda create -n sdta python=3.11 -y && conda activate sdta
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

# 2. Data (downloaded from public sources; nothing is redistributed here)
python download/ETT.py --dataset ETTh1   # repeat for ETTh2 ETTm1 ETTm2
python download/TimeSeries.py            # tslib CSVs + PEMS (MD5-checked)
python data_preprocess/build_informer_windows.py --dataset all --max_overlap 96

# 3. Smoke test (~2 min): full pretrain -> fine-tune -> test chain on ETTh1
python entry/batch.py _smoke/etth1
```

The smoke test writes `outputs/test/SDTA/ETTh1/il96/.../results.json`;
running it again skips everything (resume via `done` sentinels).

## Repository layout

```
model/                  One package per method, run as python -m model.<id>.<stage>
  SDTA/               the proposed method (SDTA)
  TimeSiam/ TimeDART/ TimeMAE_CI/ PatchTST/ SimMTM/ CPC/ SimTS/   baselines
  TimeSiam_W/             TimeSiam + SDTA's multi-window sampler (axis control)
  _baseline_forecast.py conv_encoder.py _config_inherit.py _test_io.py   shared infra
entry/
  batch.py              config-driven experiment runner (GPU scheduling, resume)
  batch_queue.py        runs several batch configs in sequence
  batch_configs/        every final experiment configuration (JSON) + queue lists
download/               dataset downloaders
data_preprocess/        raw -> data/fine training store
scripts/                result aggregation, timing analysis, VRAM probe, param counts
  make_code_appendix.sh   build + verify the anonymous submission archive
```

Key documents:

| Document | Content |
|---|---|
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md) | Where each AAAI Reproducibility Checklist item is satisfied |
| [docs/paper_code_map.md](docs/paper_code_map.md) | Paper concept ↔ code mapping; table ↔ config mapping; run-name templates |
| [docs/datasets.md](docs/datasets.md) | Dataset sources, licenses, splits, preprocessing |
| [entry/batch_configs/README.md](entry/batch_configs/README.md) | Batch config format and execution model |
| [THIRD_PARTY.md](THIRD_PARTY.md) | Attributions for baseline re-implementations and datasets |

## Reproducing the paper

All experiments run through the batch runner. A config expands into
(dataset × grid × seed) chains of `pretrain -> train (per pred_len) -> test`
subprocesses; finished stages leave a `done` sentinel and are skipped on
re-runs, so interrupted campaigns resume where they stopped.

```bash
python entry/batch.py SDTA/medium/light --dry-run   # preview the exact commands
```

Configs are organised by capacity (`small` = E1D1, `medium` = E2D1, `large` =
E2D2) and dataset cohort; `medium` with `W=4` is the paper's default.

| Paper table | Command |
|---|---|
| Main results (8 methods) | `python entry/batch_queue.py --list main_table.txt` |
| W sweep / capacity grid | `python entry/batch_queue.py --list capacity_sweep.txt` |
| Baseline axis sweeps | `TimeSiam_W/medium/*` (W ∈ {2,4}), `CPC/medium/*` (K ∈ {1,2,4}) |
| s_max sweep (supplement) | `python entry/batch.py SDTA/medium/smax_sweep/light` |
| Extended horizon | `python entry/batch_queue.py --list extended.txt` |
| Ablations (8 arms) | `python entry/batch_queue.py --list SDTA/medium/ablation/all.txt` |
| Extended-horizon ablations (2 arms) | `python entry/batch_queue.py --list SDTA/medium/ablation/all_ext.txt` |
| Cost (time + VRAM) | `python entry/batch_queue.py --list time/list.txt --concurrent 1 --delay 0 --config-delay 0` |

Cohort naming throughout `entry/batch_configs/`: `light` = the 6 small
datasets, `mid` = electricity, `heavy` = traffic, `pems_light` = PEMS03/04/08,
`pems_heavy` = PEMS07. `_ext` configs are the extended-horizon variants
(11 datasets; exchange_rate is too short for the 1080-step horizon).

Notes:

- The shipped configs target a single GPU (`cuda_visible_devices: "0"`) with
  per-cohort memory thresholds. With more GPUs, widen that list and raise
  `concurrent`; the scheduler places each job on the card with the most free
  memory. These knobs change throughput only, never results.
- Standard and extended horizons are separate configs, so a standard run is
  never contaminated by the long horizon. `_ext` configs reuse the standard
  pretraining (they resume-skip it) and tag their outputs `_PLX`.
- Performance configs run 5 seeds (2021–2025) per setting: every number in
  the paper is a 5-seed average. Timing configs use a single seed (97). The
  s_max sweep is the one exception, 3 seeds (2021–2023) on 6 datasets, as its
  supplement table states; do not read its cells against the 5-seed numbers.
- Only two ablation arms were run at the extended horizon, `no_tda` and `k=v`;
  the third row of that table is the default configuration from `extended.txt`.
  The other six arms exist at the standard horizon only.
- After each non-dry batch, the runner appends one summary row to
  `docs/batch/runs.md` inside the repository (an in-repo run ledger; created
  on first use).
- Approximate cost of the full main table on one 24 GB GPU: budget several
  GPU-days end to end (dominated by traffic/PEMS07 pretraining). Start with
  the `light` cohorts.

## Aggregating results

```bash
python scripts/aggregate_results.py            # main table (MSE + MAE)
python scripts/aggregate_results.py --std      # with cross-seed std (per dataset and per group)
python scripts/aggregate_results.py --extended # extended-horizon table
python scripts/aggregate_results.py --sdta-w 2 # SDTA at W=2
python scripts/epoch_time.py --contains time_  # wall-clock (drop epoch 1)
python scripts/count_params.py                 # verify the paper's parameter counts
python scripts/significance_test.py            # Friedman + Wilcoxon (paper's significance table)
```

`significance_test.py` takes `--axis-pair LABEL=DIR:TPL_LOW,DIR:TPL_HIGH` to test
any supervision-axis step beyond the built-in ladder, and `epoch_time.py` adds a
`source` column (`elapsed_sec` or `mm:ss`) recording how each row's timing was
read, since PatchTST-SSL logs only whole seconds. `aggregate_results.py` takes
`--seeds` for sub-sweeps run at a different seed budget; the s_max sweep needs
`--seeds 2021 2022 2023`.

`significance_test.py` reproduces the paper's statistical tests from the same
`results.json` files, so it re-runs nothing: a Friedman omnibus over the eight
methods, two-sided Wilcoxon signed-rank of SDTA against each baseline with
Holm correction (one paired observation per dataset, following Demšar 2006),
and the same test applied to each method's own supervision axis. It needs
`scipy`, which is in `requirements.txt`.

The standard tables use only the canonical four prediction lengths
({96,192,336,720} non-PEMS, {12,24,48,96} PEMS); the script enforces this.
W-sweep capacities, baseline axis sweeps, and every ablation arm are
aggregated with `--pattern` — [docs/paper_code_map.md](docs/paper_code_map.md)
§ Run-name templates lists the exact template for each paper table, e.g. the
random-init arm:

```bash
python scripts/aggregate_results.py --pre-tag no_pretrain --pattern \
  'random-init=SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_W_4_S_{s}'
```

Peak-VRAM measurement (paper's memory table; see `scripts/vram_probe/`). The
command below is the table's SDTA row at capacity Medium, `W=4`, on the widest
dataset (PEMS07, 883 channels); two epochs are enough for the peak to settle,
and the probe prints `[vram_probe] device 0: peak allocated ... MiB` on exit:

```bash
PYTHONPATH=scripts/vram_probe python -u -m model.SDTA.pretrain \
  --dataset PEMS07 --enc_in 883 --model_id SDTA --run_name vram_S_97 \
  --device auto --features M --input_len 96 --patch_len 8 --stride 8 \
  --batch_size 16 --train_epochs 2 --d_model 32 --n_heads 4 --d_ff 64 \
  --time_steps 1000 --diffusion_space patch --sampling_min 1 --sampling_max 12 \
  --use_norm 1 --scheduler cosine --pretrain_causal 1 --num_workers 1 \
  --w_chunk_size 0 --checkpoint_every 10 --learning_rate 0.001 --dropout 0.2 \
  --head_dropout 0.1 --lr_decay 0.9 --e_layers 2 --d_layers 1 --mask_ratio 0.5 \
  --lineage_mlp_layers 1 --current_views 4 --lineage_type relative \
  --kv_share_lineage 0 --forced_first_shift_one 1 --lineage_disabled 0 \
  --diffusion_noise_disabled 0 --past_disabled 0 --seed 97
```

Every flag after `--dataset` is `entry/batch_configs/time/SDTA_E2_D1.json`
read out loud: its `stage_defaults.pretrain` block is that flag list one for
one, `grid` pins the swept flags (`--e_layers`, `--d_layers`,
`--current_views`, `--seed`), and `auto_enc_in` supplies `--enc_in` per dataset
(ETTh1 7, weather 21, PEMS08 170, electricity 321, PEMS07 883). The other rows
of the memory table are the same recipe over the other `time/*.json` configs,
so print their commands rather than transcribing them:

```bash
python entry/batch.py time/TimeSiam --dry-run   # any entry/batch_configs/time/*.json
```

Each `cmd:` line is one complete stage invocation. Take the pretrain line for
the dataset you want, prefix it with `PYTHONPATH=scripts/vram_probe`, and lower
`--train_epochs` to 2; `--run_name` only names the output directory, so change
it (as above) to keep the probe out of the timing run's outputs.

## Environment

Tested on: Ubuntu 22.04.5, Python 3.11.14, PyTorch 2.5.1 (cu124 wheels),
single NVIDIA RTX 4090 (24 GB), driver 550.144.03, Intel Core i7-13700K,
70 GB RAM. The paper's accuracy numbers were produced on NVIDIA RTX 5090
GPUs; its wall-clock and VRAM measurements were taken on the single RTX 4090
above. Every experiment fits on one 24 GB GPU.

## Reproducibility notes

- Every stage takes `--seed`; `seed_everything` seeds Python, NumPy, and
  PyTorch (CPU + CUDA) and disables cuDNN benchmark. cuDNN determinism is not
  forced, so single runs are seeded but not bitwise-reproducible; cross-seed
  std of the reported aggregates is ~0.0003–0.005 (reported in the paper).
- Architecture flags live only in the pretrain stage; fine-tuning inherits
  them from the pretrain run's `config.json` (`model/_config_inherit.py`),
  which prevents encoder/head mismatches.
- See [REPRODUCIBILITY.md](REPRODUCIBILITY.md) for the full checklist mapping.

## Citation

If you use this code, please cite the SDTA paper. The paper is under
double-blind review; this entry will be updated with the full reference at
camera-ready (see also [CITATION.cff](CITATION.cff)):

```bibtex
@article{sdta2027,
  title  = {Shifted-Window Diffusion with Temporal-Distance Attention
            for Self-Supervised Time-Series Forecasting},
  author = {Anonymous},
  year   = {2026},
  note   = {Under review}
}
```

## License

MIT (see [LICENSE](LICENSE)). One exception: `model/PatchTST/src/` is vendored
from the official PatchTST implementation and remains under Apache License 2.0,
a complete copy of which is included at
[LICENSE-Apache-2.0](LICENSE-Apache-2.0); each of those files carries a
provenance header. Baseline re-implementations follow their original papers;
see [THIRD_PARTY.md](THIRD_PARTY.md) for code attributions (including RevIN,
vendored transitively via PatchTST) and dataset licenses.
