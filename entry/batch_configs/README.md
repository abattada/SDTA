# Experiment Configurations

`entry/batch.py` runs experiments described by the JSON configs in this
directory. Invoke by name (relative to this directory, without `.json`) or by
path:

```bash
python entry/batch.py SDTA/medium/light --dry-run   # print the plan, run nothing
python entry/batch.py SDTA/medium/light             # run it
python entry/batch_queue.py --list main_table.txt   # run a list of configs in sequence
```

## Layout

```
SDTA/{small,medium,large}/            capacity: small = E1D1, medium = E2D1, large = E2D2
  {cohort}.json                       standard horizons, supervision-axis sweep W in {1,2,4,8}
  {cohort}_ext.json                   the single extended horizon (1080 / 144 for PEMS)
  all.txt  all_ext.txt                queue lists for the five cohorts
  ablation/{arm}/{cohort}.json        one ablation arm, at the default window W=4
  ablation/all.txt                    queue list for all eight arms
{baseline}/medium/                    baselines run at the shared encoder capacity (E=2)
  {cohort}.json  {cohort}_ext.json  all.txt  all_ext.txt
time/                                 wall-clock and VRAM cost measurement (single seed 97)
_smoke/etth1.json                     ~2-minute end-to-end smoke test
main_table.txt  extended.txt  capacity_sweep.txt        top-level queue lists
```

`medium` with `W=4` is the paper's default configuration (Medium_4).

### Cohorts

Datasets are grouped into cohorts so a campaign can be run in pieces, smallest
and cheapest first:

| Cohort | Datasets |
|---|---|
| `light` | ETTh1, ETTh2, ETTm1, ETTm2, exchange_rate, weather |
| `mid` | electricity |
| `heavy` | traffic |
| `pems_light` | PEMS03, PEMS04, PEMS08 |
| `pems_heavy` | PEMS07 |

`_ext` configs drop exchange_rate from `light` (its series is too short for the
extended horizon), so they cover 11 datasets in total.

### Ablation arms

Folder names use the paper's terminology; the flags they set keep the code's
internal names.

| Folder | Flag set | Run-name token |
|---|---|---|
| `no_diffusion` | `pretrain.diffusion_noise_disabled=1` | `NoDiff_1` |
| `no_forced_s1` | `pretrain.forced_first_shift_one=0` | `F1_0` |
| `random_init` | train variant `load_pretrain_weights=0` | outputs under `pre_no_pretrain/` |
| `no_anchor` | `pretrain.past_disabled=1` | `NoPast_1` |
| `linear_probe` | `train.freeze_encoder=1` | `Probe_1` |
| `no_tda` | `pretrain.lineage_disabled=1` | `NoLin_1` |
| `broadcast` | `pretrain.lineage_type=learnable_token` | `LType_learnable_token` |
| `k=v` | `pretrain.kv_share_lineage=1` | `KVShare_1` |

The paper's ablation table reports the **medium** capacity; the same arms are
provided at small and large so the grid is complete and runnable.

## Execution model

A config expands into one **chain** per (dataset × grid combination):
`pretrain -> train (one process per pred_len) -> test (one process covering all
pred_lens)`. Steps within a chain run in order; different chains run in
parallel, admitted by live free GPU memory. Each step is a subprocess:
`python -u -m model.{model_id}.{stage} --dataset ... --run_name ... <flags>`.

Completed stages leave an empty `done` sentinel in their output directory; on
re-invocation those steps are skipped (`--rerun` disables this). If a pretrain
step re-runs, its downstream train/test steps re-run too. Failures whose log
tail looks like CUDA OOM are re-queued (up to `max_oom_retries`); other
failures stop the batch unless `--keep-going`.

After every non-dry run, one summary row is appended to `docs/batch/runs.md`
(created on first use) — the in-repo run ledger.

## Config keys

| Key | Meaning |
|---|---|
| `_doc` | Free-text description of the config's intent. |
| `name` | Batch id; runner artifacts go to `outputs/batch/{name}/{timestamp}/`. |
| `model_id` | Model package: `model/{model_id}/` must exist. |
| `datasets` | Datasets to run; each needs `data/fine/{dataset}/`. |
| `stages` | Subset of `["pretrain", "train", "test"]`. |
| `auto_enc_in` | Read the channel count from `data/fine/{ds}/train/metadata.json` and pass it as `--enc_in`. |
| `cuda_visible_devices` | GPU pool, e.g. `"0"` or `"0,1"`. Ids absent on the machine are ignored. |
| `min_free_mb`, `concurrent`, `delay`, `max_oom_retries` | Scheduler knobs: free-VRAM admission threshold, concurrency cap, seconds between launches, OOM retry budget. |
| `gpu_memory_fraction` | Optional per-process `torch.cuda.set_per_process_memory_fraction` cap. |
| `env` | Extra environment variables for child processes (the shipped configs pin the BLAS thread counts to 1). |
| `run_name` | `{base, include, aliases}` — run-name template, see below. |
| `grid` | Dict of lists; the cartesian product defines the combinations. Keys may be stage-prefixed (`pretrain.current_views`); unprefixed keys (like `seed`) apply to every stage. |
| `stage_defaults` | Per-stage flag defaults (`pretrain` / `train` / `test`); every entry becomes a `--key value` CLI flag. |
| `dataset_settings` | Per-dataset overrides of stage defaults. |
| `variants` | Train/test-stage overrides applied per chain (see below). |
| `resume_skip` | Default `true`; set `false` to always re-run. |

Merge order (later wins): `stage_defaults[stage]` < `dataset_settings` < grid
combination (unprefixed, then stage-prefixed). `stride` defaults to `patch_len`.

### Variants

The shipped configs use three:

| Variant | Effect |
|---|---|
| `{"name": "pretrained", "load_pretrain_weights": 1}` | Standard: fine-tune from the pretrained encoder. |
| `{"name": "ext", "suffix": "PLX", "load_pretrain_weights": 1}` | Extended horizon: reuses the same pretraining (so it resume-skips) and appends `_PLX` to the train/test run name, keeping its results separate from the standard run. |
| `{"name": "no_pretrain", "load_pretrain_weights": 0}` | Random-init control: the architecture is still inherited from the named pretrain run, no weights are loaded, and outputs land under `pre_no_pretrain/`. |

## Run names

`run_name.base` plus `_{alias}_{value}` for every grid key listed in `include`,
in order. For example base `arch_scan` with E=2, D=1, mask 0.5, s_max 12,
Lmlp 1, W=4, seed 2021 yields

```
arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_W_4_S_2021
```

Values are sanitized (`.` -> `p`, `-` -> `m`). Every grid key must appear in
`include` (checked at startup) so distinct combinations within one config can
never collide on a single output path. The seed is part of the run name, so
each seed is an independent chain, and outputs land under

```
outputs/pretrain/{model_id}/{dataset}/il{input_len}/{run_name}/
outputs/train/{model_id}/{dataset}/il{input_len}/pre_{tag}/{train_run}/pl{pred_len}/
outputs/test/{model_id}/{dataset}/il{input_len}/pre_{tag}/{train_run}/results.json
```

where `tag` is the pretrain run name, or `no_pretrain` for the random-init
control, and `train_run` carries the `_PLX` suffix for extended-horizon runs.

**Run names are the identity of a result.** The `done` sentinel that drives
resume is path-based and records no hyperparameters, so two configs that
generate the same run name for the same (model_id, dataset, input_len) are
treated as the same experiment — the second one silently inherits the first
one's checkpoints. That is *intended* where configs deliberately share
pretraining (`{cohort}_ext.json` and `ablation/random_init/` both reuse the
standard pretrained encoder), and it is a bug anywhere else. If you add a
config with a different training schedule, give it a distinct `run_name.base`;
this is why the smoke test uses the base `smoke` rather than `arch_scan`. The
shipped tree is collision-free: across all configs, every run name that is
shared is shared by configs with identical pretraining hyperparameters.

## The `w_chunk_size` memory knob

`w_chunk_size` (pretrain stage) trades time for memory on the supervision axis.
`0` forwards all W target windows in one batch of `B*W*C`. A value `n > 0`
splits W into chunks of `n`, doing one forward/backward per chunk and scaling
each chunk's loss by `n/W`, so the accumulated gradient equals the single-shot
gradient in exact arithmetic. **It only engages when `W > w_chunk_size`**, so
with the shipped value it is a no-op at W = 1, 2 and 4, including the paper's
default configuration (Medium, W=4).

Chunking is not bit-identical, because the decoder's partial mask and the
dropout masks are resampled per forward call: chunking draws them once per
chunk instead of once per step. It is therefore a different random realization
of the same objective. Measured on ETTh1 at W=8, seed 2021, two epochs:

| `w_chunk_size` | train loss | val loss | s/epoch |
|---:|---:|---:|---:|
| 0 | 0.160547 | 0.177422 | 8.2 |
| 4 | 0.160437 | 0.178202 | 17.5 |

The gap (~1e-3) sits inside the cross-seed standard deviation the paper
reports (0.0003–0.005), and the cost is roughly 2x wall-clock.

The shipped configs use `0` everywhere. The original W=8 campaigns used `4` on
the high-channel cohorts, where the machine had a per-process VRAM cap; the
W=8 row of the paper's capacity sweep therefore averages over cells trained
both ways. If you hit CUDA OOM at W=8 on `heavy` / `pems_heavy`, set
`w_chunk_size` to `4` in that config: it halves peak activation memory and
leaves the objective unchanged.

## Scheduling defaults

The shipped configs target **one GPU** (`cuda_visible_devices: "0"`) with a
per-cohort free-VRAM threshold and concurrency cap:

| Cohort | `min_free_mb` | `concurrent` |
|---|---:|---:|
| `light` | 2000 | 4 |
| `mid` | 4000 | 2 |
| `heavy` | 6000 | 1 |
| `pems_light` | 4000 | 2 |
| `pems_heavy` | 8000 | 1 |

With more GPUs, widen `cuda_visible_devices` and raise `concurrent`; the
scheduler places each job on the card with the most free memory and never
starts one below `min_free_mb`. These knobs affect throughput only, never
results.
