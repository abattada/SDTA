# Paper ↔ Code Map

How each part of the paper maps to this repository. The proposed method lives
in `model/SDTA/`; each baseline is its own package under `model/`.

## Method (paper Section: Method)

| Paper concept | Code |
|---|---|
| Shifted-window pretext: one clean anchor window, W target windows at patch-unit shifts s drawn independently at random from [1, s_max]; forced s=1 for the first window | `model/SDTA/dataset.py` — `SiamesePretrainDataset` (patch-aligned shift sampling; `forced_first_shift_one`) |
| Forward diffusion corruption: cosine schedule, T=1000, step t drawn independently per patch, x0-prediction target | `model/SDTA/diffusion.py` — `Diffusion`, `sample_t_per_patch`, `NoiseStepEmbedding` |
| Channel-independent patching (patch 8, stride 8, N=12 tokens) + instance normalization | `model/SDTA/encoder.py` — `PatchEmbedding`; per-window normalize/denormalize in `model/SDTA/pretrain.py` |
| Causal Transformer encoder (shared, encodes the anchor once) | `model/SDTA/encoder.py` — `TransformerEncoder` |
| Causal-masked cross-attention decoder conditioned on the anchor encoding | `model/SDTA/encoder.py` — `CrossAttentionDecoder`, `generate_partial_mask` (`mask_ratio` = fraction of admissible past key positions dropped) |
| **Temporal-distance attention**: continuous code ℓ(p, s) = MLP(concat[SE(p), SE(s)]) injected on the attention **key side only** (K = z + ℓ, V = z) | `model/SDTA/encoder.py` — `RelativeLineagePredictor`; key-side injection in `PretrainModel.forward` (`model/SDTA/pretrain.py`: `kv_key = past_enc + lineage`, `kv_value = past_enc`) |
| Training objective: single MSE on denoised reconstruction in raw (standardized) space | `model/SDTA/pretrain.py` — `train_pretrain_model` loss |
| Transfer: only patch embedding + encoder move downstream; linear flatten head; fine-tune | `model/SDTA/forecast.py` — `_load_pretrained_encoder`, `Forecaster`; `model/SDTA/forecast_head.py` — `ForecastHead` |
| Supervision axis W (windows per anchor) | `--current_views` (pretrain flag); W-chunked gradient accumulation `--w_chunk_size` is the constant-memory variant |
| Capacity classes Small / Medium / Large | (E, D) = `--e_layers` / `--d_layers`: Small (1,1), Medium (2,1), Large (2,2); notation Size_W, paper default Medium_4 |

### Symbols

The code uses the paper's symbols: `s` is the shifted distance in patch units
(`sampling_min`/`sampling_max` bound it, `s_max = 12`), `p` is the key position
along the attention K axis, `W` is the window count (`current_views`), and `t`
is the per-patch diffusion step. One unrelated collision is flagged in the
source: the `s` argument of `Diffusion._cosine_beta_schedule` is the cosine
schedule offset of Nichol & Dhariwal (2021), not the shifted distance.

## Ablation arms (paper Table: ablations) ↔ CLI flags

All flags belong to `python -m model.SDTA.pretrain` unless noted.

| Paper arm | Config folder | Flag |
|---|---|---|
| random init (no pretraining) | `random_init` | `--load_pretrain_weights 0` (train stage; architecture still inherited from the pretrain config, weights stay random; outputs tagged `pre_no_pretrain`) |
| linear probe (frozen encoder) | `linear_probe` | `--freeze_encoder 1` (train stage, `model.SDTA.train`) |
| no anchor conditioning | `no_anchor` | `--past_disabled 1` |
| no diffusion (clean targets) | `no_diffusion` | `--diffusion_noise_disabled 1` |
| broadcast token (discrete, TimeSiam-style) | `broadcast` | `--lineage_type learnable_token` |
| no temporal-distance attention | `no_tda` | `--lineage_disabled 1` |
| no forced s=1 target | `no_forced_s1` | `--forced_first_shift_one 0` |
| K=V (code on keys and values) | `k=v` | `--kv_share_lineage 1` |

The distance-only sinusoidal broadcast code used in the shift-invariance
analysis is `--lineage_type sinusoidal`.

## Paper tables ↔ configs

Run a config with `python entry/batch.py <name>`; run a queue list with
`python entry/batch_queue.py --list <file>`. Every performance config pins
seeds 2021–2025 (5 chains per setting).

| Paper table | Configs / queue list |
|---|---|
| Main results (8 methods, 12 datasets) | `main_table.txt` — SDTA at `SDTA/medium/*` (take W=4) plus `{TimeMAE_CI,TimeDART,TimeSiam,SimMTM,PatchTST,CPC,SimTS}/medium/*` |
| W sweep (W ∈ {1,2,4,8} × Small/Medium/Large) | `capacity_sweep.txt` — `SDTA/{small,medium,large}/*` (each config sweeps all four W) |
| Baseline supervision axes (TimeSiam W, CPC K) | `TimeSiam_W/medium/*` (W ∈ {2,4}; W=1 is the TimeSiam main-table entry), `CPC/medium/*` (K ∈ {4,2,1}) |
| Extended horizon (1080 / 144) | `extended.txt` — every `*_ext` config, 11 datasets |
| Ablations | `SDTA/medium/ablation/all.txt` (all eight arms); small/large equivalents under `SDTA/{small,large}/ablation/` |
| Peak pretraining VRAM | `time/*.json` + `scripts/vram_probe/` (peak *allocated* MiB, `torch.cuda.max_memory_allocated`, 2-epoch probe) |
| Wall-clock cost | `time/list.txt` (6 epochs, seed 97, serial, one GPU) + `scripts/epoch_time.py` (drop epoch 1, average the rest) |
| Transferred/head parameter counts | `scripts/count_params.py` |

## Run-name templates for aggregation

`scripts/aggregate_results.py` covers the main and extended tables directly;
every other table is one `--pattern 'LABEL=MODEL_DIR:RUN_TPL'` invocation away
(`{s}` is the seed placeholder).

| Runs | MODEL_DIR : RUN_TPL |
|---|---|
| SDTA capacity (E,D) at window W | `SDTA:arch_scan_Enc_{E}_Dec_{D}_Mask_0p5_SMax_12_Lmlp_1_W_{W}_S_{s}` — (E,D) ∈ {(1,1) Small, (2,1) Medium, (2,2) Large}, W ∈ {1,2,4,8} |
| TimeSiam_W at W ∈ {2,4} | `TimeSiam_W:w_scan_W_{W}_S_{s}` (W=1 anchor = the TimeSiam main-table entry) |
| CPC at K ∈ {1,2,4} | `CPC:arch_scan_Enc_2_K_{K}_Neg_64_S_{s}` |
| no temporal-distance attention | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_NoLin_1_SMax_12_Lmlp_1_W_4_S_{s}` |
| broadcast token | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_LType_learnable_token_W_4_S_{s}` |
| K=V code sharing | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_KVShare_1_W_4_S_{s}` |
| no forced s=1 | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_F1_0_Lmlp_1_W_4_S_{s}` |
| no diffusion | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_NoDiff_1_SMax_12_Lmlp_1_W_4_S_{s}` |
| no anchor conditioning | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_NoPast_1_SMax_12_Lmlp_1_W_4_S_{s}` |
| linear probe | `SDTA:arch_scan_Enc_2_Dec_1_Mask_0p5_SMax_12_Lmlp_1_W_4_S_{s}_Probe_1` |
| random init | same template as the standard runs, plus `--pre-tag no_pretrain` |
| extended horizon | same template as the standard runs with `_PLX` appended; `--extended` handles this automatically |

## One fixed configuration (paper Section: benchmark)

Values live in the `stage_defaults` of every config
(e.g. `entry/batch_configs/SDTA/medium/light.json`):

| Group | Setting |
|---|---|
| Model | input 96, patch 8, stride 8 (N=12 tokens); d_model 32, d_ff 64, 4 heads; E=2 encoder blocks, D=1 decoder block; dropout 0.2, head dropout 0.1 |
| Pretraining | Adam, lr 1e-3, ExponentialLR gamma 0.9 per epoch, 50 epochs, batch 16; diffusion T=1000, cosine schedule, x0-prediction, per-patch t; mask 0.5; s_max 12; W=4 |
| Fine-tuning | Adam, max lr 1e-4, OneCycleLR pct_start 0.3 (`--lradj step`), 10 epochs, early-stop patience 3, batch 16 |
| Selection | best validation loss checkpoint (both stages); no per-dataset tuning of any hyperparameter for any method |
