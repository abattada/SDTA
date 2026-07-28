# Reproducibility Checklist — Where Each Item Is Satisfied

Mapping from the AAAI Reproducibility Checklist (dataset and computational
items) to the artifacts in this repository.

## 3. Dataset usage

| Item | Where |
|---|---|
| 3.2 Motivation for the selected datasets | Paper (benchmark section); `docs/datasets.md` (12 standard benchmarks spanning energy, weather, finance, traffic; three channel-scale groups) |
| 3.3 / 3.4 Novel datasets | N/A — no novel datasets are introduced |
| 3.5 Citations for existing datasets | Paper references; `docs/datasets.md` § Licenses and citations |
| 3.6 Public availability | All 12 datasets are public; `download/ETT.py` and `download/TimeSeries.py` fetch them from their public sources (Zenodo files MD5-pinned) |
| 3.7 Non-public datasets | N/A |

## 4. Computational experiments

| Item | Where |
|---|---|
| 4.2 Hyper-parameter ranges tried + selection criterion | Paper, benchmark section § "What we searched, and what we did not": one fixed configuration for all methods and datasets, no per-dataset tuning. The studied axes are exactly the sweeps shipped as configs: W ∈ {1,2,4,8}, capacity ∈ {Small, Medium, Large}, baseline axes (TimeSiam W, CPC K ∈ {1,2,4}); selection = 12-dataset average MSE after seed averaging, applied once and globally. Learning rates, schedulers, batch size, dropouts, widths and patch geometry are inherited unchanged from TimeDART/PatchTST and were never searched. See `docs/paper_code_map.md` § tables. |
| 4.3 Pre-processing code | `download/`, `data_preprocess/build_informer_windows.py` (`docs/datasets.md` gives the exact commands) |
| 4.4 Source code for experiments and analysis | `model/` (method + 7 baselines), `entry/` (runner + all experiment configs, including every ablation arm and the random-init control), `scripts/` (aggregation incl. `--pattern` templates for every table, timing, VRAM probe, parameter-count verification) |
| 4.5 Code released under a research-friendly license | `LICENSE` (MIT); third-party attributions in `THIRD_PARTY.md` |
| 4.6 Comments referencing the paper | Module headers in `model/SDTA/*.py`; full mapping in `docs/paper_code_map.md` |
| 4.7 Seed handling | Every run takes `--seed`; final configs pin seeds 2021–2025 (5 per setting); `seed_everything` in `model/SDTA/utils.py` seeds `random`, NumPy, and PyTorch (CPU+CUDA) and disables cuDNN benchmark. Caveat: `cudnn.deterministic` is not forced, so runs are seeded but not bitwise-deterministic; expect cross-seed std ~0.0003–0.005 on pred-length-averaged MSE (reported in the paper). |
| 4.8 Computing infrastructure | `README.md` § Environment: Ubuntu 22.04.5, Python 3.11.14, PyTorch 2.5.1+cu124, single NVIDIA RTX 4090 24 GB (driver 550.144.03), Intel Core i7-13700K, 70 GB RAM; accuracy runs on RTX 5090 GPUs, cost measurements on the RTX 4090; exact package pins in `requirements.txt` |
| 4.9 Evaluation metrics | MSE and MAE on the globally standardized scale (no inverse transform), sample-weighted over the test split; computed in each family's test loop (e.g. `model/SDTA/forecast.py`) and formally defined, with the motivation for reporting both, in the paper's benchmark section § "Evaluation metrics" |
| 4.10 Number of runs per reported result | 5 independent pretrain→fine-tune→test chains (seeds 2021–2025); timing/VRAM measurements use a single seed (97) and are labeled as cost, not performance |
| 4.11 Distributional information | `scripts/aggregate_results.py --std` prints cross-seed standard deviations per dataset and per group aggregate (the latter reproduces the paper's headline ±std values exactly); `scripts/significance_test.py` adds average ranks and per-dataset win counts |
| 4.12 Statistical tests | `scripts/significance_test.py` — Friedman omnibus over the 8 methods plus two-sided Wilcoxon signed-rank of SDTA against each baseline (one paired observation per dataset, Demšar 2006), Holm-corrected over the 7 comparisons, for MSE and MAE at both the standard and extended horizons; it also tests each method's own supervision axis. Reported in the paper's Results § "Statistical Significance", including the comparison that is **not** significant (SDTA vs TimeMAE-CI). Reads the same `results.json` files as `aggregate_results.py`, so it re-runs no experiments. |
| 4.13 Final hyper-parameters | `entry/batch_configs/**/*.json` (every final run's exact flags); summarized in `docs/paper_code_map.md` § One fixed configuration |
