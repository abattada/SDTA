# Third-Party Attributions

This repository re-implements several published self-supervised time-series
methods as capacity-matched baselines. The re-implementations are wired to this
repository's shared data pipeline and small shared encoder (see
`docs/paper_code_map.md` and the paper's "A Fair, Capacity-Matched Benchmark"
section); pretext mechanisms follow the official implementations.

## Baseline method code

| Directory | Method | Original paper | Official repository (license) |
|---|---|---|---|
| `model/PatchTST/` | PatchTST (supervised + SSL) | Nie et al., ICLR 2023 | https://github.com/yuqinie98/PatchTST (Apache-2.0) |
| `model/TimeSiam/`, `model/TimeSiam_W/` | TimeSiam | Dong et al., ICML 2024 | https://github.com/thuml/TimeSiam (MIT) |
| `model/TimeDART/` | TimeDART | Wang et al., ICML 2025 | https://github.com/ustc-time-series/TimeDART (no license file) |
| `model/TimeMAE_CI/` | TimeMAE (channel-independent variant) | Cheng et al., 2023 | https://github.com/ustc-time-series/TimeMAE (no license file) |
| `model/SimMTM/` | SimMTM | Dong et al., NeurIPS 2023 | https://github.com/thuml/SimMTM (no license file) |
| `model/CPC/` | Contrastive Predictive Coding | van den Oord et al., 2018 | (multiple community implementations) |
| `model/SimTS/` | SimTS | Zheng et al., 2023 | https://github.com/xingyu617/SimTS_Representation_Learning (Apache-2.0) |

`model/PatchTST/src/` is directly derived from the official PatchTST code
(Apache License 2.0), with the encoder backbone swapped for this repository's
shared small encoder (`model/conv_encoder.py`) and I/O rewired to `data/fine/`.

The TimeDART, TimeMAE, and SimMTM upstream repositories publish no license
file; the corresponding directories here are re-implementations of the
published pretext mechanisms against this repository's shared infrastructure,
not copies of that code, and no license is claimed on their behalf.

## Datasets

No dataset files are redistributed with this repository. `download/` fetches
every dataset from its public source at build time; see `docs/datasets.md` for
URLs, checksums, licenses, and required citations.

| Dataset | Source | License |
|---|---|---|
| ETTh1/ETTh2/ETTm1/ETTm2 | https://github.com/zhouhaoyi/ETDataset | CC BY-ND 4.0 |
| electricity, weather, traffic, exchange_rate | Autoformer/TimesNet community CSVs via https://huggingface.co/datasets/thuml/Time-Series-Library | public underlying sources (UCI, MPI-BGC Jena, Caltrans PeMS, LSTNet) |
| PEMS03/04/07/08 | Zenodo record 7816008 (MD5-pinned) | CC BY 4.0 |
