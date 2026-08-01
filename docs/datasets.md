# Datasets

All 12 datasets are standard public long-term-forecasting benchmarks. Nothing
is redistributed with this repository: `download/` fetches every file from its
public source, and `data_preprocess/` rebuilds the exact training store.

## Download

```bash
# ETT (one call per dataset)
python download/ETT.py --dataset ETTh1
python download/ETT.py --dataset ETTh2
python download/ETT.py --dataset ETTm1
python download/ETT.py --dataset ETTm2

# electricity / weather / traffic / exchange_rate + PEMS03/04/07/08
python download/TimeSeries.py
```

Sources:

| Datasets | Source | Integrity check |
|---|---|---|
| ETTh1, ETTh2, ETTm1, ETTm2 | github.com/zhouhaoyi/ETDataset (raw.githubusercontent + jsdelivr mirror) | header must contain the `OT` column |
| electricity, weather, traffic, exchange_rate | huggingface.co/datasets/thuml/Time-Series-Library (the community-standard Autoformer/TimesNet CSVs) | header must contain `date` |
| PEMS03, PEMS04, PEMS07, PEMS08 | zenodo.org/records/7816008 | pinned MD5 per file (see `download/TimeSeries.py`) |

## Preprocess

```bash
python data_preprocess/build_informer_windows.py --dataset all --max_overlap 96
```

`--max_overlap 96` matches the paper (input length 96); it prepends 96 rows of
previous-split context to the validation/test files so that every input window
has valid history while all prediction targets stay inside the split.

Per dataset this writes `data/fine/{dataset}/`:

- `train/ validation/ test/` each holding `data.npy` (float32, rows x channels,
  globally standardized) and `metadata.json` (row offsets, channel count);
- `scaler.npz` (per-channel mean/std, **fit on the training split only**);
- `borders.json` (split provenance).

Split conventions (as stated in the paper):

| Group | Datasets (channels) | Split | Prediction lengths |
|---|---|---|---|
| 4-ETT | ETTh1, ETTh2, ETTm1, ETTm2 (7) | 12 / 4 / 4 months (Informer) | 96 / 192 / 336 / 720 |
| 4-Wide | exchange_rate (8), weather (21), electricity (321), traffic (862) | 0.7 / 0.1 / 0.2 | 96 / 192 / 336 / 720 |
| 4-PEMS | PEMS03 (358), PEMS04 (307), PEMS07 (883), PEMS08 (170) | 0.6 / 0.2 / 0.2 | 12 / 24 / 48 / 96 |

The extended-horizon stress test uses a single long horizon (1080 non-PEMS,
144 PEMS) on 11 datasets — exchange_rate is excluded because its series is too
short. Metrics are computed on the standardized scale (no inverse transform).

## Licenses and citations

This repository redistributes **no data at all** — no raw file, no derived
window store, no cached tensor. It ships only download code (`download/`) and
preprocessing code (`data_preprocess/`); every byte of data is fetched by the
user from the original public source and transformed locally. Each dataset
therefore reaches the user under its own license, direct from its own host, and
nothing here constitutes redistribution of a dataset.

- **ETT**: CC BY-ND 4.0 (see the ETDataset repository). Cite Informer
  (Zhou et al., AAAI 2021). The **ND** (NoDerivatives) term is why the
  preprocessed window store is never shipped: `data_preprocess/` writes it into
  the gitignored `data/` directory on the user's own machine, and the release
  contains neither the original CSVs nor any derivative of them.
- **electricity / weather / traffic / exchange_rate**: underlying sources are
  public — UCI ElectricityLoadDiagrams20112014, the MPI-BGC Jena weather
  station, Caltrans PeMS, and the LSTNet exchange-rate collection
  (Lai et al., SIGIR 2018). The exact CSVs are the community-standard
  preprocessed versions; cite Autoformer (Wu et al., NeurIPS 2021) and/or
  the Time-Series-Library (TimesNet, Wu et al., ICLR 2023) for the bundle.
- **PEMS03/04/07/08**: CC BY 4.0 (Zenodo record 7816008); underlying data is
  public Caltrans PeMS. Cite ASTGCN (Guo et al., AAAI 2019) / STSGCN
  (Song et al., AAAI 2020).
