# Short-Horizon BTC Price Direction from Limit Order Book Data

MAT629 final project (Spring 2026).

Predict whether the BTC mid-price will go **up / flat / down over the next 10 seconds** from
1-second limit-order-book (LOB) snapshots. The project compares a **1-D residual CNN**, a
**bidirectional LSTM** and a **Transformer encoder**, and runs an ablation over three feature
sets. Each model is evaluated on classification metrics and on a fee-aware trading backtest.

## Data

[High-frequency crypto limit order book data](https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data)
(Kaggle, `martinsn`): 1-second snapshots of the BTC-USD and ETH-USD books, 15 levels per
side, April 7–19, 2021 (~1.03M rows). The data is **not** in this repo. The code downloads it
with `kagglehub`, or you can point it at a local folder containing `BTC_1sec.csv` and
`ETH_1sec.csv`.

## Repository layout

```
lob/                     shared code, used by every notebook
  config.py              horizon, label threshold, split fractions, fee …
  features.py            raw CSV → features, labels, z-score, split, PCA
  data.py                feature sets, sliding-window Dataset, DataLoaders
  models.py              LOBCNN, LOBLSTM, LOBTransformer
  training.py            training loop (early stopping on validation)
  backtest.py            non-overlapping, fee-aware backtest + threshold sweep
  experiment.py          train → pick threshold on val → test once → plots
scripts/build_dataset.py build data/features.parquet from the command line
notebooks/
  01_build_dataset.ipynb
  02_cnn.ipynb
  03_lstm.ipynb
  04_transformer.ipynb
  05_compare.ipynb       combines results/*_summary.csv
results/                 per-model summary CSVs (written by notebooks 02–04)
```

## Quick start

**Colab:** open a notebook from GitHub, select a GPU runtime and run all cells. The first cell
clones this repo and installs the dependencies. Set `REPO_URL` in that cell to your fork.

**Locally:**

```bash
pip install -r requirements.txt
python scripts/build_dataset.py            # or: --data-dir path/to/csvs
jupyter lab notebooks/
```

Building the full table takes about 1 minute and about 2 GB of RAM. Each model notebook
builds the table automatically if `data/features.parquet` is missing.

## Method

**Labels.** `future_return[t] = mid(t + 10 s) / mid(t) − 1`, looked up by timestamp (the feed
has gaps, so "10 rows later" is not always 10 seconds later). The label is Up / Down when
|return| > 1.5 bps, and Flat otherwise.

**Features** (all normalised with a backward-looking 1-hour rolling z-score, clipped to ±10):

| Group | # | Content |
|---|---|---|
| LOB | 60 | bid/ask distance-to-mid and notional, 15 levels |
| PCA | ≤10 | StandardScaler + PCA of the LOB block, fitted on train only, Kaiser criterion |
| Market | 15 | BTC & ETH spread, top-of-book imbalance, 10s/60s volatility and momentum, BTC–ETH cross terms |
| Classic | 13 | weighted queue imbalance, order-flow imbalance (Cont et al. 2014), Kyle's λ, depth ratio (Gould et al. 2013), tick rule, Amihud-style illiquidity |

**Feature sets** (the same definitions are used for all models):

| Name | Features |
|---|---|
| `full` | PCA + market + classic |
| `no_pca` | raw LOB + market + classic |
| `no_market` | PCA + classic |

**Evaluation protocol** (identical for all models):

1. Chronological split: 70% train, 15% validation, 15% test. Rows whose 10-second label window
   crosses into the next split are purged from the targets.
2. Early stopping on validation loss. The best-validation weights are restored.
3. Trading rule: go short if P(Down) > θ, go long if P(Up) > θ, otherwise stay flat. θ is
   chosen by **validation** Sharpe and then applied once to test. Argmax trading is reported
   too.
4. Backtest on non-overlapping 10-second steps, cost = 1.5 bps × |Δposition| (so a reversal
   pays twice). Sharpe is annualised with √(steps per year) ≈ 1776. Compare these Sharpe
   values across models only, not with daily-frequency strategies.

## Results

Run notebooks 02–05 to fill in this table (`results/*_summary.csv`).

| Model | Feature set | Test acc | Dir-F1 | Argmax return | θ (val) | θ return | θ Sharpe |
|---|---|---|---|---|---|---|---|
| CNN | full | | | | | | |
| … | | | | | | | |

## Notes on changes from the original course notebooks

This code is a cleaned-up version of the notebooks used for the course report. While
refactoring, several issues were fixed, so the numbers here will **not** match the report:

- **No test-set selection.** Early stopping and the trading threshold were originally chosen
  on the test set. They are now chosen on a separate validation split.
- **One shared pipeline.** Previously the models read differently-built files, and
  `no_pca` meant different feature sets for different models.
- **Order-flow imbalance** compared `*_distance_0` (a distance to the mid) instead of prices.
  Best bid/ask prices are now rebuilt from mid × (1 + distance) and rounded to the tick.
- **Z-score of tiny-scale factors.** `std + 1e-8` squashed Amihud and Kyle's λ (≈1e-10) to
  ~0. Zero-std windows are now handled explicitly instead.
- **Transformer normalisation** used a single mean/std from the last training hour for the
  whole series. All models now share the rolling z-score.
- **Consistent backtest.** All models now use the same costs (reversals pay twice, the first
  entry is charged), the same non-overlapping steps and a purge at split boundaries.
- Labels use timestamps instead of row offsets.

## Limitations

- About 12 days of data: the validation and test windows are about 1.8 days each, so a single
  regime can dominate the results.
- The backtest assumes execution at the mid-price with a flat fee and no latency or slippage.
- The Amihud factor uses book depth in place of traded volume.
