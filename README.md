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
  data.py                feature sets, labels, target rows
  loaders.py             sliding-window Dataset + DataLoaders (PyTorch)
  models.py              LOBCNN, LOBLSTM, LOBTransformer
  baselines.py           LightGBM baseline (current-second features only)
  training.py            training loop (early stopping on validation)
  backtest.py            non-overlapping, fee-aware backtest; threshold and hysteresis rules
  experiment.py          train → pick trading rule on val → test once → plots, saved predictions
  walkforward.py         rolling folds (5 d train / 1 d val / 1 d test), per-fold PCA refit
  hparams.py             per-model hyperparameters
scripts/
  build_dataset.py       build data/features.parquet from the command line
  run_model.py           train + evaluate one model over feature sets × horizons (resumable)
  walk_forward.py        walk-forward evaluation with several seeds + seed ensemble (resumable)
  analyze_preds.py       long/short, per-day and parameter-sensitivity checks from saved predictions
notebooks/
  colab_runner.ipynb     one-click Colab run of everything, results kept in Google Drive
  01_build_dataset.ipynb
  02_cnn.ipynb
  03_lstm.ipynb
  04_transformer.ipynb
  05_compare.ipynb       combines results/*_summary.csv
results/h<horizon>/      per-model summary CSVs, figures, val/test probabilities
```

## Quick start

**Colab:** open [`notebooks/colab_runner.ipynb`](https://colab.research.google.com/github/lshangro-png/mat629-lob-prediction/blob/main/notebooks/colab_runner.ipynb),
select a GPU runtime, and run the setup cell, then one cell per model. Data and results are
stored in `MyDrive/mat629-lob/`, and an interrupted run resumes where it stopped. The
per-model notebooks (02–04) can also be opened in Colab directly for interactive work.

**Locally:**

```bash
pip install -r requirements.txt
python scripts/build_dataset.py            # or: --data-dir path/to/csvs
python scripts/run_model.py --model cnn    # or open notebooks/ in Jupyter
```

Building the full table takes about 1 minute and about 2 GB of RAM. Each model notebook
builds the table automatically if it is missing.

**One day first, then all data.** Development was done on the first ~1 day of data, then
scaled up to the full 12 days. To do the same, set `SUBSET_ROWS = 100_000` (~1.2 days) in the
first cell of each notebook, or run `python scripts/build_dataset.py --nrows 100000 --out
data/features_100000.parquet`. Subset runs use their own feature file and write to
`results/subset_<rows>/`, so they never overwrite the full-data results.

## Method

**Labels.** For each horizon h ∈ {10, 30, 60, 300} s, `future_return_h[t] = mid(t + h) / mid(t) − 1`,
looked up by timestamp (the feed has gaps, so "h rows later" is not always h seconds later).
The label is Up / Down when |return| exceeds a threshold, and Flat otherwise. The threshold is
1.5 bps at 10 s (as in the original project) and 3 bps (the round-trip cost) for longer
horizons, so an Up label is a move that is still profitable after paying to enter and exit.

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
3. Trading rules, all chosen by **validation** Sharpe and then applied once to test:
   - *argmax*: trade every prediction;
   - *threshold θ*: short if P(Down) > θ, long if P(Up) > θ, otherwise flat;
   - *hysteresis (enter / exit)*: open a position when its class probability > enter, keep it
     while that probability stays ≥ exit, and reverse directly on a strong opposite signal.
     The lower exit bar avoids paying to close and reopen on every small dip in confidence.
4. Backtest on non-overlapping steps of length h, cost = 1.5 bps × |Δposition| (so a reversal
   pays twice). Sharpe is annualised with √(steps per year) (≈ 1776 at 10 s). Compare these
   Sharpe values across models only, not with daily-frequency strategies.
5. Baselines: always-Flat (`flat_acc`) and LightGBM on the current second's features.

## Results

Full data, test split = 2021-04-17 15:00 → 04-19 09:54 UTC (154,544 targets, 15,457
non-overlapping 10-second steps). Fee 1.5 bps. θ is chosen on validation and applied once to
test. Run on a Colab T4 with `notebooks/colab_runner.ipynb`.

| Model | Feature set | Test acc | Dir-F1 | Argmax return | Argmax Sharpe | θ (val) | θ return | θ Sharpe | θ turnover |
|---|---|---|---|---|---|---|---|---|---|
| CNN | full | 0.4447 | 0.4543 | −40.05% | −96.6 | 0.850 | +0.72% | 16.8 | 28 |
| CNN | no_pca | 0.4444 | 0.4478 | −40.27% | −99.2 | 0.850 | −0.35% | −9.7 | 42 |
| CNN | no_market | 0.4429 | 0.4374 | −42.23% | −104.8 | 0.775 | −0.71% | −17.0 | 62 |
| LSTM | full | 0.4338 | 0.4470 | −48.55% | −124.7 | 0.800 | −0.16% | −2.2 | 96 |
| LSTM | no_pca | 0.4342 | 0.4490 | −46.85% | −116.9 | 0.850 | +0.27% | 18.2 | 22 |
| LSTM | no_market | 0.4363 | 0.4451 | −46.52% | −115.7 | 0.825 | −0.12% | −3.0 | 54 |
| Transformer | full | 0.4373 | 0.4660 | −42.80% | −98.6 | 0.825 | +0.32% | 13.2 | 152 |
| Transformer | no_pca | 0.4340 | 0.4636 | −39.69% | −89.5 | 0.850 | −0.06% | −8.7 | 24 |
| Transformer | no_market | 0.4361 | 0.4559 | −42.48% | −101.3 | 0.850 | +0.05% | 5.6 | 30 |

Buy & hold over the same test window: −6.52% (max drawdown −14.8%).

**Takeaways**

- **Accuracy is at the majority-class level.** Flat is 43.1% of test labels, so 0.43–0.44
  accuracy is no better than always predicting Flat. The directional F1 (~0.45–0.47) shows some
  signal, and the Transformer scores best on it.
- **Trading every prediction loses 40–49% after fees.** The edge per 10-second step is much
  smaller than the 1.5 bps cost of changing position.
- **With a high confidence threshold the models almost stop trading.** θ = 0.78–0.85, with
  22–152 position changes over 1.8 days and returns between −0.7% and +0.7%. With so few trades
  the Sharpe values are not statistically meaningful, and no feature set wins consistently.
- **The Transformer overfits fast.** Its best validation epoch is 2–4.

### Longer horizons, LightGBM baseline and the hysteresis rule

Full data, `full` feature set unless noted (LightGBM also on the other two sets, see
[`results/horizons_summary.csv`](results/horizons_summary.csv)). Hysteresis = enter/exit
thresholds chosen on validation. Buy & hold over the test window ≈ −6.6% at every horizon.

| Horizon | Model | Test acc | Always-Flat acc | Dir-F1 | Argmax return | Hysteresis return | Hyst. turnover |
|---|---|---|---|---|---|---|---|
| 10 s | CNN | 0.445 | 0.431 | 0.454 | −40.1% | **+26.8%** | 2,249 |
| 10 s | LSTM | 0.434 | 0.431 | 0.447 | −48.6% | +4.5% | 428 |
| 10 s | Transformer | 0.437 | 0.431 | 0.466 | −42.8% | +3.5% | 284 |
| 10 s | LightGBM | 0.409 | 0.431 | 0.465 | −39.3% | +6.3% | 974 |
| 30 s | CNN | 0.395 | 0.359 | 0.413 | −2.4% | +0.1% | 8 |
| 30 s | LSTM | 0.390 | 0.359 | 0.426 | −16.0% | +0.7% | 16 |
| 30 s | Transformer | 0.399 | 0.359 | 0.437 | −8.5% | +1.0% | 8 |
| 30 s | LightGBM | 0.390 | 0.359 | 0.446 | −3.0% | +13.3% | 803 |
| 60 s | CNN | 0.394 | 0.244 | 0.436 | −12.3% | +8.3% | 1,059 |
| 60 s | LSTM | 0.400 | 0.244 | 0.445 | −7.5% | +4.0% | 156 |
| 60 s | Transformer | 0.382 | 0.244 | 0.416 | +1.9% | −0.3% | 648 |
| 60 s | LightGBM | 0.419 | 0.244 | 0.466 | −2.4% | +8.6% | 28 |
| 300 s | CNN | 0.373 | 0.107 | 0.420 | −2.6% | −4.4% | 37 |
| 300 s | LSTM | 0.406 | 0.107 | 0.447 | −1.2% | −3.9% | 9 |
| 300 s | Transformer | 0.323 | 0.107 | 0.372 | −11.1% | −12.3% | 21 |
| 300 s | LightGBM | 0.443 | 0.107 | 0.476 | +18.0% | −2.4% | 18 |

**Robustness of the hysteresis results** ([`scripts/analyze_preds.py`](scripts/analyze_preds.py),
[`results/hysteresis_robustness.csv`](results/hysteresis_robustness.csv)). For each run: the
P&L split into long and short positions, the P&L per day, and the test return over the 3×3
grid of neighbouring (enter, exit) values.

- **CNN at 10 s holds up best.** Both sides make money (long +15%, short +23%, before
  −14% fees). It is positive on each of the three test days (+3.6% / +16.9% / +3.5%), and every
  neighbouring (enter, exit) pair stays positive (+14% … +49%). LightGBM `no_pca` at 10 s is
  also positive in every neighbour (+0.8% … +48%). LSTM and Transformer at 10 s are small but
  positive on every day.
- **The 30 s and 60 s gains are mostly the 04-18 sell-off.** Most of their P&L comes from
  short positions on 04-18, and nearby (enter, exit) values often turn negative (for example,
  LightGBM `full` at 30 s ranges from −12% to +15%).
- **Longer horizons did not help.** Accuracy rises above the always-Flat baseline only
  because Flat becomes rare, and Dir-F1 barely moves. At 300 s every hysteresis result is
  negative.
- **LightGBM matches the deep models on classification.** Its Dir-F1 is equal or better at
  every horizon. The deep models' advantage, mainly the CNN at 10 s, shows up only in how
  well their probabilities rank trades.

**Caveats** (addressed in the walk-forward section below). There is a single 1.8-day test window, dominated by one sell-off. Many
configurations were compared on it (24 here, plus the earlier runs), so the best one is
optimistically selected. Execution is assumed at the mid-price with no latency, and 2,249
position changes in 1.8 days is very active. Walk-forward evaluation over several windows
and multiple seeds is needed before calling any of these results an edge.

### Walk-forward evaluation and reality checks

To test whether the 10-second CNN + hysteresis result holds beyond one test window, the
model is retrained on **rolling folds**: 5 days train → 1 day validation → 1 day test, moved
forward one day at a time (6 test days, 04-13 → 04-19). The PCA is refitted on each fold's
training days, the trading rule is picked on each fold's validation day, and the CNN is
trained with 3 seeds plus a seed-averaged ensemble. Files:
[`results/walk_forward_cnn.csv`](results/walk_forward_cnn.csv),
[`results/walk_forward_lgbm.csv`](results/walk_forward_lgbm.csv).

| Hysteresis return per test day | 04-13 | 04-14 | 04-15 | 04-16 | 04-17 | 04-18 | mean | days > 0 |
|---|---|---|---|---|---|---|---|---|
| CNN, 3-seed ensemble | +4.2% | +3.1% | +9.8% | +9.0% | +6.2% | +10.0% | +7.0% | 6 / 6 |
| CNN, single seeds (18 runs) | | | | | | | +7.7% | 18 / 18 |
| LightGBM `no_pca` (1 seed) | +3.2% | +8.9% | +7.9% | +5.9% | −0.8% | +11.3% | +6.1% | 5 / 6 |
| LightGBM `full` (1 seed) | −2.2% | +1.0% | +12.4% | +9.3% | −2.7% | +10.8% | +4.8% | 4 / 6 |
| Buy & hold | +2.3% | −3.0% | −3.0% | +0.7% | −10.9% | +4.5% | −1.6% | |

On paper this looks like a strong, consistent edge. Two checks show it is **not tradable**:

**1. Execution delay** ([`scripts/check_walk_forward.py`](scripts/check_walk_forward.py)).
The backtest assumes that we trade at the mid-price of the same snapshot the features were
computed from. Delaying execution by d seconds (same model, same thresholds):

| Mean daily return | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| CNN ensemble | +7.0% | +0.9% | −1.9% | −4.0% | −6.8% |
| CNN single seeds | +7.7% | +2.3% | −1.0% | −2.9% | −4.9% |
| LightGBM | +5.2% | −4.3% | −4.6% | −6.6% | −9.5% |

Per day: [`results/walk_forward_delay_cnn_ensemble.csv`](results/walk_forward_delay_cnn_ensemble.csv).
The same check on the main 70/15/15 split (test 04-17 → 04-19, 10 s, hysteresis) covers all
four models ([`results/main_split_delay_h10.csv`](results/main_split_delay_h10.csv)):

| Test return | 0 s | 1 s | 2 s | 3 s | 5 s |
|---|---|---|---|---|---|
| CNN | +26.8% | +4.7% | −5.2% | −9.4% | −10.1% |
| LSTM | +4.5% | +1.4% | +0.4% | −0.5% | −2.1% |
| Transformer | +3.5% | +0.9% | +0.0% | −1.0% | −1.6% |
| LightGBM `no_pca` | +17.1% | −2.5% | −9.8% | −13.4% | −12.3% |

LSTM and Transformer trade less (higher enter thresholds), so they lose less when delayed,
but they show the same pattern: most of the profit is gone after 1 s and it is negative by 3 s.

The whole profit sits in the first ~1 second after each snapshot. The models do predict the
next move of the mid-price, but that move comes almost immediately. Capturing it would need
sub-second execution at the mid, with no queueing, no adverse selection and a 1.5 bps fee.
The 1-second snapshot data cannot show whether that is achievable.

**2. BTC/ETH timestamp alignment.** BTC and ETH snapshots are joined on the floored second.
Within a matched second the ETH snapshot is taken **later** than the BTC one 24.5% of the time
(up to 0.74 s later; 19.7% of seconds are > 0.1 s later). The ETH features therefore sometimes
contain information from slightly after the BTC mid they are compared with. This is a small
look-ahead leak in every feature set that uses market features (`full`, `no_pca`). A 1-second
execution delay also removes this leak, which is one more reason to read the delayed numbers
as the realistic ones.

**Conclusion.** Under realistic execution assumptions, none of the models or trading rules
here has a demonstrated edge after a 1.5 bps fee. What the models learn is genuine
sub-second order-book information (they predict direction better than chance, and the
hysteresis rule turns that into profit when execution is instantaneous). That signal decays
within about a second, faster than this data can support trading on.

<details>
<summary>Development run: first ~1.2 days (<code>SUBSET_ROWS = 100_000</code>)</summary>

Test split = 2021-04-08 11:09 → 15:19 UTC (14,958 targets, 1,496 steps). Buy & hold: +2.1%.

| Model | Feature set | Test acc | Dir-F1 | Argmax return | θ (val) | θ return | θ Sharpe | θ turnover |
|---|---|---|---|---|---|---|---|---|
| CNN | full | 0.4578 | 0.4165 | −6.74% | 0.800 | +0.09% | 31.3 | 20 |
| CNN | no_pca | 0.4596 | 0.4178 | −6.69% | 0.775 | −0.02% | −15.3 | 18 |
| CNN | no_market | 0.4521 | 0.4218 | −6.04% | 0.700 | −0.07% | −22.3 | 30 |
| LSTM | full | 0.3983 | 0.3966 | −8.28% | 0.675 | −0.45% | −71.9 | 128 |
| LSTM | no_pca | 0.4426 | 0.3898 | −7.40% | 0.775 | −0.10% | −106.3 | 10 |
| LSTM | no_market | 0.4193 | 0.4012 | −7.46% | 0.775 | −0.07% | −31.3 | 18 |
| Transformer | full | 0.4487 | 0.4169 | −7.17% | 0.800 | −0.03% | −18.6 | 16 |
| Transformer | no_pca | 0.4569 | 0.4184 | −5.99% | 0.800 | +0.15% | 47.9 | 14 |
| Transformer | no_market | 0.4598 | 0.4199 | −5.85% | 0.750 | −0.05% | −44.7 | 8 |

</details>

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
