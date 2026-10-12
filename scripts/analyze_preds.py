"""Robustness checks for the hysteresis rule, from saved predictions (no retraining).

    python scripts/analyze_preds.py --work-dir /content/drive/MyDrive/mat629-lob

For every results/h<h>/<model>_summary.csv row it reloads preds/<model>_<fs>.npz,
rebuilds the test positions with the (enter, exit) chosen on validation, and prints
  * the return split into long and short positions,
  * the return per calendar day (UTC),
  * test returns for neighbouring (enter, exit) values (sensitivity),
  * the share of time spent short vs long.
A result driven by shorting the 2021-04-18 sell-off shows up as most of the profit
coming from short positions on that day.
"""
import argparse
import glob
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lob import config as C  # noqa: E402
from lob.backtest import backtest, nonoverlap_steps, positions_hysteresis  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work-dir", default=".")
    p.add_argument("--subset", type=int, default=None)
    a = p.parse_args()

    tag = "" if a.subset is None else f"_{a.subset}"
    root = os.path.join(a.work_dir, "results" if a.subset is None else f"results/subset_{a.subset}")
    cols = ["time"] + [f"future_return_{h}" for h in C.HORIZONS]
    df = pd.read_parquet(os.path.join(a.work_dir, "data", f"features{tag}.parquet"), columns=cols)
    day = df["time"].dt.strftime("%m-%d").to_numpy()

    rows = []
    for csv in sorted(glob.glob(os.path.join(root, "h*", "*_summary.csv"))):
        for r in pd.read_csv(csv).to_dict("records"):
            h, model, fs = int(r["horizon"]), r["model"], r["feature_set"]
            npz = os.path.join(os.path.dirname(csv), "preds", f"{model}_{fs}.npz")
            if not os.path.exists(npz):
                continue
            d = np.load(npz)
            st = nonoverlap_steps(d["test_rows"], h)
            rows_t = d["test_rows"][st]
            probs = d["test_probs"][st]
            ret = df[f"future_return_{h}"].to_numpy()[rows_t]
            enter, exit_ = r["hy_enter"], r["hy_exit"]

            pos = positions_hysteresis(probs, enter, exit_)
            trades = np.abs(np.diff(pos, prepend=0))
            net = pos * ret - C.FEE * trades
            out = {"horizon": h, "model": model, "feature_set": fs,
                   "enter": enter, "exit": exit_,
                   "total": backtest(pos, ret, C.FEE, h)["total_return"],
                   "long_pnl": net[pos > 0].sum(), "short_pnl": net[pos < 0].sum(),
                   "flat_fees": net[pos == 0].sum(),
                   "pct_long": (pos > 0).mean(), "pct_short": (pos < 0).mean()}
            for dd in sorted(set(day[rows_t])):
                out[f"day_{dd}"] = net[day[rows_t] == dd].sum()
            sens = []
            for de in (-0.05, 0.0, 0.05):
                for dx in (-0.05, 0.0, 0.05):
                    e2, x2 = round(enter + de, 2), round(exit_ + dx, 2)
                    if 0 <= x2 < e2 < 1:
                        sens.append(backtest(positions_hysteresis(probs, e2, x2), ret, C.FEE, h)
                                    ["total_return"])
            out["sens_min"], out["sens_median"], out["sens_max"] = (
                np.min(sens), np.median(sens), np.max(sens))
            rows.append(out)

    res = pd.DataFrame(rows).sort_values(["horizon", "model", "feature_set"])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("Sums of per-step net returns (not compounded); sens_* = test total return over the "
          "3x3 grid of neighbouring (enter, exit) values.")
    print(res.round(4).to_csv(index=False))


if __name__ == "__main__":
    main()
