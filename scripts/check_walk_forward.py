"""Sanity checks for the walk-forward results (no retraining).

    python scripts/check_walk_forward.py --model cnn --work-dir /content/drive/MyDrive/mat629-lob

1. Execution delay. The backtest assumes we trade at the mid of the same
   snapshot the features come from. Here the position for the step starting at
   row t is decided from the predictions of row t - d (d seconds earlier), with
   the (enter, exit) chosen on each fold's validation day. If the profit
   disappears at d = 1-2 s, the edge cannot be captured in practice.
2. BTC/ETH timestamp alignment (needs the raw CSVs; downloaded with kagglehub
   unless --raw-dir is given). BTC and ETH snapshots are joined on the floored
   second; if the ETH snapshot is usually taken later within that second, the
   ETH features peek slightly into the future relative to the BTC mid.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lob import config as C  # noqa: E402
from lob.backtest import backtest, nonoverlap_steps, positions_hysteresis  # noqa: E402
from lob.features import find_data_dir  # noqa: E402


def delay_check(work_dir, model, horizon, delays):
    out_dir = os.path.join(work_dir, "results", "walk_forward", f"h{horizon}")
    runs = pd.read_csv(os.path.join(out_dir, f"{model}_runs.csv"))
    ret_all = pd.read_parquet(os.path.join(work_dir, "data", "features.parquet"),
                              columns=[f"future_return_{horizon}"]).iloc[:, 0].to_numpy()
    rows = []
    for r in runs.to_dict("records"):
        seed = str(r["seed"])
        tag = f"{model}_{r['feature_set']}_fold{int(r['fold'])}"
        if seed == "ens":
            seeds = sorted({str(s) for s in runs.loc[(runs["fold"] == r["fold"])
                                                       & (runs["feature_set"] == r["feature_set"]), "seed"]
                            if str(s) != "ens"})
            d = [np.load(os.path.join(out_dir, "preds", f"{tag}_seed{s}.npz")) for s in seeds]
            test_rows, probs = d[0]["test_rows"], np.mean([x["test_probs"] for x in d], axis=0)
        else:
            d = np.load(os.path.join(out_dir, "preds", f"{tag}_seed{seed}.npz"))
            test_rows, probs = d["test_rows"], d["test_probs"]

        prob_of_row = dict(zip(test_rows.tolist(), range(len(test_rows))))
        out = {"fold": int(r["fold"]), "test_day": r["test_day"], "seed": seed,
               "enter": r["hy_enter"], "exit": r["hy_exit"]}
        for dl in delays:
            # steps whose delayed prediction row exists
            ok = np.array([t - dl in prob_of_row for t in test_rows])
            rows_ok = test_rows[ok]
            st = nonoverlap_steps(rows_ok, horizon)
            step_rows = rows_ok[st]
            p = probs[[prob_of_row[t - dl] for t in step_rows]]
            pos = positions_hysteresis(p, r["hy_enter"], r["hy_exit"])
            out[f"delay_{dl}s"] = backtest(pos, ret_all[step_rows], C.FEE, horizon)["total_return"]
        rows.append(out)
    res = pd.DataFrame(rows)
    print(f"\n1. Hysteresis test return with execution delayed by d seconds ({model}):")
    print(res.round(4).to_csv(index=False))
    for kind, m in [("single seed", res["seed"] != "ens"), ("ensemble", res["seed"] == "ens")]:
        sub = res.loc[m, [c for c in res.columns if c.startswith("delay_")]]
        print(f"{kind:<12} mean: " + " | ".join(f"{c} {v:+.4f}" for c, v in sub.mean().items())
              + "   share > 0: " + " | ".join(f"{(sub[c] > 0).mean():.2f}" for c in sub.columns))


def timestamp_check(raw_dir, nrows=300_000):
    data_dir = find_data_dir(raw_dir)
    read = lambda f: pd.read_csv(os.path.join(data_dir, f), usecols=["system_time"], nrows=nrows)
    btc, eth = read("BTC_1sec.csv"), read("ETH_1sec.csv")
    for d in (btc, eth):
        d["t"] = pd.to_datetime(d["system_time"], utc=True)
        d["key"] = d["t"].dt.floor("s")
    m = btc.drop_duplicates("key", keep="last").merge(
        eth.drop_duplicates("key", keep="last"), on="key", suffixes=("_btc", "_eth"))
    lag = (m["t_eth"] - m["t_btc"]).dt.total_seconds()
    print(f"\n2. ETH minus BTC snapshot time within the same second ({len(m):,} matched seconds):")
    print(lag.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(4).to_string())
    print(f"share with ETH later than BTC: {(lag > 0).mean():.3f} | "
          f"share > 0.1 s later: {(lag > 0.1).mean():.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="cnn")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--horizon", type=int, default=10)
    p.add_argument("--delays", nargs="+", type=int, default=[0, 1, 2, 3, 5])
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--skip-timestamps", action="store_true")
    a = p.parse_args()
    delay_check(a.work_dir, a.model, a.horizon, a.delays)
    if not a.skip_timestamps:
        timestamp_check(a.raw_dir)


if __name__ == "__main__":
    main()
