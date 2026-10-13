"""Walk-forward evaluation over rolling 1-day test windows, with several seeds.

    python scripts/walk_forward.py --model cnn --work-dir /content/drive/MyDrive/mat629-lob
    python scripts/walk_forward.py --model lgbm --work-dir ...
    python scripts/walk_forward.py --report --work-dir ...      # aggregate table only

Per (fold, feature set, seed) the trading rules are chosen on that fold's
validation day and applied once to its test day. After all seeds of a fold, an
ensemble (seed-averaged probabilities) is evaluated as seed = "ens".

Writes results/walk_forward/h<h>/<model>_runs.csv (resumable) and
results/walk_forward/h<h>/preds/<model>_<fs>_fold<k>_seed<s>.npz.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lob import config as C  # noqa: E402
from lob.data import load_dataset  # noqa: E402
from lob.experiment import evaluate, run_experiment  # noqa: E402
from lob.features import build_dataset  # noqa: E402
from lob.hparams import HPARAMS  # noqa: E402
from lob.walkforward import apply_fold, make_folds  # noqa: E402


def report(out_dir):
    files = [os.path.join(out_dir, f) for f in sorted(os.listdir(out_dir)) if f.endswith("_runs.csv")]
    if not files:
        print("No results yet.")
        return
    runs = pd.concat(map(pd.read_csv, files))
    runs["kind"] = np.where(runs["seed"].astype(str) == "ens", "ensemble", "single seed")
    g = runs.groupby(["model", "feature_set", "kind"])
    agg = pd.DataFrame({
        "n_runs": g.size(),
        "dir_f1": g["test_dir_f1"].mean(),
        "argmax_mean": g["argmax_return"].mean(),
        "th_mean": g["th_return"].mean(),
        "hy_mean": g["hy_return"].mean(),
        "hy_median": g["hy_return"].median(),
        "hy_worst": g["hy_return"].min(),
        "hy_best": g["hy_return"].max(),
        "hy_pos_share": g["hy_return"].apply(lambda r: (r > 0).mean()),
        "hy_turnover": g["hy_turnover"].mean(),
        "bh_mean": g["bh_return"].mean(),
    })
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\nPer test day (return over one day, fee included):")
    cols = ["model", "feature_set", "fold", "test_day", "seed", "test_dir_f1", "argmax_return",
            "th_return", "hy_return", "hy_turnover", "bh_return"]
    print(runs[cols].sort_values(["model", "feature_set", "fold", "seed"]).round(4).to_csv(index=False))
    print("Aggregate over folds:")
    print(agg.round(4).to_csv())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=list(HPARAMS))
    p.add_argument("--work-dir", default=".")
    p.add_argument("--raw-dir", default=None)
    p.add_argument("--feature-sets", nargs="+", default=["full"])
    p.add_argument("--horizon", type=int, default=10, choices=C.HORIZONS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--train-days", type=int, default=5)
    p.add_argument("--folds", nargs="+", type=int, default=None, help="only these fold numbers")
    p.add_argument("--epochs", type=int, default=None, help="override (for smoke tests)")
    p.add_argument("--subset", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--report", action="store_true", help="only print the aggregate table")
    a = p.parse_args()

    tag = "" if a.subset is None else f"_{a.subset}"
    out_dir = os.path.join(a.work_dir, "results" if a.subset is None else f"results/subset_{a.subset}",
                           "walk_forward", f"h{a.horizon}")
    os.makedirs(os.path.join(out_dir, "preds"), exist_ok=True)
    if a.report:
        report(out_dir)
        return
    if a.model is None:
        p.error("--model is required unless --report")

    features = os.path.join(a.work_dir, "data", f"features{tag}.parquet")
    if not os.path.exists(features):
        build_dataset(a.raw_dir, a.subset, out_path=features)
    df, meta = load_dataset(features)

    hp = {k: (dict(v) if isinstance(v, dict) else v) for k, v in HPARAMS[a.model].items()}
    if a.epochs is not None and "train_kwargs" in hp:
        hp["train_kwargs"]["epochs"] = a.epochs
    device = None
    if a.model != "lgbm":   # torch and LightGBM clash in one process on macOS
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs_csv = os.path.join(out_dir, f"{a.model}_runs.csv")
    rows = pd.read_csv(runs_csv).to_dict("records") if os.path.exists(runs_csv) else []
    done = {(int(r["fold"]), r["feature_set"], str(r["seed"])) for r in rows}

    folds = make_folds(df, a.train_days)
    print(f"{len(folds)} folds, train {a.train_days} days / val 1 day / test 1 day | device {device}")
    for fold in folds:
        if a.folds is not None and fold["fold"] not in a.folds:
            continue
        todo = [(fs, s) for fs in a.feature_sets for s in a.seeds + ["ens"]
                if (fold["fold"], fs, str(s)) not in done]
        if not todo:
            continue
        df = apply_fold(df, meta, fold, a.horizon)
        info = {"fold": fold["fold"], "test_day": f"{fold['test_start']:%m-%d %H:%M}"}
        for fs in a.feature_sets:
            pred_paths = []
            for seed in a.seeds:
                pred = os.path.join(out_dir, "preds", f"{a.model}_{fs}_fold{fold['fold']}_seed{seed}.npz")
                pred_paths.append(pred)
                if (fold["fold"], fs, str(seed)) in done:
                    continue
                summary, _, _ = run_experiment(df, meta, a.model, fs, hp, device, a.horizon,
                                               num_workers=a.num_workers, plot=False,
                                               pred_path=pred, seed=seed)
                rows.append({**info, **summary})
                pd.DataFrame(rows).to_csv(runs_csv, index=False)

            if (fold["fold"], fs, "ens") not in done:
                d = [np.load(pp) for pp in pred_paths]
                p_val = np.mean([x["val_probs"] for x in d], axis=0)
                p_test = np.mean([x["test_probs"] for x in d], axis=0)
                targets = {"val": d[0]["val_rows"], "test": d[0]["test_rows"]}
                y_test = (df[f"label_{a.horizon}"].to_numpy()[targets["test"]] + 1).astype(np.int64)
                print(f"\n--- Seed ensemble ({len(d)} seeds) | fold {fold['fold']} | {fs} ---")
                summary, _ = evaluate(df, p_val, p_test, y_test, targets, a.horizon)
                rows.append({**info, "model": a.model, "feature_set": fs, "horizon": a.horizon,
                             "seed": "ens", **summary})
                pd.DataFrame(rows).to_csv(runs_csv, index=False)
            print(f"Saved → {runs_csv}", flush=True)

    report(out_dir)


if __name__ == "__main__":
    main()
