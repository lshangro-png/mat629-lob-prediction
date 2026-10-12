"""Train + evaluate one model on several feature sets and horizons; resumable.

    python scripts/run_model.py --model cnn --subset 100000 --work-dir /content/drive/MyDrive/mat629-lob
    python scripts/run_model.py --model lgbm --horizons 10 30 60 300

Writes into <work-dir>:
    data/features[_<subset>].parquet                       (built on first use)
    results[/subset_<subset>]/h<h>/<model>_summary.csv      (updated after every run)
    results[/subset_<subset>]/h<h>/figures/<model>_<feature_set>.png
    results[/subset_<subset>]/h<h>/preds/<model>_<feature_set>.npz   (val/test probabilities)
(horizon, feature set) pairs already in a summary CSV are skipped, so a
disconnected run can simply be restarted.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pandas as pd  # noqa: E402

from lob import config as C  # noqa: E402
from lob.data import FEATURE_SETS, load_dataset  # noqa: E402
from lob.experiment import run_experiment, save_summary  # noqa: E402
from lob.features import build_dataset  # noqa: E402
from lob.hparams import HPARAMS  # noqa: E402


def features_up_to_date(path, horizons):
    meta_path = os.path.splitext(path)[0] + "_meta.json"
    if not (os.path.exists(path) and os.path.exists(meta_path)):
        return False
    with open(meta_path) as f:
        return set(horizons) <= set(json.load(f).get("horizons", []))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(HPARAMS))
    p.add_argument("--subset", type=int, default=None, help="first N rows only (e.g. 100000 ≈ 1 day)")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--raw-dir", default=None, help="folder with BTC_1sec.csv / ETH_1sec.csv (default: kagglehub)")
    p.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    p.add_argument("--horizons", nargs="+", type=int, default=[C.HORIZON], choices=C.HORIZONS)
    p.add_argument("--epochs", type=int, default=None, help="override (for smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()

    tag = "" if a.subset is None else f"_{a.subset}"
    features = os.path.join(a.work_dir, "data", f"features{tag}.parquet")
    results_root = os.path.join(a.work_dir, "results" if a.subset is None
                                else f"results/subset_{a.subset}")

    if not features_up_to_date(features, a.horizons):
        build_dataset(a.raw_dir, a.subset, out_path=features)
    df, meta = load_dataset(features)

    hp = {k: (dict(v) if isinstance(v, dict) else v) for k, v in HPARAMS[a.model].items()}
    if a.epochs is not None and "train_kwargs" in hp:
        hp["train_kwargs"]["epochs"] = a.epochs
    device = None
    if a.model != "lgbm":   # torch and LightGBM clash in one process on macOS
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | rows: {len(df):,}")

    for h in a.horizons:
        results = os.path.join(results_root, f"h{h}")
        summary_csv = os.path.join(results, f"{a.model}_summary.csv")
        rows = pd.read_csv(summary_csv).to_dict("records") if os.path.exists(summary_csv) else []
        done = {r["feature_set"] for r in rows}
        for fs in a.feature_sets:
            if fs in done:
                print(f"Skipping horizon {h}s / {fs} (already in {summary_csv})")
                continue
            summary, _, _ = run_experiment(
                df, meta, a.model, fs, hp, device, horizon=h, num_workers=a.num_workers,
                fig_path=os.path.join(results, "figures", f"{a.model}_{fs}.png"),
                pred_path=os.path.join(results, "preds", f"{a.model}_{fs}.npz"))
            rows.append(summary)
            save_summary(rows, a.model, results)
            print(f"Saved → {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
