"""Train + evaluate one model on every feature set; resumable.

    python scripts/run_model.py --model cnn --subset 100000 --work-dir /content/drive/MyDrive/mat629-lob

Writes into <work-dir>:
    data/features[_<subset>].parquet              (built on first use)
    results[/subset_<subset>]/<model>_summary.csv  (updated after every feature set)
    results[/subset_<subset>]/figures/<model>_<feature_set>.png
Feature sets already in the summary CSV are skipped, so a disconnected run can
simply be restarted.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib  # noqa: E402
matplotlib.use("Agg")

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from lob.data import FEATURE_SETS, load_dataset  # noqa: E402
from lob.experiment import run_experiment, save_summary  # noqa: E402
from lob.features import build_dataset  # noqa: E402
from lob.hparams import HPARAMS  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=list(HPARAMS))
    p.add_argument("--subset", type=int, default=None, help="first N rows only (e.g. 100000 ≈ 1 day)")
    p.add_argument("--work-dir", default=".")
    p.add_argument("--raw-dir", default=None, help="folder with BTC_1sec.csv / ETH_1sec.csv (default: kagglehub)")
    p.add_argument("--feature-sets", nargs="+", default=FEATURE_SETS)
    p.add_argument("--epochs", type=int, default=None, help="override (for smoke tests)")
    p.add_argument("--num-workers", type=int, default=2)
    a = p.parse_args()

    tag = "" if a.subset is None else f"_{a.subset}"
    features = os.path.join(a.work_dir, "data", f"features{tag}.parquet")
    results = os.path.join(a.work_dir, "results" if a.subset is None else f"results/subset_{a.subset}")
    summary_csv = os.path.join(results, f"{a.model}_summary.csv")

    if not os.path.exists(features):
        build_dataset(a.raw_dir, a.subset, out_path=features)
    df, meta = load_dataset(features)

    hp = HPARAMS[a.model]
    train_kwargs = dict(hp["train_kwargs"])
    if a.epochs is not None:
        train_kwargs["epochs"] = a.epochs

    rows = pd.read_csv(summary_csv).to_dict("records") if os.path.exists(summary_csv) else []
    done = {r["feature_set"] for r in rows}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | rows: {len(df):,} | already done: {sorted(done) or '-'}")

    for fs in a.feature_sets:
        if fs in done:
            print(f"Skipping {fs} (already in {summary_csv})")
            continue
        summary, _, _ = run_experiment(
            df, meta, a.model, fs, hp["model_kwargs"], train_kwargs, hp["seq_len"],
            hp["batch_size"], device, num_workers=a.num_workers,
            fig_path=os.path.join(results, "figures", f"{a.model}_{fs}.png"))
        rows.append(summary)
        save_summary(rows, a.model, results)
        print(f"Saved → {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
