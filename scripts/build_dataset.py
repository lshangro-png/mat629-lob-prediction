"""Build the feature table once and save it as parquet.

    python scripts/build_dataset.py                      # downloads via kagglehub
    python scripts/build_dataset.py --data-dir data/raw  # use local BTC_1sec.csv / ETH_1sec.csv
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lob.features import build_dataset  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default=None,
                   help="folder containing BTC_1sec.csv and ETH_1sec.csv "
                        "(default: download with kagglehub)")
    p.add_argument("--out", default="data/features.parquet")
    p.add_argument("--nrows", type=int, default=None, help="only read the first N rows (for quick tests)")
    a = p.parse_args()
    build_dataset(a.data_dir, a.nrows, a.out)
