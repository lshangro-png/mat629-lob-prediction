"""Feature sets, labels and target rows shared by all models (no torch here,
so the LightGBM baseline can run without it)."""
import json
import os

import numpy as np
import pandas as pd

from . import config as C
from .features import CLASSIC_COLS, LOB_COLS, MARKET_COLS

FEATURE_SETS = ["full", "no_pca", "no_market"]
FEATURE_SET_DESC = {
    "full": "PCA(LOB) + market + classic",
    "no_pca": "raw LOB + market + classic",
    "no_market": "PCA(LOB) + classic",
}
CLASS_NAMES = ["Down", "Flat", "Up"]   # model classes 0 / 1 / 2 ↔ labels -1 / 0 / +1


def load_dataset(path):
    df = pd.read_parquet(path)
    with open(os.path.splitext(path)[0] + "_meta.json") as f:
        meta = json.load(f)
    return df, meta


def feature_columns(feature_set, pca_cols):
    groups = {
        "full": [pca_cols, MARKET_COLS, CLASSIC_COLS],
        "no_pca": [LOB_COLS, MARKET_COLS, CLASSIC_COLS],
        "no_market": [pca_cols, CLASSIC_COLS],
    }
    if feature_set not in groups:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    return [c for g in groups[feature_set] for c in g]


def target_indices(df, split, seq_len=1, horizon=C.HORIZON):
    idx = np.flatnonzero(((df["split"] == split) & df[f"is_target_{horizon}"]).to_numpy())
    return idx[idx >= seq_len - 1]


def class_labels(df, horizon=C.HORIZON):
    """Model classes 0 / 1 / 2 (Down / Flat / Up) for every row."""
    return (df[f"label_{horizon}"].to_numpy() + 1).astype(np.int64)
