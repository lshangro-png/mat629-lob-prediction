"""Feature sets, sliding-window datasets and data loaders shared by all models."""
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

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


class WindowDataset(Dataset):
    """Sample i = (features of rows t-seq_len+1 .. t, label of row t), t = targets[i].

    `X` is shared between the train / val / test datasets; only the target rows
    differ. A window may reach back across a split boundary, which is fine: the
    features are backward-looking and labels never leave their own split.
    """

    def __init__(self, X, y, targets, seq_len):
        self.X, self.y, self.seq_len = X, y, seq_len
        self.targets = torch.as_tensor(targets, dtype=torch.long)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        t = int(self.targets[i])
        return self.X[t - self.seq_len + 1: t + 1], self.y[t]


def target_indices(df, split, seq_len):
    idx = np.flatnonzero(((df["split"] == split) & df["is_target"]).to_numpy())
    return idx[idx >= seq_len - 1]


def make_loaders(df, meta, feature_set, seq_len, batch_size, device, num_workers=2):
    cols = feature_columns(feature_set, meta["pca_cols"])
    X = torch.from_numpy(df[cols].to_numpy(dtype=np.float32, copy=True))
    y = torch.from_numpy((df["label"].to_numpy() + 1).astype(np.int64))

    targets = {s: target_indices(df, s, seq_len) for s in ["train", "val", "test"]}
    pin = device.type == "cuda"
    loaders = {
        s: DataLoader(WindowDataset(X, y, targets[s], seq_len),
                      batch_size=batch_size if s == "train" else batch_size * 2,
                      shuffle=(s == "train"), num_workers=num_workers, pin_memory=pin)
        for s in targets
    }

    counts = np.bincount(y[targets["train"]].numpy(), minlength=3).astype(float)
    class_w = torch.tensor(counts.sum() / (3 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)

    print(f"Feature set: {feature_set} ({FEATURE_SET_DESC[feature_set]}) | {len(cols)} features")
    print("Samples: " + " | ".join(f"{s} {len(t):,}" for s, t in targets.items()))
    print(f"Class weights (down/flat/up): {class_w.cpu().numpy().round(3)}")
    return loaders, targets, class_w, len(cols)
