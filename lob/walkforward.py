"""Walk-forward (rolling-window) evaluation.

The series is cut into 24-hour blocks from its first timestamp. Fold k trains on
blocks [k, k + train_days), validates on the next block and tests on the one
after, so every fold is tested on a different, later day:

    fold 0: train ■■■■■ val □ test ▲
    fold 1:   train ■■■■■ val □ test ▲
    ...

For each fold the train / val / test labels are rewritten in place (with the
same purging as the main split) and the LOB PCA is refitted on that fold's
training rows only.
"""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as C
from .features import LOB_COLS


def make_folds(df, train_days=5):
    t0 = df["time"].iloc[0]
    day = pd.Timedelta(days=1)
    n_blocks = int(np.ceil((df["time"].iloc[-1] - t0) / day))
    folds = []
    for k in range(n_blocks - train_days - 1):
        train_start = t0 + k * day
        val_start = train_start + train_days * day
        test_start = val_start + day
        folds.append({"fold": k, "train_start": train_start, "val_start": val_start,
                      "test_start": test_start, "test_end": test_start + day})
    return folds


def apply_fold(df, meta, fold, horizon=C.HORIZON):
    """Rewrite df['split'] / df[f'is_target_{horizon}'] / PCA columns for one fold."""
    t = df["time"]
    split = np.full(len(df), "none", dtype=object)
    split[((t >= fold["train_start"]) & (t < fold["val_start"])).to_numpy()] = "train"
    split[((t >= fold["val_start"]) & (t < fold["test_start"])).to_numpy()] = "val"
    split[((t >= fold["test_start"]) & (t < fold["test_end"])).to_numpy()] = "test"
    df["split"] = split

    # purging: a target's label window must end inside its own split
    label_end = (t + pd.Timedelta(seconds=horizon)).reset_index(drop=True)
    has_label = df[f"future_return_{horizon}"].notna().to_numpy()
    is_target = np.zeros(len(df), dtype=bool)
    for s, end in [("train", fold["val_start"]), ("val", fold["test_start"]),
                   ("test", fold["test_end"])]:
        m = split == s
        is_target[m] = has_label[m] & (label_end[m] < end).to_numpy()
    df[f"is_target_{horizon}"] = is_target

    # PCA basis from this fold's training rows only; keep the same number of components
    train = split == "train"
    scaler = StandardScaler().fit(df.loc[train, LOB_COLS])
    pca = PCA(n_components=len(meta["pca_cols"]), random_state=C.SEED)
    pca.fit(scaler.transform(df.loc[train, LOB_COLS]))
    df[meta["pca_cols"]] = pca.transform(scaler.transform(df[LOB_COLS])).astype("float32")

    counts = {s: int(((split == s) & df[f"is_target_{horizon}"].to_numpy()).sum())
              for s in ["train", "val", "test"]}
    print(f"Fold {fold['fold']}: train {fold['train_start']:%m-%d %H:%M} → "
          f"val {fold['val_start']:%m-%d %H:%M} → test {fold['test_start']:%m-%d %H:%M}"
          f"–{fold['test_end']:%m-%d %H:%M} | targets {counts}")
    return df
