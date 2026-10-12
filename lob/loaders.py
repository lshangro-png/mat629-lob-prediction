"""Sliding-window datasets and data loaders for the PyTorch models."""
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from . import config as C
from .data import FEATURE_SET_DESC, class_labels, feature_columns, target_indices


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


def make_loaders(df, meta, feature_set, seq_len, batch_size, device, num_workers=2,
                 horizon=C.HORIZON):
    cols = feature_columns(feature_set, meta["pca_cols"])
    X = torch.from_numpy(df[cols].to_numpy(dtype=np.float32, copy=True))
    y = torch.from_numpy(class_labels(df, horizon))

    targets = {s: target_indices(df, s, seq_len, horizon) for s in ["train", "val", "test"]}
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

    print(f"Feature set: {feature_set} ({FEATURE_SET_DESC[feature_set]}) | {len(cols)} features "
          f"| horizon {horizon}s")
    print("Samples: " + " | ".join(f"{s} {len(t):,}" for s, t in targets.items()))
    print(f"Class weights (down/flat/up): {class_w.cpu().numpy().round(3)}")
    return loaders, targets, class_w, len(cols)
