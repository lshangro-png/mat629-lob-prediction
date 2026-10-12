"""Raw BTC/ETH order-book snapshots → model-ready feature table.

Pipeline (see `build_dataset`):
    load + align BTC/ETH on 1-second timestamps
    → market features → labels → classic LOB factors
    → backward-looking rolling z-score
    → chronological train / val / test split (with purging)
    → StandardScaler + PCA on LOB features, fitted on train rows only
"""
import gc
import json
import os

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from . import config as C

EPS = 1e-8

LOB_COLS = [f"{side}_{kind}_{i}"
            for i in range(C.N_LEVELS)
            for kind, side in [("distance", "bids"), ("distance", "asks"),
                               ("notional", "bids"), ("notional", "asks")]]

MARKET_COLS = [
    "spread", "btc_oimb_1",
    "btc_volatility_10s", "btc_volatility_60s",
    "btc_momentum_10s", "btc_momentum_60s",
    "eth_spread", "eth_oimb_1",
    "eth_volatility_10s", "eth_volatility_60s",
    "eth_momentum_10s", "eth_momentum_60s",
    "btc_eth_momentum_diff_10s", "btc_eth_momentum_diff_60s",
    "btc_eth_vol_ratio_10s",
]

CLASSIC_COLS = [
    "factor_queue_imbalance",
    "factor_ofi", "factor_ofi_10s", "factor_ofi_60s",
    "factor_kyle_lambda",
    "factor_depth_ratio", "factor_depth_imb", "factor_depth_ratio_smooth",
    "factor_tick_sign", "factor_tick_momentum_10s", "factor_tick_momentum_60s",
    "factor_amihud", "factor_amihud_60s",
]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load + align
# ══════════════════════════════════════════════════════════════════════════════
def find_data_dir(data_dir=None):
    """Use `data_dir` if given, otherwise download the Kaggle dataset."""
    if data_dir is not None:
        return data_dir
    import kagglehub
    return kagglehub.dataset_download(C.KAGGLE_DATASET)


def _read_snapshots(path, cols, nrows):
    # Prices / best-level distances stay float64 (needed to rebuild exact tick
    # prices for OFI); everything else is float32 to halve memory.
    keep64 = {"midpoint", "bids_distance_0", "asks_distance_0"}
    dtype = {c: ("float64" if c in keep64 else "float32")
             for c in cols if c != "system_time"}
    df = pd.read_csv(path, usecols=cols, dtype=dtype, nrows=nrows)
    df["time"] = pd.to_datetime(df.pop("system_time"), utc=True).dt.floor("s")
    return df.drop_duplicates("time", keep="last")


def load_raw(data_dir=None, nrows=None):
    """Load BTC (full book) and ETH (top of book), inner-joined on the second."""
    data_dir = find_data_dir(data_dir)
    btc = _read_snapshots(os.path.join(data_dir, "BTC_1sec.csv"),
                          ["system_time", "midpoint", "spread", *LOB_COLS], nrows)
    eth = _read_snapshots(os.path.join(data_dir, "ETH_1sec.csv"),
                          ["system_time", "midpoint", "spread",
                           "bids_notional_0", "asks_notional_0"], nrows)
    eth = eth.rename(columns=lambda c: c if c == "time" else f"eth_{c}")

    df = btc.merge(eth, on="time", how="inner").sort_values("time")
    df = df.reset_index(drop=True)
    print(f"BTC {len(btc):,} rows | ETH {len(eth):,} rows | merged {len(df):,} rows")
    print(f"Time range: {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. Features + labels
# ══════════════════════════════════════════════════════════════════════════════
def add_market_features(df):
    ws, wl = C.WINDOW_SHORT, C.WINDOW_LONG
    for asset, mid, bid0, ask0 in [
        ("btc", "midpoint", "bids_notional_0", "asks_notional_0"),
        ("eth", "eth_midpoint", "eth_bids_notional_0", "eth_asks_notional_0"),
    ]:
        r = np.log(df[mid] / df[mid].shift(1))
        df[f"{asset}_log_return"] = r
        df[f"{asset}_oimb_1"] = (df[bid0] - df[ask0]) / (df[bid0] + df[ask0] + EPS)
        df[f"{asset}_volatility_10s"] = r.rolling(ws).std()
        df[f"{asset}_volatility_60s"] = r.rolling(wl).std()
        df[f"{asset}_momentum_10s"] = r.rolling(ws).sum()
        df[f"{asset}_momentum_60s"] = r.rolling(wl).sum()

    df["btc_eth_momentum_diff_10s"] = df["btc_momentum_10s"] - df["eth_momentum_10s"]
    df["btc_eth_momentum_diff_60s"] = df["btc_momentum_60s"] - df["eth_momentum_60s"]
    df["btc_eth_vol_ratio_10s"] = df["btc_volatility_10s"] / (df["eth_volatility_10s"] + EPS)
    return df


def add_labels(df):
    """future_return[t] = mid(t + HORIZON seconds) / mid(t) - 1.

    Looked up by timestamp rather than `shift(-HORIZON)`, because the feed has
    gaps: 10 rows later is not always 10 seconds later. Rows whose future
    timestamp is missing get `has_label = False` and are never used as targets.
    """
    mid = df.set_index("time")["midpoint"]
    future = mid.reindex(df["time"] + pd.Timedelta(seconds=C.HORIZON)).to_numpy()
    df["future_return"] = future / df["midpoint"].to_numpy() - 1
    df["has_label"] = ~np.isnan(df["future_return"])
    df["label"] = np.select(
        [df["future_return"] > C.LABEL_THRESHOLD,
         df["future_return"] < -C.LABEL_THRESHOLD],
        [1, -1], default=0).astype("int8")
    return df


def add_classic_factors(df):
    ws, wl = C.WINDOW_SHORT, C.WINDOW_LONG
    r = df["btc_log_return"]

    # 1. Weighted multi-level queue imbalance (Cont, Kukanov & Stoikov 2014)
    levels = 5
    w = np.exp(-0.5 * np.arange(levels))
    w /= w.sum()
    bid = df[[f"bids_notional_{i}" for i in range(levels)]].to_numpy()
    ask = df[[f"asks_notional_{i}" for i in range(levels)]].to_numpy()
    df["factor_queue_imbalance"] = ((bid - ask) / (bid + ask + EPS) * w).sum(axis=1)

    # 2. Order flow imbalance (Cont, Kukanov & Stoikov 2014).
    # `*_distance_0` is the relative distance to the midpoint, not a price, so
    # the best bid / ask prices are rebuilt and rounded to the tick before
    # comparing them across snapshots.
    bp = (df["midpoint"] * (1 + df["bids_distance_0"]) / C.TICK_SIZE).round()
    ap = (df["midpoint"] * (1 + df["asks_distance_0"]) / C.TICK_SIZE).round()
    bv, av = df["bids_notional_0"], df["asks_notional_0"]
    e_bid = bv * (bp >= bp.shift(1)) - bv.shift(1) * (bp <= bp.shift(1))
    e_ask = av.shift(1) * (ap >= ap.shift(1)) - av * (ap <= ap.shift(1))
    df["factor_ofi"] = e_bid - e_ask
    df["factor_ofi_10s"] = df["factor_ofi"].rolling(ws).sum()
    df["factor_ofi_60s"] = df["factor_ofi"].rolling(wl).sum()

    # 3. Kyle's lambda (Kyle 1985): rolling regression slope of return on OFI
    ofi = df["factor_ofi"].fillna(0)
    df["factor_kyle_lambda"] = r.rolling(wl).cov(ofi) / (ofi.rolling(wl).var() + EPS)

    # 4. Depth ratio / imbalance over 10 levels (Gould et al. 2013)
    total_bid = df[[f"bids_notional_{i}" for i in range(10)]].sum(axis=1)
    total_ask = df[[f"asks_notional_{i}" for i in range(10)]].sum(axis=1)
    df["factor_depth_ratio"] = total_ask / (total_bid + EPS)
    df["factor_depth_imb"] = (total_bid - total_ask) / (total_bid + total_ask + EPS)
    df["factor_depth_ratio_smooth"] = df["factor_depth_ratio"].rolling(ws).mean()

    # 5. Tick rule (Lee & Ready 1991), applied to midpoint changes
    tick = np.sign(r.fillna(0)).replace(0, np.nan).ffill().fillna(1)
    df["factor_tick_sign"] = tick
    df["factor_tick_momentum_10s"] = tick.rolling(ws).sum()
    df["factor_tick_momentum_60s"] = tick.rolling(wl).sum()

    # 6. Amihud-style illiquidity (Amihud 2002), with top-5 book depth standing
    # in for traded volume
    depth5 = (df[[f"bids_notional_{i}" for i in range(5)]].sum(axis=1)
              + df[[f"asks_notional_{i}" for i in range(5)]].sum(axis=1))
    df["factor_amihud"] = r.abs() / (depth5 + EPS)
    df["factor_amihud_60s"] = df["factor_amihud"].rolling(wl).mean()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3. Normalisation
# ══════════════════════════════════════════════════════════════════════════════
def rolling_zscore(df, cols, window=C.ZSCORE_WINDOW, min_periods=C.ZSCORE_MIN_PERIODS,
                   clip=10.0, chunk=16):
    """z[t] = (x[t] - mean(x[t-window+1 .. t])) / std(same window).

    Only past and current rows are used, so this is free of look-ahead and can
    be applied to the whole series before splitting.

    No additive epsilon in the denominator: Amihud and Kyle's lambda live
    around 1e-10, so `std + 1e-8` would squash them to ~0. A zero std gives 0
    instead. Values are clipped to ±`clip`: near-constant columns (e.g. a
    spread pinned at one tick) otherwise explode whenever they move.
    """
    for i in range(0, len(cols), chunk):
        c = cols[i:i + chunk]
        x = df[c].astype("float64")
        roll = x.rolling(window=window, min_periods=min_periods)
        mean, std = roll.mean(), roll.std()
        z = ((x - mean) / std.where(std > 0)).where(mean.notna())
        z = z.fillna(0.0).where(mean.notna())   # zero-std → 0; warm-up rows stay NaN
        df[c] = z.clip(-clip, clip).astype("float32")
        del x, roll, mean, std, z
        gc.collect()
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4. Split + PCA
# ══════════════════════════════════════════════════════════════════════════════
def assign_splits(df):
    """Chronological train / val / test split with purging.

    A row is a usable target (`is_target`) only if its whole label window
    [t, t + HORIZON] lies inside its own split, so no training label is built
    from validation or test prices (and likewise val → test).
    """
    n = len(df)
    cut1 = int(n * C.TRAIN_FRAC)
    cut2 = int(n * (C.TRAIN_FRAC + C.VAL_FRAC))
    split = np.empty(n, dtype=object)
    split[:cut1], split[cut1:cut2], split[cut2:] = "train", "val", "test"
    df["split"] = split

    label_end = df["time"] + pd.Timedelta(seconds=C.HORIZON)
    next_start = df["split"].map({"train": df["time"].iloc[cut1],
                                  "val": df["time"].iloc[cut2]})   # NaT for test
    df["is_target"] = df["has_label"] & ((label_end < next_start) | next_start.isna())

    for s in ["train", "val", "test"]:
        m = df["split"] == s
        print(f"  {s:<5}: {m.sum():>9,} rows | {df.loc[m, 'is_target'].sum():>9,} targets | "
              f"{df.loc[m, 'time'].iloc[0]} → {df.loc[m, 'time'].iloc[-1]}")
    return df


def add_pca(df):
    """StandardScaler + PCA on the 60 LOB columns, fitted on train rows only.

    Keeps the components with eigenvalue > 1 (Kaiser), capped at
    MAX_PCA_COMPONENTS.
    """
    train = (df["split"] == "train").to_numpy()
    scaler = StandardScaler().fit(df.loc[train, LOB_COLS])
    pca = PCA(n_components=0.90, random_state=C.SEED)
    pca.fit(scaler.transform(df.loc[train, LOB_COLS]))

    k = int(min(max((pca.explained_variance_ > 1).sum(), 1), C.MAX_PCA_COMPONENTS))
    Z = pca.transform(scaler.transform(df[LOB_COLS]))[:, :k].astype("float32")
    pca_cols = [f"pca_{i + 1}" for i in range(k)]
    df = pd.concat([df, pd.DataFrame(Z, columns=pca_cols, index=df.index)], axis=1)

    info = {
        "n_components_90pct": int(pca.n_components_),
        "n_components_kept": k,
        "explained_variance_kept": float(pca.explained_variance_ratio_[:k].sum()),
        "explained_variance_ratio": pca.explained_variance_ratio_.round(5).tolist(),
    }
    print(f"  PCA: {info['n_components_90pct']} comps for 90% variance, keeping top {k} "
          f"(Kaiser, cap {C.MAX_PCA_COMPONENTS}) → {info['explained_variance_kept']:.3f} variance")
    return df, pca_cols, info


# ══════════════════════════════════════════════════════════════════════════════
# End-to-end
# ══════════════════════════════════════════════════════════════════════════════
def build_dataset(data_dir=None, nrows=None, out_path=None):
    """Run the whole pipeline. Returns (df, meta); optionally writes parquet + json."""
    print("1. Load")
    df = load_raw(data_dir, nrows)

    print("2. Features + labels")
    df = add_market_features(df)
    df = add_labels(df)
    df = add_classic_factors(df)
    feature_cols = LOB_COLS + MARKET_COLS + CLASSIC_COLS
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_cols)
    df = df.reset_index(drop=True)

    print("3. Rolling z-score")
    df = rolling_zscore(df, feature_cols)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)

    print("4. Split")
    df = assign_splits(df)

    print("5. PCA")
    df, pca_cols, pca_info = add_pca(df)

    keep = (["time", "split", "is_target", "label", "future_return"]
            + pca_cols + LOB_COLS + MARKET_COLS + CLASSIC_COLS)
    df = df[keep].copy()
    df["future_return"] = df["future_return"].astype("float32")

    meta = {"pca_cols": pca_cols, "pca": pca_info, "n_rows": len(df),
            "label_dist": df.loc[df["is_target"], "label"].value_counts(normalize=True)
                            .sort_index().round(4).to_dict()}
    print(f"Done: {df.shape} | label dist (down/flat/up): {meta['label_dist']}")

    if out_path is not None:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        df.to_parquet(out_path, index=False)
        with open(os.path.splitext(out_path)[0] + "_meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"Saved → {out_path}")
    return df, meta
