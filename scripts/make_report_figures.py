"""Figures for the report's results section, from the CSVs in results/.

    python scripts/make_report_figures.py --out report/figures
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = os.path.join(os.path.dirname(__file__), "..", "results")

# categorical slots in fixed order + neutral for the benchmark
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRAY = "#8a8985"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "legend.frameon": False, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "savefig.bbox": "tight",
})
pct = matplotlib.ticker.PercentFormatter(1.0)


def walk_forward_bars(out):
    cnn = pd.read_csv(os.path.join(HERE, "walk_forward_cnn.csv"))
    lgb = pd.read_csv(os.path.join(HERE, "walk_forward_lgbm.csv"))
    ens = cnn[cnn["seed"] == "ens"].sort_values("fold")
    lgb_np = lgb[(lgb["feature_set"] == "no_pca") & (lgb["seed"] == "42")].sort_values("fold")
    days = ens["test_day"].str[:5].tolist()
    series = [("CNN (3-seed ensemble)", ens["hy_return"].to_numpy(), BLUE),
              ("LightGBM (no_pca)", lgb_np["hy_return"].to_numpy(), ORANGE),
              ("Buy & hold", ens["bh_return"].to_numpy(), GRAY)]

    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x, w = np.arange(len(days)), 0.26
    for k, (name, vals, color) in enumerate(series):
        ax.bar(x + (k - 1) * w, vals, w, color=color, edgecolor=SURFACE, linewidth=1.2, label=name)
    ax.axhline(0, color=INK2, linewidth=0.8)
    ax.set_xticks(x, [f"Fold {i}\n{d}" for i, d in enumerate(days)])
    ax.yaxis.set_major_formatter(pct)
    ax.set_ylabel("Test-day return (hysteresis, 1.5 bp)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=8)
    fig.savefig(os.path.join(out, "fig_walk_forward.pdf"))
    fig.savefig(os.path.join(out, "fig_walk_forward.png"), dpi=200)
    plt.close(fig)


def delay_lines(out):
    delays = [0, 1, 2, 3, 5]
    cols = [f"delay_{d}s" for d in delays]
    main = pd.read_csv(os.path.join(HERE, "main_split_delay_h10.csv"))
    wf = pd.read_csv(os.path.join(HERE, "walk_forward_delay_cnn_ensemble.csv"))

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    panels = [
        (axes[0], "CNN ensemble, walk-forward (mean of 6 days)",
         [("CNN ensemble", wf[cols].mean().to_numpy(), BLUE)]),
        (axes[1], "Main split (test 04-17 to 04-19)",
         [(lbl, main.loc[(main["model"] == m) & (main["feature_set"] == fs), cols].iloc[0].to_numpy(), c)
          for lbl, m, fs, c in [("CNN", "cnn", "full", BLUE), ("LightGBM", "lgbm", "no_pca", ORANGE),
                                ("LSTM", "lstm", "full", AQUA), ("Transformer", "transformer", "full", YELLOW)]]),
    ]
    for ax, title, lines in panels:
        for name, vals, color in lines:
            ax.plot(delays, vals, color=color, linewidth=2, marker="o", markersize=4.5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=name)
        ax.axhline(0, color=INK2, linewidth=0.8)
        ax.set_xticks(delays)
        ax.set_xlabel("Execution delay (seconds)")
        ax.yaxis.set_major_formatter(pct)
        ax.set_title(title, fontsize=9, color=INK)
    axes[0].set_ylabel("Hysteresis test return")
    for ax, _, lines in panels:
        if len(lines) > 1:
            ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_delay.pdf"))
    fig.savefig(os.path.join(out, "fig_delay.png"), dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="report/figures")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    walk_forward_bars(a.out)
    delay_lines(a.out)
    print("saved to", a.out)
