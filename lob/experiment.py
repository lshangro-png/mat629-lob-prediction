"""train → validate → (choose threshold on val) → test once, for one model and
one feature set."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config as C
from .backtest import (backtest, format_stats, nonoverlap_steps, positions_from_probs,
                       sweep_thresholds)
from .data import CLASS_NAMES, FEATURE_SET_DESC, make_loaders
from .models import MODELS
from .training import fit, predict_proba, set_seed

MIN_TURNOVER = 20   # thresholds that barely trade on val are not eligible


def run_experiment(df, meta, model_name, feature_set, model_kwargs, train_kwargs,
                   seq_len, batch_size, device, fee=C.FEE, num_workers=2, plot=True):
    title = f"{model_name.upper()} | {feature_set} ({FEATURE_SET_DESC[feature_set]})"
    print("=" * 90 + f"\n{title}\n" + "=" * 90)
    set_seed(C.SEED)

    loaders, targets, class_w, n_features = make_loaders(
        df, meta, feature_set, seq_len, batch_size, device, num_workers)
    model = MODELS[model_name](n_features, **model_kwargs).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    history = fit(model, loaders, class_w, device, **train_kwargs)

    # ── Classification metrics on test ───────────────────────────────────────
    p_val, _ = predict_proba(model, loaders["val"], device)
    p_test, y_test = predict_proba(model, loaders["test"], device)
    y_pred = p_test.argmax(1)
    print("\nTest classification report (argmax):")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, digits=4,
                                zero_division=0))

    # ── Backtests on non-overlapping steps ───────────────────────────────────
    ret = df["future_return"].to_numpy()
    sv = nonoverlap_steps(targets["val"])
    st = nonoverlap_steps(targets["test"])
    r_val, r_test = ret[targets["val"][sv]], ret[targets["test"][st]]

    sweep = sweep_thresholds(p_val[sv], r_val, fee)
    eligible = sweep[sweep["turnover"] >= MIN_TURNOVER]
    best = (eligible if len(eligible) else sweep).sort_values("sharpe").iloc[-1]
    th = float(best["threshold"])
    print(f"\nThreshold chosen on VAL: {th:.4f} (val Sharpe {best['sharpe']:.2f}, "
          f"val return {best['total_return']:.2%})")

    bt_argmax = backtest(positions_from_probs(p_test[st]), r_test, fee)
    bt_th = backtest(positions_from_probs(p_test[st], th), r_test, fee)
    bt_bh = backtest(np.ones(len(st)), r_test, fee=0.0)
    print(f"\nTest backtest ({len(st):,} non-overlapping {C.HORIZON}s steps, "
          f"fee {fee * 1e4:.1f} bps):")
    print(format_stats("Argmax", bt_argmax))
    print(format_stats(f"Threshold {th:.3f} (from val)", bt_th))
    print(format_stats("Buy & hold (no fee)", bt_bh))

    if plot:
        _plot(title, history, y_test, y_pred, sweep, th, bt_argmax, bt_th, bt_bh, fee)

    summary = {
        "model": model_name, "feature_set": feature_set, "n_features": n_features,
        "best_epoch": history.get("best_epoch"),
        "test_acc": float((y_pred == y_test).mean()),
        "test_dir_f1": float(f1_score(y_test, y_pred, labels=[0, 2], average="macro",
                                      zero_division=0)),
        "argmax_return": bt_argmax["total_return"], "argmax_sharpe": bt_argmax["sharpe"],
        "threshold": th,
        "th_return": bt_th["total_return"], "th_sharpe": bt_th["sharpe"],
        "th_mdd": bt_th["max_drawdown"], "th_turnover": bt_th["turnover"],
        "bh_return": bt_bh["total_return"],
    }
    return summary, model, history


def save_summary(rows, model_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"{model_name}_summary.csv"), index=False)
    return out


def _plot(title, history, y_true, y_pred, sweep, th, bt_argmax, bt_th, bt_bh, fee):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(title, fontsize=14)
    ep = range(1, len(history["train_loss"]) + 1)

    ax[0, 0].plot(ep, history["train_loss"], label="train")
    ax[0, 0].plot(ep, history["val_loss"], label="val")
    ax[0, 0].axvline(history["best_epoch"], color="gray", ls=":", label="best")
    ax[0, 0].set_title("Loss"); ax[0, 0].legend()

    ax[0, 1].plot(ep, history["val_acc"], label="val acc")
    ax[0, 1].plot(ep, history["val_dir_f1"], label="val DirF1")
    ax[0, 1].axvline(history["best_epoch"], color="gray", ls=":")
    ax[0, 1].set_title("Validation metrics"); ax[0, 1].legend()

    sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]), annot=True, fmt="d",
                cmap="Blues", ax=ax[0, 2], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax[0, 2].set_title("Test confusion matrix (argmax)")
    ax[0, 2].set_xlabel("Predicted"); ax[0, 2].set_ylabel("True")

    a = ax[1, 0]
    a.plot(sweep["threshold"], sweep["sharpe"], "o-", color="tab:red")
    a.axvline(th, color="tab:red", ls=":")
    a.set_xlabel("threshold"); a.set_ylabel("val Sharpe", color="tab:red")
    b = a.twinx()
    b.plot(sweep["threshold"], sweep["turnover"], "x--", color="tab:blue")
    b.set_ylabel("val turnover", color="tab:blue")
    a.set_title("Threshold sweep (validation)")

    a = plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)
    ax[1, 1].remove(); ax[1, 2].remove()
    a.plot(bt_bh["equity"], color="gray", alpha=0.6, label="Buy & hold")
    a.plot(bt_argmax["equity"], ls="--", label="Argmax")
    a.plot(bt_th["equity"], lw=2, label=f"Threshold {th:.3f}")
    a.set_title(f"Test equity curve (fee {fee * 1e4:.1f} bps, non-overlapping {C.HORIZON}s steps)")
    a.set_xlabel("step"); a.legend()

    for x in fig.axes:
        if x is not ax[0, 2]:
            x.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
