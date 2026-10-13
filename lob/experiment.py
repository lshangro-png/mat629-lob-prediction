"""train → validate → (choose trading rule on val) → test once, for one model,
one feature set and one prediction horizon."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from . import config as C
from .backtest import (backtest, format_stats, nonoverlap_steps, pick_best,
                       positions_from_probs, positions_hysteresis, sweep_hysteresis,
                       sweep_thresholds)
from .data import CLASS_NAMES, FEATURE_SET_DESC

MIN_TURNOVER = 20   # trading rules that barely trade on val are not eligible


def _fit_predict(df, meta, model_name, feature_set, hp, device, horizon, num_workers, seed):
    """Train one model; return (p_val, p_test, y_test, targets, history, model)."""
    if model_name == "lgbm":
        from .baselines import fit_predict_lgbm
        p_val, p_test, y_test, targets, history = fit_predict_lgbm(
            df, meta, feature_set, hp["params"], horizon, seed)
        return p_val, p_test, y_test, targets, history, None

    from .loaders import make_loaders
    from .models import MODELS
    from .training import fit, predict_proba, set_seed

    set_seed(seed)
    loaders, targets, class_w, n_features = make_loaders(
        df, meta, feature_set, hp["seq_len"], hp["batch_size"], device, num_workers, horizon)
    model = MODELS[model_name](n_features, **hp["model_kwargs"]).to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    history = fit(model, loaders, class_w, device, **hp["train_kwargs"])
    p_val, _ = predict_proba(model, loaders["val"], device)
    p_test, y_test = predict_proba(model, loaders["test"], device)
    return p_val, p_test, y_test, targets, history, model


def run_experiment(df, meta, model_name, feature_set, hp, device=None, horizon=C.HORIZON,
                   fee=C.FEE, num_workers=2, plot=True, fig_path=None, pred_path=None,
                   seed=C.SEED):
    """`hp`: one entry of lob.hparams.HPARAMS. `fig_path`: save the figure there
    instead of showing it. `pred_path`: save val/test probabilities (.npz) so
    other trading rules can be evaluated later without retraining."""
    title = (f"{model_name.upper()} | {feature_set} ({FEATURE_SET_DESC[feature_set]}) "
             f"| horizon {horizon}s")
    print("=" * 100 + f"\n{title}\n" + "=" * 100)

    p_val, p_test, y_test, targets, history, model = _fit_predict(
        df, meta, model_name, feature_set, hp, device, horizon, num_workers, seed)

    if pred_path is not None:
        os.makedirs(os.path.dirname(pred_path) or ".", exist_ok=True)
        np.savez_compressed(pred_path, horizon=horizon,
                            val_rows=targets["val"], val_probs=p_val.astype(np.float32),
                            test_rows=targets["test"], test_probs=p_test.astype(np.float32))

    summary, curves = evaluate(df, p_val, p_test, y_test, targets, horizon, fee)
    summary = {"model": model_name, "feature_set": feature_set, "horizon": horizon,
               "seed": seed, "best_epoch": history.get("best_epoch"), **summary}
    if plot:
        _plot(title, history, y_test, p_test.argmax(1), curves, horizon, fee, fig_path)
    return summary, model, history


def evaluate(df, p_val, p_test, y_test, targets, horizon=C.HORIZON, fee=C.FEE):
    """Classification metrics + backtests. Trading rules are chosen on val only."""
    y_pred = p_test.argmax(1)
    print("\nTest classification report (argmax):")
    print(classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES,
                                digits=4, zero_division=0))

    ret = df[f"future_return_{horizon}"].to_numpy()
    sv = nonoverlap_steps(targets["val"], horizon)
    st = nonoverlap_steps(targets["test"], horizon)
    r_val, r_test = ret[targets["val"][sv]], ret[targets["test"][st]]
    pv, pt = p_val[sv], p_test[st]

    th_sweep = sweep_thresholds(pv, r_val, fee, horizon)
    th_best = pick_best(th_sweep, MIN_TURNOVER)
    th = float(th_best["threshold"])
    hy_sweep = sweep_hysteresis(pv, r_val, fee, horizon)
    hy_best = pick_best(hy_sweep, MIN_TURNOVER)
    enter, exit_ = float(hy_best["enter"]), float(hy_best["exit"])
    print(f"\nChosen on VAL: threshold {th:.3f} (val Sharpe {th_best['sharpe']:.2f}) | "
          f"hysteresis enter {enter:.2f} / exit {exit_:.2f} (val Sharpe {hy_best['sharpe']:.2f})")

    bt = {
        "argmax": backtest(positions_from_probs(pt), r_test, fee, horizon),
        "th": backtest(positions_from_probs(pt, th), r_test, fee, horizon),
        "hy": backtest(positions_hysteresis(pt, enter, exit_), r_test, fee, horizon),
        "bh": backtest(np.ones(len(st)), r_test, 0.0, horizon),
    }
    print(f"\nTest backtest ({len(st):,} non-overlapping {horizon}s steps, fee {fee * 1e4:.1f} bps):")
    print(format_stats("Argmax", bt["argmax"]))
    print(format_stats(f"Threshold {th:.3f}", bt["th"]))
    print(format_stats(f"Hysteresis {enter:.2f}/{exit_:.2f}", bt["hy"]))
    print(format_stats("Buy & hold (no fee)", bt["bh"]))

    summary = {
        "test_acc": float((y_pred == y_test).mean()),
        "flat_acc": float((y_test == 1).mean()),     # always-Flat baseline
        "test_dir_f1": float(f1_score(y_test, y_pred, labels=[0, 2], average="macro",
                                      zero_division=0)),
        "argmax_return": bt["argmax"]["total_return"], "argmax_sharpe": bt["argmax"]["sharpe"],
        "argmax_turnover": bt["argmax"]["turnover"],
        "threshold": th,
        "th_return": bt["th"]["total_return"], "th_sharpe": bt["th"]["sharpe"],
        "th_mdd": bt["th"]["max_drawdown"], "th_turnover": bt["th"]["turnover"],
        "hy_enter": enter, "hy_exit": exit_,
        "hy_return": bt["hy"]["total_return"], "hy_sharpe": bt["hy"]["sharpe"],
        "hy_mdd": bt["hy"]["max_drawdown"], "hy_turnover": bt["hy"]["turnover"],
        "bh_return": bt["bh"]["total_return"],
    }
    curves = {"th_sweep": th_sweep, "th": th, "enter": enter, "exit": exit_, "bt": bt}
    return summary, curves


def save_summary(rows, model_name, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(out_dir, f"{model_name}_summary.csv"), index=False)
    return out


def _plot(title, history, y_true, y_pred, curves, horizon, fee, fig_path=None):
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(title, fontsize=14)
    best = history.get("best_epoch")

    ep = range(1, len(history["train_loss"]) + 1)
    ax[0, 0].plot(ep, history["train_loss"], label="train")
    ax[0, 0].plot(ep, history["val_loss"], label="val")
    if best:
        ax[0, 0].axvline(best, color="gray", ls=":", label="best")
    ax[0, 0].set_title("Loss"); ax[0, 0].legend()

    if "val_acc" in history:
        ax[0, 1].plot(ep, history["val_acc"], label="val acc")
        ax[0, 1].plot(ep, history["val_dir_f1"], label="val DirF1")
        if best:
            ax[0, 1].axvline(best, color="gray", ls=":")
        ax[0, 1].set_title("Validation metrics"); ax[0, 1].legend()
    else:
        ax[0, 1].axis("off")

    sns.heatmap(confusion_matrix(y_true, y_pred, labels=[0, 1, 2]), annot=True, fmt="d",
                cmap="Blues", ax=ax[0, 2], xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    ax[0, 2].set_title("Test confusion matrix (argmax)")
    ax[0, 2].set_xlabel("Predicted"); ax[0, 2].set_ylabel("True")

    sw, th = curves["th_sweep"], curves["th"]
    a = ax[1, 0]
    a.plot(sw["threshold"], sw["sharpe"], "o-", color="tab:red")
    a.axvline(th, color="tab:red", ls=":")
    a.set_xlabel("threshold"); a.set_ylabel("val Sharpe", color="tab:red")
    b = a.twinx()
    b.plot(sw["threshold"], sw["turnover"], "x--", color="tab:blue")
    b.set_ylabel("val turnover", color="tab:blue")
    a.set_title("Threshold sweep (validation)")

    a = plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)
    ax[1, 1].remove(); ax[1, 2].remove()
    bt = curves["bt"]
    a.plot(bt["bh"]["equity"], color="gray", alpha=0.6, label="Buy & hold")
    a.plot(bt["argmax"]["equity"], ls="--", label="Argmax")
    a.plot(bt["th"]["equity"], lw=2, label=f"Threshold {th:.3f}")
    a.plot(bt["hy"]["equity"], lw=2, label=f"Hysteresis {curves['enter']:.2f}/{curves['exit']:.2f}")
    a.set_title(f"Test equity curve (fee {fee * 1e4:.1f} bps, non-overlapping {horizon}s steps)")
    a.set_xlabel("step"); a.legend()

    for x in fig.axes:
        if x is not ax[0, 2]:
            x.grid(alpha=0.3)
    plt.tight_layout()
    if fig_path is None:
        plt.show()
    else:
        os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
        fig.savefig(fig_path, dpi=110)
        plt.close(fig)
