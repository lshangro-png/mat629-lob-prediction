"""One backtest used by every model, so the numbers are directly comparable.

Conventions
-----------
* A position in {-1, 0, +1} is opened at a target row t and held for exactly
  HORIZON seconds, earning future_return[t].
* Steps are non-overlapping: the next step starts at least HORIZON rows
  (≥ HORIZON seconds) later. Each 10-second window is counted once.
* Cost = FEE × |position change|, so a reversal (-1 → +1) pays 2 × FEE and the
  first entry is charged too.
* Sharpe is annualised with sqrt(steps per year). With 10-second steps that
  factor is ~1776, so the values look huge; compare them across models,
  not with daily-frequency strategies.
"""
import numpy as np
import pandas as pd

from . import config as C

ANNUAL_FACTOR = C.SECONDS_PER_YEAR / C.HORIZON


def nonoverlap_steps(rows, horizon=C.HORIZON):
    """Greedy subset of sorted row indices, each ≥ `horizon` rows after the last.
    Returns positions into `rows`."""
    keep, nxt = [], -np.inf
    for i, r in enumerate(rows):
        if r >= nxt:
            keep.append(i)
            nxt = r + horizon
    return np.asarray(keep, dtype=np.int64)


def positions_from_probs(probs, threshold=None):
    """threshold=None → argmax. Otherwise per-class: short if P(Down) > th,
    long if P(Up) > th, else flat. For th ≥ 0.5 both cannot fire at once."""
    if threshold is None:
        return probs.argmax(1) - 1
    pos = np.zeros(len(probs), dtype=np.int64)
    pos[probs[:, 0] > threshold] = -1
    pos[probs[:, 2] > threshold] = 1
    return pos


def backtest(pos, returns, fee=C.FEE):
    pos = np.asarray(pos, dtype=np.float64)
    returns = np.asarray(returns, dtype=np.float64)
    trades = np.abs(np.diff(pos, prepend=0.0))
    net = pos * returns - fee * trades
    equity = np.cumprod(1 + net)
    std = net.std()
    active = pos != 0
    return {
        "total_return": float(equity[-1] - 1),
        "sharpe": float(net.mean() / std * np.sqrt(ANNUAL_FACTOR)) if std > 0 else 0.0,
        "max_drawdown": float((equity / np.maximum.accumulate(equity) - 1).min()),
        "win_rate": float((net[active] > 0).mean()) if active.any() else float("nan"),
        "exposure": float(active.mean()),
        "turnover": float(trades.sum()),
        "n_steps": int(len(pos)),
        "equity": equity,
    }


def format_stats(name, s):
    return (f"{name:<28} return {s['total_return']:>8.2%} | Sharpe {s['sharpe']:>8.2f} | "
            f"MDD {s['max_drawdown']:>8.2%} | win {s['win_rate']:>6.1%} | "
            f"exposure {s['exposure']:>6.1%} | turnover {s['turnover']:>7,.0f}")


def sweep_thresholds(probs, returns, fee=C.FEE, thresholds=None):
    """Sharpe / return / turnover for each threshold. Run this on the VALIDATION
    split only and carry the chosen threshold to test."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.35, 0.90, 23), 4)
    rows = []
    for th in thresholds:
        s = backtest(positions_from_probs(probs, th), returns, fee)
        rows.append({"threshold": th, "sharpe": s["sharpe"],
                     "total_return": s["total_return"], "turnover": s["turnover"]})
    return pd.DataFrame(rows)
