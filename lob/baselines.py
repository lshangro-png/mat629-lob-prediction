"""Non-deep baseline: LightGBM on the features of the current second only.

If this matches the sequence models, the deep architectures are not adding
anything beyond the hand-crafted features.
"""
import numpy as np

from . import config as C
from .data import class_labels, feature_columns, target_indices


def fit_predict_lgbm(df, meta, feature_set, params, horizon=C.HORIZON):
    """Train on train targets, early-stop on val multi-logloss.

    Returns (p_val, p_test, y_test, targets, history) in the same shape as the
    sequence-model path in `experiment.run_experiment`.
    """
    import lightgbm as lgb

    cols = feature_columns(feature_set, meta["pca_cols"])
    y_all = class_labels(df, horizon)
    targets = {s: target_indices(df, s, 1, horizon) for s in ["train", "val", "test"]}
    X_all = df[cols].to_numpy(dtype=np.float32)
    X = {s: X_all[t] for s, t in targets.items()}
    y = {s: y_all[t] for s, t in targets.items()}
    print(f"LightGBM | {feature_set} | {len(cols)} features | horizon {horizon}s | "
          + " | ".join(f"{s} {len(t):,}" for s, t in targets.items()))

    params = dict(params)
    early_stopping = params.pop("early_stopping_rounds", 50)
    model = lgb.LGBMClassifier(objective="multiclass", class_weight="balanced",
                               random_state=C.SEED, verbose=-1, **params)
    evals = {}
    model.fit(X["train"], y["train"],
              eval_set=[(X["train"], y["train"]), (X["val"], y["val"])],
              eval_names=["train", "val"], eval_metric="multi_logloss",
              callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                         lgb.log_evaluation(100), lgb.record_evaluation(evals)])
    print(f"Best iteration: {model.best_iteration_}")

    history = {"train_loss": evals["train"]["multi_logloss"],
               "val_loss": evals["val"]["multi_logloss"],
               "best_epoch": model.best_iteration_}
    p_val = model.predict_proba(X["val"])
    p_test = model.predict_proba(X["test"])
    return p_val, p_test, y["test"], targets, history
