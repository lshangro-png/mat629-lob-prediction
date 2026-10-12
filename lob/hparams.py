"""Per-model hyperparameters, shared by the notebooks and scripts/run_model.py."""

HPARAMS = {
    "cnn": dict(
        seq_len=10, batch_size=256,
        model_kwargs=dict(n_filters=64, kernel_size=3, n_blocks=2, dropout=0.3),
        train_kwargs=dict(epochs=50, lr=1e-4, weight_decay=1e-2, patience=10,
                          scheduler="cosine", monitor="loss"),
    ),
    "lstm": dict(
        seq_len=10, batch_size=256,
        model_kwargs=dict(hidden_size=32, n_layers=2, dropout=0.4),
        train_kwargs=dict(epochs=40, lr=1e-4, weight_decay=2e-3, patience=6,
                          scheduler="cosine", monitor="loss"),
    ),
    "transformer": dict(
        seq_len=30, batch_size=512,
        model_kwargs=dict(d_model=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1),
        train_kwargs=dict(epochs=30, lr=1e-3, weight_decay=1e-4, patience=6,
                          scheduler="onecycle", monitor="loss"),
    ),
    "lgbm": dict(
        params=dict(n_estimators=2000, learning_rate=0.05, num_leaves=31,
                    min_child_samples=200, subsample=0.8, subsample_freq=1,
                    colsample_bytree=0.8, reg_lambda=1.0, n_jobs=-1,
                    early_stopping_rounds=50),
    ),
}
