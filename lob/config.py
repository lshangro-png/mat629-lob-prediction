"""Shared constants. Every notebook and script reads from here so that the
three models are trained and evaluated under exactly the same protocol."""

KAGGLE_DATASET = "martinsn/high-frequency-crypto-limit-order-book-data"

SEED = 42

# ── Labels ────────────────────────────────────────────────────────────────────
HORIZON = 10              # prediction horizon, in seconds
LABEL_THRESHOLD = 1.5e-4  # |future return| above this → Up / Down, else Flat

# ── Features ──────────────────────────────────────────────────────────────────
N_LEVELS = 15             # order-book levels per side
WINDOW_SHORT = 10         # seconds
WINDOW_LONG = 60          # seconds
ZSCORE_WINDOW = 3600      # backward-looking rolling z-score window (1 hour)
ZSCORE_MIN_PERIODS = 60
MAX_PCA_COMPONENTS = 10   # Kaiser criterion (eigenvalue > 1), capped at this
TICK_SIZE = 0.01          # used to rebuild best bid / ask prices for OFI

# ── Chronological split (fractions of rows) ───────────────────────────────────
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15           # early stopping + threshold selection
# test = the remaining 0.15, touched only once for the final numbers

# ── Backtest ──────────────────────────────────────────────────────────────────
FEE = 1.5e-4              # per unit of position change (1.5 bps)
SECONDS_PER_YEAR = 365 * 24 * 3600
