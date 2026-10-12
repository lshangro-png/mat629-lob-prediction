"""Shared constants. Every notebook and script reads from here so that the
three models are trained and evaluated under exactly the same protocol."""

KAGGLE_DATASET = "martinsn/high-frequency-crypto-limit-order-book-data"

SEED = 42

# ── Labels ────────────────────────────────────────────────────────────────────
HORIZON = 10                    # default prediction horizon, in seconds
HORIZONS = (10, 30, 60, 300)    # labels are built for all of these
# |future return| above the threshold → Up / Down, else Flat. 10 s keeps the
# original 1.5 bps; longer horizons use the round-trip cost (2 × FEE = 3 bps),
# so an "Up" label is a move that would still be profitable after fees.
LABEL_THRESHOLDS = {10: 1.5e-4, 30: 3e-4, 60: 3e-4, 300: 3e-4}

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
