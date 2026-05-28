"""
End-to-end training data preparation (Sprint 4 prep).

Replaces ``models/split.py`` + ``models/scale_features.py`` with a single
pipeline that picks one consistent train/test split and emits both
scaled and unscaled variants of it.

Why this replaces the two earlier modules

    - ``models/split.py`` did the time-aware split but no scaling.
    - ``models/scale_features.py`` did scaling but on a random shuffle
      split decoupled from the time-aware one.

    Models trained against the two could not be compared fairly because
    the test sets differed. The unified pipeline here picks the time-
    aware split, runs scaling on top, and emits both forms so each
    model family uses the form best suited to it (trees → unscaled;
    Logistic Regression / NN → scaled) — but all on the same test rows.

Pipeline

    labeled_bank_calls.csv (from models/targets.py)
            │
            ├── add_temporal_features (hour, weekday, is_weekend,
            │                          is_business_hours) — derived
            │                          BEFORE the categorical encoder
            │                          drops the timestamp column.
            │
            ├── normalize_timestamp + sort_values ascending
            │
            ├── build_features (from models/encoding.py) — drops
            │                  timestamp + identifiers, encodes
            │                  channel via one-hot, frequency-encodes
            │                  caller_number_prefix.
            │
            ├── drop confidence_score — it is derived from the label,
            │                  including it as a feature leaks the
            │                  target.
            │
            ├── time-aware iloc-split (first 80% → train, last 20% test)
            │
            ├── StandardScaler.fit_transform on train numeric columns
            │                  (excluding booleans and the target);
            │                  scaler.transform on test using the same
            │                  fit. joblib-persisted for dashboard reuse.
            │
            └── write train_features.csv, test_features.csv (unscaled)
                  + train_scaled.csv, test_scaled.csv (scaled)
                  + models/scaler.joblib

Run (from project root):
    python -m models.prepare
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dataset.utils import normalize_timestamp
from models.encoding import build_features

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "labeled_bank_calls.csv"

TRAIN_OUT = ROOT / "data" / "train_features.csv"
TEST_OUT = ROOT / "data" / "test_features.csv"
TRAIN_SCALED_OUT = ROOT / "data" / "train_scaled.csv"
TEST_SCALED_OUT = ROOT / "data" / "test_scaled.csv"
SCALER_OUT = ROOT / "models" / "scaler.joblib"

TEST_SIZE = 0.2
FRAUD_RATE_TOLERANCE_PP = 1.0

# Dropped before training because their value depends on the label
# itself; treating them as features would leak the target.
LEAK_COLUMNS = ["confidence_score"]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour, weekday, is_weekend, is_business_hours from timestamp.

    Timestamps are normalized via the VP-2 utility first so mixed-format
    raw values (epoch, DD/MM/YYYY, ISO+TZ, Estonian free-form) parse
    uniformly.
    """
    df = df.copy()
    normalized = df["timestamp"].apply(
        lambda v: normalize_timestamp(v) if pd.notna(v) else v
    )
    df["timestamp"] = normalized
    ts = pd.to_datetime(normalized, errors="coerce")
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(bool)
    df["is_business_hours"] = (
        (df["hour"] >= 9) & (df["hour"] < 17) & (~df["is_weekend"])
    )
    return df


def time_aware_split(df: pd.DataFrame, test_size: float = TEST_SIZE) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered frame by position (no shuffle)."""
    split_idx = int(len(df) * (1 - test_size))
    train = df.iloc[:split_idx].copy().reset_index(drop=True)
    test = df.iloc[split_idx:].copy().reset_index(drop=True)
    return train, test


def numeric_columns_to_scale(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns suitable for StandardScaler.

    Excludes:
        - the target (``is_fraud``)
        - boolean columns (already in 0/1 form, scaling adds no value
          and would muddy downstream interpretation)
    """
    cols: list[str] = []
    for c in df.columns:
        if c == "is_fraud":
            continue
        if df[c].dtype == "bool":
            continue
        if df[c].dtype.kind in ("i", "f"):  # int / float
            cols.append(c)
    return cols


def main() -> None:
    df = pd.read_csv(LABELED)
    n_input = len(df)

    df = add_temporal_features(df)
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df = build_features(df)
    df = df.drop(columns=[c for c in LEAK_COLUMNS if c in df.columns])

    train, test = time_aware_split(df)

    scale_cols = numeric_columns_to_scale(train)
    scaler = StandardScaler()
    train_scaled = train.copy()
    test_scaled = test.copy()
    train_scaled[scale_cols] = scaler.fit_transform(train[scale_cols])
    test_scaled[scale_cols] = scaler.transform(test[scale_cols])

    for path in [TRAIN_OUT, TEST_OUT, TRAIN_SCALED_OUT, TEST_SCALED_OUT, SCALER_OUT]:
        path.parent.mkdir(parents=True, exist_ok=True)

    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    train_scaled.to_csv(TRAIN_SCALED_OUT, index=False)
    test_scaled.to_csv(TEST_SCALED_OUT, index=False)
    joblib.dump(scaler, SCALER_OUT)

    train_rate = train["is_fraud"].mean()
    test_rate = test["is_fraud"].mean()
    diff_pp = abs(train_rate - test_rate) * 100

    print(f"input:  {LABELED.relative_to(ROOT)} ({n_input} rows)")
    print(f"train:  {len(train)} rows ({train_rate:.3f} fraud rate)")
    print(f"test:   {len(test)} rows ({test_rate:.3f} fraud rate)")
    print(f"fraud-rate gap: {diff_pp:.2f} pp ({'PASS' if diff_pp <= FRAUD_RATE_TOLERANCE_PP else 'WARN — expected, see PR #47 discussion'})")
    print(f"\nfeatures ({len(train.columns) - 1} columns, target = is_fraud):")
    print("  " + ", ".join(c for c in train.columns if c != "is_fraud"))
    print(f"\nscaled numeric columns ({len(scale_cols)}):")
    print("  " + ", ".join(scale_cols))
    print(f"\nfiles written:")
    for p in [TRAIN_OUT, TEST_OUT, TRAIN_SCALED_OUT, TEST_SCALED_OUT, SCALER_OUT]:
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
