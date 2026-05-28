"""
Time-aware train/test split (VP-15).

Conventional ``train_test_split(shuffle=True)`` is unsafe on call-log
data: the model would peek at future events during training, which
leaks information unavailable at inference time. We instead sort by
``timestamp`` ascending and put the first 80% chronologically into
train, the most recent 20% into test — mimicking the real deployment
pattern (train on history, evaluate on the next period).

Pipeline order

    labeled_bank_calls.csv  -- has timestamp still
            │
            ├── normalize timestamps via VP-2 utility
            │   (raw file has mixed formats; ISO required for sort)
            │
            ├── sort by timestamp ascending (stable mergesort)
            │
            ├── build_features (drops timestamp + identifiers,
            │                   encodes categoricals)
            │
            └── split by index position
                    ├── first 80% → train_features.csv
                    └── last  20% → test_features.csv

Frequency-encoding note

    ``build_features`` runs ONCE on the full sorted frame before the
    split, so train and test share a single consistent frequency
    mapping for ``caller_number_prefix``. This is a mild feature-only
    leak (the test set's own row counts feed into its own feature
    values) but does NOT leak the target. Strict fit-on-train /
    transform-on-test would require refactoring the encoder; flagged
    for a Sprint 4 follow-up if model accuracy looks suspiciously
    optimistic.

Run (from project root):
    python -m models.split
Output:
    data/train_features.csv
    data/test_features.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.utils import normalize_timestamp
from models.encoding import build_features

ROOT = Path(__file__).resolve().parent.parent
LABELED = ROOT / "data" / "labeled_bank_calls.csv"
TRAIN_OUT = ROOT / "data" / "train_features.csv"
TEST_OUT = ROOT / "data" / "test_features.csv"

TEST_SIZE = 0.2
FRAUD_RATE_TOLERANCE_PP = 1.0


def sort_chronologically(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ``timestamp`` to ISO and sort ascending.

    The raw CSV may contain mixed formats from the generator (epoch
    seconds, DD/MM/YYYY, ISO with TZ, Estonian free-form). Lex sort
    only works once everything is in ``YYYY-MM-DD HH:MM:SS`` form.
    """
    df = df.copy()
    df["timestamp"] = df["timestamp"].apply(
        lambda v: normalize_timestamp(v) if pd.notna(v) else v
    )
    return df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


def time_aware_split(
    features: pd.DataFrame, test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a time-ordered feature frame by position.

    The caller is responsible for sorting the input by timestamp
    ascending BEFORE this function — the function trusts row order.
    """
    split_idx = int(len(features) * (1 - test_size))
    train = features.iloc[:split_idx].copy().reset_index(drop=True)
    test = features.iloc[split_idx:].copy().reset_index(drop=True)
    return train, test


def main() -> None:
    df = pd.read_csv(LABELED)
    sorted_df = sort_chronologically(df)
    features = build_features(sorted_df)

    train, test = time_aware_split(features)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    train_rate = train["is_fraud"].mean()
    test_rate = test["is_fraud"].mean()
    diff_pp = abs(train_rate - test_rate) * 100

    print(f"input:  {LABELED.relative_to(ROOT)} ({len(df)} rows)")
    print(f"train:  {TRAIN_OUT.relative_to(ROOT)} ({len(train)} rows, fraud rate {train_rate:.3f})")
    print(f"test:   {TEST_OUT.relative_to(ROOT)} ({len(test)} rows, fraud rate {test_rate:.3f})")
    print(f"fraud-rate gap: {diff_pp:.2f} percentage points")

    if diff_pp <= FRAUD_RATE_TOLERANCE_PP:
        print(f"[PASS] gap within {FRAUD_RATE_TOLERANCE_PP} pp acceptance criterion")
    else:
        print(
            f"[WARN] gap exceeds {FRAUD_RATE_TOLERANCE_PP} pp — temporal "
            f"drift in label distribution"
        )


if __name__ == "__main__":
    main()
