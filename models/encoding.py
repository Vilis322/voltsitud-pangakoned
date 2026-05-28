"""
Categorical feature encoding for the labeled bank-call dataset (VP-13).

The labeled dataset (output of ``models.targets``) carries a mix of
column types. This module reshapes the non-numeric columns into a
form a tabular classifier can consume.

Per-column treatment

    channel                 — one-hot encoded with ``drop_first=True``
                              (3 levels → 2 indicator columns).
                              Low cardinality so one-hot is cheap; the
                              dropped baseline avoids the dummy trap.

    caller_number_prefix    — extracted as the first 5 digits of
                              ``caller_number``, then frequency-encoded
                              (each prefix replaced by how many calls
                              share it). Captures "is this number from
                              a heavily-used bucket?" without exploding
                              cardinality, and does not leak the target
                              (count is independent of ``is_fraud``).

    caller_number           — dropped. Raw 11-digit numbers are
                              identifiers, not features; encoding them
                              directly would overfit. The prefix above
                              keeps the useful signal.

    called_number           — dropped, same reasoning. The
                              ``distinct_victims`` aggregate exposed in
                              Sprint 2 already captured the
                              mass-targeting signal.

    timestamp               — dropped here. ``models.targets`` left it
                              as a string; the temporal features
                              (hour, weekday, is_business_hours, …) are
                              the responsibility of VP-12 (Temporal
                              feature extraction), so they appear after
                              this step in the pipeline.

    complaint_text          — dropped. Free text is not a "categorical"
                              feature; turning it into numeric signal
                              is the Sprint 5 "Additional behavioral
                              features" task (keyword-score column).

    call_id                 — dropped. Pure row identifier.

    label_5way              — dropped. The multi-class label is kept
                              in ``models.targets``'s output for
                              auditing, but should not survive into a
                              feature matrix used to predict
                              ``is_fraud``.

Frequency encoding vs target / label encoding

    Target encoding (replace each prefix by the per-prefix mean of
    ``is_fraud``) is strictly more informative, but it leaks the
    target if computed on the full set before the train/test split.
    Doing it correctly requires fitting on train only — that belongs
    in the same module as the split (VP-15). Frequency encoding is
    target-free, computable on the full set, and good enough for the
    Sprint 4 baselines.

Run (from project root):
    python -m models.encoding
Output:
    data/features_bank_calls.csv  — feature matrix + is_fraud column.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "labeled_bank_calls.csv"
OUTPUT = ROOT / "data" / "features_bank_calls.csv"

PREFIX_LEN = 5
DROP_COLUMNS = [
    "call_id",
    "caller_number",
    "called_number",
    "timestamp",
    "complaint_text",
    "label_5way",
]


def add_caller_number_prefix(df: pd.DataFrame, prefix_len: int = PREFIX_LEN) -> pd.DataFrame:
    """Return df with a new ``caller_number_prefix`` string column."""
    df = df.copy()
    df["caller_number_prefix"] = df["caller_number"].astype(str).str[:prefix_len]
    return df


def frequency_encode(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Replace ``column`` with its per-value count.

    The original column is removed and a new ``{column}_freq`` column
    is added. Frequency comes from the input frame itself — safe for
    a "fit on train, transform on test" pattern as long as the train
    frequencies are reused when transforming test.
    """
    df = df.copy()
    counts = df[column].value_counts()
    df[f"{column}_freq"] = df[column].map(counts)
    df = df.drop(columns=[column])
    return df


def one_hot_encode(df: pd.DataFrame, columns: list[str], drop_first: bool = True) -> pd.DataFrame:
    """Wrapper around pd.get_dummies with our project defaults."""
    return pd.get_dummies(df, columns=columns, drop_first=drop_first)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end transform: labeled rows → numeric feature matrix.

    Order matters: derive the prefix from caller_number *before*
    dropping caller_number; one-hot channel after prefix to keep the
    column dance readable.
    """
    df = add_caller_number_prefix(df)
    df = frequency_encode(df, "caller_number_prefix")
    df = one_hot_encode(df, ["channel"])
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return df


def main() -> None:
    df = pd.read_csv(INPUT)
    features = build_features(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(OUTPUT, index=False)

    print(f"input:  {INPUT.relative_to(ROOT)} ({len(df)} rows, {len(df.columns)} cols)")
    print(f"output: {OUTPUT.relative_to(ROOT)} ({len(features)} rows, {len(features.columns)} cols)\n")

    print("Feature dtypes:")
    print(features.dtypes.to_string())
    print(f"\nTarget balance (is_fraud): {features['is_fraud'].mean():.3f}")


if __name__ == "__main__":
    main()
