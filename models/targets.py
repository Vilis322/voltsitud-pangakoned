"""
Target variable construction for the fraud-detection models (VP-11).

Produces two target columns from the canonical ``label_5way``:

    is_fraud (binary, used by the main classifier)
        1  – Confirmed_Fraud, High_Risk_Reported
        0  – Verified_Legitimate
        — (row dropped) – Community_Flagged, Unknown

    label_5way (multi-class, used for analysis and future calibration)
        Five canonical classes kept as-is.

Rationale for the binary mapping

    Sprint 4 trains a primary binary classifier. The two ambiguous
    classes (``community_flagged`` is external signal only; ``unknown``
    is "suspicious but evidence insufficient") would dilute the binary
    signal in either direction:

        - calling them positives inflates false positives (since the
          underlying behavior may be legitimate);
        - calling them negatives buries true fraud cases in the
          negative class.

    Dropping them yields a cleaner binary supervised problem and
    matches the "Confidence score" guidance in the course ML
    methodology: train on the highest-confidence labels first; the
    grey-zone rows can later be used for semi-supervised or active
    learning experiments if there is time.

    The multi-class label is preserved for class-aware calibration
    (e.g. predicting the 5-class distribution and aggregating into a
    fraud probability) — left as a Sprint 5 stretch goal.

Run (from project root):
    python -m models.targets
Output:
    data/labeled_bank_calls.csv  — same rows as cleaned + is_fraud column,
                                   ambiguous rows dropped
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "data" / "cleaned_bank_calls.csv"
OUTPUT = ROOT / "data" / "labeled_bank_calls.csv"

# Canonical CamelCase labels produced by analysis/cleaning.py.
POSITIVE_LABELS = {"Confirmed_Fraud", "High_Risk_Reported"}
NEGATIVE_LABELS = {"Verified_Legitimate"}
AMBIGUOUS_LABELS = {"Community_Flagged", "Unknown"}


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Return a new frame with ``is_fraud`` added and ambiguous rows dropped.

    The multi-class ``label_5way`` column is left untouched. The
    function does not mutate the input.
    """
    df = df.copy()
    keep_mask = df["label_5way"].isin(POSITIVE_LABELS | NEGATIVE_LABELS)
    df = df.loc[keep_mask].copy()
    df["is_fraud"] = df["label_5way"].isin(POSITIVE_LABELS).astype(int)
    return df.reset_index(drop=True)


def summarize(df: pd.DataFrame, original_size: int) -> str:
    """Build the printable distribution report used by the CLI."""
    dropped = original_size - len(df)
    drop_pct = 100 * dropped / original_size if original_size else 0.0
    pos = int((df["is_fraud"] == 1).sum())
    neg = int((df["is_fraud"] == 0).sum())
    pos_pct = 100 * pos / len(df) if len(df) else 0.0

    multiclass = df["label_5way"].value_counts().to_string()

    return (
        f"Input rows: {original_size}\n"
        f"Kept rows:  {len(df)} ({100 - drop_pct:.1f}% of input)\n"
        f"Dropped:    {dropped} ({drop_pct:.1f}%) — Community_Flagged + Unknown\n"
        f"\n"
        f"Binary is_fraud distribution:\n"
        f"  1 (fraud):     {pos} ({pos_pct:.1f}%)\n"
        f"  0 (legitimate): {neg} ({100 - pos_pct:.1f}%)\n"
        f"\n"
        f"Multi-class label_5way distribution (kept rows only):\n"
        f"{multiclass}\n"
    )


def main() -> None:
    df = pd.read_csv(INPUT)
    labeled = build_targets(df)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(OUTPUT, index=False)

    print(f"input:  {INPUT.relative_to(ROOT)}")
    print(f"output: {OUTPUT.relative_to(ROOT)}\n")
    print(summarize(labeled, original_size=len(df)))


if __name__ == "__main__":
    main()
