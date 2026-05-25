"""
Quality-assurance script for the generated raw bank-call dataset.

Reads the CSV produced by dataset/generate.py and runs a set of structural
and statistical checks. Exits with a non-zero status if any hard check
fails so the script is usable in CI.

Two kinds of checks are reported:

    [PASS/FAIL] hard checks — invariants the downstream pipeline relies on.
    [INFO]      statistics  — printed for inspection / defense.

Run:
    python dataset/qa.py
    python dataset/qa.py --input some/other.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DEFAULT = Path(__file__).resolve().parent.parent / "data" / "raw_bank_calls.csv"

REQUIRED_COLUMNS = [
    "call_id", "caller_number", "called_number", "timestamp",
    "duration_sec", "was_answered", "channel", "was_hangup_by_client",
    "login_after_call", "twofa_confirmed_after_call", "transfer_after_call",
    "new_payee_after_call", "settings_changed_after_call",
    "time_to_next_action_min", "complaint_text", "manual_severity",
    "label_5way", "confidence_score",
]

# Canonical (lower-case) fraud labels — case variants are normalized first.
FRAUD_LABELS_CANONICAL = {"confirmed_fraud", "high_risk_reported"}
ALL_LABELS_CANONICAL = {
    "confirmed_fraud", "high_risk_reported",
    "community_flagged", "unknown", "verified_legitimate",
}


def _canon_label(s: str) -> str:
    """Lower-case + replace dashes; matches the standardization sprint task."""
    if pd.isna(s):
        return s
    return str(s).strip().lower().replace("-", "_")


def run_checks(df: pd.DataFrame) -> tuple[list[tuple[str, bool, str]], list[tuple[str, str]]]:
    """Run hard checks + collect informational stats.

    Returns (hard_results, info_lines). Each hard result is
    (name, passed, message).
    """
    hard: list[tuple[str, bool, str]] = []
    info: list[tuple[str, str]] = []

    # ---- Structural checks ----
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    hard.append((
        "all required columns present",
        not missing_cols,
        f"missing: {missing_cols}" if missing_cols else "ok",
    ))

    # ---- Row count: should be near 5000 plus duplicate injection ~3%. ----
    row_count_ok = 4500 <= len(df) <= 5500
    hard.append((
        "row count within expected band (4500..5500)",
        row_count_ok,
        f"actual: {len(df)}",
    ))

    # ---- Fraud rate (canonicalize labels first). ----
    canon = df["label_5way"].apply(_canon_label)
    fraud_rate = canon.isin(FRAUD_LABELS_CANONICAL).mean()
    fraud_rate_ok = 0.10 <= fraud_rate <= 0.20
    hard.append((
        "fraud rate within band (10%..20%)",
        fraud_rate_ok,
        f"actual: {fraud_rate:.3f}",
    ))

    # ---- All labels are recognized (modulo case/dash variants). ----
    unknown_labels = set(canon.dropna().unique()) - ALL_LABELS_CANONICAL
    hard.append((
        "no unrecognized labels after canonicalization",
        not unknown_labels,
        f"found: {sorted(unknown_labels)}" if unknown_labels else "ok",
    ))

    # ---- duration_sec range. ----
    if "duration_sec" in df:
        valid = df["duration_sec"].dropna()
        dur_ok = (valid >= 0).all() and (valid <= 3600).all()
        hard.append((
            "duration_sec in [0, 3600] (or NA)",
            dur_ok,
            f"min={valid.min()}, max={valid.max()}",
        ))

    # ---- Confidence score in [0, 1]. ----
    if "confidence_score" in df:
        cs = df["confidence_score"].dropna()
        cs_ok = (cs >= 0).all() and (cs <= 1).all()
        hard.append((
            "confidence_score in [0, 1]",
            cs_ok,
            f"min={cs.min():.3f}, max={cs.max():.3f}",
        ))

    # ---- Informational stats ----
    info.append(("rows", str(len(df))))
    info.append(("columns", str(len(df.columns))))
    info.append(("NA cells (total)", str(int(df.isna().sum().sum()))))
    info.append(("NA per column", "\n" + df.isna().sum().to_string()))

    info.append((
        "label_5way distribution (raw, with case variants)",
        "\n" + df["label_5way"].value_counts(dropna=False).to_string(),
    ))
    info.append((
        "label_5way distribution (canonicalized)",
        "\n" + canon.value_counts(dropna=False).to_string(),
    ))

    # Duplicate accounting on the natural composite key.
    dup_key = ["caller_number", "called_number", "timestamp", "duration_sec"]
    exact_dupes = df.duplicated(subset=dup_key).sum()
    info.append(("exact duplicates on (caller, called, ts, dur)", str(int(exact_dupes))))

    # Mass-fraud signal: top callers by victim count, fraud rate per number.
    by_caller = (
        df.assign(_canon=canon)
          .groupby("caller_number")
          .agg(calls=("call_id", "count"),
               fraud_rate=("_canon", lambda s: s.isin(FRAUD_LABELS_CANONICAL).mean()))
          .sort_values("calls", ascending=False)
          .head(10)
    )
    info.append(("top 10 callers by volume", "\n" + by_caller.to_string()))

    # Post-call action share by canonical label.
    behavior_cols = [
        "login_after_call", "twofa_confirmed_after_call",
        "transfer_after_call", "new_payee_after_call",
        "settings_changed_after_call",
    ]
    behavior_by_label = (
        df.assign(_canon=canon)
          .groupby("_canon")[behavior_cols]
          .mean()
          .round(3)
    )
    info.append(("behavior share by label (canonical)", "\n" + behavior_by_label.to_string()))

    return hard, info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DEFAULT)
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[FAIL] file not found: {args.input}")
        print("       run `python dataset/generate.py` first.")
        return 2

    df = pd.read_csv(args.input)
    hard, info = run_checks(df)

    print(f"loaded {args.input} ({len(df)} rows, {len(df.columns)} cols)\n")

    print("=== hard checks ===")
    failed = 0
    for name, ok, msg in hard:
        tag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print(f"  [{tag}] {name}  ({msg})")

    print("\n=== info ===")
    for k, v in info:
        print(f"\n--- {k} ---{v if v.startswith(chr(10)) else ' ' + v}")

    print(f"\nresult: {len(hard) - failed}/{len(hard)} hard checks passed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
