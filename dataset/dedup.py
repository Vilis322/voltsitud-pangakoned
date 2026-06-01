"""
Duplicate detection and removal for the raw bank-call dataset.

Two duplicate classes are handled:

    1. Exact duplicates — every column identical (including duration_sec).
       These come from the generator cloning rows verbatim.
    2. Near-duplicates — same caller / called / minute-rounded timestamp,
       with duration_sec drifting by only a few seconds. These come from
       the generator perturbing a small batch with small numeric noise.

The dedup runs in two passes so each class can be counted independently
and reported. Order matters: exact duplicates are removed first, then
near-duplicates are detected on what remains.

Run (from project root):
    python -m dataset.dedup [--input data/raw_bank_calls.csv]
                            [--output data/dedup_bank_calls.csv]
                            [--report docs/dedup_report.md]
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dataset.utils import normalize_timestamp

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DOCS_DIR = Path(__file__).resolve().parent.parent / "docs"
DEFAULT_INPUT = DATA_DIR / "raw_bank_calls.csv"
DEFAULT_OUTPUT = DATA_DIR / "dedup_bank_calls.csv"
DEFAULT_REPORT = DOCS_DIR / "dedup_report.md"

# Composite key required by the task spec (excluding duration so we can
# separately reason about near-dupes that differ only in duration).
GROUP_KEY = ["caller_number", "called_number", "timestamp_minute"]

# How close two durations must be (in seconds) to count as a near-duplicate.
NEAR_DURATION_TOLERANCE_SEC = 5


@dataclass
class DedupReport:
    """Aggregated counters produced by a single dedup run."""

    rows_before: int
    exact_duplicates: int
    near_duplicates: int
    rows_after: int
    near_dup_groups: int

    def as_markdown(self) -> str:
        removed = self.exact_duplicates + self.near_duplicates
        pct = 100 * removed / self.rows_before if self.rows_before else 0.0
        return (
            "# Deduplication report\n\n"
            f"Input file: {self.rows_before} rows.\n\n"
            "| Metric | Value |\n"
            "|---|---|\n"
            f"| Rows before | {self.rows_before} |\n"
            f"| Exact duplicates removed | {self.exact_duplicates} |\n"
            f"| Near-duplicates removed | {self.near_duplicates} |\n"
            f"| Distinct near-dup groups | {self.near_dup_groups} |\n"
            f"| Rows after | {self.rows_after} |\n"
            f"| Total removed | {removed} ({pct:.2f}%) |\n\n"
            "## Definitions\n\n"
            "- **Exact duplicate** — every column is identical to a row that\n"
            "  appeared earlier in the file. Removed via `df.duplicated(keep='first')`.\n"
            "- **Near-duplicate** — same `caller_number`, `called_number`, and\n"
            "  timestamp rounded to the nearest minute, with `duration_sec`\n"
            "  within ±5 s of an earlier row in the same group. These are the\n"
            "  rows the generator perturbed by a small numeric offset.\n\n"
            "## Removal order\n\n"
            "Exact duplicates are removed first; near-duplicate detection is\n"
            "then run on the surviving rows. This keeps the two counts disjoint\n"
            "and avoids double-counting a row that is both a strict duplicate\n"
            "and inside a near-dup cluster.\n"
        )


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the helper columns used by both passes.

    Specifically, parse the timestamp via the Sprint 1 normalizer and add a
    minute-rounded variant used for grouping. The original `timestamp`
    column is left untouched so the cleaned CSV preserves the input value.
    """
    df = df.copy()
    parsed = df["timestamp"].apply(
        lambda v: normalize_timestamp(v) if pd.notna(v) else pd.NaT
    )
    df["_timestamp_iso"] = parsed
    df["timestamp_minute"] = pd.to_datetime(parsed, errors="coerce").dt.floor("min")
    return df


def remove_exact_duplicates(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop rows that are bit-identical to an earlier row.

    Returns the surviving frame and the number of rows removed.
    """
    mask = df.duplicated(keep="first")
    return df.loc[~mask].copy(), int(mask.sum())


def remove_near_duplicates(
    df: pd.DataFrame, duration_tolerance_sec: int = NEAR_DURATION_TOLERANCE_SEC,
) -> tuple[pd.DataFrame, int, int]:
    """Detect and drop near-duplicates within each composite-key group.

    Vectorized: sort the frame by the group key plus `duration_sec`, then
    compute the per-group consecutive `duration_sec` difference. Any row
    whose difference to its in-group predecessor is at most
    `duration_tolerance_sec` is treated as a near-duplicate and dropped.
    Rows with a NaN difference (either the first of a group or NaN
    duration) are kept.

    O(N log N) overall (dominated by the sort), versus the nested-loop
    O(N^2 / |groups|) of the previous implementation.

    Returns (deduped_frame, rows_removed, distinct_groups_with_dupes).
    """
    sort_cols = GROUP_KEY + ["duration_sec"]
    sorted_df = df.sort_values(sort_cols, kind="mergesort", na_position="last")

    diffs = sorted_df.groupby(GROUP_KEY, dropna=False)["duration_sec"].diff()
    near_mask = diffs.notna() & (diffs.abs() <= duration_tolerance_sec)

    if near_mask.any():
        groups_with_dupes = (
            sorted_df.loc[near_mask, GROUP_KEY].drop_duplicates().shape[0]
        )
    else:
        groups_with_dupes = 0

    cleaned = sorted_df.loc[~near_mask].copy()
    return cleaned.reset_index(drop=True), int(near_mask.sum()), int(groups_with_dupes)


def dedup(df: pd.DataFrame) -> tuple[pd.DataFrame, DedupReport]:
    """Run both dedup passes and return cleaned frame + report counters."""
    rows_before = len(df)
    prepared = _prepare(df)
    after_exact, exact_count = remove_exact_duplicates(prepared)
    after_near, near_count, near_groups = remove_near_duplicates(after_exact)
    cleaned = after_near.drop(columns=["_timestamp_iso", "timestamp_minute"])
    report = DedupReport(
        rows_before=rows_before,
        exact_duplicates=exact_count,
        near_duplicates=near_count,
        near_dup_groups=near_groups,
        rows_after=len(cleaned),
    )
    return cleaned, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    cleaned, report = dedup(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False)
    args.report.write_text(report.as_markdown(), encoding="utf-8")

    print(f"input:  {args.input}  ({report.rows_before} rows)")
    print(f"output: {args.output} ({report.rows_after} rows)")
    print(f"report: {args.report}")
    print(
        f"removed: {report.exact_duplicates} exact + "
        f"{report.near_duplicates} near "
        f"(across {report.near_dup_groups} groups)"
    )


if __name__ == "__main__":
    main()
