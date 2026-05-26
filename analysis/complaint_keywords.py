"""
Complaint keyword analysis (VP-9 / Issue #10).

Extracts a small fixed vocabulary of fraud-related Estonian keywords
from the free-text ``complaint_text`` column, computes how often each
keyword appears per label class, and surfaces which keywords correlate
most strongly with ``Confirmed_Fraud``.

Output:
    docs/figures/complaint_keyword_frequencies.png  — grouped bar chart
                                                      of keyword counts by
                                                      label class.

Why a fixed vocabulary instead of a free word cloud:
    The course brief on the topic explicitly lists the fraud-signalling
    keywords ("pank", "pettus", "kahtlane", "turvakonto", "Mobiil-ID",
    "Smart-ID"). Restricting the analysis to that vocabulary keeps the
    output directly defensible against the source material and avoids
    noise from common stop-words in the Estonian text.

Run (from project root):
    python -m analysis.complaint_keywords
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cleaned_bank_calls.csv"
FIG_OUT = ROOT / "docs" / "figures" / "complaint_keyword_frequencies.png"

# Canonical label order in figures and tables.
LABEL_ORDER = [
    "Confirmed_Fraud",
    "High_Risk_Reported",
    "Community_Flagged",
    "Unknown",
    "Verified_Legitimate",
]

# Keyword vocabulary from the topic brief, section 4. Each value is a
# regex pattern, applied case-insensitively. Estonian inflection means we
# match common stems rather than exact tokens (``pank``, ``panga``,
# ``pangast`` all count as ``bank``).
KEYWORD_PATTERNS: dict[str, str] = {
    "bank (pank/panga)":         r"pang",
    "fraud (pettus)":            r"pettus",
    "suspicious (kahtlane)":     r"kahtlas",
    "secure account (turvakonto)": r"turvakont",
    "Mobiil-ID":                 r"mobiil[\-\s]?id",
    "Smart-ID":                  r"smart[\-\s]?id",
    "transaction (tehing)":      r"tehing",
    "card (kaart)":              r"kaardi|kaart",
}


def count_keywords(df: pd.DataFrame) -> pd.DataFrame:
    """Per-class count of complaint rows containing each keyword pattern.

    A row is counted once per keyword regardless of how many times the
    keyword appears in its text — we care about *prevalence*, not raw
    frequency, so a single ranting complaint doesn't dominate a class.
    """
    text = df["complaint_text"].fillna("").astype(str).str.lower()
    out_rows = []
    for label in LABEL_ORDER:
        label_text = text[df["label_5way"] == label]
        n_rows = len(label_text)
        row = {"label": label, "n_complaints": n_rows}
        for name, pattern in KEYWORD_PATTERNS.items():
            hits = label_text.str.contains(pattern, regex=True, na=False).sum()
            row[name] = int(hits)
        out_rows.append(row)
    return pd.DataFrame(out_rows).set_index("label")


def keyword_correlations(counts: pd.DataFrame) -> pd.DataFrame:
    """For each keyword, the share of Confirmed_Fraud complaints vs others.

    Returns a DataFrame with one row per keyword and columns
    ``fraud_share`` and ``other_share`` — useful for the defense to
    answer "which words separate fraud from legitimate the most".
    """
    keywords = [c for c in counts.columns if c != "n_complaints"]
    fraud_n = counts.loc["Confirmed_Fraud", "n_complaints"]
    other_n = counts["n_complaints"].drop("Confirmed_Fraud").sum()

    rows = []
    for kw in keywords:
        f_share = counts.loc["Confirmed_Fraud", kw] / fraud_n if fraud_n else 0.0
        o_share = (counts[kw].drop("Confirmed_Fraud").sum() / other_n) if other_n else 0.0
        rows.append({
            "keyword": kw,
            "fraud_share": round(f_share, 3),
            "other_share": round(o_share, 3),
            "lift": round(f_share / o_share, 2) if o_share else float("inf"),
        })
    return (
        pd.DataFrame(rows)
        .set_index("keyword")
        .sort_values("lift", ascending=False)
    )


def plot_grouped_bar(counts: pd.DataFrame, output_path: Path) -> None:
    """Grouped bar chart: x = keyword, one bar per label class.

    Heights are *shares* (hits / complaints in that class) so classes
    with very different complaint counts are comparable.
    """
    keywords = [c for c in counts.columns if c != "n_complaints"]
    shares = counts[keywords].div(counts["n_complaints"].replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(13, 6))
    bar_width = 0.15
    x = np.arange(len(keywords))
    colors = {
        "Confirmed_Fraud":     "#c0392b",
        "High_Risk_Reported":  "#e67e22",
        "Community_Flagged":   "#f1c40f",
        "Unknown":             "#95a5a6",
        "Verified_Legitimate": "#2980b9",
    }
    for i, label in enumerate(LABEL_ORDER):
        ax.bar(
            x + i * bar_width,
            shares.loc[label].values,
            width=bar_width,
            label=label,
            color=colors[label],
        )

    ax.set_xticks(x + bar_width * (len(LABEL_ORDER) - 1) / 2)
    ax.set_xticklabels(keywords, rotation=30, ha="right")
    ax.set_ylabel("Share of complaints containing the keyword")
    ax.set_title("Complaint keyword prevalence by label class")
    ax.legend(loc="upper right")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA)
    counts = count_keywords(df)

    print("Keyword hit counts by class:")
    print(counts.to_string())

    corr = keyword_correlations(counts)
    print("\nKeyword separation power (share among Confirmed_Fraud vs others):")
    print(corr.to_string())

    plot_grouped_bar(counts, FIG_OUT)
    print(f"\nSaved chart to {FIG_OUT.relative_to(ROOT)}")

    top3 = corr.head(3).index.tolist()
    print(
        f"\nTop 3 fraud-signalling keywords (by lift over other classes): "
        f"{', '.join(top3)}."
    )


if __name__ == "__main__":
    main()
