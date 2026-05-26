"""
Top suspicious caller numbers analysis (VP-7 / Issue #8).

For every caller_number in the cleaned dataset, compute:

    total_calls       — how many calls came from the number
    distinct_victims  — how many distinct customers it reached
    fraud_rate        — share of calls labelled as fraud
    avg_duration_sec  — mean call duration in seconds

Then keep numbers with at least MIN_CALLS_TO_RANK calls (cuts noise from
one-offs) and produce the top-20 by (fraud_rate, total_calls). Renders
a horizontal bar chart with colour-coded fraud rate and prints a short
pattern discussion.

Run (from project root):
    python -m analysis.suspicious_callers
Output:
    docs/figures/top_suspicious_callers.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "cleaned_bank_calls.csv"
FIG_OUT = ROOT / "docs" / "figures" / "top_suspicious_callers.png"

# Labels treated as fraud for the rate computation. Names are the
# CamelCase canonical form produced by analysis/cleaning.py.
FRAUD_LABELS = {"Confirmed_Fraud", "High_Risk_Reported"}

# Numbers with fewer calls than this are excluded from the ranking so
# the chart is not dominated by single-call outliers with 100% fraud_rate.
MIN_CALLS_TO_RANK = 20

TOP_N = 20


def aggregate_by_caller(df: pd.DataFrame) -> pd.DataFrame:
    """Per-caller volume, distinct victims, fraud share and avg duration.

    Returns a DataFrame indexed by caller_number, sorted by fraud_rate
    descending and total_calls descending as the tiebreaker.
    """
    df = df.copy()
    df["_is_fraud"] = df["label_5way"].isin(FRAUD_LABELS)
    agg = df.groupby("caller_number").agg(
        total_calls=("call_id", "count"),
        distinct_victims=("called_number", "nunique"),
        fraud_rate=("_is_fraud", "mean"),
        avg_duration_sec=("duration_sec", "mean"),
    )
    return agg.sort_values(
        ["fraud_rate", "total_calls"], ascending=[False, False]
    )


def top_suspicious(
    agg: pd.DataFrame, n: int = TOP_N, min_calls: int = MIN_CALLS_TO_RANK
) -> pd.DataFrame:
    """Filter low-volume noise and take the top-n by the current order."""
    return agg[agg["total_calls"] >= min_calls].head(n)


def plot_top(top: pd.DataFrame, output_path: Path) -> None:
    """Horizontal bar chart of total_calls per caller, coloured by fraud rate.

    Red = >50% fraud, orange = 20-50%, blue = <20%. Each bar is annotated
    with the fraud percentage and distinct-victim count for quick reading.
    """
    fig, ax = plt.subplots(figsize=(11, 9))
    colors = [
        "#c0392b" if r > 0.5 else "#e67e22" if r > 0.2 else "#2980b9"
        for r in top["fraud_rate"]
    ]
    ax.barh(top.index.astype(str), top["total_calls"], color=colors)

    for i, (_, row) in enumerate(top.iterrows()):
        ax.text(
            row["total_calls"],
            i,
            f"  {row['fraud_rate']:.0%} fraud · {int(row['distinct_victims'])} victims",
            va="center",
            fontsize=9,
        )

    ax.set_xlabel("Total calls")
    ax.set_ylabel("Caller number")
    ax.set_title(
        f"Top {len(top)} suspicious caller numbers "
        f"(min {MIN_CALLS_TO_RANK} calls, ranked by fraud rate)"
    )
    ax.invert_yaxis()

    legend_items = [
        plt.Rectangle((0, 0), 1, 1, color="#c0392b", label=">50% fraud"),
        plt.Rectangle((0, 0), 1, 1, color="#e67e22", label="20-50% fraud"),
        plt.Rectangle((0, 0), 1, 1, color="#2980b9", label="<20% fraud"),
    ]
    ax.legend(handles=legend_items, loc="lower right")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def discuss(top: pd.DataFrame) -> None:
    """Print a short narrative summary of the patterns visible in the ranking."""
    mass = top[(top["fraud_rate"] > 0.5) & (top["total_calls"] >= 50)]
    targeted = top[(top["fraud_rate"] > 0.5) & (top["total_calls"] < 30)]
    legit = top[top["fraud_rate"] < 0.1]

    print("\n=== Pattern discussion ===")
    print(
        f"Mass campaigns ( >50% fraud, >=50 calls ): {len(mass)} numbers, "
        f"{int(mass['distinct_victims'].sum())} distinct victims combined."
    )
    print(
        f"Targeted attempts ( >50% fraud, <30 calls ): {len(targeted)} numbers — "
        f"likely individual scammers rather than mass campaigns."
    )
    if len(legit):
        print("Legitimate-looking high-volume numbers ( <10% fraud ):")
        for number, row in legit.iterrows():
            print(
                f"  {number}: {int(row['total_calls'])} calls, "
                f"{row['fraud_rate']:.1%} fraud — likely a bank service number."
            )


def main() -> None:
    df = pd.read_csv(DATA)
    agg = aggregate_by_caller(df)
    top = top_suspicious(agg, n=TOP_N)

    print(f"loaded {DATA} ({len(df)} rows, {df['caller_number'].nunique()} unique callers)")
    print(f"\nTop {len(top)} ranking:")
    print(top.to_string())

    plot_top(top, FIG_OUT)
    print(f"\nSaved chart to {FIG_OUT.relative_to(ROOT)}")

    discuss(top)


if __name__ == "__main__":
    main()
