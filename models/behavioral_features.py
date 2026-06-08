from pathlib import Path

import pandas as pd

from dataset.utils import normalize_timestamp
from models.encoding import build_features
from models.prepare import time_aware_split

ROOT = Path(__file__).resolve().parent.parent
LABELED_PATH = ROOT / "data" / "labeled_bank_calls.csv"

TRAIN_OUT = ROOT / "data" / "train_features_behavioral.csv"
TEST_OUT = ROOT / "data" / "test_features_behavioral.csv"

KEYWORDS = [
    "pettus",
    "kahtlane",
    "turvakonto",
    "mobiil-id",
    "smart-id",
    "tehing",
    "kaart",
    "pank",
    "panga",
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    normalized = df["timestamp"].apply(lambda v: normalize_timestamp(v) if pd.notna(v) else v)
    df["timestamp"] = normalized
    ts = pd.to_datetime(normalized, errors="coerce")
    df["hour"] = ts.dt.hour
    df["weekday"] = ts.dt.weekday
    df["is_weekend"] = (df["weekday"] >= 5).astype(bool)
    df["is_business_hours"] = (df["hour"] >= 9) & (df["hour"] < 17) & (~df["is_weekend"])
    return df


def complaint_keyword_score(text) -> int:
    if pd.isna(text):
        return 0
    text = str(text).lower()
    return sum(1 for keyword in KEYWORDS if keyword in text)


def add_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp_dt", kind="mergesort").reset_index(drop=True)

    df["calls_from_number_last_24h"] = 0
    df["distinct_victims_last_7d"] = 0

    for caller, group in df.groupby("caller_number", sort=False):
        times = group["timestamp_dt"]
        indexes = group.index

        for idx, current_time in zip(indexes, times):
            if pd.isna(current_time):
                continue

            last_24h_mask = (times < current_time) & (times >= current_time - pd.Timedelta(hours=24))
            last_7d_mask = (times < current_time) & (times >= current_time - pd.Timedelta(days=7))

            df.at[idx, "calls_from_number_last_24h"] = int(last_24h_mask.sum())
            df.at[idx, "distinct_victims_last_7d"] = int(group.loc[last_7d_mask, "called_number"].nunique())

    df["mean_call_duration_per_number"] = (
        df.groupby("caller_number")["duration_sec"]
        .transform("mean")
    )

    df["complaint_keyword_score"] = df["complaint_text"].apply(complaint_keyword_score)

    df = df.drop(columns=["timestamp_dt"])

    return df


def main():
    df = pd.read_csv(LABELED_PATH)

    df = add_temporal_features(df)
    df = add_aggregate_features(df)

    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    df = build_features(df)

    if "confidence_score" in df.columns:
        df = df.drop(columns=["confidence_score"])

    train, test = time_aware_split(df)

    TRAIN_OUT.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)

    added = [
        "calls_from_number_last_24h",
        "distinct_victims_last_7d",
        "mean_call_duration_per_number",
        "complaint_keyword_score",
    ]

    print("Behavioral feature datasets written:")
    print(f"  {TRAIN_OUT.relative_to(ROOT)}")
    print(f"  {TEST_OUT.relative_to(ROOT)}")
    print("\nAdded features:")
    for col in added:
        print(f"  {col}")


if __name__ == "__main__":
    main()