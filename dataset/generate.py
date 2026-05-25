"""
Synthetic dataset generator for the Võltsitud Pangakõned defense project.

Produces a CSV that mirrors the data model described in the course reference
material on fake bank calls. See docs/methodology_alignment.md for the
mapping between the course methodology sections and the columns produced
here.

Design goals:
    1. Reflect the three data groups from section 4 of the topic brief:
       call event, post-call customer behavior, complaint/text data.
    2. Embed the risk signals from section 5 (mass calls from one number,
       authentication or payment right after the call, temporal clustering)
       so downstream EDA can rediscover them.
    3. Use the 5-class label scheme from section 7 plus a confidence score
       as recommended in the general ML methodology document.
    4. Inject the kinds of quality issues the Sprint 1 cleaning tasks
       expect: malformed timestamps, NA, duplicates, inconsistent spellings.

Run:
    python dataset/generate.py
Output:
    data/raw_bank_calls.csv
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 42
DEFAULT_ROWS = 5000
DEFAULT_FRAUD_RATE = 0.15
DATA_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw_bank_calls.csv"

# Known legitimate bank service numbers (verified_legitimate class).
# A small whitelist mirrors section 8 of the topic brief: negative examples
# must be explicitly verified, not just "no complaint".
LEGITIMATE_SERVICE_NUMBERS = [
    "+3726310310",  # SEB
    "+3726310410",  # Swedbank
    "+3726800800",  # LHV
    "+3726123456",  # Coop Pank
    "+3726998000",  # Luminor
]

# Channels used in caller_id metadata. Spellings deliberately vary so that
# Sprint 1 task "Standardize categorical values" has something to clean.
CHANNEL_VARIANTS = ["voip", "VoIP", "voIP", "mobile", "Mobile", "MOBILE", "landline", "Landline"]

LABEL_CLASSES = [
    "confirmed_fraud",
    "high_risk_reported",
    "community_flagged",
    "unknown",
    "verified_legitimate",
]

# Mixed casing for label spellings — second source of categorical noise.
LABEL_VARIANTS = {
    "confirmed_fraud": ["confirmed_fraud", "Confirmed_Fraud", "CONFIRMED_FRAUD"],
    "high_risk_reported": ["high_risk_reported", "High_Risk_Reported", "high-risk-reported"],
    "community_flagged": ["community_flagged", "Community_Flagged"],
    "unknown": ["unknown", "Unknown", "UNKNOWN"],
    "verified_legitimate": ["verified_legitimate", "Verified_Legitimate", "verified-legitimate"],
}

# Fraud campaign generators. Each campaign reuses one caller number across
# many victims within a short time window — this is the textbook mass-fraud
# signal called out in section 5 of the topic brief.
NUM_FRAUD_CAMPAIGNS = 50
VICTIMS_PER_CAMPAIGN = (30, 100)  # min, max victims per campaign

# Complaint text templates in Estonian. The keywords ("pank", "pettus",
# "kahtlane", "turvakonto", "Mobiil-ID", "Smart-ID") are taken directly from
# section 4 of the topic brief to make Sprint 2 keyword analysis findable.
FRAUD_COMPLAINTS = [
    "Helistas keegi panga nimel, küsis kaardi andmeid. Väga kahtlane.",
    "Kõneleja palus kinnitada Mobiil-ID toiming, väitis et tegu on turvakontrolliga.",
    "Pettus, klient suunati raha kandma turvakontole.",
    "Kahtlane tehing teatatud, helistaja ütles et konto on ohus.",
    "Klient kinnitas Smart-ID, hiljem märkas raha mahaarvamist.",
    "Helistaja survestas otsustama kiiresti, ei lubanud kõnet katkestada.",
    "Pankur palus loobuda kahtlasest tehingust, klient kinnitas tundliku info.",
]

LEGITIMATE_COMPLAINTS = [
    "Pank tuletas meelde kaardi aegumist, info kontrollitud.",
    "Klienditeenindus selgitas tehingu staatust, kõik korras.",
    "Reklaamkõne uue hoiusepakkumise kohta.",
    "Panga kinnitus laenupakkumise kohta.",
    "",  # most legitimate calls do not generate a complaint
    "",
    "",
]


def _random_estonian_mobile(rng: np.random.Generator) -> str:
    """Random Estonian-format mobile number (+3725XXXXXXX)."""
    return f"+3725{rng.integers(1_000_000, 9_999_999)}"


def _random_estonian_called(rng: np.random.Generator) -> str:
    """Random Estonian-format called number (any mobile)."""
    return f"+3725{rng.integers(1_000_000, 9_999_999)}"


def _format_timestamp_corrupted(ts: pd.Timestamp, rng: np.random.Generator) -> str:
    """Render one timestamp in one of several legal-looking formats.

    Used to seed the Sprint 1 normalization task with realistic noise:
    ISO, slash-separated, dash-separated with 2-digit year, ISO with
    timezone, epoch seconds, and a free-form Estonian phrasing.
    """
    formats = [
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        ts.strftime("%d/%m/%Y %H:%M"),
        ts.strftime("%m-%d-%y %H:%M:%S"),
        ts.strftime("%Y-%m-%dT%H:%M:%S+02:00"),
        str(int(ts.timestamp())),
        ts.strftime("%d.%m.%Y kell %H:%M"),
    ]
    weights = [0.55, 0.15, 0.10, 0.10, 0.05, 0.05]
    idx = rng.choice(len(formats), p=weights)
    return formats[idx]


def _make_campaign_caller_numbers(rng: np.random.Generator, n: int) -> list[str]:
    """Pool of caller numbers reused across mass fraud campaigns."""
    return [_random_estonian_mobile(rng) for _ in range(n)]


def _label_for_campaign(rng: np.random.Generator) -> str:
    """Label distribution for calls produced by a mass fraud campaign.

    Campaign calls skew heavily toward confirmed and high-risk labels;
    a smaller tail lands in community_flagged or unknown when complaints
    were inconclusive.
    """
    return rng.choice(
        ["confirmed_fraud", "high_risk_reported", "community_flagged", "unknown"],
        p=[0.45, 0.30, 0.15, 0.10],
    )


def _label_for_organic(rng: np.random.Generator) -> str:
    """Label distribution for organic (non-campaign) calls.

    Most organic traffic is legitimate or unverified; a small share are
    community-flagged or individual fraud attempts that did not belong
    to a recognized campaign.
    """
    return rng.choice(
        ["verified_legitimate", "unknown", "community_flagged", "confirmed_fraud"],
        p=[0.45, 0.40, 0.10, 0.05],
    )


def _confidence_for_label(label: str, rng: np.random.Generator) -> float:
    """Confidence score per label class — higher for verified states.

    Modelled on the "Confidence score" recommendation in the general
    ML methodology document: complaint-driven sources should carry a
    score reflecting how strong the evidence behind the label is.
    """
    centers = {
        "confirmed_fraud": 0.95,
        "verified_legitimate": 0.95,
        "high_risk_reported": 0.75,
        "community_flagged": 0.55,
        "unknown": 0.35,
    }
    noise = rng.normal(0, 0.05)
    return float(np.clip(centers[label] + noise, 0.0, 1.0))


def _post_call_behavior(label: str, rng: np.random.Generator) -> dict:
    """Generate post-call customer actions linked to the label.

    Fraud labels are far more likely to produce login/transfer/new-payee
    events shortly after the call — this is the "Järgnev sündmusahel"
    feature group from section 6 of the topic brief. The generator
    intentionally tightens the time-to-next-action window for fraud so
    that downstream models can pick up the temporal coupling.
    """
    if label in ("confirmed_fraud", "high_risk_reported"):
        prob_login = 0.75
        prob_2fa = 0.65
        prob_transfer = 0.55
        prob_new_payee = 0.35
        prob_settings = 0.20
        time_to_next_min = float(np.clip(rng.normal(8, 5), 0.5, 120))
    elif label == "community_flagged":
        prob_login = 0.30
        prob_2fa = 0.15
        prob_transfer = 0.10
        prob_new_payee = 0.05
        prob_settings = 0.05
        time_to_next_min = float(np.clip(rng.normal(45, 30), 1, 480))
    elif label == "verified_legitimate":
        prob_login = 0.20
        prob_2fa = 0.10
        prob_transfer = 0.05
        prob_new_payee = 0.02
        prob_settings = 0.05
        time_to_next_min = float(np.clip(rng.normal(120, 80), 1, 720))
    else:  # unknown
        prob_login = 0.15
        prob_2fa = 0.08
        prob_transfer = 0.04
        prob_new_payee = 0.02
        prob_settings = 0.03
        time_to_next_min = float(np.clip(rng.normal(200, 120), 1, 1440))

    return {
        "login_after_call": rng.random() < prob_login,
        "twofa_confirmed_after_call": rng.random() < prob_2fa,
        "transfer_after_call": rng.random() < prob_transfer,
        "new_payee_after_call": rng.random() < prob_new_payee,
        "settings_changed_after_call": rng.random() < prob_settings,
        "time_to_next_action_min": time_to_next_min,
    }


def _complaint_text(label: str, rng: np.random.Generator) -> str:
    """Pick a complaint text consistent with the label.

    Fraud labels always produce a complaint; community-flagged calls
    produce one ~60% of the time; legitimate or unknown calls usually
    have no complaint at all.
    """
    if label in ("confirmed_fraud", "high_risk_reported"):
        return str(rng.choice(FRAUD_COMPLAINTS))
    if label == "community_flagged":
        if rng.random() < 0.6:
            return str(rng.choice(FRAUD_COMPLAINTS))
        return ""
    return str(rng.choice(LEGITIMATE_COMPLAINTS))


def _random_call_hour(label: str, rng: np.random.Generator) -> int:
    """Sample a call hour with class-dependent shape.

    Fraud calls cluster during business hours (10-15) when customers are
    available and likely to act on urgent prompts. Legitimate calls are
    spread more evenly across the working day.
    """
    if label in ("confirmed_fraud", "high_risk_reported", "community_flagged"):
        return int(np.clip(rng.normal(12, 2), 7, 20))
    return int(np.clip(rng.normal(13, 4), 6, 22))


def _generate_rows(n_rows: int, fraud_rate: float, rng: np.random.Generator) -> list[dict]:
    """Build the raw rows before any noise injection.

    Two phases: first emit fraud campaigns (one caller, many victims,
    tight time window); then fill the remaining quota with organic
    rows drawn from the legitimate-service pool or one-off numbers.
    """
    n_fraud_rows = int(n_rows * fraud_rate)
    n_organic_rows = n_rows - n_fraud_rows

    campaign_numbers = _make_campaign_caller_numbers(rng, NUM_FRAUD_CAMPAIGNS)
    rows: list[dict] = []
    start_date = datetime(2026, 1, 1)
    horizon_days = 120

    remaining_fraud = n_fraud_rows
    while remaining_fraud > 0:
        caller = str(rng.choice(campaign_numbers))
        victims = int(rng.integers(*VICTIMS_PER_CAMPAIGN))
        victims = min(victims, remaining_fraud)
        campaign_day = int(rng.integers(0, horizon_days))
        for _ in range(victims):
            label = _label_for_campaign(rng)
            hour = _random_call_hour(label, rng)
            ts = start_date + timedelta(
                days=campaign_day,
                hours=hour,
                minutes=int(rng.integers(0, 60)),
                seconds=int(rng.integers(0, 60)),
            )
            rows.append(_assemble_row(rng, caller, ts, label))
            remaining_fraud -= 1
            if remaining_fraud <= 0:
                break

    for _ in range(n_organic_rows):
        # 30% of organic traffic is a known legitimate service number,
        # 70% is a random one-off number with a noisier label mix.
        if rng.random() < 0.30:
            caller = str(rng.choice(LEGITIMATE_SERVICE_NUMBERS))
            label = "verified_legitimate"
        else:
            caller = _random_estonian_mobile(rng)
            label = _label_for_organic(rng)
        hour = _random_call_hour(label, rng)
        ts = start_date + timedelta(
            days=int(rng.integers(0, horizon_days)),
            hours=hour,
            minutes=int(rng.integers(0, 60)),
            seconds=int(rng.integers(0, 60)),
        )
        rows.append(_assemble_row(rng, caller, ts, label))

    rng.shuffle(rows)
    return rows


def _assemble_row(rng: np.random.Generator, caller: str, ts: datetime, label: str) -> dict:
    """Assemble one full row from the components above."""
    duration_sec = int(np.clip(rng.normal(180, 90), 5, 1200))
    was_answered = rng.random() < 0.85
    behavior = _post_call_behavior(label, rng)
    channel = str(rng.choice(CHANNEL_VARIANTS))
    label_text = str(rng.choice(LABEL_VARIANTS[label]))
    return {
        "call_id": f"C{rng.integers(10_000_000, 99_999_999)}",
        "caller_number": caller,
        "called_number": _random_estonian_called(rng),
        "timestamp": ts,
        "duration_sec": duration_sec,
        "was_answered": was_answered,
        "channel": channel,
        "was_hangup_by_client": rng.random() < 0.40,
        **behavior,
        "complaint_text": _complaint_text(label, rng),
        "manual_severity": int(rng.integers(1, 6)),
        "label_5way": label_text,
        "confidence_score": _confidence_for_label(label, rng),
    }


def _inject_noise(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Apply the quality issues the Sprint 1 cleaning tasks expect.

    Order matters: first corrupt timestamp formats, then null out a
    selection of columns, then clone exact duplicates, then create
    near-duplicates with small numeric drift.
    """
    df = df.copy()
    n = len(df)

    df["timestamp"] = df["timestamp"].apply(lambda t: t.strftime("%Y-%m-%d %H:%M:%S"))
    corrupt_idx = rng.choice(n, size=int(n * 0.05), replace=False)
    for i in corrupt_idx:
        df.at[i, "timestamp"] = _format_timestamp_corrupted(
            pd.Timestamp(df.at[i, "timestamp"]), rng
        )

    na_targets = {
        "channel": 0.04,
        "duration_sec": 0.02,
        "complaint_text": 0.08,
        "manual_severity": 0.06,
        "time_to_next_action_min": 0.05,
    }
    for col, rate in na_targets.items():
        mask = rng.random(n) < rate
        df.loc[mask, col] = np.nan

    dup_idx = rng.choice(n, size=int(n * 0.02), replace=False)
    duplicates = df.iloc[dup_idx].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    near_idx = rng.choice(len(df), size=int(len(df) * 0.01), replace=False)
    near = df.iloc[near_idx].copy().reset_index(drop=True)
    near["duration_sec"] = near["duration_sec"].apply(
        lambda d: (d + rng.integers(-2, 3)) if pd.notna(d) else d
    )
    df = pd.concat([df, near], ignore_index=True)

    df = df.sample(frac=1, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
    return df


def generate_dataset(
    n_rows: int = DEFAULT_ROWS,
    fraud_rate: float = DEFAULT_FRAUD_RATE,
    seed: int = RNG_SEED,
    output: Path = DATA_OUTPUT,
) -> pd.DataFrame:
    """Generate the dataset and write it to disk.

    Returns the resulting DataFrame so callers can run further checks
    in memory without re-reading the CSV.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    rows = _generate_rows(n_rows, fraud_rate, rng)
    df = pd.DataFrame(rows)
    df = _inject_noise(df, rng)

    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
    return df


def _summary(df: pd.DataFrame) -> str:
    """One-line-per-fact summary of the generated frame."""
    fraud_labels = ["confirmed_fraud", "Confirmed_Fraud", "CONFIRMED_FRAUD",
                    "high_risk_reported", "High_Risk_Reported", "high-risk-reported"]
    fraud_rate = df["label_5way"].isin(fraud_labels).mean()
    return (
        f"rows: {len(df)}\n"
        f"columns: {len(df.columns)}\n"
        f"fraud_rate: {fraud_rate:.3f}\n"
        f"NA cells: {df.isna().sum().sum()}\n"
        f"label distribution:\n{df['label_5way'].value_counts().to_string()}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--fraud-rate", type=float, default=DEFAULT_FRAUD_RATE)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--output", type=Path, default=DATA_OUTPUT)
    args = parser.parse_args()

    df = generate_dataset(
        n_rows=args.rows,
        fraud_rate=args.fraud_rate,
        seed=args.seed,
        output=args.output,
    )
    print(f"saved {args.output}")
    print(_summary(df))


if __name__ == "__main__":
    main()
