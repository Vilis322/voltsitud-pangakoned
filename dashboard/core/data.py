from __future__ import annotations

from pathlib import Path
from datetime import date
import pandas as pd

from .config import CLEANED_CALLS, FILTER_COLUMNS
from .model import lookup_risk_info

try:
    import streamlit as st
except ImportError:
    class _DummyCacheDecorator:
        def __call__(self, func):
            return func

    st = type("st", (), {"cache_data": _DummyCacheDecorator()})


# =========================
# DATA LOADING
# =========================

@st.cache_data
def load_call_data() -> pd.DataFrame:
    if not CLEANED_CALLS.exists():
        return _mock_call_data()

    df = pd.read_csv(CLEANED_CALLS)

    # --- SAFE TIMESTAMP PARSING (важно) ---
    df[FILTER_COLUMNS["timestamp"]] = pd.to_datetime(
        df[FILTER_COLUMNS["timestamp"]],
        errors="coerce",
        format="mixed",
        utc=True
    )

    # --- type normalization ---
    df[FILTER_COLUMNS["caller_number"]] = df[FILTER_COLUMNS["caller_number"]].astype(str)
    df["called_number"] = df["called_number"].astype(str)

    return df


def _mock_call_data() -> pd.DataFrame:
    sample = [
        {
            "call_id": "C00000001",
            "caller_number": "37255063883",
            "called_number": "37255500001",
            "timestamp": pd.Timestamp("2026-01-15 09:00:00"),
            "duration_sec": 120.0,
            "was_answered": True,
            "channel": "Mobile",
            "was_hangup_by_client": False,
            "login_after_call": True,
            "twofa_confirmed_after_call": False,
            "transfer_after_call": False,
            "new_payee_after_call": False,
            "settings_changed_after_call": False,
            "time_to_next_action_min": 3.0,
            "complaint_text": "Mocked call history.",
            "manual_severity": 1.0,
            "label_5way": "Unknown",
            "confidence_score": 0.25,
        },
        {
            "call_id": "C00000002",
            "caller_number": "37255063883",
            "called_number": "37255500002",
            "timestamp": pd.Timestamp("2026-02-01 13:20:00"),
            "duration_sec": 65.0,
            "was_answered": True,
            "channel": "Landline",
            "was_hangup_by_client": True,
            "login_after_call": False,
            "twofa_confirmed_after_call": False,
            "transfer_after_call": False,
            "new_payee_after_call": False,
            "settings_changed_after_call": False,
            "time_to_next_action_min": 5.2,
            "complaint_text": "Mocked call history.",
            "manual_severity": 3.0,
            "label_5way": "Confirmed_Fraud",
            "confidence_score": 0.88,
        },
    ]
    return pd.DataFrame(sample)


# =========================
# META HELPERS
# =========================

def available_labels(df: pd.DataFrame) -> list[str]:
    return sorted(df[FILTER_COLUMNS["label"]].dropna().unique().tolist())


def available_channels(df: pd.DataFrame) -> list[str]:
    return sorted(df[FILTER_COLUMNS["channel"]].dropna().unique().tolist())


def available_date_range(df: pd.DataFrame) -> tuple[date, date]:
    series = pd.to_datetime(df[FILTER_COLUMNS["timestamp"]])

    start = series.min().date()
    end = series.max().date()

    return start, end


# =========================
# FILTERING
# =========================

from datetime import date
import pandas as pd


def filter_calls(
    df: pd.DataFrame,
    start_date: date | None = None,
    end_date: date | None = None,
    labels: list[str] | None = None,
    caller_prefix: str | None = None,
    channels: list[str] | None = None,
) -> pd.DataFrame:
    result = df.copy()

    # =========================
    # FIX: date -> UTC timestamp alignment
    # =========================
    if start_date is not None:
        start_ts = pd.to_datetime(start_date, utc=True)
        result = result[result[FILTER_COLUMNS["timestamp"]] >= start_ts]

    if end_date is not None:
        end_ts = (
            pd.to_datetime(end_date, utc=True)
            + pd.Timedelta(days=1)
            - pd.Timedelta(seconds=1)
        )
        result = result[result[FILTER_COLUMNS["timestamp"]] <= end_ts]

    # =========================
    # filters (unchanged)
    # =========================
    if labels:
        result = result[result[FILTER_COLUMNS["label"]].isin(labels)]

    if channels:
        result = result[result[FILTER_COLUMNS["channel"]].isin(channels)]

    if caller_prefix:
        prefix = "".join(ch for ch in str(caller_prefix) if ch.isdigit())
        if prefix:
            result = result[
                result[FILTER_COLUMNS["caller_number"]]
                .astype(str)
                .str.startswith(prefix)
            ]

    return result


# =========================
# LOOKUP LOGIC
# =========================

def lookup_number_history(df: pd.DataFrame, number: str) -> pd.DataFrame:
    if not number:
        return df.iloc[0:0]

    query = "".join(ch for ch in str(number) if ch.isdigit())
    if not query:
        return df.iloc[0:0]

    exact_mask = df[FILTER_COLUMNS["caller_number"]].astype(str) == query
    prefix_mask = df[FILTER_COLUMNS["caller_number"]].astype(str).str.startswith(query)

    history = df[exact_mask].copy() if exact_mask.any() else df[prefix_mask].copy()

    return history.sort_values(FILTER_COLUMNS["timestamp"], ascending=False)


def lookup_number_risk(df: pd.DataFrame, number: str) -> dict:
    history = lookup_number_history(df, number)
    risk = lookup_risk_info(history)

    return {
        "history": history,
        "risk": risk,
    }