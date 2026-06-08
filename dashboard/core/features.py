import pandas as pd

FEATURE_COLUMNS = [
    "caller_number",
    "called_number",
    "timestamp",
    "duration_sec",
    "was_answered",
    "channel",
    "was_hangup_by_client",
    "login_after_call",
    "twofa_confirmed_after_call",
    "transfer_after_call",
    "new_payee_after_call",
    "settings_changed_after_call",
    "time_to_next_action_min",
    "label_5way",
    "confidence_score",
]


def prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=FEATURE_COLUMNS)

    result = df.copy()
    result["caller_number"] = result["caller_number"].astype(str)
    result["called_number"] = result["called_number"].astype(str)
    if "timestamp" in result.columns:
        result["timestamp"] = pd.to_datetime(result["timestamp"], errors="coerce")

    return result[FEATURE_COLUMNS].copy()
