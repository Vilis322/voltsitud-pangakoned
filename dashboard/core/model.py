import pandas as pd


def risk_level(score: float | None) -> str:
    if score is None:
        return "Unknown"
    if score >= 0.75:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def normalize_score(score: float | None) -> float | None:
    if score is None or pd.isna(score):
        return None
    return float(min(max(score, 0.0), 1.0))


def aggregate_risk_score(series: pd.Series) -> float | None:
    if series is None or len(series) == 0:
        return None

    valid_scores = [float(value) for value in series.tolist() if pd.notna(value)]
    if not valid_scores:
        return None

    return sum(valid_scores) / len(valid_scores)


def lookup_risk_info(history_df: pd.DataFrame) -> dict:
    score = aggregate_risk_score(history_df.get("confidence_score", pd.Series(dtype=float)))
    return {
        "confidence_score": normalize_score(score),
        "risk_level": risk_level(score),
        "history_count": len(history_df),
    }
