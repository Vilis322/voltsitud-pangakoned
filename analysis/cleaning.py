import pandas as pd

# Normalization mappings (module-level so other modules can import and reuse).
CHANNEL_MAP = {
    'landline': 'Landline',
    'MOBILE': 'Mobile',
    'mobile': 'Mobile',
    'voip': 'VoIP',
    'voIP': 'VoIP'
}

LABEL_MAP = {
    'unknown': 'Unknown',
    'UNKNOWN': 'Unknown',
    'CONFIRMED_FRAUD': 'Confirmed_Fraud',
    'confirmed_fraud': 'Confirmed_Fraud',
    'verified-legitimate': 'Verified_Legitimate',
    'verified_legitimate': 'Verified_Legitimate',
    'high-risk-reported': 'High_Risk_Reported',
    'high_risk_reported': 'High_Risk_Reported',
    'community_flagged': 'Community_Flagged',
    'community-flagged': 'Community_Flagged'
}


def normalize_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with channel/label_5way standardized to canonical form."""
    df = df.copy()
    df['channel'] = df['channel'].replace(CHANNEL_MAP)
    df['label_5way'] = df['label_5way'].replace(LABEL_MAP)
    return df


def main() -> None:
    df = pd.read_csv('data/raw_bank_calls.csv')
    cleaned = normalize_categoricals(df)
    cleaned.to_csv('data/cleaned_bank_calls.csv', index=False)
    print("Dataset cleaned and saved to data/cleaned_bank_calls.csv")


if __name__ == "__main__":
    main()
