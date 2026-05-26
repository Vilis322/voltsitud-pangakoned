import pandas as pd

# Load the dataset
file_path = 'data/raw_bank_calls.csv'
df = pd.read_csv(file_path)

# Normalization Mappings
channel_map = {
    'landline': 'Landline',
    'MOBILE': 'Mobile',
    'mobile': 'Mobile',
    'voip': 'VoIP',
    'voIP': 'VoIP'
}

label_map = {
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

# Apply mappings
df['channel'] = df['channel'].replace(channel_map)
df['label_5way'] = df['label_5way'].replace(label_map)

# Save the cleaned dataset
df.to_csv('data/cleaned_bank_calls.csv', index=False)
print("Dataset cleaned and saved to data/cleaned_bank_calls.csv")
