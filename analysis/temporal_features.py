import pandas as pd
import numpy as np

def derive_temporal_features(file_path):
    df = pd.read_csv(file_path)
    
    # 1. Convert timestamp to datetime
    df['timestamp_parsed'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Derive features
    df['hour'] = df['timestamp_parsed'].dt.hour
    df['weekday'] = df['timestamp_parsed'].dt.weekday
    
    # is_weekend: 5=Saturday, 6=Sunday
    df['is_weekend'] = (df['weekday'] >= 5).astype(float)
    
    # is_business_hours: e.g. 9 to 17 on weekdays
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] < 17) & (df['is_weekend'] == 0)).astype(float)
    
    # time_to_next_action_min is already provided in minutes
    if 'time_to_next_action_min' in df.columns:
        df['time_to_next_action_sec'] = df['time_to_next_action_min'] * 60

    return df

def analyze_correlation(df):
    fraud_labels = ['Confirmed_Fraud', 'High_Risk_Reported', 'confirmed_fraud', 'high_risk_reported']
    
    if 'label_5way' in df.columns:
        df['is_fraud'] = df['label_5way'].isin(fraud_labels).astype(int)
    else:
        print("label_5way column not found!")
        return
        
    features = ['hour', 'weekday', 'is_business_hours', 'is_weekend', 'time_to_next_action_min', 'time_to_next_action_sec']
    
    print("Correlation with target (is_fraud):")
    for feature in features:
        if feature in df.columns:
            corr = df[feature].corr(df['is_fraud'])
            print(f"{feature}: {corr:.4f}")

if __name__ == "__main__":
    file_path = 'data/cleaned_bank_calls.csv'
    df = derive_temporal_features(file_path)
    
    output_path = 'data/cleaned_bank_calls_with_temporal.csv'
    df.to_csv(output_path, index=False)
    print(f"Saved dataset with temporal features to {output_path}")
    
    analyze_correlation(df)
