import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def main():
    # Load data
    df = pd.read_csv('data/cleaned_bank_calls_with_temporal.csv')
    
    # Identify numerical columns, excluding identifiers and targets
    exclude_cols = ['caller_number', 'called_number', 'confidence_score']
    # Select numeric datatypes
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out excluded columns
    num_cols = [c for c in num_cols if c not in exclude_cols]
    
    print(f"Numerical columns to scale: {num_cols}")
    
    # Split the dataset
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    
    # Initialize scaler
    scaler = StandardScaler()
    
    # Fit only on training data, then transform both
    train_df_scaled = train_df.copy()
    test_df_scaled = test_df.copy()
    
    train_df_scaled[num_cols] = scaler.fit_transform(train_df[num_cols])
    test_df_scaled[num_cols] = scaler.transform(test_df[num_cols])
    
    # Save the scaler using joblib
    joblib.dump(scaler, 'models/scaler.joblib')
    print("\nScaler successfully persisted to models/scaler.joblib")
    
    # Verify results on train set
    print("\nVerification on Train Set (should be mean ~ 0, std ~ 1):")
    train_means = train_df_scaled[num_cols].mean()
    train_stds = train_df_scaled[num_cols].std()
    
    for col in num_cols:
        print(f"  {col}: mean = {train_means[col]:.4f}, std = {train_stds[col]:.4f}")
        
    # Verify results on test set
    print("\nVerification on Test Set:")
    test_means = test_df_scaled[num_cols].mean()
    test_stds = test_df_scaled[num_cols].std()
    
    for col in num_cols:
        print(f"  {col}: mean = {test_means[col]:.4f}, std = {test_stds[col]:.4f}")
        
    # Save the scaled data
    train_df_scaled.to_csv('data/train_scaled.csv', index=False)
    test_df_scaled.to_csv('data/test_scaled.csv', index=False)
    print("\nSaved scaled train and test sets to data/train_scaled.csv and data/test_scaled.csv")

if __name__ == "__main__":
    main()
