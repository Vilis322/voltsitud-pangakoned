import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('data/cleaned_bank_calls.csv')

# Preprocess timestamps to handle non-standard formats
# Remove 'kell' and standardize common separators
df['timestamp'] = df['timestamp'].astype(str).str.replace(' kell ', ' ', regex=False)
df['timestamp'] = df['timestamp'].str.replace('.', '-', regex=False)
df['timestamp'] = df['timestamp'].str.replace('/', '-', regex=False)

# Convert timestamp to datetime, coercing errors to NaT
df['timestamp'] = pd.to_datetime(df['timestamp'], format='mixed', errors='coerce', utc=True)

# Drop rows where timestamp couldn't be parsed
df = df.dropna(subset=['timestamp'])

# Extract hour and day of week
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.day_name()

# Define order for day of week
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Filter data for fraud vs legitimate
# Assuming 'Confirmed_Fraud' is the fraud class
fraud_data = df[df['label_5way'] == 'Confirmed_Fraud']
legit_data = df[df['label_5way'] == 'Verified_Legitimate']

# Aggregate by hour and day
fraud_counts = fraud_data.groupby(['day_of_week', 'hour']).size().reset_index(name='count')

# Pivot for heatmap
pivot_fraud = fraud_counts.pivot(index='day_of_week', columns='hour', values='count').reindex(day_order)

# Plotting
plt.figure(figsize=(12, 6))
sns.heatmap(pivot_fraud, annot=True, fmt='g', cmap='YlOrRd')
plt.title('Fraud Call Volume by Day of Week and Hour')
plt.savefig('data/fraud_call_frequency.png')
print("Analysis saved to data/fraud_call_frequency.png")
