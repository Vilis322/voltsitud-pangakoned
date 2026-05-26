import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add project root to sys.path to allow importing from dataset
root_path = Path(__file__).resolve().parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dataset.utils import normalize_timestamp

# Load dataset
df = pd.read_csv('data/cleaned_bank_calls.csv')

# Preprocess timestamps using normalized utility
df['timestamp'] = df['timestamp'].apply(normalize_timestamp)
df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

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
plt.savefig('docs/figures/fraud_call_frequency.png')
print("Analysis saved to docs/figures/fraud_call_frequency.png")
