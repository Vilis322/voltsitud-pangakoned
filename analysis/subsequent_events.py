import pandas as pd
import matplotlib.pyplot as plt

def analyze_subsequent_events(file_path):
    df = pd.read_csv(file_path)
    
    # Define criteria: event occurred AND within 1 hour (60 minutes)
    df['login_within_1h'] = df['login_after_call'] & (df['time_to_next_action_min'] <= 60)
    df['transfer_within_1h'] = df['transfer_after_call'] & (df['time_to_next_action_min'] <= 60)
    df['new_payee_within_1h'] = df['new_payee_after_call'] & (df['time_to_next_action_min'] <= 60)
    
    # Calculate means (share of calls) per label
    results = df.groupby('label_5way')[['login_within_1h', 'transfer_within_1h', 'new_payee_within_1h']].mean()
    
    return results

def plot_results(results):
    results.plot(kind='bar', figsize=(10, 6))
    plt.title('Share of Subsequent Events within 1 Hour by Label Class')
    plt.ylabel('Share of Calls')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('docs/figures/subsequent_events_risk_signal.png')
    print("Chart saved to docs/figures/subsequent_events_risk_signal.png")

if __name__ == "__main__":
    file_path = 'data/cleaned_bank_calls.csv'
    results = analyze_subsequent_events(file_path)
    print(results)
    plot_results(results)
