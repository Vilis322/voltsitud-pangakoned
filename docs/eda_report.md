# Exploratory Data Analysis Report

This report summarizes the findings from the analysis of bank call data (`cleaned_bank_calls.csv`), focusing on identifying patterns associated with fraud.

## 1. Fraud Call Frequency
Fraud calls exhibit temporal clusters. The heatmap shows specific time-of-day and day-of-week patterns, indicating when scammers are most active.
![Fraud Call Volume by Time](figures/fraud_call_frequency.png)

## 2. Complaint Keyword Prevalence
Analysis of free-text complaints reveals that certain keywords, such as "mobiil-id" and "smart-id", are strong indicators of potential fraud attempts.
![Complaint Keyword Frequencies](figures/complaint_keyword_frequencies.png)

## 3. Subsequent Events Risk
Tracking actions taken by customers within one hour of a call (login, transfer, new payee) provides a strong behavioral signal of whether the call resulted in a successful fraud attempt.
![Subsequent Events Risk](figures/subsequent_events_risk_signal.png)

## 4. Suspicious Callers
High-volume callers with high fraud rates have been identified. We distinguish between "mass campaigns" (high volume, high fraud) and "targeted attempts" (low volume, high fraud).
![Top Suspicious Callers](figures/top_suspicious_callers.png)

## Hypotheses for Modeling
Based on the EDA, we propose the following hypotheses to test in the modeling phase:

1.  **Temporal Predictivity:** Calls occurring during specific "off-peak" or "high-fraud" clusters (identified in heatmap) significantly increase the baseline probability of fraud.
2.  **Authentication Signal:** The mention of "Mobiil-ID" or "Smart-ID" in complaints is a stronger predictor of fraud than generic banking keywords.
3.  **Behavioral Trigger:** A login or new payee setup within 60 minutes of a call is a causal signal of compromised account security.
4.  **Caller Reputation:** Caller reputation scores (based on fraud rate in training data) can be used as a primary feature to block or flag incoming calls in real-time.
5.  **Campaign Detection:** Combining caller-number volume with distinct-victim counts can differentiate between automated mass-fraud campaigns and targeted social engineering attacks, enabling tailored defensive responses.
