# Imputation Strategy for Bank Calls Dataset

Based on the initial analysis of the synthetic dataset, several columns contain missing values (NA). The following strategies will be applied to prepare the data for downstream tasks:

## Missing Value Analysis
The following columns have identified missing values:
- `duration_sec` (numerical)
- `channel` (categorical)
- `complaint_text` (categorical/text)
- `manual_severity` (numerical)
- `time_to_next_action_min` (numerical)

## Imputation Strategy

| Column | Type | Strategy | Reason |
| :--- | :--- | :--- | :--- |
| `duration_sec` | Numerical | Median | Robust to outliers in call duration. |
| `manual_severity` | Numerical | Median | Standard approach for ordinal/numeric severity ratings. |
| `time_to_next_action_min` | Numerical | Median | Represents a time delta; median provides a representative typical value. |
| `channel` | Categorical | 'unknown' | Captures missing metadata as a distinct category. |
| `complaint_text` | Categorical/Text | 'unknown' | No text reported defaults to a neutral 'unknown' state. |
| `*_after_call` flags | Behavioral | False | Missing behavior implies no action was recorded. |

## Implementation Note
The imputation should be performed in a dedicated data cleaning pipeline before model training. Behavioral flags (e.g., `login_after_call`, `twofa_confirmed_after_call`) should be explicitly filled with `False` to ensure type consistency (Boolean).
