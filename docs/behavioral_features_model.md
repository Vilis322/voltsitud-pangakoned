# Behavioral Features Model

The best model was re-trained with additional aggregate behavioral features.

## Added features

- calls_from_number_last_24h
- distinct_victims_last_7d
- mean_call_duration_per_number
- complaint_keyword_score

## Test set metrics

| Metric | Value |
|---------|---------|
| Accuracy | 0.9913 |
| ROC-AUC | 0.9999 |
| F1 | 0.9865 |
| Precision | 0.9910 |
| Recall | 0.9821 |
| Inference latency (ms) | 7.75 |

## Confusion matrix

```text
[[464   2]
 [  4 220]]
```
