# Hyperparameter Optimization

Best single model from Sprint 4 was Gradient Boosting, so GridSearchCV was applied to GradientBoostingClassifier.

## Search setup

- Search method: GridSearchCV
- Cross-validation: cv=3
- Scoring: ROC-AUC
- Tuned hyperparameters: n_estimators, learning_rate, max_depth

## Best parameters

- learning_rate: 0.1
- max_depth: 2
- n_estimators: 200

Best cross-validation ROC-AUC: 0.9983

## Test set metrics

| Metric | Value |
|---------|---------|
| Accuracy | 0.9667 |
| ROC-AUC | 0.9976 |
| F1 | 0.9471 |
| Precision | 0.9763 |
| Recall | 0.9196 |
| Inference latency (ms) | 6.36 |

## Confusion matrix

```text
[[461   5]
 [ 18 206]]
```
