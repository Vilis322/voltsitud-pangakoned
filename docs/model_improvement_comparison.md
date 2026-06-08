# Baseline vs Tuned Model Comparison

## Metrics Comparison

| Metric    | Baseline Model | Tuned Model | Delta   |
| --------- | -------------- | ----------- | ------- |
| Accuracy  | 0.9667         | 0.9913      | +0.0246 |
| ROC-AUC   | 0.9976         | 0.9999      | +0.0023 |
| F1-score  | 0.9471         | 0.9865      | +0.0394 |
| Precision | 0.9763         | 0.9910      | +0.0147 |
| Recall    | 0.9196         | 0.9821      | +0.0625 |

## Confusion Matrix Comparison

### Baseline

```text
[[461   5]
 [ 18 206]]
```

### Tuned Model

```text
[[464   2]
 [  4 220]]
```

### Differences

* False Positives reduced from 5 to 2.
* False Negatives reduced from 18 to 4.
* True Positives increased from 206 to 220.
* True Negatives increased from 461 to 464.

## Narrative

The addition of aggregate behavioral features produced a substantial improvement in model performance.

The largest gain was observed in recall (+6.25 percentage points), meaning the model successfully identified significantly more fraudulent calls while simultaneously reducing missed fraud cases.

The newly engineered features captured behavioral patterns that were not available in the original dataset, including recent calling activity, victim diversity, average caller behavior, and complaint keyword signals.

These features allowed the Gradient Boosting model to distinguish fraudulent campaigns more effectively, resulting in fewer false negatives and improved overall classification quality.

The tuned model outperformed the baseline across every evaluation metric and represents the strongest model developed during the project.
