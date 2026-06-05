# VP-18 — Gradient Boosting Model Report

## 1. Context

In this experiment (VP-18), we evaluate a Gradient Boosting model for bank call fraud detection and compare it against a Logistic Regression baseline (VP-16).

The goal is to assess:
- Predictive performance improvement
- Trade-offs in precision/recall
- Feature behavior and potential risk signals

---

## 2. Models Evaluated

### Logistic Regression (baseline)
- Linear model
- Scaled features (`train_scaled.csv`)
- Strong but limited in capturing nonlinear interactions

### Gradient Boosting Classifier (candidate)
- Non-linear ensemble model
- Trained on raw engineered features (`train_features.csv`)
- Able to capture feature interactions and thresholds

---

## 3. Results Summary

| Metric      | Logistic Regression | Gradient Boosting |
|------------|---------------------|-------------------|
| ROC-AUC    | 0.9876              | 0.9977            |
| Accuracy   | 0.9333              | 0.9710            |
| Precision  | 0.8708              | 0.9766            |
| Recall     | 0.9330              | 0.9330            |
| F1-score   | 0.9009              | 0.9543            |

### Confusion Matrix Comparison

**Logistic Regression:**
- FP: 31
- FN: 15

**Gradient Boosting:**
- FP: 5
- FN: 15

---

## 4. Key Findings

### 4.1 Performance Improvement
Gradient Boosting significantly improves:
- Precision (0.87 → 0.98)
- False positives reduction (~6x decrease)
- Overall ROC-AUC

Recall remains unchanged, indicating stable fraud detection sensitivity.

---

### 4.2 Feature Importance (GB Model)

The model is heavily dominated by a single feature:

- `time_to_next_action_min`: ~0.76 importance

Secondary features:
- `twofa_confirmed_after_call`
- `login_after_call`
- `transfer_after_call`
- `hour`
- `caller_number_prefix_freq`

---

## 5. Interpretation

The model behavior suggests:
- Strong dependency on behavioral timing signals
- Fraud detection is highly influenced by post-call reaction time
- Remaining features contribute marginally compared to the dominant signal

This indicates the dataset contains a very strong predictive feature that captures most of the separability between classes.

---

## 6. Risks and Observations

### 6.1 Feature Dominance
A single feature contributes more than 75% of model importance.  
This creates potential risks:
- Reduced robustness if feature distribution shifts
- Over-reliance on a single behavioral proxy

---

### 6.2 Dataset Structure
High ROC-AUC values (~0.99) suggest:
- Strong signal-to-noise ratio
- Possibly structured or semi-synthetic behavioral patterns
- Low ambiguity in class separation

No direct evidence of data leakage was observed from current audits.

---

## 7. Recommendations

### Short-term
- Perform feature ablation test (remove `time_to_next_action_min`)
- Validate model stability under feature removal
- Compare degradation vs current performance

### Mid-term
- Introduce SHAP-based interpretability analysis
- Investigate feature distribution stability across time

### Long-term
- Assess whether dataset reflects real-world fraud variability
- Consider adding noise or additional behavioral complexity for robustness testing

---

## 8. Conclusion

Gradient Boosting significantly outperforms Logistic Regression in predictive performance while maintaining stable recall.

However, the model relies heavily on a single dominant feature, which should be further investigated for robustness and generalization.

From a deployment perspective, the model is strong, but requires additional validation before production usage due to feature concentration risk.