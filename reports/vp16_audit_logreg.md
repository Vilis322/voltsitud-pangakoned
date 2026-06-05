# 📄 Dataset Audit Note — Synthetic Feature Determinism Risk

## 1. Summary

Model performance is very high (ROC-AUC ~0.98 with Logistic Regression) and permutation tests confirm **no classical leakage** (shuffled AUC ≈ 0.5).

However, feature-level analysis suggests the presence of **synthetic generation artifacts**, where target variable is implicitly encoded into a small subset of highly predictive features.

This does not indicate data leakage in the technical sense, but rather **over-deterministic synthetic data generation logic**.

---

## 2. Key Findings

### 2.1 Permutation Test (Leakage Check)

* Baseline AUC: ~0.987
* Shuffled labels AUC: ~0.495 ± 0.026

**Interpretation:**

* No evidence of label leakage
* Model is learning real structure present in dataset

---

### 2.2 Single Feature Predictive Power (Critical Signal)

Observed unusually high predictive performance from individual features:

* `time_to_next_action_min` → AUC ≈ 0.90
* `login_after_call` → AUC ≈ 0.79
* `twofa_confirmed_after_call` → AUC ≈ 0.76
* `transfer_after_call` → strong standalone signal (similar regime)

**Expected behavior in real-world fraud datasets:**

* Single feature AUC typically ~0.55–0.70

**Observation here:**

* Single features are nearly sufficient for classification

👉 This indicates **high structural determinism in feature generation**

---

### 2.3 Feature Grouping Pattern: Post-Event Behavioral Signals

A significant portion of predictive power comes from “post-call behavior” features:

* login_after_call
* transfer_after_call
* new_payee_after_call
* twofa_confirmed_after_call

These variables exhibit strong alignment with target label.

**Concern:**

* These features appear to be directly or indirectly conditioned on the fraud label in the synthetic generation process
* This introduces **label-to-feature dependency through generation rules**

---

### 2.4 Model Behavior

* Logistic Regression achieves ROC-AUC ≈ 0.98
* Indicates near-linear separability in feature space
* Suggests limited interaction complexity and strong rule-like structure

---

## 3. Root Cause Hypothesis

The dataset likely follows a generation schema closer to:

```
label → feature generation rules → observable variables
```

instead of a realistic causal structure:

```
latent risk factors → user behavior → partial observability → label
```

As a result, certain features encode near-explicit information about the target variable.

---

## 4. Affected Feature Groups

### 4.1 High-risk deterministic features

* time_to_next_action_min
  → appears strongly coupled to label definition or simulation rules

### 4.2 Post-event behavioral indicators

* login_after_call
* transfer_after_call
* twofa_confirmed_after_call
* new_payee_after_call

→ likely generated conditional on fraud label or near-direct proxy logic

### 4.3 Aggregated behavioral score (if derived from above)

* behavior_score
  → potential indirect compression of deterministic signals

---

## 5. Impact

### 5.1 Model performance inflation

* ROC-AUC ≈ 0.98 likely overestimates real-world performance

### 5.2 Reduced realism

* Dataset behaves closer to a rule system than a noisy observational process

### 5.3 Limited generalization validity

* High risk of performance collapse under real-world distribution shift

---

## 6. Recommendations

### 6.1 Decouple label from direct feature generation

Replace deterministic conditional logic:

❌ Current pattern:

```
if fraud:
    login_after_call = 1
```

✔ Recommended:

```
P(login_after_call) = sigmoid(
    w1 * caller_risk +
    w2 * temporal_context +
    noise
)
```

---

### 6.2 Introduce controlled overlap between classes

For key features (especially temporal ones):

* Ensure distributions overlap significantly between classes
* Target single-feature AUC range: **0.60–0.75**

---

### 6.3 Add stochasticity to temporal features

For `time_to_next_action_min`:

* introduce noise (log-normal / Gaussian)
* avoid near-deterministic separation

---

### 6.4 Introduce ambiguity in behavioral outcomes

* Add false positives: legitimate users exhibiting “fraud-like” behavior
* Add false negatives: fraud cases without strong behavioral signals

Suggested range:

* 10–20% controlled label noise in behavioral proxies

---

### 6.5 Break post-event determinism

Reduce strong coupling between:

* call event → immediate action flags

Introduce:

* delayed actions
* missing logs
* probabilistic execution chains

---

## 7. Validation Criteria After Fix

Expected post-fix metrics:

| Feature Type      | Target Single Feature AUC |
| ----------------- | ------------------------- |
| Temporal features | 0.60–0.75                 |
| Behavioral flags  | 0.55–0.70                 |
| Caller features   | 0.55–0.65                 |

Model-level ROC-AUC (LogReg):

* expected drop to ~0.75–0.90 depending on noise level

---

## 8. Conclusion

The dataset is internally consistent and free of classical leakage, but exhibits **strong synthetic determinism**, where a small number of features encode disproportionate information about the target variable.

This leads to inflated model performance and reduces realism relative to production fraud detection environments.

Recommended next step is to re-design feature generation to enforce:

* probabilistic dependency structure
* noise injection
* partial observability

---

Если хочешь, я могу дальше сделать тебе ещё более “боевой” вариант — типа:

* shorter Slack message version (5–7 строк)
* или “senior ML engineer review comment”
* или даже “ticket description for Jira”
