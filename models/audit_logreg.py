from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train_scaled.csv"
TEST_PATH = ROOT / "data" / "test_scaled.csv"


# -----------------------------
# DATA LOADING
# -----------------------------
def load_data(path: Path):
    df = pd.read_csv(path)

    if "is_fraud" not in df.columns:
        raise ValueError(f"is_fraud missing in {path}")

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


# -----------------------------
# LEAKAGE TEST 1
# -----------------------------
def permutation_test(model, X, y, n_iter: int = 20):
    """
    If model is real → AUC should collapse when labels are shuffled.
    If leakage exists → AUC stays suspiciously high.
    """
    base_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    shuffled_aucs = []

    for _ in range(n_iter):
        y_shuffled = np.random.permutation(y)
        auc = roc_auc_score(y_shuffled, model.predict_proba(X)[:, 1])
        shuffled_aucs.append(auc)

    return {
        "base_auc": base_auc,
        "shuffled_mean_auc": float(np.mean(shuffled_aucs)),
        "shuffled_std_auc": float(np.std(shuffled_aucs)),
    }


# -----------------------------
# LEAKAGE TEST 2
# -----------------------------
def feature_sanity_check(model, feature_names):
    """
    Inspect extreme coefficients in Logistic Regression.
    Large absolute weights often indicate proxy leakage features.
    """
    if not hasattr(model, "coef_"):
        return {"error": "model has no coef_"}

    coefs = model.coef_[0]
    df = pd.DataFrame({
        "feature": feature_names,
        "coef": coefs,
        "abs_coef": np.abs(coefs),
    }).sort_values("abs_coef", ascending=False)

    return {
        "top_positive": df.head(10)[["feature", "coef"]].to_dict(orient="records"),
        "top_negative": df.tail(10)[["feature", "coef"]].to_dict(orient="records"),
    }


# -----------------------------
# LEAKAGE TEST 3
# -----------------------------
def train_on_single_feature(X_train, y_train, X_test, y_test):
    """
    Tests if one feature alone can already achieve very high AUC.
    That often signals leakage or overly deterministic generation.
    """
    results = {}

    for col in X_train.columns[:10]:  # limit for speed
        model = LogisticRegression(max_iter=500)
        model.fit(X_train[[col]].fillna(0), y_train)

        auc = roc_auc_score(
            y_test,
            model.predict_proba(X_test[[col]].fillna(0))[:, 1]
        )

        results[col] = auc

    return sorted(results.items(), key=lambda x: x[1], reverse=True)


# -----------------------------
# MAIN AUDIT
# -----------------------------
def main():
    print("\n=== LOADING DATA ===")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    print("\n=== TRAIN BASELINE MODEL ===")
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train.fillna(0), y_train)

    base_auc = roc_auc_score(
        y_test,
        model.predict_proba(X_test.fillna(0))[:, 1]
    )

    print(f"Baseline ROC-AUC: {base_auc:.4f}")

    print("\n=== PERMUTATION TEST ===")
    perm = permutation_test(model, X_test.fillna(0), y_test)

    print(f"Base AUC: {perm['base_auc']:.4f}")
    print(f"Shuffled mean AUC: {perm['shuffled_mean_auc']:.4f}")
    print(f"Shuffled std AUC: {perm['shuffled_std_auc']:.4f}")

    print("\n=== FEATURE SANITY CHECK ===")
    feat = feature_sanity_check(model, X_train.columns)

    print("\nTop positive coefficients:")
    for r in feat["top_positive"]:
        print(r)

    print("\nTop negative coefficients:")
    for r in feat["top_negative"]:
        print(r)

    print("\n=== SINGLE FEATURE SIGNAL TEST ===")
    single = train_on_single_feature(X_train.fillna(0), y_train, X_test.fillna(0), y_test)

    print("\nTop single-feature AUC signals:")
    for name, score in single[:10]:
        print(f"{name:40s} {score:.4f}")


if __name__ == "__main__":
    main()