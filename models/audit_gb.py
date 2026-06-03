from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import GradientBoostingClassifier


ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH = ROOT / "data" / "test_features.csv"


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
# PERMUTATION TEST (GB version)
# -----------------------------
def permutation_test(model, X, y, n_iter: int = 20):
    base_auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    shuffled_aucs = []

    for _ in range(n_iter):
        y_shuffled = np.random.permutation(y)

        auc = roc_auc_score(
            y_shuffled,
            model.predict_proba(X)[:, 1]
        )
        shuffled_aucs.append(auc)

    return {
        "base_auc": float(base_auc),
        "shuffled_mean_auc": float(np.mean(shuffled_aucs)),
        "shuffled_std_auc": float(np.std(shuffled_aucs)),
    }


# -----------------------------
# FEATURE IMPORTANCE CHECK
# -----------------------------
def feature_importance(model, feature_names):
    if not hasattr(model, "feature_importances_"):
        return {"error": "model has no feature_importances_"}

    importances = model.feature_importances_

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False)

    return df.head(10).to_dict(orient="records")


# -----------------------------
# MAIN AUDIT
# -----------------------------
def main():
    print("\n=== LOADING DATA ===")
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    print("\n=== TRAIN GB MODEL ===")

    model = GradientBoostingClassifier(
        random_state=42
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

    print("\n=== FEATURE IMPORTANCE ===")
    imp = feature_importance(model, X_train.columns)

    for r in imp:
        print(r)


if __name__ == "__main__":
    main()