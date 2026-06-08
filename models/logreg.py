from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate, print_metrics


def get_model(use_class_weight=False):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=3000,
            solver="lbfgs",
            class_weight="balanced" if use_class_weight else None,
            random_state=42,
        ))
    ])


# -----------------------------
# Optional standalone execution
# -----------------------------
def main():
    from pathlib import Path
    import pandas as pd
    from sklearn.metrics import (
        accuracy_score,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
    )

    ROOT = Path(__file__).resolve().parent.parent
    TRAIN_PATH = ROOT / "data" / "train_features.csv"
    TEST_PATH = ROOT / "data" / "test_features.csv"

    def load_data(path):
        df = pd.read_csv(path)
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"]
        return X, y

    def apply_imputation(X_train, X_test):
        X_train = X_train.copy()
        X_test = X_test.copy()
        med = X_train.median(numeric_only=True)
        return X_train.fillna(med), X_test.fillna(med)

    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    X_train, X_test = apply_imputation(X_train, X_test)

    # baseline
    model_base = get_model(use_class_weight=False)
    model_base.fit(X_train, y_train)
    metrics_base = evaluate(model_base, X_test, y_test)

    print("BASELINE (no imbalance handling):")
    print_metrics(metrics_base)


    # balanced
    model_bal = get_model(use_class_weight=True)
    model_bal.fit(X_train, y_train)
    metrics_bal = evaluate(model_bal, X_test, y_test)

    print("\nBALANCED (class_weight='balanced'):")
    print_metrics(metrics_bal)


if __name__ == "__main__":
    main()