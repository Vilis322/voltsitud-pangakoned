from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evaluation import evaluate, print_metrics


def get_model():
    """
    Factory function for ensemble usage.
    Returns an UNTRAINED model.
    """
    return LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )


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

    model = get_model()
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print("LR metrics:")
    print_metrics(metrics)


if __name__ == "__main__":
    main()