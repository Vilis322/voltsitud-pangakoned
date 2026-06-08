from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from evaluation import evaluate, print_metrics

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH = ROOT / "data" / "test_features.csv"


def load_data(path: Path):
    df = pd.read_csv(path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


def apply_temporary_imputation(X_train, X_test):
    """Same philosophy as in LogReg module:

    simple median imputation for safety.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test


def get_model():
    """Return a configured GradientBoostingClassifier for ensemble use."""
    return GradientBoostingClassifier(random_state=42)


def main():
    print("\n=== LOADING DATA (FEATURES VERSION) ===")

    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # NaN diagnostics
    print("\n=== NA DIAGNOSTICS ===")

    nan_train = X_train.isna().sum()
    nan_train = nan_train[nan_train > 0]

    nan_test = X_test.isna().sum()
    nan_test = nan_test[nan_test > 0]

    if len(nan_train) == 0:
        print("Train: no NaN values")
    else:
        print("Train NaN columns:")
        print(nan_train)

    if len(nan_test) == 0:
        print("Test: no NaN values")
    else:
        print("Test NaN columns:")
        print(nan_test)

    # Imputation
    X_train, X_test = apply_temporary_imputation(X_train, X_test)

    # === MODEL ===
    print("GB metrics:")
    model = get_model()

    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    print_metrics(metrics)


if __name__ == "__main__":
    main()
