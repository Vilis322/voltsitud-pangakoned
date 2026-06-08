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

# Removed 'evaluate' from here to prevent namespace overriding
from evaluation import print_metrics

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


def evaluate_local(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    print("\n=== GRADIENT BOOSTING EVALUATION ===\n")

    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


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
    print("\nTraining GB Model...")
    model = get_model()  # Using the helper function properly now

    model.fit(X_train, y_train)

    # Call the explicitly renamed local evaluation function
    metrics = evaluate_local(model, X_test, y_test)

    # === AUDIT BLOCK ===
    print("\n=== AUDIT SUMMARY ===")
    print("Model: GradientBoostingClassifier")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")


if __name__ == "__main__":
    main()
