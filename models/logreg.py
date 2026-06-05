from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train_scaled.csv"
TEST_PATH = ROOT / "data" / "test_scaled.csv"


def load_data(path: Path):
    df = pd.read_csv(path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


def apply_temporary_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    TEMPORARY WORKAROUND.

    LogisticRegression cannot train with NaN values.

    Project documentation defines an imputation strategy,
    but it does not appear to be implemented yet in the
    shared preprocessing pipeline.

    To keep VP-16 self-contained, we impute using training-set
    medians only and reuse the same values for the test set.
    """

    X_train = X_train.copy()
    X_test = X_test.copy()

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n=== MODEL EVALUATION ===\n")

    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))


def main():
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    # === DIAGNOSTICS: NaN CHECK (before imputation) ===
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

    # Temporary VP-16 workaround
    X_train, X_test = apply_temporary_imputation(X_train, X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )

    model.fit(X_train, y_train)
    evaluate(model, X_test, y_test)


if __name__ == "__main__":
    main()