from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Imported from ensembling branch for factory support
from evaluation import evaluate as ext_evaluate
from evaluation import print_metrics

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH = ROOT / "data" / "test_features.csv"
FIGURE_PATH = ROOT / "docs" / "figures" / "random_forest_top15_feature_importances.png"


def load_data(path: Path):
    df = pd.read_csv(path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


def apply_temporary_imputation(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    train_medians = X_train.median(numeric_only=True)

    X_train = X_train.fillna(train_medians)
    X_test = X_test.fillna(train_medians)

    return X_train, X_test


def get_model():
    """Factory function for ensemble usage.

    Returns an UNTRAINED model.
    """
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
    }

    print("\n=== RANDOM FOREST EVALUATION ===\n")

    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return metrics


def plot_feature_importances(model, feature_names):
    importances = pd.Series(
        model.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(15)

    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind="barh")
    plt.title("Top-15 Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=150)
    plt.close()

    print(f"\nFeature importance plot saved to: {FIGURE_PATH.relative_to(ROOT)}")


def main():
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

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

    X_train, X_test = apply_temporary_imputation(X_train, X_test)

    model = get_model()

    model.fit(X_train, y_train)
    metrics = evaluate(model, X_test, y_test)
    plot_feature_importances(model, X_train.columns)

    print("\nRF metrics:")
    print_metrics(metrics)


if __name__ == "__main__":
    main()
