from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH = ROOT / "data" / "test_features.csv"
OUTPUT_PATH = ROOT / "docs" / "hyperparameter_optimization.md"


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y


def impute_train_test(X_train, X_test):
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_test.fillna(medians)


def evaluate_model(model, X_test, y_test):
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "f1": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "latency_ms": latency_ms,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }


def main():
    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    X_train, X_test = impute_train_test(X_train, X_test)

    base_model = GradientBoostingClassifier(random_state=42)

    param_grid = {
        "n_estimators": [100, 150, 200],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [2, 3, 4],
    }

    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=3,
        n_jobs=-1,
    )

    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    metrics = evaluate_model(best_model, X_test, y_test)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    markdown = "# Hyperparameter Optimization\n\n"
    markdown += "Best single model from Sprint 4 was Gradient Boosting, so GridSearchCV was applied to GradientBoostingClassifier.\n\n"
    markdown += "## Search setup\n\n"
    markdown += "- Search method: GridSearchCV\n"
    markdown += "- Cross-validation: cv=3\n"
    markdown += "- Scoring: ROC-AUC\n"
    markdown += "- Tuned hyperparameters: n_estimators, learning_rate, max_depth\n\n"
    markdown += "## Best parameters\n\n"
    for key, value in search.best_params_.items():
        markdown += f"- {key}: {value}\n"

    markdown += f"\nBest cross-validation ROC-AUC: {search.best_score_:.4f}\n\n"

    markdown += "## Test set metrics\n\n"
    markdown += "| Metric | Value |\n"
    markdown += "|---------|---------|\n"
    markdown += f"| Accuracy | {metrics['accuracy']:.4f} |\n"
    markdown += f"| ROC-AUC | {metrics['roc_auc']:.4f} |\n"
    markdown += f"| F1 | {metrics['f1']:.4f} |\n"
    markdown += f"| Precision | {metrics['precision']:.4f} |\n"
    markdown += f"| Recall | {metrics['recall']:.4f} |\n"
    markdown += f"| Inference latency (ms) | {metrics['latency_ms']:.2f} |\n"

    markdown += "\n## Confusion matrix\n\n"
    markdown += "```text\n"
    markdown += str(metrics["confusion_matrix"])
    markdown += "\n```\n"

    OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    print("Best parameters:")
    print(search.best_params_)
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("\nTest metrics:")
    for key, value in metrics.items():
        if key == "confusion_matrix":
            print(f"{key}:\n{value}")
        else:
            print(f"{key}: {value:.4f}")
    print(f"\nSaved report to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()