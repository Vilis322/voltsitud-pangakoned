from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

TRAIN_SCALED_PATH = ROOT / "data" / "train_scaled.csv"
TEST_SCALED_PATH = ROOT / "data" / "test_scaled.csv"
TRAIN_FEATURES_PATH = ROOT / "data" / "train_features.csv"
TEST_FEATURES_PATH = ROOT / "data" / "test_features.csv"
OUTPUT_PATH = ROOT / "docs" / "models_comparison.md"


def load_data(path: Path):
    df = pd.read_csv(path)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y


def impute_train_test(X_train, X_test):
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_test.fillna(medians)


def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)

    start = time.perf_counter()
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - start) * 1000

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
        "F1": f1_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "Inference latency (ms)": latency_ms,
    }


def main():
    X_train_scaled, y_train_scaled = load_data(TRAIN_SCALED_PATH)
    X_test_scaled, y_test_scaled = load_data(TEST_SCALED_PATH)

    X_train_features, y_train_features = load_data(TRAIN_FEATURES_PATH)
    X_test_features, y_test_features = load_data(TEST_FEATURES_PATH)

    X_train_scaled, X_test_scaled = impute_train_test(X_train_scaled, X_test_scaled)
    X_train_features, X_test_features = impute_train_test(X_train_features, X_test_features)

    results = [
        evaluate_model(
            "Logistic Regression",
            LogisticRegression(
                max_iter=1000,
                solver="lbfgs",
                class_weight="balanced",
                random_state=42,
            ),
            X_train_scaled,
            y_train_scaled,
            X_test_scaled,
            y_test_scaled,
        ),
        evaluate_model(
            "Random Forest",
            RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            X_train_features,
            y_train_features,
            X_test_features,
            y_test_features,
        ),
        evaluate_model(
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
            X_train_features,
            y_train_features,
            X_test_features,
            y_test_features,
        ),
    ]

    df = pd.DataFrame(results)

    for col in ["Accuracy", "ROC-AUC", "F1", "Precision", "Recall"]:
        df[col] = df[col].map(lambda x: f"{x:.4f}")

    df["Inference latency (ms)"] = df["Inference latency (ms)"].map(lambda x: f"{x:.2f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    markdown = "# Models Comparison\n\n"
    markdown += "| Model | Accuracy | ROC-AUC | F1 | Precision | Recall | Inference latency (ms) |\n"
    markdown += "|---------|---------|---------|---------|---------|---------|---------|\n"

    for _, row in df.iterrows():
        markdown += (
            f"| {row['Model']} "
            f"| {row['Accuracy']} "
            f"| {row['ROC-AUC']} "
            f"| {row['F1']} "
            f"| {row['Precision']} "
            f"| {row['Recall']} "
            f"| {row['Inference latency (ms)']} |\n"
        )
    markdown += "\n\n"
    markdown += (
        "Gradient Boosting achieved the strongest overall performance, "
        "with the highest accuracy, ROC-AUC, F1-score and precision. "
        "Random Forest performed very similarly and clearly improved over "
        "the Logistic Regression baseline. Logistic Regression remains useful "
        "as a simple baseline model, but tree-based models fit this dataset better.\n"
    )

    OUTPUT_PATH.write_text(markdown, encoding="utf-8")

    print(df.to_string(index=False))
    print(f"\nSaved comparison table to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()