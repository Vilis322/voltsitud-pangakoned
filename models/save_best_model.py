from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent

TRAIN_SCALED_PATH = ROOT / "data" / "train_scaled.csv"
TEST_SCALED_PATH = ROOT / "data" / "test_scaled.csv"
TRAIN_FEATURES_PATH = ROOT / "data" / "train_features.csv"
TEST_FEATURES_PATH = ROOT / "data" / "test_features.csv"

BEST_MODEL_PATH = ROOT / "models" / "best_model.joblib"
FEATURE_COLUMNS_PATH = ROOT / "models" / "feature_columns.joblib"


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]
    return X, y


def impute_train_test(X_train, X_test):
    medians = X_train.median(numeric_only=True)
    return X_train.fillna(medians), X_test.fillna(medians)


def train_and_score(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, y_proba)

    return {
        "name": name,
        "model": model,
        "score": score,
        "feature_columns": X_train.columns.tolist(),
    }


def main():
    X_train_scaled, y_train_scaled = load_data(TRAIN_SCALED_PATH)
    X_test_scaled, y_test_scaled = load_data(TEST_SCALED_PATH)

    X_train_features, y_train_features = load_data(TRAIN_FEATURES_PATH)
    X_test_features, y_test_features = load_data(TEST_FEATURES_PATH)

    X_train_scaled, X_test_scaled = impute_train_test(X_train_scaled, X_test_scaled)
    X_train_features, X_test_features = impute_train_test(X_train_features, X_test_features)

    results = [
        train_and_score(
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
        train_and_score(
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
        train_and_score(
            "Gradient Boosting",
            GradientBoostingClassifier(random_state=42),
            X_train_features,
            y_train_features,
            X_test_features,
            y_test_features,
        ),
    ]

    best = max(results, key=lambda item: item["score"])

    joblib.dump(best["model"], BEST_MODEL_PATH)
    joblib.dump(best["feature_columns"], FEATURE_COLUMNS_PATH)

    print(f"Best model: {best['name']}")
    print(f"ROC-AUC: {best['score']:.4f}")
    print(f"Saved model to: {BEST_MODEL_PATH.relative_to(ROOT)}")
    print(f"Saved feature columns to: {FEATURE_COLUMNS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()