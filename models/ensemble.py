from pathlib import Path

from sklearn.ensemble import VotingClassifier

from logreg import get_model as get_lr
from random_forest import get_model as get_rf
from gb_model import get_model as get_gb

from evaluation import evaluate, print_metrics
from gb_model import apply_temporary_imputation


ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data" / "train_features.csv"
TEST_PATH = ROOT / "data" / "test_features.csv"


def load_data(path: Path):
    import pandas as pd

    df = pd.read_csv(path)

    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"]

    return X, y


def main():
    print("\n=== LOADING DATA ===")

    X_train, y_train = load_data(TRAIN_PATH)
    X_test, y_test = load_data(TEST_PATH)

    X_train, X_test = apply_temporary_imputation(X_train, X_test)  

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # === BASE MODELS ===
    lr = get_lr()
    rf = get_rf()
    gb = get_gb()

    # === ENSEMBLE ===
    ensemble = VotingClassifier(
        estimators=[
            ("lr", lr),
            ("rf", rf),
            ("gb", gb),
        ],
        voting="soft",
        n_jobs=-1,
    )

    print("\n=== TRAINING ENSEMBLE ===")
    ensemble.fit(X_train, y_train)

    print("\n=== ENSEMBLE EVALUATION ===")
    ens_metrics = evaluate(ensemble, X_test, y_test)
    print_metrics(ens_metrics)

    # === BASELINE COMPARISON (GB) ===
    print("\n=== BASELINE (GRADIENT BOOSTING) ===")

    gb.fit(X_train, y_train)
    gb_metrics = evaluate(gb, X_test, y_test)
    print_metrics(gb_metrics)

    # === FINAL SUMMARY ===
    print("\n=== FINAL COMPARISON ===")
    print(f"GB ROC-AUC:        {gb_metrics['roc_auc']:.4f}")
    print(f"ENSEMBLE ROC-AUC:  {ens_metrics['roc_auc']:.4f}")

    print(f"GB F1:             {gb_metrics['f1']:.4f}")
    print(f"ENSEMBLE F1:       {ens_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()