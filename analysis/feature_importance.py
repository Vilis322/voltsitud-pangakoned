import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


# =========================================================
# Permutation Importance
# =========================================================
def compute_permutation_importance(
    model,
    X_val,
    y_val,
    scoring="roc_auc",
    n_repeats=10,
    random_state=42
):
    """
    Computes permutation importance for a trained model.
    Returns a sorted DataFrame.
    """

    result = permutation_importance(
        model,
        X_val,
        y_val,
        scoring=scoring,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    df = pd.DataFrame({
        "feature": X_val.columns,
        "importance": result.importances_mean,
        "std": result.importances_std
    })

    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# =========================================================
# Random Forest Importance
# =========================================================
def extract_rf_importance(model, feature_names):
    """
    Extracts feature importance from Random Forest.
    Works with both pipeline and standalone RF.
    """

    if hasattr(model, "named_steps") and "model" in model.named_steps:
        rf = model.named_steps["model"]
    else:
        rf = model

    df = pd.DataFrame({
        "feature": feature_names,
        "importance": rf.feature_importances_
    })

    return df.sort_values("importance", ascending=False).reset_index(drop=True)


# =========================================================
# Comparison
# =========================================================
def compare_importances(perm_df, rf_df):
    """
    Merges permutation and RF importance.
    """

    merged = perm_df.merge(
        rf_df,
        on="feature",
        suffixes=("_perm", "_rf")
    )

    return merged.sort_values("importance_perm", ascending=False).reset_index(drop=True)


# =========================================================
# Utility
# =========================================================
def get_top_k(df, k=20, column="importance"):
    """
    Returns top-k features from a dataframe.
    """
    return df.sort_values(column, ascending=False).head(k)


# =========================================================
# CLI / manual execution
# =========================================================
if __name__ == "__main__":
    """
    Manual run mode.
    Replace load_model/load_data with your project functions.
    """

    ROOT = Path(__file__).resolve().parent.parent
    TRAIN_PATH = ROOT / "data" / "train_features.csv"
    TARGET_COLUMN = "is_fraud"

    def load_data(test_size=0.2, random_state=42):
        df = pd.read_csv(TRAIN_PATH)

        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=random_state,
        )

        train_medians = X_train.median(numeric_only=True)
        X_train = X_train.fillna(train_medians)
        X_val = X_val.fillna(train_medians)

        return X_train, X_val, y_train, y_val

    def load_model(X_train, y_train):
        model = RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        return model

    X_train, X_val, y_train, y_val = load_data()
    model = load_model(X_train, y_train)

    print("\n[INFO] Computing permutation importance...")
    perm = compute_permutation_importance(model, X_val, y_val)

    print("\n[INFO] Computing RF importance...")
    rf = extract_rf_importance(model, X_val.columns)

    print("\n[INFO] Comparing...")
    comp = compare_importances(perm, rf)

    print("\nTOP PERMUTATION FEATURES:")
    print(get_top_k(perm, 20))

    print("\nTOP RF FEATURES:")
    print(get_top_k(rf, 20))

    print("\nTOP COMPARISON:")
    print(comp.head(20))