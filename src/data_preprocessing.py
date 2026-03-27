"""Module de prétraitement du dataset de détection de fraude bancaire."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

CATEGORICAL_COLS = ["category", "gender", "state", "job"]
NUMERIC_COLS = ["amt", "lat", "long", "city_pop", "merch_lat", "merch_long"]
TARGET_COL = "is_fraud"


def compute_age(dob_series: pd.Series) -> pd.Series:
    """Compute age in years from date of birth."""
    dob = pd.to_datetime(dob_series)
    now = pd.Timestamp("2019-01-01")
    return ((now - dob).dt.days / 365.25).round(1)


def compute_distance(df: pd.DataFrame) -> pd.Series:
    """Compute Euclidean distance between client and merchant."""
    return np.sqrt(
        (df["lat"] - df["merch_lat"]) ** 2 + (df["long"] - df["merch_long"]) ** 2
    )


def load_and_prepare(filepath: str, test_size: float = 0.2, random_state: int = 42):
    """
    Load, clean and prepare dataset for training.

    Args:
        filepath: Path to CSV file.
        test_size: Test set proportion.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple (X_train, X_test, y_train, y_test).
    """
    df = pd.read_csv(filepath)

    df["age"] = compute_age(df["dob"])
    df["distance"] = compute_distance(df)

    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = le.fit_transform(df[col].astype(str))

    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS + ["age", "distance"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df[TARGET_COL]

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_prepare("data/train.csv")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Taux de fraude (train): {y_train.mean():.4f}")
