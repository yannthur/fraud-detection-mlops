"""Module de prétraitement des données pour la détection de fraude."""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(filepath: str) -> pd.DataFrame:
    """Charge les données depuis un fichier CSV."""
    return pd.read_csv(filepath)


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Prétraite les données pour l'entraînement."""
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")

    feature_cols = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]
    X = df[feature_cols].copy()
    y = df["is_fraud"].copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test, scaler
