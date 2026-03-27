"""Script d'entraînement du modèle de détection de fraude."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_data, preprocess_data
from src.model import FraudDetector


def main():
    """Point d'entrée pour l'entraînement."""
    data_path = Path(__file__).parent / "data" / "train.csv"

    print(f"Chargement des données depuis {data_path}...")
    df = load_data(str(data_path))

    print("Prétraitement des données...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    print("Entraînement du modèle...")
    model = FraudDetector()
    model.scaler = scaler
    model.fit(X_train, y_train)

    print("Évaluation du modèle...")
    report, cm = model.evaluate(X_test, y_test)

    print(f"\nClassification Report:\n{report}")
    print(f"\nConfusion Matrix:\n{cm}")

    model_path = Path(__file__).parent / "models" / "fraud_model.pkl"
    print(f"\nSauvegarde du modèle vers {model_path}...")
    model.save(str(model_path))

    print("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
