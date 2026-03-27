"""Script d'entraînement du modèle de détection de fraude."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_preprocessing import load_and_prepare
from src.model import FraudDetector


def main():
    """Point d'entrée pour l'entraînement."""
    data_path = Path(__file__).parent / "data" / "train.csv"

    print(f"Chargement des données depuis {data_path}...")
    X_train, X_test, y_train, y_test = load_and_prepare(str(data_path))

    print(f"Dimensions train: {X_train.shape}, test: {X_test.shape}")
    print(f"Taux de fraude (train): {y_train.mean():.4f}")

    print("Entraînement du modèle...")
    model = FraudDetector()
    model.fit(X_train, y_train)

    print("Évaluation du modèle...")
    report, cm = model.evaluate(X_test, y_test)

    print("\nClassification Report:")
    for label, metrics in report.items():
        if isinstance(metrics, dict):
            print(
                f"  {label}: precision={metrics.get('precision', 0):.3f}, "
                f"recall={metrics.get('recall', 0):.3f}, "
                f"f1-score={metrics.get('f1-score', 0):.3f}"
            )

    print("\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}, FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}, TP={cm[1][1]}")

    model_path = Path(__file__).parent / "models" / "fraud_model.pkl"
    print(f"\nSauvegarde du modèle vers {model_path}...")
    model.save(str(model_path))

    print("Entraînement terminé avec succès!")


if __name__ == "__main__":
    main()
