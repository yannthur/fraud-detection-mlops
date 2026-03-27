"""Script d'entraînement du modèle de détection de fraude bancaire."""

import os
import json
import sys

# Attention: ce module doit exister pour que le code fonctionne !
from src.data_preprocessing import load_and_prepare
from src.model import build_model, evaluate_model, save_model, ACCURACY_THRESHOLD, MODEL_PATH

def main():
    """Pipeline complet d'entraînement et de validation."""
    print("=" * 60)
    print("  Fraud Detection — Pipeline d'entraînement")
    print("=" * 60)

    # 1. Chargement et prétraitement des données
    print("\n[1/4] Chargement des données...")
    X_train, X_test, y_train, y_test = load_and_prepare("data/train.csv")
    print(f"      Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"      Taux de fraude (train): {y_train.mean():.4f}")

    # 2. Construction et entraînement du modèle
    print("\n[2/4] Entraînement du modèle...")
    model = build_model()
    model.fit(X_train, y_train)
    print("      Modèle entraîné avec succès.")

    # 3. Évaluation
    print("\n[3/4] Évaluation des performances...")
    metrics = evaluate_model(model, X_test, y_test)
    for metric, value in metrics.items():
        print(f"      {metric:<12} : {value:.4f}")

    # 4. Validation du seuil et sauvegarde
    print(f"\n[4/4] Validation du seuil (accuracy >= {ACCURACY_THRESHOLD})...")
    if metrics["accuracy"] < ACCURACY_THRESHOLD:
        print(f"      ❌ ÉCHEC : accuracy {metrics['accuracy']:.4f} < {ACCURACY_THRESHOLD}")
        sys.exit(1)

    save_model(model)
    print(f"      ✅ Modèle sauvegardé : {MODEL_PATH}")

    # Sauvegarder les métriques pour le CI/CD
    os.makedirs("models", exist_ok=True)
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("      Métriques sauvegardées : models/metrics.json")

    print("\n✅ Pipeline d'entraînement terminé avec succès.")
    return metrics

if __name__ == "__main__":
    main()

