"""Tests unitaires et de validation du modèle de détection de fraude."""

import os
import json
import pytest
import numpy as np
import pandas as pd
import joblib

from src.data_preprocessing import load_and_prepare, compute_age, compute_distance
from src.model import build_model, evaluate_model, ACCURACY_THRESHOLD, MODEL_PATH


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def data():
    """Charge le dataset une seule fois pour tous les tests."""
    return load_and_prepare("data/train.csv")


@pytest.fixture(scope="session")
def trained_model(data):
    """Entraîne ou charge le modèle."""
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    X_train, _, y_train, _ = data
    model = build_model()
    model.fit(X_train, y_train)
    return model


# ── Tests de prétraitement ───────────────────────────────────────────────────


class TestDataPreprocessing:
    """Tests du pipeline de prétraitement des données."""

    def test_data_loads_correctly(self, data):
        """Le dataset se charge sans erreur et contient les bonnes dimensions."""
        X_train, X_test, y_train, y_test = data
        assert X_train.shape[0] > 0, "X_train est vide"
        assert X_test.shape[0] > 0, "X_test est vide"
        assert X_train.shape[1] == X_test.shape[1], "Nombre de features incohérent"

    def test_no_missing_values(self, data):
        """Les données préparées ne contiennent pas de valeurs manquantes."""
        X_train, X_test, y_train, y_test = data
        assert not X_train.isnull().any().any(), "Valeurs manquantes dans X_train"
        assert not X_test.isnull().any().any(), "Valeurs manquantes dans X_test"

    def test_target_is_binary(self, data):
        """La variable cible est bien binaire (0 ou 1)."""
        _, _, y_train, y_test = data
        assert set(y_train.unique()).issubset({0, 1}), "y_train contient des valeurs non binaires"
        assert set(y_test.unique()).issubset({0, 1}), "y_test contient des valeurs non binaires"

    def test_stratified_split_preserves_ratio(self, data):
        """Le split stratifié préserve approximativement le taux de fraude."""
        _, _, y_train, y_test = data
        ratio_train = y_train.mean()
        ratio_test = y_test.mean()
        assert abs(ratio_train - ratio_test) < 0.02, (
            f"Déséquilibre du split stratifié : train={ratio_train:.4f}, test={ratio_test:.4f}"
        )

    def test_train_test_split_ratio(self, data):
        """La proportion train/test est approximativement 80/20."""
        X_train, X_test, _, _ = data
        total = X_train.shape[0] + X_test.shape[0]
        test_ratio = X_test.shape[0] / total
        assert 0.18 <= test_ratio <= 0.22, f"Ratio test inattendu : {test_ratio:.3f}"


# ── Tests du modèle ──────────────────────────────────────────────────────────


class TestModel:
    """Tests de validation du modèle entraîné."""

    def test_model_file_exists(self):
        """Le fichier du modèle entraîné existe."""
        assert os.path.exists(MODEL_PATH), f"Modèle introuvable : {MODEL_PATH}"

    def test_model_loads_correctly(self, trained_model):
        """Le modèle se charge sans erreur."""
        assert trained_model is not None

    def test_model_has_predict_method(self, trained_model):
        """Le modèle expose bien les méthodes predict et predict_proba."""
        assert hasattr(trained_model, "predict")
        assert hasattr(trained_model, "predict_proba")

    def test_model_accuracy_above_threshold(self, trained_model, data):
        """
        TEST CRITIQUE : l'accuracy du modèle dépasse le seuil minimum requis.
        Ce test échoue le pipeline CI/CD si la performance est insuffisante.
        """
        _, X_test, _, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        accuracy = metrics["accuracy"]
        assert accuracy >= ACCURACY_THRESHOLD, (
            f"Accuracy {accuracy:.4f} en dessous du seuil requis {ACCURACY_THRESHOLD}. "
            f"Le modèle ne peut pas être déployé."
        )

    def test_model_f1_score_positive(self, trained_model, data):
        """Le F1-score sur la classe 1 (fraude) est supérieur à 0.50."""
        _, X_test, _, y_test = data
        metrics = evaluate_model(trained_model, X_test, y_test)
        assert metrics["f1_score"] > 0.50, (
            f"F1-score insuffisant : {metrics['f1_score']:.4f} < 0.50"
        )

    def test_prediction_output_shape(self, trained_model, data):
        """Les prédictions ont la bonne forme."""
        _, X_test, _, y_test = data
        preds = trained_model.predict(X_test)
        assert preds.shape[0] == X_test.shape[0]

    def test_prediction_values_are_binary(self, trained_model, data):
        """Les prédictions du modèle sont bien 0 ou 1."""
        _, X_test, _, _ = data
        preds = trained_model.predict(X_test[:100])
        assert set(preds).issubset({0, 1}), "Prédictions non binaires détectées"

    def test_proba_output_is_valid(self, trained_model, data):
        """Les probabilités prédites sont entre 0 et 1 et somment à 1."""
        _, X_test, _, _ = data
        probas = trained_model.predict_proba(X_test[:100])
        assert probas.shape[1] == 2
        assert np.allclose(probas.sum(axis=1), 1.0, atol=1e-6)
        assert (probas >= 0).all() and (probas <= 1).all()


# ── Tests de reproductibilité ────────────────────────────────────────────────


class TestReproducibility:
    """Tests de reproductibilité du pipeline."""

    def test_training_is_deterministic(self, data):
        """Deux entraînements avec la même graine produisent le même modèle."""
        X_train, X_test, y_train, y_test = data
        model1 = build_model()
        model2 = build_model()
        model1.fit(X_train, y_train)
        model2.fit(X_train, y_train)
        preds1 = model1.predict(X_test[:50])
        preds2 = model2.predict(X_test[:50])
        assert np.array_equal(preds1, preds2), "L'entraînement n'est pas déterministe"