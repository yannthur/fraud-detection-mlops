"""Tests unitaires pour le module model."""
import numpy as np

from src.model import FraudDetector


class TestFraudDetector:
    """Tests pour la classe FraudDetector."""

    def test_init(self):
        """Test l'initialisation du modèle."""
        model = FraudDetector()
        assert model.model is not None
        assert model.scaler is None

    def test_fit_predict(self):
        """Test l'entraînement et la prédiction."""
        model = FraudDetector()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        predictions = model.predict(X[:10])

        assert len(predictions) == 10
        assert all(p in [0, 1] for p in predictions)

    def test_predict_proba(self):
        """Test les probabilités de prédiction."""
        model = FraudDetector()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        probas = model.predict_proba(X[:10])

        assert probas.shape == (10, 2)
        assert np.allclose(probas.sum(axis=1), 1.0)
