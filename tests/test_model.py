"""Tests unitaires pour le module model et data_preprocessing."""

import numpy as np

from src.model import FraudDetector


class TestFraudDetector:
    """Tests pour la classe FraudDetector."""

    def test_init(self):
        """Test l'initialisation du modèle."""
        model = FraudDetector()
        assert model.model is not None
        assert model.feature_names is None

    def test_init_with_params(self):
        """Test l'initialisation avec paramètres."""
        model = FraudDetector(n_estimators=50, max_depth=5)
        assert model.model.n_estimators == 50
        assert model.model.max_depth == 5

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

    def test_evaluate(self):
        """Test l'évaluation du modèle."""
        model = FraudDetector()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        report, cm = model.evaluate(X, y)

        assert "0" in report or "0.0" in report
        assert cm.shape == (2, 2)

    def test_feature_importance(self):
        """Test l'importance des features."""
        model = FraudDetector()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        importance = model.get_feature_importance()

        assert len(importance) == 5

    def test_save_load(self, tmp_path):
        """Test la sauvegarde et le chargement."""
        model = FraudDetector()
        X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)

        model.fit(X, y)
        model_path = str(tmp_path / "test_model.pkl")
        model.save(model_path)

        new_model = FraudDetector()
        new_model.load(model_path)

        assert new_model.model is not None
        predictions_new = new_model.predict(X[:5])
        predictions_old = model.predict(X[:5])
        assert np.array_equal(predictions_new, predictions_old)


class TestDataPreprocessing:
    """Tests pour le module data_preprocessing."""

    def test_load_and_prepare_shape(self):
        """Test que le chargement retourne les bonnes dimensions."""
        from src.data_preprocessing import load_and_prepare

        X_train, X_test, y_train, y_test = load_and_prepare("data/train.csv")
        assert X_train.shape[0] + X_test.shape[0] == 1000
        assert y_train.shape[0] == X_train.shape[0]
        assert y_test.shape[0] == X_test.shape[0]

    def test_load_and_prepare_stratified(self):
        """Test que le split est stratifié."""
        from src.data_preprocessing import load_and_prepare

        X_train, X_test, y_train, y_test = load_and_prepare("data/train.csv")
        train_fraud_rate = y_train.mean()
        test_fraud_rate = y_test.mean()
        assert abs(train_fraud_rate - test_fraud_rate) < 0.01

    def test_model_accuracy_threshold(self):
        """Test que le modèle atteint le seuil d'accuracy >= 0.80."""
        from sklearn.metrics import accuracy_score

        from src.data_preprocessing import load_and_prepare

        X_train, X_test, y_train, y_test = load_and_prepare("data/train.csv")
        model = FraudDetector(n_estimators=50, max_depth=10)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        assert accuracy >= 0.80, f"Accuracy {accuracy:.3f} < seuil 0.80"
