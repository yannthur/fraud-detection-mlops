"""Module de modèle ML pour la détection de fraude."""

from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class FraudDetector:
    """Classifieur pour la détection de fraude bancaire."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 10):
        """Initialize the FraudDetector with a RandomForest classifier."""
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        )
        self.feature_names = None

    def fit(self, X, y):
        """Entraîne le modèle."""
        self.model.fit(X, y)
        if hasattr(X, "columns"):
            self.feature_names = list(X.columns)
        return self

    def predict(self, X):
        """Prédit les classes."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Retourne les probabilités."""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Évalue le modèle."""
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        return report, cm

    def get_feature_importance(self):
        """Retourne l'importance des features."""
        if self.feature_names is None:
            return dict(
                zip(
                    range(len(self.model.feature_importances_)),
                    self.model.feature_importances_,
                )
            )
        return dict(zip(self.feature_names, self.model.feature_importances_))

    def save(self, path: str):
        """Sauvegarde le modèle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "feature_names": self.feature_names}, path)

    def load(self, path: str):
        """Charge le modèle."""
        data = joblib.load(path)
        self.model = data["model"]
        self.feature_names = data.get("feature_names")
        return self
