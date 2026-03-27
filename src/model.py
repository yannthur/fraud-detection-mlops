"""Module de modèle ML pour la détection de fraude."""

import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix


class FraudDetector:
    """Classifieur pour la détection de fraude bancaire."""

    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.scaler = None

    def fit(self, X, y):
        """Entraîne le modèle."""
        self.model.fit(X, y)
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
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm

    def save(self, path: str):
        """Sauvegarde le modèle."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "scaler": self.scaler}, path)

    def load(self, path: str):
        """Charge le modèle."""
        data = joblib.load(path)
        self.model = data["model"]
        self.scaler = data.get("scaler")
        return self
