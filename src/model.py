"""Définition et utilitaires du modèle de détection de fraude."""

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    classification_report,
)
import joblib
import os

MODEL_PATH = "models/fraud_model.pkl"
ACCURACY_THRESHOLD = 0.80

def build_model():
    """Construit et retourne le modèle RandomForest."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        class_weight="balanced",  # Gère le déséquilibre des classes
        random_state=42,
        n_jobs=-1,
    )

def evaluate_model(model, X_test, y_test) -> dict:
    """Évalue le modèle sur le jeu de test."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    return metrics

def save_model(model, path: str = MODEL_PATH) -> None:
    """Sauvegarde le modèle entraîné."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)

def load_model(path: str = MODEL_PATH):
    """Charge le modèle sauvegardé."""
    return joblib.load(path)
