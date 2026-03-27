"""Application Gradio pour la détection de fraude."""

import sys
from pathlib import Path

import gradio as gr
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from src.model import FraudDetector

MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.pkl"


def load_model():
    """Charge le modèle."""
    model = FraudDetector()
    if MODEL_PATH.exists():
        model.load(str(MODEL_PATH))
    return model


def predict(
    amount: float,
    city_pop: int,
    lat: float,
    long: float,
    merch_lat: float,
    merch_long: float,
) -> dict:
    """Prédit si une transaction est frauduleuse."""
    model = load_model()

    features = np.array([[amount, city_pop, lat, long, merch_lat, merch_long]])

    if model.scaler is not None:
        features = model.scaler.transform(features)

    proba = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    return {
        "Frauduleuse": bool(prediction),
        "Probabilité fraud": f"{proba[1] * 100:.1f}%",
        "Probabilité légitime": f"{proba[0] * 100:.1f}%",
    }


def main():
    """Lance l'application Gradio."""
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Number(label="Montant (amt)"),
            gr.Number(label="Population ville (city_pop)"),
            gr.Number(label="Latitude"),
            gr.Number(label="Longitude"),
            gr.Number(label="Latitude marchand (merch_lat)"),
            gr.Number(label="Longitude marchand (merch_long)"),
        ],
        outputs="json",
        title="Détection de Fraude Bancaire",
        description="Entrez les caractéristiques d'une transaction.",
    )

    iface.launch()


if __name__ == "__main__":
    main()
