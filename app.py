"""Application Gradio pour la detection de fraude."""

from pathlib import Path

import gradio as gr
import numpy as np

from src.model import FraudDetector

MODEL_PATH = Path(__file__).parent / "models" / "fraud_model.pkl"

CATEGORIES = [
    "entertainment",
    "food_dining",
    "gas_transport",
    "grocery_net",
    "grocery_pos",
    "health_fitness",
    "home",
    "kids_pets",
    "misc_net",
    "misc_pos",
    "personal_care",
    "shopping_net",
    "shopping_pos",
    "travel",
]

_model = None


def get_model():
    """Charge le model (cache)."""
    global _model
    if _model is None:
        _model = FraudDetector()
        if MODEL_PATH.exists():
            _model.load(str(MODEL_PATH))
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print(f"WARNING: Model not found at {MODEL_PATH}")
    return _model


def predict(
    amt: float,
    category: str,
    gender: str,
    city_pop: int,
    lat: float,
    long: float,
    merch_lat: float,
    merch_long: float,
    age: float,
) -> dict:
    """Predit si une transaction est frauduleuse."""
    model = get_model()

    if not MODEL_PATH.exists():
        return {"Erreur": "Modele non entraine. Lancez python train.py d'abord."}

    category_encoded = CATEGORIES.index(category) if category in CATEGORIES else 0
    gender_encoded = 1 if gender == "M" else 0

    features = np.array(
        [
            [
                float(amt),
                float(lat),
                float(long),
                float(city_pop),
                float(merch_lat),
                float(merch_long),
                float(category_encoded),
                float(gender_encoded),
                float(age),
                float(np.sqrt((lat - merch_lat) ** 2 + (long - merch_long) ** 2)),
            ]
        ]
    )

    proba = model.predict_proba(features)[0]
    prediction = model.predict(features)[0]

    return {
        "Prediction": "FRAUDE" if prediction == 1 else "Legitime",
        "Probabilite fraude": f"{proba[1] * 100:.1f}%",
        "Probabilite legitime": f"{proba[0] * 100:.1f}%",
    }


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Montant (amt)", value=100.0),
        gr.Dropdown(choices=CATEGORIES, label="Categorie", value=CATEGORIES[0]),
        gr.Radio(choices=["M", "F"], label="Genre", value="M"),
        gr.Number(label="Population ville", value=10000),
        gr.Number(label="Latitude client", value=40.0),
        gr.Number(label="Longitude client", value=-100.0),
        gr.Number(label="Latitude marchand", value=40.1),
        gr.Number(label="Longitude marchand", value=-100.1),
        gr.Number(label="Age", value=35),
    ],
    outputs="json",
    title="Detection de Fraude Bancaire",
    description=(
        "Entrez les caracteristiques d'une transaction "
        "pour predire si elle est frauduleuse."
    ),
    examples=[
        [150.0, "gas_transport", "M", 50000, 40.7, -74.0, 40.8, -74.1, 30],
        [2000.0, "shopping_net", "F", 100000, 34.0, -118.0, 34.1, -118.1, 45],
        [50.0, "grocery_pos", "M", 25000, 41.0, -87.0, 41.1, -87.1, 25],
    ],
)

if __name__ == "__main__":
    demo.launch()
