"""Application Streamlit pour la détection de fraude."""

import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.resolve()))

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


@st.cache_resource
def load_model():
    """Charge le modele (cache)."""
    model = FraudDetector()
    if MODEL_PATH.exists():
        model.load(str(MODEL_PATH))
        st.success(f"Modele charge depuis {MODEL_PATH}")
    else:
        st.error(f"Modele non trouve a {MODEL_PATH}")
    return model


def main():
    """Lance l'application Streamlit."""
    st.set_page_config(
        page_title="Detection de Fraude Bancaire",
        page_icon="\U0001f52e",
        layout="centered",
    )

    st.title("\U0001f52e Detection de Fraude Bancaire")
    st.markdown(
        """
    Entrez les caracteristiques d'une transaction pour predire si elle est frauduleuse.
    """
    )

    col1, col2 = st.columns(2)

    with col1:
        amt = st.number_input("Montant (amt)", value=100.0, min_value=0.0)
        category = st.selectbox("Categorie", CATEGORIES)
        gender = st.radio("Genre", ["M", "F"])
        city_pop = st.number_input("Population ville", value=10000, min_value=0)
        age = st.number_input("Age", value=35, min_value=0, max_value=120)

    with col2:
        lat = st.number_input("Latitude client", value=40.0)
        long = st.number_input("Longitude client", value=-100.0)
        merch_lat = st.number_input("Latitude marchand", value=40.1)
        merch_long = st.number_input("Longitude marchand", value=-100.1)

    if st.button("Predire", type="primary"):
        model = load_model()

        if not MODEL_PATH.exists():
            st.error("Modele non disponible")
            return

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

        if prediction == 1:
            st.error("\U0001f6d1 FRAUDE DETECTEE")
            st.metric("Probabilite de fraude", f"{proba[1] * 100:.1f}%")
        else:
            st.success("\u2705 Transaction LEGITIME")
            st.metric("Probabilite de legitime", f"{proba[0] * 100:.1f}%")

        st.progress(proba[1] if prediction == 1 else proba[0])


if __name__ == "__main__":
    main()
