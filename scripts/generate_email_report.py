"""Génère un rapport email basé sur les résultats du pipeline."""

import os


def generate_email_report(pipeline_results: dict, hf_space_url: str) -> str:
    """Génère un rapport email utilisant Gemini."""
    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""Tu es un assistant qui génère des rapports professionnels.

Voici les résultats du pipeline de détection de fraude:
- Exactitude (Accuracy): {pipeline_results.get("accuracy", "N/A")}
- Précision fraude: {pipeline_results.get("precision_fraud", "N/A")}
- Rappel fraude: {pipeline_results.get("recall_fraud", "N/A")}
- Score F1: {pipeline_results.get("f1_score", "N/A")}

L'application est déployée sur: {hf_space_url}

Génère un email professionnel résumant ces résultats."""

    response = model.generate_content(prompt)

    return response.text


if __name__ == "__main__":
    results = {
        "accuracy": 0.95,
        "precision_fraud": 0.87,
        "recall_fraud": 0.82,
        "f1_score": 0.84,
    }

    report = generate_email_report(
        results, "https://huggingface.co/spaces/yannthur/fraud-detection"
    )
    print(report)
