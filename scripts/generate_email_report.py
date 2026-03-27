"""Génère et envoie un rapport email basé sur les résultats du pipeline."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def generate_report_content(pipeline_results: dict, hf_space_url: str) -> str:
    """Génère le contenu du rapport utilisant Gemini."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

        model = genai.GenerativeModel("gemini-2.0-flash")

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
    except Exception as e:
        print(f"Gemini API error: {e}")
        return f"""
        <p>Le pipeline MLOps de détection de fraude s'est exécuté avec succès.</p>
        <p>Résultats:</p>
        <ul>
            <li>Accuracy: {pipeline_results.get("accuracy", "N/A")}</li>
            <li>Precision fraude: {pipeline_results.get("precision_fraud", "N/A")}</li>
            <li>Rappel fraude: {pipeline_results.get("recall_fraud", "N/A")}</li>
            <li>F1 Score: {pipeline_results.get("f1_score", "N/A")}</li>
        </ul>
        """


def send_email(subject: str, body: str) -> bool:
    """Envoie un email via SMTP."""
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = os.getenv("SMTP_PORT")
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")
    mail_to = os.getenv("MAIL_TO")

    if not all([smtp_server, smtp_port, smtp_username, smtp_password, mail_to]):
        print("ERREUR: Variables SMTP manquantes dans l'environnement.")
        return False

    try:
        msg = MIMEMultipart()
        msg["From"] = smtp_username  # type: ignore
        msg["To"] = mail_to  # type: ignore
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html", "utf-8"))

        with smtplib.SMTP(smtp_server, int(smtp_port)) as server:  # type: ignore
            server.starttls()
            server.login(smtp_username, smtp_password)  # type: ignore
            server.send_message(msg)

        print(f"Email envoyé à {mail_to}")
        return True
    except Exception as e:
        print(f"ERREUR lors de l'envoi de l'email: {e}")
        return False


def generate_and_send_email_report(
    pipeline_results: dict, hf_space_url: str, commit_sha: str = "unknown"
) -> bool:
    """Génère et envoie le rapport email."""
    accuracy = pipeline_results.get("accuracy", "N/A")
    subject = f"[MLOps] Rapport Pipeline - Precision: {accuracy}"

    report_content = generate_report_content(pipeline_results, hf_space_url)

    html_body = f"""
    <html>
    <body>
        <h2>Rapport de Pipeline MLOps</h2>
        <p><strong>Commit:</strong> {commit_sha}</p>
        <hr/>
        {report_content}
        <hr/>
        <p><strong>URL Space:</strong> <a href="{hf_space_url}">{hf_space_url}</a></p>
    </body>
    </html>
    """

    return send_email(subject, html_body)


if __name__ == "__main__":
    results = {
        "accuracy": 0.95,
        "precision_fraud": 0.87,
        "recall_fraud": 0.82,
        "f1_score": 0.84,
    }

    hf_url = os.getenv(
        "HF_SPACE_URL", "https://huggingface.co/spaces/yannthur/fraud-detection"
    )
    commit = os.getenv("GITHUB_SHA", "local")[:7]

    generate_and_send_email_report(results, hf_url, commit)
