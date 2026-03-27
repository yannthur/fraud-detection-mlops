"""Met à jour le HuggingFace Space avec la dernière version."""

import os
import sys
import tempfile

from huggingface_hub import HfApi, login


def update_spaces(repo_name: str = "fraud-detection-streamlit"):
    """Met à jour le HuggingFace Space."""
    token = os.getenv("HF_TOKEN")

    if not token:
        print("ERREUR: HF_TOKEN non trouvé dans l'environnement.")
        sys.exit(1)

    login(token=token)

    api = HfApi()
    repo_id = f"yannthur/{repo_name}"

    readme_content = """---
title: Fraud Detection Streamlit
emoji: "\U0001f52e"
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# Fraud Detection App

Application de détection de fraude bancaire avec un modèle RandomForest et Streamlit.

## Utilisation

Entrez les caractéristiques d'une transaction pour prédire si elle est frauduleuse.
"""

    dockerfile_content = """FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \\\\
    build-essential \\\\
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app_streamlit.py ./app.py
COPY src/ ./src/
COPY models/ ./models/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "app.py",
    "--server.port=8501", "--server.address=0.0.0.0"]
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".dockerfile", delete=False) as f:
        f.write(dockerfile_content)
        dockerfile_path = f.name

    api.create_repo(
        repo_id=repo_id, repo_type="space", exist_ok=True, space_sdk="docker"
    )

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_file(
        path_or_fileobj=dockerfile_path,
        path_in_repo="Dockerfile",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_file(
        path_or_fileobj="app_streamlit.py",
        path_in_repo="app_streamlit.py",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_folder(
        folder_path="src",
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="src",
    )

    api.upload_file(
        path_or_fileobj="requirements.txt",
        path_in_repo="requirements.txt",
        repo_id=repo_id,
        repo_type="space",
    )

    api.upload_folder(
        folder_path="models",
        repo_id=repo_id,
        repo_type="space",
        path_in_repo="models",
    )

    os.unlink(readme_path)
    os.unlink(dockerfile_path)

    print(f"Space mis à jour sur https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    update_spaces()
