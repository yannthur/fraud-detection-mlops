"""Met à jour le HuggingFace Space avec la dernière version."""

import os
import sys

from huggingface_hub import HfApi, login


def update_spaces(repo_name: str = "fraud-detection"):
    """Met à jour le HuggingFace Space."""
    token = os.getenv("HF_TOKEN")

    if not token:
        print("ERREUR: HF_TOKEN non trouvé dans l'environnement.")
        sys.exit(1)

    login(token=token)

    api = HfApi()
    repo_id = f"yannthur/{repo_name}"

    api.create_repo(
        repo_id=repo_id, repo_type="space", exist_ok=True, space_sdk="gradio"
    )

    readme_content = """---
title: Fraud Detection
emoji: "\U0001f52e"
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
---

# Fraud Detection App

Application de détection de fraude bancaire utilisant un modèle RandomForest.

## Utilisation

Entrez les caractéristiques d'une transaction pour prédire si elle est frauduleuse.
"""

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(readme_content)
        readme_path = f.name

    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )

    os.unlink(readme_path)

    api.upload_file(
        path_or_fileobj="app.py",
        path_in_repo="app.py",
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

    print(f"Space mis à jour sur https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    update_spaces()
