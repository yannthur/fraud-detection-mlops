"""Met à jour le HuggingFace Space avec la dernière version."""

import os
import sys

from huggingface_hub import HfApi, login


def update_spaces(repo_name: str = "fraud-detection-v2"):
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
