"""Met à jour le HuggingFace Space avec la dernière version."""

import os
import sys

from huggingface_hub import HfApi, login


def update_spaces(repo_name: str = "fraud-detection", space_file: str = "app.py"):
    """Met à jour le HuggingFace Space."""
    token = os.getenv("HF_TOKEN")

    if not token:
        print("ERREUR: HF_TOKEN non trouvé dans l'environnement.")
        sys.exit(1)

    login(token=token)

    api = HfApi()
    repo_id = f"yannthur/{repo_name}"

    api.create_repo(repo_id=repo_id, repo_type="space", exist_ok=True)

    api.upload_file(
        path_or_fileobj=space_file,
        path_in_repo="app.py",
        repo_id=repo_id,
        repo_type="space",
    )

    print(f"Space mis à jour sur https://huggingface.co/spaces/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        update_spaces(space_file=sys.argv[1])
    else:
        update_spaces()
