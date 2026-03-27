"""Met à jour le HuggingFace Space avec la dernière version."""

import sys
from pathlib import Path

from huggingface_hub import HfApi, login


HF_TOKEN_FILE = Path("~/.hf_token").expanduser()


def update_spaces(repo_name: str = "fraud-detection", space_file: str = "app.py"):
    """Met à jour le HuggingFace Space."""
    if not HF_TOKEN_FILE.exists():
        print("ERREUR: Token HF non trouvé. Crée ~/.hf_token avec ton token.")
        sys.exit(1)

    token = HF_TOKEN_FILE.read_text().strip()
    login(token=token)

    api = HfApi()
    repo_id = f"yannthur/{repo_name}"

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
