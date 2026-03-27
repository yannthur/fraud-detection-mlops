"""Upload le modèle vers HuggingFace Hub."""

import sys
from pathlib import Path

from huggingface_hub import HfApi, login


def upload_model_to_hub(model_path: str, repo_name: str = "fraud-detection-model"):
    """Upload le modèle vers HuggingFace Hub."""
    token = (
        Path("~/.hf_token").expanduser().read_text().strip()
        if Path("~/.hf_token").exists()
        else None
    )

    if not token:
        print("ERREUR: Token HF non trouvé. Crée ~/.hf_token avec ton token.")
        sys.exit(1)

    login(token=token)

    api = HfApi()
    repo_id = f"yannthur/{repo_name}"

    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo="fraud_model.pkl",
        repo_id=repo_id,
    )

    print(f"Modèle uploadé sur https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        upload_model_to_hub(sys.argv[1])
    else:
        default_path = Path(__file__).parent.parent / "models" / "fraud_model.pkl"
        upload_model_to_hub(str(default_path))
