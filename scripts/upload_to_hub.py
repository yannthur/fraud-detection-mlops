"""Upload le modèle vers HuggingFace Hub."""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, login


def upload_model_to_hub(model_path: str, repo_name: str = "fraud-detection-model"):
    """Upload le modèle vers HuggingFace Hub."""
    token = os.getenv("HF_TOKEN")

    if not token:
        print("ERREUR: HF_TOKEN non trouvé dans l'environnement.")
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
        models_dir = Path(__file__).parent.parent / "models"
        default_path = models_dir / "fraud_model.pkl"
        upload_model_to_hub(str(default_path))
