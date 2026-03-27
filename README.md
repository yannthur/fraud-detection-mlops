# Fraud Detection MLOps

Pipeline MLOps complet pour la détection de fraude bancaire - Projet M2 IABD 2026.

## Description du projet

Ce projet implémente un pipeline MLOps complet pour détecter les transactions bancaires frauduleuses à partir d'un dataset de 100 000 transactions avec 23 features.

## Structure du projet

```
fraud-detection-mlops/
├── .github/
│   └── workflows/
│       └── main.yml           # Pipeline CI/CD GitHub Actions
├── .git/
│   └── hooks/                 # Hooks Git manuels
├── data/
│   └── train.csv             # Dataset (tracké par Git LFS)
├── models/                    # Modèles entraînés (Git LFS)
├── scripts/
│   ├── generate_email_report.py  # Génération rapport email (LLM)
│   ├── upload_to_hub.py          # Upload modèle vers HF Hub
│   └── update_spaces.py          # Mise à jour HF Spaces
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py      # Prétraitement des données
│   └── model.py                   # Modèle ML (RandomForest)
├── tests/
│   ├── __init__.py
│   └── test_model.py              # Tests unitaires
├── app.py                     # Application Gradio
├── train.py                   # Script d'entraînement
├── requirements.txt           # Dépendances Python
├── pyproject.toml             # Configuration outils
├── .pre-commit-config.yaml    # Hooks pre-commit
├── .gitignore
├── .gitattributes             # Configuration Git LFS
├── .secrets.baseline          # Baseline detect-secrets
├── CHANGELOG.md
└── README.md
```

## Installation

### Prérequis

- Python 3.10+
- Git
- Git LFS

### Setup

```bash
# Cloner le repository
git clone https://github.com/yannthur/fraud-detection-mlops.git
cd fraud-detection-mlops

# Installer Git LFS
git lfs install

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou .venv\Scripts\activate sur Windows

# Installer les dépendances
pip install -r requirements.txt

# Installer les hooks pre-commit
pre-commit install
pre-commit install --hook-type commit-msg
```

## Utilisation

### Entraînement du modèle

```bash
python train.py
```

Le modèle sera sauvegardé dans `models/fraud_model.pkl`.

### Application Gradio

```bash
python app.py
```

L'interface sera disponible sur `http://localhost:7860`.

### Tests

```bash
pytest tests/ -v
```

## Architecture Git Flow

```
main (production)
  ↑
  └── release/v1.0.0
         ↑
         └── develop (intégration)
                ↑
                ├── feature/data-preprocessing
                ├── feature/model-training
                ├── feature/tests
                └── feature/gradio-app
```

### Branches permanentes

- `main` : Code de production, protégée (PR obligatoire + 1 approbation + tests)
- `develop` : Code d'intégration, protégée (PR obligatoire + 1 approbation)

### Workflow

1. Créer une feature branch depuis `develop`
2. Développer et committer (format: `type(scope): description`)
3. Créer une PR vers `develop`
4. Après review, merger dans `develop`
5. Pour une release, merge `develop` → `main`

## Git Hooks

### Pre-commit (automatique)

Exécuté à chaque `git commit`:
- `black` : Formatage Python
- `isort` : Tri des imports
- `flake8` : Linting PEP 8
- `detect-secrets` : Détection de secrets

### Commit-msg (automatique)

Valide le format du message de commit selon Conventional Commits:
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`, etc.
- Format: `type(scope): description`

### Pre-push (manuel)

Vérifie que les fichiers >5 Mo sont trackés par Git LFS.

## Pipeline CI/CD

Le workflow GitHub Actions (`.github/workflows/main.yml`) exécute:

1. **Job test-and-validate** :
   - Linting (`flake8`, `black`)
   - Tests unitaires (`pytest`)
   - Entraînement du modèle
   - Validation de l'accuracy >= 80%

2. **Job notify** :
   - Génération de rapport via Gemini LLM
   - Envoi d'email aux équipes

3. **Job deploy** (main seulement) :
   - Upload du modèle vers Hugging Face Hub
   - Mise à jour du HF Space Gradio

## Git LFS

Les fichiers suivants sont trackés par Git LFS:
- `*.csv` : Datasets
- `*.pkl` : Modèles sérialisés
- `*.pt`, `*.pth` : Modèles PyTorch
- `*.h5` : Modèles Keras/TensorFlow
- `*.onnx` : Modèles ONNX
- `*.parquet` : Données Parquet
- `*.joblib` : Objets Joblib

## Configuration des secrets GitHub

Configurer dans Settings → Secrets and variables → Actions:

| Secret | Description |
|--------|-------------|
| `HF_TOKEN` | Token Hugging Face (write access) |
| `GEMINI_API_KEY` | Clé API Google Gemini |
| `SMTP_SERVER` | Serveur SMTP (ex: smtp.gmail.com) |
| `SMTP_PORT` | Port SMTP (ex: 587) |
| `SMTP_USERNAME` | Adresse email expéditrice |
| `SMTP_PASSWORD` | Mot de passe d'application |
| `MAIL_TO` | Email(s) destinataires |
| `HF_SPACE_URL` | URL du HuggingFace Space |

## Dataset

Format du dataset (`data/train.csv`):

| Colonne | Description |
|---------|-------------|
| `amt` | Montant de la transaction |
| `lat`, `long` | Latitude/Longitude du client |
| `merch_lat`, `merch_long` | Coordonnées du marchand |
| `city_pop` | Population de la ville |
| `category` | Catégorie de commerce |
| `gender` | Genre du client |
| `state` | État |
| `job` | Profession |
| `dob` | Date de naissance |
| `is_fraud` | Cible (0 = légitime, 1 = fraude) |

## Modèle

- **Type**: RandomForest Classifier
- **Features**: `amt`, `lat`, `long`, `city_pop`, `merch_lat`, `merch_long`, `category`, `gender`, `state`, `job`, `age`, `distance`
- **Métriques cibles**: Accuracy >= 80%

## Liens

- **GitHub**: https://github.com/yannthur/fraud-detection-mlops
- **Hugging Face Hub**: https://huggingface.co/yannthur/fraud-detection-model
- **Hugging Face Space**: https://huggingface.co/spaces/yannthur/fraud-detection

## Licence

MIT

## Auteurs

Projet M2 IABD 2026
