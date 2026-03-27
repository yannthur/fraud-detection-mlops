# fraud-detection-mlops

Pipeline MLOps complet pour la détection de fraude bancaire.

## Structure du projet

```
fraud-detection-mlops/
├── .github/workflows/    # CI/CD pipelines
├── data/                # Dataset d'entraînement
├── models/              # Modèles entraînés
├── scripts/             # Scripts utilitaires
├── src/                 # Code source
├── tests/               # Tests unitaires
├── app.py               # Application Gradio
└── train.py             # Script d'entraînement
```

## Installation

```bash
uv sync
```

## Utilisation

### Entraînement
```bash
uv run python train.py
```

### Application Gradio
```bash
uv run python app.py
```

## License

MIT
