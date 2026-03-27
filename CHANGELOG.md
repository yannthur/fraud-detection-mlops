# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.0] - 2026-03-27

### Added
- Initial project structure with Git Flow branching
- Git LFS configuration for ML assets (csv, pkl, pt, h5, onnx, parquet, joblib)
- Pre-commit framework with black, isort, flake8, detect-secrets
- Commit-msg hook for Conventional Commits validation
- Pre-push hook for large file verification
- Data preprocessing module with feature engineering
  - Age calculation from date of birth
  - Distance calculation between client and merchant
  - Categorical encoding with LabelEncoder
  - Stratified train/test split
- ML model module (RandomForest Classifier)
  - Training, prediction, evaluation methods
  - Model save/load with joblib
  - Feature importance extraction
- Training script (`train.py`)
- Gradio application for inference (`app.py`)
- Unit tests with pytest
- CI/CD pipeline with GitHub Actions
  - Linting (flake8, black)
  - Testing (pytest)
  - Model training and validation
  - Email notification via Gemini LLM
  - Hugging Face deployment
- Comprehensive documentation (README, CHANGELOG)
- GitHub Secrets configuration for CI/CD

### Fixed
- Updated workflow to use GEMINI_API_KEY instead of ANTHROPIC_API_KEY
- Fixed train.py imports to match data_preprocessing module
- Updated model.py to use load_and_prepare from data_preprocessing

### Security
- Detect-secrets baseline for secret detection
- Branch protection rules on main and develop
