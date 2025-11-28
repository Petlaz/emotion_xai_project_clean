# Emotion-Aware Customer Feedback Analysis with Explainable AI

[![CI/CD Pipeline](https://github.com/Petlaz/emotion-xai-project/workflows/CI%2FCD%20Pipeline/badge.svg)](https://github.com/Petlaz/emotion-xai-project/actions)
[![codecov](https://codecov.io/gh/Petlaz/emotion-xai-project/branch/main/graph/badge.svg)](https://codecov.io/gh/Petlaz/emotion-xai-project)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A comprehensive package for analyzing customer feedback using transformer models, explainable AI techniques, and clustering for theme discovery.

## Features

- **Multi-label emotion classification** using fine-tuned transformer models (DistilRoBERTa)
- **Explainable AI** with SHAP and LIME explanations for model interpretability
- **Theme discovery** through clustering analysis using UMAP + HDBSCAN
- **Interactive web interface** built with Gradio for real-time analysis
- **Comprehensive evaluation** metrics and visualizations
- **Production-ready** with Docker support and CI/CD pipelines

## Quick Start

### Installation

```bash
pip install emotion-xai
```

### Basic Usage

```python
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import train_baseline
from emotion_xai.explainability.explanations import explain_with_shap

# Load and prepare data (data files are in project root data/ directory)
data = load_dataset("data/raw/goemotions.csv")  # Raw data location
features = prepare_features(data, "text_column")

# Train baseline model
model = train_baseline(features, labels)

# Generate explanations
explanations = explain_with_shap(model, sample_texts)
```

## Project Structure

```
emotion_xai_project/
├── emotion_xai/                 # 📦 Main Python package (installable)
│   ├── __init__.py             #    Package initialization & exports
│   ├── cli.py                  #    Command-line interface entry point
│   ├── data/                   #    📝 Data processing modules (Python code)
│   │   ├── __init__.py         #    NOT actual data - just processing code!
│   │   └── preprocessing.py    #    Text cleaning & feature preparation functions
│   ├── models/                 #    Machine learning models
│   │   ├── __init__.py         
│   │   ├── baseline.py         #    TF-IDF + Logistic Regression baseline
│   │   └── transformer.py     #    DistilRoBERTa fine-tuning utilities
│   ├── explainability/         #    Model interpretation & XAI
│   │   ├── __init__.py         
│   │   └── explanations.py    #    SHAP & LIME explanation generators
│   ├── clustering/             #    Theme discovery & clustering
│   │   ├── __init__.py         
│   │   └── feedback_clustering.py  # UMAP + HDBSCAN clustering
│   └── utils/                  #    Shared utilities & configuration
│       ├── __init__.py         
│       └── config.py           #    Configuration management classes
│
├── tests/                      # 🧪 Test suite (pytest-based)
│   ├── conftest.py             #    Shared test fixtures & configuration
│   ├── unit/                   #    Unit tests for individual modules
│   │   ├── test_preprocessing.py
│   │   └── test_baseline.py
│   ├── integration/            #    Integration & end-to-end tests
│   │   └── test_pipeline.py
│   └── fixtures/               #    Test data & mock objects
│       └── __init__.py
│
├── docs/                       # 📚 Documentation (Markdown & Sphinx)
│   ├── README.md               #    Documentation overview
│   ├── getting_started.md      #    Installation & quick start guide
│   ├── development.md          #    Development setup & contributing
│   └── documentation_report.md #    Project documentation report
│
├── config/                     # ⚙️ Configuration files (YAML-based)
│   ├── default.yaml            #    Default configuration settings
│   ├── development.yaml        #    Development environment config
│   └── production.yaml         #    Production environment config
│
├── data/                       # 💾 ACTUAL data files (at project root)
│   ├── raw/                    #    📁 Original datasets (e.g., goemotions.csv)
│   │   └── .gitkeep            #    (data files themselves are gitignored)
│   └── processed/              #    📁 Cleaned & preprocessed data files  
│       └── .gitkeep            #    (processed data files are gitignored)
│
├── models/                     # 🤖 Saved model artifacts & checkpoints
│   ├── distilroberta_finetuned/ #    Fine-tuned transformer models
│   │   └── .gitkeep            #    (model files gitignored)
│   └── cluster_embeddings/     #    Clustering model artifacts
│       └── .gitkeep            #    (model files gitignored)
│
├── notebooks/                  # 📓 Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb     # EDA & data understanding
│   ├── 02_finetuning.ipynb          # Model training experiments
│   ├── 03_explainability.ipynb      # XAI analysis & visualization
│   └── 04_clustering_analysis.ipynb # Theme discovery analysis
│
├── app/                        # 🌐 Web application (Gradio interface)
│   └── gradio_app.py           #    Interactive demo & API server
│
├── docker/                     # 🐳 Containerization & deployment
│   ├── Dockerfile              #    Multi-stage Docker build
│   └── requirements.txt        #    Docker-specific dependencies
│
├── scripts/                    # 🛠️ Utility scripts for setup & automation
│   └── download_goemotions.py  #    GoEmotions dataset download utility
│
├── logs/                       # 📝 Application logs (gitignored content)
│   └── .gitignore              #    Log files exclusion rules
│
├── .github/workflows/          # 🚀 CI/CD automation (GitHub Actions)
│   └── ci.yml                  #    Test, lint, & deployment pipeline
│
├── pyproject.toml              # 📋 Modern Python packaging configuration
├── requirements.txt            # 📦 Production dependencies
├── requirements-dev.txt        # 🛠️ Development dependencies
├── setup.cfg                   # ⚙️ Tool configuration (pytest, flake8, mypy)
├── .pre-commit-config.yaml     # 🔍 Pre-commit hooks for code quality
├── MANIFEST.in                 # 📄 Package distribution files
├── CHANGELOG.md                # 📅 Version history & release notes
└── CONTRIBUTING.md             # 🤝 Contribution guidelines & workflow
```

**Key Design Principles:**
- **Separation of Concerns**: Each directory has a single, clear responsibility
- **Scalability**: Structure supports growing from prototype to production  
- **Reproducibility**: Configuration management & environment isolation
- **Collaboration**: Clear testing, documentation & contribution workflows

> **🚨 IMPORTANT - Data vs Code Separation:**
> - `emotion_xai/data/` = **Python modules** for data processing (code)
> - `data/` (project root) = **Actual dataset files** (CSV, JSON, etc.)
> - This follows best practices: code is installable, data stays with project

> **Note**: This structure follows Python packaging best practices with:
> - **Package code** in `emotion_xai/` (installable via pip)
> - **Project data** in `data/` (separate from package code)
> - **Configuration** externalized in `config/` 
> - **Tests** isolated in `tests/` with proper fixtures

## Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/Petlaz/emotion-xai-project.git
cd emotion-xai-project
```

2. Create virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install in development mode:
```bash
pip install -e ".[dev]"
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emotion_xai --cov-report=html

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Code Quality

```bash
# Format code
black emotion_xai tests

# Sort imports
isort emotion_xai tests

# Lint code
flake8 emotion_xai tests

# Type checking
mypy emotion_xai
```

## CLI Usage

The package provides a command-line interface for common tasks:

```bash
# Train baseline model (using data from project root data/ directory)
emotion-xai train-baseline --data-path data/raw/goemotions.csv --text-column text

# Train transformer model with custom parameters
emotion-xai train-transformer --model-name distilroberta-base --epochs 3 --batch-size 16

# Show help for all available commands
emotion-xai --help
```

## Documentation

- [Getting Started](docs/getting_started.md)
- [User Guide](docs/user_guide.md)
- [Development Guide](docs/development.md)
- [API Reference](docs/api_reference.md)

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers/) by Hugging Face
- Uses [SHAP](https://github.com/slundberg/shap) for explainable AI
- Clustering powered by [UMAP](https://umap-learn.readthedocs.io/) and [HDBSCAN](https://hdbscan.readthedocs.io/)
- Web interface built with [Gradio](https://gradio.app/)