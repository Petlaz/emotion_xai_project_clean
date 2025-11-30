<<<<<<< HEAD
# **Project:** Explainable AI for Emotion Detection in Social Media Text

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)

A comprehensive package for analyzing social media text using transformer models, explainable AI techniques, and clustering for theme discovery. This project implements multi-label emotion classification with state-of-the-art transformers and provides interpretable insights through explainable AI.

## 🎯 Project Overview

### Features
- **🤖 Multi-label emotion classification** using fine-tuned DistilRoBERTa (82M parameters)
- **🔍 Explainable AI** with SHAP and LIME explanations for model interpretability  
- **📊 Theme discovery** through clustering analysis using UMAP + HDBSCAN
- **🌐 Interactive web interface** built with Gradio for real-time analysis
- **📈 Comprehensive evaluation** metrics and visualizations
- **🚀 Production-ready** with optimized training pipelines

### Dataset
- **GoEmotions**: 211,225 Reddit comments with 28 emotion labels
- **Multi-label classification**: Comments can have multiple emotions
- **Processed data**: 147K training, 21K validation, 42K test samples

### Performance Targets
- **✅ Baseline Achieved**: F1-macro 0.161 (TF-IDF + Logistic Regression)
- **✅ Production Model**: F1-macro 0.196 (19.6% - DistilRoBERTa fine-tuned)
- **🎯 Ultimate Target**: F1-macro > 0.6 (60% - future optimization)

## 🚀 Quick Start
=======
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
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88

### Installation

```bash
<<<<<<< HEAD
# Clone the repository
git clone https://github.com/Petlaz/emotion_xai_project.git
cd emotion_xai_project

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
=======
pip install emotion-xai
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
```

### Basic Usage

```python
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import BaselineModel
from emotion_xai.explainability.explanations import explain_with_shap
<<<<<<< HEAD

# Load and prepare data
data = load_dataset("data/processed/train_data_*.csv")
features = prepare_features(data, "text")

# Train baseline model
model = BaselineModel()
model.fit(train_texts, train_labels)

# Evaluate performance
=======
from emotion_xai.utils.device import resolve_device

# Setup device optimization (automatic detection: CUDA/MPS/CPU)
device = resolve_device()

# Load GoEmotions dataset (211,225 samples with 28 emotion labels)
data = load_dataset("data/raw/goemotions.csv")
features = prepare_features(data, "text")

# Train baseline model (TF-IDF + Logistic Regression)
model = BaselineModel()
model.fit(train_texts, train_labels)

# Evaluate model performance (Current: F1-macro 0.161)
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
metrics = model.evaluate(val_texts, val_labels)
print(f"F1-macro: {metrics['f1_macro']:.3f}")

# Generate explanations
explanations = explain_with_shap(model, sample_texts)
```

<<<<<<< HEAD
## 🔥 Production Transformer Training

### Quick Training Options

```bash
# Option 1: Using the runner script (recommended)
./run_production_training.sh test    # Quick test (5K samples, ~5-10 min)
./run_production_training.sh full    # Full training (147K samples, ~30-60 min)

# Option 2: Direct Python execution  
python scripts/train_transformer_production.py --config configs/test_training.json
python scripts/train_transformer_production.py --config configs/production_training.json

# Option 3: Resume from checkpoint
python scripts/train_transformer_production.py \
  --config configs/production_training.json \
  --resume models/distilroberta_production_*/checkpoint-1500
```

### Training Configuration

The training system supports:
- **Mac M1/M2 optimization** with MPS acceleration
- **Memory management** for 8GB+ systems  
- **Automatic checkpointing** every 500 steps
- **Early stopping** with patience monitoring
- **Comprehensive evaluation** with multiple F1 metrics

### Current Best Model
- **✅ Production Model**: `models/distilroberta_production_20251130_044054/`
- **🏆 Training Complete**: 6,500/11,540 steps (56% of 5 epochs completed)
- **🎯 Performance**: F1-macro 19.6%, F1-micro 30.4%, Hamming Acc 96.2%
- **📈 Achievement**: 87% loss reduction (0.695 → 0.089), 1.2x baseline improvement

## 📊 Project Structure

```
emotion_xai_project/
├── 📊 data/                     # Dataset and processed features
│   ├── raw/                     # Original GoEmotions data
│   └── processed/               # Cleaned and split datasets
├── 📔 notebooks/                # Jupyter analysis notebooks (clean structure)
│   ├── 01_data_exploration.ipynb    # ✅ EDA and data quality analysis (Phase 1)
│   ├── 02_modeling.ipynb           # ✅ Baseline model development (Phase 2)
│   ├── 03_finetuning.ipynb         # ✅ Transformer model training (Phase 3)
│   ├── 04_explainability.ipynb     # ✅ Production XAI analysis (Phase 4)
│   └── 05_clustering_analysis.ipynb # ✅ Clustering & theme discovery (Phase 5)
├── 🤖 models/                   # Trained model artifacts
│   ├── distilroberta_production_20251130_044054/  # ✅ Best production model
│   └── cluster_embeddings/      # ✅ Clustering pipeline & embeddings cache
├── 🔧 scripts/                  # Production scripts and utilities
│   ├── train_transformer_production.py  # Main training script
│   ├── use_trained_model.py     # Model inference utilities
│   ├── test_clustering.py       # ✅ Clustering functionality tests
│   └── download_goemotions.py   # Dataset download utility
├── ⚙️  config/                   # General application configurations (YAML)
│   ├── production.yaml          # Production deployment settings
│   ├── development.yaml         # Development environment config
│   └── mac_optimizations.yaml   # Mac M1/M2 specific optimizations
├── 📋 configs/                  # Training-specific configurations (JSON)
│   ├── test_training.json       # Quick test config (5K samples)
│   └── production_training.json # Full training config (147K samples)
├── 📈 results/                  # Training results and visualizations
│   ├── plots/                   # All generated visualizations
│   │   ├── eda_plots/          # ✅ Data exploration plots
│   │   ├── explainability/     # ✅ XAI explanation plots
│   │   └── clustering/         # ✅ Clustering analysis plots
│   ├── production_training/     # Training logs and metrics
│   └── clustering_analysis_*.json # ✅ Clustering analysis results
├── 📦 emotion_xai/              # Core library package
│   ├── __init__.py             # Package initialization
│   ├── data/                   # ✅ Data processing utilities
│   ├── models/                 # ✅ Model implementations
│   ├── explainability/        # ✅ XAI explanations (SHAP/LIME)
│   ├── clustering/            # ✅ Theme discovery & clustering pipeline
│   └── utils/                 # ✅ Utility functions and helpers
├── 🌐 app/                      # Web interface
│   └── gradio_app.py           # Interactive Gradio application
├── 🐳 docker/                   # Containerization
│   ├── Dockerfile              # Production container setup
│   └── requirements.txt        # Docker-specific dependencies
├── 📚 docs/                     # Documentation
│   ├── documentation_report.md # Comprehensive project report
│   ├── development.md          # Development setup guide
│   └── mac_optimization.md     # Mac M1/M2 optimization guide
├── 🧪 tests/                    # Test suite
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── 📋 logs/                     # Application logs
└── 🔧 Configuration Files       # Project configuration
    ├── requirements.txt         # Main Python dependencies
    ├── pyproject.toml          # Package configuration
    ├── setup.cfg               # Setup tools configuration
    └── .pre-commit-config.yaml # Code quality hooks
```

## 🔍 Model Usage

### Using Trained Models

```python
# Load the trained model for inference
from scripts.use_trained_model import EmotionPredictor

# Initialize predictor with best model
predictor = EmotionPredictor("models/distilroberta_production_20251130_044054")

# Single prediction
emotions = predictor.predict("I love this product but the delivery was slow")
print(emotions)  # {'joy': 0.85, 'disappointment': 0.73, ...}

# Batch prediction
results = predictor.predict_batch([
    "Amazing customer service!",
    "The product broke after one day",
    "Decent quality for the price"
])

# Interactive demo
predictor.run_interactive_demo()  # Launches Gradio interface
```

### Web Interface

Launch the interactive Gradio interface:

```bash
python app/gradio_app.py
```

Access at `http://localhost:7860` for:
- Real-time emotion prediction
- Model explanations with SHAP/LIME
- Batch processing capabilities
- Visualization dashboards

## 🧠 Explainable AI

### SHAP Explanations

```python
from emotion_xai.explainability.explanations import explain_with_shap

# Generate SHAP explanations for predictions
explanations = explain_with_shap(
    model=predictor.model,
    tokenizer=predictor.tokenizer, 
    text="The service was excellent but expensive",
    top_k_emotions=5
)

# Visualize feature importance
explanations.plot()    # Shows word-level contributions
```

### LIME Explanations

```python
from emotion_xai.explainability.explanations import explain_with_lime

# Generate LIME explanations
lime_exp = explain_with_lime(
    predictor=predictor,
    text="Fast shipping, great quality product!",
    num_features=10
)

lime_exp.show_in_notebook()  # Interactive visualization
```

## 📈 Performance Monitoring

### Training Progress

Monitor training with built-in logging:

```python
# Check latest training metrics
from pathlib import Path
import json

# Load final training results
results_path = Path("results/production_training/production_results_20251130_074958.json")
with open(results_path) as f:
    results = json.load(f)
    
print(f"Final F1-macro: {results['test_results']['f1_macro']:.4f}")
print(f"Training duration: {results['training_info']['duration_minutes']:.1f} minutes")
print(f"Model location: {results['model_path']}")
```

### Evaluation Metrics

The system tracks comprehensive metrics:
- **F1-macro**: Primary metric for multi-label performance
- **F1-micro**: Overall performance across all labels  
- **F1-weighted**: Weighted by label frequency
- **Exact match**: Perfect multi-label predictions
- **Hamming accuracy**: Per-label accuracy

## 🛠️ Development

=======
### Current Performance Benchmarks (Phase 1-2 Complete)

**Data Processing Pipeline:**
- **Dataset**: GoEmotions (211,225 → 211,008 samples, 99.90% retention)
- **Quality Filtering**: 6 types of quality issues identified and handled
- **Train/Val/Test Split**: 70%/10%/20% (147,705/21,101/42,202 samples)

**Baseline Model Performance:**
- **Algorithm**: TF-IDF (10K features) + One-vs-Rest Logistic Regression
- **F1-Macro Score**: 0.161 (benchmark for transformer comparison)
- **Training Time**: ~10 seconds on Apple M1
- **Best Emotions**: Amusement (0.390), Admiration (0.356), Joy (0.319)
- **Challenging Emotions**: Annoyance (0.020), Approval (0.046), Caring (0.069)

### Advanced Training Configuration

```python
from emotion_xai.models.transformer import TrainingConfig, train_model

# Create training configuration
config = TrainingConfig(
    model_name="distilroberta-base",
    batch_size=16,
    learning_rate=2e-5,
    num_epochs=3,
    device=None  # Auto-detect best available device
)

# Train with automatic device optimization
trainer = train_model(config, train_dataset, eval_dataset)
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
│   ├── models/                 #    🤖 Machine learning models
│   │   ├── __init__.py         
│   │   ├── baseline.py         #    TF-IDF + Logistic Regression baseline
│   │   └── transformer.py     #    DistilRoBERTa fine-tuning utilities
│   ├── explainability/         #    🔍 Model interpretation & XAI
│   │   ├── __init__.py         
│   │   └── explanations.py    #    SHAP & LIME explanation generators
│   ├── clustering/             #    🎯 Theme discovery & clustering
│   │   ├── __init__.py         
│   │   └── feedback_clustering.py  # UMAP + HDBSCAN clustering
│   └── utils/                  #    🛠️ Shared utilities & configuration
│       ├── __init__.py         
│       ├── config.py           #    Configuration management classes
│       └── device.py           #    Device detection and optimization utilities
│
├── tests/                      # 🧪 Test suite (pytest-based)
│   ├── conftest.py             #    Shared test fixtures & configuration
│   ├── unit/                   #    Unit tests for individual modules
│   │   ├── test_preprocessing.py
│   │   ├── test_baseline.py
│   │   └── test_device.py      #    Device optimization tests
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
│   ├── raw/                    #    📁 Original datasets
│   │   ├── .gitkeep            #    (data files themselves are gitignored)
│   │   └── goemotions.csv      #    ✅ GoEmotions dataset (211,225 samples, 28 emotions)
│   └── processed/              #    📁 Cleaned & preprocessed data files  
│       └── .gitkeep            #    (processed data files are gitignored)
│
├── models/                     # 🤖 Saved model artifacts & checkpoints
│   ├── distilroberta_finetuned/ #    Fine-tuned transformer models
│   │   └── .gitkeep            #    (model files gitignored)
│   ├── cluster_embeddings/     #    Clustering model artifacts
│   │   └── .gitkeep            #    (model files gitignored)
│   └── .cache/                 #    HuggingFace model cache
│
├── notebooks/                  # 📓 Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb    # EDA & data understanding
│   ├── 02_modeling.ipynb            # ✅ Baseline modeling & preprocessing (COMPLETED)
│   ├── 03_finetuning.ipynb          # Transformer fine-tuning experiments (NEXT)
│   ├── 04_explainability.ipynb      # XAI analysis & visualization
│   └── 05_clustering_analysis.ipynb # Theme discovery analysis
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
├── results/                    # 📊 Analysis results & outputs
│   ├── README.md               #    Results overview & documentation
│   ├── metrics/                #    📈 EDA statistics, model performance, evaluations
│   │   ├── eda_statistics_*.json     # Comprehensive EDA findings
│   │   ├── emotion_distribution_*.csv # Emotion frequency analysis
│   │   ├── emotion_correlations_*.csv # Correlation matrices
│   │   └── text_quality_assessment_*.csv # Data quality metrics
│   └── plots/                  #    📊 Visualizations & charts
│       ├── eda_plots/          #    EDA visualization outputs
│       ├── model_performance/  #    Model evaluation charts
│       └── explainability/     #    XAI visualization outputs
│
├── logs/                       # 📝 Application logs (gitignored content)
│   └── .gitignore              #    Log files exclusion rules
│
├── .github/workflows/          # 🚀 CI/CD automation (GitHub Actions)
│   └── ci.yml                  #    Test, lint, & deployment pipeline
│
├── .venv/                      # 🐍 Virtual environment (gitignored)
│   └── ...                     #    Python 3.11.13 with all dependencies installed
│
├── pyproject.toml              # 📋 Modern Python packaging configuration
├── requirements.txt            # 📦 Production dependencies
├── PROJECT_PLAN.md             # 📋 Detailed implementation roadmap
├── IMPLEMENTATION_CHECKLIST.md # ✅ Task tracking with priorities
├── LICENSE                     # 📄 MIT License
├── .pre-commit-config.yaml     # 🔍 Pre-commit hooks for code quality
├── .gitignore                  # 📄 Git ignore patterns
└── README.md                   # � This file
```

**🎯 Current Implementation Status:**
- ✅ **Infrastructure Complete**: Professional package structure & tooling
- ✅ **Device Optimization**: Multi-platform device detection and acceleration  
- ✅ **Dataset Ready**: GoEmotions dataset downloaded (211,225 samples, 28 emotions)
- ✅ **Environment Setup**: Virtual environment with all ML dependencies
- ✅ **Phase 1 Complete**: Data preprocessing pipeline (99.90% quality retention)
- ✅ **Phase 2 Complete**: Baseline modeling (TF-IDF + LogReg, F1-macro: 0.161)
- 🚧 **Next Phase**: Transformer fine-tuning (DistilRoBERTa target: F1-macro >0.6)

**Key Design Principles:**
- **Cross-Platform Development**: Optimized for various hardware configurations
- **Separation of Concerns**: Each directory has a single, clear responsibility
- **Scalability**: Structure supports growing from prototype to production  
- **Reproducibility**: Configuration management & environment isolation
- **Collaboration**: Clear testing, documentation & contribution workflows

> **🚨 IMPORTANT - Data vs Code Separation:**
> - `emotion_xai/data/` = **Python modules** for data processing (code)
> - `data/` (project root) = **Actual dataset files** (CSV, JSON, etc.)
> - This follows best practices: code is installable, data stays with project

> **📊 Dataset Information:**
> - **GoEmotions**: 211,225 Reddit comments with 28 emotion labels
> - **Multi-label**: Average 1.18 emotions per comment
> - **Diversity**: Comments from 483 different subreddits
> - **Quality**: Human-annotated for research purposes

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

>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
<<<<<<< HEAD
pytest --cov=emotion_xai tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
=======
pytest --cov=emotion_xai --cov-report=html

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
```

### Code Quality

```bash
<<<<<<< HEAD
# Code formatting
black emotion_xai/ scripts/ tests/
isort emotion_xai/ scripts/ tests/

# Linting  
flake8 emotion_xai/ scripts/ tests/
mypy emotion_xai/

# Security check
bandit -r emotion_xai/
```

### Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`black`, `flake8`, `pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 🚀 Deployment

### Docker Support

```bash
# Build the Docker image
docker build -t emotion-xai .

# Run the container
docker run -p 7860:7860 emotion-xai

# With GPU support
docker run --gpus all -p 7860:7860 emotion-xai
```

### Production Deployment

The system is designed for production with:
- **Scalable inference** with batch processing
- **API endpoints** via Gradio/FastAPI integration
- **Model versioning** with checkpoint management
- **Monitoring** with comprehensive metrics
- **Docker containerization** for consistent deployments

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **GoEmotions Dataset**: Google Research for the comprehensive emotion dataset
- **Hugging Face**: Transformers library and model hub
- **SHAP/LIME**: Explainable AI framework contributions
- **Gradio**: Interactive ML interface framework

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/Petlaz/emotion_xai_project/issues)
- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: See `notebooks/` for usage examples

---

**Status**: ✅ **Phase 5 Complete** | **🚀 Phase 6 Ready**: Complete emotion analysis pipeline with clustering & theme discovery, explainable AI, and production transformer model (F1-macro 0.196). Next: Interactive web interface development for deployment.
=======
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
# Test device optimizations (automatic CUDA/MPS/CPU detection)
python emotion_xai/utils/device.py

# Train baseline model with optimized settings
emotion-xai train-baseline --data-path data/raw/goemotions.csv --text-column text

# Train transformer model with automatic device detection
emotion-xai train-transformer --model-name distilroberta-base --epochs 3 --batch-size 16

# Download GoEmotions dataset
python scripts/download_goemotions.py

# Show help for all available commands
emotion-xai --help
```

### Development Commands

```bash
# Verify device optimizations
python -c "from emotion_xai.utils.device import resolve_device; print('Device:', resolve_device())"

# Check dataset status
python -c "import pandas as pd; df = pd.read_csv('data/raw/goemotions.csv'); print(f'Dataset: {len(df):,} samples ready')"

# Run tests
pytest tests/

# Code quality checks
black emotion_xai tests && isort emotion_xai tests && flake8 emotion_xai tests
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

## Project Progress Status

### ✅ Completed Phases

**Phase 1: Data Preprocessing & EDA**
- GoEmotions dataset integration (211,225 samples, 28 emotions)
- Comprehensive data quality assessment (99.90% retention rate)
- Text cleaning pipeline with conservative/aggressive approaches
- Interactive visualizations and statistical analysis

**Phase 2: Baseline Modeling**
- TF-IDF feature extraction with optimization
- One-vs-Rest Logistic Regression for multi-label classification
- Comprehensive evaluation framework (F1-macro: 0.161)
- Model persistence and artifact management
- Complete notebook implementation (`notebooks/02_modeling.ipynb`)

### 🚀 Next Phase: Transformer Fine-tuning

**Target Goals:**
- DistilRoBERTa fine-tuning for emotion classification
- Target performance: F1-macro > 0.6 (4x improvement over baseline)
- Multi-label classification head with label smoothing
- Comprehensive comparison with baseline model

## Acknowledgments

- Built with [Transformers](https://huggingface.co/transformers/) by Hugging Face
- Uses [SHAP](https://github.com/slundberg/shap) for explainable AI
- Clustering powered by [UMAP](https://umap-learn.readthedocs.io/) and [HDBSCAN](https://hdbscan.readthedocs.io/)
- Web interface built with [Gradio](https://gradio.app/)
- Dataset: [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) by Google Research
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
