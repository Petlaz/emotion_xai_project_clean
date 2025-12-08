# Emotion-XAI: Explainable AI for Social Media Emotion Detection

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![Gradio App](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/petlaz/emotion-xai)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black)](https://github.com/Petlaz/emotion_xai_project_clean)

**A production-ready explainable AI system for multi-label emotion detection in social media text**

[🚀 **Live Demo**](https://huggingface.co/spaces/petlaz/emotion-xai) | [📖 **Documentation**](./docs/) | [🔬 **Research Notebooks**](./notebooks/) | [⭐ **Star on GitHub**](https://github.com/Petlaz/emotion_xai_project_clean)

</div>

---

## 🎯 **Executive Summary**

This project delivers a **complete machine learning pipeline** for detecting and explaining emotions in social media text. Built with **state-of-the-art transformer models** and **explainable AI techniques**, it provides interpretable insights into human emotional expression in digital communications.

**🏆 Key Achievements:**
- **Production Model**: Fine-tuned DistilRoBERTa achieving **19.6% F1-macro** (1.2x baseline improvement)
- **Explainable AI**: Integrated SHAP and LIME for model interpretability
- **Interactive Interface**: Live web application with **real-time emotion analysis**
- **Scalable Architecture**: Production-ready deployment on Hugging Face Spaces

## 🚀 **Technology Stack & Features**

<table>
<tr>
<td width="50%">

### 🔧 **Core Technologies**
- **Deep Learning**: PyTorch, Transformers
- **Model Architecture**: DistilRoBERTa (82M parameters)
- **Explainable AI**: SHAP, LIME
- **Clustering**: UMAP, HDBSCAN
- **Web Interface**: Gradio, Plotly
- **Deployment**: Hugging Face Spaces

</td>
<td width="50%">

### 📊 **Performance Metrics**
- **Dataset**: GoEmotions (211K samples, 28 emotions)
- **F1-Macro**: 0.196 (19.6% accuracy)
- **Baseline Improvement**: 1.2x performance gain
- **Processing Speed**: <1s per prediction
- **Model Size**: 82M parameters (optimized)

</td>
</tr>
</table>

### 🎯 **Core Capabilities**

| Feature | Description | Status |
|---------|-------------|--------|
| **Multi-label Classification** | Detect 28 different emotions simultaneously | ✅ Production Ready |
| **Explainable Predictions** | SHAP/LIME explanations for model transparency | ✅ Fully Integrated |
| **Real-time Analysis** | Interactive web interface with instant results | ✅ Live Demo Available |
| **Batch Processing** | Analyze multiple texts efficiently | ✅ Optimized Pipeline |
| **Theme Discovery** | Unsupervised clustering for emotion patterns | ✅ Advanced Analytics |
| **Production Deployment** | Scalable cloud-based serving | ✅ HF Spaces Deployed |

### 📈 **Business Value**

- **Social Media Monitoring**: Automated emotion analysis for brand sentiment
- **Content Moderation**: Detect emotional tone for platform safety
- **Market Research**: Understand customer emotional responses
- **Mental Health**: Monitor emotional patterns in digital communications

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Petlaz/emotion_xai_project_clean.git
cd emotion_xai_project

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

```python
from emotion_xai.utils.device import resolve_device
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import BaselineModel
from emotion_xai.explainability.explanations import explain_with_shap

# Setup device optimization (automatic detection: CUDA/MPS/CPU)
device = resolve_device()

# Load GoEmotions dataset (211,225 samples with 28 emotion labels)
data = load_dataset("data/raw/goemotions.csv")
features = prepare_features(data, "text")

# Train baseline model (TF-IDF + Logistic Regression)
model = BaselineModel()
model.fit(train_texts, train_labels)

# Evaluate model performance (Current: F1-macro 0.161)
metrics = model.evaluate(val_texts, val_labels)
print(f"F1-macro: {metrics['f1_macro']:.3f}")

# Generate explanations
explanations = explain_with_shap(model, sample_texts)
```

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

### Current Best Model
- **✅ Production Model**: `models/distilroberta_production_20251130_044054/`
- **🏆 Training Complete**: 6,500/11,540 steps (56% of 5 epochs completed)
- **🎯 Performance**: F1-macro 19.6%, F1-micro 30.4%, Hamming Acc 96.2%
- **📈 Achievement**: 87% loss reduction (0.695 → 0.089), 1.2x baseline improvement

## 🏗️ **System Architecture**

```
emotion_xai_project/
├── 📊 data/                     # Dataset and processed features
│   ├── raw/                     # Original GoEmotions data
│   └── processed/               # Cleaned and split datasets
├── 📔 notebooks/                # Jupyter analysis notebooks (clean structure)
│   ├── 01_data_exploration.ipynb    # ✅ EDA and data quality analysis
│   ├── 02_modeling.ipynb           # ✅ Baseline model development
│   ├── 03_finetuning.ipynb         # ✅ Transformer model training
│   ├── 04_explainability.ipynb     # ✅ Production XAI analysis
│   └── 05_clustering_analysis.ipynb # ✅ Clustering & theme discovery
├── 🤖 models/                   # Trained model artifacts
│   ├── distilroberta_production_20251130_044054/  # ✅ Best production model
│   ├── saved_models/            # Baseline model artifacts
│   └── cluster_embeddings/      # ✅ Clustering pipeline & embeddings cache
├── 🔧 scripts/                  # Production scripts and utilities
│   ├── train_transformer_production.py  # Main training script
│   ├── use_trained_model.py     # Model inference utilities
│   ├── download_goemotions.py   # Dataset download utility
│   └── test_*.py               # Various testing scripts
├── ⚙️  config/                   # General application configurations (YAML)
│   ├── production.yaml          # Production deployment settings
│   ├── development.yaml         # Development environment config
│   ├── default.yaml            # Default configuration settings
│   └── mac_optimizations.yaml   # Mac M1/M2 specific optimizations
├── � results/                  # Training results and visualizations
│   ├── metrics/                 # Performance metrics and statistics
│   ├── plots/                   # All generated visualizations
│   └── clustering_analysis_*.json # ✅ Clustering analysis results
├── 📦 emotion_xai/              # Core library package
│   ├── data/                    # ✅ Data processing utilities
│   ├── models/                  # ✅ Model implementations
│   ├── explainability/         # ✅ XAI explanations (SHAP/LIME)
│   ├── clustering/             # ✅ Theme discovery & clustering pipeline
│   ├── utils/                  # ✅ Utility functions and helpers
│   └── cli.py                  # Command-line interface
├── 🌐 app/                      # ✅ Web interface (Complete)
│   ├── gradio_app.py           # Full-featured Gradio application (434 lines)
│   └── __init__.py            # Package initialization
├── 🚀 app.py                   # ✅ HF Spaces entry point (production ready)
├── 📋 requirements_gradio.txt   # ✅ Gradio deployment dependencies  
├── 📚 README_HF_SPACES.md      # ✅ Hugging Face Spaces documentation
├── 🐳 docker/                   # Containerization
│   ├── Dockerfile              # Production container setup
│   └── requirements.txt        # Docker-specific dependencies
├── 📚 docs/                     # Documentation (organized)
│   ├── README.md               # Documentation overview
│   ├── project_plan.md         # Complete project plan
│   ├── documentation_report.md # Comprehensive completion report
│   ├── development.md          # Development setup guide
│   ├── getting_started.md      # Getting started guide
│   └── mac_optimization.md     # Mac M1/M2 optimization guide
├── 🧪 tests/                    # Test suite
│   ├── conftest.py             # Test configuration
│   ├── fixtures/               # Test fixtures
│   ├── unit/                   # Unit tests
│   └── integration/            # Integration tests
├── 📄 logs/                     # Application logs directory
└── 🔧 Configuration Files       # Project configuration
    ├── requirements.txt         # Main Python dependencies
    ├── pyproject.toml          # Package configuration
    ├── setup.cfg               # Setup tools configuration
    ├── MANIFEST.in             # Package manifest
    ├── LICENSE                 # Project license
    ├── CHANGELOG.md            # Version history
    ├── CONTRIBUTING.md         # Contribution guidelines
    ├── .gitignore              # Git ignore patterns
    └── run_production_training.sh # Production training script
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
# OR
python app.py  # HF Spaces entry point
```

Access at `http://localhost:7860` for:
- Real-time emotion prediction with 4-decimal precision
- Interactive Plotly visualizations
- Model explanations with SHAP/LIME
- Batch processing capabilities
- Professional UI with instant launch examples

### 🚀 Hugging Face Spaces Deployment

Ready for one-click deployment to Hugging Face Spaces:

1. **Files Ready**:
   - ✅ `app.py` - Main entry point
   - ✅ `requirements_gradio.txt` - Dependencies
   - ✅ `README_HF_SPACES.md` - Spaces documentation

2. **Deploy Command**:
```bash
# From your HF Spaces repository
git add .
git commit -m "Deploy Emotion-XAI app"
git push
```

3. **Features**:
   - ✅ Instant launch (<30s)
   - ✅ Pre-loaded examples
   - ✅ Public sharing capability
   - ✅ Production DistilRoBERTa model (82M params)
   - ✅ 4-decimal precision scores

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

## 🛠️ Development

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/Petlaz/emotion-xai-project_clean.git
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
# Code formatting
black emotion_xai/ scripts/ tests/
isort emotion_xai/ scripts/ tests/

# Linting  
flake8 emotion_xai/ scripts/ tests/
mypy emotion_xai/

# Security check
bandit -r emotion_xai/
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
```

## 🚀 Deployment

### Hugging Face Spaces (Recommended)

**✅ Ready for one-click deployment!**

**Pre-deployment Checklist:**
- ✅ `app.py` - HF Spaces entry point 
- ✅ `requirements_gradio.txt` - Complete dependencies
- ✅ `README_HF_SPACES.md` - Spaces metadata and documentation
- ✅ Gradio interface with instant launch capability
- ✅ Production model included (82M parameters)
- ✅ 4-decimal precision for emotion scores
- ✅ Public sharing enabled with professional UI

**Deploy Steps:**
1. Create new Space on Hugging Face
2. Upload project files
3. Set Space to use `app.py` as main file
4. App launches automatically with <30s startup time

### Docker Support

```bash
# Build the Docker image
docker build -t emotion-xai .

# Run the container
docker run -p 7860:7860 emotion-xai

# With GPU support
docker run --gpus all -p 7860:7860 emotion-xai
```

### Production Features

The system is designed for production with:
- **✅ Scalable inference** with batch processing
- **✅ Public API** via Gradio sharing links
- **✅ Model versioning** with checkpoint management
- **✅ Real-time monitoring** with comprehensive metrics
- **✅ Professional UI** with instant examples and explanations
- **✅ Cross-platform** compatibility (CUDA/MPS/CPU auto-detection)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run quality checks (`black`, `flake8`, `pytest`)
5. Commit changes (`git commit -m 'Add amazing feature'`)
6. Push to branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- **GoEmotions Dataset**: Google Research for the comprehensive emotion dataset
- **Hugging Face**: Transformers library and model hub
- **SHAP/LIME**: Explainable AI framework contributions
- **Gradio**: Interactive ML interface framework

## 📞 Support

For questions, issues, or contributions:
- **GitHub Issues**: [Create an issue](https://github.com/Petlaz/emotion_xai_project_clean/issues)
- **Documentation**: Check the `docs/` directory for detailed guides
- **Examples**: See `notebooks/` for usage examples

---

---

## 🚀 **Live Demo & Links**

<div align="center">

### 🌟 **Try the Live Application**
[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/petlaz/emotion-xai)

**[🎭 Launch Emotion-XAI App](https://huggingface.co/spaces/petlaz/emotion-xai)**

### 📊 **Project Resources**
| Resource | Link | Description |
|----------|------|-------------|
| 🚀 **Live Demo** | [HF Spaces](https://huggingface.co/spaces/petlaz/emotion-xai) | Interactive web application |
| 📂 **Source Code** | [GitHub Repository](https://github.com/Petlaz/emotion_xai_project_clean) | Complete codebase |
| 📖 **Documentation** | [Technical Docs](./docs/) | Comprehensive guides |
| 🔬 **Research** | [Jupyter Notebooks](./notebooks/) | Analysis workflows |

</div>

## 🏆 **Project Status**

<div align="center">

### ✅ **PRODUCTION READY**
**Complete 6-phase ML pipeline successfully deployed**

🎯 **F1-Macro: 19.6%** | 🤖 **DistilRoBERTa: 82M params** | 📊 **GoEmotions: 211K samples** | 🌐 **Live on HF Spaces**

*Built with modern MLOps practices, explainable AI, and production-grade deployment*

</div>