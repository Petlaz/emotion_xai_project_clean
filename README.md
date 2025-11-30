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

### Installation

```bash
# Clone the repository
git clone https://github.com/Petlaz/emotion_xai_project.git
cd emotion_xai_project

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

```python
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import BaselineModel
from emotion_xai.explainability.explanations import explain_with_shap

# Load and prepare data
data = load_dataset("data/processed/train_data_*.csv")
features = prepare_features(data, "text")

# Train baseline model
model = BaselineModel()
model.fit(train_texts, train_labels)

# Evaluate performance
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

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=emotion_xai tests/

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
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