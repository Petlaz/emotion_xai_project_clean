# Getting Started

## Installation

### From PyPI (when available)
```bash
pip install emotion-xai
```

### Development Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/emotion-xai-project.git
cd emotion-xai-project
```

2. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install the package in development mode:
```bash
pip install -e ".[dev]"
```

## Prerequisites

- Python 3.9 or higher
- PyTorch (CPU or GPU version)
- At least 4GB RAM for basic usage
- Additional space for model downloads

## Basic Usage

### 1. Data Preparation

```python
from emotion_xai.data.preprocessing import load_dataset, prepare_features

# Load your dataset
data = load_dataset("path/to/your/data.csv")

# Prepare text features
features = prepare_features(data, text_column="feedback_text")
```

### 2. Baseline Model Training

```python
from emotion_xai.models.baseline import train_baseline, evaluate_baseline

# Train baseline model
model = train_baseline(features, labels)

# Evaluate performance
results = evaluate_baseline(model, test_features, test_labels)
print(f"Accuracy: {results['accuracy']:.3f}")
```

### 3. Transformer Fine-tuning

```python
from emotion_xai.models.transformer import TrainingConfig, train_model

# Configure training
config = TrainingConfig(
    model_name="distilroberta-base",
    num_epochs=3,
    batch_size=16
)

# Train the model
train_model(config)
```

### 4. Generate Explanations

```python
from emotion_xai.explainability.explanations import explain_with_shap

# Generate SHAP explanations
explanations = explain_with_shap(model, input_texts)
```

### 5. Clustering Analysis

```python
from emotion_xai.clustering.feedback_clustering import embed_sentences, cluster_embeddings

# Generate embeddings
embeddings = embed_sentences(texts)

# Perform clustering
clusters = cluster_embeddings(embeddings, config)
```

## Next Steps

- Read the [User Guide](user_guide.md) for detailed usage instructions
- Check the [API Reference](api_reference.md) for complete documentation
- See [Examples](examples.md) for practical use cases