# Emotion-XAI Documentation

Welcome to the Emotion-XAI project documentation.

## Table of Contents

- [Getting Started](getting_started.md)
- [API Reference](api_reference.md)
- [User Guide](user_guide.md)
- [Development Guide](development.md)
- [Examples](examples.md)

## Overview

Emotion-XAI is a comprehensive package for analyzing customer feedback using transformer models, explainable AI techniques, and clustering for theme discovery.

## Quick Start

```python
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import train_baseline

# Load and prepare data
data = load_dataset("data/raw/feedback.csv")
features = prepare_features(data, "text_column")

# Train baseline model
model = train_baseline(features, labels)
```

## Features

- **Multi-label emotion classification** using fine-tuned transformer models
- **Explainable AI** with SHAP and LIME explanations
- **Theme discovery** through clustering analysis
- **Interactive web interface** with Gradio
- **Comprehensive evaluation** metrics and visualizations