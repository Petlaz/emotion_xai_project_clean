# 📚 Emotion-XAI Documentation

**Complete technical documentation for the Emotion-XAI project**

## 📋 Documentation Index

| Document | Description | Target Audience |
|----------|-------------|-----------------|
| [🚀 Getting Started](getting_started.md) | Quick setup and first steps | New users, developers |
| [💻 Development Guide](development.md) | Technical setup and contribution | Contributors, maintainers |
| [📊 Project Plan](project_plan.md) | Complete project roadmap | Project managers, stakeholders |
| [📈 Documentation Report](documentation_report.md) | Comprehensive project completion | Technical reviewers |
| [🍎 Mac Optimization](mac_optimization.md) | macOS-specific optimizations | Mac developers |

## 🎯 **Project Overview**

Emotion-XAI is a **production-ready explainable AI system** for multi-label emotion detection in social media text. Built with state-of-the-art transformer models and comprehensive explainability features.

### 🏆 **Key Achievements**
- **F1-Macro**: 0.196 (19.6% accuracy on 28-emotion classification)
- **Model**: Fine-tuned DistilRoBERTa (82M parameters)
- **Dataset**: GoEmotions (211,225 Reddit comments)
- **Deployment**: Live on Hugging Face Spaces

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