---
title: Emotion-XAI
emoji: 🎭
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
---

# 🎭 Emotion-XAI: Explainable AI for Emotion Detection

**Analyze emotions in text with state-of-the-art AI and explainable insights**

## 🚀 Features

- **Multi-label Emotion Classification**: Detect multiple emotions simultaneously using fine-tuned DistilRoBERTa
- **Explainable AI**: Understand model predictions with SHAP and LIME explanations
- **Real-time Analysis**: Instant emotion detection with interactive visualizations
- **Production Ready**: Based on F1-macro 0.196 performance (19.6% accuracy)
- **Batch Processing**: Analyze multiple texts at once

## 📊 Model Performance

- **Dataset**: GoEmotions (211K Reddit comments, 28 emotion categories)
- **Architecture**: DistilRoBERTa (82M parameters)
- **Performance**: F1-macro 0.196 (1.2x improvement over baseline)
- **Training**: 147K samples, optimized for efficiency

## 🎯 Supported Emotions

admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral

## 🛠️ Technical Details

- **Framework**: Gradio + Transformers + Explainable AI
- **Device Support**: Auto-detection (CUDA/MPS/CPU)
- **Explainability**: SHAP and LIME explanations
- **Visualization**: Interactive Plotly charts

## 📚 Project Phases

✅ **Phase 1-5 Complete**: Data processing, baseline modeling, transformer fine-tuning, explainable AI, clustering analysis  
🎯 **Phase 6**: Interactive web interface deployment

## 🔗 Links

- **GitHub**: [emotion_xai_project_clean](https://github.com/Petlaz/emotion_xai_project_clean)
- **License**: MIT
- **Framework**: Gradio + Transformers + Explainable AI

---

Built with ❤️ using Gradio, Transformers, SHAP, and LIME