<<<<<<< HEAD
"""
Explainability utilities for emotion classification models.

This module provides comprehensive explainability tools for transformer-based
emotion classification models using SHAP, LIME, and attention visualization.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForSequenceClassification
=======
"""Explainability utilities for the emotion classification model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88


@dataclass
class ExplanationResult:
    """Container for explanation artifacts."""
<<<<<<< HEAD
    
    text: str
    predictions: Dict[str, float]
    shap_values: Optional[np.ndarray] = None
    lime_explanation: Optional[Any] = None
    attention_weights: Optional[Dict[str, np.ndarray]] = None
    tokens: Optional[List[str]] = None
    feature_importance: Optional[Dict[str, float]] = None


class ExplainerFactory:
    """Factory class to create appropriate explainers for different model types."""
    
    @staticmethod
    def create_explainer(model_type: str, model_path: str, **kwargs):
        """Create an explainer instance based on model type."""
        if model_type.lower() == 'transformer':
            return TransformerExplainer(model_path, **kwargs)
        elif model_type.lower() == 'baseline':
            return BaselineExplainer(model_path, **kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class SHAPExplainer:
    """SHAP-based explainer for emotion classification models."""
    
    def __init__(self, model_path: str, emotion_labels: Optional[List[str]] = None):
        """
        Initialize SHAP explainer.
        
        Args:
            model_path: Path to the trained transformer model
            emotion_labels: List of emotion labels (will auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Set up emotion labels
        if emotion_labels is None:
            # GoEmotions emotion labels
            self.emotion_labels = [
                'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
            ]
        else:
            self.emotion_labels = emotion_labels
        
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Please install with: pip install shap")
    
    def _model_predict(self, texts: List[str]) -> np.ndarray:
        """Prediction function for SHAP."""
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for text in texts:
                # Tokenize input
                inputs = self.tokenizer(
                    text, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=512
                ).to(self.device)
                
                # Get predictions
                outputs = self.model(**inputs)
                probs = torch.sigmoid(outputs.logits).cpu().numpy()
                predictions.append(probs[0])
        
        return np.array(predictions)
    
    def explain_text(self, text: str, num_samples: int = 100) -> ExplanationResult:
        """
        Create SHAP explanations for a single text.
        
        Args:
            text: Input text to explain
            num_samples: Number of samples for SHAP estimation
        
        Returns:
            ExplanationResult with SHAP values and predictions
        """
        # Get model predictions
        predictions = self._model_predict([text])[0]
        pred_dict = {label: float(pred) for label, pred in zip(self.emotion_labels, predictions)}
        # Create SHAP explainer
        explainer = shap.Explainer(self._model_predict, self.tokenizer)
        # Compute SHAP values
        shap_values = explainer([text])
        # Extract tokens
        tokens = self.tokenizer.tokenize(text)
        # Calculate feature importance
        feature_importance = {}
        if hasattr(shap_values, 'values') and len(shap_values.values) > 0:
            # Average SHAP values across all emotions
            avg_shap = np.mean(np.abs(shap_values.values[0]), axis=-1)
            if len(tokens) == len(avg_shap):
                feature_importance = {token: float(importance) 
                                    for token, importance in zip(tokens, avg_shap)}
        return ExplanationResult(
            text=text,
            predictions=pred_dict,
            shap_values=shap_values.values[0] if hasattr(shap_values, 'values') else None,
            tokens=tokens,
            feature_importance=feature_importance
        )


class TransformerExplainer:
    """Comprehensive explainer for transformer-based emotion classification."""
    
    def __init__(self, model_path: str, emotion_labels: Optional[List[str]] = None):
        """
        Initialize transformer explainer.
        
        Args:
            model_path: Path to the trained transformer model
            emotion_labels: List of emotion labels (will auto-detect if None)
        """
        self.model_path = Path(model_path)
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_path))
        self.model.to(self.device)
        self.model.eval()
        
        # Set up emotion labels
        if emotion_labels is None:
            # GoEmotions emotion labels
            self.emotion_labels = [
                'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                'relief', 'remorse', 'sadness', 'surprise', 'neutral'
            ]
        else:
            self.emotion_labels = emotion_labels
    
    def predict(self, text: str) -> Dict[str, float]:
        """Get model predictions for a text."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.sigmoid(outputs.logits).cpu().numpy()[0]
        
        return {label: float(prob) for label, prob in zip(self.emotion_labels, probs)}
    
    def get_attention_weights(self, text: str) -> Dict[str, np.ndarray]:
        """Extract attention weights from the model."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        attention_weights = {}
        for layer_idx, attention in enumerate(outputs.attentions):
            # Average across heads: (batch, heads, seq, seq) -> (seq, seq)  
            avg_attention = attention[0].mean(dim=0).cpu().numpy()
            attention_weights[f'layer_{layer_idx}'] = avg_attention
            
        return attention_weights
    
    def explain_with_shap(self, text: str, num_samples: int = 100) -> ExplanationResult:
        """Create SHAP explanations."""
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")
        shap_explainer = SHAPExplainer(self.model_path, self.emotion_labels)
        return shap_explainer.explain_text(text, num_samples)
    
    def get_top_emotions(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top-k predicted emotions with confidence scores."""
        predictions = self.predict(text)
        sorted_emotions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        return sorted_emotions[:top_k]


class BaselineExplainer:
    """Explainer for baseline TF-IDF models."""
    
    def __init__(self, model_path: str):
        """Initialize baseline explainer."""
        self.model_path = model_path
        # Implementation for baseline model explanations
        # This would load the saved baseline model artifacts
        pass
    
    def explain(self, text: str) -> ExplanationResult:
        """Create explanations for baseline model."""
        # Implementation for baseline model explanations
        pass


def explain_with_shap(model_path: str, text: str, **kwargs) -> ExplanationResult:
    """
    Convenience function to create SHAP explanations.
    
    Args:
        model_path: Path to the trained model
        text: Text to explain
        **kwargs: Additional arguments
    
    Returns:
        ExplanationResult with SHAP explanations
    """
    explainer = TransformerExplainer(model_path)
    return explainer.explain_with_shap(text, **kwargs)


def explain_with_lime(model_path: str, text: str, **kwargs) -> ExplanationResult:
    """
    Convenience function to create LIME explanations.
    
    Args:
        model_path: Path to the trained model  
        text: Text to explain
        **kwargs: Additional arguments
    
    Returns:
        ExplanationResult with LIME explanations
    """
    # This will be implemented in lime_utils.py
    from .lime_utils import LIMEExplainer
    explainer = LIMEExplainer(model_path)
    return explainer.explain_text(text, **kwargs)
=======

    raw_output: Any
    shap_values: Any | None = None
    lime_explanation: Any | None = None


def explain_with_shap(model: Any, inputs: np.ndarray) -> ExplanationResult:
    """Placeholder SHAP explanation stub."""
    # TODO: Integrate SHAP once model artifacts are available.
    return ExplanationResult(raw_output=model.predict(inputs), shap_values=None)


def explain_with_lime(model: Any, inputs: np.ndarray) -> ExplanationResult:
    """Placeholder LIME explanation stub."""
    # TODO: Integrate LIME text explainer.
    return ExplanationResult(raw_output=model.predict(inputs), lime_explanation=None)
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
