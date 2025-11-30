"""
LIME (Local Interpretable Model-agnostic Explanations) utilities for emotion classification.

This module provides LIME-based explanations for text-based emotion classification models,
offering local interpretability through perturbation-based analysis.
"""

from __future__ import annotations

import numpy as np
import torch
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path

try:
    from lime import lime_text
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from .explanations import ExplanationResult


class LIMEExplainer:
    """LIME-based explainer for emotion classification models."""
    
    def __init__(self, model_path: str, emotion_labels: Optional[List[str]] = None):
        """
        Initialize LIME explainer.
        
        Args:
            model_path: Path to the trained transformer model
            emotion_labels: List of emotion labels (will auto-detect if None)
        """
        if not LIME_AVAILABLE:
            raise ImportError("LIME is not installed. Please install with: pip install lime")
        
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
        
        # Initialize LIME explainer
        self.lime_explainer = LimeTextExplainer(
            class_names=self.emotion_labels
        )
    
    def _prediction_function(self, texts: List[str]) -> np.ndarray:
        """
        Prediction function for LIME perturbations.
        
        Args:
            texts: List of text samples
            
        Returns:
            Prediction probabilities for all samples
        """
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
    
    def explain_text(self, 
                    text: str, 
                    num_features: int = 10,
                    num_samples: int = 1000,
                    top_emotions: Optional[List[str]] = None) -> ExplanationResult:
        """
        Create LIME explanations for a single text.
        
        Args:
            text: Input text to explain
            num_features: Number of features to include in explanation
            num_samples: Number of samples for LIME estimation
            top_emotions: List of emotions to focus on (if None, uses top predictions)
        
        Returns:
            ExplanationResult with LIME explanations
        """
        # Get model predictions
        predictions = self._prediction_function([text])[0]
        pred_dict = {label: float(pred) for label, pred in zip(self.emotion_labels, predictions)}
        
        # Determine which emotions to explain
        if top_emotions is None:
            # Get top 5 predicted emotions
            sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
            emotion_indices = [idx for idx, _ in sorted_preds[:5] if predictions[idx] > 0.1]
        else:
            emotion_indices = [self.emotion_labels.index(emotion) 
                             for emotion in top_emotions if emotion in self.emotion_labels]
        
    # Create LIME explanation
        lime_explanation = self.lime_explainer.explain_instance(
            text,
            self._prediction_function,
            num_features=num_features,
            num_samples=num_samples,
            labels=emotion_indices
        )
        
        # Extract feature importance
        feature_importance = {}
        for emotion_idx in emotion_indices:
            emotion_name = self.emotion_labels[emotion_idx]
            explanation_list = lime_explanation.as_list(label=emotion_idx)
            feature_importance[emotion_name] = {
                word: importance for word, importance in explanation_list
            }
        
        return ExplanationResult(
            text=text,
            predictions=pred_dict,
            lime_explanation=lime_explanation,
            feature_importance=feature_importance
        )
    
    def explain_top_emotions(self, text: str, top_k: int = 3, **kwargs) -> ExplanationResult:
        """
        Explain the top-k predicted emotions.
        
        Args:
            text: Input text to explain
            top_k: Number of top emotions to explain
            **kwargs: Additional arguments for explain_text
            
        Returns:
            ExplanationResult with explanations for top emotions
        """
        # Get predictions to find top emotions
        predictions = self._prediction_function([text])[0]
        sorted_preds = sorted(enumerate(predictions), key=lambda x: x[1], reverse=True)
        
        # Get top emotions above threshold
        top_emotions = []
        for idx, score in sorted_preds[:top_k]:
            if score > 0.1:  # Only explain emotions with reasonable confidence
                top_emotions.append(self.emotion_labels[idx])
        
        return self.explain_text(text, top_emotions=top_emotions, **kwargs)
    
    def get_word_importance_summary(self, explanation_result: ExplanationResult) -> Dict[str, Dict[str, float]]:
        """
        Extract word importance summary from LIME explanation.
        
        Args:
            explanation_result: Result from explain_text
            
        Returns:
            Dictionary mapping emotions to word importance scores
        """
        if explanation_result.lime_explanation is None:
            return {}
        
        summary = {}
        lime_exp = explanation_result.lime_explanation
        
        # Get available labels (emotions) in the explanation
        available_labels = lime_exp.available_labels()
        
        for label_idx in available_labels:
            emotion_name = self.emotion_labels[label_idx]
            word_scores = dict(lime_exp.as_list(label=label_idx))
            summary[emotion_name] = word_scores
        
        return summary
    
    def visualize_explanation(self, explanation_result: ExplanationResult, emotion: str) -> str:
        """
        Create HTML visualization for a specific emotion.
        
        Args:
            explanation_result: Result from explain_text
            emotion: Emotion to visualize
        
        Returns:
            HTML string for visualization
        """
        if explanation_result.lime_explanation is None:
            return "<p>No LIME explanation available</p>"
        
        if emotion not in self.emotion_labels:
            return f"<p>Emotion '{emotion}' not found in labels</p>"
        
        emotion_idx = self.emotion_labels.index(emotion)
        
        try:
            html = explanation_result.lime_explanation.as_html(label=emotion_idx)
            return html
        except Exception as e:
            return f"<p>Error creating visualization: {str(e)}</p>"


class MultiLabelLIME:
    """Enhanced LIME explainer specifically for multi-label emotion classification."""
    
    def __init__(self, model_path: str, emotion_labels: Optional[List[str]] = None):
        """Initialize multi-label LIME explainer."""
        self.base_explainer = LIMEExplainer(model_path, emotion_labels)
        self.emotion_labels = self.base_explainer.emotion_labels
    
    def explain_all_emotions(self, text: str, threshold: float = 0.05, **kwargs) -> Dict[str, ExplanationResult]:
        """
        Create explanations for all emotions above threshold.
        
        Args:
            text: Input text to explain
            threshold: Minimum prediction threshold to include explanation
            **kwargs: Additional arguments for LIME
        
        Returns:
            Dictionary mapping emotion names to their explanations
        """
        # Get predictions
        predictions = self.base_explainer._prediction_function([text])[0]
        
        # Find emotions above threshold
        significant_emotions = [
            self.emotion_labels[i] for i, score in enumerate(predictions) 
            if score > threshold
        ]
        
        explanations = {}
        
    # Create explanation for each significant emotion
        for emotion in significant_emotions:
            try:
                result = self.base_explainer.explain_text(
                    text, 
                    top_emotions=[emotion], 
                    **kwargs
                )
                explanations[emotion] = result
            except Exception as e:
                print(f"Error explaining emotion '{emotion}': {str(e)}")
                continue
        
        return explanations
    
    def compare_emotions(self, text: str, emotion_pairs: List[tuple], **kwargs) -> Dict[tuple, Dict]:
        """
        Compare explanations between pairs of emotions.
        
        Args:
            text: Input text to explain
            emotion_pairs: List of (emotion1, emotion2) tuples to compare
            **kwargs: Additional arguments for LIME
            
        Returns:
            Comparison results for each emotion pair
        """
        comparisons = {}
        
        for emotion1, emotion2 in emotion_pairs:
            if emotion1 not in self.emotion_labels or emotion2 not in self.emotion_labels:
                continue
            
            # Get explanations for both emotions
            result1 = self.base_explainer.explain_text(text, top_emotions=[emotion1], **kwargs)
            result2 = self.base_explainer.explain_text(text, top_emotions=[emotion2], **kwargs)
            
            # Extract word importance
            words1 = result1.feature_importance.get(emotion1, {})
            words2 = result2.feature_importance.get(emotion2, {})
            
            # Find common and unique words
            all_words = set(words1.keys()) | set(words2.keys())
            
            comparison = {
                'emotion1': emotion1,
                'emotion2': emotion2,
                'prediction1': result1.predictions[emotion1],
                'prediction2': result2.predictions[emotion2],
                'common_words': {
                    word: (words1.get(word, 0), words2.get(word, 0))
                    for word in all_words if word in words1 and word in words2
                },
                'unique_to_1': {word: score for word, score in words1.items() if word not in words2},
                'unique_to_2': {word: score for word, score in words2.items() if word not in words1}
            }
            
            comparisons[(emotion1, emotion2)] = comparison
        
        return comparisons


def create_lime_explainer(model_path: str, **kwargs) -> LIMEExplainer:
    """
    Factory function to create a LIME explainer.
    
    Args:
        model_path: Path to the trained model
        **kwargs: Additional arguments for LIMEExplainer
        
    Returns:
        Initialized LIMEExplainer instance
    """
    return LIMEExplainer(model_path, **kwargs)


def explain_text_with_lime(model_path: str, text: str, **kwargs) -> ExplanationResult:
    """
    Convenience function for LIME text explanation.
    
    Args:
        model_path: Path to the trained model
        text: Text to explain
        **kwargs: Additional arguments for explain_text
        
    Returns:
        ExplanationResult with LIME explanations
    """
    explainer = LIMEExplainer(model_path)
    return explainer.explain_text(text, **kwargs)