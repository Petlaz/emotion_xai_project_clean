"""Emotion-Aware Customer Feedback Analysis with Explainable AI.

A comprehensive package for analyzing customer feedback using transformer models,
explainable AI techniques, and clustering for theme discovery.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from emotion_xai.data import preprocessing
from emotion_xai.models import baseline, transformer
from emotion_xai.explainability import explanations
from emotion_xai.clustering import feedback_clustering

__all__ = [
    "preprocessing",
    "baseline",
    "transformer", 
    "explanations",
    "feedback_clustering",
]