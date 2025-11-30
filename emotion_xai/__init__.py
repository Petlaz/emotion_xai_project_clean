"""Emotion-Aware Customer Feedback Analysis with Explainable AI.

A comprehensive package for analyzing customer feedback using transformer models,
explainable AI techniques, and clustering for theme discovery.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import modules with error handling
try:
    from emotion_xai.data import preprocessing
except ImportError as e:
    preprocessing = None

try:
    from emotion_xai.models import baseline
except ImportError as e:
    baseline = None

try:
    from emotion_xai.models import transformer
except ImportError as e:
    transformer = None

try:
    from emotion_xai.explainability import explanations
except ImportError as e:
    explanations = None

try:
    from emotion_xai.clustering import feedback_clustering
except ImportError as e:
    feedback_clustering = None

__all__ = [
    "preprocessing",
    "baseline",
    "transformer", 
    "explanations",
    "feedback_clustering",
]