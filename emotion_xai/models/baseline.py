"""
Baseline model implementation for emotion classification.

This module implements a TF-IDF + Logistic Regression baseline model
for multi-label emotion classification using the GoEmotions dataset.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import joblib
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings

from emotion_xai.data.preprocessing import (
    load_dataset, prepare_features, prepare_emotion_labels,
    filter_quality_issues, split_dataset, EMOTION_COLUMNS
)


class BaselineModel:
    """
    TF-IDF + Logistic Regression baseline model for emotion classification.
    
    This model serves as a baseline for comparing more advanced approaches.
    It uses TF-IDF vectorization followed by One-vs-Rest Logistic Regression
    for multi-label emotion classification.
    """
    
    def __init__(self, 
                 max_features: int = 10000,
                 ngram_range: Tuple[int, int] = (1, 2),
                 C: float = 1.0,
                 max_iter: int = 1000,
                 random_state: int = 42):
        """
        Initialize the baseline model.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: Range of n-grams to extract
            C: Regularization strength for logistic regression
            max_iter: Maximum iterations for training
            random_state: Random state for reproducibility
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize components
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'(?u)\b\w\w+\b'  # Only words with 2+ characters
        )
        
        self.classifier = OneVsRestClassifier(
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                random_state=random_state,
                solver='liblinear'  # Good for small datasets
            ),
            n_jobs=-1  # Use all available cores
        )
        
        # Store model information
        self.is_fitted = False
        self.emotion_labels = None
        self.feature_names = None
        self.training_info = {}
        
    def fit(self, texts: List[str], labels: np.ndarray, 
            emotion_names: List[str] = None) -> 'BaselineModel':
        """
        Train the baseline model on the provided data.
        
        Args:
            texts: List of text samples
            labels: Multi-label binary array of shape (n_samples, n_emotions)
            emotion_names: Names of emotion labels
            
        Returns:
            Self for method chaining
        """
        print(f"🚀 Training baseline model...")
        print(f"   📊 Training samples: {len(texts)}")
        print(f"   🏷️  Emotions: {labels.shape[1]}")
        
        start_time = datetime.now()
        
        # Store emotion information
        self.emotion_labels = emotion_names or [f"emotion_{i}" for i in range(labels.shape[1])]
        
        # Vectorize texts
        print("   🔤 Vectorizing texts with TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        print(f"   📈 TF-IDF shape: {X_tfidf.shape}")
        print(f"   📝 Vocabulary size: {len(self.feature_names)}")
        
        # Train classifier
        print("   🎯 Training multi-label classifier...")
        self.classifier.fit(X_tfidf, labels)
        
        # Record training information
        training_time = (datetime.now() - start_time).total_seconds()
        self.training_info = {
            'training_time_seconds': training_time,
            'training_samples': len(texts),
            'n_emotions': labels.shape[1],
            'n_features': X_tfidf.shape[1],
            'vocabulary_size': len(self.feature_names),
            'hyperparameters': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'C': self.C,
                'max_iter': self.max_iter
            }
        }
        
        self.is_fitted = True
        
        print(f"   ✅ Training completed in {training_time:.1f} seconds")
        return self
    
    def predict(self, texts: List[str]) -> np.ndarray:
        """
        Predict emotion labels for given texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Binary predictions of shape (n_samples, n_emotions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_tfidf = self.vectorizer.transform(texts)
        return self.classifier.predict(X_tfidf)
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Predict emotion probabilities for given texts.
        
        Args:
            texts: List of text samples
            
        Returns:
            Probability predictions of shape (n_samples, n_emotions)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        X_tfidf = self.vectorizer.transform(texts)
        
        # OneVsRestClassifier returns decision function, convert to probabilities
        decision_scores = self.classifier.decision_function(X_tfidf)
        
        # Apply sigmoid to convert to probabilities
        probabilities = 1 / (1 + np.exp(-decision_scores))
        
        return probabilities
    
    def evaluate(self, texts: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the model on provided data.
        
        Args:
            texts: List of text samples
            labels: True multi-label binary array
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before evaluation")
        
        print("📊 Evaluating model performance...")
        
        # Get predictions
        predictions = self.predict(texts)
        probabilities = self.predict_proba(texts)
        
        # Calculate metrics
        metrics = {}
        
        # Overall metrics (micro and macro averaging)
        metrics['accuracy'] = accuracy_score(labels, predictions)
        metrics['f1_micro'] = f1_score(labels, predictions, average='micro')
        metrics['f1_macro'] = f1_score(labels, predictions, average='macro')
        metrics['f1_weighted'] = f1_score(labels, predictions, average='weighted')
        metrics['precision_micro'] = precision_score(labels, predictions, average='micro')
        metrics['precision_macro'] = precision_score(labels, predictions, average='macro')
        metrics['recall_micro'] = recall_score(labels, predictions, average='micro')
        metrics['recall_macro'] = recall_score(labels, predictions, average='macro')
        
        # Per-emotion metrics
        per_emotion_f1 = f1_score(labels, predictions, average=None)
        per_emotion_precision = precision_score(labels, predictions, average=None)
        per_emotion_recall = recall_score(labels, predictions, average=None)
        
        metrics['per_emotion'] = {}
        for i, emotion in enumerate(self.emotion_labels):
            metrics['per_emotion'][emotion] = {
                'f1': float(per_emotion_f1[i]),
                'precision': float(per_emotion_precision[i]),
                'recall': float(per_emotion_recall[i])
            }
        
        # Multi-label specific metrics
        n_samples = len(texts)
        n_predicted = np.sum(predictions)
        n_actual = np.sum(labels)
        
        metrics['multilabel_stats'] = {
            'avg_labels_per_sample': float(np.mean(np.sum(labels, axis=1))),
            'avg_predictions_per_sample': float(np.mean(np.sum(predictions, axis=1))),
            'total_predicted_labels': int(n_predicted),
            'total_actual_labels': int(n_actual)
        }
        
        # Sample statistics
        metrics['sample_stats'] = {
            'n_samples': n_samples,
            'n_emotions': len(self.emotion_labels)
        }
        
        print(f"   🎯 Overall Accuracy: {metrics['accuracy']:.3f}")
        print(f"   📊 F1-Score (macro): {metrics['f1_macro']:.3f}")
        print(f"   📊 F1-Score (micro): {metrics['f1_micro']:.3f}")
        print(f"   🏷️  Avg labels per sample: {metrics['multilabel_stats']['avg_labels_per_sample']:.2f}")
        
        return metrics
    
    def get_top_features(self, emotion_idx: int, n_features: int = 20) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a specific emotion.
        
        Args:
            emotion_idx: Index of the emotion
            n_features: Number of top features to return
            
        Returns:
            List of (feature, coefficient) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting features")
        
        # Get coefficients for the specific emotion classifier
        coef = self.classifier.estimators_[emotion_idx].coef_[0]
        
        # Get top positive features
        top_indices = np.argsort(coef)[-n_features:][::-1]
        top_features = [(self.feature_names[i], coef[i]) for i in top_indices]
        
        return top_features
    
    def save(self, filepath: Path) -> Path:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
            
        Returns:
            Path where model was saved
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model components
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'emotion_labels': self.emotion_labels,
            'feature_names': self.feature_names,
            'training_info': self.training_info,
            'hyperparameters': {
                'max_features': self.max_features,
                'ngram_range': self.ngram_range,
                'C': self.C,
                'max_iter': self.max_iter,
                'random_state': self.random_state
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Model saved to: {filepath}")
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'BaselineModel':
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded BaselineModel instance
        """
        model_data = joblib.load(filepath)
        
        # Create instance with saved hyperparameters
        hyperparams = model_data['hyperparameters']
        instance = cls(
            max_features=hyperparams['max_features'],
            ngram_range=tuple(hyperparams['ngram_range']),
            C=hyperparams['C'],
            max_iter=hyperparams['max_iter'],
            random_state=hyperparams['random_state']
        )
        
        # Restore model components
        instance.vectorizer = model_data['vectorizer']
        instance.classifier = model_data['classifier']
        instance.emotion_labels = model_data['emotion_labels']
        instance.feature_names = model_data['feature_names']
        instance.training_info = model_data['training_info']
        instance.is_fitted = True
        
        print(f"📂 Model loaded from: {filepath}")
        
        return instance


def save_evaluation_results(metrics: Dict[str, Any], 
                          model_info: Dict[str, Any],
                          results_dir: Path) -> Path:
    """
    Save evaluation results to the results directory.
    
    Args:
        metrics: Evaluation metrics dictionary
        model_info: Model training information
        results_dir: Results directory path
        
    Returns:
        Path to saved results file
    """
    # Create results directory
    metrics_dir = results_dir / 'metrics' / 'model_performance'
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # Create comprehensive results
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'baseline_tfidf_logistic_regression',
        'model_info': model_info,
        'evaluation_metrics': metrics,
        'summary': {
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_micro': metrics['f1_micro'],
            'training_time': model_info.get('training_time_seconds', 0),
            'n_samples': metrics['sample_stats']['n_samples'],
            'n_emotions': metrics['sample_stats']['n_emotions']
        }
    }
    
    # Save with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = metrics_dir / f'baseline_evaluation_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Evaluation results saved to: {results_file}")
    
    return results_file


if __name__ == "__main__":
    print("🧪 Running baseline model unit tests...")
    
    # Create sample data for testing
    sample_texts = [
        "I am so happy today!",
        "This makes me really sad and disappointed.",
        "I'm feeling quite angry about this situation.",
        "What an amazing and joyful experience!",
        "I'm scared and worried about the future."
    ]
    
    # Create sample multi-label data
    sample_labels = np.array([
        [1, 0, 0],  # joy
        [0, 1, 0],  # sadness
        [0, 0, 1],  # anger
        [1, 0, 0],  # joy
        [0, 1, 1]   # sadness + anger
    ])
    
    emotion_names = ['joy', 'sadness', 'anger']
    
    print(f"📝 Sample data: {len(sample_texts)} texts, {len(emotion_names)} emotions")
    
    # Test model training
    model = BaselineModel(max_features=1000, random_state=42)
    model.fit(sample_texts, sample_labels, emotion_names)
    
    # Test prediction
    test_texts = ["I feel great!", "This is terrible"]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    print(f"🔮 Predictions shape: {predictions.shape}")
    print(f"🎲 Probabilities shape: {probabilities.shape}")
    
    # Test evaluation
    metrics = model.evaluate(sample_texts, sample_labels)
    
    print(f"\n✅ Baseline model tests completed!")
    print(f"   🎯 Test accuracy: {metrics['accuracy']:.3f}")
    print(f"   📊 Test F1 (macro): {metrics['f1_macro']:.3f}")