<<<<<<< HEAD
"""
Comprehensive theme discovery pipeline using UMAP + HDBSCAN clustering.

This module provides advanced clustering capabilities for emotion and theme analysis
using dimensionality reduction and density-based clustering.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import pickle
import warnings

# Clustering libraries
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class ClusteringConfig:
    """Configuration for UMAP + HDBSCAN clustering pipeline."""
    
    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_n_components: int = 2
    umap_metric: str = 'cosine'
    umap_random_state: int = 42
    
    # HDBSCAN parameters
    hdbscan_min_cluster_size: int = 15
    hdbscan_min_samples: int = 5
    hdbscan_cluster_selection_epsilon: float = 0.0
    hdbscan_metric: str = 'euclidean'
    
    # Processing parameters
    standardize_features: bool = True
    noise_threshold: float = 0.1

class ThemeClusteringPipeline:
    """Advanced clustering pipeline for theme discovery."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """
        Initialize the clustering pipeline.
        
        Args:
            config: Clustering configuration parameters
        """
        self.config = config or ClusteringConfig()
        self.scaler = StandardScaler()
        
        # Models
        self.umap_model = None
        self.hdbscan_model = None
        
        # Results
        self.embeddings_2d = None
        self.cluster_labels = None
        self.cluster_probabilities = None
        
        logger.info("✅ Theme clustering pipeline initialized")
    
    def fit_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Fit UMAP dimensionality reduction.
        
        Args:
            embeddings: High-dimensional embeddings
            
        Returns:
            2D UMAP embeddings
        """
        logger.info(f"Fitting UMAP on {embeddings.shape} embeddings")
        
        # Initialize UMAP
        self.umap_model = umap.UMAP(
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            n_components=self.config.umap_n_components,
            metric=self.config.umap_metric,
            random_state=self.config.umap_random_state,
            verbose=True
        )
        
        # Fit and transform
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.embeddings_2d = self.umap_model.fit_transform(embeddings)
        
        logger.info(f"✅ UMAP completed: {self.embeddings_2d.shape}")
        return self.embeddings_2d
    
    def fit_hdbscan(self, embeddings_2d: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit HDBSCAN clustering on reduced embeddings.
        
        Args:
            embeddings_2d: 2D embeddings (uses self.embeddings_2d if None)
            
        Returns:
            Cluster labels
        """
        if embeddings_2d is None:
            embeddings_2d = self.embeddings_2d
            
        if embeddings_2d is None:
            raise ValueError("No 2D embeddings available. Run fit_umap first.")
        
        logger.info(f"Fitting HDBSCAN on {embeddings_2d.shape} embeddings")
        
        # Initialize HDBSCAN
        self.hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            min_samples=self.config.hdbscan_min_samples,
            cluster_selection_epsilon=self.config.hdbscan_cluster_selection_epsilon,
            metric=self.config.hdbscan_metric,
            prediction_data=True  # Enable prediction for new points
        )
        
        # Fit clustering
        self.cluster_labels = self.hdbscan_model.fit_predict(embeddings_2d)
        self.cluster_probabilities = self.hdbscan_model.probabilities_
        
        # Report results
        n_clusters = len(set(self.cluster_labels)) - (1 if -1 in self.cluster_labels else 0)
        n_noise = list(self.cluster_labels).count(-1)
        
        logger.info(f"✅ HDBSCAN completed: {n_clusters} clusters, {n_noise} noise points")
        return self.cluster_labels
    
    def fit(self, embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Complete clustering pipeline: UMAP + HDBSCAN.
        
        Args:
            embeddings: High-dimensional semantic embeddings
            
        Returns:
            Tuple of (2D embeddings, cluster labels)
        """
        logger.info("🚀 Starting complete clustering pipeline")
        
        # Standardize if configured
        if self.config.standardize_features:
            logger.info("Standardizing features")
            embeddings = self.scaler.fit_transform(embeddings)
        
        # Step 1: UMAP dimensionality reduction
        embeddings_2d = self.fit_umap(embeddings)
        
        # Step 2: HDBSCAN clustering
        cluster_labels = self.fit_hdbscan(embeddings_2d)
        
        logger.info("🎉 Clustering pipeline completed successfully")
        return embeddings_2d, cluster_labels
    
    def get_cluster_summary(self) -> Dict:
        """
        Get comprehensive cluster analysis summary.
        
        Returns:
            Dictionary with clustering statistics
        """
        if self.cluster_labels is None:
            raise ValueError("No clustering results available. Run fit() first.")
        
        # Basic statistics
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = counts[unique_labels == -1][0] if -1 in unique_labels else 0
        
        # Quality metrics
        non_noise_mask = self.cluster_labels != -1
        if np.sum(non_noise_mask) > 1 and n_clusters > 1:
            silhouette_avg = silhouette_score(
                self.embeddings_2d[non_noise_mask], 
                self.cluster_labels[non_noise_mask]
            )
            calinski_harabasz = calinski_harabasz_score(
                self.embeddings_2d[non_noise_mask], 
                self.cluster_labels[non_noise_mask]
            )
        else:
            silhouette_avg = 0.0
            calinski_harabasz = 0.0
        
        # Cluster sizes
        cluster_sizes = {}
        for label, count in zip(unique_labels, counts):
            if label != -1:  # Exclude noise
                cluster_sizes[f"cluster_{label}"] = int(count)
        
        return {
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_percentage': float(n_noise / len(self.cluster_labels) * 100),
            'silhouette_score': float(silhouette_avg),
            'calinski_harabasz_score': float(calinski_harabasz),
            'cluster_sizes': cluster_sizes,
            'total_points': len(self.cluster_labels)
        }
    
    def predict_new_points(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Predict clusters for new data points.
        
        Args:
            new_embeddings: New high-dimensional embeddings
            
        Returns:
            Predicted cluster labels
        """
        if self.umap_model is None or self.hdbscan_model is None:
            raise ValueError("Models not fitted. Run fit() first.")
        
        # Transform to 2D space
        if self.config.standardize_features:
            new_embeddings = self.scaler.transform(new_embeddings)
        
        new_embeddings_2d = self.umap_model.transform(new_embeddings)
        
        # Predict clusters
        cluster_labels, strengths = hdbscan.approximate_predict(
            self.hdbscan_model, new_embeddings_2d
        )
        
        return cluster_labels
    
    def save_model(self, filepath: Union[str, Path]) -> None:
        """
        Save the fitted clustering models.
        
        Args:
            filepath: Output file path
        """
        model_data = {
            'config': self.config,
            'umap_model': self.umap_model,
            'hdbscan_model': self.hdbscan_model,
            'scaler': self.scaler,
            'embeddings_2d': self.embeddings_2d,
            'cluster_labels': self.cluster_labels,
            'cluster_probabilities': self.cluster_probabilities,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"✅ Model saved to {filepath}")
    
    def load_model(self, filepath: Union[str, Path]) -> None:
        """
        Load previously saved clustering models.
        
        Args:
            filepath: Model file path
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.config = model_data['config']
        self.umap_model = model_data['umap_model']
        self.hdbscan_model = model_data['hdbscan_model']
        self.scaler = model_data['scaler']
        self.embeddings_2d = model_data['embeddings_2d']
        self.cluster_labels = model_data['cluster_labels']
        self.cluster_probabilities = model_data['cluster_probabilities']
        
        logger.info(f"✅ Model loaded from {filepath}")

def cluster_embeddings(
    embeddings: np.ndarray, 
    config: Optional[ClusteringConfig] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function for complete clustering pipeline.
    
    Args:
        embeddings: High-dimensional semantic embeddings
        config: Clustering configuration
        
    Returns:
        Tuple of (2D embeddings, cluster labels)
    """
    pipeline = ThemeClusteringPipeline(config)
    return pipeline.fit(embeddings)

def analyze_emotion_themes(
    embeddings: np.ndarray,
    texts: List[str],
    emotions: Optional[np.ndarray] = None,
    config: Optional[ClusteringConfig] = None
) -> Dict:
    """
    Complete emotion theme analysis pipeline.
    
    Args:
        embeddings: Semantic embeddings
        texts: Original text data
        emotions: Emotion labels/probabilities
        config: Clustering configuration
        
    Returns:
        Comprehensive analysis results
    """
    logger.info(f"🎭 Starting emotion theme analysis for {len(texts)} samples")
    
    # Clustering
    pipeline = ThemeClusteringPipeline(config)
    embeddings_2d, cluster_labels = pipeline.fit(embeddings)
    
    # Compile results
    results = {
        'embeddings_2d': embeddings_2d,
        'cluster_labels': cluster_labels,
        'texts': texts,
        'cluster_summary': pipeline.get_cluster_summary(),
        'pipeline': pipeline
    }
    
    # Add emotion analysis if provided
    if emotions is not None:
        results['emotions'] = emotions
        results['emotion_clusters'] = _analyze_emotion_clusters(
            cluster_labels, emotions, texts
        )
    
    logger.info("🎉 Emotion theme analysis completed")
    return results

def _analyze_emotion_clusters(
    cluster_labels: np.ndarray,
    emotions: np.ndarray,
    texts: List[str]
) -> Dict:
    """Analyze emotion distributions within clusters."""
    cluster_emotion_stats = {}
    
    unique_clusters = np.unique(cluster_labels)
    for cluster_id in unique_clusters:
        if cluster_id == -1:  # Skip noise
            continue
            
        cluster_mask = cluster_labels == cluster_id
        cluster_emotions = emotions[cluster_mask]
        cluster_texts = [texts[i] for i in np.where(cluster_mask)[0]]
        
        # Calculate emotion statistics
        if len(cluster_emotions.shape) > 1:  # Multi-label emotions
            emotion_means = np.mean(cluster_emotions, axis=0)
            top_emotions = np.argsort(emotion_means)[-3:][::-1]  # Top 3
        else:  # Single emotion labels
            emotion_means = np.bincount(cluster_emotions.astype(int))
            top_emotions = np.argsort(emotion_means)[-3:][::-1]
        
        cluster_emotion_stats[f"cluster_{cluster_id}"] = {
            'size': int(np.sum(cluster_mask)),
            'top_emotions': [int(x) for x in top_emotions],
            'emotion_distribution': emotion_means.tolist(),
            'sample_texts': cluster_texts[:5]  # First 5 examples
        }
    
    return cluster_emotion_stats

if __name__ == "__main__":
    # Example usage
    logger.info("🧪 Testing clustering pipeline")
    
    # Prepare sample data
    np.random.seed(42)
    sample_embeddings = np.random.randn(1000, 384)  # Simulate sentence embeddings
    
    # Test clustering
    pipeline = ThemeClusteringPipeline()
    embeddings_2d, cluster_labels = pipeline.fit(sample_embeddings)
    
    # Print results
    summary = pipeline.get_cluster_summary()
    print(f"✅ Clustering test completed:")
    print(f"   Clusters found: {summary['n_clusters']}")
    print(f"   Noise points: {summary['n_noise_points']}")
    print(f"   Silhouette score: {summary['silhouette_score']:.3f}")
    
    logger.info("🎉 Clustering pipeline test completed successfully")
=======
"""Theme discovery utilities using sentence embeddings + clustering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class ClusteringConfig:
    """Configuration for UMAP + HDBSCAN clustering."""

    n_neighbors: int = 15
    min_dist: float = 0.1
    min_cluster_size: int = 10


def embed_sentences(texts: Iterable[str]) -> np.ndarray:
    """Placeholder embedding function."""
    # TODO: Replace with sentence-transformers embedding generation.
    text_list = list(texts)
    return np.zeros((len(text_list), 768))


def cluster_embeddings(embeddings: np.ndarray, config: ClusteringConfig) -> np.ndarray:
    """Placeholder clustering function."""
    # TODO: Integrate UMAP + HDBSCAN pipeline here.
    return np.zeros(len(embeddings), dtype=int)
>>>>>>> be3b044594b375f6fcd55554c1c72425f0629c88
