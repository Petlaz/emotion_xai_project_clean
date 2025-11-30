"""
Semantic embedding extraction for text clustering and theme discovery.

This module provides utilities for extracting semantic embeddings from text data
using sentence-transformers, optimized for emotion and theme analysis.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from datetime import datetime
import pickle
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import warnings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemanticEmbeddingGenerator:
    """Extract semantic embeddings for clustering analysis."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the semantic embedding system.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to use ('cuda', 'mps', 'cpu'). Auto-detected if None
            cache_dir: Directory to cache embeddings
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.cache_dir = Path(cache_dir) if cache_dir else Path("models/cluster_embeddings")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        logger.info(f"Loading sentence transformer: {model_name}")
        self.model = SentenceTransformer(model_name, device=self.device)
        
        # Embedding cache
        self.embedding_cache = {}
        self.scaler = StandardScaler()
        
        logger.info(f"✅ Embedding system initialized on {self.device}")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine the best available device."""
        if device:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    
    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Extract semantic embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Show progress bar
            
        Returns:
            numpy array of embeddings (n_texts, embedding_dim)
        """
        logger.info(f"Processing embeddings for {len(texts)} texts")
        
        # Extract embeddings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
        
        logger.info(f"✅ Extracted embeddings: {embeddings.shape}")
        return embeddings
    
    def process_emotion_data(
        self,
        data_path: Union[str, Path],
        text_column: str = "text",
        emotion_columns: Optional[List[str]] = None,
        sample_size: Optional[int] = None,
        cache_key: Optional[str] = None
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Process emotion dataset and extract embeddings.
        
        Args:
            data_path: Path to the dataset
            text_column: Name of the text column
            emotion_columns: List of emotion column names
            sample_size: Limit to N samples for testing
            cache_key: Cache key for embeddings
            
        Returns:
            Tuple of (embeddings, dataframe)
        """
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        if sample_size and len(df) > sample_size:
            logger.info(f"Sampling {sample_size} rows from {len(df)} total")
            df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
        
        # Check cache
        if cache_key:
            cached_path = self.cache_dir / f"{cache_key}_embeddings.pkl"
            if cached_path.exists():
                logger.info(f"Loading cached embeddings from {cached_path}")
                with open(cached_path, 'rb') as f:
                    embeddings = pickle.load(f)
                return embeddings, df
        
        # Extract embeddings
        texts = df[text_column].astype(str).tolist()
        embeddings = self.generate_embeddings(texts)
        
        # Cache results
        if cache_key:
            cached_path = self.cache_dir / f"{cache_key}_embeddings.pkl"
            logger.info(f"Caching embeddings to {cached_path}")
            with open(cached_path, 'wb') as f:
                pickle.dump(embeddings, f)
        
        return embeddings, df
    
    def prepare_clustering_data(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        emotion_columns: Optional[List[str]] = None,
        standardize: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Prepare data for clustering analysis.
        
        Args:
            embeddings: Semantic embeddings
            df: Original dataframe
            emotion_columns: Emotion label columns
            standardize: Whether to standardize embeddings
            
        Returns:
            Dictionary with prepared data arrays
        """
        logger.info("Preparing clustering data")
        
        # Standardize embeddings if requested
        if standardize:
            embeddings = self.scaler.fit_transform(embeddings)
        
        result = {
            'embeddings': embeddings,
            'texts': df['text'].values if 'text' in df.columns else None
        }
        
        # Add emotion labels if available
        if emotion_columns:
            available_emotions = [col for col in emotion_columns if col in df.columns]
            if available_emotions:
                emotion_matrix = df[available_emotions].values
                result['emotions'] = emotion_matrix
                result['emotion_names'] = available_emotions
                
                # Calculate dominant emotions
                dominant_emotions = df[available_emotions].idxmax(axis=1).values
                result['dominant_emotions'] = dominant_emotions
                
                logger.info(f"Added {len(available_emotions)} emotion features")
        
        logger.info(f"✅ Clustering data prepared: {embeddings.shape}")
        return result
    
    def save_embeddings(
        self,
        embeddings: np.ndarray,
        df: pd.DataFrame,
        filename: str
    ) -> Path:
        """
        Save embeddings and associated data.
        
        Args:
            embeddings: Extracted embeddings
            df: Original dataframe
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.cache_dir / f"{filename}_{timestamp}.pkl"
        
        data = {
            'embeddings': embeddings,
            'dataframe': df,
            'model_name': self.model_name,
            'timestamp': timestamp,
            'shape': embeddings.shape
        }
        
        logger.info(f"Saving embeddings to {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
            
        return output_path
    
    def load_embeddings(self, filepath: Union[str, Path]) -> Dict:
        """
        Load previously saved embeddings.
        
        Args:
            filepath: Path to embedding file
            
        Returns:
            Dictionary with embeddings and metadata
        """
        logger.info(f"Loading embeddings from {filepath}")
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        logger.info(f"✅ Loaded embeddings: {data['shape']}")
        return data

def extract_emotion_embeddings(
    data_path: str,
    sample_size: int = 10000,
    model_name: str = "all-MiniLM-L6-v2"
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Convenience function to extract embeddings from emotion data.
    
    Args:
        data_path: Path to emotion dataset
        sample_size: Number of samples to process
        model_name: Sentence transformer model
        
    Returns:
        Tuple of (embeddings, dataframe)
    """
    generator = SemanticEmbeddingGenerator(model_name=model_name)
    
    # Standard GoEmotions emotion columns
    emotion_labels = [
        'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
        'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
        'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
        'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
        'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]
    
    cache_key = f"goemotions_{sample_size}_{model_name.replace('/', '_')}"
    
    embeddings, df = generator.process_emotion_data(
        data_path=data_path,
        emotion_columns=emotion_labels,
        sample_size=sample_size,
        cache_key=cache_key
    )
    
    return embeddings, df

if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent.parent
    sys.path.append(str(project_root))
    
    # Test embedding extraction
    generator = SemanticEmbeddingGenerator()
    
    # Test with sample texts
    sample_texts = [
        "I love this amazing product!",
        "This service is terrible and disappointing.",
        "The quality is decent but could be better.",
        "Outstanding customer service experience!",
        "I'm confused about how this works."
    ]
    
    embeddings = generator.generate_embeddings(sample_texts)
    print(f"Extracted embeddings shape: {embeddings.shape}")
    print("✅ Embedding extraction test completed successfully!")