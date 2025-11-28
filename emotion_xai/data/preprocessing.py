"""
Data preprocessing utilities for the emotion-aware feedback pipeline.

This module implements comprehensive text preprocessing based on EDA findings:
- 211,225 samples with 28 emotion labels
- Average text length: 69 characters, 13 words
- 3.8% quality issues identified in EDA
- Multi-label classification (17% have multiple emotions)
"""

from __future__ import annotations

import json
import re
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


# Emotion columns based on GoEmotions dataset
EMOTION_COLUMNS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]


class DataQualityMetrics:
    """Container for data quality assessment metrics."""
    
    def __init__(self):
        self.total_samples = 0
        self.clean_samples = 0
        self.quality_issues = {}
        self.removed_samples = 0
        self.text_stats = {}
    
    @property
    def quality_percentage(self) -> float:
        """Calculate quality percentage."""
        return (self.clean_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'total_samples': self.total_samples,
            'clean_samples': self.clean_samples,
            'quality_issues': self.quality_issues,
            'removed_samples': self.removed_samples,
            'text_stats': self.text_stats,
            'quality_percentage': (self.clean_samples / self.total_samples * 100) if self.total_samples > 0 else 0
        }


def load_dataset(path: Union[Path, str]) -> pd.DataFrame:
    """
    Load GoEmotions dataset from CSV file.
    
    Args:
        path: Path to CSV file containing GoEmotions data
        
    Returns:
        DataFrame with text and emotion labels
        
    Raises:
        FileNotFoundError: If dataset file doesn't exist
        ValueError: If required columns are missing
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    
    df = pd.read_csv(path)
    
    # Verify essential columns exist
    if 'text' not in df.columns:
        raise ValueError("Dataset must contain 'text' column")
    
    # Check for emotion columns
    available_emotions = [col for col in EMOTION_COLUMNS if col in df.columns]
    if len(available_emotions) == 0:
        raise ValueError("Dataset must contain at least one emotion column")
    
    print(f"📊 Dataset loaded: {len(df):,} samples with {len(available_emotions)} emotion labels")
    return df


def assess_text_quality(texts: pd.Series) -> Dict[str, int]:
    """
    Assess text quality issues based on EDA findings.
    
    Args:
        texts: Series of text strings
        
    Returns:
        Dictionary with counts of different quality issues
    """
    issues = {}
    
    # Very short texts (< 5 characters) - 0.03% in EDA
    issues['very_short'] = (texts.str.len() < 5).sum()
    
    # Very long texts (> 500 characters) - potential truncation needed
    issues['very_long'] = (texts.str.len() > 500).sum()
    
    # Texts with mostly punctuation - 0.08% in EDA
    mostly_punct = texts.apply(lambda x: len(re.findall(r'[^\w\s]', str(x))) > len(re.findall(r'\w', str(x))) if pd.notna(x) else False)
    issues['mostly_punctuation'] = mostly_punct.sum()
    
    # Texts with repeated characters (like "yessssss") - 1.05% in EDA
    repeated_chars = texts.str.contains(r'(.)\1{4,}', na=False)
    issues['repeated_chars'] = repeated_chars.sum()
    
    # Texts that are all uppercase (emotional intensity) - 1.03% in EDA
    all_caps = texts.apply(lambda x: str(x).isupper() and len(str(x).strip()) > 5 if pd.notna(x) else False)
    issues['all_caps'] = all_caps.sum()
    
    # Texts with no letters - 0.01% in EDA
    no_letters = texts.apply(lambda x: not re.search(r'[a-zA-Z]', str(x)) if pd.notna(x) else False)
    issues['no_letters'] = no_letters.sum()
    
    # Empty or null texts
    issues['empty_null'] = texts.isna().sum() + (texts == '').sum()
    
    return issues


def clean_text(text_series: pd.Series, aggressive: bool = False) -> pd.Series:
    """
    Clean and normalize text based on EDA findings.
    
    Args:
        text_series: Series of text strings to clean
        aggressive: If True, apply more aggressive cleaning
        
    Returns:
        Series of cleaned text strings
    """
    cleaned = text_series.astype(str).copy()
    
    # Basic cleaning
    # 1. Normalize whitespace (multiple spaces, tabs, newlines)
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True)
    
    # 2. Strip leading/trailing whitespace
    cleaned = cleaned.str.strip()
    
    # 3. Handle repeated characters (from EDA: 1.05% of texts)
    # Reduce repeated characters (e.g., "yesssss" -> "yesss")
    cleaned = cleaned.str.replace(r'(.)\1{3,}', r'\1\1\1', regex=True)
    
    if aggressive:
        # Aggressive cleaning for baseline models
        
        # 4. Normalize case (but preserve some capitalization for emotion)
        # Convert to lowercase but keep track of ALL-CAPS words
        caps_pattern = cleaned.str.findall(r'\b[A-Z]{2,}\b')
        cleaned = cleaned.str.lower()
        
        # 5. Handle special Reddit patterns
        # Replace [NAME] tokens (common in Reddit data)
        cleaned = cleaned.str.replace(r'\[NAME\]', 'person', regex=True)
        
        # 6. Normalize punctuation patterns
        # Reduce multiple punctuation marks
        cleaned = cleaned.str.replace(r'[!]{2,}', '!!', regex=True)
        cleaned = cleaned.str.replace(r'[?]{2,}', '??', regex=True)
        cleaned = cleaned.str.replace(r'[.]{3,}', '...', regex=True)
        
        # 7. Remove URLs (0% in EDA but good practice)
        cleaned = cleaned.str.replace(r'http[s]?://\S+|www\.\S+', '', regex=True)
        
        # 8. Normalize mentions and hashtags (0.02% mentions, 0.3% hashtags in EDA)
        cleaned = cleaned.str.replace(r'@\w+', '@user', regex=True)
        cleaned = cleaned.str.replace(r'#(\w+)', r'\1', regex=True)  # Keep hashtag content
    
    # Final whitespace cleanup
    cleaned = cleaned.str.replace(r'\s+', ' ', regex=True).str.strip()
    
    return cleaned


def filter_quality_issues(df: pd.DataFrame, text_column: str = 'text', 
                         remove_issues: bool = True) -> Tuple[pd.DataFrame, DataQualityMetrics]:
    """
    Filter out samples with quality issues based on EDA findings.
    
    Args:
        df: DataFrame with text data
        text_column: Name of text column to assess
        remove_issues: Whether to remove problematic samples
        
    Returns:
        Tuple of (filtered_df, quality_metrics)
    """
    metrics = DataQualityMetrics()
    metrics.total_samples = len(df)
    
    # Assess quality issues
    quality_issues = assess_text_quality(df[text_column])
    metrics.quality_issues = quality_issues
    
    if not remove_issues:
        metrics.clean_samples = len(df)
        return df.copy(), metrics
    
    # Create filter mask
    texts = df[text_column].astype(str)
    
    # Keep samples that don't have severe quality issues
    keep_mask = pd.Series(True, index=df.index)
    
    # Remove very short texts (< 5 characters)
    keep_mask &= texts.str.len() >= 5
    
    # Remove texts with no letters
    keep_mask &= texts.apply(lambda x: bool(re.search(r'[a-zA-Z]', str(x))))
    
    # Remove empty/null texts
    keep_mask &= ~(texts.isna() | (texts == ''))
    
    # Optionally remove texts with mostly punctuation
    mostly_punct = texts.apply(lambda x: len(re.findall(r'[^\w\s]', str(x))) > len(re.findall(r'\w', str(x))))
    keep_mask &= ~mostly_punct
    
    filtered_df = df[keep_mask].copy().reset_index(drop=True)
    
    metrics.clean_samples = len(filtered_df)
    metrics.removed_samples = metrics.total_samples - metrics.clean_samples
    
    # Calculate text statistics on clean data
    clean_texts = filtered_df[text_column].astype(str)
    metrics.text_stats = {
        'avg_length': float(clean_texts.str.len().mean()),
        'median_length': float(clean_texts.str.len().median()),
        'avg_words': float(clean_texts.str.split().str.len().mean()),
        'min_length': int(clean_texts.str.len().min()),
        'max_length': int(clean_texts.str.len().max())
    }
    
    print(f"🧹 Quality filtering: {metrics.total_samples:,} → {metrics.clean_samples:,} samples")
    print(f"   Removed {metrics.removed_samples:,} ({metrics.removed_samples/metrics.total_samples*100:.2f}%) problematic samples")
    
    return filtered_df, metrics


def prepare_emotion_labels(df: pd.DataFrame, emotion_columns: Optional[List[str]] = None) -> np.ndarray:
    """
    Prepare multi-label emotion arrays for training.
    
    Args:
        df: DataFrame with emotion columns
        emotion_columns: List of emotion column names (default: all available)
        
    Returns:
        Binary array of shape (n_samples, n_emotions)
    """
    if emotion_columns is None:
        emotion_columns = [col for col in EMOTION_COLUMNS if col in df.columns]
    
    if len(emotion_columns) == 0:
        raise ValueError("No emotion columns found in dataset")
    
    # Extract emotion labels and convert to binary array
    emotion_data = df[emotion_columns].values.astype(int)
    
    print(f"🏷️  Emotion labels prepared: {len(emotion_columns)} emotions for {len(df):,} samples")
    print(f"   Multi-label samples: {(emotion_data.sum(axis=1) > 1).sum():,} ({(emotion_data.sum(axis=1) > 1).mean()*100:.1f}%)")
    
    return emotion_data


def split_dataset(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1, 
                 emotion_columns: Optional[List[str]] = None, random_state: int = 42) -> Dict[str, Any]:
    """
    Split dataset into train/validation/test sets with stratification for multi-label.
    
    Args:
        df: DataFrame with text and emotion data
        test_size: Proportion for test set
        val_size: Proportion for validation set (from remaining data after test split)
        emotion_columns: List of emotion columns for stratification
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing train/val/test splits
    """
    if emotion_columns is None:
        emotion_columns = [col for col in EMOTION_COLUMNS if col in df.columns]
    
    # For multi-label stratification, we'll use a simplified approach
    # Create a stratification key based on the most frequent emotion
    emotion_data = df[emotion_columns].values
    primary_emotion = emotion_data.argmax(axis=1)  # Index of strongest emotion
    
    # First split: separate test set
    train_val_idx, test_idx = train_test_split(
        range(len(df)), 
        test_size=test_size, 
        stratify=primary_emotion,
        random_state=random_state
    )
    
    # Second split: separate validation from training
    if val_size > 0:
        train_idx, val_idx = train_test_split(
            train_val_idx,
            test_size=val_size / (1 - test_size),  # Adjust val_size for remaining data
            stratify=primary_emotion[train_val_idx],
            random_state=random_state
        )
    else:
        train_idx = train_val_idx
        val_idx = []
    
    # Create splits
    splits = {
        'train': df.iloc[train_idx].copy().reset_index(drop=True),
        'test': df.iloc[test_idx].copy().reset_index(drop=True)
    }
    
    if val_size > 0:
        splits['val'] = df.iloc[val_idx].copy().reset_index(drop=True)
    
    # Print split information
    print(f"📊 Dataset splits created:")
    print(f"   Train: {len(splits['train']):,} samples ({len(splits['train'])/len(df)*100:.1f}%)")
    if val_size > 0:
        print(f"   Val:   {len(splits['val']):,} samples ({len(splits['val'])/len(df)*100:.1f}%)")
    print(f"   Test:  {len(splits['test']):,} samples ({len(splits['test'])/len(df)*100:.1f}%)")
    
    return splits


def prepare_features(df: pd.DataFrame, text_column: str = 'text', 
                    aggressive_cleaning: bool = False) -> List[str]:
    """
    Prepare cleaned text features ready for vectorization.
    
    Args:
        df: DataFrame with text data
        text_column: Name of column containing text
        aggressive_cleaning: Whether to apply aggressive cleaning for baseline models
        
    Returns:
        List of cleaned text strings
    """
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame")
    
    cleaned_texts = clean_text(df[text_column], aggressive=aggressive_cleaning)
    
    print(f"📝 Text features prepared: {len(cleaned_texts):,} samples")
    print(f"   Average length: {cleaned_texts.str.len().mean():.1f} characters")
    print(f"   Aggressive cleaning: {'Yes' if aggressive_cleaning else 'No'}")
    
    return cleaned_texts.tolist()


def save_preprocessing_results(quality_metrics: DataQualityMetrics, 
                             splits_info: Dict[str, int],
                             output_dir: Union[Path, str] = "results/metrics") -> str:
    """
    Save preprocessing results to JSON file in results directory.
    
    Args:
        quality_metrics: Data quality assessment results
        splits_info: Information about dataset splits
        output_dir: Directory to save results
        
    Returns:
        Path to saved results file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    results = {
        'timestamp': timestamp,
        'preprocessing_info': {
            'emotion_columns_count': len(EMOTION_COLUMNS),
            'emotion_columns': EMOTION_COLUMNS,
            'cleaning_applied': True,
            'quality_filtering': True
        },
        'quality_metrics': quality_metrics.to_dict(),
        'dataset_splits': splits_info,
        'text_statistics': quality_metrics.text_stats
    }
    
    output_file = output_dir / f"preprocessing_results_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"💾 Preprocessing results saved to: {output_file}")
    return str(output_file)


if __name__ == "__main__":
    raise SystemExit("This module provides helper functions and is not intended for CLI use.")