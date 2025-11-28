"""
Unit tests for the comprehensive data preprocessing module.
Tests all preprocessing functions with various data scenarios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

# Try importing pytest, fallback to basic testing if not available
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    PYTEST_AVAILABLE = False

from emotion_xai.data.preprocessing import (
    load_dataset,
    assess_text_quality,
    clean_text,
    filter_quality_issues,
    prepare_emotion_labels,
    split_dataset,
    prepare_features,
    save_preprocessing_results,
    DataQualityMetrics,
    EMOTION_COLUMNS
)


class TestPreprocessing:
    """Test comprehensive preprocessing functionality."""
    
    def test_clean_text_basic(self):
        """Test basic text cleaning functionality."""
        text_series = pd.Series([
            "  This   has    extra    spaces  ",
            "Normal text",
            "   \t  Mixed    whitespace \n  "
        ])
        
        cleaned = clean_text(text_series)
        
        assert cleaned.iloc[0] == "This has extra spaces"
        assert cleaned.iloc[1] == "Normal text"
        assert cleaned.iloc[2] == "Mixed whitespace"
    
    def test_clean_text_repeated_chars(self):
        """Test repeated character handling."""
        texts = pd.Series(['yesssssss', 'nooooooo'])
        
        cleaned = clean_text(texts)
        
        assert cleaned.iloc[0] == 'yesss'
        assert cleaned.iloc[1] == 'nooo'
    
    def test_assess_text_quality(self):
        """Test text quality assessment."""
        texts = pd.Series([
            'Hello world',  # Good text
            'Hi',          # Short text
            '!!!!',        # Mostly punctuation
            '',            # Empty
            None           # Null
        ])
        
        issues = assess_text_quality(texts)
        
        assert issues['very_short'] >= 1
        assert issues['mostly_punctuation'] >= 1
        assert issues['empty_null'] >= 2
    
    def test_prepare_features_conservative(self):
        """Test conservative feature preparation."""
        sample_data = pd.DataFrame({
            'text': ['Hello world', 'Good morning', 'Nice day']
        })
        
        features = prepare_features(sample_data, aggressive_cleaning=False)
        
        assert len(features) == 3
        assert isinstance(features, list)
        assert all(isinstance(text, str) for text in features)
    
    def test_prepare_features_aggressive(self):
        """Test aggressive feature preparation."""
        sample_data = pd.DataFrame({
            'text': ['Hello WORLD', 'Good [NAME]', '@user #test']
        })
        
        features = prepare_features(sample_data, aggressive_cleaning=True)
        
        assert len(features) == 3
        assert features[0] == 'hello world'
        assert 'person' in features[1]
    
    def test_prepare_emotion_labels(self):
        """Test emotion label preparation."""
        df = pd.DataFrame({
            'text': ['Hello', 'World'],
            'joy': [1, 0],
            'sadness': [0, 1]
        })
        
        labels = prepare_emotion_labels(df, ['joy', 'sadness'])
        
        assert labels.shape == (2, 2)
        assert labels[0, 0] == 1  # First sample has joy
        assert labels[1, 1] == 1  # Second sample has sadness
    
    def test_split_dataset(self):
        """Test dataset splitting functionality."""
        df = pd.DataFrame({
            'text': [f'Text {i}' for i in range(50)],
            'joy': [1 if i % 2 == 0 else 0 for i in range(50)],
            'sadness': [1 if i % 2 == 1 else 0 for i in range(50)]
        })
        
        splits = split_dataset(df, test_size=0.2, val_size=0.1, random_state=42)
        
        assert 'train' in splits
        assert 'val' in splits
        assert 'test' in splits
        
        total = len(splits['train']) + len(splits['val']) + len(splits['test'])
        assert total == len(df)
    
    def test_filter_quality_issues(self):
        """Test quality filtering functionality."""
        df = pd.DataFrame({
            'text': ['Hello world', 'Hi', '!!!!', '123', ''],
            'neutral': [1, 0, 1, 0, 1]
        })
        
        filtered_df, metrics = filter_quality_issues(df, remove_issues=True)
        
        assert len(filtered_df) <= len(df)
        assert metrics.total_samples == 5
        assert isinstance(metrics.to_dict(), dict)
    
    def test_data_quality_metrics(self):
        """Test DataQualityMetrics class."""
        metrics = DataQualityMetrics()
        metrics.total_samples = 100
        metrics.clean_samples = 95
        
        result = metrics.to_dict()
        
        assert result['total_samples'] == 100
        assert result['clean_samples'] == 95
        assert result['quality_percentage'] == 95.0


@pytest.fixture
def sample_dataframe():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        'text': ['Hello world', 'Good morning', 'Test message'],
        'joy': [1, 0, 0],
        'sadness': [0, 1, 0],
        'neutral': [0, 0, 1]
    })


def test_integration_preprocessing_pipeline(sample_dataframe, tmp_path):
    """Integration test for full preprocessing pipeline."""
    # Test the complete preprocessing workflow
    
    # Step 1: Quality filtering
    filtered_df, quality_metrics = filter_quality_issues(sample_dataframe)
    assert len(filtered_df) <= len(sample_dataframe)
    
    # Step 2: Emotion labels
    labels = prepare_emotion_labels(filtered_df)
    assert labels.shape[0] == len(filtered_df)
    
    # Step 3: Feature preparation
    features = prepare_features(filtered_df)
    assert len(features) == len(filtered_df)
    
    # Step 4: Save results
    splits_info = {'train_size': len(filtered_df), 'val_size': 0, 'test_size': 0}
    result_file = save_preprocessing_results(quality_metrics, splits_info, tmp_path)
    
    assert Path(result_file).exists()
    
    # Verify saved content
    with open(result_file, 'r') as f:
        data = json.load(f)
    
    assert 'timestamp' in data
    assert 'preprocessing_info' in data


if __name__ == "__main__":
    # Run basic functionality test
    print("Running basic preprocessing tests...")
    
    # Create sample data
    sample_df = pd.DataFrame({
        'text': ['Hello world', 'Good morning', 'Test message'],
        'joy': [1, 0, 0],
        'neutral': [0, 1, 1]
    })
    
    # Test basic functions
    features = prepare_features(sample_df)
    print(f"✅ Features prepared: {len(features)} samples")
    
    labels = prepare_emotion_labels(sample_df)
    print(f"✅ Labels prepared: {labels.shape}")
    
    print("✅ Basic preprocessing tests passed!")