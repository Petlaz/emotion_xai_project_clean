"""Tests for data preprocessing module."""

import pytest
import pandas as pd
from pathlib import Path

from emotion_xai.data.preprocessing import load_dataset, clean_text, prepare_features


class TestPreprocessing:
    """Test data preprocessing functions."""
    
    def test_clean_text(self, sample_data):
        """Test text cleaning functionality."""
        text_series = pd.Series([
            "  This   has    extra    spaces  ",
            "Normal text",
            "   \t  Mixed    whitespace \n  "
        ])
        
        cleaned = clean_text(text_series)
        
        assert cleaned.iloc[0] == "This has extra spaces"
        assert cleaned.iloc[1] == "Normal text"
        assert cleaned.iloc[2] == "Mixed whitespace"
    
    def test_prepare_features(self, sample_data):
        """Test feature preparation."""
        features = prepare_features(sample_data, 'text')
        
        assert len(features) == len(sample_data)
        assert isinstance(features, list)
        assert all(isinstance(text, str) for text in features)
    
    def test_load_dataset_file_not_found(self, temp_dir):
        """Test load_dataset with non-existent file."""
        non_existent_path = temp_dir / "missing.csv"
        
        with pytest.raises(FileNotFoundError):
            load_dataset(non_existent_path)
    
    def test_load_dataset_success(self, temp_dir, sample_data):
        """Test successful dataset loading."""
        csv_path = temp_dir / "test_data.csv"
        sample_data.to_csv(csv_path, index=False)
        
        loaded_data = load_dataset(csv_path)
        
        pd.testing.assert_frame_equal(loaded_data, sample_data)