"""Integration tests for the complete pipeline."""

import tempfile
from pathlib import Path
import pandas as pd

from emotion_xai.data.preprocessing import prepare_features
from emotion_xai.models.baseline import train_baseline, evaluate_baseline


class TestIntegrationPipeline:
    """Integration tests for the complete ML pipeline."""
    
    def test_end_to_end_baseline_pipeline(self, sample_data):
        """Test complete pipeline from data to evaluation."""
        # Prepare features
        features = prepare_features(sample_data, 'text')
        
        # Create simple binary labels for testing
        labels = [1 if 'joy' in emotion or 'excited' in emotion 
                 else 0 for emotion in sample_data['emotions']]
        
        # Train model
        model = train_baseline(features, labels)
        
        # Evaluate model
        results = evaluate_baseline(model, features, labels)
        
        assert "accuracy" in results
        assert isinstance(results["accuracy"], float)