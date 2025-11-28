"""Tests for baseline model functionality."""

import numpy as np
from emotion_xai.models.baseline import build_baseline_pipeline, train_baseline, evaluate_baseline


class TestBaselineModel:
    """Test baseline model functionality."""
    
    def test_build_baseline_pipeline(self):
        """Test pipeline creation."""
        pipeline = build_baseline_pipeline(max_features=1000)
        
        assert len(pipeline.steps) == 2
        assert pipeline.steps[0][0] == "tfidf"
        assert pipeline.steps[1][0] == "clf"
    
    def test_train_baseline(self, sample_texts):
        """Test baseline model training."""
        X = sample_texts
        y = np.array([1, 0, 1, 0, 1])  # Binary labels for testing
        
        model = train_baseline(X, y)
        
        # Check that model can make predictions
        predictions = model.predict(X)
        assert len(predictions) == len(X)
        assert all(pred in [0, 1] for pred in predictions)
    
    def test_evaluate_baseline(self, sample_texts):
        """Test baseline model evaluation."""
        X = sample_texts
        y = np.array([1, 0, 1, 0, 1])
        
        model = train_baseline(X, y)
        results = evaluate_baseline(model, X, y)
        
        assert "accuracy" in results
        assert 0.0 <= results["accuracy"] <= 1.0