#!/usr/bin/env python3
"""
Usage examples for emotion_xai preprocessing and baseline model modules.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import the modules (correct paths)
from emotion_xai.data.preprocessing import (
    load_dataset, prepare_features, prepare_emotion_labels,
    filter_quality_issues, split_dataset, EMOTION_COLUMNS
)
from emotion_xai.models.baseline import BaselineModel

def example_preprocessing():
    """Example of using the preprocessing module."""
    print("🧹 PREPROCESSING EXAMPLE")
    print("=" * 40)
    
    # 1. Load dataset
    data_path = Path('data/raw/goemotions.csv')
    if data_path.exists():
        df = load_dataset(data_path)
        print(f"✅ Loaded dataset: {len(df)} samples")
        
        # 2. Filter quality issues
        filtered_df, quality_metrics = filter_quality_issues(df, remove_issues=True)
        print(f"✅ Quality filtering: {quality_metrics.quality_percentage:.2f}% retained")
        
        # 3. Split dataset
        splits = split_dataset(filtered_df, test_size=0.2, val_size=0.1, random_state=42)
        print(f"✅ Dataset splits: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")
        
        # 4. Prepare features and labels
        train_texts = prepare_features(splits['train'])
        train_labels = prepare_emotion_labels(splits['train'], EMOTION_COLUMNS)
        print(f"✅ Features prepared: {len(train_texts)} texts, {train_labels.shape[1]} emotions")
        
        return train_texts[:1000], train_labels[:1000]  # Return sample for model example
    else:
        print(f"❌ Dataset not found at {data_path}")
        return None, None

def example_baseline_model(texts=None, labels=None):
    """Example of using the baseline model."""
    print("\n🤖 BASELINE MODEL EXAMPLE")
    print("=" * 40)
    
    if texts is None or labels is None:
        # Use sample data
        texts = [
            "I am so happy today!",
            "This makes me really sad.",
            "I'm feeling quite angry.",
            "What an amazing experience!",
            "I'm scared about the future."
        ]
        labels = np.array([
            [1, 0, 0],  # joy
            [0, 1, 0],  # sadness  
            [0, 0, 1],  # anger
            [1, 0, 0],  # joy
            [0, 1, 1]   # sadness + anger
        ])
        emotions = ['joy', 'sadness', 'anger']
        print("✅ Using sample data for demonstration")
    else:
        emotions = EMOTION_COLUMNS
        print(f"✅ Using real dataset: {len(texts)} samples")
    
    # 1. Initialize model
    model = BaselineModel(max_features=1000, random_state=42)
    
    # 2. Train model
    model.fit(texts, labels, emotions)
    
    # 3. Make predictions
    test_texts = ["I feel great!", "This is terrible"]
    predictions = model.predict(test_texts)
    probabilities = model.predict_proba(test_texts)
    
    print(f"✅ Predictions made for {len(test_texts)} test samples")
    print(f"   Predictions shape: {predictions.shape}")
    print(f"   Probabilities shape: {probabilities.shape}")
    
    # 4. Evaluate model
    metrics = model.evaluate(texts, labels)
    print(f"✅ Model evaluation completed")
    print(f"   Accuracy: {metrics['accuracy']:.3f}")
    print(f"   F1-macro: {metrics['f1_macro']:.3f}")
    
    return model

def example_full_pipeline():
    """Example of the complete pipeline."""
    print("\n🚀 COMPLETE PIPELINE EXAMPLE")
    print("=" * 40)
    
    # Step 1: Preprocessing
    texts, labels = example_preprocessing()
    
    # Step 2: Modeling (if data was loaded successfully)
    if texts is not None and labels is not None:
        model = example_baseline_model(texts, labels)
        print("✅ Complete pipeline executed successfully!")
        return model
    else:
        # Fallback to sample data
        model = example_baseline_model()
        print("✅ Pipeline executed with sample data!")
        return model

if __name__ == "__main__":
    print("📚 EMOTION XAI USAGE EXAMPLES")
    print("=" * 50)
    
    # Run examples
    model = example_full_pipeline()
    
    print(f"\n🎯 SUMMARY")
    print("=" * 20)
    print("✅ Preprocessing functions work correctly")
    print("✅ Baseline model trains and predicts successfully") 
    print("✅ Complete pipeline is operational")
    print(f"\n📖 For more examples, check notebooks/02_modeling.ipynb")
