#!/usr/bin/env python3
"""
Trained Transformer Model Usage Script
Phase 3: Using the fine-tuned DistilRoBERTa for emotion prediction

This script demonstrates how to use the production-trained transformer model for:
- Single text emotion prediction
- Batch text processing
- Confidence scores and explanations
- Integration with existing systems

Usage:
    python scripts/use_trained_model.py [--model MODEL_PATH] [--text "Your text here"]
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple
import warnings

import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    pipeline
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class EmotionPredictor:
    """
    Production-ready emotion prediction using trained DistilRoBERTa model
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        """
        Initialize the emotion predictor
        
        Args:
            model_path: Path to the trained model directory
            device: Device to run on ("auto", "cpu", "mps", "cuda")
        """
        self.model_path = Path(model_path)
        
        # Setup device
        if device == "auto":
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        print(f"🤖 Loading emotion predictor...")
        print(f"   Model: {self.model_path}")
        print(f"   Device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load emotion labels (from processed data)
        self.emotion_labels = self._load_emotion_labels()
        
        print(f"✅ Model loaded successfully!")
        print(f"   Parameters: {self.model.num_parameters():,}")
        print(f"   Emotions: {len(self.emotion_labels)}")
        
    def _load_emotion_labels(self) -> List[str]:
        """Load emotion labels from processed data"""
        try:
            processed_data_dir = Path("data/processed")
            # Find latest processed features file
            feature_files = list(processed_data_dir.glob("processed_features_*.pkl"))
            if feature_files:
                import pickle
                with open(sorted(feature_files)[-1], 'rb') as f:
                    features = pickle.load(f)
                return features['emotion_columns']
            else:
                # Fallback to GoEmotions standard labels
                return [
                    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
                    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
                    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
                    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
                    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
                ]
        except Exception as e:
            print(f"⚠️  Warning: Could not load emotion labels: {e}")
            return [f"emotion_{i}" for i in range(28)]  # Fallback
    
    def predict_single(self, text: str, threshold: float = 0.5, top_k: int = 5) -> Dict:
        """
        Predict emotions for a single text
        
        Args:
            text: Input text to analyze
            threshold: Probability threshold for positive prediction
            top_k: Number of top emotions to return
            
        Returns:
            Dictionary with predictions, probabilities, and metadata
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            
        # Convert to probabilities
        probs = torch.sigmoid(logits).cpu().numpy()[0]
        
        # Create predictions
        predictions = (probs > threshold).astype(int)
        
        # Get top emotions
        top_indices = np.argsort(probs)[::-1][:top_k]
        top_emotions = [
            {
                'emotion': self.emotion_labels[i],
                'probability': float(probs[i]),
                'predicted': bool(predictions[i])
            }
            for i in top_indices
        ]
        
        # Get all predicted emotions
        predicted_emotions = [
            self.emotion_labels[i] for i in range(len(predictions)) 
            if predictions[i] == 1
        ]
        
        return {
            'text': text,
            'predicted_emotions': predicted_emotions,
            'num_emotions': len(predicted_emotions),
            'top_emotions': top_emotions,
            'all_probabilities': {
                self.emotion_labels[i]: float(probs[i]) 
                for i in range(len(probs))
            },
            'confidence': float(np.max(probs)),
            'threshold': threshold
        }
    
    def predict_batch(self, texts: List[str], threshold: float = 0.5) -> List[Dict]:
        """
        Predict emotions for a batch of texts
        
        Args:
            texts: List of input texts
            threshold: Probability threshold for positive prediction
            
        Returns:
            List of prediction dictionaries
        """
        print(f"🔄 Processing batch of {len(texts)} texts...")
        
        results = []
        batch_size = 32  # Process in batches for memory efficiency
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize batch
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
            
            # Convert to probabilities
            probs = torch.sigmoid(logits).cpu().numpy()
            predictions = (probs > threshold).astype(int)
            
            # Process each text in batch
            for j, text in enumerate(batch_texts):
                predicted_emotions = [
                    self.emotion_labels[k] for k in range(len(predictions[j])) 
                    if predictions[j][k] == 1
                ]
                
                results.append({
                    'text': text,
                    'predicted_emotions': predicted_emotions,
                    'num_emotions': len(predicted_emotions),
                    'confidence': float(np.max(probs[j])),
                    'all_probabilities': {
                        self.emotion_labels[k]: float(probs[j][k]) 
                        for k in range(len(probs[j]))
                    }
                })
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"   Processed {i+len(batch_texts):,}/{len(texts):,} texts")
        
        print(f"✅ Batch processing complete!")
        return results
    
    def analyze_text_detailed(self, text: str) -> Dict:
        """
        Provide detailed analysis of a text including confidence intervals
        """
        result = self.predict_single(text, threshold=0.3)  # Lower threshold for analysis
        
        # Categorize emotions by probability
        high_conf = [e for e in result['top_emotions'] if e['probability'] > 0.7]
        med_conf = [e for e in result['top_emotions'] if 0.3 < e['probability'] <= 0.7]
        low_conf = [e for e in result['top_emotions'] if 0.1 < e['probability'] <= 0.3]
        
        return {
            **result,
            'analysis': {
                'high_confidence_emotions': high_conf,
                'medium_confidence_emotions': med_conf,
                'low_confidence_emotions': low_conf,
                'dominant_emotion': result['top_emotions'][0]['emotion'] if result['top_emotions'] else 'neutral',
                'emotional_complexity': len([e for e in result['top_emotions'] if e['probability'] > 0.5]),
                'text_length': len(text.split()),
                'sentiment_summary': self._get_sentiment_summary(result['top_emotions'])
            }
        }
    
    def _get_sentiment_summary(self, top_emotions: List[Dict]) -> str:
        """Create a human-readable sentiment summary"""
        if not top_emotions:
            return "neutral"
        
        top_emotion = top_emotions[0]
        prob = top_emotion['probability']
        
        if prob > 0.8:
            confidence = "very high"
        elif prob > 0.6:
            confidence = "high"
        elif prob > 0.4:
            confidence = "moderate"
        else:
            confidence = "low"
        
        return f"{top_emotion['emotion']} ({confidence} confidence: {prob:.1%})"


def find_latest_model() -> str:
    """Find the most recently trained model"""
    models_dir = Path("models")
    model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith("distilroberta_production")]
    
    if not model_dirs:
        raise FileNotFoundError("No trained models found in models/ directory")
    
    # Return the most recent model
    latest_model = sorted(model_dirs, key=lambda x: x.stat().st_mtime)[-1]
    return str(latest_model)


def interactive_demo():
    """Interactive demo for testing the model"""
    print("🎭 Interactive Emotion Analysis Demo")
    print("=" * 40)
    
    # Load model
    try:
        model_path = find_latest_model()
        predictor = EmotionPredictor(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print(f"\n💡 Enter text to analyze emotions (or 'quit' to exit)")
    print(f"📝 Example: 'I am so excited about this new project!'")
    print()
    
    while True:
        try:
            text = input("📄 Your text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                continue
            
            # Analyze text
            result = predictor.analyze_text_detailed(text)
            
            print(f"\n🎯 Analysis Results:")
            print(f"   📊 Dominant emotion: {result['analysis']['dominant_emotion']}")
            print(f"   🎭 Sentiment: {result['analysis']['sentiment_summary']}")
            print(f"   📈 Emotional complexity: {result['analysis']['emotional_complexity']} strong emotions")
            
            if result['predicted_emotions']:
                print(f"   ✅ Predicted emotions: {', '.join(result['predicted_emotions'])}")
            else:
                print(f"   😐 No strong emotions detected (mostly neutral)")
            
            print(f"\n🔍 Top 5 Emotions:")
            for emotion in result['top_emotions'][:5]:
                bar = "█" * int(emotion['probability'] * 20)
                print(f"   {emotion['emotion']:>15}: {emotion['probability']:.3f} {bar}")
            
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


def batch_analysis_demo():
    """Demo batch processing with sample texts"""
    print("📊 Batch Analysis Demo")
    print("=" * 30)
    
    # Load model
    try:
        model_path = find_latest_model()
        predictor = EmotionPredictor(model_path)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Sample texts for demonstration
    sample_texts = [
        "I am absolutely thrilled about this opportunity!",
        "This is really frustrating and annoying.",
        "I feel so grateful for all the support I've received.",
        "I'm quite nervous about the presentation tomorrow.",
        "The sunset was incredibly beautiful tonight.",
        "I can't believe this disappointing news.",
        "This is the most amazing thing ever!",
        "I'm feeling a bit confused about the instructions.",
        "Thank you so much for your help, I really appreciate it.",
        "I'm worried something might go wrong."
    ]
    
    # Analyze batch
    results = predictor.predict_batch(sample_texts)
    
    # Display results
    print(f"\n📊 Batch Analysis Results:")
    print("=" * 60)
    
    for i, result in enumerate(results, 1):
        text = result['text'][:50] + "..." if len(result['text']) > 50 else result['text']
        emotions = ', '.join(result['predicted_emotions']) if result['predicted_emotions'] else 'neutral'
        
        print(f"{i:2}. {text}")
        print(f"    🎭 Emotions: {emotions}")
        print(f"    📈 Confidence: {result['confidence']:.3f}")
        print()


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(description='Use trained emotion prediction model')
    parser.add_argument('--model', type=str, help='Path to trained model directory')
    parser.add_argument('--text', type=str, help='Text to analyze')
    parser.add_argument('--demo', choices=['interactive', 'batch'], help='Run demo mode')
    parser.add_argument('--batch-file', type=str, help='File with texts to analyze (one per line)')
    
    args = parser.parse_args()
    
    if args.demo == 'interactive':
        interactive_demo()
    elif args.demo == 'batch':
        batch_analysis_demo()
    elif args.text:
        # Single text analysis
        model_path = args.model or find_latest_model()
        predictor = EmotionPredictor(model_path)
        
        result = predictor.analyze_text_detailed(args.text)
        
        print(f"📄 Text: {args.text}")
        print(f"🎭 Emotions: {', '.join(result['predicted_emotions']) if result['predicted_emotions'] else 'neutral'}")
        print(f"📊 Sentiment: {result['analysis']['sentiment_summary']}")
        print(f"📈 Top emotions:")
        for emotion in result['top_emotions'][:3]:
            print(f"   {emotion['emotion']}: {emotion['probability']:.3f}")
    
    elif args.batch_file:
        # Batch file analysis
        model_path = args.model or find_latest_model()
        predictor = EmotionPredictor(model_path)
        
        with open(args.batch_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        results = predictor.predict_batch(texts)
        
        # Save results
        output_file = f"batch_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {output_file}")
    
    else:
        # Default: show usage and run interactive demo
        print("🎭 Emotion Prediction Model Usage")
        print("=" * 40)
        print()
        print("Options:")
        print("  --demo interactive    # Interactive text analysis")
        print("  --demo batch         # Batch analysis demo")
        print("  --text 'Your text'   # Analyze single text")
        print("  --batch-file file.txt # Analyze texts from file")
        print()
        print("Running interactive demo...")
        print()
        interactive_demo()


if __name__ == "__main__":
    main()