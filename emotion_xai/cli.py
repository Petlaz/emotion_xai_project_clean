"""Command-line interface for emotion-xai package."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from emotion_xai.utils.config import get_config
from emotion_xai.data.preprocessing import load_dataset, prepare_features
from emotion_xai.models.baseline import train_baseline
from emotion_xai.models.transformer import TrainingConfig, train_model


def train_baseline_cli(args):
    """Train baseline model from CLI."""
    config = get_config()
    
    # Load data
    data = load_dataset(Path(args.data_path))
    features = prepare_features(data, args.text_column)
    
    # For demo purposes, create dummy labels
    labels = [0] * len(features)  # Replace with actual label logic
    
    # Train model
    model = train_baseline(features, labels)
    print("Baseline model training completed!")


def train_transformer_cli(args):
    """Train transformer model from CLI."""
    config = TrainingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=Path(args.output_dir)
    )
    
    train_model(config)
    print("Transformer model training completed!")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Emotion-XAI CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train baseline command
    baseline_parser = subparsers.add_parser("train-baseline", help="Train baseline model")
    baseline_parser.add_argument("--data-path", required=True, help="Path to training data")
    baseline_parser.add_argument("--text-column", default="text", help="Text column name")
    baseline_parser.set_defaults(func=train_baseline_cli)
    
    # Train transformer command
    transformer_parser = subparsers.add_parser("train-transformer", help="Train transformer model")
    transformer_parser.add_argument("--model-name", default="distilroberta-base", help="Model name")
    transformer_parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    transformer_parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    transformer_parser.add_argument("--output-dir", default="models/distilroberta_finetuned", help="Output directory")
    transformer_parser.set_defaults(func=train_transformer_cli)
    
    args = parser.parse_args()
    
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()