#!/usr/bin/env python3
"""
Production Transformer Fine-tuning Script
Phase 3: Full-scale DistilRoBERTa training on GoEmotions dataset

This script performs full-scale transformer fine-tuning with:
- Full dataset (147K+ samples) 
- Optimized hyperparameters for F1-macro > 0.6 target
- Mac M1/MPS optimization with CPU fallback
- Comprehensive logging and checkpointing
- Automatic model evaluation and saving

Usage:
    python scripts/train_transformer_production.py [--config CONFIG_FILE] [--resume CHECKPOINT]
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Transformers and datasets
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification,
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset
from sklearn.metrics import f1_score, precision_recall_fscore_support, classification_report

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import project modules
try:
    from emotion_xai.utils.device import setup_mac_optimizations
    from emotion_xai.data.preprocessing import load_dataset
except ImportError as e:
    print(f"⚠️  Warning: Could not import project modules: {e}")
    print("Running in standalone mode...")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)


class ProductionConfig:
    """Production configuration for transformer fine-tuning"""
    
    def __init__(self, config_file: Optional[str] = None):
        # Model configuration
        self.model_name = "distilroberta-base"
        self.num_labels = 28  # GoEmotions emotions
        self.max_length = 128  # Optimal for memory and performance
        
        # Training hyperparameters (optimized for performance)
        self.learning_rate = 2e-5
        self.num_epochs = 5  # Increased for better performance
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        # Batch sizes (optimized for Mac M1)
        self.batch_size_train = 16  # Larger for better performance
        self.batch_size_eval = 32   # Even larger for evaluation
        self.gradient_accumulation_steps = 4  # Effective batch size: 64
        
        # Device configuration
        self.device = self._setup_device()
        self.use_fp16 = False  # Disable fp16 for compatibility (PyTorch 2.9.1 has MPS fp16 issues)
        
        # Training settings
        self.logging_steps = 50
        self.eval_steps = 250  # Less frequent for speed
        self.save_steps = 500
        self.early_stopping_patience = 3
        
        # Data settings
        self.use_full_dataset = True  # Use all available data
        self.max_train_samples = None  # No limit
        self.max_val_samples = None
        
        # Output paths
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"models/distilroberta_production_{timestamp}")
        self.results_dir = Path("results/production_training")
        
        # Load custom config if provided
        if config_file and Path(config_file).exists():
            self._load_config(config_file)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_device(self) -> torch.device:
        """Setup optimal device for training"""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _load_config(self, config_file: str):
        """Load configuration from JSON file"""
        with open(config_file, 'r') as f:
            custom_config = json.load(f)
        
        for key, value in custom_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def save_config(self):
        """Save current configuration"""
        config_dict = {}
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                if isinstance(v, Path):
                    config_dict[k] = str(v)
                elif isinstance(v, torch.device):
                    config_dict[k] = str(v)
                else:
                    config_dict[k] = v
        
        config_path = self.results_dir / "production_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        return config_path


def load_processed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str]]:
    """Load processed datasets from Phase 2"""
    print("📁 Loading processed datasets...")
    
    processed_data_dir = Path("data/processed")
    
    # Find latest processed files
    latest_files = sorted(processed_data_dir.glob("*_20251128_045051.*"))
    if not latest_files:
        raise FileNotFoundError("No processed data files found. Run Phase 2 first.")
    
    # Load datasets
    train_df = pd.read_csv(processed_data_dir / "train_data_20251128_045051.csv")
    val_df = pd.read_csv(processed_data_dir / "val_data_20251128_045051.csv")
    test_df = pd.read_csv(processed_data_dir / "test_data_20251128_045051.csv")
    
    # Load emotion columns
    with open(processed_data_dir / "processed_features_20251128_045051.pkl", 'rb') as f:
        processed_features = pickle.load(f)
    
    emotion_columns = processed_features['emotion_columns']
    
    print(f"✅ Data loaded:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Validation: {len(val_df):,} samples")
    print(f"   Test: {len(test_df):,} samples")
    print(f"   Emotions: {len(emotion_columns)}")
    
    return train_df, val_df, test_df, emotion_columns


def create_datasets(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    test_df: pd.DataFrame,
    emotion_columns: List[str],
    tokenizer,
    config: ProductionConfig
) -> Tuple[Dataset, Dataset, Dataset]:
    """Create optimized HuggingFace datasets for production training"""
    
    print("📊 Creating production datasets...")
    
    def prepare_data(df, split_name):
        # Sample data if limits are set
        if split_name == "train" and config.max_train_samples:
            df = df.head(config.max_train_samples)
        elif split_name in ["val", "test"] and config.max_val_samples:
            df = df.head(config.max_val_samples)
        
        texts = df['text'].tolist()
        labels = df[emotion_columns].values.astype(np.float32)
        
        print(f"   Processing {split_name}: {len(texts):,} samples")
        
        # Tokenize in batches for memory efficiency
        batch_size = 1000
        all_encodings = {"input_ids": [], "attention_mask": []}
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            encodings = tokenizer(
                batch_texts,
                truncation=True,
                padding=False,  # Dynamic padding in collator
                max_length=config.max_length,
                return_tensors=None
            )
            
            all_encodings["input_ids"].extend(encodings["input_ids"])
            all_encodings["attention_mask"].extend(encodings["attention_mask"])
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"     Tokenized {i+len(batch_texts):,}/{len(texts):,} texts")
        
        # Create dataset
        dataset_dict = {
            'input_ids': all_encodings["input_ids"],
            'attention_mask': all_encodings["attention_mask"],
            'labels': labels.tolist()
        }
        
        return Dataset.from_dict(dataset_dict)
    
    # Create datasets
    train_dataset = prepare_data(train_df, "train")
    val_dataset = prepare_data(val_df, "val")
    test_dataset = prepare_data(test_df, "test")
    
    print(f"✅ Production datasets created:")
    print(f"   Train: {len(train_dataset):,} samples")
    print(f"   Validation: {len(val_dataset):,} samples")
    print(f"   Test: {len(test_dataset):,} samples")
    
    return train_dataset, val_dataset, test_dataset


def compute_metrics(eval_pred):
    """Compute comprehensive metrics for multi-label classification"""
    predictions, labels = eval_pred
    
    # Apply sigmoid to get probabilities
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(predictions))
    
    # Convert to predictions
    y_pred = (probs > 0.5).int().numpy()
    y_true = labels
    
    # Calculate metrics
    try:
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Exact match accuracy
        exact_match = np.mean((y_pred == y_true).all(axis=1))
        
        # Hamming accuracy (per-label)
        hamming_acc = np.mean(y_pred == y_true)
        
    except Exception as e:
        print(f"⚠️  Metrics computation error: {e}")
        f1_macro = f1_micro = f1_weighted = exact_match = hamming_acc = 0.0
    
    return {
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'f1_weighted': f1_weighted,
        'exact_match': exact_match,
        'hamming_accuracy': hamming_acc,
        'f1': f1_macro  # Primary metric
    }


def setup_training(
    model,
    tokenizer,
    train_dataset: Dataset,
    val_dataset: Dataset,
    config: ProductionConfig
) -> Trainer:
    """Setup production trainer with optimized arguments"""
    
    print("⚙️  Setting up production training...")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(config.output_dir),
        
        # Training schedule
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size_train,
        per_device_eval_batch_size=config.batch_size_eval,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        
        # Optimization
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        
        # Evaluation and saving
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        
        # Model selection
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1_macro",
        greater_is_better=True,
        
        # Hardware optimization
        fp16=config.use_fp16,
        dataloader_num_workers=4 if config.device.type != "mps" else 0,
        dataloader_pin_memory=config.device.type == "cuda",
        
        # Misc
        report_to=[],  # Disable external logging for compatibility
        save_total_limit=3,
        seed=42,
        data_seed=42,
        
        # Early stopping
        # early_stopping_patience=config.early_stopping_patience,  # Handled by callback
    )
    
    # Data collator
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, 
        return_tensors="pt"
    )
    
    # Early stopping callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping_patience
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
        callbacks=[early_stopping]
    )
    
    print(f"✅ Trainer configured:")
    print(f"   Effective batch size: {config.batch_size_train * config.gradient_accumulation_steps}")
    print(f"   Total steps: ~{len(train_dataset) // (config.batch_size_train * config.gradient_accumulation_steps) * config.num_epochs:,}")
    print(f"   Device: {config.device}")
    print(f"   FP16: {config.use_fp16}")
    
    return trainer


def train_model(trainer: Trainer, config: ProductionConfig, resume_from_checkpoint: Optional[str] = None) -> Dict:
    """Execute production training with monitoring"""
    
    print("\n" + "="*60)
    print("🚀 STARTING PRODUCTION TRANSFORMER TRAINING")
    print("="*60)
    
    # Record start time
    start_time = datetime.now()
    print(f"⏰ Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Start training (with resume if specified)
        if resume_from_checkpoint:
            print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
            train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            print("🔥 Beginning fine-tuning...")
            train_result = trainer.train()
        
        # Record end time
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n✅ TRAINING COMPLETED SUCCESSFULLY!")
        print(f"⏰ End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️  Duration: {duration}")
        print(f"📈 Final loss: {train_result.training_loss:.4f}")
        
        # Save training results
        training_results = {
            'training_loss': train_result.training_loss,
            'global_step': train_result.global_step,
            'duration_seconds': duration.total_seconds(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'success': True
        }
        
        return training_results
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        end_time = datetime.now()
        duration = end_time - start_time
        
        training_results = {
            'error': str(e),
            'duration_seconds': duration.total_seconds(),
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat(),
            'success': False
        }
        
        return training_results


def evaluate_model(trainer: Trainer, test_dataset: Dataset, config: ProductionConfig) -> Dict:
    """Comprehensive model evaluation"""
    
    print("\n📊 COMPREHENSIVE MODEL EVALUATION")
    print("="*50)
    
    # Evaluate on validation set
    print("🔍 Validation set evaluation...")
    val_metrics = trainer.evaluate()
    
    # Evaluate on test set
    print("🧪 Test set evaluation...")
    test_metrics = trainer.evaluate(test_dataset)
    
    # Display results
    print(f"\n🎯 Validation Results:")
    print(f"   F1-Macro: {val_metrics['eval_f1_macro']:.4f}")
    print(f"   F1-Micro: {val_metrics['eval_f1_micro']:.4f}")
    print(f"   F1-Weighted: {val_metrics['eval_f1_weighted']:.4f}")
    print(f"   Exact Match: {val_metrics['eval_exact_match']:.4f}")
    print(f"   Hamming Acc: {val_metrics['eval_hamming_accuracy']:.4f}")
    
    print(f"\n🎯 Test Results:")
    print(f"   F1-Macro: {test_metrics['eval_f1_macro']:.4f}")
    print(f"   F1-Micro: {test_metrics['eval_f1_micro']:.4f}")
    print(f"   F1-Weighted: {test_metrics['eval_f1_weighted']:.4f}")
    print(f"   Exact Match: {test_metrics['eval_exact_match']:.4f}")
    print(f"   Hamming Acc: {test_metrics['eval_hamming_accuracy']:.4f}")
    
    # Compare with baseline
    baseline_f1_macro = 0.161
    improvement = test_metrics['eval_f1_macro'] / baseline_f1_macro if test_metrics['eval_f1_macro'] > 0 else 0
    
    print(f"\n📈 Performance vs Baseline:")
    print(f"   Baseline F1-Macro: {baseline_f1_macro:.3f}")
    print(f"   Transformer F1-Macro: {test_metrics['eval_f1_macro']:.3f}")
    print(f"   Improvement: {improvement:.1f}x better")
    
    # Target achievement
    target_f1 = 0.6
    target_achieved = test_metrics['eval_f1_macro'] >= target_f1
    
    if target_achieved:
        print(f"🎉 TARGET ACHIEVED! F1-Macro {test_metrics['eval_f1_macro']:.3f} ≥ {target_f1}")
    else:
        progress = test_metrics['eval_f1_macro'] / target_f1 * 100
        print(f"📊 Target progress: {progress:.1f}% toward F1-Macro {target_f1}")
    
    return {
        'validation_metrics': val_metrics,
        'test_metrics': test_metrics,
        'baseline_comparison': {
            'baseline_f1_macro': baseline_f1_macro,
            'improvement_factor': improvement
        },
        'target_analysis': {
            'target_f1_macro': target_f1,
            'achieved': target_achieved,
            'progress_percent': test_metrics['eval_f1_macro'] / target_f1 * 100
        }
    }


def save_results(
    config: ProductionConfig,
    training_results: Dict,
    evaluation_results: Dict,
    trainer: Trainer,
    tokenizer
):
    """Save all training results and artifacts"""
    
    print("\n💾 SAVING RESULTS AND ARTIFACTS")
    print("="*40)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save model and tokenizer
    print(f"🤖 Saving model to: {config.output_dir}")
    trainer.save_model(str(config.output_dir))
    tokenizer.save_pretrained(str(config.output_dir))
    
    # Save configuration
    config_path = config.save_config()
    print(f"⚙️  Configuration saved to: {config_path}")
    
    # Combine all results
    final_results = {
        'timestamp': timestamp,
        'model_info': {
            'model_name': config.model_name,
            'num_parameters': trainer.model.num_parameters(),
            'device': str(config.device)
        },
        'config': config.__dict__,
        'training_results': training_results,
        'evaluation_results': evaluation_results,
        'phase': 'Phase 3: Production Transformer Training'
    }
    
    # Save comprehensive results
    results_path = config.results_dir / f"production_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        # Convert non-serializable objects to strings
        serializable_results = {}
        for key, value in final_results.items():
            if isinstance(value, dict):
                serializable_results[key] = {k: str(v) if isinstance(v, (Path, torch.device)) else v 
                                           for k, v in value.items()}
            else:
                serializable_results[key] = str(value) if isinstance(value, (Path, torch.device)) else value
        
        json.dump(serializable_results, f, indent=2)
    
    print(f"📊 Results saved to: {results_path}")
    
    # Create summary
    summary = f"""
🎉 PRODUCTION TRAINING COMPLETED!
═════════════════════════════════════════

📊 Final Results:
   Test F1-Macro: {evaluation_results['test_metrics']['eval_f1_macro']:.4f}
   Test F1-Micro: {evaluation_results['test_metrics']['eval_f1_micro']:.4f}
   Improvement: {evaluation_results['baseline_comparison']['improvement_factor']:.1f}x over baseline

🎯 Target Achievement:
   Target F1-Macro: {evaluation_results['target_analysis']['target_f1_macro']}
   Status: {'✅ ACHIEVED' if evaluation_results['target_analysis']['achieved'] else '⏳ In Progress'}
   Progress: {evaluation_results['target_analysis']['progress_percent']:.1f}%

⏱️  Training Info:
   Duration: {training_results.get('duration_seconds', 0)/60:.1f} minutes
   Final Loss: {training_results.get('training_loss', 'N/A')}
   
💾 Artifacts:
   Model: {config.output_dir}
   Results: {results_path}
   Config: {config_path}

🚀 Ready for Phase 4: Explainable AI!
    """
    
    print(summary)
    
    # Save summary
    summary_path = config.results_dir / f"production_summary_{timestamp}.txt"
    with open(summary_path, 'w') as f:
        f.write(summary)
    
    return results_path, summary_path


def main():
    """Main training pipeline"""
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='Production Transformer Training')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--dry-run', action='store_true', help='Dry run without training')
    args = parser.parse_args()
    
    print("🚀 PRODUCTION TRANSFORMER FINE-TUNING SCRIPT")
    print("=" * 60)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # 1. Setup configuration
    config = ProductionConfig(args.config)
    
    # Handle resume: use existing directory instead of creating new one
    if args.resume:
        checkpoint_path = Path(args.resume)
        if checkpoint_path.exists():
            # Extract the parent model directory from checkpoint path
            # e.g., models/distilroberta_production_20251130_015947/checkpoint-1500 -> models/distilroberta_production_20251130_015947
            config.output_dir = checkpoint_path.parent
            print(f"🔄 Resuming from: {args.resume}")
            print(f"📁 Using existing output directory: {config.output_dir}")
        else:
            print(f"❌ Checkpoint not found: {args.resume}")
            return
    
    print(f"✅ Configuration loaded:")
    print(f"   Model: {config.model_name}")
    print(f"   Device: {config.device}")
    print(f"   Epochs: {config.num_epochs}")
    print(f"   Batch size: {config.batch_size_train}")
    print(f"   Output: {config.output_dir}")
    
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - Configuration only")
        print(f"📁 Would save to: {config.output_dir}")
        return
    
    # 2. Load data
    train_df, val_df, test_df, emotion_columns = load_processed_data()
    
    # 3. Setup tokenizer
    print(f"\n🔤 Loading tokenizer: {config.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 4. Create datasets
    train_dataset, val_dataset, test_dataset = create_datasets(
        train_df, val_df, test_df, emotion_columns, tokenizer, config
    )
    
    # 5. Load model
    print(f"\n🤖 Loading model: {config.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=config.num_labels,
        problem_type="multi_label_classification"
    )
    
    # Move to device
    model.to(config.device)
    print(f"✅ Model loaded ({model.num_parameters():,} parameters) on {config.device}")
    
    # 6. Setup trainer
    trainer = setup_training(model, tokenizer, train_dataset, val_dataset, config)
    
    # 7. Train model
    training_results = train_model(trainer, config, args.resume)
    
    if not training_results['success']:
        print("❌ Training failed. Check logs for details.")
        return
    
    # 8. Evaluate model
    evaluation_results = evaluate_model(trainer, test_dataset, config)
    
    # 9. Save results
    results_path, summary_path = save_results(
        config, training_results, evaluation_results, trainer, tokenizer
    )
    
    print(f"\n🎊 PRODUCTION TRAINING COMPLETE!")
    print(f"📈 Best F1-Macro: {evaluation_results['test_metrics']['eval_f1_macro']:.4f}")
    print(f"💾 Results: {results_path}")


if __name__ == "__main__":
    main()