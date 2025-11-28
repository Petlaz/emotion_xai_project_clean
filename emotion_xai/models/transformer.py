"""Fine-tuning routines for DistilRoBERTa on emotion classification with Mac optimizations."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        EarlyStoppingCallback
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers library not available. Install with: pip install transformers")

from ..utils.device import MacDeviceManager, setup_mac_optimizations


@dataclass
class MacOptimizedTrainingConfig:
    """Configuration for Mac-optimized transformer fine-tuning."""

    # Model configuration
    model_name: str = "distilroberta-base"
    num_labels: int = 27  # GoEmotions has 27 emotion labels
    
    # Training hyperparameters (Mac optimized)
    learning_rate: float = 2e-5
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    
    # Mac-specific batch sizes
    batch_size_train: Optional[int] = None  # Auto-detected based on device
    batch_size_eval: Optional[int] = None   # Auto-detected based on device
    gradient_accumulation_steps: int = 4    # For effective larger batch size
    
    # Device and optimization
    device: Optional[str] = None  # Auto-detected (mps, cuda, cpu)
    use_mac_optimizations: bool = True
    mixed_precision: bool = False  # MPS doesn't fully support AMP yet
    
    # Data loading (Mac optimized)
    dataloader_num_workers: Optional[int] = None  # Auto-detected
    dataloader_pin_memory: Optional[bool] = None  # Auto-detected
    
    # Output and logging
    output_dir: Path = Path("models/distilroberta_finetuned")
    logging_steps: int = 50
    eval_steps: int = 500
    save_steps: int = 500
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Mac memory management
    max_seq_length: int = 128  # Reduced for Mac memory efficiency
    clear_cache_every_n_steps: int = 100  # Clear MPS cache regularly


class MacOptimizedTransformerTrainer:
    """Transformer trainer optimized for Mac/Apple Silicon."""
    
    def __init__(self, config: MacOptimizedTrainingConfig):
        self.config = config
        self.device_manager = MacDeviceManager() if config.use_mac_optimizations else None
        self.device = None
        self.model = None
        self.tokenizer = None
        
    def setup_device_and_optimization(self) -> torch.device:
        """Setup device and apply Mac-specific optimizations."""
        if self.config.use_mac_optimizations:
            self.device, device_info = setup_mac_optimizations(verbose=True)
            
            # Update config with device-specific batch sizes
            if self.config.batch_size_train is None:
                self.config.batch_size_train = self.device_manager.get_optimal_batch_size(
                    "transformer", "train"
                )
            
            if self.config.batch_size_eval is None:
                self.config.batch_size_eval = self.device_manager.get_optimal_batch_size(
                    "transformer", "inference"
                )
            
            # Set DataLoader kwargs
            dataloader_kwargs = self.device_manager.get_dataloader_kwargs(self.device)
            if self.config.dataloader_num_workers is None:
                self.config.dataloader_num_workers = dataloader_kwargs.get('num_workers', 4)
            if self.config.dataloader_pin_memory is None:
                self.config.dataloader_pin_memory = dataloader_kwargs.get('pin_memory', False)
        else:
            # Standard device detection
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        
        return self.device
    
    def load_model_and_tokenizer(self) -> Tuple[nn.Module, object]:
        """Load model and tokenizer with Mac optimizations."""
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True  # Use fast tokenizers when available
        )
        
        # Load model with Mac-specific settings
        model_kwargs = {
            "num_labels": self.config.num_labels,
            "torch_dtype": torch.float32,  # MPS works best with float32
        }
        
        # Add Mac-specific optimizations
        if self.device.type == "mps":
            model_kwargs["low_cpu_mem_usage"] = True
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Apply Mac-specific model optimizations
        if self.config.use_mac_optimizations and self.device.type == "mps":
            # Enable MPS fallback for unsupported operations
            torch.backends.mps.enable_fallback(True)
        
        return self.model, self.tokenizer
    
    def get_training_arguments(self) -> TrainingArguments:
        """Get Mac-optimized training arguments."""
        # Base arguments
        args = {
            "output_dir": str(self.config.output_dir),
            "num_train_epochs": self.config.num_epochs,
            "per_device_train_batch_size": self.config.batch_size_train,
            "per_device_eval_batch_size": self.config.batch_size_eval,
            "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
            "learning_rate": self.config.learning_rate,
            "weight_decay": self.config.weight_decay,
            "warmup_ratio": self.config.warmup_ratio,
            "logging_steps": self.config.logging_steps,
            "eval_steps": self.config.eval_steps,
            "save_steps": self.config.save_steps,
            "evaluation_strategy": "steps",
            "save_strategy": "steps",
            "load_best_model_at_end": True,
            "metric_for_best_model": "eval_f1",
            "greater_is_better": True,
            "dataloader_num_workers": self.config.dataloader_num_workers,
            "dataloader_pin_memory": self.config.dataloader_pin_memory,
            "remove_unused_columns": False,
            "push_to_hub": False,
        }
        
        # Mac-specific optimizations
        if self.device.type == "mps":
            args.update({
                "fp16": False,  # MPS doesn't support fp16 yet
                "bf16": False,  # MPS doesn't support bf16 yet
                "torch_compile": False,  # Not yet stable on MPS
                "dataloader_persistent_workers": False,  # Can cause issues on MPS
            })
        elif self.device.type == "cuda":
            args.update({
                "fp16": self.config.mixed_precision,
                "dataloader_persistent_workers": True,
            })
        else:  # CPU
            args.update({
                "fp16": False,
                "bf16": False,
                "dataloader_persistent_workers": False,
            })
        
        return TrainingArguments(**args)
    
    def create_trainer(self, train_dataset, eval_dataset, compute_metrics_fn) -> Trainer:
        """Create Mac-optimized Trainer."""
        training_args = self.get_training_arguments()
        
        # Callbacks for Mac optimization
        callbacks = []
        if self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience,
                    early_stopping_threshold=self.config.early_stopping_threshold
                )
            )
        
        # Custom trainer class for Mac optimizations
        if self.config.use_mac_optimizations:
            trainer_class = MacOptimizedTrainer
        else:
            trainer_class = Trainer
        
        trainer = trainer_class(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics_fn,
            callbacks=callbacks,
            tokenizer=self.tokenizer,
        )
        
        # Set custom attributes for Mac optimization
        if self.config.use_mac_optimizations:
            trainer.device_manager = self.device_manager
            trainer.clear_cache_every_n_steps = self.config.clear_cache_every_n_steps
        
        return trainer


if TRANSFORMERS_AVAILABLE:
    class MacOptimizedTrainer(Trainer):
        """Custom Trainer with Mac/MPS optimizations."""
        
        def training_step(self, model, inputs):
            """Override training step with Mac optimizations."""
            # Standard training step
            result = super().training_step(model, inputs)
            
            # Clear MPS cache periodically to prevent memory issues
            if hasattr(self, 'clear_cache_every_n_steps') and self.state.global_step % self.clear_cache_every_n_steps == 0:
                if hasattr(self, 'device_manager') and self.device_manager:
                    self.device_manager.clear_cache(self.args.device)
            
            return result
else:
    MacOptimizedTrainer = None


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the best available device with Mac optimizations."""
    if preferred:
        return torch.device(preferred)
    
    # Use Mac device manager for better detection
    try:
        device_manager = MacDeviceManager()
        return device_manager.detect_device()
    except Exception:
        # Fallback to basic detection
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")


def train_model(
    config: MacOptimizedTrainingConfig,
    train_dataset=None,
    eval_dataset=None,
    compute_metrics_fn=None
) -> Optional[Trainer]:
    """
    Train transformer model with Mac optimizations.
    
    Args:
        config: Training configuration
        train_dataset: Training dataset (HuggingFace Dataset)
        eval_dataset: Evaluation dataset (HuggingFace Dataset)
        compute_metrics_fn: Function to compute evaluation metrics
        
    Returns:
        Trained Trainer object
    """
    # Create trainer
    trainer_wrapper = MacOptimizedTransformerTrainer(config)
    
    # Setup device and optimizations
    device = trainer_wrapper.setup_device_and_optimization()
    print(f"🚀 Training on {device} with Mac optimizations: {config.use_mac_optimizations}")
    
    # Load model and tokenizer
    model, tokenizer = trainer_wrapper.load_model_and_tokenizer()
    
    # Create output directory
    config.output_dir.mkdir(parents=True, exist_ok=True)
    
    if train_dataset is None or eval_dataset is None:
        print("⚠️  No datasets provided. Model and tokenizer loaded, ready for training.")
        print(f"   Model: {config.model_name}")
        print(f"   Device: {device}")
        print(f"   Batch size (train): {config.batch_size_train}")
        print(f"   Batch size (eval): {config.batch_size_eval}")
        return None
    
    # Create trainer
    trainer = trainer_wrapper.create_trainer(train_dataset, eval_dataset, compute_metrics_fn)
    
    try:
        # Train the model
        print("🏋️  Starting training...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        tokenizer.save_pretrained(config.output_dir)
        
        print(f"✅ Training completed! Model saved to {config.output_dir}")
        return trainer
        
    except Exception as e:
        print(f"❌ Training failed: {str(e)}")
        # Clear cache on failure
        if config.use_mac_optimizations and trainer_wrapper.device_manager:
            trainer_wrapper.device_manager.clear_cache(device)
        raise


# Convenience function for backward compatibility
def train_transformer_model(config: MacOptimizedTrainingConfig) -> None:
    """Legacy function name for backward compatibility."""
    warnings.warn("train_transformer_model is deprecated. Use train_model instead.", DeprecationWarning)
    train_model(config)


if __name__ == "__main__":
    if TRANSFORMERS_AVAILABLE:
        config = MacOptimizedTrainingConfig()
        train_model(config)
    else:
        print("Transformers library not available. Install with: pip install transformers")