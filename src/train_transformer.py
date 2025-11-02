"""Fine-tuning routines for DistilRoBERTa on emotion classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader


@dataclass
class TrainingConfig:
    """Configuration for transformer fine-tuning."""

    model_name: str = "distilroberta-base"
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 3
    output_dir: Path = Path("models/distilroberta_finetuned")
    device: Optional[str] = None


def resolve_device(preferred: Optional[str] = None) -> torch.device:
    """Resolve the best available device with macOS MPS fallback."""
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(config: TrainingConfig) -> None:
    """Placeholder training loop to be implemented later."""
    device = resolve_device(config.device)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    # TODO: Implement Hugging Face Trainer fine-tuning pipeline.
    print(f"Training would run on {device} for {config.num_epochs} epochs.")


if __name__ == "__main__":
    train_model(TrainingConfig())
