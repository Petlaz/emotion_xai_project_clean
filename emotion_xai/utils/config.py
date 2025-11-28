"""Configuration management for emotion-xai project."""

from pathlib import Path
from typing import Dict, Any
import os
import yaml


class Config:
    """Base configuration class."""
    
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    # Model paths
    MODELS_DIR = Path("models")
    BASELINE_MODEL_DIR = MODELS_DIR / "baseline"
    TRANSFORMER_MODEL_DIR = MODELS_DIR / "distilroberta_finetuned"
    CLUSTERING_MODEL_DIR = MODELS_DIR / "cluster_embeddings"
    
    # Training parameters
    BATCH_SIZE = 16
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 3
    MAX_LENGTH = 512
    
    # Model parameters
    MODEL_NAME = "distilroberta-base"
    NUM_LABELS = 28  # GoEmotions has 28 emotion labels
    
    # Clustering parameters
    N_NEIGHBORS = 15
    MIN_DIST = 0.1
    MIN_CLUSTER_SIZE = 10
    
    # Explainability parameters
    MAX_SHAP_SAMPLES = 100
    
    @classmethod
    def from_yaml(cls, config_path: Path) -> "Config":
        """Load configuration from YAML file."""
        if not config_path.exists():
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = cls()
        for key, value in config_data.items():
            if hasattr(config, key.upper()):
                setattr(config, key.upper(), value)
        
        return config


class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    LOG_LEVEL = "INFO"


class TestingConfig(Config):
    """Testing configuration."""
    DEBUG = True
    LOG_LEVEL = "DEBUG"
    BATCH_SIZE = 4  # Smaller batch size for testing


def get_config() -> Config:
    """Get configuration based on environment variable."""
    env = os.getenv("EMOTION_XAI_ENV", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()