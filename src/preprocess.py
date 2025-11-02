"""Data preprocessing utilities for the emotion-aware feedback pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def load_dataset(path: Path) -> pd.DataFrame:
    """Load raw feedback data from ``path`` into a DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def clean_text(text_series: pd.Series) -> pd.Series:
    """Basic text normalization placeholder. Extend with real cleaning rules."""
    return text_series.str.replace(r"\s+", " ", regex=True).str.strip()


def prepare_features(df: pd.DataFrame, text_column: str) -> Iterable[str]:
    """Yield cleaned text examples ready for feature extraction."""
    cleaned = clean_text(df[text_column].astype(str))
    return cleaned.tolist()


if __name__ == "__main__":
    raise SystemExit("This module provides helper functions and is not intended for CLI use.")
